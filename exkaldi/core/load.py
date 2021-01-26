# coding=utf-8
#
# Yu Wang (University of Yamanashi)
# Mar,2020
#
# Licensed under the Apache License,Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exkaldi core functions to interact with kaldi"""

import numpy as np
import copy
import os
from io import BytesIO

from exkaldi.version import info as ExkaldiInfo
from exkaldi.error import *
from exkaldi.utils.utils import run_shell_command,type_name,list_files
from exkaldi.utils.utils import FileHandleManager
from exkaldi.utils import declare
from exkaldi.core.archive import BytesArchive,BytesMatrix,BytesFeat,BytesCMVN,BytesProb,BytesFmllr,BytesAliTrans
from exkaldi.core.archive import NumpyMatrix,NumpyFeat,NumpyCMVN,NumpyProb,NumpyAliTrans,NumpyFmllr
from exkaldi.core.archive import NumpyAli,NumpyAliPhone,NumpyAliPdf
from exkaldi.core.archive import Transcription,IndexTable,ListTable,WavSegment

# load list table
def load_list_table(target,name="listTable"):
	'''
	Generate a list table object from dict object or file.

	Args:
		<target>: dict object or a file path.
	
	Return:
		a ListTable object.
	'''
	declare.is_classes("target",target,[dict,ListTable,str])

	newTable = ListTable(name=name)
	if type_name(target) in ["dict","ListTable"]:
		newTable.update(target)
		return newTable

	else:
		files = list_files(target)
		for filePath in files:
			with open(filePath,"r",encoding="utf-8") as fr:
				lines = fr.readlines()
			for index,line in enumerate(lines,start=1):
				t = line.strip().split(maxsplit=1)
				if len(t) < 2:
					raise WrongDataFormat(f"Line Number: {index}\n"+f"Line Content: {line}\n"+f"Missing paired key and value information in file:{filePath}.")
				else:
					newTable[t[0]] = t[1]

		return newTable

def __read_one_record_from_ark(fp):
	'''
	Read a utterance from opened file pointer of an archive file.
	It is used to generate bytes index table.
	'''
	# read utterance ID
	utt = ''
	while True:
		char = fp.read(1).decode()
		if (char == '') or (char == ' '):
			break
		utt += char
	utt = utt.strip()
	if utt == '':
		if fp.read() == b'':
			return (None,None,None)
		else:
			fp.close()
			raise WrongDataFormat("Miss utterance ID before utterance. This may not be complete Kaldi archeve table file.")
	# read data
	binarySymbol = fp.read(2).decode()
	if binarySymbol == '\0B':
		sizeSymbol = fp.read(1).decode()
		if sizeSymbol == '\4':
			frames = int(np.frombuffer(fp.read(4),dtype='int32',count=1)[0])
			buf = fp.read( frames * 5)  # move the handle
			del buf
			dataSize = len(utt) + 8 + frames * 5
			return (utt,frames,dataSize)
		else:
			dataType = sizeSymbol + fp.read(2).decode() 
			if dataType == 'CM ':
				fp.close()
				raise UnsupportedType("Unsupported to generate index table from compressed archive table. Please decompress it firstly.")                    
			elif dataType == 'FM ':
				sampleSize = 4
			elif dataType == 'DM ':
				sampleSize = 8
			else:
				fp.close()
				raise WrongDataFormat(f"This may not be Kaldi archeve table file.")
			s1,rows,s2,cols = np.frombuffer(fp.read(10),dtype="int8,int32,int8,int32",count=1)[0]
			rows = int(rows)
			cols = int(cols)
			buf = fp.read(rows * cols * sampleSize) # move the handle
			del buf
			dataSize = len(utt) + 16 + rows * cols * sampleSize
			return (utt,rows,dataSize)
	else:
		fp.close()
		raise WrongDataFormat("Miss binary symbol before utterance. This may not be Kaldi binary archeve table file.")

def __read_index_table_from_ark_file(fileName):
	'''
	Read index table from ark file.
	'''
	fileName = os.path.abspath(fileName)
	newTable = IndexTable()
	startIndex = 0
	with open(fileName,"rb") as fr:
		while True:
			utt,frames,dataSize = __read_one_record_from_ark(fr)
			if utt is None:
				break
			else:
				newTable[utt] = newTable.spec(frames,startIndex,dataSize,fileName)
				startIndex += dataSize
	
	return newTable

def __read_index_table_from_scp_file(fileName):
	'''
	Read index table from scp file.
	'''
	newTable = IndexTable()
	
	with FileHandleManager() as fhm:

		fr = fhm.open(fileName,"r",encoding="utf-8")
		lines = fr.readlines()

		for lineID,lineTxt in enumerate(lines):
			line = lineTxt.strip().split()
			if len(line) == 0:
				continue
			elif len(line) == 1:
				raise WrongDataFormat(f"line {lineID}: {lineTxt}\n"+"Missed complete utterance-filepath information.")
			elif len(line) > 2:
				raise WrongDataFormat("We don't support reading index table from binary data generated via PIPE line. The second value should be ark file path and the shift.")
			else:
				uttID = line[0]
				line = line[1].split(":")
				if len(line) != 2:
					raise WrongDataFormat(f"line {lineID}: {lineTxt}\n"+"Missed complete file path and shift value information.")
				arkFileName = line[0]
				startIndex = int(line[1]) - 1 - len(uttID)

				fr = fhm.call(arkFileName)
				if fr is None:
					fr = fhm.open(arkFileName,"rb")
				
				fr.seek(startIndex)
				_,frames,dataSize = __read_one_record_from_ark(fr)
				arkFileName = os.path.abspath(arkFileName)
				newTable[uttID] = newTable.spec( frames,startIndex,dataSize,arkFileName)

	return newTable

def load_index_table(target,name="index",useSuffix=None):
	'''
	Load an index table from dict,or archive table file.

	Args:
		<target>: dict object,.ark or .scp file,IndexTable object,bytes archive object.
		<name>: a string.
		<useSuffix>: "ark" or "scp". We will check the file type by its suffix. 
								But if <target> is file path and not default suffix (ark or scp),you have to declare which type it is.

	Return:
		an exkaldi IndexTable object.
	'''
	newTable = IndexTable(name=name)

	if type_name(target) == "dict":
		for key,value in target.items():
			if isinstance(value,(list,tuple)):
				assert len(value) in [3,4],f"Expected (frames,start index,data size[,file path]) but {value} does not match."
				newTable[key] = newTable.spec(*value)
			elif type_name(value) == "Index":
				newTable[key] = value
			else:
				raise WrongDataFormat(f"Expected list or tuple but got wrong index info format: {value}.")	
		
		return newTable

	elif type_name(target) == "IndexTable":
		newTable.update(target)
		return newTable
	
	elif isinstance(target,BytesArchive):
		newTable.update( target.indexTable )
		return newTable
	
	else:
		fileList = list_files(target)

		if useSuffix is not None:
			declare.is_valid_string("useSuffix",useSuffix)
			useSuffix = useSuffix.strip()[-3:].lower()
			declare.is_instances("useSuffix",useSuffix,["ark","scp"])
		else:
			useSuffix = ""

		for fileName in fileList:

			if fileName.rstrip().endswith(".ark"):
				t = __read_index_table_from_ark_file(fileName)
			elif fileName.rstrip().endswith(".scp"):
				t = __read_index_table_from_scp_file(fileName)
			elif useSuffix == "ark":
				t = __read_index_table_from_ark_file(fileName)
			elif useSuffix == "scp":
				t = __read_index_table_from_scp_file(fileName)
			else:
				raise UnsupportedType("Unknown file suffix. Specify <useSuffix> please.")

			newTable.update(t)
	
		return newTable

# load archive data
def __read_data_from_file(fileName,useSuffix=None):
	'''
	Read data from file. If the file suffix is unknown,<useSuffix> is necessary.
	'''
	declare.kaldi_existed()

	if useSuffix != None:
		declare.is_valid_string("useSuffix",useSuffix)
		useSuffix = useSuffix.strip().lower()[-3:]
		declare.is_instances("useSuffix",useSuffix,["ark","scp","npy"])
	else:
		useSuffix = ""
	
	allFiles = list_files(fileName)

	allData_bytes = []
	allData_numpy = {}

	def loadNpyFile(fileName):
		try:
			temp = np.load(fileName,allow_pickle=True)
			data = {}
			for utt_mat in temp:
				assert isinstance(utt_mat[0],str) and isinstance(utt_mat[1],np.ndarray)
				data[utt_mat[0]] = utt_mat[1]           
		except:
			raise UnsupportedType(f'This is not a valid Exkaldi npy file: {fileName}.')
		else:
			return data
	
	def loadArkScpFile(fileName,suffix):
		declare.kaldi_existed()

		if suffix == "ark":
			cmd = 'copy-feats ark:'
		else:
			cmd = 'copy-feats scp:'

		cmd += '{} ark:-'.format(fileName)
		out,err,cod = run_shell_command(cmd,stdout="PIPE",stderr="PIPE")
		if (isinstance(cod,int) and cod != 0) or out == b'':
			raise KaldiProcessError('Failed to read archive table.',err.decode())
		else:
			#if sys.getsizeof(out) > 10000000000:
			#    print('Warning: Data is extramely large. We don't recommend use load_index_table to replace it.') 
			return out

	for fileName in allFiles:
		sfx = fileName.strip()[-3:].lower()
		if sfx == "npy":
			allData_numpy.update( loadNpyFile(fileName) )
		elif sfx in ["ark","scp"]:
			allData_bytes.append( loadArkScpFile(fileName,sfx) )
		elif useSuffix == "npy":
			allData_numpy.update( loadNpyFile(fileName) )
		elif useSuffix in ["ark","scp"]:
			allData_bytes.append( loadArkScpFile(fileName,sfx) )
		else:
			raise UnsupportedType('Unknown file suffix. You can appoint the <useSuffix> option with "scp","ark" or "npy".')
	
	allData_bytes = b"".join(allData_bytes)

	if useSuffix == "":
		useSuffix = allFiles[0].strip()[-3:].lower()

	if useSuffix == "npy":
		dataType = "numpy"
	else:
		dataType = "bytes"
	
	return allData_bytes,allData_numpy,dataType

def load_feat(target,name="feat",useSuffix=None):
	'''
	Load feature data.

	Args:
		<target>: Python dict object,bytes object,exkaldi feature object,.ark file,.scp file,.npy file.
		<name>: a string.
		<useSuffix>: "ark" or "scp" or "npy". We will check the file type by its suffix. 
								But if <target> is file path and not default suffix (ark or scp),you have to declare which type it is.

	Return:
		A BytesFeat or NumpyFeat object.
	'''
	declare.is_valid_string("name",name)

	if isinstance(target,dict):
		result = NumpyFeat(target,name)
		result.check_format()
		return result

	elif isinstance(target,bytes):
		result = BytesFeat(target,name)
		result.check_format()
		return result

	elif isinstance(target,(NumpyFeat,BytesFeat)):
		result = copy.deepcopy(target)
		result.rename(name)
		return result

	elif isinstance(target,str):
		allData_bytes,allData_numpy,dataType = __read_data_from_file(target,useSuffix)
		if dataType == "numpy":
			result = NumpyFeat(allData_numpy) + BytesFeat(allData_bytes)
		else:
			result = BytesFeat(allData_bytes) + NumpyFeat(allData_numpy)
		result.rename(name)
		return result

	elif isinstance(target,IndexTable):
		return target.fetch(arkType="feat",name=name)

	else:
		raise UnsupportedType(f"Expected Python dict,bytes object,exkaldi feature or indexTable object or file path but got{type_name(target)}.")

def load_cmvn(target,name="cmvn",useSuffix=None):
	'''
	Load CMVN statistics data.

	Args:
		<target>: Python dict object,bytes object,exkaldi feature or index table object,.ark file,.scp file,npy file.
		<name>: a string.
		<useSuffix>: "ark" or "scp" or "npy". We will check the file type by its suffix. 
								But if <target> is file path and not default suffix (ark or scp),you have to declare which type it is.

	Return:
		A BytesFeat or NumpyFeat object.
	'''
	declare.is_valid_string("name",name)

	if isinstance(target,dict):
		result = NumpyCMVN(target,name)
		result.check_format()
		return result

	elif isinstance(target,bytes):
		result = BytesCMVN(target,name)
		result.check_format()
		return result

	elif isinstance(target,(NumpyCMVN,BytesCMVN)):
		result = copy.deepcopy(target)
		result.rename(name)
		return result

	elif isinstance(target,str):
		allData_bytes,allData_numpy,dataType = __read_data_from_file(target,useSuffix)
		if dataType == "npy":
			result = NumpyCMVN(allData_numpy) + BytesCMVN(allData_bytes)
		else:
			result = BytesCMVN(allData_bytes) + NumpyCMVN(allData_numpy)
		result.rename(name)
		return result

	elif isinstance(target,IndexTable):
		return target.fetch(arkType="cmvn",name=name)

	else:
		raise UnsupportedType(f"Expected Python dict,bytes object,exkaldi feature or index table object or file path but got{type_name(target)}.")

def load_prob(target,name="prob",useSuffix=None):
	'''
	Load post probability data.

	Args:
		<target>: Python dict object,bytes object,exkaldi feature object,.ark file,.scp file,.npy file.
		<name>: a string.
		<useSuffix>: "ark" or "scp" or "npy". We will check the file type by its suffix. 
								But if <target> is file path and not default suffix (ark or scp),you have to declare which type it is.
							
	Return:
		A BytesProb or NumpyProb object.
	'''
	declare.is_valid_string("name",name)

	if isinstance(target,dict):
		result = NumpyProb(target,name)
		result.check_format()
		return result

	elif isinstance(target,bytes):
		result = BytesProb(target,name)
		result.check_format()
		return result

	elif isinstance(target,(NumpyProb,BytesProb)):
		result = copy.deepcopy(target)
		result.rename(name)
		return result

	elif isinstance(target,str):
		allData_bytes,allData_numpy,dataType = __read_data_from_file(target,useSuffix)
		if dataType == "numpy":
			result = NumpyProb(allData_numpy) + BytesProb(allData_bytes)
		else:
			result = BytesProb(allData_bytes) + NumpyProb(allData_numpy)
		result.rename(name)
		return result

	elif isinstance(target,IndexTable):
		return target.fetch(arkType="prob",name=name)

	else:
		raise UnsupportedType(f"Expected Python dict,bytes object,exkaldi feature object or file path but got{type_name(target)}.")

def load_fmllr(target,name="prob",useSuffix=None):
	'''
	Load fmllr transform matrix data.

	Args:
		<target>: Python dict object,bytes object,exkaldi feature or index table object,.ark file,.scp file,npy file.
		<name>: a string.
		<useSuffix>: "ark" or "scp" or "npy". We will check the file type by its suffix. 
								But if <target> is file path and not default suffix (ark or scp),you have to declare which type it is.

	Return:
		A BytesFmllr or NumpyFmllr object.
	'''
	declare.is_valid_string("name",name)

	if isinstance(target,dict):
		result = NumpyFmllr(target,name)
		result.check_format()
		return result

	elif isinstance(target,bytes):
		result = BytesFmllr(target,name)
		result.check_format()
		return result

	elif isinstance(target,(NumpyFmllr,BytesFmllr)):
		result = copy.deepcopy(target)
		result.rename(name)
		return result

	elif isinstance(target,str):
		allData_bytes,allData_numpy,dataType = __read_data_from_file(target,useSuffix)
		if dataType == "npy":
			result = NumpyFmllr(allData_numpy) + BytesFmllr(allData_bytes)
		else:
			result = BytesFmllr(allData_bytes) + NumpyFmllr(allData_numpy)
		result.rename(name)
		return result

	elif isinstance(target,IndexTable):
		return target.fetch(arkType="fmllr",name=name)

	else:
		raise UnsupportedType(f"Expected Python dict,bytes object,exkaldi fmllr matrix object,index table object or file path but got{type_name(target)}.")

def load_ali(target,aliType="transitionID",name="ali",hmm=None):
	'''
	Load alignment data.

	Args:
		<target>: Python dict object,bytes object,exkaldi alignment object,kaldi alignment file or .npy file.
		<aliType>: None,or one of 'transitionID','phoneID','pdfID'. It will return different alignment object.
		<name>: a string.
		<hmm>: file path or exkaldi HMM object.

	Return:
		exkaldi alignment objects.
	'''
	declare.is_valid_string("name",name)
	declare.is_instances("aliType",aliType,[None,"transitionID","phoneID","pdfID"])
	declare.kaldi_existed()

	def transform(data,cmd):
		out,err,cod = run_shell_command(cmd,stdin="PIPE",stdout="PIPE",stderr="PIPE",inputs=data)
		if (isinstance(cod,int) and cod != 0) and out == b'':
			raise KaldiProcessError('Failed to transform alignment.',err.decode())
		else:
			result = {}
			sp = BytesIO(out)
			for line in sp.readlines():
				line = line.decode()
				line = line.strip().split()
				utt = line[0]
				matrix = np.array(line[1:],dtype=np.int32)
				result[utt] = matrix
			return result

	if isinstance(target,dict):
		if aliType is None:
			result = NumpyAli(target,name)
		elif aliType == "transitionID":
			result = NumpyAliTrans(target,name)
		elif aliType == "phoneID":
			result = NumpyAliPhone(target,name)
		elif aliType == "pdfID":
			result = NumpyAliPdf(target,name)
		else:
			raise WrongOperation(f"<aliType> should be None,'transitionID','phoneID' or 'pdfID' but got {aliType}.")
		result.check_format()
		return result

	elif isinstance(target,(NumpyAli,NumpyAliTrans,BytesAliTrans)):
		result = copy.deepcopy(target)
		result.rename(name)
		return result

	elif isinstance(target,IndexTable):
		result = target.fetch(arkType="ali")
		if aliType in ["phoneID","pdfID"]:
			result = result.to_numpy(aliType,hmm)
		result.rename(name)
		return result

	elif isinstance(target,str):
		allFiles = list_files(target)
		numpyAli = {}
		bytesAli = []

		for fileName in allFiles:
			fileName = fileName.strip()
			if fileName.endswith(".npy"):
				try:
					temp = np.load(fileName,allow_pickle=True)
					numpyAli.update(temp)          
				except:
					raise UnsupportedType(f'This is not a valid Exkaldi npy file: {fileName}.')
			else:
				if fileName.endswith('.gz'):
					cmd = f'gunzip -c {fileName}'
				else:
					cmd = f'cat {fileName}'

				if aliType is None or aliType == "transitionID":
					out,err,cod = run_shell_command(cmd,stdout="PIPE",stderr="PIPE")
					if (isinstance(cod,int) and cod != 0 ) or out == b'':
						raise ShellProcessError(f"Failed to get the alignment data from file: {fileName}.",err.decode())
					else:
						bytesAli.append( out )
				
				else:
					with FileHandleManager() as fhm:
						declare.is_potential_hmm("hmm",hmm)
						if not isinstance(hmm,str):
							hmmTemp = fhm.create("wb+")
							hmm.save(hmmTemp)
							hmm = hmmTemp.name

						if aliType == "phoneID":
							cmd += f" | ali-to-phones --per-frame=true {hmm} ark:- ark,t:-"
							temp = transform(None,cmd)

						else:
							cmd += f" | ali-to-pdf {hmm} ark:- ark,t:-"
							temp = transform(None,cmd)

					numpyAli.update(temp) 

		bytesAli = b"".join(bytesAli)
		if aliType is None:
			if len(numpyAli) == 0:
				return BytesAliTrans(bytesAli,name=name)
			elif len(bytesAli) == 0:
				return NumpyAli(numpyAli,name=name)
			else:
				result = NumpyAliTrans(numpyAli) + BytesAliTrans(bytesAli)
				result.rename(name)
				return result
		elif aliType == "transitionID":
			if len(numpyAli) == 0:
				return BytesAliTrans(bytesAli,name=name)
			elif len(bytesAli) == 0:
				return NumpyAliTrans(numpyAli,name=name)
			else:
				result = NumpyAliTrans(numpyAli) + BytesAliTrans(bytesAli)
				result.rename(name)
				return result
		elif aliType == "phoneID":		
			return NumpyAliPhone(numpyAli,name=name)
		else:
			return NumpyAliPdf(numpyAli,name=name)

	else:
		raise UnsupportedType(f"<target> should be dict,file name or exkaldi alignment or index table object but got: {type_name(target)}.")

def load_transcription(target,name="transcription",checkSpace=True):
	'''
	Load transcription from file.

	Args:
		<target>: transcription file path.
		<name>: a string.
		<checkSpace>: a bbol value. If True,we will check the validity of the number of spaces.

	Return:
		An exkaldi Transcription object.
	'''
	declare.is_classes("target",target,["dict","Transcription","ListTable","str"])
	declare.is_bool("checkSpace",checkSpace)

	if isinstance(target,str):
		declare.is_file("target",target)
		with open(target,"r",encoding="utf-8") as fr:
			lines = fr.readlines()
		result = Transcription(name=name)
		for index,line in enumerate(lines,start=1):
			t = line.strip().split(maxsplit=1)
			if len(t) < 2:
				raise WrongDataFormat(f"Line Number: {index}\n"+f"Line Content: {line}\n"+"Missing entire key and value information.")
			else:
				result[t[0]] = t[1]
	else:
		for utt,utterance in target.items():
			declare.is_valid_string("utterance ID",utt)
			declare.is_valid_string("utterance",utterance)
		result = Transcription(target,name=name)

	if checkSpace:
		sampleText = result.subset(nRandom=100)
		spaceCount = 0
		for key,value in sampleText.items():
			spaceCount += value.count(" ")
		if spaceCount < len(sampleText)//2:
			errMes = "The transcription doesn't seem to be separated by spaces or extremely short."
			errMes += "If it actually has right format, set the <checkSpace>=False and run this function again."
			raise WrongDataFormat(errMes)

	return result




		


		


		




