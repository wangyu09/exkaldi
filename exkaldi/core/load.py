# coding=utf-8
#
# Yu Wang (University of Yamanashi)
# Mar, 2020
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exkaldi core functions to interact with kaldi"""

import numpy as np
import copy
import os
import subprocess
import tempfile
from io import BytesIO

from exkaldi.version import info as ExkaldiInfo
from exkaldi.version import WrongPath, WrongOperation, WrongDataFormat, UnsupportedType, ShellProcessError, KaldiProcessError
from exkaldi.utils.utils import run_shell_command, type_name, list_files
from exkaldi.utils.utils import FileHandleManager
from exkaldi.utils import declare
from exkaldi.core.archive import BytesArchive, BytesMatrix, BytesFeature, BytesCMVNStatistics, BytesProbability, BytesAlignmentTrans, BytesFmllrMatrix
from exkaldi.core.archive import NumpyMatrix, NumpyFeature, NumpyCMVNStatistics, NumpyProbability, NumpyAlignmentTrans, NumpyFmllrMatrix
from exkaldi.core.archive import NumpyAlignment, NumpyAlignmentPhone, NumpyAlignmentPdf
from exkaldi.core.archive import Transcription, ArkIndexTable, ListTable

# load list table
def load_list_table(target, name="table"):
	'''
	Generate a list table object from dict object or file.

	Args:
		<target>: dict object or the text file path.
	
	Return:
		a ListTable object.
	'''
	declare.is_classes("target", target, [dict, ListTable, str])

	newTable = ListTable(name=name)
	if type_name(target) in ["dict","ListTable"]:
		newTable.update(target)
		return newTable

	else:
		declare.is_file("target", target)

		with open(target, "r", encoding="utf-8") as fr:
			lines = fr.readlines()
		for index, line in enumerate(lines, start=1):
			t = line.strip().split(maxsplit=1)
			if len(t) < 2:
				print(f"Line Number: {index}")
				print(f"Line Content: {line}")
				raise WrongDataFormat("Missing paired key and value information.")
			else:
				newTable[t[0]] = t[1]
		
		return newTable

# load index table
def __read_one_record_from_ark(fp):
	'''
	Read a utterance from opened file pointer of an archive file.
	It is used to generate bytes index table.
	'''
	# read utterance ID
	utt = ''
	while True:
		char = fp.read(1).decode()
		if (char == '') or (char == ' '):break
		utt += char
	utt = utt.strip()
	if utt == '':
		if fp.read() == b'':
			return (None, None, None)
		else:
			fp.close()
			raise WrongDataFormat("Miss utterance ID before utterance. This may not be complete Kaldi archeve table file.")
	# read data
	binarySymbol = fp.read(2).decode()
	if binarySymbol == '\0B':
		sizeSymbol = fp.read(1).decode()
		if sizeSymbol == '\4':
			frames = int(np.frombuffer(fp.read(4), dtype='int32', count=1)[0])
			buf = fp.read( frames * 5)
			del buf
			dataSize = len(utt) + 8 + frames * 5
			return (utt, frames, dataSize)
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
			s1,rows,s2,cols = np.frombuffer(fp.read(10), dtype="int8, int32, int8, int32", count=1)[0]
			rows = int(rows)
			cols = int(cols)
			buf = fp.read(rows * cols * sampleSize)
			del buf
			dataSize = len(utt) + 16 + rows * cols * sampleSize
			return (utt, rows, dataSize)
	else:
		fp.close()
		raise WrongDataFormat("Miss binary symbol before utterance. This may not be Kaldi binary archeve table file.")

def __read_index_table_from_ark_file(fileName):
	'''
	Read index table from ark file.
	'''
	newTable = ArkIndexTable()
	startIndex = 0
	with open(fileName, "rb") as fr:
		while True:
			utt, frames, dataSize = __read_one_record_from_ark(fr)
			if utt is None:
				break
			else:
				newTable[utt] = newTable.spec( frames, startIndex, dataSize, fileName)
				startIndex += dataSize
	
	return newTable

def __read_index_table_from_scp_file(fileName):
	'''
	Read index table scp file.
	'''
	newTable = ArkIndexTable()
	
	with FileHandleManager() as fhm:

		fr = fhm.open(fileName, "r", encoding="utf-8")
		lines = fr.readlines()

		for lineID, lineTxt in enumerate(lines):
			line = lineTxt.strip().split()
			if len(line) == 0:
				continue
			elif len(line) == 1:
				print(f"line {lineID}: {lineTxt}")
				raise WrongDataFormat("Missed complete utterance-filepath information.")
			elif len(line) > 2:
				raise WrongDataFormat("We don't support reading index table from binary data generated via pip line. The second value should be ark file path and shift value.")
			else:
				#uttID = line[0]
				line = line[1].split(":")
				if len(line) != 2:
					print(f"line {lineID}: {lineTxt}")
					raise WrongDataFormat("Missed complete file path and shift value information.")
				arkFileName = line[0]
				startIndex = int(line[1])

				fr = fhm.call(arkFileName)
				if fr is None:
					fr = fhm.open(arkFileName, "rb")
				
				fr.seek(startIndex)
				uttID, frames, dataSize = __read_one_record_from_ark(fr)
				newTable[uttID] = newTable.spec( frames, startIndex, dataSize, arkFileName)

	return newTable

def load_index_table(target, name="index", useSuffix=None):
	'''
	Load an index table from dict, or archive table file.

	Args:
		<target>: dict object, ArkIndexTable object, bytes archive object or archive table file .
		<name>: a string.
		<useSuffix>: if <target> is file path and not default suffix, specified it.

	Return:
		an exkaldi ArkIndexTable object.
	'''
	newTable = ArkIndexTable(name=name)

	if type_name(target) == "dict":
		for key, value in target.items():
			if isinstance(value, (list,tuple)):
				assert len(value) in [3,4], f"Expected (frames, start index, data size[, file path]) but {value} does not match."
				newTable[key] = newTable.spec(*value)
			elif type_name(value) == "Index":
				newTable[key] = value
			else:
				raise WrongDataFormat(f"Expected list or tuple but got wrong index info format: {value}.")	
		
		return newTable

	elif type_name(target) == "ArkIndexTable":
		newTable.update(target)
		return newTable
	
	elif isinstance(target, BytesArchive):
		newTable.update( target.indexTable )
		return newTable
	
	else:
		fileList = list_files(target)

		if useSuffix is not None:
			declare.is_valid_string("useSuffix", useSuffix)
			useSuffix = useSuffix.strip()[-3:].lower()
			declare.is_instances("useSuffix", useSuffix, ["ark","scp"])
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
def __read_data_from_file(fileName, useSuffix=None):
	'''
	Read data from file. If the file suffix is unknown, <useSuffix> should be assigned.
	'''
	declare.kaldi_existed()

	if useSuffix != None:
		declare.is_valid_string("useSuffix", useSuffix)
		useSuffix = useSuffix.strip().lower()[-3:]
		declare.is_instances("useSuffix", useSuffix, ["ark","scp","npy"])
	else:
		useSuffix = ""
	
	allFiles = list_files(fileName)

	allData_bytes = []
	allData_numpy = {}

	def loadNpyFile(fileName):
		try:
			temp = np.load(fileName, allow_pickle=True)
			data = {}
			for utt_mat in temp:
				data[utt_mat[0]] = utt_mat[1]           
		except:
			raise UnsupportedType(f'This is not a valid exkaldi npy file: {fileName}.')
		else:
			return data
	
	def loadArkScpFile(fileName, suffix):
		declare.kaldi_existed()

		if suffix == "ark":
			cmd = 'copy-feats ark:'
		else:
			cmd = 'copy-feats scp:'

		cmd += '{} ark:-'.format(fileName)
		out, err, cod = run_shell_command(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		if (isinstance(cod, int) and cod != 0) or out == b'':
			print(err.decode())
			raise KaldiProcessError('Copy feat defeated.')
		else:
			#if sys.getsizeof(out) > 10000000000:
			#    print('Warning: Data is extramely large. It could not be used correctly sometimes.') 
			return out

	for fileName in allFiles:
		sfx = fileName[-3:].lower()
		if sfx == "npy":
			allData_numpy.update( loadNpyFile(fileName) )
		elif sfx in ["ark", "scp"]:
			allData_bytes.append( loadArkScpFile(fileName, sfx) )
		elif useSuffix == "npy":
			allData_numpy.update( loadNpyFile(fileName) )
		elif useSuffix in ["ark", "scp"]:
			allData_bytes.append( loadArkScpFile(fileName, sfx) )
		else:
			raise UnsupportedType('Unknown file suffix. You can assign the <useSuffix> with "scp", "ark" or "npy".')
	
	allData_bytes = b"".join(allData_bytes)

	if useSuffix == "":
		useSuffix = allFiles[0][-3:].lower()
	if useSuffix == "npy":
		dataType = "numpy"
	else:
		dataType = "bytes"

	return allData_bytes, allData_numpy, dataType

def load_feat(target, name="feat", useSuffix=None):
	'''
	Load feature data.

	Args:
		<target>: Python dict object, bytes object, exkaldi feature object, .ark file, .scp file, npy file.
		<name>: a string.
		<useSuffix>: a string. When target is file path, use this to specify file.
	Return:
		A BytesFeature or NumpyFeature object.
	'''
	declare.is_valid_string("name", name)

	if isinstance(target, dict):
		result = NumpyFeature(target, name)
		result.check_format()
		return result

	elif isinstance(target, bytes):
		result = BytesFeature(target, name)
		result.check_format()
		return result

	elif isinstance(target, (NumpyFeature, BytesFeature)):
		result = copy.deepcopy(target)
		result.rename(name)
		return result

	elif isinstance(target, str):
		allData_bytes, allData_numpy, dataType = __read_data_from_file(target, useSuffix)
		if dataType == "npy":
			result = NumpyFeature(allData_numpy) + BytesFeature(allData_bytes)
		else:
			result = BytesFeature(allData_bytes) + NumpyFeature(allData_numpy)
		result.rename(name)
		return result

	elif isinstance(target, ArkIndexTable):
		return target.fetch(arkType="feat", name=name)

	else:
		raise UnsupportedType(f"Expected Python dict, bytes object, exkaldi feature object or file path but got{type_name(target)}.")

def load_cmvn(target, name="cmvn", useSuffix=None):
	'''
	Load CMVN statistics data.

	Args:
		<target>: Python dict object, bytes object, exkaldi feature object, .ark file, .scp file, npy file or index table object.
		<name>: a string.
		<useSuffix>: a string. When target is file path, use this to specify file.
	Return:
		A BytesFeature or NumpyFeature object.
	'''
	declare.is_valid_string("name", name)

	if isinstance(target, dict):
		result = NumpyCMVNStatistics(target, name)
		result.check_format()
		return result

	elif isinstance(target, bytes):
		result = BytesCMVNStatistics(target, name)
		result.check_format()
		return result

	elif isinstance(target, (NumpyCMVNStatistics, BytesCMVNStatistics)):
		result = copy.deepcopy(target)
		result.rename(name)
		return result

	elif isinstance(target, str):
		allData_bytes, allData_numpy, dataType = __read_data_from_file(target, useSuffix)
		if dataType == "npy":
			result = NumpyCMVNStatistics(allData_numpy) + BytesCMVNStatistics(allData_bytes)
		else:
			result = BytesCMVNStatistics(allData_bytes) + NumpyCMVNStatistics(allData_numpy)
		result.rename(name)
		return result

	elif isinstance(target, ArkIndexTable):
		return target.fetch(arkType="cmvn", name=name)

	else:
		raise UnsupportedType(f"Expected Python dict, bytes object, exkaldi feature object or file path but got{type_name(target)}.")

def load_prob(target, name="prob", useSuffix=None):
	'''
	Load post probability data.

	Args:
		<target>: Python dict object, bytes object, exkaldi feature object, .ark file, .scp file, npy file.
		<name>: a string.
		<useSuffix>: a string. When target is file path, use this to specify file.
	Return:
		A BytesProbability or NumpyProbability object.
	'''
	declare.is_valid_string("name", name)

	if isinstance(target, dict):
		result = NumpyProbability(target, name)
		result.check_format()
		return result

	elif isinstance(target, bytes):
		result = BytesProbability(target, name)
		result.check_format()
		return result

	elif isinstance(target, (NumpyProbability, BytesProbability)):
		result = copy.deepcopy(target)
		result.rename(name)
		return result

	elif isinstance(target, str):
		allData_bytes, allData_numpy, dataType = __read_data_from_file(target, useSuffix)
		if dataType == "numpy":
			result = NumpyProbability(allData_numpy) + BytesProbability(allData_bytes)
		else:
			result = BytesProbability(allData_bytes) + NumpyProbability(allData_numpy)
		result.rename(name)
		return result

	elif isinstance(target, ArkIndexTable):
		return target.fetch(arkType="prob", name=name)

	else:
		raise UnsupportedType(f"Expected Python dict, bytes object, exkaldi feature object or file path but got{type_name(target)}.")

def load_fmllr(target, name="prob", useSuffix=None):
	'''
	Load fmllr transform matrix data.

	Args:
		<target>: Python dict object, bytes object, exkaldi feature object, .ark file, .scp file, npy file.
		<name>: a string.
		<useSuffix>: a string. When target is file path, use this to specify file.
	Return:
		A BytesFmllrMatrix or NumpyFmllrMatrix object.
	'''
	declare.is_valid_string("name", name)

	if isinstance(target, dict):
		result = NumpyFmllrMatrix(target, name)
		result.check_format()
		return result

	elif isinstance(target, bytes):
		result = BytesFmllrMatrix(target, name)
		result.check_format()
		return result

	elif isinstance(target, (NumpyFmllrMatrix, BytesFmllrMatrix)):
		result = copy.deepcopy(target)
		result.rename(name)
		return result

	elif isinstance(target, str):
		allData_bytes, allData_numpy, dataType = __read_data_from_file(target, useSuffix)
		if dataType == "npy":
			result = NumpyFmllrMatrix(allData_numpy) + BytesFmllrMatrix(allData_bytes)
		else:
			result = BytesFmllrMatrix(allData_bytes) + NumpyFmllrMatrix(allData_numpy)
		result.rename(name)
		return result

	elif isinstance(target, ArkIndexTable):
		return target.fetch(arkType="fmllrMat", name=name)

	else:
		raise UnsupportedType(f"Expected Python dict, bytes object, exkaldi fmllr matrix object or file path but got{type_name(target)}.")

def load_ali(target, aliType=None, name="ali", hmm=None):
	'''
	Load alignment data.

	Args:
		<target>: Python dict object, bytes object, exkaldi alignment object, kaldi alignment file or .npy file.
		<aliType>: None, or one of 'transitionID', 'phoneID', 'pdfID'. It will return different alignment object.
		<name>: a string.
		<hmm>: file path or exkaldi HMM object.
	Return:
		exkaldi alignment data objects.
	'''
	declare.is_valid_string("name", name)
	declare.is_instances("aliType", aliType, [None, "transitionID", "phoneID","pdfID"])
	declare.kaldi_existed()

	def transform(data, cmd):
		out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=data)
		if (isinstance(cod,int) and cod != 0) and out == b'':
			print(err.decode())
			raise KaldiProcessError('Failed to transform alignment.')
		else:
			result = {}
			sp = BytesIO(out)
			for line in sp.readlines():
				line = line.decode()
				line = line.strip().split()
				utt = line[0]
				matrix = np.array(line[1:], dtype=np.int32)
				result[utt] = matrix
			return result

	if isinstance(target, dict):
		if aliType is None:
			result = NumpyAlignment(target, name)
		elif aliType == "transitionID":
			result = NumpyAlignmentTrans(target, name)
		elif aliType == "phoneID":
			result = NumpyAlignmentPhone(target, name)
		elif aliType == "pdfID":
			result = NumpyAlignmentPdf(target, name)
		else:
			raise WrongOperation(f"<aliType> should be None, 'transitionID', 'phoneID' or 'pdfID' but got {aliType}.")
		result.check_format()
		return result

	elif isinstance(target, (NumpyAlignment, NumpyAlignmentTrans, BytesAlignmentTrans)):
		result = copy.deepcopy(target)
		result.rename(name)
		return result

	elif isinstance(target, ArkIndexTable):
		result = target.fetch(arkType="ali")
		if aliType in ["phoneID","pdfID"]:
			result = result.to_numpy(aliType, hmm)
		result.rename(name)
		return result

	elif isinstance(target, str):
				
		allFiles = list_files(target)

		numpyAli = {}
		bytesAli = []

		for fileName in allFiles:
			
			if fileName.endswith(".npy"):
				try:
					temp = np.load(fileName, allow_pickle=True)
					for utt, mat in temp:
						numpyAli[ utt ] = mat             
				except:
					raise UnsupportedType(f'This is not a valid exkaldi npy file: {fileName}.')
			
			else:
				if fileName.endswith('.gz'):
					cmd = f'gunzip -c {fileName}'
				else:
					cmd = f'cat {fileName}'

					if aliType is None or aliType == "transitionID":
						out, err, cod = run_shell_command(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
						if (isinstance(cod, int) and cod != 0 ) or out == b'':
							print(err.decode())
							raise ShellProcessError(f"Failed to get the alignment data from file: {fileName}.")
						else:
							bytesAli.append( out )
					
					else:
						with FileHandleManager() as fhm:

							declare.is_potential_hmm("hmm", hmm)
							if not isinstance(hmm, str):
								hmmTemp = fhm.create("wb+")
								hmm.save(hmmTemp)
								hmm = hmmTemp.name

							if aliType == "phoneID":
								cmd += f" | ali-to-phones --per-frame=true {hmm} ark:- ark,t:-"
								temp = transform(None, cmd)

							else:
								cmd = f" | ali-to-pdf {hmm} ark:- ark,t:-"
								temp = transform(None, cmd)

						numpyAli.update( temp )	
			
			bytesAli = b"".join(bytesAli)
			if aliType is None:
				if len(numpyAli) == 0:
					return BytesAlignmentTrans(bytesAli, name=name)
				elif len(bytesAli) == 0:
					return NumpyAlignment(numpyAli, name=name)
				else:
					result = NumpyAlignmentTrans(numpyAli) + BytesAlignmentTrans(bytesAli)
					result.rename(name)
					return result
			elif aliType == "transitionID":
				if len(numpyAli) == 0:
					return BytesAlignmentTrans(bytesAli, name=name)
				elif len(bytesAli) == 0:
					return NumpyAlignmentTrans(numpyAli, name=name)
				else:
					result = NumpyAlignmentTrans(numpyAli) + BytesAlignmentTrans(bytesAli)
					result.rename(name)
					return result
			elif aliType == "phoneID":		
				return NumpyAlignmentPhone(numpyAli, name=name)
			else:
				return NumpyAlignmentPdf(numpyAli, name=name)

	else:
		raise UnsupportedType(f"<target> should be dict, file name or exkaldi alignment or index table object but got: {type_name(target)}.")

def load_transcription(target, name="transcription"):
	'''
	Load transcription from file.

	Args:
		<target>: transcription file path.
		<name>: a string.

	Return:
		An exkaldi Transcription object.
	'''
	declare.is_classes("target", target, ["dict","Transcription","ListTable","str"])

	if isinstance(target, str):
		declare.is_file("target", target)
		with open(target, "r", encoding="utf-8") as fr:
			lines = fr.readlines()
		result = Transcription(name=name)
		for index, line in enumerate(lines, start=1):
			t = line.strip().split(maxsplit=1)
			if len(t) < 2:
				print(f"Line Number: {index}")
				print(f"Line Content: {line}")
				raise WrongDataFormat("Missing entire key and value information.")
			else:
				result[t[0]] = t[1]
	else:
		for utt, utterance in target.items():
			declare.is_valid_string("utterance ID", utt)
			declare.is_valid_string("utterance", utterance)
		result = Transcription(target,name=name)

	sampleText = result.subset(nRandom=100)
	spaceCount = 0
	for key,value in sampleText.items():
		spaceCount += value.count(" ")
	if spaceCount < len(sampleText)//2:
		raise WrongDataFormat("The transcription doesn't seem to be separated by spaces or extremely short.")

	return result
