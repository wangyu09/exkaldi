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

from exkaldi.version import version as ExkaldiInfo
from exkaldi.version import WrongPath, WrongOperation, WrongDataFormat, UnsupportedType, ShellProcessError, KaldiProcessError
from exkaldi.utils.utils import run_shell_command, type_name, list_files
from exkaldi.core.archieve import BytesMatrix, BytesFeature, BytesCMVNStatistics, BytesProbability, BytesAlignmentTrans
from exkaldi.core.archieve import NumpyMatrix, NumpyFeature, NumpyCMVNStatistics, NumpyProbability, NumpyAlignmentTrans
from exkaldi.core.archieve import NumpyAlignment, NumpyAlignmentPhone, NumpyAlignmentPdf
from exkaldi.core.archieve import Transcription

def __read_data_from_file(fileName, useSuffix=None):
	'''
	Read data from file. If the file suffix is unknown, <useSuffix> should be assigned.
	'''
	if useSuffix != None:
		assert isinstance(useSuffix, str), "Expected <useSuffix> is a string."
		useSuffix = useSuffix.strip().lower()[-3:]
	else:
		useSuffix = ""

	assert useSuffix in ["", "scp", "ark", "npy"], f'Expected <useSuffix> is "ark", "scp" or "npy" but got "{useSuffix}".'

	if isinstance(fileName, str):
		if os.path.isdir(fileName):
			raise WrongOperation(f"Expected file name but got a directory:{fileName}.")
		else:
			allFiles = list_files(fileName)
	else:
		raise UnsupportedType(f'Expected <fileName> is file name-like string but got a {type_name(fileName)}.')

	allData_bytes = BytesMatrix()
	allData_numpy = NumpyMatrix()

	def loadNpyFile(fileName):
		try:
			temp = np.load(fileName, allow_pickle=True)
			data = {}
			#totalSize = 0
			for utt_mat in temp:
				data[utt_mat[0]] = utt_mat[1]
				#totalSize += sys.getsizeof(utt_mat[1])
			#if totalSize > 10000000000:
			#    print('Warning: Data is extramely large. It could not be used correctly sometimes.')                
		except:
			raise UnsupportedType(f'Expected "npy" data with exkaldi format but got {fileName}.')
		else:
			return NumpyMatrix(data)
	
	def loadArkScpFile(fileName, suffix):
		ExkaldiInfo.vertify_kaldi_existed()

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
			return BytesMatrix(out)

	for fileName in allFiles:
		sfx = fileName[-3:].lower()
		if sfx == "npy":
			allData_numpy += loadNpyFile(fileName)
		elif sfx in ["ark", "scp"]:
			allData_bytes += loadArkScpFile(fileName, sfx)
		elif useSuffix == "npy":
			allData_numpy += loadNpyFile(fileName)
		elif useSuffix in ["ark", "scp"]:
			allData_bytes += loadArkScpFile(fileName, useSuffix)
		else:
			raise UnsupportedType('Unknown file suffix. You can assign the <useSuffix> with "scp", "ark" or "npy".')
	
	if useSuffix == "":
		if allFiles[0][-3:].lower() == "npy":
			result = allData_numpy + allData_bytes.to_numpy()
		else:
			result = allData_bytes + allData_numpy.to_bytes()
	elif useSuffix == "npy":
		result = allData_numpy + allData_bytes.to_numpy()
	else:
		result = allData_bytes + allData_numpy.to_bytes() 

	result.check_format()
	return result

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
	assert isinstance(name, str) and len(name) > 0, "Name shoud be a string avaliable."

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
		result = __read_data_from_file(target, useSuffix)
		if isinstance(result, BytesMatrix):
			return BytesFeature(result.data, name)
		else:
			return NumpyFeature(result.data, name)	
			
	else:
		raise UnsupportedType(f"Expected Python dict, bytes object, exkaldi feature object or file path but got{type_name(target)}.")

def load_cmvn(target, name="cmvn", useSuffix=None):
	'''
	Load CMVN statistics data.

	Args:
		<target>: Python dict object, bytes object, exkaldi feature object, .ark file, .scp file, npy file.
		<name>: a string.
		<useSuffix>: a string. When target is file path, use this to specify file.
	Return:
		A BytesFeature or NumpyFeature object.
	'''
	assert isinstance(name, str) and len(name) > 0, "Name shoud be a string avaliable."

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
		result = __read_data_from_file(target, useSuffix)
		if isinstance(result, BytesMatrix):
			return BytesCMVNStatistics(result.data, name)
		else:
			return NumpyCMVNStatistics(result.data, name)	

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
	assert isinstance(name, str) and len(name) > 0, "Name shoud be a string avaliable."

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
		result = __read_data_from_file(target, useSuffix)
		if isinstance(result, BytesMatrix):
			return BytesProbability(result.data, name)
		else:
			return NumpyProbability(result.data, name)	
			
	else:
		raise UnsupportedType(f"Expected Python dict, bytes object, exkaldi feature object or file path but got{type_name(target)}.")

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
	assert isinstance(name, str) and len(name) > 0, "Name shoud be a string avaliable."

	ExkaldiInfo.vertify_kaldi_existed()

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
			return results

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

	elif type_name(target) in ["NumpyAlignment","NumpyAlignmentTrans","NumpyAlignmentPhone","NumpyAlignmentPdf", "BytesAlignmentTrans"]:
		result = copy.deepcopy(target)
		result.rename(name)
		return result

	elif isinstance(target, str):
				
		allFiles = list_files(target)

		results = {"NumpyAlignment": NumpyAlignment(),
				   "NumpyAlignmentTrans": NumpyAlignmentTrans(),
				   "NumpyAlignmentPhone": NumpyAlignmentPhone(),
				   "NumpyAlignmentPdf": NumpyAlignmentPdf(),
				   "BytesAlignmentTrans": BytesAlignmentTrans(),
				}

		for fileName in allFiles:
			fileName = os.path.abspath(fileName)

			if fileName.endswith(".npy"):
				temp = __read_data_from_file(fileName, "npy")
				if aliType is None:
					temp = NumpyAlignment(temp.data)
					results["NumpyAlignment"] += temp
				elif aliType == "transitionID":
					temp = NumpyAlignmentTrans(temp.data)
					results["NumpyAlignmentTrans"] += temp					
				elif aliType == "phoneID":
					temp = NumpyAlignmentPhone(temp.data)
					results["NumpyAlignmentPhone"] += temp	
				elif aliType == "pdfID":
					temp = NumpyAlignmentPdf(temp.data)
					results["NumpyAlignmentPdf"] += temp
				else:
					raise WrongOperation(f"<aliType> should be None, 'transitionID','phoneID' or 'pdfID' but got {aliType}.")
			
			else:
				if fileName.endswith('.gz'):
					cmd = f'gunzip -c {fileName}'
				else:
					cmd = f'cat {fileName}'
				
				if aliType is None or aliType == "transitionID":
					out, err, cod = run_shell_command(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
					if (isinstance(cod, int) and cod != 0 ) or out == b'':
						print(err.decode())
						raise ShellProcessError("Failed to get the alignment data from file.")
					else:
						temp = BytesAlignmentTrans(out)
						results["BytesAlignmentTrans"] += temp
				
				else:
					temp = tempfile.NamedTemporaryFile("wb+")
					try:
						if type_name(hmm) in ("HMM", "MonophoneHMM", "TriphoneHMM"):
							hmm.save(temp)
							hmmFileName = temp.name
						elif isinstance(hmm, str):
							if not os.path.isfile(hmm):
								raise WrongPath(f"No such file:{hmm}.")
							hmmFileName = hmm
						else:
							raise UnsupportedType(f"<hmm> should be a filePath or exkaldi HMM and its sub-class object. but got {type_name(hmm)}.") 

						if aliType == "phoneID":
							cmd += f" | ali-to-phones --per-frame=true {hmmFileName} ark:- ark,t:-"
							temp = transform(None, cmd)
							temp = NumpyAlignmentPhone(temp)
							results["NumpyAlignmentPhone"] += temp

						elif target == "pdfID":
							cmd = f" | ali-to-pdf {hmmFileName} ark:- ark,t:-"
							temp = transform(None, cmd)
							temp = NumpyAlignmentPdf(temp)
							results["NumpyAlignmentPdf"] += temp
						else:
							raise WrongOperation(f"<target> should be 'trainsitionID', 'phoneID' or 'pdfID' but got {target}.")
					
					finally:
						temp.close()

		finalResult = []
		for obj in results.values():
			if not obj.is_void:
				obj.rename(name)
				finalResult.append(obj)
		
		if len(finalResult) == 0:
			raise WrongOperation("<target> dose not include any data avaliable.")
		elif len(finalResult) == 1:
			finalResult = finalResult[0]
		
		return finalResult

def load_trans(target, name="transcription"):
	'''
	Load transcription from file.

	Args:
		<target>: transcription file path.
		<name>: a string.

	Return:
		An exkaldi Transcription object.
	'''
	if type_name(target) in ["dict", "Transcription", "ScriptTable"]:
		for utt, utterance in target.items():
			assert isinstance(utt, str) and len(utt) > 0, "Utterance ID should be a string."
			assert isinstance(utterance, str), "Utterance text should a string."
		
		return Transcription(target, name)
	
	elif isinstance(target, str):
		assert os.path.isfile(target), f"No such file:{target}."

		result = Transcription(name=name)
		result.load(target)

		return result
	
	else:
		raise UnsupportedType("<target> should be file path, dict object or ScriptTable object.")

