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

"""feature processing functions"""
import subprocess
import tempfile
import math
import os
import numpy as np
from io import BytesIO
import struct

from exkaldi.version import version as ExkaldiInfo
from exkaldi.version import WrongPath, UnsupportedType, KaldiProcessError, WrongOperation, ShellProcessError, WrongDataFormat
from exkaldi.utils.utils import run_shell_command, type_name, check_config
from exkaldi.core.achivements import BytesFeature, BytesCMVNStatistics, ScriptTable

def __compute_feature(wavFile, kaldiTool, useSuffix=None, name="feat"):

	if useSuffix != None:
		assert isinstance(useSuffix, str), "Expected <useSuffix> is a string."
		useSuffix = useSuffix.strip().lower()[-3:]
	else:
		useSuffix = ""
	assert useSuffix in ["","scp","wav"], 'Expected <useSuffix> is "scp" or "wav".'

	ExkaldiInfo.vertify_kaldi_existed()

	wavFileTemp = tempfile.NamedTemporaryFile("w+", suffix=".scp", encoding="utf-8")
	try:
		if isinstance(wavFile, str):
			if os.path.isdir(wavFile):
				raise WrongOperation(f'Expected <wavFile> is file path but got a directory:{wavFile}.')
			else:
				out, err, cod = run_shell_command(f'ls {wavFile}', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
				if out == b'':
					raise WrongPath(f"No such file:{wavFile}.")
				else:
					allFiles = out.decode().strip().split('\n')
		elif isinstance(wavFile, ScriptTable):
			wavFile = wavFile.sort()
			wavFile.save(wavFileTemp)
			allFiles = [wavFileTemp.name,]
		else:
			raise UnsupportedType(f'Expected filename-like string but got a {type_name(wavFile)}.')
		
		results = []
		for wavFile in allFiles:
			wavFile = os.path.abspath(wavFile)
			if wavFile[-3:].lower() == "wav":
				dirName = os.path.dirname(wavFile)
				fileName = os.path.basename(wavFile)
				uttID = "".join(fileName[0:-4].split("."))
				cmd = f"echo {uttID} {wavFile} | {kaldiTool} scp,p:- ark:-"
			elif wavFile[-3:].lower() == 'scp':
				cmd = f"{kaldiTool} scp,p:{wavFile} ark:-"
			elif "wav" in useSuffix:
				dirName = os.path.dirname(wavFile)
				fileName = os.path.basename(wavFile)
				uttID = "".join(fileName[0:-4].split("."))
				cmd = f"echo {uttID} {wavFile} | {kaldiTool} scp,p:- ark:-"
			elif "scp" in useSuffix:        
				cmd = f"{kaldiTool} scp,p:{wavFile} ark:-" 
			else:
				raise UnsupportedType('Unknown file suffix. You can declare it by making <useSuffix> "wav" or "scp".')

			out, err, cod = run_shell_command(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
			if (isinstance(out, int) and cod != 0) or out == b'':
				print(err.decode())
				raise KaldiProcessError(f'Failed to compute feature:{name}.')
			else:
				results.append(BytesFeature(out))
	finally:
		wavFileTemp.close()

	if len(results) == 0:
		raise WrongOperation("No any feature date in file path.")
	else:
		result = results[0]
		for i in results[1:]:
			result += i
		result.rename(name)
		return result	

def compute_mfcc(wavFile, rate=16000, frameWidth=25, frameShift=10, 
				melBins=23, featDim=13, windowType='povey', useSuffix=None,
				config=None, name="mfcc"):
	'''
	Compute MFCC feature.

	Args:
		<wavFile>: wave file or scp file or exkaldi SriptTable object. If it is wave file, use it's file name as utterance ID.
		<rate>: sample rate.
		<frameWidth>: sample windows width.
		<frameShift>: shift windows width.
		<melbins>: the numbers of mel filter banks.
		<featDim>: the output dinmensionality of MFCC feature.
		<windowType>: sample windows type.
		<useSuffix>: If the suffix of file is not .scp, use this to specify it.
		<config>: use this to assign more extra optional configures.
		<name>: the name of feature.

		Some usual options can be assigned directly. If you want use more, set <config> = your-configure.
		You can use .check_config('compute_mfcc') function to get configure information that you can set.
		Also you can run shell command "compute-mfcc-feats" to look their meaning.

	Return:
		A exkaldi bytes feature object.
	'''
	assert isinstance(frameWidth, int) and frameWidth > 0,  f"<frameWidth> should be a positive int value but got {type_name(frameWidth)}."
	assert isinstance(frameShift, int) and frameShift > 0,  f"<frameShift> should be a positive int value but got {type_name(frameShift)}."
	assert frameWidth > frameShift,  f"<frameWidth> and <frameShift> is unavaliable."
	assert windowType in ["hamming","hanning","povey","rectangular","blackmann"], f'<windowType> should be "hamming","hanning","povey","rectangular","blackmann", but got: {windowType}.'

	kaldiTool = 'compute-mfcc-feats --allow-downsample --allow-upsample '
	kaldiTool += f'--sample-frequency={rate} '
	kaldiTool += f'--frame-length={frameWidth} '
	kaldiTool += f'--frame-shift={frameShift} '
	kaldiTool += f'--num-mel-bins={melBins} '
	kaldiTool += f'--num-ceps={featDim} '
	kaldiTool += f'--window-type={windowType} '

	if config is not None:
		if check_config(name='compute_mfcc', config=config):
			for key,value in config.items():
				if isinstance(value, bool):
					if value is True:
						kaldiTool += f"{key} "
				else:
					kaldiTool += f" {key}={value}"
	
	result = __compute_feature(wavFile, kaldiTool, useSuffix, name)
	return result

def compute_fbank(wavFile, rate=16000, frameWidth=25, frameShift=10, 
					melBins=23, windowType='povey', useSuffix=None,
					config=None, name="fbank"):
	'''
	Compute fbank feature.

	Args:
		<wavFile>: wave file or scp file or exkaldi SriptTable object. If it is wave file, use it's file name as utterance ID.
		<rate>: sample rate.
		<frameWidth>: sample windows width.
		<frameShift>: shift windows width.
		<melbins>: the numbers of mel filter banks.
		<windowType>: sample windows type.
		<useSuffix>: If the suffix of file is not .scp, use this to specify it.
		<config>: use this to assign more configures. If use it, all these configures above will be skipped.
		<name>: the name of feature.
		
		Some usual options can be assigned directly. If you want use more, set <config> = your-configure, but if you do this, these usual configures we provided will be ignored.
		You can use .check_config('compute_fbank') function to get configure information that you can set.
		Also you can run shell command "compute-fbank-feats" to look their meaning.

	Return:
		A exkaldi bytes feature object.
	'''
	assert isinstance(frameWidth, int) and frameWidth > 0,  f"<frameWidth> should be a positive int value but got {type_name(frameWidth)}."
	assert isinstance(frameShift, int) and frameShift > 0,  f"<frameShift> should be a positive int value but got {type_name(frameShift)}."
	assert frameWidth > frameShift,  f"<frameWidth> and <frameShift> is unavaliable."
	assert windowType in ["hamming","hanning","povey","rectangular","blackmann"], f'<windowType> should be "hamming","hanning","povey","rectangular","blackmann", but got: {windowType}.'

	kaldiTool = 'compute-fbank-feats --allow-downsample --allow-upsample '
	kaldiTool += f'--sample-frequency={rate} '
	kaldiTool += f'--frame-length={frameWidth} '
	kaldiTool += f'--frame-shift={frameShift} '
	kaldiTool += f'--num-mel-bins={melBins} '
	kaldiTool += f'--window-type={windowType} '

	if config is not None:
		if check_config(name='compute_fbank', config=config):
			for key,value in config.items():
				if isinstance(value, bool):
					if value is True:
						kaldiTool += f"{key} "
				else:
					kaldiTool += f" {key}={value}"
	
	result = __compute_feature(wavFile, kaldiTool, useSuffix, name)
	return result

def compute_plp(wavFile, rate=16000, frameWidth=25, frameShift=10,
				melBins=23, featDim=13, windowType='povey', useSuffix=None,
				config=None, name="plp"):
	'''
	Compute fbank feature.

	Args:
		<wavFile>: wave file or scp file or exkaldi SriptTable object. If it is wave file, use it's file name as utterance ID.
		<rate>: sample rate.
		<frameWidth>: sample windows width.
		<frameShift>: shift windows width.
		<melbins>: the numbers of mel filter banks.
		<featDim>: the output dinmensionality of PLP feature.
		<windowType>: sample windows type.
		<useSuffix>: If the suffix of file is not .scp, use this to specify it.
		<config>: use this to assign more configures.
		<name>: the name of feature.

		Some usual options can be assigned directly. If you want use more, set <config> = your-configure.
		You can use .check_config('compute_plp') function to get configure information that you can set.
		Also you can run shell command "compute-plp-feats" to look their meaning.

	Return:
		A exkaldi bytes feature object.
	'''
	assert isinstance(frameWidth, int) and frameWidth > 0,  f"<frameWidth> should be a positive int value but got {type_name(frameWidth)}."
	assert isinstance(frameShift, int) and frameShift > 0,  f"<frameShift> should be a positive int value but got {type_name(frameShift)}."
	assert frameWidth > frameShift,  f"<frameWidth> and <frameShift> is unavaliable."
	assert windowType in ["hamming","hanning","povey","rectangular","blackmann"], f'<windowType> should be "hamming","hanning","povey","rectangular","blackmann", but got: {windowType}.'

	kaldiTool = 'compute-plp-feats --allow-downsample --allow-upsample '
	kaldiTool += f'--sample-frequency={rate} '
	kaldiTool += f'--frame-length={frameWidth} '
	kaldiTool += f'--frame-shift={frameShift} '
	kaldiTool += f'--num-mel-bins={melBins} '
	kaldiTool += f'--num-ceps={featDim} '
	kaldiTool += f'--window-type={windowType} '

	if config is not None:
		if check_config(name='compute_plp', config=config):
			for key,value in config.items():
				if isinstance(value, bool):
					if value is True:
						kaldiTool += f"{key} "
				else:
					kaldiTool += f" {key}={value}"
	
	result = __compute_feature(wavFile, kaldiTool, useSuffix, name)
	return result
	
def compute_spectrogram(wavFile, rate=16000, frameWidth=25, frameShift=10,
						windowType='povey', useSuffix=None, config=None, name="spectrogram"):
	'''
	Compute power spectrogram feature.

	Args:
		<wavFile>: wave file or scp file or exkaldi SriptTable object. If it is wave file, use it's file name as utterance ID.
		<rate>: sample rate.
		<frameWidth>: sample windows width.
		<frameShift>: shift windows width.
		<windowType>: sample windows type.
		<useSuffix>: If the suffix of file is not .scp, use this to specify it.
		<config>: use this to assign more configures. If use it, all these configures above will be skipped.
		<name>: the name of feature.

		Some usual options can be assigned directly. If you want use more, set <config> = your-configure, but if you do this, these usual configures we provided will be ignored.
		You can use .check_config('compute_spectrogram') function to get configure information that you can set.
		Also you can run shell command "compute-spectrogram-feats" to look their meaning.

	Return:
		A exkaldi bytes feature object.
	'''
	assert isinstance(frameWidth, int) and frameWidth > 0,  f"<frameWidth> should be a positive int value but got {type_name(frameWidth)}."
	assert isinstance(frameShift, int) and frameShift > 0,  f"<frameShift> should be a positive int value but got {type_name(frameShift)}."
	assert frameWidth > frameShift,  f"<frameWidth> and <frameShift> is unavaliable."
	assert windowType in ["hamming","hanning","povey","rectangular","blackmann"], f'<windowType> should be "hamming","hanning","povey","rectangular","blackmann", but got: {windowType}.'

	kaldiTool = 'compute-spetrogram-feats --allow-downsample --allow-upsample '
	kaldiTool += f'--sample-frequency={rate} '
	kaldiTool += f'--frame-length={frameWidth} '
	kaldiTool += f'--frame-shift={frameShift} '
	kaldiTool += f'--window-type={windowType} '

	if config is not None:
		if check_config(name='compute_spetrogram', config=config):
			for key,value in config.items():
				if isinstance(value, bool):
					if value is True:
						kaldiTool += f"{key} "
				else:
					kaldiTool += f" {key}={value}"
	
	result = __compute_feature(wavFile, kaldiTool, useSuffix, name)
	return result

def transform_feat(feat, matrixFile):
	'''
	Transform feat by a transform matrix. Typically, LDA, MLLt matrixes.

	Args:
		<feat>: exkaldi feature object.
		<matrixFile>: file name.
	
	Return:
		a new exkaldi feature object.
	'''
	assert isinstance(matrixFile, str), f"<transformMatrix> should be a file path but got: {type_name(matrixFile)}."
	if not os.path.isfile(matrixFile):
		raise WrongPath(f"No such file: {matrixFile}.")

	if type_name(feat) == "BytesFeature":
		bytesFlag = True
	elif type_name(feat) == "NumpyFeature":
		bytesFlag = False
		feat = feat.to_bytes()
	else:
		raise UnsupportedType(f"<feat> should exkaldi feature object but got: {type_name(feat)}.")
	
	cmd = f'transform-feats {matrixFile} ark:- ark:-'

	out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=feat.data)

	if cod != 0 :
		print(err.decode())
		raise KaldiProcessError("Failed to transform feature.")
	else:
		newName = f"tansform({feat.name})"
		newFeat = BytesFeature(out, name=newName)
		if bytesFlag:
			return newFeat
		else:
			return newFeat.to_numpy()

def use_fmllr(feat, transMatrix, utt2spkFile):
	'''
	Transform feat by a transform matrix. Typically, LDA, MLLt matrixes.

	Args:
		<feat>: exkaldi feature object.
		<transFile>: exkaldi fMLLR transform matrix object.
		<utt2spkFile>: utt2spk file name.
	
	Return:
		a new exkaldi feature object.
	'''
	if type_name(feat) == "BytesFeature":
		bytesFlag = True
		feat = feat.sort(by="utt")
	elif type_name(feat) == "NumpyFeature":
		bytesFlag = False
		feat = feat.sort(by="utt").to_bytes()
	else:
		raise UnsupportedType(f"<feat> should exkaldi feature object but got: {type_name(feat)}.")

	if type_name(transMatrix) == "BytesFmllrMatrix":
		transMatrix = transMatrix.sort(by="utt")
	elif type_name(transMatrix) == "NumpyFmllrMatrix":
		transMatrix = transMatrix.sort(by="utt").to_bytes()
	else:
		raise UnsupportedType(f"<transMatrix> should exkaldi fMLLR transform matrix object but got: {type_name(transMatrix)}.")
	
	transTemp = tempfile.NamedTemporaryFile("wb+", suffix="_trans.ark")
	try:
		transTemp.write(transMatrix.data)
		transTemp.seek(0)

		cmd = f'transform-feats --utt2spk=ark:{utt2spkFile} ark:{transTemp.name} ark:- ark:-'

		out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=feat.data)

		if cod != 0 :
			print(err.decode())
			raise KaldiProcessError("Failed to transform feature to fMLLR feature.")
		else:
			newName = f"fmllr({feat.name})"
			newFeat = BytesFeature(out, name=newName)
			if bytesFlag:
				return newFeat
			else:
				return newFeat.to_numpy()
	finally:
		transTemp.close()

def use_cmvn(feat, cmvn, utt2spk=None, std=False):
	'''
	Apply CMVN statistics to feature.

	Args:
		<feat>: exkaldi feature object.
		<cmvn>: exkaldi CMVN statistics object.
		<utt2spk>: utt2spk file path or ScriptTable object.
		<std>: If true, apply std normalization.

	Return:
		A new feature object.
	''' 
	ExkaldiInfo.vertify_kaldi_existed()

	if type_name(feat) == "BytesFeature":
		feat = feat.sort(by="utt")
	elif type_name(feat) == "NumpyFeature":
		feat = feat.sort(by="utt").to_bytes()
	else:
		raise UnsupportedType(f"Expected exkaldi feature but got {type_name(feat)}.")

	if type_name(cmvn) == "BytesCMVNStatistics":
		cmvn = cmvn.sort(by="utt")
	elif type_name(cmvn) == "NumpyCMVNStatistics":
		cmvn = cmvn.sort(by="utt").to_bytes()
	else:
		raise UnsupportedType(f"Expected exkaldi CMVN statistics but got {type_name(cmvn)}.")

	cmvnTemp = tempfile.NamedTemporaryFile('wb+', suffix='_cmvn.ark')
	utt2spkTemp = tempfile.NamedTemporaryFile('w+', suffix="_utt2spk",encoding="utf-8")
	try:
		cmvnTemp.write(cmvn.data)
		cmvnTemp.seek(0)

		if std is True:
			stdOption = " --norm-vars true"
		else:
			stdOption = ""

		if utt2spk is None:
			cmd = f'apply-cmvn{stdOption} ark:{cmvnTemp.name} ark:- ark:-'
		else:
			if isinstance(utt2spk, str):
				if not os.path.isfile(utt2spk):
					raise WrongPath(f"No such file:{utt2spk}.")
				utt2spkSorted = ScriptTable(name="utt2spk").load(utt2spk).sort()
				utt2spkSorted.save(utt2spkTemp)
			elif isinstance(utt2spk, ScriptTable):
				utt2spkSorted = utt2spk.sort()
				utt2spkSorted.save(utt2spkTemp)
			else:
				raise UnsupportedType(f"<utt2spk> should be a file path or ScriptTable object but got {type_name(utt2spk)}.")
			utt2spkTemp.seek(0)	

			cmd = f'apply-cmvn{stdOption} --utt2spk=ark:{utt2spkTemp.name} ark:{cmvnTemp.name} ark:- ark:-'	

		out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=feat.data)

		if (isinstance(cod,int) and cod != 0) or out == b'':
			print(err.decode())
			raise KaldiProcessError('Failed to apply CMVN statistics.')
		else:
			newName = f"cmvn({feat.name},{cmvn.name})"
			if type_name(feat) == "NumpyFeature":
				return BytesFeature(out, newName, indexTable=None).to_numpy()
			else:
				return BytesFeature(out, newName, indexTable=None)
	finally:
		cmvnTemp.close()
		utt2spkTemp.close()

def compute_cmvn_stats(feat, spk2utt=None, name="cmvn"):
	'''
	Compute CMVN statistics.

	Args:
		<feat>: exkaldi feature object.
		<spk2utt>: spk2utt file or exkaldi ScriptTable object.
		<name>: a string.

	Return:
		A exkaldi CMVN statistics object.
	''' 
	ExkaldiInfo.vertify_kaldi_existed()

	if type_name(feat) == "BytesFeature":
		feat = feat.sort("utt")
	elif type_name(feat) == "NumpyFeature":
		feat = feat.sort("utt").to_bytes()
	else:
		raise UnsupportedType(f"Expected <feat> is a exkaldi feature object but got {type_name(feat)}.")
	
	spk2uttTemp = tempfile.NamedTemporaryFile("w+", encoding="utf-8")
	try:
		if spk2utt is None:
			cmd = 'compute-cmvn-stats ark:- ark:-'
		else:
			if isinstance(spk2utt, str):
				if not os.path.isfile(spk2utt):
					raise WrongPath(f"No such file:{spk2utt}.")
				spk2uttSorted = ScriptTable(name="spk2utt").load(spk2utt).sort()
				spk2uttSorted.save(spk2uttTemp)
			elif isinstance(spk2utt, ScriptTable):
				spk2uttSorted = spk2utt.sort()
				spk2uttSorted.save(spk2uttTemp)
			else:
				raise UnsupportedType(f"<spk2utt> should be a file path or ScriptTable object but got {type_name(spk2utt)}.")			
			spk2uttTemp.seek(0)

			cmd = f'compute-cmvn-stats --spk2utt=ark:{spk2uttTemp.name} ark:- ark:-'

		out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=feat.data)

		if (isinstance(cod,int) and cod != 0) or out == b'':
			print(err.decode())
			raise KaldiProcessError('Failed to compute CMVN statistics.')
		else:
			return BytesCMVNStatistics(out, name, indexTable=None)
	finally:
		spk2uttTemp.close()

def use_cmvn_sliding(feat, windowsSize=None, std=False):
	'''
	Allpy sliding CMVN statistics.

	Args:
		<feat>: exkaldi feature object.
		<windowsSize>: windows size, If None, use windows size larger than the frames of feature.
		<std>: a bool value.

	Return:
		An exkaldi feature object.
	'''
	ExkaldiInfo.vertify_kaldi_existed()

	if isinstance(feat, BytesFeature):
		pass
	elif type_name(feat) == "NumpyFeature":
		feat = feat.to_bytes()
	else:
		raise UnsupportedType(f"Expected <feat> is an exkaldi feature object but got {type_name(feat)}.")
	
	if windowsSize == None:
		featLen = feat.lens[1]
		maxLen = max([length for utt, length in featLen])
		windowsSize = math.ceil(maxLen/100)*100
	else:
		assert isinstance(windowsSize,int), "Expected <windowsSize> is an int value."

	if std==True:
		std='true'
	else:
		std='false'

	cmd = f'apply-cmvn-sliding --cmn-window={windowsSize} --min-cmn-window=100 --norm-vars={std} ark:- ark:-'
	out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=feat.data)
	if (isinstance(cod,int) and cod != 0) or out == b'':
		print(err.decode())
		raise KaldiProcessError('Failed to use sliding CMVN.')
	else:
		newName = f"cmvn({feat.name},{windowsSize})"
		return BytesFeature(out, newName, indexTable=None)

def decompress_feat(feat):
	'''
	Decompress a kaldi conpressed feature whose data-type is "CM"
	
	Args:
		<feat>: an exkaldi feature object.
	Return:
		An new exkaldi feature object.

	This function is a cover of kaldi-io-for-python tools. 
	For more information about it, please access to https://github.com/vesis84/kaldi-io-for-python/blob/master/kaldi_io/kaldi_io.py 
	'''
	assert isinstance(feat, BytesFeature), "Expected <feat> is a exkaldi bytes feature object."

	def _read_compressed_mat(fd):

		# Format of header 'struct',
		global_header = np.dtype([('minvalue','float32'),('range','float32'),('num_rows','int32'),('num_cols','int32')]) # member '.format' is not written,
		per_col_header = np.dtype([('percentile_0','uint16'),('percentile_25','uint16'),('percentile_75','uint16'),('percentile_100','uint16')])

		# Read global header,
		globmin, globrange, rows, cols = np.frombuffer(fd.read(16), dtype=global_header, count=1)[0]
		cols = int(cols)
		rows = int(rows)

		# The data is structed as [Colheader, ... , Colheader, Data, Data , .... ]
		#                         {           cols           }{     size         }
		col_headers = np.frombuffer(fd.read(cols*8), dtype=per_col_header, count=cols)
		col_headers = np.array([np.array([x for x in y]) * globrange * 1.52590218966964e-05 + globmin for y in col_headers], dtype=np.float32)
		data = np.reshape(np.frombuffer(fd.read(cols*rows), dtype='uint8', count=cols*rows), newshape=(cols,rows)) # stored as col-major,

		mat = np.zeros((cols,rows), dtype='float32')
		p0 = col_headers[:, 0].reshape(-1, 1)
		p25 = col_headers[:, 1].reshape(-1, 1)
		p75 = col_headers[:, 2].reshape(-1, 1)
		p100 = col_headers[:, 3].reshape(-1, 1)
		mask_0_64 = (data <= 64)
		mask_193_255 = (data > 192)
		mask_65_192 = (~(mask_0_64 | mask_193_255))

		mat += (p0  + (p25 - p0) / 64. * data) * mask_0_64.astype(np.float32)
		mat += (p25 + (p75 - p25) / 128. * (data - 64)) * mask_65_192.astype(np.float32)
		mat += (p75 + (p100 - p75) / 63. * (data - 192)) * mask_193_255.astype(np.float32)

		return mat.T,rows,cols        
	
	with BytesIO(feat.data) as sp:
		newData = []

		while True:
			data = b''
			utt = ''
			while True:
				char = sp.read(1)
				data += char
				char = char.decode()
				if (char == '') or (char == ' '):break
				utt += char
			utt = utt.strip()
			if utt == '':break
			binarySymbol = sp.read(2)
			data += binarySymbol
			binarySymbol = binarySymbol.decode()
			if binarySymbol == '\0B':
				dataType = sp.read(3).decode()
				if dataType == 'CM ':
					data += 'FM '.encode()
					matrix,rows,cols = _read_compressed_mat(sp)
					data += '\04'.encode()
					data += struct.pack(np.dtype('uint32').char, rows)
					data += '\04'.encode()
					data += struct.pack(np.dtype('uint32').char, cols)
					data += matrix.tobytes()
					newData.append(data)
				else:
					raise UnsupportedType("This is not a compressed binary data.")
			else:
				raise WrongDataFormat('Miss right binary symbol.')

	return BytesFeature(b''.join(newData), name=feat.name)