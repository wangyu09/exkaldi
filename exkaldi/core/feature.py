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

"""feature extracting functions"""

import os
import subprocess

from exkaldi.core.achivements import BytesFeature
from exkaldi.version import version as ExkaldiInfo
from exkaldi.version import WrongPath
from exkaldi.utils.utils import UnsupportedDataType, KaldiProcessError, WrongOperation
from exkaldi.utils.utils import run_shell_command, type_name, check_config

def __compute_feature(wavFile, kaldiTool, useSuffix=None, name="feat"):

	if useSuffix != None:
		assert isinstance(useSuffix, str), "Expected <useSuffix> is a string."
		useSuffix = useSuffix.strip().lower()[-3:]
	else:
		useSuffix = ""
	assert useSuffix in ["","scp","wav"], 'Expected <useSuffix> is "scp" or "wav".'

	ExkaldiInfo.vertify_kaldi_existed()

	if isinstance(wavFile, str):
		if os.path.isdir(wavFile):
			raise WrongOperation(f'Expected <wavFile> is file path but got a directory:{wavFile}.')
		else:
			out, err, _ = run_shell_command(f'ls {wavFile}', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
			if out == b'':
				raise WrongPath(f"No such file:{wavFile}.")
			else:
				allFiles = out.decode().strip().split('\n')
	else:
		raise UnsupportedDataType(f'Expected filename-like string but got a {type_name(wavFile)}.')
	
	results = []
	for wavFile in allFiles:
		wavFile = os.path.abspath(wavFile)
		if wavFile[-3:].lower() == "wav":
			dirName = os.path.dirname(wavFile)
			fileName = os.path.basename(wavFile)
			uttID = "".join(fileName[0:-4].split("."))
			cmd = f"echo {uttID} {wavFile} | {kaldiTool} scp:- ark:-"
		elif wavFile[-3:].lower() == 'scp':
			cmd = f"{kaldiTool} scp:{wavFile} ark:-"
		elif "wav" in useSuffix:
			dirName = os.path.dirname(wavFile)
			fileName = os.path.basename(wavFile)
			uttID = "".join(fileName[0:-4].split("."))
			cmd = f"echo {uttID} {wavFile} | {kaldiTool} scp:- ark:-"
		elif "scp" in useSuffix:        
			cmd = f"{kaldiTool} scp:{wavFile} ark:-" 
		else:
			raise UnsupportedDataType('Unknown file suffix. You can declare it by making <useSuffix> "wav" or "scp".')

		out, err, _ = run_shell_command(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		if out == b'':
			print(err.decode())
			raise KaldiProcessError(f'Failed to compute feature:{name}.')
		else:
			results.append(BytesFeature(out))

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
		<wavFile>: wave file or scp file. If it is wave file, use it's file name as utterance ID.
		<rate>: sample rate.
		<frameWidth>: sample windows width.
		<frameShift>: shift windows width.
		<melbins>: the numbers of mel filter banks.
		<featDim>: the output dinmensionality of MFCC feature.
		<windowType>: sample windows type.
		<useSuffix>: If the suffix of file is not .scp, use this to specify it.
		<config>: use this to assign more configures. If use it, all these configures above will be skipped.
		<name>: the name of feature.

		Some usual options can be assigned directly. If you want use more, set <config> = your-configure, but if you do this, these usual configures we provided will be ignored.
		You can use .check_config('compute_mfcc') function to get configure information that you can set.
		Also you can run shell command "compute-mfcc-feats" to look their meaning.

	Return:
		A exkaldi bytes feature object.
	'''
	assert isinstance(wavFile, str), f"<wavFile> should be a string but got {type_name(wavFile)}."
	assert isinstance(frameWidth, int) and frameWidth > 0,  f"<frameWidth> should be a positive int value but got {type_name(frameWidth)}."
	assert isinstance(frameShift, int) and frameShift > 0,  f"<frameShift> should be a positive int value but got {type_name(frameShift)}."
	assert frameWidth > frameShift,  f"<frameWidth> and <frameShift> is unavaliable."

	kaldiTool = 'compute-mfcc-feats'
	if config == None:    
		config = {}
		config["--allow-downsample"] = "true"
		config["--allow-upsample"] = "true"
		config["--sample-frequency"] = rate
		config["--frame-length"] = frameWidth
		config["--frame-shift"] = frameShift
		config["--num-mel-bins"] = melBins
		config["--num-ceps"] = featDim
		config["--window-type"] = windowType
	if check_config(name='compute_mfcc', config=config):
		for key in config.keys():
			kaldiTool += f" {key}={config[key]}"
	
	result = __compute_feature(wavFile, kaldiTool, useSuffix, name)
	return result

def compute_fbank(wavFile, rate=16000, frameWidth=25, frameShift=10, 
					melBins=23, windowType='povey', useSuffix=None,
					config=None, name="fbank"):
	'''
	Compute fbank feature.

	Args:
		<wavFile>: wave file or scp file. If it is wave file, use it's file name as utterance ID.
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
	assert isinstance(wavFile, str), f"<wavFile> should be a string but got {type_name(wavFile)}."
	assert isinstance(frameWidth, int) and frameWidth > 0,  f"<frameWidth> should be a positive int value but got {type_name(frameWidth)}."
	assert isinstance(frameShift, int) and frameShift > 0,  f"<frameShift> should be a positive int value but got {type_name(frameShift)}."
	assert frameWidth > frameShift,  f"<frameWidth> and <frameShift> is unavaliable."

	kaldiTool = 'compute-fbank-feats'
	if config == None:    
		config = {}
		config["--allow-downsample"] = "true"
		config["--allow-upsample"] = "true"
		config["--sample-frequency"] = rate
		config["--frame-length"] = frameWidth
		config["--frame-shift"] = frameShift
		config["--num-mel-bins"] = melBins
		config["--window-type"] = windowType
	if check_config(name='compute_fbank', config=config):
		for key in config.keys():
			kaldiTool += f' {key}={config[key]}'
	
	result = __compute_feature(wavFile, kaldiTool, useSuffix, name)
	return result

def compute_plp(wavFile, rate=16000, frameWidth=25, frameShift=10,
				melBins=23, featDim=13, windowType='povey', useSuffix=None,
				config=None, name="plp"):
	'''
	Compute fbank feature.

	Args:
		<wavFile>: wave file or scp file. If it is wave file, use it's file name as utterance ID.
		<rate>: sample rate.
		<frameWidth>: sample windows width.
		<frameShift>: shift windows width.
		<melbins>: the numbers of mel filter banks.
		<featDim>: the output dinmensionality of PLP feature.
		<windowType>: sample windows type.
		<useSuffix>: If the suffix of file is not .scp, use this to specify it.
		<config>: use this to assign more configures. If use it, all these configures above will be skipped.
		<name>: the name of feature.

		Some usual options can be assigned directly. If you want use more, set <config> = your-configure, but if you do this, these usual configures we provided will be ignored.
		You can use .check_config('compute_plp') function to get configure information that you can set.
		Also you can run shell command "compute-plp-feats" to look their meaning.

	Return:
		A exkaldi bytes feature object.
	'''
	assert isinstance(wavFile, str), f"<wavFile> should be a string but got {type_name(wavFile)}."
	assert isinstance(frameWidth, int) and frameWidth > 0,  f"<frameWidth> should be a positive int value but got {type_name(frameWidth)}."
	assert isinstance(frameShift, int) and frameShift > 0,  f"<frameShift> should be a positive int value but got {type_name(frameShift)}."
	assert frameWidth > frameShift,  f"<frameWidth> and <frameShift> is unavaliable."

	kaldiTool = 'compute-plp-feats'
	if config == None:    
		config = {}
		config["--allow-downsample"] = "true"
		config["--allow-upsample"] = "true"
		config["--sample-frequency"] = rate
		config["--frame-length"] = frameWidth
		config["--frame-shift"] = frameShift
		config["--num-mel-bins"] = melBins
		config["--num-ceps"] = featDim
		config["--window-type"] = windowType
	if check_config(name='compute_plp',config=config):
		for key in config.keys():
			kaldiTool += f' {key}={config[key]}'
	
	result = __compute_feature(wavFile, kaldiTool, useSuffix, name)
	return result
	
def compute_spectrogram(wavFile, rate=16000, frameWidth=25, frameShift=10,
						windowType='povey', useSuffix=None, config=None, name="spectrogram"):
	'''
	Compute power spectrogram feature.

	Args:
		<wavFile>: wave file or scp file. If it is wave file, use it's file name as utterance ID.
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
	assert isinstance(wavFile, str), f"<wavFile> should be a string but got {type_name(wavFile)}."
	assert isinstance(frameWidth, int) and frameWidth > 0,  f"<frameWidth> should be a positive int value but got {type_name(frameWidth)}."
	assert isinstance(frameShift, int) and frameShift > 0,  f"<frameShift> should be a positive int value but got {type_name(frameShift)}."
	assert frameWidth > frameShift,  f"<frameWidth> and <frameShift> is unavaliable."

	kaldiTool = 'compute-spetrogram-feats'
	if config == None: 
		config = {}
		config["--allow-downsample"] = "true"
		config["--allow-upsample"] = "true"
		config["--sample-frequency"] = rate
		config["--frame-length"] = frameWidth
		config["--frame-shift"] = frameShift
		config["--window-type"] = windowType
	if check_config(name='compute_spetrogram', config=config):
		for key in config.keys():
			kaldiTool += f' {key}={config[key]}'
	
	result = __compute_feature(wavFile, kaldiTool, useSuffix, name)
	return result
