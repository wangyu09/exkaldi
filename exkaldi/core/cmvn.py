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

"""CMVN"""
import subprocess
import tempfile
import math
import os

from exkaldi.version import version as ExkaldiInfo
from exkaldi.version import WrongPath
from exkaldi.utils.utils import run_shell_command, type_name

from exkaldi.core.achivements import UnsupportedDataType, KaldiProcessError
from exkaldi.core.achivements import BytesFeature, BytesCMVNStatistics

def use_cmvn(feat, cmvn, utt2spkFile=None, std=False, ordered=False):
	'''
	Apply CMVN statistics to feature.

	Args:
		<feat>: exkaldi feature object.
		<cmvn>: exkaldi CMVN statistics object.
		<utt2spkFile>: utt2spk file path.

	Return:
		A new feature object.
	''' 
	ExkaldiInfo.vertify_kaldi_existed()

	if type_name(feat) == "BytesFeature":
		if not ordered:
			feat = feat.sort(by="utt")
	elif type_name(feat) == "NumpyFeature":
		if ordered:
			feat = feat.to_bytes()
		else:
			feat = feat.sort(by="utt").to_bytes()
	else:
		raise UnsupportedDataType(f"Expected exkaldi feature but got {type_name(feat)}.")

	if type_name(cmvn) == "BytesCMVNStatistics":
		if not ordered:
			cmvn = cmvn.sort(by="utt")
	elif type_name(cmvn) == "NumpyCMVNStatistics":
		if ordered:
			cmvn = cmvn.to_bytes()
		else:
			cmvn = cmvn.sort(by="utt").to_bytes()
	else:
		raise UnsupportedDataType(f"Expected exkaldi CMVN statistics but got {type_name(cmvn)}.")

	cmvnTemp = tempfile.NamedTemporaryFile('wb+', suffix='.ark')
	utt2spkTemp = tempfile.NamedTemporaryFile('w+', encoding="utf-8")
	try:
		cmvnTemp.write(cmvn.data)
		cmvnTemp.seek(0)

		if std is True:
			stdOption = " --norm-vars true"
		else:
			stdOption = ""

		if utt2spkFile is None:
			cmd = f'apply-cmvn{stdOption} ark,c,cs:{cmvnTemp.name} ark,c,cs:- ark:-'
		else:
			assert isinstance(utt2spkFile, str), "<utt2spkFile> should be file path."
			if not os.path.isfile(utt2spkFile):
				raise WrongPath(f"No such file:{utt2spkFile}.")
			cmd = f"sort {utt2spkFile} > {utt2spkTemp.name}"
			out, err, code = run_shell_command(cmd, stderr=subprocess.PIPE)
			if (isinstance(code,int) and code != 0) or os.path.getsize(utt2spkTemp.name) == 0:
				print(err.decode())
				raise Exception("Failed to sort utt2spk file.")
			cmd = f'apply-cmvn{stdOption} --utt2spk=ark:{utt2spkTemp.name} ark,c,cs:{cmvnTemp.name} ark,c,cs:- ark:-'

		out, err, _ = run_shell_command(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, inputs=feat.data)

		if out == b'':
			print(err.decode())
			raise KaldiProcessError('Failed to apply CMVN statistics.')
		else:
			newName = f"cmvn({feat.name},{cmvn.name})"
			if type_name(feat) == "NumpyFeature":
				return BytesFeature(out, newName).to_numpy()
			else:
				return BytesFeature(out, newName)
	finally:
		cmvnTemp.close()
		utt2spkTemp.close()

def compute_cmvn_stats(feat, spk2uttFile=None, name="cmvn", ordered=False):
	'''
	Compute CMVN statistics.

	Args:
		<feat>: exkaldi feature object.
		<spk2uttFile>: spk2utt file.
		<name>: a string.

	Return:
		A exkaldi CMVN statistics object.
	''' 
	ExkaldiInfo.vertify_kaldi_existed()

	if type_name(feat) == "BytesFeature":
		if not ordered:
			feat = feat.sort("utt")
	elif type_name(feat) == "NumpyFeature":
		if ordered:
			feat = feat.to_bytes()
		else:
			feat = feat.sort("utt").to_bytes()
	else:
		raise UnsupportedDataType(f"Expected <feat> is a exkaldi feature object but got {type_name(feat)}.")
	
	utt2spkSorted = tempfile.NamedTemporaryFile("w+", encoding="utf-8")
	try:
		if spk2uttFile != None:
			assert isinstance(spk2uttFile, str), "<utt2spkFile> should be a file path."
			if not os.path.isfile(spk2uttFile):
				raise WrongPath(f"No such file:{spk2uttFile}.")
			else:
				cmd = f"sort {spk2uttFile} > {utt2spkSorted.name}"
				out, err, cod = run_shell_command(cmd, stderr=subprocess.PIPE)
				if (isinstance(cod, int) and cod != 0) or os.path.getsize(utt2spkSorted.name) == 0:
					print(err.decode())
					raise Exception("Failed to sort utt2spk file.")
				
				cmd = f'compute-cmvn-stats --spk2utt=ark,c,cs:{utt2spkSorted.name} ark,c,cs:- ark:-'
		else:
			cmd = 'compute-cmvn-stats ark,c,cs:- ark:-'

		out, err, _ = run_shell_command(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, inputs=feat.data)

		if len(out) == 0:
			print(err.decode())
			raise KaldiProcessError('Failed to compute CMVN statistics.')
		else:
			return BytesCMVNStatistics(out, name)
	finally:
		utt2spkSorted.close()

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
		raise UnsupportedDataType(f"Expected <feat> is an exkaldi feature object but got {type_name(feat)}.")
	
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
	out, err, _ = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=feat.data)
	if out == b'':
		print(err.decode())
		raise KaldiProcessError('Failed to use sliding CMVN.')
	else:
		newName = f"cmvn({feat.name},{windowsSize})"
		return BytesFeature(out,newName)
