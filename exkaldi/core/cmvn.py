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
from exkaldi.version import WrongPath, UnsupportedType, KaldiProcessError
from exkaldi.utils.utils import run_shell_command, type_name
from exkaldi.core.achivements import BytesFeature, BytesCMVNStatistics, ScriptTable

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

	cmvnTemp = tempfile.NamedTemporaryFile('wb+', suffix='.ark')
	utt2spkTemp = tempfile.NamedTemporaryFile('w+', encoding="utf-8")
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
