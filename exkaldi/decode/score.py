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
"""Score the decoding results"""
import tempfile
from collections import namedtuple, Iterable
import os
import re
import subprocess

from exkaldi.version import version as ExkaldiInfo
from exkaldi.version import WrongPath
from exkaldi.utils.utils import type_name, flatten, run_shell_command
from exkaldi.utils.utils import UnsupportedDataType, WrongDataFormat, KaldiProcessError, WrongOperation

def wer(ref, hyp, ignore=None, mode='all'):
	'''
	Compute WER (word error rate) between <ref> and <hyp>. 

	Args:
		<ref>, <hyp>: exkaldi transcription object or file path.
		<ignore>: ignore a symbol.
		<mode>: "all" or "present".
	Return:
		a namedtuple of score information.
	'''
	assert mode in ['all', 'present'], 'Expected <mode> to be "present" or "all".'
	ExkaldiInfo.vertify_kaldi_existed()

	hypTemp = tempfile.NamedTemporaryFile("w+", suffix=".txt", encoding="utf-8")
	refTemp = tempfile.NamedTemporaryFile("w+", suffix=".txt", encoding="utf-8")
	try:
		if ignore is None:
			if type_name(hyp) == "Transcription":
				hyp.save(hypTemp)
				hypTemp.seek(0)
				hypFileName = hypTemp.name
			elif isinstance(hyp, str):
				if not os.path.isfile(hyp):
					raise WrongPath(f"No such file:{hyp}.")
				else:
					hypFileName = hyp
			else:
				raise UnsupportedDataType('<hyp> should be exkaldi Transcription object or file path.')

			if type_name(ref) == "Transcription":
				ref.save(refTemp)
				refTemp.seek(0)
				refFileName = refTemp.name
			elif isinstance(ref, str):
				if not os.path.isfile(ref):
					raise WrongPath(f"No such file:{ref}.")
				else:
					refFileName = ref
			else:
				raise UnsupportedDataType('<ref> should be exkaldi Transcription object or file path.')

			cmd = f'compute-wer --text --mode={mode} ark:{refFileName} ark,p:{hypFileName}'
			scoreOut, scoreErr, _ = run_shell_command(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		else:
			if type_name(hyp) == "Transcription":
				hyp = hyp.save()
			elif isinstance(hyp, str):
				if not os.path.isfile(hyp):
					raise WrongPath(f"No such file:{hyp}.")
				else:
					with open(hyp, "r", encoding="utf-8") as fr:
						hyp = fr.read()
			else:
				raise UnsupportedDataType('<hyp> should be exkaldi Transcription object or file path.')
			
			cmd = f'sed "s/{ignore} //g" > {hypTemp.name}'
			hypOut, err, _ = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=hyp.encode())
			if len(hypOut) == 0:
				print(err.decode())
				raise WrongDataFormat("<hyp> has wrong data formation.")

			if type_name(ref) == "Transcription":
				ref = ref.save()
			elif isinstance(ref, str):
				if not os.path.isfile(ref):
					raise WrongPath(f"No such file:{ref}.")
				else:
					with open(ref, "r", encoding="utf-8") as fr:
						ref = fr.read()
			else:
				raise UnsupportedDataType('<ref> should be exkaldi Transcription object or file path.')
			
			cmd = f'sed "s/{ignore} //g" > {refTemp.name}'
			refOut, err, _ = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=hyp.encode())
			if len(refOut) == 0:
				print(err.decode())
				raise WrongDataFormat("<ref> has wrong data formation.")

			cmd = f'compute-wer --text --mode={mode} ark:{refTemp.name} ark,p:{hypTemp.name}'
			scoreOut, scoreErr, _ = run_shell_command(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

	finally:
		hypTemp.close()
		refTemp.close()
	
	if len(scoreOut) == 0:
		print(scoreErr.decode())
		raise KaldiProcessError("Failed to compute WER.")

	else:
		out = scoreOut.decode().split("\n")
		pattern1 = '%WER (.*) \[ (.*) \/ (.*), (.*) ins, (.*) del, (.*) sub \]'
		pattern2 = "%SER (.*) \[ (.*) \/ (.*) \]"
		pattern3 = "Scored (.*) sentences, (.*) not present in hyp."
		s1 = re.findall(pattern1,out[0])[0]
		s2 = re.findall(pattern2,out[1])[0]
		s3 = re.findall(pattern3,out[2])[0]    

		return namedtuple("Score",["WER", "words", "ins", "del", "sub", "SER", "sentences", "wrongSentences", "missedSentences"])(
						float(s1[0]), #WER
						int(s1[2]),   #words
						int(s1[3]),	  #ins
						int(s1[4]),   #del
						int(s1[5]),   #sub
						float(s2[0]), #SER
						int(s2[1]),   #sentences
						int(s2[2]),   #wrong sentences
						int(s3[1])    #missed sentences
				)

def accuracy(ref, hyp, ignore=None, mode='all'):
	'''
	Score one-2-one matching score between two items.

	Args:
		<ref>, <hyp>: iterable objects like list, tuple or NumPy array. It will be flattened before scoring.
		<ignore>: Ignoring specific symbols.
		<model>: If <mode> is "all", compute one-one matching score. For example, <ref> is (1,2,3,4), and <hyp> is (1,2,2,4), the score will be 0.75.
				 If <mode> is "present", only the members of <hyp> which appeared in <ref> will be scored no matter which position it is. 
	Return:
		a namedtuple object of score information.
	'''
	assert type_name(ref)=="Transcription" or type_name(hyp) == "Transcription", "Exkaldi Transcription objects are unsupported in this function."

	assert mode in ['all','present'], 'Expected <mode> to be "present" or "all".'

	x = flatten(ref)
	x = list( filter(lambda i:i!=ignore, x) ) 
	y = flatten(hyp)
	y = list( filter(lambda i:i!=ignore, y) ) 

	if mode == 'all':
		i = 0
		score = 0
		while True:
			if i >= len(x) or i >= len(y):
				break
			elif x[i] == y[i]:
				score += 1
			i += 1
		if i < len(x) or i < len(y):
			raise WrongOperation('<ref> and <hyp> have different length to score.')
		else:
			if len(x) == 0:
				accuracy = 1.0
			else:
				accuracy = score/len(x)

			return namedtuple("Score",["accuracy", "items", "rightItems"])(
						accuracy, len(x), score
					)
	else:
		x = sorted(x)
		score = 0
		for i in y:
			if i in x:
				score += 1
		if len(y) == 0:
			if len(x) == 0:
				accuracy = 1.0
			else:
				accuracy = 0.0
		else:
			accuracy = score/len(y)
		
		return namedtuple("Score", ["accuracy", "items", "rightItems"])(
					accuracy, len(y), score
				)

def __edit_distance(ref, hyp, ignore=None):
	'''
	Compute edit-distance score.

	Args:
		<ref>, <hyp>: iterable objects like list, tuple or NumPy array. It will be flattened before scoring.
		<ignore>: Ignoring specific symbols.	 
	Return:
		a namedtuple object including score information.	
	'''
	assert isinstance(ref, Iterable), "<ref> is not a iterable object."
	assert isinstance(hyp, Iterable), "<hyp> is not a iterable object."
	
	x = flatten(ref)
	x = list( filter(lambda i:i!=ignore, x) ) 
	y = flatten(hyp)
	y = list( filter(lambda i:i!=ignore, y) ) 

	lenX = len(x)
	lenY = len(y)

	mapping = np.zeros((lenX+1,lenY+1))

	for i in range(lenX+1):
		mapping[i][0] = i
	for j in range(lenY+1):
		mapping[0][j] = j
	for i in range(1,lenX+1):
		for j in range(1,lenY+1):
			if x[i-1] == y[j-1]:
				delta = 0
			else:
				delta = 1       
			mapping[i][j] = min(mapping[i-1][j-1]+delta, min(mapping[i-1][j]+1, mapping[i][j-1]+1))
	
	score = int(mapping[lenX][lenY])
	return namedtuple("Score",["editDistance", "items"])(
				score, len(x)
			)

def edit_distance(ref, hyp, ignore=None, mode='present'):
	'''
	Compute edit-distance score.

	Args:
		<ref>, <hyp>: Transcription objects or iterable objects like list, tuple or NumPy array. It will be flattened before scoring.
		<ignore>: Ignoring specific symbols.
		<mode>: When both are Transcription objects, if mode is 'present', skip the missed utterances.
	Return:
		a namedtuple object including score information.	
	'''
	if type_name(hyp) == "Transcription" and type_name(hyp) == "Transcription":
		allED = 0
		words = 0
		SER = 0
		sentences = 0
		wrongSentences = 0
		missedSentences = 0

		ref = ref.sort()
		hyp = hyp.sort()
		for utt, hypTrans in hyp.items():
			try:
				refTrans = ref[utt]
			except KeyError as e:
				if mode == "all":
					raise Exception("Missing transcription in reference, set <mode> as 'all' to skip it.")
				else:
					missedSentences += 1
			else:
				sentences += 1
				refTrans = refTrans.split()
				hypTrans = hypTrans.split()
				ed, wds = __edit_distance(refTrans, hypTrans, ignore=ignore)
				allED += ed
				words += wds
				if ed > 0:
					wrongSentences += 1
		if sentences == 0:
			raise Exception("Missing all transcription in reference.")

		return namedtuple("Score",["editDistance", "words", "SER", "sentences", "wrongSentences", "missedSentences"])(
				allED, words, wrongSentences/sentences, sentences, wrongSentences, missedSentences
				)
	
	else:
		if type_name(hyp) == "Transcription" or type_name(hyp) == "Transcription":
			raise WrongOperation("<hyp> and <ref> should be exkaldi Transcription simultaneously.")
		else:
			ed, wds = __edit_distance(ref, hyp, ignore=ignore)
			return namedtuple("Score",["editDistance", "words"])(ed, wds)			

