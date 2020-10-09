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
"""Score the decoding results"""
import os
import re
from collections import namedtuple,Iterable

from exkaldi.version import info as ExKaldiInfo
from exkaldi.error import *
from exkaldi.utils.utils import type_name,flatten,run_shell_command
from exkaldi.utils.utils import FileHandleManager
from exkaldi.utils import declare
from exkaldi.core.load import load_transcription
from exkaldi.nn.nn import pure_edit_distance

def wer(ref,hyp,ignore=None,mode='all'):
	'''
	Compute WER (word error rate) between <ref> and <hyp>. 

	Args:
		<ref>,<hyp>: exkaldi transcription object or file path.
		<ignore>: ignore a symbol.
		<mode>: "all" or "present".

	Return:
		a namedtuple of score information.
	'''
	declare.is_potential_transcription("ref",ref)
	declare.is_potential_transcription("hyp",hyp)
	declare.is_instances("mode",mode,['all','present'])
	declare.kaldi_existed()

	if ignore is not None:
		declare.is_valid_string("ignore",ignore)

	with FileHandleManager() as fhm:

		if ignore is None:

			if type_name(hyp) == "Transcription":
				hypTemp = fhm.create("w+",suffix=".txt",encoding="utf-8")
				hyp.save(hypTemp)
				hyp = hypTemp.name

			if type_name(ref) == "Transcription":
				refTemp = fhm.create("w+",suffix=".txt",encoding="utf-8")
				ref.save(refTemp)
				ref = refTemp.name

			cmd = f'compute-wer --text --mode={mode} ark:{ref} ark,p:{hyp}'
			scoreOut,scoreErr,_ = run_shell_command(cmd,stdout="PIPE",stderr="PIPE")

		else:
			# remove the ingored symbol in hyp
			if type_name(hyp) == "Transcription":
				hyp = hyp.save()
			else:
				with open(hyp,"r",encoding="utf-8") as fr:
					hyp = fr.read()
			hypTemp = fhm.create("w+",suffix=".txt",encoding="utf-8")
			cmd = f'sed "s/{ignore} //g" > {hypTemp.name}'
			hypOut,err,_ = run_shell_command(cmd,stdin="PIPE",stdout="PIPE",stderr="PIPE",inputs=hyp)
			if len(hypOut) == 0:
				raise WrongDataFormat("<hyp> has wrong data formation.",err.decode())
			# remove the ingored symbol in ref
			if type_name(ref) == "Transcription":
				ref = ref.save()
			else:
				with open(ref,"r",encoding="utf-8") as fr:
					ref = fr.read()
			refTemp = fhm.create("w+",suffix=".txt",encoding="utf-8")
			cmd = f'sed "s/{ignore} //g" > {refTemp.name}'
			refOut,err,cod = run_shell_command(cmd,stdin="PIPE",stdout="PIPE",stderr="PIPE",inputs=ref)
			if cod != 0 or len(refOut) == 0:
				raise WrongDataFormat("<ref> has wrong data formation.",err.decode())
			# score
			cmd = f'compute-wer --text --mode={mode} ark:{refTemp.name} ark,p:{hypTemp.name}'
			scoreOut,scoreErr,_ = run_shell_command(cmd,stdout="PIPE",stderr="PIPE")
	
	if len(scoreOut) == 0:
		raise KaldiProcessError("Failed to compute WER.",scoreErr.decode())
	else:
		out = scoreOut.decode().split("\n")
		pattern1 = '%WER (.*) \[ (.*) \/ (.*),(.*) ins,(.*) del,(.*) sub \]'
		pattern2 = "%SER (.*) \[ (.*) \/ (.*) \]"
		pattern3 = "Scored (.*) sentences,(.*) not present in hyp."
		s1 = re.findall(pattern1,out[0])[0]
		s2 = re.findall(pattern2,out[1])[0]
		s3 = re.findall(pattern3,out[2])[0]    

		return namedtuple("Score",["WER","words","insErr","delErr","subErr","SER","sentences","wrongSentences","missedSentences"])(
						float(s1[0]),#WER
						int(s1[2]),  #words
						int(s1[3]),	  #ins
						int(s1[4]),  #del
						int(s1[5]),  #sub
						float(s2[0]),#SER
						int(s2[1]),  #sentences
						int(s2[2]),  #wrong sentences
						int(s3[1])    #missed sentences
				)

def edit_distance(ref,hyp,ignore=None,mode='present'):
	'''
	Compute edit-distance score.

	Args:
		<ref>,<hyp>: exkaldi Transcription objects.
		<ignore>: Ignoring specific symbols.
		<mode>: When both are Transcription objects,if mode is 'present',skip the missed utterances.

	Return:
		a namedtuple object including score information.	
	'''
	declare.is_potential_transcription("ref",ref)
	declare.is_potential_transcription("hyp",hyp)
	declare.is_instances("mode",mode,['all','present'])

	if ignore is not None:
		declare.is_valid_string("ignore",ignore)

	if isinstance(ref,str):
		ref = load_transcription(ref)

	if isinstance(hyp,str):
		hyp = load_transcription(hyp)

	allED = 0
	words = 0
	SER = 0
	sentences = 0
	wrongSentences = 0
	missedSentences = 0

	ref = ref.sort()
	hyp = hyp.sort()

	for utt,hypTrans in hyp.items():
		try:
			refTrans = ref[utt]
		except KeyError as e:
			if mode == "all":
				raise Exception("Missing transcription in reference,set <mode> as 'all' to skip it.")
			else:
				missedSentences += 1
		else:
			sentences += 1
			refTrans = refTrans.split()
			hypTrans = hypTrans.split()
			ed,wds = pure_edit_distance(refTrans,hypTrans,ignore=ignore)
			allED += ed
			words += wds
			if ed > 0:
				wrongSentences += 1

	if sentences == 0:
		raise Exception("Missing all transcription in reference. We don't think it's a reasonable result. Check the file please.")

	return namedtuple("Score",["editDistance","words","SER","sentences","wrongSentences","missedSentences"])(
			allED,words,wrongSentences/sentences,sentences,wrongSentences,missedSentences
			)
