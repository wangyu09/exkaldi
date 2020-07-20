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

"""Exkaldi LM training & HCLG generating associates."""

import os
import subprocess
import tempfile
import kenlm
import math
import sys

from exkaldi.utils.utils import check_config, make_dependent_dirs, type_name, run_shell_command
from exkaldi.core.archieve import BytesArchieve, Metric
from exkaldi.version import version as ExkaldiInfo
from exkaldi.version import WrongPath, KaldiProcessError, ShellProcessError, KenlmProcessError, UnsupportedType, WrongOperation, WrongDataFormat

def train_ngrams_srilm(lexicons, order, textFile, outFile, config=None):
	'''
	Train n-grams language model with Srilm tookit.

	Args:
		<lexicons>: words.txt file path or Exkaldi LexiconBank object.
		<order>: the maxinum order of n-grams.
		<textFile>: text corpus file.
		<outFile>: ARPA out file name.
		<config>: configures, a Python dict object.

	You can use .check_config("train_ngrams_srilm") function to get configure information that you can set.
	Also you can run shell command "lmplz" to look their meaning.
	'''
	assert isinstance(order, int) and order > 0 and order < 10, "Expected <n> is a positive int value and it must be smaller than 10."
	assert isinstance(textFile,str), "Expected <textFile> is name-like string."
	assert isinstance(outFile,str), "Expected <outFile> is name-like string."
	assert type_name(lexicons) == "LexiconBank", f"Expected <lexicons> is exkaldi LexiconBank object but got {type_name(lexicons)}."

	ExkaldiInfo.prepare_srilm()

	if not os.path.isfile(textFile):
		raise WrongPath(f"No such file:{textFile}")
	else:
		## Should check the numbers of lines
		cmd = f"shuf {textFile} -n 100"
		out, err, cod = run_shell_command(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		if (isinstance(cod,int) and cod != 0):
			print(err.decode())
			raise ShellProcessError("Failed to sample from text file.")
		elif out == b'':
			raise WrongDataFormat("Void text file.")
		else:
			out = out.decode().strip().split("\n")
			spaceCount = 0
			for line in out:
				spaceCount += line.count(" ")
			if spaceCount < len(out)//2:
				raise WrongDataFormat("The text file doesn't seem to be separated by spaces or extremely short.")

	wordlist = tempfile.NamedTemporaryFile("w+", encoding='utf-8', suffix=".txt")
	unkSymbol = lexicons("oov")
	try:
		lexiconp = lexicons("lexiconp")
		words = [ x[0] for x in lexiconp.keys() ]
		wordlist.write( "\n".join(words) )
		wordlist.seek(0)

		#cmd2 = f"ngram-count -text {textFile} -order {order}"
		extraConfig = " "
		specifyDiscount = False
		if config is not None:
			if check_config(name='train_ngrams_srilm', config=config):
				for key,value in config.items():
					if isinstance(value,bool):
						if value is True:
							extraConfig += f"{key} "
						if key.endswith("discount"):
							specifyDiscount = True
					else:
						extraConfig += f" {key} {value}"

		cmd = f"ngram-count -text {textFile} -order {order} -limit-vocab -vocab {wordlist.name} -unk -map-unk {unkSymbol} "
		if specifyDiscount is False:
			cmd += "-kndiscount "
		cmd += "-interpolate "

		if not outFile.rstrip().endswith(".arpa"):
			outFile += ".arpa"
		make_dependent_dirs(outFile, pathIsFile=True)

		cmd += f" -lm {outFile}"
		
		out, err, cod = run_shell_command(cmd, stderr=subprocess.PIPE)

		if (isinstance(cod, int) and cod != 0) or (not os.path.isfile(outFile)) or os.path.getsize(outFile) == 0:
			print(err.decode())
			if os.path.isfile(outFile):
				os.remove(outFile)
			raise KaldiProcessError(f'Failed to generate ngrams language model.')
		else:
			return os.path.abspath(outFile)
	
	finally:
		wordlist.close()

def train_ngrams_kenlm(lexicons, order, textFile, outFile, config=None):
	'''
	Train n-grams language model with KenLm tookit.

	Args:
		<lexicons>: words.txt file path or Exkaldi LexiconBank object.
		<order>: the maxinum order of n-grams.
		<textFile>: text corpus file.
		<outFile>: ARPA out file name.
		<config>: configures, a Python dict object.

	You can use .check_config("train_ngrams_kenlm") function to get configure information that you can set.
	Also you can run shell command "lmplz" to look their meaning.
	'''
	assert isinstance(order, int) and 0 < order <= 6, "We support maximum 6-grams LM in current version."

	if not os.path.isfile(textFile):
		raise WrongPath("No such file:{}".format(textFile))
	else:
		## Should check the numbers of lines
		cmd = f"shuf {textFile} -n 100"
		out, err, cod = run_shell_command(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		if (isinstance(cod,int) and cod != 0):
			print(err.decode())
			raise ShellProcessError("Failed to sample from text file.")
		elif out == b'':
			raise WrongDataFormat("Void text file.")
		else:
			out = out.decode().strip().split("\n")
			spaceCount = 0
			for line in out:
				spaceCount += line.count(" ")
			if spaceCount < len(out)//2:
				raise WrongDataFormat("The text file doesn't seem to be separated by spaces or extremely short.")

	extraConfig = " "
	if config != None:
		assert isinstance(config, dict), f"<config> should be dict object but got: {type_name(config)}."
		if check_config(name='train_ngrams_kenlm', config=config):
			if "--temp_prefix" in config.keys() and "-T" in config.keys():
				raise WrongOperation(f'"--temp_prefix" and "-T" is the same configure so only one of them is expected.')
			if "--memory" in config.keys() and "-S" in config.keys():
				raise WrongOperation(f'"--memory" and "-S" is the same configure so only one of them is expected.')
			for key,value in config.items():
				if isinstance(value,bool):
					if value is True:
						extraConfig += f"{key} "
				else:
					extraConfig += f"{key} {value} "

	assert isinstance(outFile, str), f"<outFile> should be a string."
	if not outFile.rstrip().endswith(".arpa"):
		outFile += ".arpa"
	make_dependent_dirs(outFile, pathIsFile=True)
	
	words = tempfile.NamedTemporaryFile("w+", suffix=".txt", encoding="utf-8")
	try:
		if type_name(lexicons) == "LexiconBank":
			ws = lexicons("words")
			words_count = math.ceil(len(ws)/10) * 10
			ws = "\n".join(ws.keys())
		elif isinstance(lexicons, str):
			if not os.path.isfile(lexicons):
				raise WrongPath(f"No such file:{lexicons}.")
			with open(lexicons, "r", encoding="utf-8") as fr:
				lines = fr.readlines()
			ws = []
			for line in lines:
				line = line.strip().split(maxsplit=1)
				if len(line) < 1:
					continue
				else:
					ws.append(line[0])
			words_count = math.ceil(len(ws)/10) * 10
			ws = "\n".join(ws)
		else:
			raise UnsupportedType("<lexicons> should be LexiconBank object or file path.")

		words.write(ws)
		words.seek(0)

		KenLMTool = os.path.join(sys.prefix, "exkaldisrc", "tools", "lmplz")

		cmd = f"{KenLMTool}{extraConfig}-o {order} --vocab_estimate {words_count} --text {textFile} --arpa {outFile} --limit_vocab_file {words.name}"
		out, err, cod = run_shell_command(cmd, stderr=subprocess.PIPE)

		if (isinstance(cod, int) and cod != 0) or (not os.path.isfile(outFile)) or (os.path.getsize(outFile)==0):
			print(err.decode())
			raise KenlmProcessError("Failed to generate arpa file.")
		else:
			return os.path.abspath(outFile)

	finally:
		words.close()

def arpa_to_binary(arpaFile, outFile):
	'''
	Transform ARPA language model file to KenLM binary format file.

	Args:
		<arpaFile>: ARPA file path.
		<outFile>: output binary file path.
	Return:
		Then absolute path of output file.
	'''
	assert isinstance(arpaFile, str), f"<arpaFile> should be a string."
	if not os.path.isfile(arpaFile):
		raise WrongPath(f"No such file:{arpaFile}.")

	assert isinstance(outFile, str), f"<outFile> should be a string."
	make_dependent_dirs(outFile)

	cmd = os.path.join(sys.prefix,"exkaldisrc","tools","build_binary")
	cmd += f" -s {arpaFile} {outFile}"
	out, err, cod = run_shell_command(cmd, stderr=subprocess.PIPE)

	if (cod != 0) or (not os.path.isfile(outFile)) or (os.path.getsize(outFile)==0):
		print(err.decode())
		raise KenlmProcessError("Failed to tansform ARPA to binary format.")
	
	else:
		return os.path.abspath(outFile)

class KenNGrams(BytesArchieve):
	'''
	This is a wrapper of kenlm.Model, and we only support n-grams model with binary format.
	'''
	def __init__(self, filePath, name="ngram"):
		assert isinstance(filePath, str), f"<filePath> should be string but got {type_name(filePath)}."
		if not os.path.isfile(filePath):
			raise WrongPath(f"No such file:{filePath}.")
		else:
			with open(filePath, "rb") as fr:
				t = fr.read(50).decode().strip()
			if t != "mmap lm http://kheafield.com/code format version 5":
				raise UnsupportedType("This is not a KenLM binary model formation.")
		
		super(KenNGrams, self).__init__(data=b"kenlm", name=name)
		self.__model = kenlm.Model(filePath)

	@property
	def path(self):
		'''
		Gen the model file path.
		'''
		return self.__model.path

	@property
	def order(self):
		'''
		Gen the maxinum value of order.
		'''
		return self.__model.order

	def score(self, sentence, bos=True, eos=True):
		'''
		Score a sentence.

		Args:
			<sentence>: a string with out boundary symbols or exkaldi Transcription object.
			<bos>: If True, add <s> to the head.
			<eos>: If True, add </s> to the tail.
		Return:
			If <sentence> is string, return a float log-value.
			Else, return an exkaldi Metric object.
		'''
		def score_one(one, bos, eos):
			if one.count(" ") < 1:
				print(f"Warning: sentence doesn't seem to be separated by spaces or extremely short: {one}.")
			return self.__model.score(one, bos, eos)
		
		if isinstance(sentence, str):
			return score_one(sentence, bos, eos)
		elif type_name(sentence) == "Transcription":
			scores = {}
			for uttID, txt in sentence.items():
				assert isinstance(txt,str), f"Transcription should be string od words but got:{type_name(txt)} at utt-ID {uttID}."
				scores[uttID] = score_one(txt, bos, eos )
			return Metric(scores, name=f"LMscore({sentence.name})")
		else:
			raise UnsupportedType(f"<sentence> should be string or exkaldi Transcription object ut got: {type_name(sentence)}.")

	def full_scores(self, sentence, bos=True, eos=True):
		'''
		Generate full scores (prob, ngram length, oov).

		Args:
			<sentence>: a string with out boundary symbols.
			<bos>: If True, add <s> to the head.
			<eos>: If True, add </s> to the tail.
		Return:
			a iterator of (prob, ngram length, oov).
		'''
		return self.__model.full_scores(sentence, bos, eos)

	def perplexity(self, sentence):
		'''
		Compute perplexity of a sentence.

		Args:
			<sentence>: a sentence which has words-in blank and has not boundary or exkaldi Transcription object.

		Return:
			If <sentence> is string, return a perplexity value.
			Else return an exkaldi Metric object.
		'''
		def perplexity_one(one):
			if one.count(" ") < 1:
				print(f"Warning: sentence doesn't seem to be separated by spaces or extremely short: {one}.")
			return self.__model.perplexity(one)
		
		if isinstance(sentence, str):
			return perplexity_one(sentence)
		elif type_name(sentence) == "Transcription":
			scores = {}
			for uttID, txt in sentence.items():
				assert isinstance(txt,str), f"Transcription should be string od words but got:{type_name(txt)} at utt-ID {uttID}."
				scores[uttID] = perplexity_one(txt)
			return Metric(scores, name=f"LMperplexity({sentence.name})")
		else:
			raise UnsupportedType(f"<sentence> should be string or exkaldi Transcription object ut got: {type_name(sentence)}.")
