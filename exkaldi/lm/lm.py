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
from exkaldi.utils.utils import FileHandleManager
from exkaldi.utils import declare
from exkaldi.core.archieve import BytesArchieve, Metric
from exkaldi.version import info as ExkaldiInfo
from exkaldi.version import WrongPath, KaldiProcessError, ShellProcessError, KenlmProcessError, UnsupportedType, WrongOperation, WrongDataFormat

def train_ngrams_srilm(lexicons, order, text, outFile, config=None):
	'''
	Train n-grams language model with Srilm tookit.

	Args:
		<lexicons>: words.txt file path or Exkaldi LexiconBank object.
		<order>: the maxinum order of n-grams.
		<text>: text corpus file or exkaldi transcription object.
		<outFile>: ARPA out file name.
		<config>: configures, a Python dict object.

	You can use .check_config("train_ngrams_srilm") function to get configure information that you can set.
	Also you can run shell command "ngram-count" to look their meaning.
	'''
	declare.is_positive_int("order", order)
	declare.smaller("order", order, "max order", 9)
	declare.is_valid_file_name("outFile", outFile)
	declare.is_lexicon_bank("lexicons", lexicons)
	declare.is_potential_transcription("text", text)

	ExkaldiInfo.prepare_srilm()

	with FileHandleManager() as fhm:

		if isinstance(text,str):
			## Should check the numbers of lines
			cmd = f"shuf {text} -n 100"
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
		else:
			sampleText = text.subset(nRandom=100)
			spaceCount = 0
			for key,value in sampleText.items():
				assert isinstance(value,str), f"Transcription object must be string but got: {type_name(value)}."
				spaceCount += value.count(" ")
			if spaceCount < len(sampleText)//2:
				raise WrongDataFormat("The transcription doesn't seem to be separated by spaces or extremely short.")
					
			textTemp = fhm.create("a+", suffix=".txt", encoding="utf-8")
			text.save(textTemp, discardUttID=True)
			text = textTemp.name

		unkSymbol = lexicons("oov")

		wordlistTemp = fhm.create("w+", encoding='utf-8', suffix=".txt")
		words = lexicons("words")
		words = "\n".join(words.keys())

		wordlistTemp.write(words)
		wordlistTemp.seek(0)

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

		cmd = f"ngram-count -text {text} -order {order} -limit-vocab -vocab {wordlistTemp.name} -unk -map-unk {unkSymbol} "
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

		return outFile

def train_ngrams_kenlm(lexicons, order, text, outFile, config=None):
	'''
	Train n-grams language model with KenLm tookit.

	Args:
		<lexicons>: words.txt file path or Exkaldi LexiconBank object.
		<order>: the maxinum order of n-grams.
		<text>: text corpus file or exkaldi transcription object.
		<outFile>: ARPA out file name.
		<config>: configures, a Python dict object.

	You can use .check_config("train_ngrams_kenlm") function to get configure information that you can set.
	Also you can run shell command "lmplz" to look their meaning.
	'''
	declare.is_positive_int("order", order)
	declare.smaller("order", order, "max order", 6)
	declare.is_valid_file_name("outFile", outFile)
	declare.is_lexicon_bank("lexicons", lexicons)
	declare.is_potential_transcription("text", text)

	with FileHandleManager() as fhm:

		if isinstance(text,str):
			## Should check the numbers of lines
			cmd = f"shuf {text} -n 100"
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
		else:
			sampleText = text.subset(nRandom=100)
			spaceCount = 0
			for key,value in sampleText.items():
				assert isinstance(value,str), f"Transcription object must be string but got: {type_name(value)}."
				spaceCount += value.count(" ")
			if spaceCount < len(sampleText)//2:
				raise WrongDataFormat("The transcription doesn't seem to be separated by spaces or extremely short.")
					
			textTemp = fhm.create("a+", suffix=".txt", encoding="utf-8")
			text.save(textTemp, discardUttID=True)
			text = textTemp.name

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

		if not outFile.rstrip().endswith(".arpa"):
			outFile += ".arpa"
		make_dependent_dirs(outFile, pathIsFile=True)

		wordlistTemp = fhm.create("w+", encoding='utf-8', suffix=".txt")
		words = lexicons("words")
		words_count = math.ceil(len(words)/10) * 10
		
		words = "\n".join(words.keys())

		wordlistTemp.write(words)
		wordlistTemp.seek(0)

		KenLMTool = os.path.join(sys.prefix, "exkaldisrc", "tools", "lmplz")

		cmd = f"{KenLMTool}{extraConfig}-o {order} --vocab_estimate {words_count} --text {text} --arpa {outFile} --limit_vocab_file {wordlistTemp.name}"
		out, err, cod = run_shell_command(cmd, stderr=subprocess.PIPE)

		if (isinstance(cod, int) and cod != 0) or (not os.path.isfile(outFile)) or (os.path.getsize(outFile)==0):
			print(err.decode())
			raise KenlmProcessError("Failed to generate arpa file.")

		return outFile

def arpa_to_binary(arpaFile, outFile):
	'''
	Transform ARPA language model file to KenLM binary format file.

	Args:
		<arpaFile>: ARPA file path.
		<outFile>: output binary file path.
	Return:
		output file name with suffix ".binary".
	'''
	declare.is_file("arpaFile", arpaFile)
	declare.is_valid_string("outFile", outFile)
	outFile = outFile.strip()
	if not outFile.endswith(".binary"):
		outFile += ".binary"

	declare.is_valid_file_name("outFile", outFile)
	make_dependent_dirs(outFile)

	cmd = os.path.join(sys.prefix,"exkaldisrc","tools","build_binary")
	cmd += f" -s {arpaFile} {outFile}"
	out, err, cod = run_shell_command(cmd, stderr=subprocess.PIPE)

	if (cod != 0) or (not os.path.isfile(outFile)) or (os.path.getsize(outFile)==0):
		print(err.decode())
		raise KenlmProcessError("Failed to tansform ARPA to binary format.")
	
	else:
		return outFile

class KenNGrams(BytesArchieve):
	'''
	This is a wrapper of kenlm.Model, and we only support n-grams model with binary format.
	'''
	def __init__(self, filePath, name="ngram"):
		declare.is_file("filePath", filePath)

		with open(filePath, "rb") as fr:
			t = fr.read(50).decode().strip()
		if t != "mmap lm http://kheafield.com/code format version 5":
			raise UnsupportedType("This is not a KenLM binary model formation.")
		
		super(KenNGrams, self).__init__(data=b"kenlm", name=name)
		self.__model = kenlm.Model(filePath)
		self._path = None

	@property
	def path(self):
		'''
		Gen the model file path.
		'''
		if self._path is None:
			return self.__model.path
		else:
			return self._path

	@property
	def order(self):
		'''
		Get the maxinum value of order.
		'''
		return self.__model.order

	def score_sentence(self, sentence, bos=True, eos=True):
		'''
		Score a sentence.

		Args:
			<sentence>: a string with out boundary symbols.
			<bos>: If True, add <s> to the head.
			<eos>: If True, add </s> to the tail.
		Return:
			a float log-value.
		'''
		declare.is_valid_string("sentence", sentence)
		declare.is_bool("bos", bos)
		declare.is_bool("eos", eos)

		if sentence.count(" ") < 1:
			print(f"Warning: sentence doesn't seem to be separated by spaces or extremely short: {one}.")		

		return self.__model.score(sentence, bos, eos)

	def score(self, transcription, bos=True, eos=True):
		'''
		Score transcription.

		Args:
			<transcription>: file path or exkaldi Transcription object.
			<bos>: If True, add <s> to the head.
			<eos>: If True, add </s> to the tail.
		Return:
			an exkaldi Metric object.
		'''
		declare.is_potential_transcription("transcription", transcription)

		if isinstance(transcription, str):
			transcription = load_transcription(transcription)
		
		scores = {}
		for uttID, txt in transcription.items():
			scores[uttID] = self.score_sentence(txt, bos, eos)

		return Metric(scores, name=f"LMscore({transcription.name})")

	def full_scores_sentence(self, sentence, bos=True, eos=True):
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

	def perplexity_sentence(self, sentence):
		'''
		Compute perplexity of a sentence.

		Args:
			<sentence>: a string with out boundary symbols.
		Return:
			a float log-value.
		'''
		declare.is_valid_string("sentence", sentence)

		if sentence.count(" ") < 1:
			print(f"Warning: sentence doesn't seem to be separated by spaces or extremely short: {one}.")		

		return self.__model.perplexity(sentence)

	def perplexity(self, transcription):
		'''
		Compute perplexity of transcription.

		Args:
			<transcription>: file path or exkaldi Transcription object.

		Return:
			an exkaldi Metric object.
		'''
		declare.is_potential_transcription("transcription", transcription)

		if isinstance(transcription, str):
			transcription = load_transcription(transcription)
		
		scores = {}
		for uttID, txt in transcription.items():
			scores[uttID] = self.perplexity_sentence(txt)

		return Metric(scores, name=f"LMperplexity({transcription.name})")

def load_ngrams(target, name="gram"):
	'''
	Load a ngrams from arpa or binary language model file.

	Args:
		<target>: file path with suffix .arpa or .binary.
	
	Return:
		KenNGrams object
	'''
	declare.is_file("target", target)
	
	with FileHandleManager() as fhm:

		if target.endswith(".arpa"):
			modelTemp = fhm.create("wb+", suffix=".binary")
			arpa_to_binary(target, modelTemp.name)
			modelTemp.seek(0)
			model = KenNGrams(modelTemp.name, name=name)

		elif not target.endswith(".binary"):
			raise UnsupportedType(f"Unknown suffix. Language model file should has a suffix .arpa or .binary but got: {target}.")
			model = KenNGrams(target, name=name)

		model._path = target
		return model

