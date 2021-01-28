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

"""Language Model"""

import os
try:
    import kenlm
except Exception:
    print("WARNING: kenlm is not installed, decoder's method can't use")
import math
import sys
from collections import namedtuple

from exkaldi.utils.utils import check_config, make_dependent_dirs, type_name, run_shell_command
from exkaldi.utils.utils import FileHandleManager
from exkaldi.utils import declare
from exkaldi.core.archive import BytesArchive, Metric
from exkaldi.version import info as ExKaldiInfo
from exkaldi.error import *

def train_ngrams_srilm(lexicons, order, text, outFile, config=None):
	'''
	Train N-Grams language model with SriLM tookit.
	If you don't specified the discount by the <config> option, We defaultly use "kndiscount".

	Args:
		<lexicons>: an exkaldi LexiconBank object.
		<order>: the maximum order of N-Grams.
		<text>: a text corpus file or an exkaldi transcription object.
		<outFile>: output file name of arpa LM.
		<config>: extra configurations, a Python dict object.

	You can use .check_config("train_ngrams_srilm") function to get a reference of extra configurations.
	Also you can run shell command "ngram-count" to look their usage.
	'''
	declare.is_lexicon_bank("lexicons", lexicons)
	declare.is_positive_int("order", order)
	declare.is_potential_transcription("text", text)
	declare.is_valid_file_name("outFile", outFile)
	# verify the max order
	declare.less_equal("order", order, "max order", 9)
  # prepare srilm tool
	ExKaldiInfo.prepare_srilm()

	with FileHandleManager() as fhm:
		# check whether this is a reasonable text corpus that should be splited by space.
		if isinstance(text, str):
			cmd = f"shuf {text} -n 100"
			out,err,cod = run_shell_command(cmd, stdout="PIPE", stderr="PIPE")
			if (isinstance(cod,int) and cod != 0):
				raise ShellProcessError(f"Failed to sample from text file:{text}.",err.decode())
			elif out == b'':
				raise WrongDataFormat(f"Void text file:{text}.")
			else:
				out = out.decode().strip().split("\n")
				spaceCount = 0
				for line in out:
					spaceCount += line.count(" ")
				if spaceCount < len(out)//2:
					raise WrongDataFormat("The text file doesn't seem to be separated by spaces or sentences are extremely short.")
		
		else:
			sampleText = text.subset(nRandom=100)
			spaceCount = 0
			for key,value in sampleText.items():
				assert isinstance(value,str), f"Transcription must be string but got: {type_name(value)}."
				spaceCount += value.count(" ")
			if spaceCount < len(sampleText)//2:
				raise WrongDataFormat("The text file doesn't seem to be separated by spaces or sentences are extremely short.")
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

		cmd = f'ngram-count -text {text} -order {order} -limit-vocab -vocab {wordlistTemp.name} -unk -map-unk "{unkSymbol}" '
		if specifyDiscount is False:
			cmd += "-kndiscount "
		cmd += "-interpolate "

		if not outFile.rstrip().endswith(".arpa"):
			outFile += ".arpa"
		make_dependent_dirs(outFile, pathIsFile=True)
		cmd += f" -lm {outFile}"
		
		out,err,cod = run_shell_command(cmd, stderr="PIPE")

		if (isinstance(cod,int) and cod != 0) or (not os.path.isfile(outFile)) or os.path.getsize(outFile) == 0:
			if os.path.isfile(outFile):
				os.remove(outFile)
			raise KaldiProcessError(f'Failed to generate N-Grams language model.',err.decode())

		return outFile

def train_ngrams_kenlm(lexicons, order, text, outFile, config=None):
	'''
	Train N-Grams language model with SriLM tookit.

	Args:
		<lexicons>: an exkaldi LexiconBank object.
		<order>: the maximum order of N-Grams.
		<text>: a text corpus file or an exkaldi transcription object.
		<outFile>: output file name of arpa LM.
		<config>: extra configurations, a Python dict object.

	You can use .check_config("train_ngrams_kenlm") function to get a reference of extra configurations.
	Also you can run shell command "lmplz" to look their usage.
	'''
	declare.is_lexicon_bank("lexicons", lexicons)
	declare.is_positive_int("order", order)
	declare.is_potential_transcription("text", text)
	declare.is_valid_file_name("outFile", outFile)

	declare.less_equal("order", order, "max order", 9)

	with FileHandleManager() as fhm:
		# check whether this is a reasonable text corpus that should be splited by space.
		if isinstance(text, str):
			cmd = f"shuf {text} -n 100"
			out,err,cod = run_shell_command(cmd, stdout="PIPE", stderr="PIPE")
			if (isinstance(cod,int) and cod != 0):
				raise ShellProcessError(f"Failed to sample from text file:{text}.",err.decode())
			elif out == b'':
				raise WrongDataFormat(f"Void text file:{text}.")
			else:
				out = out.decode().strip().split("\n")
				spaceCount = 0
				for line in out:
					spaceCount += line.count(" ")
				if spaceCount < len(out)//2:
					raise WrongDataFormat("The text file doesn't seem to be separated by spaces or sentences are extremely short.")
		
		else:
			sampleText = text.subset(nRandom=100)
			spaceCount = 0
			for key,value in sampleText.items():
				assert isinstance(value,str), f"Transcription must be string but got: {type_name(value)}."
				spaceCount += value.count(" ")
			if spaceCount < len(sampleText)//2:
				raise WrongDataFormat("The text file doesn't seem to be separated by spaces or sentences are extremely short.")
			textTemp = fhm.create("a+", suffix=".txt", encoding="utf-8")
			text.save(textTemp, discardUttID=True)
			text = textTemp.name

		extraConfig = " "
		if config is not None:
			if check_config(name='train_ngrams_kenlm', config=config):
				if "--temp_prefix" in config.keys() and "-T" in config.keys():
					raise WrongOperation(f'"--temp_prefix" and "-T" is the same configuration so only one of them is expected.')
				if "--memory" in config.keys() and "-S" in config.keys():
					raise WrongOperation(f'"--memory" and "-S" is the same configuration so only one of them is expected.')
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

		KenLMTool = os.path.join(sys.prefix,"exkaldisrc","tools","lmplz")

		cmd = f"{KenLMTool}{extraConfig}-o {order} --vocab_estimate {words_count} --text {text} --arpa {outFile} --limit_vocab_file {wordlistTemp.name}"
		out,err,cod = run_shell_command(cmd,stderr="PIPE")

		if (isinstance(cod, int) and cod != 0) or (not os.path.isfile(outFile)) or (os.path.getsize(outFile)==0):
			if os.path.isfile(outFile):
				os.remove(outFile)
			raise KenlmProcessError("Failed to generate arpa file.",err.decode())

		return outFile

def arpa_to_binary(arpaFile, outFile):
	'''
	Transform ARPA language model to KenLM binary format.

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
	out, err, cod = run_shell_command(cmd, stderr="PIPE")

	if (cod != 0) or (not os.path.isfile(outFile)) or (os.path.getsize(outFile)==0):
		if os.path.isfile(outFile):
			os.remove(outFile)
		raise KenlmProcessError("Failed to tansform ARPA to binary format.",err.decode())
	
	else:
		return outFile

class KenNGrams(BytesArchive):
	'''
	This is a wrapper of kenlm.Model, and we only support n-grams model with binary format.

	If you want to load ARPA format file directly, use the exkaldi.lm.load_ngrams() function.
	'''
	def __init__(self, filePath, name="ngram"):
		declare.is_file("filePath", filePath)

		with open(filePath, "rb") as fr:
			t = fr.read(50).decode().strip()
		if t != "mmap lm http://kheafield.com/code format version 5":
			raise UnsupportedType("This may be not a KenLM binary model format.")
		
		super(KenNGrams, self).__init__(data=b"placeholder", name=name)
		self.__model = kenlm.Model(filePath)
		self._path = None

	@property
	def info(self):
		'''
		Get the info of this N-grams model.
		'''
		order = self.__model.order
		if self._path is None:
			path = self.__model.path
		else:
			path = self._path
		
		return namedtuple("NgramInfo",["order","path"])(order,str(path))

	def score_sentence(self, sentence, bos=True, eos=True):
		'''
		Score a sentence.

		Args:
			<sentence>: a string with out boundary symbols.
			<bos>: If True, add <s> to the head.
			<eos>: If True, add </s> to the tail.

		Return:
			a float value.
		'''
		declare.is_valid_string("sentence", sentence)
		declare.is_bool("bos", bos)
		declare.is_bool("eos", eos)

		return self.__model.score(sentence, bos, eos)

	def score(self, transcription, bos=True, eos=True):
		'''
		Score a transcription.

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
		
		scores = Metric(name=f"LMscore({transcription.name})")
		for uttID, txt in transcription.items():
			scores[uttID] = self.score_sentence(txt, bos, eos)

		return scores

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
		declare.is_valid_string("sentence", sentence)
		declare.is_bool("bos", bos)
		declare.is_bool("eos", eos)

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

		return self.__model.perplexity(sentence)

	def perplexity(self, transcription):
		'''
		Compute perplexity of a transcription.

		Args:
			<transcription>: file path or exkaldi Transcription object.

		Return:
			an named tuple: PPL(probs,sentences,words,ppl,ppl1).
		'''
		declare.is_potential_transcription("transcription", transcription)

		if isinstance(transcription, str):
			transcription = load_transcription(transcription)
		
		prob = self.score(transcription)
		sentences = len(prob)
		words = transcription.sentence_length().sum()

		sumProb = prob.sum()
		ppl = 10**(-sumProb/(sentences+words))
		ppl1 = 10**(-sumProb/(words))

		return namedtuple("PPL",["prob","sentences","words","ppl","ppl1"])(round(sumProb,2),sentences,words,round(ppl,2),round(ppl1,2))

def load_ngrams(target, name="gram"):
	'''
	Load a N-Grams from arpa or binary language model file.

	Args:
		<target>: file path with suffix .arpa or .binary.
	
	Return:
		a KenNGrams object.
	'''
	declare.is_file("target", target)
	target = target.strip()
	
	with FileHandleManager() as fhm:

		if target.endswith(".arpa"):
			modelTemp = fhm.create("wb+", suffix=".binary")
			arpa_to_binary(target, modelTemp.name)
			modelTemp.seek(0)
			model = KenNGrams(modelTemp.name, name=name)
			model._path = target
			
		elif target.endswith(".binary"):
			model = KenNGrams(target, name=name)

		else:
			raise UnsupportedType(f"Unknown suffix. Language model file should has a suffix .arpa or .binary but got: {target}.")
			
		return model

