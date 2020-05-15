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

from exkaldi.utils.utils import check_config, make_dependent_dirs,type_name, run_shell_command
from exkaldi.core.achivements import BytesAchievement
from exkaldi.version import version as ExkaldiInfo
from exkaldi.version import WrongPath, KaldiProcessError, UnsupportedType, WrongOperation, WrongDataFormat

def train_ngrams_srilm(lexicons, order, textFile, outFile, discount="kndiscount", config=None):
	'''
	Usage:  obj = ngrams(3,"text.txt","lm.3g.gz","word.txt")

	Generate ARPA n-grams language model. Return abspath of generated LM.
	<order> is the orders of n-grams model. <textFile> is the text file. <outFile> is expected .gz file name of generated LM.
	Notion that we will use srilm language model toolkit.
	Some usual options can be assigned directly. If you want use more, set <config> = your-configure, but if you do this, these usual configures we provided will be ignored.
	You can use .check_config("ngrams") function to get configure information that you can set.
	Also you can run shell command "ngram-count -help" to look their meaning.    
	'''
	assert isinstance(order, int) and order > 0 and order < 10, "Expected <n> is a positive int value and it must be smaller than 10."
	assert isinstance(textFile,str), "Expected <textFile> is name-like string."
	assert isinstance(outFile,str), "Expected <outFile> is name-like string."
	assert type_name(lexicons) == "LexiconBank", f"Expected <lexicons> is exkaldi LexiconBank object but got {type_name(lexicons)}."

	if not os.path.isfile(textFile):
		raise WrongPath(f"No such file:{textFile}")
	else:
		## Should check the numbers of lines
		cmd1 = f'shuf {textFile} -n 100 | sed "s/ /\\n<space>\\n/g" | sort | uniq -c | sort -n | tail -n 1'
		if int(subprocess.check_output(cmd1,shell=True).decode().strip().split()[0]) < 50:
			raise WrongDataFormat("Text file sames that it were not splited by spaces.")
	
	wordlist = tempfile.NamedTemporaryFile("w+", encoding='utf-8', suffix=".txt")
	unkSymbol = lexicons("oov")

	try:
		lexiconp = lexicons("lexiconp")
		words = [ x[0] for x in lexiconp.keys() ]
		wordlist.write( "\n".join(words) )
		wordlist.seek(0)

		cmd2 = f"ngram-count -text {textFile} -order {order}"

		if config is None:  
			config = {}
			config["-limit-vocab -vocab"] = wordlist.name
			config["-unk -map-unk"] = unkSymbol
			assert discount in ["wbdiscount","kndiscount"], "Expected <discount> is wbdiscount or kndiscount."
			config[f"-{discount}"] = True
			config["-interpolate"] = True
		else:
			raise WrongOperation("<config> of train_ngrams function is unavaliable now.")

		if check_config(name='ngrams', config=config):
			for key in config.keys():
				if config[key] is True:
					cmd2 += " {}".format(key)
				elif not (config[key] is False):
					if key == "-unk -map-unk":
						cmd2 += f' {key} "{config[key]}"'
					else:
						cmd2 += f' {key} {config[key]}'

		if not outFile.endswith(".gz"):
			outFile += ".gz"
		make_dependent_dirs(outFile, pathIsFile=True)

		cmd2 += f" -lm {outFile}"
		
		out, err, cod = run_shell_command(cmd2, stderr=subprocess.PIPE)

		if (not os.path.isfile(outFile)) or os.path.getsize(outFile) == 0:
			print(err.decode())
			if os.path.isfile(outFile):
				os.remove(outFile)
			raise KaldiProcessError(f'Failed to generate ngrams language model.')
		else:
			return os.path.abspath(outFile)
	
	finally:
		wordlist.close()

def train_ngrams_kenlm(lexicons, order, textFile, outFile):
	'''
	Train n-grams language model with KenLm tookit.

	Args:
		<lexicons>: words.txt file path or Exkaldi LexiconBank object.
		<order>: the maxinum order of n-grams.
		<textFile>: 

	'''
	assert isinstance(order, int) and 0 < order <= 6, "We support maximum 6-grams LM in current version."

	if not os.path.isfile(textFile):
		raise WrongPath("No such file:{}".format(textFile))
	else:
		cmd1 = f'shuf {textFile} -n 100 | sed "s/ /\\n<space>\\n/g" | sort | uniq -c | sort -n | tail -n 1'
		if int(subprocess.check_output(cmd1,shell=True).decode().strip().split()[0]) < 50: ## bug should check the lines
			raise WrongDataFormat("Text file sames that it were not splited by spaces.")
	
	assert isinstance(outFile, str), f"<outFile> should be a string."
	if outFile.rstrip().endswith(".arpa"):
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

		cmd = f"lmplz -o {order} --vocab_estimate {words_count} --verbose_header --text {textFile} --arpa {outFile} --prune --limit_vocab_file {words.name}"
		out, err, cod = run_shell_command(cmd, stderr=subprocess.PIPE)

		if (cod != 0) or (not os.path.isfile(outFile)) or (os.path.getsize(outFile)==0):
			print(err.decode())
			raise Exception("Failed to generate arpa file.")
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

	cmd= f"build_binary -s {arpaFile} {outFile}"
	out, err, cod = run_shell_command(cmd, stderr=subprocess.PIPE)

	if (cod != 0) or (not os.path.isfile(outFile)) or (os.path.getsize(outFile)==0):
		print(err.decode())
		raise Exception("Failed to tansform ARPA to binary format.")
	
	else:
		return os.path.abspath(outFile)

class KenNGrams(BytesAchievement):
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
			<sentence>: a string with out boundary symbols.
			<bos>: If True, add <s> to the head.
			<eos>: If True, add </s> to the tail.
		Return:
			log value.
		'''
		return self.__model.score(sentence, bos, eos)

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
			<sentence>: a sentence which has words-in blank and has not boundary.

		Return:
			perplexity value.
		'''
		return self.__model.perplexity(sentence)