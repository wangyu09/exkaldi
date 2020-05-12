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

"""Exkaldi Decoding associates """
import tempfile
import copy
import os
import subprocess

from exkaldi.version import version as ExkaldiInfo
from exkaldi.version import WrongPath
from exkaldi.utils.utils import WrongOperation, WrongDataFormat, KaldiProcessError, UnsupportedDataType
from exkaldi.utils.utils import run_shell_command, make_dependent_dirs, type_name, check_config
from exkaldi.core.achivements import BytesAchievement
from exkaldi.nn.nn import log_softmax

class Transcription(dict):
	'''
	This is a subclass of Python dict and used to hold decoding results. 
	'''
	def __init__(self, *args, name="1-best", am_cost=None, lm_cost=None, **kwargs):
		assert isinstance(name, str) and len(name) >0, "Name is not a string avaliable."
		self.__name = name
		self.__am_cost = am_cost
		self.__lm_cost = lm_cost
		super(Transcription, self).__init__(*args, **kwargs)
	
	@property
	def is_void(self):
		if len(self.keys()) == 0:
			return True
		else:
			return False

	@property
	def name(self):
		return self.__name

	def rename(self, name):
		'''
		Rename.

		Args:
			<name>: a string.
		'''
		assert isinstance(name, str) and len(name) >0, "Name is not a string avaliable."
		self.__name = name

	def sort(self):
		'''
		Sort by utt.

		Return:
			A new Transcription object. 
		'''		
		results = Transcription(name=self.name)
		items = sorted(self.items(), key=lambda x:x[0])
		results.update(items)
		return results
		
	def save(self, fileName=None):
		'''
		Save as text formation.

		Args:
			<fileName>: If None, return a string of text formation.

		Return:
			A new Transcription object. 
		'''			
		results = "\n".join( map(lambda x:x[0]+" "+x[1], self.items()) )
		if fileName is None:
			return results
		elif isinstance(fileName, tempfile._TemporaryFileWrapper):
			fileName.write(results)
		else:
			assert isinstance(fileName, str) and len(fileName) > 0, "file name is unavaliable."
			make_dependent_dirs(fileName, pathIsFile=True)
			with open(fileName, "w", encoding="utf-8") as fw:
				fw.write(results)

	def load(self, fileName):
		'''
		Load a transcription from file. If utt is existed, it will be overlaped.

		Args:
			<fileName>: the txt file path.
		'''
		assert isinstance(fileName, str) and len(fileName)>0, "file name is unavaliable."
		if not os.path.isfile(fileName):
			raise WrongPath(f"No such file:{fileName}.")

		with open(fileName, "r", encoding="utf-8") as fr:
			lines = fr.readlines()
		for index, line in enumerate(lines, start=1):
			le = line.strip().split(maxsplit=1)
			if len(le) < 2:
				print(f"Line Number:{index}")
				print(f"Line Content:{line}")
				raise WrongDataFormat("Missing entire uttID and utterance information.")
			else:
				self[le[0]] = le[1]
	
	def __add__(self, other):
		'''
		Integrate two transcription objects. If utt is existed in both two objects, the former will be retained.

		Args:
			<other>: another Transcription object.
		Return:
			A new transcription object.
		'''
		assert isinstance(other, Transcription), f"Cannot add {type_name(other)}."
		new = copy.deepcopy(other)
		new.update(self)

		newName = f"add({self.name},{other.name})"
		new.rename(newName)

		return new

	def am_cost(self, utt=None):
		'''
		Get the acoustic model cost.

		Args:
			<utt>: If None, return all the cost of all utterance.
		
		Return:
			None or log float value.
		'''
		if utt is None:
			return self.__am_cost
		else:
			if self.__am_cost is None:
				return None
			try:
				score = self.__am_cost[utt]
			except KeyError:
				raise WrongOperation(f"No such utterance:{utt}.")
			else:
				return score

	def lm_cost(self, utt=None):
		'''
		Get the language model cost.

		Args:
			<utt>: If None, return all the cost of all utterance.
		
		Return:
			None or log float value.
		'''
		if utt is None:
			return self.__lm_cost
		else:
			if self.__lm_cost is None:
				return None
			try:
				score = self.__lm_cost[utt]
			except KeyError:
				raise WrongOperation(f"No such utterance:{utt}.")
			else:
				return score

class Lattice(BytesAchievement):
	'''
	Usage:  obj = KaldiLattice() or obj = KaldiLattice(lattice,hmm,wordSymbol)

	KaldiLattice holds the lattice and its related file path: HMM file and WordSymbol file. 
	The <lattice> can be lattice binary data or file path. Both <hmm> and <wordSymbol> are expected to be file path.
	decode_lattice() function will return a KaldiLattice object. Aslo, you can define a empty KaldiLattice object and load its data later.
	'''
	def __init__(self, data=None, name="lat"):
		super().__init__(data, name)
			
	def save(self, fileName):
		'''
		Save lattice as .ali file. 
		
		Args:
			<fileName>: file name.
		''' 
		assert isinstance(fileName, str) and len(fileName) > 0, "file name is unavaliable."

		if self.is_void:
			raise WrongOperation('No any data to save.')

		if not fileName.rstrip().endswith(".ali"):
			fileName += ".ali"
		
		make_dependent_dirs(fileName)

		with open(fileName, "wb") as fw:
			fw.write(self.data)

	def get_1best(self, lexicons, hmm, lmwt=1, acwt=1.0, phoneLevel=False):
		'''
		Get 1 best result with text formation.

		Args:
			<lexicons>: a LexiconBank object.
			<hmm>: HMM object.
			<lmwt>: language model weight.
			<acwt>: acoustic model weight.
			<phoneLevel>: If Ture, return phone results.
		Return:
			An exkaldi Transcription object.
		'''
		ExkaldiInfo.vertify_kaldi_existed()

		if self.is_void:
			raise WrongOperation('No any data in lattice.')

		assert isinstance(lmwt, int) and lmwt >=0, "Expected <lmwt> is a non-negative int number."

		model = tempfile.NamedTemporaryFile("wb+", suffix=".mdl")
		wordSymbol = tempfile.NamedTemporaryFile("w+", suffix=".txt", encoding="utf-8")
		try:
			if phoneLevel:
				cmd0 = ExkaldiInfo.KALDI_ROOT + f'/src/latbin/lattice-align-phones --replace-output-symbols=true {model.name} ark:- ark:- | '
				lexicons.dump_dict("words", wordSymbol, False)
			else:
				cmd0 = ""
				lexicons.dump_dict("phones", wordSymbol, False)

			cmd1 = ExkaldiInfo.KALDI_ROOT + f'/src/latbin/lattice-best-path --lm-scale={lmwt} --acoustic-scale={acwt} --word-symbol-table={wordSymbol.name} --verbose=2 ark:- ark,t:- |'
			cmd2 = ExkaldiInfo.KALDI_ROOT + f'/egs/wsj/s5/utils/int2sym.pl -f 2- {wordSymbol.name} '
			cmd = cmd0 + cmd1 + cmd2

			out, err, _ = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=self.data)
			if out == b'':
				print(err.decode())
				raise KaldiProcessError('Failed to get 1-best from lattice.')
			else:
				out = out.decode().split("\n")
				results = Transcription()
				for re in out:
					re = re.strip().split(maxsplit=1)
					results[re[0]] = re[1]

				return results
		finally:
			model.close()
			wordSymbol.close()
	
	def scale(self, acwt=1, invAcwt=1, ac2lm=0, lmwt=1, lm2ac=0):
		'''
		Scale lattice.

		Args:
			<acwt>: acoustic scale.
			<invAcwt>: inverse acoustic scale.
			<ac2lm>: acoustic to lm scale.
			<lmwt>: language lm scale.
			<lm2ac>: lm scale to acoustic.
		Return:
			An new Lattice object.
		'''
		ExkaldiInfo.vertify_kaldi_existed()

		if self.data is None:
			raise WrongOperation('No any lattice to scale.')           

		for x in [acwt, invAcwt, ac2lm, lmwt, lm2ac]:
			assert x >= 0, "Expected scale is positive value."
		
		cmd = ExkaldiInfo.KALDI_ROOT + '/src/latbin/lattice-scale'
		cmd += ' --acoustic-scale={}'.format(acwt)
		cmd += ' --acoustic2lm-scale={}'.format(ac2lm)
		cmd += ' --inv-acoustic-scale={}'.format(invAcwt)
		cmd += ' --lm-scale={}'.format(lmwt)
		cmd += ' --lm2acoustic-scale={}'.format(lm2ac)
		cmd += ' ark:- ark:-'

		out, err, _ = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=self.data)

		if out == b'':
			print(err.decode())
			raise KaldiProcessError("Failed to scale lattice.")
		else:
			return Lattice(out, name=self.name)

	def add_penalty(self, penalty=0):
		'''
		Add penalty to lattice.

		Args:
			<penalty>: penalty.
		Return:
			An new Lattice object.
		'''
		ExkaldiInfo.vertify_kaldi_existed()

		if self.is_void:
			raise WrongOperation('No any lattice to scale.')

		assert isinstance(penalty, (int,float)) and penalty >= 0, "Expected <penalty> is positive int or float value."
		
		cmd = ExkaldiInfo.KALDI_ROOT + '/src/latbin/lattice-add-penalty'
		cmd += f' --word-ins-penalty={penalty}'
		cmd += ' ark:- ark:-'

		out, err, _ = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=self.data)

		if out == b'':
			print(err.decode())
			raise KaldiProcessError("Failed to add penalty.")
		else:
			return Lattice(out, name=self.name)

	def get_nbest(self, n, lexicons, hmm, acwt=1, requireCost=False):
		'''
		Get N best result with text formation.

		Args:
			<n>: n best results.
			<acwt>: acoustic weight.
			<requireAli>: If True, return Alignment object at the same time.
			<requireCost>: If True, return acoustic model and language model cost.
		Return:
			An exkaldi Transcription object (and Alignment).
		'''
		assert isinstance(n, int) and n > 0, "Expected <n> is a positive int value."
		assert isinstance(acwt, (int,float)) and acwt > 0, "Expected <acwt> is a positive int or float value."
	
		if self.is_void:
			raise WrongOperation('No any data in lattice.')
		
		outCostFile_lm = tempfile.NamedTemporaryFile('w+', encoding='utf-8')
		outCostFile_ac = tempfile.NamedTemporaryFile('w+', encoding='utf-8')
		outAliFile = tempfile.NamedTemporaryFile('wb+')
		words = tempfile.NamedTemporaryFile('w+', suffix=".txt", encoding='utf-8')

		try:
			lexicons.dump_dict("words", words)

			cmd = f'lattice-to-nbest --acoustic-scale={acwt} --n={n} ark:- ark:- |'
			cmd += f'nbest-to-linear ark:- ark:{outAliFile.name} "ark,t:|{ExkaldiInfo.KALDI_ROOT}/egs/wsj/s5/utils/int2sym.pl -f 2- {words.name}"'   
			
			if requireCost:
				cmd += ' ark,t:{} ark,t:{}'.format(outCostFile_lm.name, outCostFile_ac.name)

			out, err, _ = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=self.data)
			
			if out == b'':
				print(err.decode())
				raise KaldiProcessError('Failed to get N best results.')
			
			out = out.decode().strip().split("\n")
			if requireCost:
				allResult = []
				outCostFile_lm.seek(0)
				outCostFile_ac.seek(0)
				lines_am = outCostFile_ac.read().strip().split("\n")
				lines_lm = outCostFile_lm.read().strip().split("\n")

				for result, ac_score, lm_score in zip(out, lines_am, lines_lm):
					allResult.append([result, float(ac_score.split()[1]), float(lm_score.split()[1])])
				out = allResult

			results = {}
			for index, trans in enumerate(out):
				trans = trans.strip().split(maxsplit=1)
				name = trans[0][0:(trans[0].rfind("-"))]
				if len(trans) == 1:
					res = ""
				else: 
					res = trans[1]
				
				if requireCost:
					am_score = float(lines_am[index].split()[1])
					lm_score = float(lines_lm[index].split()[1])
					res = (res, am_score, lm_score)
				
				if not name in results.keys():
					results[name] = [res,]
				else:
					results[name].append(res)

			finalResult = [ {} for i in range(n)]
			finalAMCost = [ {} for i in range(n)]
			finalLMCost = [ {} for i in range(n)]
			for utt, trans in results.items():
				for i in range(n):
					try:
						t = trans[i]
					except IndexError:
						if requireCost:
							finalResult[i][utt] = " "
							finalAMCost[i][utt] = None
							finalLMCost[i][utt] = None
						else:
							finalResult[i][utt] = " "
					else:
						if requireCost:
							finalResult[i][utt] = t[0]
							finalAMCost[i][utt] = t[1]
							finalLMCost[i][utt] = t[2]
						else:
							finalResult[i][utt] = t
			
			if requireCost:
				return [ Transcription(t, name=f"{n}best-{index}", am_cost=ams, lm_cost=lms) for index, t, ams, lms in zip(range(1,n+1), finalResult, finalAMCost, finalLMCost) ]
			else:
				return [ Transcription(t, name=f"{n}best-{index}") for index, t in zip(range(1,n+1), finalResult) ]
				 
		finally:
			outCostFile_lm.close()
			outCostFile_ac.close()
			outAliFile.close()
			words.close()

def load_lat(target, name="lat"):
	'''
	Load lattice data.

	Args:
		<target>: bytes object, file path or exkaldi lattice object.
		<hmm>: file path or exkaldi HMM object.
		<wordSymbol>: file path or exkaldi LexiconBank object.
		<name>: a string.
	Return:
		A exkaldi lattice object.
	'''
	if isinstance(target, bytes):
		return Lattice(target, name)

	elif isinstance(target, str):
		if not os.path.isfile(target):
			raise WrongPath(f'No such file:{target}.')

		if target.endswith('.gz'):
			cmd = 'gunzip -c {}'.format(target)
			out, err, _ = run_shell_command(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
			if out == b'':
				print(err.decode())
				raise WrongDataFormat('Failed to load Lattice.')
			else:
				return Lattice(out, name)
		else:
			try:
				with open(target, 'rb') as fr:
					out = fr.read()
			except Exception as e:
				print("Load lattice file defeated. Please make sure it is a lattice file avaliable.")
				raise e
			else:
				return Lattice(out, name)

	else:
		raise UnsupportedDataType("Expected bytes object or alignment file.")

def nn_decode(postprob, hmm, hclgFile, wordSymbol, beam=10, latBeam=8, acwt=1,
				minActive=200, maxActive=7000, maxMem=50000000, config=None, maxThreads=1):
	'''
	Decode by generating lattice from acoustic probability output by NN model.

	Args:
		<postprob>: An exkaldi probability object. We expect the probability didn't pass any activation function, or it may generate wrong results.
		<hmm>: An exkaldi HMM object or file path.
		<hclgFile>: HCLG file path.
		<wordSymbol>: words.txt file path or exkaldi LexiconBank object.
		<beam>: beam size.
		<latBeam>: lattice beam size.
		<acwt>: acoustic model weight.
		<minActivate>: .
		<maxActive>: .
		<maxMem>: .
		<config>: decode configure file.
		<maxThreads>: the number of mutiple threads.
		
		Some usual options can be assigned directly. If you want use more, set <config> = your-configure, but if you do this, these usual configures we provided will be ignored.
		You can use .check_config('decode_lattice') function to get configure information you could set.
		Also run shell command "latgen-faster-mapped" to look their meaning.
	Return:
		An Lattice object.
	''' 
	ExkaldiInfo.vertify_kaldi_existed()

	if type_name(postprob) == "BytesProbability":
		pass
	elif type_name(postprob) == "NumpyProbability":
		postprob = postprob.to_bytes()
	else:
		raise UnsupportedDataType("Expected <postprob> is aexkaldi postprobability object.")
		
	assert isinstance(hclgFile, str), "<hclgFile> should be a file path."
	if not os.path.isfile(hclgFile):
		raise WrongPath(f"No such file:{hclgFile}")

	if maxThreads > 1:
		kaldiTool = f"latgen-faster-mapped-parallel --num-threads={maxThreads}"
	else:
		kaldiTool = "latgen-faster-mapped" 

	if config == None:    
		config = {}
		config["--allow-partial"] = "true"
		config["--min-active"] = minActive
		config["--max-active"] = maxActive
		config["--max_mem"] = maxMem
		config["--beam"] = beam
		config["--lattice-beam"] = latBeam
		config["--acoustic-scale"] = acwt

	words = tempfile.NamedTemporaryFile("w+", suffix=".txt", encoding="utf-8")
	model = tempfile.NamedTemporaryFile("wb+", suffix=".mdl")

	try:
		if type_name(wordSymbol) == "LexiconBank":
			wordSymbol.dump_dict("words", words)
			wordsFile = words.name
		elif isinstance(wordSymbol, str):
			if not os.path.isfile(wordSymbol):
				raise WrongPath(f"No such file:{wordSymbol}.")
			else:
				wordsFile = wordSymbol
		else:
			raise UnsupportedDataType(f"<wordSymbol> should be a file path or exkaldi LexiconBank object but got {type_name(wordSymbol)}.")

		config["--word-symbol-table"] = wordsFile

		if check_config(name='decode_lattice',config=config):
			for key in config.keys():
				kaldiTool += f' {key}={config[key]}'

		if type_name(hmm) in ["HMM", "MonophoneHMM", "TriphoneHMM"]:
			model.write(hmm.data)
			model.seek(0)
			hmmFile = model.name
		elif isinstance(hmm, str):
			if not os.path.isfile(hmm):
				raise WrongPath(f"No such file:{hmm}.")
			else:
				hmmFile = hmm
		else:
			raise UnsupportedDataType(f"<hmm> should be exkaldi HMM object or file path but got {type_name(hmm)}.")
		
		postprob = postprob.map(func=lambda x:log_softmax(x, 1))

		cmd = f'{kaldiTool} {hmmFile} {hclgFile} ark:- ark:-'
		out, err, _ = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=postprob.data)

		if out == b'':
			print(err.decode())
			raise KaldiProcessError('Failed to generate lattice.')
		else:
			return Lattice(data=out)
	
	finally:
		words.close()
		model.close()

def gmm_decode(feats, hmm, hclgFile, wordSymbol, beam=10, latBeam=8, acwt=1,
				minActive=200, maxActive=7000, maxMem=50000000, config=None, maxThreads=1):
	'''
	Decode by generating lattice from feature and GMM model.

	Args:
		<feats>: An exkaldi feature object.
		<hmm>: An exkaldi HMM object or file path.
		<hclgFile>: HCLG file path.
		<wordSymbol>: words.txt file path or exkaldi LexiconBank object.
		<beam>: beam size.
		<latBeam>: lattice beam size.
		<acwt>: acoustic model weight.
		<minActivate>: .
		<maxActive>: .
		<maxMem>: .
		<config>: decode configure file.
		<maxThreads>: the number of mutiple threads.
		
		Some usual options can be assigned directly. If you want use more, set <config> = your-configure, but if you do this, these usual configures we provided will be ignored.
		You can use .check_config('decode_lattice') function to get configure information you could set.
		Also run shell command "gmm-latgen-faster" to look their meaning.
	Return:
		An Lattice object.
	''' 
	ExkaldiInfo.vertify_kaldi_existed()

	if type_name(feats) == "BytesFeature":
		pass
	elif type_name(feats) == "NumpyFeature":
		feats = feats.to_bytes()
	else:
		raise UnsupportedDataType("Expected <feats> is an exkaldi feature object.")
		
	assert isinstance(hclgFile, str), "<hclgFile> should be a file path."
	if not os.path.isfile(hclgFile):
		raise WrongPath(f"No such file:{hclgFile}")

	if maxThreads > 1:
		kaldiTool = f"gmm-latgen-faster-parallel --num-threads={maxThreads}"
	else:
		kaldiTool = "gmm-latgen-faster" 

	if config == None:    
		config = {}
		config["--allow-partial"] = "true"
		config["--min-active"] = minActive
		config["--max-active"] = maxActive
		config["--max_mem"] = maxMem
		config["--beam"] = beam
		config["--lattice-beam"] = latBeam
		config["--acoustic-scale"] = acwt

	words = tempfile.NamedTemporaryFile("w+", suffix=".txt", encoding="utf-8")
	model = tempfile.NamedTemporaryFile("wb+", suffix=".mdl")

	try:
		if type_name(wordSymbol) == "LexiconBank":
			wordSymbol.dump_dict("words", words)
			wordsFile = words.name
		elif isinstance(wordSymbol, str):
			if not os.path.isfile(wordSymbol):
				raise WrongPath(f"No such file:{wordSymbol}.")
			else:
				wordsFile = wordSymbol
		else:
			raise UnsupportedDataType(f"<wordSymbol> should be a file path or exkaldi LexiconBank object but got {type_name(wordSymbol)}.")

		config["--word-symbol-table"] = wordsFile

		if check_config(name='decode_lattice',config=config):
			for key in config.keys():
				kaldiTool += f' {key}={config[key]}'

		if type_name(hmm) in ["HMM", "MonophoneHMM", "TriphoneHMM"]:
			model.write(hmm.data)
			model.seek(0)
			hmmFile = model.name
		elif isinstance(hmm, str):
			if not os.path.isfile(hmm):
				raise WrongPath(f"No such file:{hmm}.")
			else:
				hmmFile = hmm
		else:
			raise UnsupportedDataType(f"<hmm> should be exkaldi HMM object or file path but got {type_name(hmm)}.")
		
		cmd = f'{kaldiTool} {hmmFile} {hclgFile} ark:- ark:-'
		out, err, _ = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=feats.data)

		if out == b'':
			print(err.decode())
			raise KaldiProcessError('Failed to generate lattice.')
		else:
			return Lattice(data=out)
	
	finally:
		words.close()
		model.close()