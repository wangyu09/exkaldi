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
import copy
import numpy as np

from exkaldi.version import version as ExkaldiInfo
from exkaldi.version import WrongPath, WrongOperation, WrongDataFormat, KaldiProcessError, UnsupportedType
from exkaldi.utils.utils import run_shell_command, make_dependent_dirs, type_name, check_config, list_files
from exkaldi.core.archieve import BytesArchieve, Transcription, ListTable, BytesAlignmentTrans, NumpyAlignmentTrans, Metric
from exkaldi.nn.nn import log_softmax
from exkaldi.hmm.hmm import load_hmm

class Lattice(BytesArchieve):
	'''
	Usage:  obj = KaldiLattice() or obj = KaldiLattice(lattice,hmm,wordSymbol)

	KaldiLattice holds the lattice and its related file path: HMM file and WordSymbol file. 
	The <lattice> can be lattice binary data or file path. Both <hmm> and <wordSymbol> are expected to be file path.
	decode_lattice() function will return a KaldiLattice object. Aslo, you can define a empty KaldiLattice object and load its data later.
	'''
	def __init__(self, data=None, wordSymbolTable=None, hmm=None, name="lat"):
		super().__init__(data, name)
		if wordSymbolTable is not None:
			assert type_name(wordSymbolTable) == "ListTable", f"<wordSymbolTable> must be exkaldi ListTable object but got: {type_name(wordSymbolTable)}."
		if hmm is not None:
			assert type_name(hmm) in ["MonophoneHMM","TriphoneHMM"], f"<hmm> must be exkaldi HMM object but got: {type_name(hmm)}."
		
		self.__wordSymbolTable = wordSymbolTable
		self.__hmm = hmm
	
	@property
	def wordSymbolTable(self):
		return copy.deepcopy(self.__wordSymbolTable)
	
	@property
	def hmm(self):
		return self.__hmm

	def save(self, fileName):
		'''
		Save lattice as .ali file. 
		
		Args:
			<fileName>: file name.
		''' 
		assert isinstance(fileName, str) and len(fileName) > 0, "file name is unavaliable."

		if self.is_void:
			raise WrongOperation('No any data to save.')

		if not fileName.rstrip().endswith(".lat"):
			fileName += ".lat"
		
		make_dependent_dirs(fileName)

		with open(fileName, "wb") as fw:
			fw.write(self.data)

		return os.path.abspath(fileName)

	def get_1best(self, wordSymbolTable=None, hmm=None, lmwt=1, acwt=1.0, phoneLevel=False):
		'''
		Get 1 best result with text formation.

		Args:
			<wordSymbolTable>: None or file path or ListTable object or LexiconBank object.
			<hmm>: None or file path or HMM object.
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

		if wordSymbolTable is None:
			assert self.wordSymbolTable is not None, "<wordSymbolTable> is necessary because no wordSymbol table is avaliable."
			wordSymbolTable = self.wordSymbolTable
		
		if hmm is None:
			assert self.hmm is not None, "<hmm> is necessary because no wordSymbol table is avaliable."
			hmm = self.hmm

		modelTemp = tempfile.NamedTemporaryFile("wb+", suffix=".mdl")
		wordSymbolTemp = tempfile.NamedTemporaryFile("w+", suffix="_words.txt", encoding="utf-8")

		try:
			if isinstance(wordSymbolTable, str):
				assert os.path.isfile(wordSymbolTable), f"No such file: {wordSymbolTable}."
				wordsFile = wordSymbolTable
			elif type_name(wordSymbolTable) == "LexiconBank":
				if phoneLevel:
					wordSymbolTable.dump_dict("phones", wordSymbolTemp)
				else:
					wordSymbolTable.dump_dict("words", wordSymbolTemp)
				wordsFile = wordSymbolTemp.name
			elif type_name(wordSymbolTable) == "ListTable":
				wordSymbolTable.save(wordSymbolTemp)
				wordSymbolTemp.seek(0)
				wordsFile = wordSymbolTemp.name
			else:
				raise UnsupportedType(f"<wordSymbolTable> should be file name, LexiconBank object or ListTable object but got: {type_name(wordSymbolTable)}.")

			if isinstance(hmm, str):
				assert os.path.isfile(hmm), f"No such file: {hmm}."
				hmmFile = hmm
			elif type_name(hmm) in ["MonophoneHMM","TriphoneHMM"]:
				hmm.save(modelTemp)
				hmmFile = modelTemp.name
			else:
				raise UnsupportedType(f"<hmm> should be file name, exkaldi HMM object but got: {type_name(hmm)}.")

			if phoneLevel:
				cmd0 = f'lattice-align-phones --replace-output-symbols=true {hmmFile} ark:- ark:- | '
			else:
				cmd0 = ""

			cmd1 = f"lattice-best-path --lm-scale={lmwt} --acoustic-scale={acwt} --word-symbol-table={wordsFile} --verbose=2 ark:- ark,t:- "
			cmd = cmd0 + cmd1

			out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=self.data)
			if cod != 0 or out == b'':
				print(err.decode())
				raise KaldiProcessError('Failed to get 1-best from lattice.')
			else:
				out = out.decode().strip().split("\n")
				if phoneLevel:
					newName = "1-best-phones"
				else:
					newName = "1-best-words"

				results = Transcription(name=newName)
				for re in out:
					re = re.strip().split(maxsplit=1)
					if len(re) == 0:
						continue
					elif len(re) == 1:
						results[re[0]] = " "
					else:
						results[re[0]] = re[1]
				return results

		finally:
			modelTemp.close()
			wordSymbolTemp.close()
	
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

		if self.is_void:
			raise WrongOperation('No any lattice to scale.')           

		for x in [acwt, invAcwt, ac2lm, lmwt, lm2ac]:
			assert x >= 0, "Expected scale is positive value."
		
		cmd = 'lattice-scale'
		cmd += ' --acoustic-scale={}'.format(acwt)
		cmd += ' --acoustic2lm-scale={}'.format(ac2lm)
		cmd += ' --inv-acoustic-scale={}'.format(invAcwt)
		cmd += ' --lm-scale={}'.format(lmwt)
		cmd += ' --lm2acoustic-scale={}'.format(lm2ac)
		cmd += ' ark:- ark:-'

		out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=self.data)

		if cod != 0 or out == b'':
			print(err.decode())
			raise KaldiProcessError("Failed to scale lattice.")
		else:
			newName = f"scale({self.name})"
			return Lattice(data=out,wordSymbolTable=self.wordSymbolTable,hmm=self.hmm,name=newName)

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
		
		cmd = f"lattice-add-penalty --word-ins-penalty={penalty} ark:- ark:-"

		out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=self.data)

		if cod != 0 or out == b'':
			print(err.decode())
			raise KaldiProcessError("Failed to add penalty.")
		else:
			newName = f"add_penalty({self.name})"
			return Lattice(data=out, wordSymbolTable=self.wordSymbolTable, hmm=self.hmm, name=newName)

	def get_nbest(self, n, wordSymbolTable=None, hmm=None, acwt=1, phoneLevel=False, requireAli=False, requireCost=False):
		'''
		Get N best result with text formation.

		Args:
			<n>: n best results.
			<wordSymbolTable>: file or ListTable object or LexiconBank object.
			<hmm>: file or HMM object.
			<acwt>: acoustic weight.
			<phoneLevel>: If True, return phone results.
			<requireAli>: If True, return alignment simultaneously.
			<requireCost>: If True, return acoustic model and language model cost simultaneously.

		Return:
			A list of exkaldi Transcription objects (and their Metric objects).
		'''
		assert isinstance(n, int) and n > 0, "Expected <n> is a positive int value."
		assert isinstance(acwt, (int,float)) and acwt > 0, "Expected <acwt> is a positive int or float value."
	
		if self.is_void:
			raise WrongOperation('No any data in lattice.')
		
		if wordSymbolTable is None:
			assert self.wordSymbolTable is not None, "<wordSymbolTable> is necessary because no wordSymbol table is avaliable."
			wordSymbolTable = self.wordSymbolTable
		
		if hmm is None:
			assert self.hmm is not None, "<hmm> is necessary because no wordSymbol table is avaliable."
			hmm = self.hmm

		wordSymbolTemp = tempfile.NamedTemporaryFile('w+', suffix="_words.txt", encoding='utf-8')
		modelTemp = tempfile.NamedTemporaryFile("wb+", suffix=".mdl")
		outAliTemp = tempfile.NamedTemporaryFile('w+', suffix=".ali", encoding='utf-8')
		outCostFile_LM = tempfile.NamedTemporaryFile('w+', suffix=".cost", encoding='utf-8')
		outCostFile_AM = tempfile.NamedTemporaryFile('w+', suffix=".cost", encoding='utf-8')

		try:
			if isinstance(wordSymbolTable, str):
				assert os.path.isfile(wordSymbolTable), f"No such file: {wordSymbolTable}."
				wordsFile = wordSymbolTable
			elif type_name(wordSymbolTable) == "LexiconBank":
				if phoneLevel:
					wordSymbolTable.dump_dict("phones", wordSymbolTemp)
				else:
					wordSymbolTable.dump_dict("words", wordSymbolTemp)
				wordsFile = wordSymbolTemp.name
			elif type_name(wordSymbolTable) == "ListTable":
				wordSymbolTable.save(wordSymbolTemp)
				wordSymbolTemp.seek(0)
				wordsFile = wordSymbolTemp.name
			else:
				raise UnsupportedType(f"<wordSymbolTable> should be file name, LexiconBank object or ListTable object but got: {type_name(wordSymbolTable)}.")

			if isinstance(hmm, str):
				assert os.path.isfile(hmm), f"No such file: {hmm}."
				hmmFile = hmm
			elif type_name(hmm) in ["MonophoneHMM","TriphoneHMM"]:
				hmm.save(modelTemp)
				hmmFile = modelTemp.name
			else:
				raise UnsupportedType(f"<hmm> should be file name, exkaldi HMM object but got: {type_name(hmm)}.")

			if phoneLevel:
				cmd = f'lattice-align-phones --replace-output-symbols=true {hmmFile} ark:- ark:- | '
			else:
				cmd = ""			

			cmd += f'lattice-to-nbest --acoustic-scale={acwt} --n={n} ark:- ark:- |'
			cmd += f'nbest-to-linear ark:- ark,t:{outAliTemp.name} ark,t:-'   
			
			if requireCost:
				cmd += f' ark,t:{outCostFile_LM.name} ark,t:{outCostFile_AM.name}'

			out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=self.data)
			
			if cod != 0 or out == b'':
				print(err.decode())
				raise KaldiProcessError('Failed to get N best results.')
			
			def sperate_n_bests(data):
				results	= {}
				for index,trans in enumerate(data):
					trans = trans.strip().split(maxsplit=1)
					if len(trans) == 0:
						continue
					name = trans[0][0:(trans[0].rfind("-"))]
					if len(trans) == 1:
						res = " "
					else:
						res = trans[1]
					if not name in results.keys():
						results[name] = [res,]
					else:
						results[name].append(res)
				
				finalResults = []
				for uttID, nbests in results.items():
					for index, one in enumerate(nbests):
						if index > len(finalResults)-1:
							finalResults.append({})
						finalResults[index][uttID] = one

				return finalResults

			out = out.decode().strip().split("\n")

			out = sperate_n_bests(out)
			NBEST = []
			for index, one in enumerate(out,start=1):
				name = f"{index}-best"
				NBEST.append( Transcription(one, name=name) )
			del out

			if requireCost:
				outCostFile_AM.seek(0)
				lines_AM = outCostFile_AM.read().strip().split("\n")
				lines_AM = sperate_n_bests(lines_AM)
				AMSCORE = []
				for index, one in enumerate(lines_AM, start=1):
					name = f"AM-{index}-best"
					AMSCORE.append( Metric(one, name=name) )
				del lines_AM			

				outCostFile_LM.seek(0)
				lines_LM = outCostFile_LM.read().strip().split("\n")
				lines_LM = sperate_n_bests(lines_LM)
				LMSCORE = []
				for index, one in enumerate(lines_LM, start=1):
					name = f"LM-{index}-best"
					LMSCORE.append( Metric(one, name=name) )
				del lines_LM

				finalResult = [NBEST,AMSCORE,LMSCORE]
			else:
				finalResult = [NBEST,]

			if requireAli:
				ALIGNMENT = []
				outAliTemp.seek(0)
				ali = outAliTemp.read().strip().split("\n")
				ali = sperate_n_bests(ali)
				for index, one in enumerate(ali, start=1):
					name = f"{index}-best"
					temp = {}
					for key, value in one.items():
						value = value.strip().split()
						temp[key] = np.array(value, dtype=np.int32)
					ALIGNMENT.append( NumpyAlignmentTrans(temp, name=name) )
				del ali
				finalResult.append(ALIGNMENT)

			if len(finalResult) == 1:
				finalResult = finalResult[0]

			return finalResult
			 
		finally:
			wordSymbolTemp.close()
			modelTemp.close()
			outAliTemp.close()
			outCostFile_LM.close()
			outCostFile_AM.close()

	def determinize(self, acwt=1.0, beam=6):
		'''
		Determinize the lattice.

		Args:
			<acwt>: acoustic scale.
			<beam>: prune beam.
		Return:
			An new Lattice object.
		'''
		ExkaldiInfo.vertify_kaldi_existed()

		if self.is_void:
			raise WrongOperation('No any lattice data.')

		assert isinstance(acwt, float) and acwt >= 0, "Expected <acwt> is positive float value."
		assert isinstance(beam, int) and beam >= 0, "Expected <beam> is positive int value."
		
		cmd = f"lattice-determinize-pruned --acoustic-scale={acwt} --beam={beam} ark:- ark:-"

		out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=self.data)

		if cod != 0 or out == b'':
			print(err.decode())
			raise KaldiProcessError("Failed to determinize lattice.")
		else:
			newName = f"determinize({self.name})"
			return Lattice(data=out, wordSymbolTable=self.wordSymbolTable, hmm=self.hmm, name=newName)		

	def am_rescore(self, hmm, feat):
		"""
		Replace the acoustic scores with new HMM-GMM model.
		"""
		'''
		Determinize the lattice.

		Args:
			<hmm>: exkaldi HMM object or file path.

		Return:
			An new Lattice object.
		'''
		ExkaldiInfo.vertify_kaldi_existed()

		if self.is_void:
			raise WrongOperation('No any lattice data.')

		hmmTemp = tempfile.NamedTemporaryFile("wb+", suffix=".mdl")
		featTemp = tempfile.NamedTemporaryFile("wb+", suffix=".mdl")
		try:
			if isinstance(hmm, str):
				assert os.path.isfile(hmm), f"No such file: {hmm}."
				hmmFile = hmm
			elif type_name(hmm) in ["BaseHMM", "MonophoneHMM", "TriphoneHMM"]:
				hmmTemp.write(hmm.data)
				hmmTemp.seek(0)
				hmmFile = hmmTemp.name
			else:
				raise UnsupportedType(f"<hmm> should be file path or exkaldi HMM object but got: {type_name(hmm)}.")
	
			if type_name(feat) == "BytesFeature":
				feat = feat.sort(by="utt")
			elif type_name(feat) == "NumpyFeature":
				feat = feat.sort(by="utt").to_numpy()
			else:
				raise UnsupportedType(f"<feat> should be exkaldi feature object but got: {type_name(feat)}.")

			featTemp.write(feat.data)
			featTemp.seek(0)
			featFile = featTemp.name

			cmd = f"gmm-rescore-lattice	{hmmFile} ark:- ark:{featFile} ark:-"

			out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=self.data)

			if cod != 0 or out == b'':
				print(err.decode())
				raise KaldiProcessError("Failed to determinize lattice.")
			else:
				newName = f"am_rescore({self.name})"
				return Lattice(data=out, wordSymbolTable=self.wordSymbolTable, hmm=self.hmm, name=newName)
		finally:
			hmmTemp.close()
			featTemp.close()

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
		target = list_files(target)
		allData = []
		for fileName in target:
			if fileName.endswith('.gz'):
				cmd = 'gunzip -c {}'.format(fileName)
				out, err, _ = run_shell_command(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
				if out == b'':
					print(err.decode())
					raise WrongDataFormat('Failed to load Lattice.')
				else:
					allData.append(out)
			else:
				try:
					with open(fileName, 'rb') as fr:
						out = fr.read()
				except Exception as e:
					print("Load lattice file defeated. Please make sure it is a lattice file avaliable.")
					raise e
				else:
					allData.append(out)
		try:
			allData = b"".join(allData)
		except Exception as e:
			raise WrongOperation("Only support binary format lattice file.")
		else:
			return Lattice(data=allData, name=name)

	else:
		raise UnsupportedType(f"Expected bytes object or lattice file but got: {type_name(target)}.")

def nn_decode(postprob, hmm, HCLGFile, wordSymbolTable, beam=10, latBeam=8, acwt=1,
				minActive=200, maxActive=7000, maxMem=50000000, config=None, maxThreads=1):
	'''
	Decode by generating lattice from acoustic probability output by NN model.

	Args:
		<postprob>: An exkaldi probability object. We expect the probability didn't pass any activation function, or it may generate wrong results.
		<hmm>: An exkaldi HMM object or file path.
		<HCLGFile>: HCLG file path.
		<wordSymbolTable>: words.txt file path or exkaldi LexiconBank object.
		<beam>: beam size.
		<latBeam>: lattice beam size.
		<acwt>: acoustic model weight.
		<minActivate>: .
		<maxActive>: .
		<maxMem>: .
		<config>: decode configure file.
		<maxThreads>: the number of mutiple threads.
		
		Some usual options can be assigned directly. If you want use more, set <config> = your-configure.
		You can use .check_config('nn_decode') function to get configure information you could set.
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
		raise UnsupportedType("Expected <postprob> is aexkaldi postprobability object.")
		
	assert isinstance(HCLGFile, str), "<HCLGFile> should be a file path."
	if not os.path.isfile(HCLGFile):
		raise WrongPath(f"No such file:{HCLGFile}")

	if maxThreads > 1:
		kaldiTool = f"latgen-faster-mapped-parallel --num-threads={maxThreads} "
	else:
		kaldiTool = "latgen-faster-mapped " 

	kaldiTool += f'--allow-partial=true '
	kaldiTool += f'--min-active={minActive} '
	kaldiTool += f'--max-active={maxActive} '  
	kaldiTool += f'--max_mem={maxMem} '
	kaldiTool += f'--beam={beam} '
	kaldiTool += f'--lattice-beam={latBeam} '
	kaldiTool += f'--acoustic-scale={acwt} '

	wordsTemp = tempfile.NamedTemporaryFile("w+", suffix=".txt", encoding="utf-8")
	modelTemp = tempfile.NamedTemporaryFile("wb+", suffix=".mdl")

	try:
		if type_name(wordSymbolTable) == "LexiconBank":
			wordSymbolTable.dump_dict("words", wordsTemp)
			wordsFile = wordsTemp.name
		elif type_name(wordSymbolTable) == "ListTable":
			wordSymbolTable.save(wordsTemp)
			wordsTemp.seek(0)
			wordsFile = wordsTemp.name
		elif isinstance(wordSymbolTable, str):
			if not os.path.isfile(wordSymbolTable):
				raise WrongPath(f"No such file:{wordSymbolTable}.")
			else:
				wordsFile = wordSymbolTable
		else:
			raise UnsupportedType(f"<wordSymbolTable> should be a file path or exkaldi LexiconBank object but got {type_name(wordSymbolTable)}.")

		kaldiTool += f'--word-symbol-table={wordsFile} '

		if config is not None:
			if check_config(name='nn_decode', config=config):
				for key,value in config.items():
					if isinstance(value, bool):
						if value is True:
							kaldiTool += f"{key} "
					else:
						kaldiTool += f" {key}={value}"

		if type_name(hmm) in ["HMM", "MonophoneHMM", "TriphoneHMM"]:
			modelTemp.write(hmm.data)
			modelTemp.seek(0)
			hmmFile = modelTemp.name
		elif isinstance(hmm, str):
			if not os.path.isfile(hmm):
				raise WrongPath(f"No such file:{hmm}.")
			else:
				hmmFile = hmm
		else:
			raise UnsupportedType(f"<hmm> should be exkaldi HMM object or file path but got {type_name(hmm)}.")
		
		cmd = f'{kaldiTool} {hmmFile} {HCLGFile} ark:- ark:-'
		out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=postprob.data)

		if cod !=0 or out == b'':
			print(err.decode())
			raise KaldiProcessError('Failed to generate lattice.')
		else:
			newName = f"lat({postprob.name})"
			return Lattice(data=out, name=newName)
	
	finally:
		wordsTemp.close()
		modelTemp.close()

def gmm_decode(feat, hmm, HCLGFile, wordSymbolTable, beam=10, latBeam=8, acwt=1,
				minActive=200, maxActive=7000, maxMem=50000000, config=None, maxThreads=1):
	'''
	Decode by generating lattice from feature and GMM model.

	Args:
		<feat>: An exkaldi feature object.
		<hmm>: An exkaldi HMM object or file path.
		<HCLGFile>: HCLG file path.
		<wordSymbolTable>: words.txt file path or exkaldi LexiconBank object or exkaldi ListTable object.
		<beam>: beam size.
		<latBeam>: lattice beam size.
		<acwt>: acoustic model weight.
		<minActivate>: .
		<maxActive>: .
		<maxMem>: .
		<config>: decode configure file.
		<maxThreads>: the number of mutiple threads.
		
		Some usual options can be assigned directly. If you want use more, set <config> = your-configure, but if you do this, these usual configures we provided will be ignored.
		You can use .check_config('gmm_decode') function to get configure information you could set.
		Also run shell command "gmm-latgen-faster" to look their meaning.
	Return:
		An exkaldi Lattice object.
	''' 
	ExkaldiInfo.vertify_kaldi_existed()

	if type_name(feat) == "BytesFeature":
		pass
	elif type_name(feat) == "NumpyFeature":
		feat = feat.to_bytes()
	else:
		raise UnsupportedType(f"Expected <feat> is an exkaldi feature object but got: {type_name(feat)}.")
		
	assert isinstance(HCLGFile, str), "<HCLGFile> should be a file path."
	if not os.path.isfile(HCLGFile):
		raise WrongPath(f"No such file:{HCLGFile}")

	if maxThreads > 1:
		kaldiTool = f"gmm-latgen-faster-parallel --num-threads={maxThreads} "
	else:
		kaldiTool = "gmm-latgen-faster " 

	kaldiTool += f'--allow-partial=true '
	kaldiTool += f'--min-active={minActive} '
	kaldiTool += f'--max-active={maxActive} '  
	kaldiTool += f'--max_mem={maxMem} '
	kaldiTool += f'--beam={beam} '
	kaldiTool += f'--lattice-beam={latBeam} '
	kaldiTool += f'--acoustic-scale={acwt} '

	wordsTemp = tempfile.NamedTemporaryFile("w+", suffix="_words.txt", encoding="utf-8")
	modelTemp = tempfile.NamedTemporaryFile("wb+", suffix=".mdl")

	try:
		if type_name(wordSymbolTable) == "LexiconBank":
			wordSymbolTable.dump_dict("words", wordsTemp)
			wordsFile = wordsTemp.name
		elif type_name(wordSymbolTable) == "ListTable":
			wordSymbolTable.save(wordsTemp)
			wordsTemp.seek(0)
			wordsFile = wordsTemp.name
		elif isinstance(wordSymbolTable, str):
			if not os.path.isfile(wordSymbolTable):
				raise WrongPath(f"No such file:{wordSymbolTable}.")
			else:
				wordsFile = wordSymbolTable
		else:
			raise UnsupportedType(f"<wordSymbolTable> should be a file path or exkaldi LexiconBank object but got {type_name(wordSymbolTable)}.")

		kaldiTool += f'--word-symbol-table={wordsFile} '

		if config is not None:
			if check_config(name='gmm_decode', config=config):
				for key,value in config.items():
					if isinstance(value, bool):
						if value is True:
							kaldiTool += f"{key} "
					else:
						kaldiTool += f" {key}={value}"

		if type_name(hmm) in ["MonophoneHMM", "TriphoneHMM"]:
			modelTemp.write(hmm.data)
			modelTemp.seek(0)
			hmmFile = modelTemp.name
		elif isinstance(hmm, str):
			if not os.path.isfile(hmm):
				raise WrongPath(f"No such file:{hmm}.")
			else:
				hmmFile = hmm
		else:
			raise UnsupportedType(f"<hmm> should be exkaldi HMM object or file path but got {type_name(hmm)}.")
		
		cmd = f'{kaldiTool} {hmmFile} {HCLGFile} ark:- ark:-'
		out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=feat.data)

		if cod != 0 or out == b'':
			print(err.decode())
			raise KaldiProcessError('Failed to generate lattice.')
		else:
			newName = f"lat({feat.name})"
			return Lattice(data=out, name=newName)
	
	finally:
		wordsTemp.close()
		modelTemp.close()

def compile_align_graph(hmm, tree, transcription, LFile, outFile, lexicons=None):
	'''
	Compile graph for training or aligning.
		<lexicons>: None. If no any lexicons provided in DecisionTree, this is expected.
					In this step, we will use "context_indep" lexicon.
	'''
	if isinstance(hmm,str):
		assert os.path.isfile(hmm), f"No such file: {hmm}."
		assert type_name(lexicons) == "LexiconBank", "Expected <lexicons> is provided in this case."
		hmm = load_hmm(hmm, lexicons=lexicons)
	else:
		assert type_name(hmm) in ["BaseHMM","MonophoneHMM","TriphoneHMM"], f"<hmm> should be exkaldi HMM object but got: {hmm}."

	return hmm.compile_train_graph(tree, transcription, LFile, outFile)

def nn_align(hmm, prob, alignGraphFile=None, tree=None, transcription=None, Lfile=None, transitionScale=1.0, acousticScale=0.1, 
				selfloopScale=0.1, beam=10, retry_beam=40, lexicons=None, name="ali"):
	'''
	Align the neural network acoustic output probability.
	'''
	if type_name(prob) == "BytesProbability":
		pass
	elif type_name(prob) == "NumpyProbability":
		prob = prob.to_bytes()
	else:
		raise UnsupportedType(f"Expected <prob> is an exkaldi probability object but got: {type_name(prob)}.")

	hmmTemp = tempfile.NamedTemporaryFile("wb+", suffix=".mdl")
	treeTemp = tempfile.NamedTemporaryFile("wb+", suffix=".tree")
	disambigTemp = tempfile.NamedTemporaryFile("w+", suffix="_disambig.int", encoding="utf-8")
	try:
		if isinstance(hmm,str):
			assert os.path.isfile(hmm), f"No such file: {hmm}."
			hmmFile = hmm
		else:
			assert type_name(hmm) in ["BaseHMM","MonophoneHMM","TriphoneHMM"], f"<hmm> should be exkaldi HMM object but got: {hmm}."
			hmmTemp.write(hmm.data)
			hmmTemp.seek(0)
			hmmFile = hmmTemp.name

		if alignGraphFile is None:
			assert None not in [tree,transcription,Lfile], "When align graph is not provided, all of <tree>, <transcription> and <Lfile> is necessary."
			assert isinstance(transcription,str) and os.path.isfile(transcription), "Transcription is not a file name or does not exist."
			assert isinstance(Lfile,str) and os.path.isfile(Lfile), "Lexicon fst is not a file name or does not exist."
			
			if lexicons is not None:
				assert type_name(lexicons) == "LexiconBank", "<lexicons> is necessary."
			else:
				if isinstance(hmm,str) or hmm.lex is None:
					raise WrongOperation("<lexicons> is necessary in this case.")
				else:
					lexicons = hmm.lex
			
			if type_name(tree) == "DecisionTree":
				treeTemp.write(tree.data)
				treeTemp.seek(0)
				treeFile = treeTemp.name
			else:
				assert isinstance(tree, str) and os.path.isfile(tree), "Tree is not a file name or does not exist."
				treeFile = tree
			
			lexicons.dump_dict(name="disambig", outFile=disambigTemp, dumpInt=True)
			
			cmd = f"align-mapped --transition-scale={transitionScale} --acoustic-scale={acousticScale} --self-loop-scale={selfloopScale} "
			cmd += f"--beam={beam} --retry-beam={retry_beam} "
			cmd += f"--read-disambig-syms={disambigTemp.name} "
			cmd += f"{treeFile} {hmmFile} {Lfile} ark:- ark:{transcription} ark:-"

		else:
			assert tree is None and transcription is None and Lfile is None, "When use compiled align graph, any of <tree>, <transcription> and <Lfile> is invalid."
			assert isinstance(alignGraphFile,str) and os.path.isfile(alignGraphFile), "Align graph is not a file name or does not exist."

			cmd = f"align-compiled-mapped --transition-scale={transitionScale} --acoustic-scale={acousticScale} --self-loop-scale={selfloopScale} "
			cmd += f"--beam={beam} --retry-beam={retry_beam} "
			cmd += f"{hmmFile} ark:{alignGraphFile} ark:- ark:-"

		out,err,cod = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=prob.data)

		if cod != 0:
			print(err.decode())
			raise KaldiProcessError("Failed to align probability.")
		else:
			return BytesAlignmentTrans(out,name=name)
	
	finally:
		hmmTemp.close()
		treeTemp.close()
		disambigTemp.close()

def gmm_align(hmm, feat, alignGraphFile=None, tree=None, transcription=None, Lfile=None, transitionScale=1.0, acousticScale=0.1, 
				selfloopScale=0.1, beam=10, retry_beam=40, boost_silence=1.0, careful=False, name="ali", lexicons=None):
	'''
	Align acoustic feature with kaldi vertibi algorithm.
		<lexicons>: None. If no any lexicons provided in DecisionTree, this is expected.
					In this step, we will use "context_indep" lexicon.
	'''
	if type_name(feat) == "BytesFeature":
		feat = feat.sort(by="utt")
	elif type_name(feat) == "NumpyFeature":
		feat = feat.sort(by="utt").to_bytes()
	else:
		raise UnsupportedType(f"Expected exkaldi feature object but got {type_name(feat)}.")

	hmmTemp = tempfile.NamedTemporaryFile("wb+", suffix=".mdl")
	treeTemp = tempfile.NamedTemporaryFile("wb+", suffix=".tree")
	disambigTemp = tempfile.NamedTemporaryFile("w+", suffix="_disambig.int", encoding="utf-8")

	try:
		if isinstance(hmm,str):
			assert os.path.isfile(hmm), f"No such file: {hmm}."
			assert type_name(lexicons) == "LexiconBank", "<lexicons> is necessary in this case."
			optionSilence = ":".join(lexicons("optional_silence", True))

			cmd = f'gmm-boost-silence --boost={boost_silence} {optionSilence} {hmm} {hmmTemp.name}'
			out, err, cod = run_shell_command(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

		else:
			assert type_name(hmm) in ["BaseHMM","MonophoneHMM","TriphoneHMM"], f"<hmm> should be exkaldi HMM object but got: {hmm}."
			if lexicons is None:
				assert hmm.lex is not None, "<lexicons> is necessary in this case."
				lexicons = hmm.lex
			else:
				assert type_name(lexicons) == "LexiconBank", f"<lexicons> should be exkaldi LexiconBank object but got: {type_name(lexicons)}."

			optionSilence = ":".join(lexicons("optional_silence", True))

			cmd = f'gmm-boost-silence --boost={boost_silence} {optionSilence} - {hmmTemp.name}'
			out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=hmm.data)
		
		if (isinstance(cod,int) and cod != 0 ) or os.path.getsize(hmmTemp.name) == 0:
			print(err.decode())
			raise KaldiProcessError("Generate new HMM defeated.")
		hmmTemp.ssek(0)

		if alignGraphFile is None:
			assert None not in [tree,transcription,Lfile], "When align graph is not provided, all of <tree>, <transcription> and <Lfile> is necessary."
			assert isinstance(transcription,str) and os.path.isfile(transcription), "Transcription is not a file name or does not exist."
			assert isinstance(Lfile,str) and os.path.isfile(Lfile), "Lexicon fst is not a file name or does not exist."

			if type_name(tree) == "DecisionTree":
				treeTemp.write(tree.data)
				treeTemp.seek(0)
				treeFile = treeTemp.name
			else:
				assert isinstance(tree, str) and os.path.isfile(tree), "Tree is not a file name or does not exist."
				treeFile = tree
			
			lexicons.dump_dict(name="disambig", outFile=disambigTemp, dumpInt=True)				
			
			cmd = f'gmm-align --transition-scale={transitionScale} --acoustic-scale={acousticScale} --self-loop-scale={selfloopScale} '
			cmd += f'--beam={beam} --retry-beam={retry_beam} --careful={careful} '
			cmd += f'{treeFile} {hmmTemp.name} {Lfile} ark:- ark:{transcription} ark:-'

		else:
			assert tree is None and transcription is None and Lfile is None, "When use compiled align graph, any of <tree>, <transcription> and <Lfile> is invalid."
			assert isinstance(alignGraphFile,str) and os.path.isfile(alignGraphFile), "Align graph is not a file name or does not exist."

			cmd = f"gmm-align-compiled --transition-scale={transitionScale} --acoustic-scale={acousticScale} --self-loop-scale={selfloopScale} "
			cmd += f"--beam={beam} --retry-beam={retry_beam} --careful={careful} "
			cmd += f"{hmmTemp.name} ark:{alignGraphFile} ark:- ark:-"

		out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=feat.data)

		if (isinstance(cod,int) and cod != 0):
			print(err.decode())
			raise KaldiProcessError("Failed to align feature.")
		
		return BytesAlignmentTrans(out, name=name)
	
	finally:
		hmmTemp.close()
		treeTemp.close()
		disambigTemp.close()
	