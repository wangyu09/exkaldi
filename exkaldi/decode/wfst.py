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
import copy
import os
import subprocess
import copy
import numpy as np

from exkaldi.version import info as ExkaldiInfo
from exkaldi.version import WrongPath, WrongOperation, WrongDataFormat, KaldiProcessError, UnsupportedType
from exkaldi.utils.utils import run_shell_command, make_dependent_dirs, type_name, check_config, list_files
from exkaldi.utils.utils import FileHandleManager
from exkaldi.utils import declare
from exkaldi.core.archieve import BytesArchieve, Transcription, ListTable, BytesAlignmentTrans, NumpyAlignmentTrans, Metric
from exkaldi.core.common import check_mutiple_resources, run_kaldi_commands_parallel 
from exkaldi.nn.nn import log_softmax
from exkaldi.hmm.hmm import load_hmm
from exkaldi.core.load import load_transcription

class Lattice(BytesArchieve):

	def __init__(self, data=None, symbolTable=None, hmm=None, name="lat"):
		super().__init__(data, name)
		if symbolTable is not None:
			declare.is_list_table("symbolTable", symbolTable)
		if hmm is not None:
			declare.is_hmm("hmm", hmm)
		
		self.__symbolTable = symbolTable
		self.__hmm = hmm
	
	@property
	def symbolTable(self):
		return copy.deepcopy(self.__symbolTable)
	
	@property
	def hmm(self):
		return self.__hmm

	def save(self,fileName):
		'''
		Save lattice as .ali file. 
		
		Args:
			<fileName>: file name.
		''' 
		declare.not_void("lattice", self)
		declare.is_valid_file_name_or_handle("fileName", fileName)

		if isinstance(fileName,str):
			make_dependent_dirs(fileName)
			with open(fileName, "wb") as fw:
				fw.write(self.data)
			return fileName
		else:
			fileName.truncate()
			fileName.write(self.data)
			fileName.seek(0)
			return fileName

	def get_1best(self, symbolTable=None, hmm=None, lmwt=1, acwt=1.0, phoneLevel=False, outFile=None):
		'''
		Get 1 best result with text format.

		Share Args:
			<symbolTable>: None or file path or ListTable object or LexiconBank object.
			<hmm>: None or file path or exkaldi HMM object.
			<phoneLevel>: If Ture, return phone results.

		Parallel Args:
			<lmwt>: language model weight.
			<acwt>: acoustic model weight.
			<outFile>: output file name.

		Return:
			exkaldi Transcription object.
		'''
		declare.is_bool("phoneLevel", phoneLevel)
		declare.kaldi_existed()
		declare.not_void("lattice", self)

		with FileHandleManager() as fhm:
			# check the fotmat of word symbol table
			if symbolTable is None:
				assert self.symbolTable is not None, "<symbolTable> is necessary because no wordSymbol table is avaliable."
				symbolTable = self.symbolTable
			
			if isinstance(symbolTable,str):
				assert os.path.isfile(symbolTable), f"No such file: {symbolTable}."
			elif type_name(symbolTable) == "LexiconBank":
				symbolTableTemp = fhm.create("w+", encoding="utf-8")
				if phoneLevel is True:
					symbolTable.dump_dict("phones", symbolTableTemp, False)
				else:
					symbolTable.dump_dict("words", symbolTableTemp, False)
				symbolTable = symbolTableTemp.name
			elif type_name(symbolTable) == "ListTable":
				symbolTableTemp = fhm.create("w+", encoding="utf-8")
				symbolTable.save(symbolTableTemp)
				symbolTable = symbolTableTemp.name
			else:
				raise UnsupportedType(f"<symbolTable> should be file name, exkaldi LexiconBank or ListTable object but got: {type_name(symbolTable)}.")
			
			if phoneLevel is True:
				# check the format of HMM
				if hmm is None:
					assert self.hmm is not None, "<hmm> is necessary because no HMM model is avaliable."
					hmm = self.hmm

				declare.is_potential_hmm("hmm", hmm)
				if not isinstance(hmm, str):
					hmmTemp = fhm.create("wb+", suffix=".mdl")
					hmm.save(hmmTemp)
					hmm = hmmTemp.name
			else:
				hmm = "placeholder"

			symbolTables,hmms,lmwts,acwts,outFiles = check_mutiple_resources(symbolTable,hmm,lmwt,acwt,outFile=outFile)
			
			if len(outFiles) > 1:
				latTemp = fhm.create("wb+",suffix=".lat")
				self.save(latTemp)
				lat = latTemp.name
			else:
				lat = self
			
			lats = []
			for lmwt, acwt in zip(lmwts, acwts):
				declare.is_positive("lmwt", lmwt)
				declare.is_positive("acwt", acwt)
				lats.append(lat)

			if phoneLevel:
				cmdPattern = 'lattice-align-phones --replace-output-symbols=true {model} ark:{lat} ark:- | '
				cmdPattern += "lattice-best-path --lm-scale={lmwt} --acoustic-scale={acwt} --word-symbol-table={words} ark:- ark,t:{outFile}"
				outputName = '1-best-phone'
			else:
				cmdPattern = "lattice-best-path --lm-scale={lmwt} --acoustic-scale={acwt} --word-symbol-table={words} ark:{lat} ark,t:{outFile}"
				outputName = '1-best-word'

			resources = {"lat":lats,"words":symbolTables,"model":hmms,"lmwt":lmwts,"acwt":acwts,"outFile":outFiles}

			results = run_kaldi_commands_parallel(resources, cmdPattern, analyzeResult=True)

			if len(outFiles) == 1:
				outFile = outFiles[0]
				if outFile == "-":
					outbuffer = results[2].decode().strip().split("\n")
					results = Transcription(name=outputName)
					for line in outbuffer:
						line = line.strip().split(maxsplit=1)
						if len(line) == 0:
							continue
						elif len(line) == 1:
							results[line[0]] = " "
						else:
							results[line[0]] = line[1]
				else:
					results = load_transcription(outFile, name=outputName)
			else:
				for i, fileName in enumerate(outFiles):
					results[i] = load_transcription(fileName, name=outputName)
			
			return results
	
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
		declare.kaldi_existed()
		declare.not_void("lattice", self)

		for x in [acwt, invAcwt, ac2lm, lmwt, lm2ac]:
			declare.is_non_negative("scales", x)
		
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
			return Lattice(data=out,symbolTable=self.symbolTable,hmm=self.hmm,name=newName)

	def add_penalty(self, penalty=0):
		'''
		Add penalty to lattice.

		Args:
			<penalty>: penalty.
		Return:
			An new Lattice object.
		'''
		declare.kaldi_existed()
		declare.not_void("lattice", self)
		declare.is_non_negative("penalty",penalty)
		
		cmd = f"lattice-add-penalty --word-ins-penalty={penalty} ark:- ark:-"

		out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=self.data)

		if cod != 0 or out == b'':
			print(err.decode())
			raise KaldiProcessError("Failed to add penalty.")
		else:
			newName = f"add_penalty({self.name})"
			return Lattice(data=out, symbolTable=self.symbolTable, hmm=self.hmm, name=newName)

	def get_nbest(self, n, symbolTable=None, hmm=None, acwt=1, phoneLevel=False, requireAli=False, requireCost=False):
		'''
		Get N best result with text format.

		Share Args:
			<n>: n best results.
			<symbolTable>: file or ListTable object or LexiconBank object.
			<hmm>: file or HMM object.
			<acwt>: acoustic weight.
			<phoneLevel>: If True, return phone results.
			<requireAli>: If True, return alignment simultaneously.
			<requireCost>: If True, return acoustic model and language model cost simultaneously.
		
		Parallel Args:
			NULL

		Return:
			A list of exkaldi Transcription objects (and their Metric objects).
		'''
		declare.is_positive_int("n", n)
		declare.is_positive("acwt", acwt)
		declare.not_void("lattice", self)

		if symbolTable is None:
			assert self.symbolTable is not None, "<symbolTable> is necessary because no wordSymbol table is avaliable."
			symbolTable = self.symbolTable
		
		with FileHandleManager() as fhm:
			
			if isinstance(symbolTable, str):
				assert os.path.isfile(symbolTable), f"No such file: {symbolTable}."
			elif type_name(symbolTable) == "LexiconBank":
				wordSymbolTemp = fhm.create('w+', suffix=".txt", encoding='utf-8')
				if phoneLevel:
					symbolTable.dump_dict("phones", wordSymbolTemp)
				else:
					symbolTable.dump_dict("words", wordSymbolTemp)
				symbolTable = wordSymbolTemp.name
			elif type_name(symbolTable) == "ListTable":
				wordSymbolTemp = fhm.create('w+', suffix=".txt", encoding='utf-8')
				symbolTable.save(wordSymbolTemp)
				symbolTable = wordSymbolTemp.name
			else:
				raise UnsupportedType(f"<symbolTable> should be file name, LexiconBank object or ListTable object but got: {type_name(symbolTable)}.")
			
			if phoneLevel is True:
				# check the format of HMM
				if hmm is None:
					assert self.hmm is not None, "<hmm> is necessary because no HMM model is avaliable."
					hmm = self.hmm

				declare.is_potential_hmm("hmm", hmm)
				if not isinstance(hmm, str):
					hmmTemp = fhm.create("wb+", suffix=".mdl")
					hmm.save(hmmTemp)
					hmm = hmmTemp.name

				cmd = f'lattice-align-phones --replace-output-symbols=true {hmm} ark:- ark:- | '
			else:
				cmd = ''

			outAliTemp = fhm.create('w+', suffix=".ali", encoding='utf-8')

			cmd += f'lattice-to-nbest --acoustic-scale={acwt} --n={n} ark:- ark:- |'
			cmd += f'nbest-to-linear ark:- ark,t:{outAliTemp.name} ark,t:-'  

			if requireCost:
				outCostFile_LM = fhm.create('w+', suffix=".cost", encoding='utf-8')
				outCostFile_AM = fhm.create('w+', suffix=".cost", encoding='utf-8')				
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

	def determinize(self, acwt=1.0, beam=6):
		'''
		Determinize the lattice.

		Args:
			<acwt>: acoustic scale.
			<beam>: prune beam.
		Return:
			An new Lattice object.
		'''
		declare.kaldi_existed()
		declare.is_positive("acwt", acwt)
		declare.is_positive_int("beam", beam)
		declare.not_void("lattice", self)
		
		cmd = f"lattice-determinize-pruned --acoustic-scale={acwt} --beam={beam} ark:- ark:-"

		out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=self.data)

		if cod != 0 or out == b'':
			print(err.decode())
			raise KaldiProcessError("Failed to determinize lattice.")
		else:
			newName = f"determinize({self.name})"
			return Lattice(data=out, symbolTable=self.symbolTable, hmm=self.hmm, name=newName)		

	def am_rescore(self, hmm, feat):
		'''
		Replace the acoustic scores with new HMM-GMM model.

		Args:
			<hmm>: exkaldi HMM object or file path.
			<feat>: exkaldi feature object or index table object.

		Return:
			a new Lattice object.
		'''
		declare.kaldi_existed()
		declare.not_void("lattice", self)
		declare.is_potential_hmm("hmm", hmm)
		declare.is_feature("feat", feat)
		
		with FileHandleManager() as fhm:

			if not isinstance(hmm, str):
				hmmTemp = fhm.create("wb+", suffix=".mdl")
				hmm.save(hmmTemp)
				hmm = hmmTemp.name
			
			if type_name(feat) == "ArkIndexTable":
				featTemp = fhm.create("w+", suffix=".scp", encoding="utf-8")
				feat.save(featTemp)
				featRepe = f"scp:{featTemp.name}"
			elif type_name(feat) == "BytesFeature":
				feat = feat.sort(by="utt")
				featTemp = fhm.create("wb+", suffix=".ark")
				feat.save(featTemp)
				featRepe = f"ark:{featTemp.name}"
			else:
				feat = feat.sort(by="utt").to_bytes()
				featTemp = fhm.create("wb+", suffix=".ark")
				feat.save(featTemp)
				featRepe = f"ark:{featTemp.name}"

			cmd = f"gmm-rescore-lattice	{hmm} ark:- {featRepe} ark:-"

			out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=self.data)

			if cod != 0 or out == b'':
				print(err.decode())
				raise KaldiProcessError("Failed to determinize lattice.")
			else:
				newName = f"am_rescore({self.name})"
				return Lattice(data=out, symbolTable=self.symbolTable, hmm=self.hmm, name=newName)

	def __add__(self, other):
		'''
		Sum two lattices to one.
		'''
		declare.is_classes("other", other, Lattice)
		name = f"plus({self.name},{other.name})"
		return Lattice( b"".join([self.data,other.data]), hmm=self.hmm, symbolTable=self.symbolTable, name=name)

def load_lat(target, name="lat"):
	'''
	Load lattice data.

	Args:
		<target>: bytes object, file path or exkaldi lattice object.
		<name>: a string.
	Return:
		An exkaldi lattice object.
	'''
	declare.is_valid_string("name", name)

	if isinstance(target, bytes):
		return Lattice(target, name=name)
	
	elif isinstance(target, Lattice):
		return Lattice(target.data, name=name)

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

def nn_decode(prob, hmm, HCLGFile, symbolTable, beam=10, latBeam=8, acwt=1,
				minActive=200, maxActive=7000, maxMem=50000000, config=None, maxThreads=1, outFile=None):
	'''
	Decode by generating lattice from acoustic probability output by NN model.

	Share Args:
		NULL
	
	Parallel Args:
		<prob>: An exkaldi probability object. We expect the probability didn't pass any activation function, or it may generate wrong results.
		<hmm>: file path or exkaldi HMM object.
		<HCLGFile>: HCLG graph file:
		<symbolTable>: words.txt file path or exkaldi LexiconBank or ListTable object.
		<beam>.
		<latBeam>.
		<acwt>.
		<minActive>.
		<maxActive>.
		<maxMem>.
		<config>.
		<maxThreads>.
		<outFile>.

		Some usual options can be assigned directly. If you want use more, set <config> = your-configure.
		You can use .check_config('nn_decode') function to get configure information you could set.
		Also run shell command "latgen-faster-mapped" to look their meaning.
	
	Return:
		exkaldi Lattice object.
	'''
	declare.kaldi_existed()

	parameters = check_mutiple_resources( prob, hmm, HCLGFile, symbolTable, 
										  beam, latBeam, acwt, minActive, maxActive, maxMem,
										  config, maxThreads, outFile=outFile,
										)

	with FileHandleManager() as fhm:

		baseCmds = []
		outFiles = parameters[-1]

		for prob,hmm,HCLGFile,symbolTable,beam,latBeam,acwt,minActive,maxActive,maxMem,config,maxThreads in zip(*parameters[:-1]):
			# check probability
			declare.is_probability("prob", prob)
			# check hmm
			declare.is_potential_hmm("hmm", hmm)
			# check HCLGFile
			declare.is_file("HCLGFile", HCLGFile)
			# check symbolTable
			if isinstance(symbolTable,str):
				assert os.path.isfile(hmm), f"No such file: {hmm}."
			elif type_name(symbolTable) == "LexiconBank":
				wordsTemp = fhm.create("w+", suffix=".words", encoding="utf-8")
				symbolTable.dump_dict("words", wordsTemp)
				symbolTable = wordsTemp.name
			elif type_name(symbolTable) == "ListTable":
				wordsTemp = fhm.create("w+", suffix=".words", encoding="utf-8")
				symbolTable.save(wordsTemp)
				symbolTable = wordsTemp.name
			else:
				raise UnsupportedType(f"<symbolTable> should be file name, LexiconBank or ListTable object but got: {symbolTable}.")
			# check other parameters
			declare.is_positive_int("maxThreads", maxThreads)
			# build the base command
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
			kaldiTool += f'--word-symbol-table={symbolTable} '
			if config is not None:
				if check_config(name='nn_decode', config=config):
					for key,value in config.items():
						if isinstance(value, bool):
							if value is True:
								kaldiTool += f"{key} "
						else:
							kaldiTool += f" {key}={value}"
			baseCmds.append( kaldiTool )
			
		# define command pattern
		cmdPattern = '{kaldiTool} {hmm} {HCLG} {prob} ark:{outFile}'
		# define resources
		resources = {"prob":parameters[0], "hmm":parameters[1], "HCLG":parameters[2], "kaldiTool":baseCmds, "outFile":outFiles}
		# run
		results = run_kaldi_commands_parallel(resources, cmdPattern)
		if len(outFiles) == 1:
			outFile = outFiles[0]
			newName = f"lat({parameters[0][0].name})"
			if outFile == "-":
				results = Lattice(data=results[2], name=newName)
			else:
				results = load_lat(outFile, name=newName)
		else:
			for i, fileName in enumerate(outFiles):
				newName = f"lat({parameters[0][i].name})"
				results[i] = load_lat(fileName, name=newName)
			
		return results

def gmm_decode(feat, hmm, HCLGFile, symbolTable, beam=10, latBeam=8, acwt=1,
				minActive=200, maxActive=7000, maxMem=50000000, config=None, maxThreads=1, outFile=None):
	'''
	Decode by generating lattice from acoustic probability output by NN model.

	Share Args:
		NULL
	
	Parallel Args:
		<feat>: An exkaldi feature or index table object.
		<hmm>: file path or exkaldi HMM object.
		<HCLGFile>: HCLG graph file:
		<symbolTable>: words.txt file path or exkaldi LexiconBank or ListTable object.
		<beam>.
		<latBeam>.
		<acwt>.
		<minActive>.
		<maxActive>.
		<maxMem>.
		<config>.
		<maxThreads>.
		<outFile>.

		Some usual options can be assigned directly. If you want use more, set <config> = your-configure.
		You can use .check_config('gmm_decode') function to get configure information you could set.
		Also run shell command "latgen-faster-mapped" to look their meaning.
	
	Return:
		exkaldi Lattice object.
	'''
	declare.kaldi_existed()

	parameters = check_mutiple_resources( feat, hmm, HCLGFile, symbolTable, 
										  beam, latBeam, acwt, minActive, maxActive, maxMem,
										  config, maxThreads, outFile=outFile,
										)

	with FileHandleManager() as fhm:

		baseCmds = []
		outFiles = parameters[-1]

		for feat,hmm,HCLGFile,symbolTable,beam,latBeam,acwt,minActive,maxActive,maxMem,config,maxThreads in zip(*parameters[:-1]):
			# check probability
			declare.is_feature("feat", feat)
			# check hmm
			declare.is_potential_hmm("hmm", hmm)
			# check HCLGFile
			declare.is_file("HCLGFile", HCLGFile)
			# check symbolTable
			if isinstance(symbolTable,str):
				assert os.path.isfile(hmm), f"No such file: {hmm}."
			elif type_name(symbolTable) == "LexiconBank":
				wordsTemp = fhm.create("w+", suffix=".words", encoding="utf-8")
				symbolTable.dump_dict("words", wordsTemp)
				symbolTable = wordsTemp.name
			elif type_name(symbolTable) == "ListTable":
				wordsTemp = fhm.create("w+", suffix=".words", encoding="utf-8")
				symbolTable.save(wordsTemp)
				symbolTable = wordsTemp.name
			else:
				raise UnsupportedType(f"<symbolTable> should be file name, LexiconBank or ListTable object but got: {symbolTable}.")
			# check other parameters
			declare.is_positive_int("maxThreads", maxThreads)
			# build the base command
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
			kaldiTool += f'--word-symbol-table={symbolTable} '
			if config is not None:
				if check_config(name='gmm_decode', config=config):
					for key,value in config.items():
						if isinstance(value, bool):
							if value is True:
								kaldiTool += f"{key} "
						else:
							kaldiTool += f" {key}={value}"
			baseCmds.append( kaldiTool )
			
		# define command pattern
		cmdPattern = '{kaldiTool} {hmm} {HCLG} {feat} ark:{outFile}'
		# define resources
		resources = {"feat":parameters[0], "hmm":parameters[1], "HCLG":parameters[2], "kaldiTool":baseCmds, "outFile":outFiles}
		# run
		results = run_kaldi_commands_parallel(resources, cmdPattern)

		if len(outFiles) == 1:
			outFile = outFiles[0]
			newName = f"lat({parameters[0][0].name})"
			if outFile == "-":
				results = Lattice(data=results[2], name=newName)
			else:
				results = load_lat(outFile, name=newName)
		else:
			for i, fileName in enumerate(outFiles):
				newName = f"lat({parameters[0][i].name})"
				results[i] = load_lat(fileName, name=newName)
			
		return results

def compile_align_graph(hmm, tree, transcription, LFile, outFile, lexicons=None):
	'''
	Compile graph for training or aligning.

	Share Args:
		<hmm>: file name or exkaldi HMM object.
		<tree>: file name or exkaldi decision tree object.
		<lexicons>: exkaldi lexicon bank object.
		<LFile>: file name.
	
	Parallel Args:
		<transcription>, file path or exkaldi trancription object.
		<outFile>, output file name.

	Return:
		output file name.
	'''
	declare.is_potential_hmm("hmm", hmm)
	declare.is_potential_tree("tree", tree)

	if isinstance(tree, str):
		treeLexicons = None
	else:
		treeLexicons = tree.lex

	if isinstance(hmm, str):
		if lexicons is not None:
			declare.is_lexicon_bank("lexicons", lexicons)
		elif treeLexicons is not None:
			lexicons = treeLexicons
		else:
			raise WrongOperation("<lexicons> is necessary on this case.")
		hmm = load_hmm(hmm, lexicons=lexicons)

	return hmm.compile_train_graph(tree, transcription, LFile, outFile)

def nn_align(hmm, prob, alignGraphFile=None, tree=None, transcription=None, Lfile=None, transitionScale=1.0, acousticScale=0.1, 
				selfloopScale=0.1, beam=10, retryBeam=40, lexicons=None, name="ali", outFile=None):
	'''
	Align the neural network acoustic output probability.

	Share Args:
		<hmm>: file name or exkaldi HMM object.
		<tree>: file name or exkaldi decision tree object.
		<Lfile>: file name.
		<lexicons>: exkaldi LexiconBank object.
	
	Parallel Args:
		<prob>: exkaldi probability object or index table object.
		<alignGraphFile>: file name.
		<transcription>: file name or exkaldi transcription object.
		<transitionScale>.
		<acousticScale>.
		<selfloopScale>.
		<beam>.
		<retryBeam>.
		<name>: string.
		<outFile>: file name.

	Return:
		exkaldi alignment object or index table object.
	'''
	declare.kaldi_existed()
	declare.is_potential_hmm("hmm", hmm)

	with FileHandleManager() as fhm:

		if isinstance(hmm,str):
			hmmLexicons = None
		else:
			hmmTemp = fhm.create("wb+",suffix=".mdl")
			hmm.save(hmmTemp)
			hmmLexicons = hmm.lex
			hmm = hmmTemp.name
		
		if alignGraphFile is None:
			assert None not in [tree, transcription, Lfile], "When align graph is not provided, all of <tree>, <transcription> and <Lfile> is necessary."
			declare.is_file("Lfile", Lfile)
			declare.is_potential_tree("tree", tree)

			if isinstance(tree,str):
				treeLexicons = None
			else:
				treeTemp = fhm.create("wb+",suffix=".tree")
				tree.save(treeTemp)
				treeLexicons = tree.lex
				tree = treeTemp.name
			
			if lexicons is None:
				if hmmLexicons is None:
					if treeLexicons is None:
						raise WrongOperation("<lexicons> is necessary on this case.")
					else:
						lexicons = treeLexicons
				else:
					lexicons = hmmLexicons
			else:
				declare.is_lexicon_bank("lexicons", lexicons)

			disambigTemp = fhm.create("w+", suffix="_disambig.int", encoding="utf-8")
			lexicons.dump_dict(name="disambig", outFile=disambigTemp, dumpInt=True)

			parameters = check_mutiple_resources(prob, transcription, transitionScale, acousticScale, selfloopScale, beam, retryBeam, outFile=outFile)
			baseCmds = []
			for prob, transcription, transitionScale, acousticScale, selfloopScale, beam, retryBeam, _ in zip(*parameters):
				declare.is_probability("prob", prob)
				declare.is_potential_transcription("transcription", transcription)

				cmd = f"align-mapped --transition-scale={transitionScale} --acoustic-scale={acousticScale} --self-loop-scale={selfloopScale} "
				cmd += f"--beam={beam} --retry-beam={retryBeam} "
				cmd += f"--read-disambig-syms={disambigTemp.name} "
				cmd += f"{tree} {hmm} {Lfile} "
				baseCmds.append(cmd)
			
			cmdPattern = "{baseTool} {prob} ark:{trans} ark:{outFile}"
			resources = {"prob":parameters[0], "baseTool":baseCmds, "trans":parameters[1], "outFile":parameters[-1]}

		else:
			assert tree is None and transcription is None and Lfile is None, "When use compiled align graph, any of <tree>, <transcription> and <Lfile> is invalid."
			
			parameters = check_mutiple_resources(prob, alignGraphFile, transitionScale, acousticScale, selfloopScale, beam, retryBeam, outFile=outFile)

			baseCmds = []
			for prob, alignGraphFile, transitionScale, acousticScale, selfloopScale, beam, retryBeam, _ in zip(*parameters):
				declare.is_probability("prob", prob)
				declare.is_file("alignGraphFile", alignGraphFile)
			
				cmd = f"align-compiled-mapped --transition-scale={transitionScale} --acoustic-scale={acousticScale} --self-loop-scale={selfloopScale} "
				cmd += f"--beam={beam} --retry-beam={retryBeam} "
				cmd += f"{hmm} ark:{alignGraphFile}"
				baseCmds.append(cmd)
			
			cmdPattern = "{baseTool} {prob} ark:{outFile}"
			resources = {"prob":parameters[0], "baseTool":baseCmds, "outFile":parameters[-1]}			

		# run
		results = run_kaldi_commands_parallel(resources, cmdPattern)
		# analyze result
		if len(outFiles) == 1:
			outFile = outFiles[0]
			if outFile == "-":
				results = BytesAlignmentTrans(results[2], name=names[0])
			else:
				results = load_index_table(outFile, name=names[0])
		else:
			for i, fileName in enumerate(outFiles):
				results = load_index_table(fileName, name=names[i])
		
		return results

def gmm_align(hmm, feat, alignGraphFile=None, tree=None, transcription=None, Lfile=None, transitionScale=1.0, acousticScale=0.1, 
				selfloopScale=0.1, beam=10, retryBeam=40, boostSilence=1.0, careful=False, name="ali", lexicons=None, outFile=None):
	'''
	Align the feature.

	Share Args:
		<hmm>: file name or exkaldi HMM object.
		<tree>: file name or exkaldi decision tree object.
		<Lfile>: file name.
		<lexicons>: exkaldi LexiconBank object.
		<_boostSilence>.
		<careful>.
	
	Parallel Args:
		<feat>: exkaldi feature object or index table object.
		<alignGraphFile>: file name.
		<transcription>: file name or exkaldi transcription object.
		<transitionScale>.
		<acousticScale>.
		<selfloopScale>.
		<beam>.
		<retryBeam>.
		<name>: string.
		<outFile>: file name.

	Return:
		exkaldi alignment object or index table object.
	'''
	declare.kaldi_existed()
	declare.is_potential_tree("tree", tree)

	with FileHandleManager() as fhm:

		if isinstance(tree,str):
			treeLexicons = None
		else:
			treeTemp = fhm.create("wb+",suffix=".tree")
			tree.save(treeTemp)
			treeLexicons = tree.lex
			tree = treeTemp.name

		hmmTemp = fhm.create("wb+",suffix=".mdl")
		declare.is_potential_hmm("hmm", hmm)

		if isinstance(hmm,str):
			if lexicons is None:
				if treeLexicons is None:
					raise WrongOperation("<lexicons> is necessary on this case.")
				else:
					lexicons = treeLexicons
			else:
				declare.is_lexicon_bank("lexicons", lexicons)
			optionSilence = ":".join(lexicons("optional_silence", True))
			cmd = f'gmm-boost-silence --boost={boostSilence} {optionSilence} {hmm} {hmmTemp.name}'
			out, err, cod = run_shell_command(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)			

		else:
			if lexicons is None:
				if hmm.lex is None:
					if treeLexicons is None:
						raise WrongOperation("<lexicons> is necessary on this case.")
					else:
						lexicons = treeLexicons
				else:
					lexicons = hmm.lex
			else:
				declare.is_lexicon_bank("lexicons", lexicons)
			optionSilence = ":".join(lexicons("optional_silence", True))
			cmd = f'gmm-boost-silence --boost={boostSilence} {optionSilence} - {hmmTemp.name}'
			out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=hmm.data)
		
		if (isinstance(cod,int) and cod != 0 ) or os.path.getsize(hmmTemp.name) == 0:
			print(err.decode())
			raise KaldiProcessError("Generate new HMM defeated.")
		hmmTemp.seek(0)
		
		if alignGraphFile is None:
			assert None not in [tree,transcription,Lfile], "When align graph is not provided, all of <tree>, <transcription> and <Lfile> is necessary."
			declare.is_file("Lfile", Lfile)

			disambigTemp = fhm.create("w+", suffix="_disambig.int", encoding="utf-8")
			lexicons.dump_dict(name="disambig", outFile=disambigTemp, dumpInt=True)

			parameters = check_mutiple_resources(feat, transcription, transitionScale, acousticScale, selfloopScale, beam, retryBeam, careful, outFile=outFile)
			baseCmds = []
			for feat, transcription, transitionScale, acousticScale, selfloopScale, beam, retryBeam, careful, _ in zip(*parameters):
				declare.is_feature("feat", feat)
				declare.is_potential_transcription("transcription", transcription)

				cmd = f"gmm-align --transition-scale={transitionScale} --acoustic-scale={acousticScale} --self-loop-scale={selfloopScale} "
				cmd += f"--beam={beam} --retry-beam={retryBeam} --careful={careful} "
				cmd += f"--read-disambig-syms={disambigTemp.name} "
				cmd += f"{tree} {hmmTemp.name} {Lfile} "
				baseCmds.append(cmd)
			
			cmdPattern = "{baseTool} {feat} ark:{trans} ark:{outFile}"
			resources = {"feat":parameters[0], "baseTool":baseCmds, "trans":parameters[1], "outFile":parameters[-1]}

		else:
			assert tree is None and transcription is None and Lfile is None, "When use compiled align graph, any of <tree>, <transcription> and <Lfile> is invalid."
			
			parameters = check_mutiple_resources(prob, alignGraphFile, transitionScale, acousticScale, selfloopScale, beam, retryBeam, careful, outFile=outFile)

			baseCmds = []
			for feat, alignGraphFile, transitionScale, acousticScale, selfloopScale, beam, retryBeam, careful, _ in zip(*parameters):
				declare.is_feature("feat", feat)
				declare.is_file("alignGraphFile", alignGraphFile)
				
				cmd = f"gmm-align-compiled --transition-scale={transitionScale} --acoustic-scale={acousticScale} --self-loop-scale={selfloopScale} "
				cmd += f"--beam={beam} --retry-beam={retryBeam}  --careful={careful} "
				cmd += f"{hmm} ark:{alignGraphFile}"
				baseCmds.append(cmd)
			
			cmdPattern = "{baseTool} {feat} ark:{outFile}"
			resources = {"feat":parameters[0], "baseTool":baseCmds, "outFile":parameters[-1]}			

		# run
		results = run_kaldi_commands_parallel(resources, cmdPattern)
		# analyze result
		if len(outFiles) == 1:
			outFile = outFiles[0]
			if outFile == "-":
				results = BytesAlignmentTrans(results[2], name=names[0])
			else:
				results = load_index_table(outFile, name=names[0])
		else:
			for i, fileName in enumerate(outFiles):
				results = load_index_table(fileName, name=names[i])
		
		return results