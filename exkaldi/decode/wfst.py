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

"""Exkaldi Decoding associates """
import copy
import os
import numpy as np

from exkaldi.version import info as ExKaldiInfo
from exkaldi.error import *
from exkaldi.utils.utils import run_shell_command,make_dependent_dirs,type_name,check_config,list_files
from exkaldi.utils.utils import FileHandleManager
from exkaldi.utils import declare
from exkaldi.core.archive import BytesArchive,Transcription,ListTable,BytesAliTrans,NumpyAliTrans,Metric
from exkaldi.core.common import check_multiple_resources,run_kaldi_commands_parallel 
from exkaldi.nn.nn import log_softmax
from exkaldi.hmm.hmm import load_hmm
from exkaldi.core.load import load_transcription

class Lattice(BytesArchive):
	'''
	A class to hold binary lattice data in memory.
	'''
	def __init__(self,data=None,symbolTable=None,hmm=None,name="lat"):
		'''
		Args:
			<data>: bytes object.
			<symbolTable>: an exkaldi ListTable object.
			<hmm>: an exkaldi HMM object.
			<name>: a string.
		'''
		super().__init__(data,name)
		if symbolTable is not None:
			declare.is_list_table("symbolTable",symbolTable)
		if hmm is not None:
			declare.is_hmm("hmm",hmm)
		
		self.__symbolTable = symbolTable
		self.__hmm = hmm
	
	@property
	def symbolTable(self):
		'''
		Get the symbol table.

		Return:
			None or a ListTable object.
		'''
		return copy.deepcopy(self.__symbolTable)
	
	@property
	def hmm(self):
		'''
		Get the HMM.

		Return:
			None or an exkaldi HMM object.
		'''
		return self.__hmm

	def save(self,fileName):
		'''
		Save lattice to file. 
		
		Args:
			<fileName>: file name or file handle.
		
		Return:
			file name or file handle.
		''' 
		declare.not_void(type_name(self),self)
		declare.is_valid_file_name_or_handle("fileName",fileName)

		if isinstance(fileName,str):
			make_dependent_dirs(fileName)
			with open(fileName,"wb") as fw:
				fw.write(self.data)
			return fileName

		else:
			fileName.truncate()
			fileName.write(self.data)
			fileName.seek(0)
			return fileName

	def get_1best(self,symbolTable=None,hmm=None,lmwt=1,acwt=1.0,phoneLevel=False,outFile=None):
		'''
		Get 1 best result with text format.

		Share Args:
			<symbolTable>: None or file path or ListTable object or LexiconBank object.
			<hmm>: None or file path or exkaldi HMM object.
			<phoneLevel>: If Ture,return phone results.

		Parallel Args:
			<lmwt>: language model weight.
			<acwt>: acoustic model weight.
			<outFile>: output file name.

		Return:
			exkaldi Transcription object.
		'''
		declare.is_bool("phoneLevel",phoneLevel)
		declare.kaldi_existed()
		declare.not_void(type_name(self),self)

		with FileHandleManager() as fhm:
			# check the format of word symbol table
			if symbolTable is None:
				assert self.symbolTable is not None,"<symbolTable> is necessary because no word symbol table is avaliable."
				symbolTable = self.symbolTable
			
			if isinstance(symbolTable,str):
				assert os.path.isfile(symbolTable),f"No such file: {symbolTable}."
			elif type_name(symbolTable) == "LexiconBank":
				symbolTableTemp = fhm.create("w+",encoding="utf-8")
				if phoneLevel is True:
					symbolTable.dump_dict("phones",symbolTableTemp,False)
				else:
					symbolTable.dump_dict("words",symbolTableTemp,False)
				symbolTable = symbolTableTemp.name
			elif type_name(symbolTable) == "ListTable":
				symbolTableTemp = fhm.create("w+",encoding="utf-8")
				symbolTable.save(symbolTableTemp)
				symbolTable = symbolTableTemp.name
			else:
				raise UnsupportedType(f"<symbolTable> should be file name,exkaldi LexiconBank or ListTable object but got: {type_name(symbolTable)}.")
			
			if phoneLevel is True:
				# check the format of HMM
				if hmm is None:
					assert self.hmm is not None,"<hmm> is necessary because no HMM model is avaliable."
					hmm = self.hmm
				declare.is_potential_hmm("hmm",hmm)
				if not isinstance(hmm,str):
					hmmTemp = fhm.create("wb+",suffix=".mdl")
					hmm.save(hmmTemp)
					hmm = hmmTemp.name
			else:
				hmm = "placeholder"

			symbolTables,hmms,lmwts,acwts,outFiles = check_multiple_resources(symbolTable,hmm,lmwt,acwt,outFile=outFile)
			
			if len(outFiles) > 1:
				latTemp = fhm.create("wb+",suffix=".lat")
				self.save(latTemp)
				lat = latTemp.name
			else:
				lat = self
			
			lats = []
			for lmwt,acwt in zip(lmwts,acwts):
				declare.is_positive("lmwt",lmwt)
				declare.is_positive("acwt",acwt)
				lats.append(lat)

			if phoneLevel:
				cmdPattern = 'lattice-align-phones --replace-output-symbols=true {model} ark:{lat} ark:- | '
				cmdPattern += "lattice-best-path --lm-scale={lmwt} --acoustic-scale={acwt} --word-symbol-table={words} ark:- ark,t:{outFile}"
				outputName = '1-best-phone'
			else:
				cmdPattern = "lattice-best-path --lm-scale={lmwt} --acoustic-scale={acwt} --word-symbol-table={words} ark:{lat} ark,t:{outFile}"
				outputName = '1-best-word'

			resources = {"lat":lats,"words":symbolTables,"model":hmms,"lmwt":lmwts,"acwt":acwts,"outFile":outFiles}

			results = run_kaldi_commands_parallel(resources,cmdPattern,analyzeResult=True)

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
					results = load_transcription(outFile,name=outputName)
			else:
				for i,fileName in enumerate(outFiles):
					results[i] = load_transcription(fileName,name=outputName)
			
			return results
	
	def scale(self,acwt=1,invAcwt=1,ac2lm=0,lmwt=1,lm2ac=0):
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
		declare.not_void(type_name(self),self)

		for x in [acwt,invAcwt,ac2lm,lmwt,lm2ac]:
			declare.is_non_negative("scales",x)
		
		cmd = 'lattice-scale'
		cmd += ' --acoustic-scale={}'.format(acwt)
		cmd += ' --acoustic2lm-scale={}'.format(ac2lm)
		cmd += ' --inv-acoustic-scale={}'.format(invAcwt)
		cmd += ' --lm-scale={}'.format(lmwt)
		cmd += ' --lm2acoustic-scale={}'.format(lm2ac)
		cmd += ' ark:- ark:-'

		out,err,cod = run_shell_command(cmd,stdin="PIPE",stdout="PIPE",stderr="PIPE",inputs=self.data)

		if cod != 0 or out == b'':
			raise KaldiProcessError("Failed to scale lattice.",err.decode())
		else:
			newName = f"scale({self.name})"
			return Lattice(data=out,symbolTable=self.symbolTable,hmm=self.hmm,name=newName)

	def add_penalty(self,penalty=0):
		'''
		Add penalty to lattice.

		Args:
			<penalty>: word insertion penalty.

		Return:
			An new Lattice object.
		'''
		declare.kaldi_existed()
		declare.not_void(type_name(self),self)
		declare.is_non_negative("penalty",penalty)
		
		cmd = f"lattice-add-penalty --word-ins-penalty={penalty} ark:- ark:-"

		out,err,cod = run_shell_command(cmd,stdin="PIPE",stdout="PIPE",stderr="PIPE",inputs=self.data)

		if cod != 0 or out == b'':
			raise KaldiProcessError("Failed to add penalty.",err.decode())
		else:
			newName = f"add_penalty({self.name})"
			return Lattice(data=out,symbolTable=self.symbolTable,hmm=self.hmm,name=newName)

	def get_nbest(self,n,symbolTable=None,hmm=None,acwt=1,phoneLevel=False,requireAli=False,requireCost=False):
		'''
		Get N best result with text format.

		Args:
			<n>: n best results.
			<symbolTable>: file or ListTable object or LexiconBank object.
			<hmm>: file or HMM object.
			<acwt>: acoustic weight.
			<phoneLevel>: If True,return phone results.
			<requireAli>: If True,return alignment simultaneously.
			<requireCost>: If True,return acoustic model and language model cost simultaneously.

		Return:
			A list of exkaldi Transcription objects (and their Metric objects).
		'''
		declare.is_positive_int("n",n)
		declare.is_positive("acwt",acwt)
		declare.not_void("lattice",self)

		if symbolTable is None:
			assert self.symbolTable is not None,"<symbolTable> is necessary because no wordSymbol table is avaliable."
			symbolTable = self.symbolTable
		
		with FileHandleManager() as fhm:
			
			if isinstance(symbolTable,str):
				assert os.path.isfile(symbolTable),f"No such file: {symbolTable}."
			elif type_name(symbolTable) == "LexiconBank":
				wordSymbolTemp = fhm.create('w+',suffix=".txt",encoding='utf-8')
				if phoneLevel:
					symbolTable.dump_dict("phones",wordSymbolTemp)
				else:
					symbolTable.dump_dict("words",wordSymbolTemp)
				symbolTable = wordSymbolTemp.name
			elif type_name(symbolTable) == "ListTable":
				wordSymbolTemp = fhm.create('w+',suffix=".txt",encoding='utf-8')
				symbolTable.save(wordSymbolTemp)
				symbolTable = wordSymbolTemp.name
			else:
				raise UnsupportedType(f"<symbolTable> should be file name,LexiconBank object or ListTable object but got: {type_name(symbolTable)}.")
			
			if phoneLevel is True:
				# check the format of HMM
				if hmm is None:
					assert self.hmm is not None,"<hmm> is necessary because no HMM model is avaliable."
					hmm = self.hmm

				declare.is_potential_hmm("hmm",hmm)
				if not isinstance(hmm,str):
					hmmTemp = fhm.create("wb+",suffix=".mdl")
					hmm.save(hmmTemp)
					hmm = hmmTemp.name

				cmd = f'lattice-align-phones --replace-output-symbols=true {hmm} ark:- ark:- | '
			else:
				cmd = ''

			outAliTemp = fhm.create('w+',suffix=".ali",encoding='utf-8')

			cmd += f'lattice-to-nbest --acoustic-scale={acwt} --n={n} ark:- ark:- |'
			cmd += f'nbest-to-linear ark:- ark,t:{outAliTemp.name} ark,t:-'  

			if requireCost:
				outCostFile_LM = fhm.create('w+',suffix=".cost",encoding='utf-8')
				outCostFile_AM = fhm.create('w+',suffix=".cost",encoding='utf-8')				
				cmd += f' ark,t:{outCostFile_LM.name} ark,t:{outCostFile_AM.name}'

			out,err,cod = run_shell_command(cmd,stdin="PIPE",stdout="PIPE",stderr="PIPE",inputs=self.data)
			
			if cod != 0 or out == b'':
				raise KaldiProcessError('Failed to get N best results.',err.decode())
			
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
				for uttID,nbests in results.items():
					for index,one in enumerate(nbests):
						if index > len(finalResults)-1:
							finalResults.append({})
						finalResults[index][uttID] = one

				return finalResults

			out = out.decode().strip().split("\n")

			out = sperate_n_bests(out)
			NBEST = []
			for index,one in enumerate(out,start=1):
				name = f"{index}-best"
				NBEST.append( Transcription(one,name=name) )
			del out

			if requireCost:
				outCostFile_AM.seek(0)
				lines_AM = outCostFile_AM.read().strip().split("\n")
				lines_AM = sperate_n_bests(lines_AM)
				AMSCORE = []
				for index,one in enumerate(lines_AM,start=1):
					name = f"AM-{index}-best"
					for key,value in one.items():
						one[key] = float(value)
					AMSCORE.append( Metric(one,name=name) )
				del lines_AM			

				outCostFile_LM.seek(0)
				lines_LM = outCostFile_LM.read().strip().split("\n")
				lines_LM = sperate_n_bests(lines_LM)
				LMSCORE = []
				for index,one in enumerate(lines_LM,start=1):
					name = f"LM-{index}-best"
					for key,value in one.items():
						one[key] = float(value)
					LMSCORE.append( Metric(one,name=name) )
				del lines_LM

				finalResult = [NBEST,AMSCORE,LMSCORE]
			else:
				finalResult = [NBEST,]

			if requireAli:
				ALIGNMENT = []
				outAliTemp.seek(0)
				ali = outAliTemp.read().strip().split("\n")
				ali = sperate_n_bests(ali)
				for index,one in enumerate(ali,start=1):
					name = f"{index}-best"
					temp = {}
					for key,value in one.items():
						value = value.strip().split()
						temp[key] = np.array(value,dtype=np.int32)
					ALIGNMENT.append( NumpyAliTrans(temp,name=name) )
				del ali
				finalResult.append(ALIGNMENT)

			if len(finalResult) == 1:
				finalResult = finalResult[0]

			return finalResult

	def determinize(self,acwt=1.0,beam=6):
		'''
		Determinize the lattice.

		Args:
			<acwt>: acoustic scale.
			<beam>: prune beam.

		Return:
			An new Lattice object.
		'''
		declare.kaldi_existed()
		declare.is_positive("acwt",acwt)
		declare.is_positive_int("beam",beam)
		declare.not_void(type_name(self),self)
		
		cmd = f"lattice-determinize-pruned --acoustic-scale={acwt} --beam={beam} ark:- ark:-"

		out,err,cod = run_shell_command(cmd,stdin="PIPE",stdout="PIPE",stderr="PIPE",inputs=self.data)

		if cod != 0 or out == b'':
			raise KaldiProcessError("Failed to determinize lattice.",err.decode())
		else:
			newName = f"determinize({self.name})"
			return Lattice(data=out,symbolTable=self.symbolTable,hmm=self.hmm,name=newName)		

	def am_rescore(self,hmm,feat):
		'''
		Replace the acoustic scores with new HMM-GMM model.

		Args:
			<hmm>: exkaldi HMM object or file path.
			<feat>: exkaldi feature object or index table object.

		Return:
			a new Lattice object.
		'''
		declare.kaldi_existed()
		declare.not_void("lattice",self)
		declare.is_potential_hmm("hmm",hmm)
		declare.is_feature("feat",feat)
		
		with FileHandleManager() as fhm:

			if not isinstance(hmm,str):
				hmmTemp = fhm.create("wb+",suffix=".mdl")
				hmm.save(hmmTemp)
				hmm = hmmTemp.name
			
			if type_name(feat) == "IndexTable":
				featTemp = fhm.create("w+",suffix=".scp",encoding="utf-8")
				feat.save(featTemp)
				featRepe = f"scp:{featTemp.name}"
			elif type_name(feat) == "BytesFeat":
				feat = feat.sort(by="utt")
				featTemp = fhm.create("wb+",suffix=".ark")
				feat.save(featTemp)
				featRepe = f"ark:{featTemp.name}"
			else:
				feat = feat.sort(by="utt").to_bytes()
				featTemp = fhm.create("wb+",suffix=".ark")
				feat.save(featTemp)
				featRepe = f"ark:{featTemp.name}"

			cmd = f"gmm-rescore-lattice	{hmm} ark:- {featRepe} ark:-"

			out,err,cod = run_shell_command(cmd,stdin="PIPE",stdout="PIPE",stderr="PIPE",inputs=self.data)

			if cod != 0 or out == b'':
				raise KaldiProcessError("Failed to determinize lattice.",err.decode())
			else:
				newName = f"am_rescore({self.name})"
				return Lattice(data=out,symbolTable=self.symbolTable,hmm=self.hmm,name=newName)

	def __add__(self,other):
		'''
		Sum two lattices to one.
		We won't check the data. So error might occur when you plus another lattice with different format.

		Args:
			<other>: another lattice object.
		'''
		declare.is_classes("other",other,Lattice)
		name = f"plus({self.name},{other.name})"
		return Lattice( b"".join([self.data,other.data]),hmm=self.hmm,symbolTable=self.symbolTable,name=name)

	def write_ctm(self,lmwt,acwt,outFile,hmm=None,phoneLevel=False,config=None):
		'''
		Generate result with CTM format.

		Share Args:
			<hmm>: file path or HMM object.
			<phoneLevel>: A bool value. If True,generate phone level result.

		Parallel Args:
			<lmwt>: a float value.
			<acwt>: a float value.
			<config>: extra arguments.
			<outFile>: output file path.
		
		Return:
			file name.
		'''
		declare.is_bool("phoneLevel",phoneLevel)

		with FileHandleManager() as fhm:

			if phoneLevel is True:
				# check the format of HMM
				if hmm is None:
					assert self.hmm is not None,"<hmm> is necessary because no HMM model is avaliable."
					hmm = self.hmm

				declare.is_potential_hmm("hmm",hmm)
				if not isinstance(hmm,str):
					hmmTemp = fhm.create("wb+",suffix=".mdl")
					hmm.save(hmmTemp)
					hmm = hmmTemp.name
			else:
				hmm = "placeholder"

			lmwts,acwts,hmms,configs,outFiles = check_multiple_resources(lmwt,acwt,hmm,config,outFile=outFile)

			extraConfigs = []
			for lmwt,acwt,config,outFile in zip(lmwts,acwts,configs,outFiles):
				declare.is_non_negative("lmwt",lmwt)
				declare.is_non_negative("acwt",acwt)
				assert outFile != "-",f"<outFile> is necessary."

				cfg = ""
				if config is not None:
					if check_config(name='write_ctm',config=config):
						for key,value in config.items():
							if isinstance(value,bool):
								if value is True:
									cfg += f"{key} "
							else:
								cfg += f" {key}={value}"
				
				extraConfigs.append(cfg)


			if phoneLevel is True:
				cmdPattern = "lattice-align-phones {hmm} {lat} ark:- | "
				cmdPattern += "lattice-to-ctm-conf --acoustic-scale={acwt} --lm-scale={lmwt} {extraCfg} ark:- {outFile}"
			else:
				cmdPattern = "lattice-to-ctm-conf --acoustic-scale={acwt} --lm-scale={lmwt} {extraCfg} ark:{lat} {outFile}"
			
			resources = {"lat":self,"acwt":acwts,"lmwt":lmwts,"hmm":hmms,"extraCfg":extraConfigs,"outFile":outFiles}

			results = run_kaldi_commands_parallel(resources,cmdPattern,analyzeResult=True)
			if len(outFiles) == 1:
				return outFiles[0]
			else:
				return outFiless

def load_lat(target,name="lat"):
	'''
	Load lattice data.

	Args:
		<target>: bytes object,file path or exkaldi Lattice object.
		<name>: a string.

	Return:
		An exkaldi lattice object.
	'''
	declare.is_valid_string("name",name)

	if isinstance(target,bytes):
		return Lattice(target,name=name)
	
	elif isinstance(target,Lattice):
		return Lattice(target.data,name=name)

	elif isinstance(target,str):
		target = list_files(target)
		allData = []
		for fileName in target:
			fileName = fileName.strip()
			if fileName.endswith('.gz'):
				cmd = f'gunzip -c {fileName}'
				out,err,_ = run_shell_command(cmd,stdout="PIPE",stderr="PIPE")
				if out == b'':
					raise WrongDataFormat('Failed to load Lattice.',err.decode())
				else:
					allData.append(out)
			else:
				try:
					with open(fileName,'rb') as fr:
						out = fr.read()
				except Exception as e:
					e.args = ("Load lattice file defeated. Please make sure it is a lattice file avaliable."+"\n"+e.args[0],)
					raise e
				else:
					allData.append(out)
		try:
			allData = b"".join(allData)
		except Exception as e:
			raise WrongOperation("Only support binary format lattice file.")
		else:
			return Lattice(data=allData,name=name)

	else:
		raise UnsupportedType(f"Expected bytes object or lattice file but got: {type_name(target)}.")

def nn_decode(prob,hmm,HCLGFile,symbolTable,beam=10,latBeam=8,acwt=1,
				minActive=200,maxActive=7000,maxMem=50000000,config=None,maxThreads=1,outFile=None):
	'''
	Decode by generating lattice from acoustic probability output by NN model.

	Share Args:
		<hmm>: file path or exkaldi HMM object.
		<HCLGFile>: HCLG graph file:
		<symbolTable>: words.txt file path or exkaldi LexiconBank or ListTable object.

	Parallel Args:
		<prob>: An exkaldi probability object. We expect the probability didn't pass any activation function,or it may generate wrong results.
		<beam>: decode beam size.
		<latBeam>: lattice beam size.
		<acwt>: acoustic model weight.
		<minActive>: minimum active.
		<maxActive>: maximum axtive.
		<maxMem>: maximum memory.
		<config>: extra configurations.
		<maxThreads>: maximum threads.
		<outFile>: output file name.

		Some usual options can be assigned directly. If you want use more,set <config> = your-configure.
		You can use .check_config('nn_decode') function to get the reference of extra configurations.
		Also run shell command "latgen-faster-mapped" to look their usage.
	
	Return:
		exkaldi Lattice object.
	'''
	declare.kaldi_existed()

	with FileHandleManager() as fhm:

		# check hmm
		declare.is_potential_hmm("hmm",hmm)
		if not isinstance(hmm,str):
			hmmTemp = fhm.create("wb+",suffix=".mdl")	
			hmm.save(hmmTemp)
			hmm = hmmTemp.name

		# check HCLGFile
		declare.is_file("HCLGFile",HCLGFile)

		# check symbolTable
		if isinstance(symbolTable,str):
			assert os.path.isfile(symbolTable),f"No such file: {symbolTable}."
		elif type_name(symbolTable) == "LexiconBank":
			wordsTemp = fhm.create("w+",suffix=".words",encoding="utf-8")
			symbolTable.dump_dict("words",wordsTemp)
			symbolTable = wordsTemp.name
		elif type_name(symbolTable) == "ListTable":
			wordsTemp = fhm.create("w+",suffix=".words",encoding="utf-8")
			symbolTable.save(wordsTemp)
			symbolTable = wordsTemp.name
		else:
			raise UnsupportedType(f"<symbolTable> should be file name,LexiconBank or ListTable object but got: {symbolTable}.")

		parameters = check_multiple_resources(prob,hmm,HCLGFile,symbolTable,
												beam,latBeam,acwt,minActive,maxActive,maxMem,
												config,maxThreads,outFile=outFile,
											)

		baseCmds = []
		outFiles = parameters[-1]

		for prob,_,_,symbolTable,beam,latBeam,acwt,minActive,maxActive,maxMem,config,maxThreads in zip(*parameters[:-1]):
			# check probability
			declare.is_probability("prob",prob)
			# check other parameters
			declare.is_positive_int("maxThreads",maxThreads)
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
				if check_config(name='nn_decode',config=config):
					for key,value in config.items():
						if isinstance(value,bool):
							if value is True:
								kaldiTool += f"{key} "
						else:
							kaldiTool += f" {key}={value}"
			baseCmds.append( kaldiTool )
			
		# define command pattern
		cmdPattern = '{kaldiTool} {hmm} {HCLG} {prob} ark:{outFile}'
		# define resources
		resources = {"prob":parameters[0],"hmm":parameters[1],"HCLG":parameters[2],"kaldiTool":baseCmds,"outFile":outFiles}
		# run
		results = run_kaldi_commands_parallel(resources,cmdPattern,analyzeResult=True)
		if len(outFiles) == 1:
			outFile = outFiles[0]
			newName = f"lat({parameters[0][0].name})"
			if outFile == "-":
				results = Lattice(data=results[2],name=newName)
			else:
				results = load_lat(outFile,name=newName)
		else:
			for i,fileName in enumerate(outFiles):
				newName = f"lat({parameters[0][i].name})"
				results[i] = load_lat(fileName,name=newName)
			
		return results

def gmm_decode(feat,hmm,HCLGFile,symbolTable,beam=10,latBeam=8,acwt=1,
				minActive=200,maxActive=7000,maxMem=50000000,config=None,maxThreads=1,outFile=None):
	'''
	Decode by generating lattice from acoustic probability output by NN model.

	Share Args:
		<hmm>: file path or exkaldi HMM object.
		<HCLGFile>: HCLG graph file:
		<symbolTable>: words.txt file path or exkaldi LexiconBank or ListTable object.
	
	Parallel Args:
		<feat>: An exkaldi feature or index table object.
		<beam>: decode beam size.
		<latBeam>: lattice beam size.
		<acwt>: acoustic model weight.
		<minActive>: minimum active.
		<maxActive>: maximum axtive.
		<maxMem>: maximum memory.
		<config>: extra configurations.
		<maxThreads>: maximum threads.
		<outFile>: output file name.

		Some usual options can be assigned directly. If you want use more,set <config> = your-configure.
		You can use .check_config('gmm_decode') function to get the reference of extra configurations.
		Also run shell command "gmm-latgen-faster" to look their usage.
	
	Return:
		exkaldi Lattice object.
	'''
	declare.kaldi_existed()

	with FileHandleManager() as fhm:

		# check hmm
		declare.is_potential_hmm("hmm",hmm)
		if not isinstance(hmm,str):
			hmmTemp = fhm.create("wb+",suffix=".mdl")	
			hmm.save(hmmTemp)
			hmm = hmmTemp.name

		# check HCLGFile
		declare.is_file("HCLGFile",HCLGFile)

		# check symbolTable
		if isinstance(symbolTable,str):
			assert os.path.isfile(symbolTable),f"No such file: {symbolTable}."
		elif type_name(symbolTable) == "LexiconBank":
			wordsTemp = fhm.create("w+",suffix=".words",encoding="utf-8")
			symbolTable.dump_dict("words",wordsTemp)
			symbolTable = wordsTemp.name
		elif type_name(symbolTable) == "ListTable":
			wordsTemp = fhm.create("w+",suffix=".words",encoding="utf-8")
			symbolTable.save(wordsTemp)
			symbolTable = wordsTemp.name
		else:
			raise UnsupportedType(f"<symbolTable> should be file name,LexiconBank or ListTable object but got: {symbolTable}.")

		parameters = check_multiple_resources(feat,hmm,HCLGFile,symbolTable,
												beam,latBeam,acwt,minActive,maxActive,maxMem,
												config,maxThreads,outFile=outFile,
											)


		baseCmds = []
		outFiles = parameters[-1]

		for feat,_,_,symbolTable,beam,latBeam,acwt,minActive,maxActive,maxMem,config,maxThreads in zip(*parameters[:-1]):
			# check feature
			declare.is_feature("feat",feat)
			# build the base command
			declare.is_positive_int("maxThreads",maxThreads)
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
				if check_config(name='gmm_decode',config=config):
					for key,value in config.items():
						if isinstance(value,bool):
							if value is True:
								kaldiTool += f"{key} "
						else:
							kaldiTool += f" {key}={value}"
			baseCmds.append( kaldiTool )
			
		# define command pattern
		cmdPattern = '{kaldiTool} {hmm} {HCLG} {feat} ark:{outFile}'
		# define resources
		resources = {"feat":parameters[0],"hmm":parameters[1],"HCLG":parameters[2],"kaldiTool":baseCmds,"outFile":outFiles}
		# run
		results = run_kaldi_commands_parallel(resources,cmdPattern)

		if len(outFiles) == 1:
			outFile = outFiles[0]
			newName = f"lat({parameters[0][0].name})"
			if outFile == "-":
				results = Lattice(data=results[2],name=newName)
			else:
				results = load_lat(outFile,name=newName)
		else:
			for i,fileName in enumerate(outFiles):
				newName = f"lat({parameters[0][i].name})"
				results[i] = load_lat(fileName,name=newName)
			
		return results

def compile_align_graph(hmm,tree,transcription,LFile,outFile,lexicons=None):
	'''
	Compile graph for training or aligning.

	Share Args:
		<hmm>: file name or exkaldi HMM object.
		<tree>: file name or exkaldi decision tree object.
		<LFile>: file name.
		<lexicons>: exkaldi lexicon bank object.
		
	Parallel Args:
		<transcription>: file path or exkaldi trancription object.
		<outFile>: output file name.

	Return:
		output file path.
	'''
	declare.is_potential_hmm("hmm",hmm)
	declare.is_potential_tree("tree",tree)

	if isinstance(tree,str):
		treeLexicons = None
	else:
		treeLexicons = tree.lex

	if isinstance(hmm,str):
		if lexicons is not None:
			declare.is_lexicon_bank("lexicons",lexicons)
		elif treeLexicons is not None:
			lexicons = treeLexicons
		else:
			raise WrongOperation("<lexicons> is necessary on this case.")
		hmm = load_hmm(hmm,lexicons=lexicons)

	return hmm.compile_train_graph(tree,transcription,LFile,outFile)

def nn_align(hmm,prob,alignGraphFile=None,tree=None,transcription=None,LFile=None,transitionScale=1.0,acousticScale=0.1,
				selfloopScale=0.1,beam=10,retryBeam=40,lexicons=None,name="ali",outFile=None):
	'''
	Align the neural network acoustic output probability.

	Share Args:
		<hmm>: file name or exkaldi HMM object.
		<tree>: file name or exkaldi decision tree object.
		<LFile>: Lexicon fst file name.
		<lexicons>: exkaldi LexiconBank object.
	
	Parallel Args:
		<prob>: exkaldi probability object or index table object.
		<alignGraphFile>: alignment graph file name.
		<transcription>: file name or exkaldi Transcription object.
		<transitionScale>: a float value,the transition weight.
		<acousticScale>: a float value,the acoustic weight.
		<selfloopScale>: a float value,the self loop weight.
		<beam>: search beam size.
		<retryBeam>: retry beam size.
		<name>: a string.
		<outFile>: output file name.

	Return:
		exkaldi alignment object or index table object.
	'''
	declare.kaldi_existed()

	with FileHandleManager() as fhm:
		# check HMM
		declare.is_potential_hmm("hmm",hmm)
		if isinstance(hmm,str):
			hmmLexicons = None
		else:
			hmmTemp = fhm.create("wb+",suffix=".mdl")
			hmm.save(hmmTemp)
			hmmLexicons = hmm.lex
			hmm = hmmTemp.name
		
		if alignGraphFile is None:
			assert None not in [tree,transcription,LFile],"When align graph is not provided,all of <tree>,<transcription> and <LFile> is necessary."
			declare.is_file("LFile",LFile)
			declare.is_potential_tree("tree",tree)
			# check tree
			if isinstance(tree,str):
				treeLexicons = None
			else:
				treeTemp = fhm.create("wb+",suffix=".tree")
				tree.save(treeTemp)
				treeLexicons = tree.lex
				tree = treeTemp.name
			# check lexicons
			if lexicons is None:
				if hmmLexicons is None:
					if treeLexicons is None:
						raise WrongOperation("<lexicons> is necessary on this case.")
					else:
						lexicons = treeLexicons
				else:
					lexicons = hmmLexicons
			else:
				declare.is_lexicon_bank("lexicons",lexicons)

			disambigTemp = fhm.create("w+",suffix="_disambig.int",encoding="utf-8")
			lexicons.dump_dict(name="disambig",outFile=disambigTemp,dumpInt=True)

			parameters = check_multiple_resources(prob,transcription,transitionScale,acousticScale,selfloopScale,beam,retryBeam,name,outFile=outFile)
			baseCmds = []
			for prob,transcription,transitionScale,acousticScale,selfloopScale,beam,retryBeam,name,_ in zip(*parameters):
				# check 
				declare.is_probability("prob",prob)
				declare.is_potential_transcription("transcription",transcription)
				declare.is_valid_string("name",name)
				# build base command
				cmd = f"align-mapped --transition-scale={transitionScale} --acoustic-scale={acousticScale} --self-loop-scale={selfloopScale} "
				cmd += f"--beam={beam} --retry-beam={retryBeam} "
				cmd += f"--read-disambig-syms={disambigTemp.name} "
				cmd += f"{tree} {hmm} {LFile} "
				baseCmds.append(cmd)
			# define command templet and resources
			cmdPattern = "{baseTool} {prob} ark:{trans} ark:{outFile}"
			resources = {"prob":parameters[0],"trans":parameters[1],"baseTool":baseCmds,"outFile":parameters[-1]}
			names = parameters[-2]

		else:
			assert tree is None and transcription is None and LFile is None,"When use compiled align graph,any of <tree>,<transcription> and <LFile> is invalid."
			
			parameters = check_multiple_resources(prob,alignGraphFile,transitionScale,acousticScale,selfloopScale,beam,retryBeam,name,outFile=outFile)

			baseCmds = []
			for prob,alignGraphFile,transitionScale,acousticScale,selfloopScale,beam,retryBeam,name,_ in zip(*parameters):
				# check
				declare.is_probability("prob",prob)
				declare.is_file("alignGraphFile",alignGraphFile)
				declare.is_valid_string("name",name)
				# build base command
				cmd = f"align-compiled-mapped --transition-scale={transitionScale} --acoustic-scale={acousticScale} --self-loop-scale={selfloopScale} "
				cmd += f"--beam={beam} --retry-beam={retryBeam} "
				cmd += f"{hmm} ark:{alignGraphFile}"
				baseCmds.append(cmd)
			# define command templet and resources
			cmdPattern = "{baseTool} {prob} ark:{outFile}"
			resources = {"prob":parameters[0],"baseTool":baseCmds,"outFile":parameters[-1]}			
			names = parameters[-2]
		# run
		return run_kaldi_commands_parallel(resources,cmdPattern,generateArchive="ali",archiveNames=names)

def gmm_align(hmm,feat,alignGraphFile=None,tree=None,transcription=None,LFile=None,transitionScale=1.0,acousticScale=0.1,
				selfloopScale=0.1,beam=10,retryBeam=40,boostSilence=1.0,careful=False,name="ali",lexicons=None,outFile=None):
	'''
	Align the feature.

	Share Args:
		<hmm>: file name or exkaldi HMM object.
		<tree>: file name or exkaldi decision tree object.
		<LFile>: Lexicon fst file name.
		<lexicons>: exkaldi LexiconBank object.
		<boostSilence>: boost silence.
		<careful>: a bool value.
	
	Parallel Args:
		<feat>: exkaldi feature object or index table object.
		<alignGraphFile>: file name.
		<transcription>: file name or exkaldi transcription object.
		<transitionScale>.
		<acousticScale>.
		<selfloopScale>.
		<beam>: search beam size.
		<retryBeam>: retry beam size.
		<name>: a string.
		<outFile>: output file name.

	Return:
		exkaldi alignment object or index table object.
	'''
	declare.kaldi_existed()

	with FileHandleManager() as fhm:
		# check tree
		if tree is not None:
			declare.is_potential_tree("tree",tree)
			if isinstance(tree,str):
				treeLexicons = None
			else:
				treeTemp = fhm.create("wb+",suffix=".tree")
				tree.save(treeTemp)
				treeLexicons = tree.lex
				tree = treeTemp.name
		# check HMM and lexicons and boost silence
		declare.is_potential_hmm("hmm",hmm)
		if isinstance(hmm,str):
			if lexicons is None:
				if treeLexicons is None:
					raise WrongOperation("<lexicons> is necessary on this case.")
				else:
					lexicons = treeLexicons
			else:
				declare.is_lexicon_bank("lexicons",lexicons)
			# boost silence
			optionSilence = ":".join(lexicons("optional_silence",True))
			hmmTemp = fhm.create("wb+",suffix=".mdl")
			cmd = f'gmm-boost-silence --boost={boostSilence} {optionSilence} {hmm} {hmmTemp.name}'
			out,err,cod = run_shell_command(cmd,stdout="PIPE",stderr="PIPE")

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
				declare.is_lexicon_bank("lexicons",lexicons)
			# boost silence
			optionSilence = ":".join(lexicons("optional_silence",True))
			hmmTemp = fhm.create("wb+",suffix=".mdl")
			cmd = f'gmm-boost-silence --boost={boostSilence} {optionSilence} - {hmmTemp.name}'
			out,err,cod = run_shell_command(cmd,stdin="PIPE",stdout="PIPE",stderr="PIPE",inputs=hmm.data)
		
		if cod != 0:
			raise KaldiProcessError("Generate new HMM defeated.",err.decode())
		hmmTemp.seek(0)

		# then align
		if alignGraphFile is None:
			assert None not in [tree,transcription,LFile],"When align graph is not provided,all of <tree>,<transcription> and <LFile> is necessary."
			declare.is_file("LFile",LFile)

			disambigTemp = fhm.create("w+",suffix="_disambig.int",encoding="utf-8")
			lexicons.dump_dict(name="disambig",outFile=disambigTemp,dumpInt=True)

			parameters = check_multiple_resources(feat,transcription,transitionScale,acousticScale,
																						selfloopScale,beam,retryBeam,careful,name,outFile=outFile)
			baseCmds = []
			for feat,transcription,transitionScale,acousticScale,selfloopScale,beam,retryBeam,careful,name,_ in zip(*parameters):
				# check
				declare.is_feature("feat",feat)
				declare.is_potential_transcription("transcription",transcription)
				declare.is_valid_string("name",name)
				# build base command
				cmd = f"gmm-align --transition-scale={transitionScale} --acoustic-scale={acousticScale} --self-loop-scale={selfloopScale} "
				cmd += f"--beam={beam} --retry-beam={retryBeam} --careful={careful} "
				cmd += f"--read-disambig-syms={disambigTemp.name} "
				cmd += f"{tree} {hmmTemp.name} {LFile} "
				baseCmds.append(cmd)
			# define command templet and resources
			cmdPattern = "{baseTool} {feat} ark:{trans} ark:{outFile}"
			resources = {"feat":parameters[0],"baseTool":baseCmds,"trans":parameters[1],"outFile":parameters[-1]}
			names = parameters[-2]

		else:
			assert tree is None and transcription is None and LFile is None,"When use compiled align graph,any of <tree>,<transcription> and <LFile> is invalid."
			
			parameters = check_multiple_resources(feat,alignGraphFile,transitionScale,acousticScale,
																						selfloopScale,beam,retryBeam,careful,name,outFile=outFile)

			baseCmds = []
			for feat,alignGraphFile,transitionScale,acousticScale,selfloopScale,beam,retryBeam,careful,name,_ in zip(*parameters):
				# check
				declare.is_feature("feat",feat)
				declare.is_file("alignGraphFile",alignGraphFile)
				declare.is_valid_string("name",name)
				# build base command				
				cmd = f"gmm-align-compiled --transition-scale={transitionScale} --acoustic-scale={acousticScale} --self-loop-scale={selfloopScale} "
				cmd += f"--beam={beam} --retry-beam={retryBeam}  --careful={careful} "
				cmd += f"{hmm} ark:{alignGraphFile} "
				baseCmds.append(cmd)
			# define command templet and resources
			cmdPattern = "{baseTool} {feat} ark:{outFile}"
			resources = {"feat":parameters[0],"baseTool":baseCmds,"outFile":parameters[-1]}	
			names = parameters[-2]		

		# run
		return run_kaldi_commands_parallel(resources,cmdPattern,generateArchive="ali",archiveNames=names)