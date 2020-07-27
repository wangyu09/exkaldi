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

"""Train a DexisionTree and HMM-GMM model"""
import os
import glob
import subprocess
import copy
import time, datetime
from collections import namedtuple
import numpy as np

from exkaldi.version import info as ExkaldiInfo
from exkaldi.version import WrongPath, KaldiProcessError, UnsupportedType, WrongOperation, WrongDataFormat
from exkaldi.core.archieve import BytesArchieve, Transcription, ListTable, BytesAlignmentTrans, BytesFmllrMatrix
from exkaldi.core.load import load_ali, load_index_table, load_transcription, load_list_table
from exkaldi.core.feature import transform_feat, use_fmllr
from exkaldi.core.common import check_mutiple_resources, run_kaldi_commands_parallel, merge_archieves, utt_to_spk
from exkaldi.utils.utils import run_shell_command, run_shell_command_parallel, check_config, make_dependent_dirs, type_name, list_files
from exkaldi.utils.utils import FileHandleManager
from exkaldi.utils import declare

class DecisionTree(BytesArchieve):
	'''
	Decision tree.
	'''
	def __init__(self, data=b"", contextWidth=3, centralPosition=1, lexicons=None, name="tree"):

		declare.kaldi_existed()

		super().__init__(data, name)

		if not lexicons is None:
			declare.is_lexicon_bank("lexicons", lexicons)
		declare.is_positive_int("contextWidth", contextWidth)
		declare.is_non_negative_int("centralPosition", centralPosition)

		self.__lex = lexicons
		self.__contextWidth = contextWidth
		self.__centralPosition = centralPosition
	
	@property
	def lex(self):
		'''
		Return lexicon bank.
		'''
		return self.__lex

	@property
	def contextWidth(self):
		return self.__contextWidth

	@property
	def centralPosition(self):
		return self.__centralPosition

	def accumulate_stats(self, feat, hmm, alignment, outFile, lexicons=None):
		'''
		Accumulate statistics in order to compile questions.

		Share Args:
			<hmm>: exkaldi HMM object or file name.
			<lexicons>: None. If no any lexicons provided in DecisionTree, this is expected.

		Parallel Args:
			<feat>: exkaldi feature or index table object.
			<alignment>: exkaldi alignment or index table object.
			<outFile>: file name.
		
		Return:
			output file paths.
		'''
		declare.is_potential_hmm("hmm", hmm)
		if isinstance(hmm, BaseHMM):
			hmmLex = hmm.lex
		else:
			hmmLex = None

		if lexicons is None:
			if self.lex is None:
				assert hmmLex is None, "<lexicons> is avaliable on this case."
				lexicons = hmmLex
			else:
				lexicons = self.lex
		else:
			declare.is_lexicon_bank("lexicons", lexicons)

		with FileHandleManager() as fhm:

			if isinstance(hmm, BaseHMM):
				hmmTemp = fhm.create("wb+", suffix=".mdl")
				hmm.save(hmmTemp)
				hmm = hmmTemp.name
			
			feats, hmms, alignments, outFiles = check_mutiple_resources(feat, hmm, alignment, outFile=outFile)

			for feat, hmm, alignment, outFile in zip(feats, hmms, alignments, outFiles):
				declare.is_feature("feat", feat)
				declare.is_alignment("alignment", alignment)
				declare.is_valid_string("outFile", outFile)
				assert outFile != "-", f"Out file name is necessary."

			# define command pattern
			ciphones = ":".join(lexicons("context_indep", True))
			cmdPattern = f'acc-tree-stats --context-width={self.contextWidth} --central-position={self.centralPosition} --ci-phones={ciphones} '
			cmdPattern += '{hmm} {feat} {alignment} {outFile}'
			# define resources
			resources = {"feat":feats, "hmm":hmms, "alignment":alignments, "outFile":outFiles}
			# run
			run_kaldi_commands_parallel(resources, cmdPattern, analyzeResult=True)
			# analysis result
			if len(outFiles) == 1:
				return outFiles[0]
			else:
				return outFiles
		
	def compile_questions(self, treeStatsFile, topoFile, outFile, lexicons=None):
		'''
		Compile questions.

		Share Args:
			<treeStatsFile>: file path.
			<topoFile>: topo file path.
			<outFile>: file name.
			<lexicons>: None. If no any lexicons provided in DecisionTree, this is necessary.
		
		Parallel Args:
			Null
		
		Return:
			output file path.
		'''
		if lexicons is None:
			assert self.lex is not None, "<lexicons> is necessary on this case."
			lexicons = self.lex
		else:
			declare.is_lexicon_bank("lexicons", lexicons)

		declare.is_valid_file_name("outFile", outFile)
		make_dependent_dirs(outFile, pathIsFile=True)
	
		declare.is_file("treeStatsFile", treeStatsFile)
		declare.is_file("topoFile", topoFile)
		
		with FileHandleManager() as fhm:
			
			# first step: cluster phones
			setsTemp = fhm.create("w+", suffix="_sets.int", encoding="utf-8")
			lexicons.dump_dict("sets", setsTemp, True)

			cmd = f'cluster-phones --context-width={self.contextWidth} --central-position={self.centralPosition} '
			cmd += f'{treeStatsFile} {setsTemp.name} -'

			out, err, cod = run_shell_command(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

			if (isinstance(cod,int) and cod != 0):
				print(err.decode())
				raise KaldiProcessError("Failed to cluster phones.")
			
			# second step: compile questions
			extra = lexicons.dump_dict("extra_questions", None, True)
			questions = "\n".join([out.decode().strip(), extra.strip()])

			cmd = f'compile-questions --context-width={self.contextWidth} --central-position={self.centralPosition} '
			cmd += f'{topoFile} - {outFile}'

			out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, inputs=questions)

			if (isinstance(cod, int) and cod != 0) or (not os.path.isfile(outFile)) or os.path.getsize(outFile) == 0:
				print(err.decode())
				raise KaldiProcessError("Failed to compile questions.")
			
			return os.path.abspath(outFile)

	def build(self, treeStatsFile, questionsFile, topoFile, numLeaves, clusterThresh=-1, lexicons=None):
		'''
		Build tree.

		Share Args:
			<treeStatsFile>: file path.
			<questionsFile>: file path.
			<numLeaves>: target numbers of leaves.
			<topoFile>: topo file path.
			<lexicons>: None. If no any lexicons provided in DecisionTree, this is expected.
						In this step, we will use "roots" lexicon.
		Parallel Args:
			Null
		
		Return:
			Absolute path of out file.
		'''
		declare.is_file("treeStatsFile", treeStatsFile)
		declare.is_file("questionsFile", questionsFile)
		declare.is_file("topoFile", topoFile)
		declare.is_positive_int("numLeaves", numLeaves)

		if lexicons is None:
			assert self.lex is not None, "<lexicons> is necessary on this case."
			lexicons = self.lex
		else:
			declare.is_lexicon_bank("lexicons", lexicons)
		
		with FileHandleManager() as fhm:

			rootsTemp = fhm.create("w+", suffix=".int", encoding="utf-8")
			lexicons.dump_dict("roots", rootsTemp, True)

			cmd = f'build-tree --context-width={self.contextWidth} --central-position={self.centralPosition} '
			cmd += f'--verbose=1 --max-leaves={numLeaves} --cluster-thresh={clusterThresh} '
			cmd += f'{treeStatsFile} {rootsTemp.name} {questionsFile} {topoFile} -'

			out, err, cod = run_shell_command(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

			if (isinstance(cod,int) and cod !=0 ) or len(out) == 0:
				print(err.decode())
				raise KaldiProcessError("Failed to build tree.")
			else:
				self.reset_data(out)
				return self			

	def train(self, feat, hmm, alignment, topoFile, numLeaves, tempDir, clusterThresh=-1, lexicons=None):
		'''
		This is a hign-level API to build a decision tree.

		Share Args:
			<hmm>: file path or exkaldi HMM object.
			<topoFile>: topo file path.
			<numLeaves>: target numbers of leaves.
			<tempDir>: a temp directory to storage some intermidiate files.
			<lexicons>: None. If no any lexicons provided in DecisionTree, this is expected.

		Parallel Args:
			<feat>: exkaldi feature object.
			<alignment>: file path or exkaldi transition-ID Alignment object.
		'''
		print("Start to build decision tree.")
		make_dependent_dirs(tempDir, pathIsFile=False)

		starttime = datetime.datetime.now().strftime("%Y/%m/%d-%H:%M:%S")
		print(f"Start Time: {starttime}")

		print("Accumulate tree statistics")
		statsFile = os.path.join(tempDir, "treeStats.acc")
		results = self.accumulate_stats(feat, hmm, alignment, outFile=statsFile, lexicons=lexicons)
		if isinstance(results, list):
			sum_tree_stats(results, statsFile)

		print("Cluster phones and compile questions")
		questionsFile = os.path.join(tempDir, "questions.qst")
		self.compile_questions(statsFile, topoFile, outFile=questionsFile, lexicons=lexicons)

		print("Build tree")
		self.build(statsFile, questionsFile, topoFile, numLeaves, clusterThresh)

		treeFile = os.path.join(tempDir,"tree")
		self.save(treeFile)

		print('Done to build the decision tree.')
		print(f"Saved Final Tree: {treeFile}")
		endtime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
		print(f"End Time: {endtime}")

	def save(self, fileName="tree"):
		'''
		Save tree to file.

		Args:
			<fileName>: a string or file handle.
		'''
		declare.is_valid_file_name_or_handle("fileName", fileName)

		if isinstance(fileName, str):
			make_dependent_dirs(fileName, pathIsFile=True)
			with open(fileName,"wb") as fw:
				fw.write(self.data)
			return fileName
		else:
			fileName.truncate()
			fileName.write(self.data)
			fileName.seek(0)
			return fileName

	def load(self, target):
		'''
		Reload a tree from file.
		The original data will be discarded.

		Args:
			<target>: file name.
		'''
		declare.is_file("target", target)
		
		cmd = f"tree-info {target}"
		out, err, cod = run_shell_command(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		if isinstance(cod,int) and cod != 0:
			print(err.decode())
			raise WrongDataFormat("Failed to load tree.")
		else:
			out = out.decode().strip().split("\n")
			self.__contextWidth = int(out[1].strip().split()[-1])
			self.__centralPosition = int(out[2].strip().split()[-1])
			with open(target, "rb") as fr:
				data = fr.read()
			self.reset_data(data)
			return self

	@property
	def info(self):
		'''
		Get the information of tree.
		'''
		declare.not_void("hmm", self)
		
		cmd = f"tree-info -"
		out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=self.data)
		if isinstance(cod, int) and cod != 0:
			print(err.decode())
			raise WrongDataFormat("Failed to get the infomation of model.")
		else:
			out = out.decode().strip().split("\n")
			names = []
			values = []
			for t1 in out:
				t1 = t1.strip().split()
				value = int(t1[-1])
				name = t1[-2].split("-")
				for index,t2 in enumerate(name[1:],start=1):
					name[index] = t2[0].upper() + t2[1:]
				name = "".join(name)

				names.append(name)
				values.append(value)
			return namedtuple("TreeInfo",names)(*values)

class BaseHMM(BytesArchieve):

	def __init__(self, data=b"", name="hmm", lexicons=None):

		declare.kaldi_existed()

		super().__init__(data, name)
		if not lexicons is None:
			declare.is_lexicon_bank("lexicons", lexicons)
		self.__lex = lexicons

	@property
	def lex(self):
		'''
		Get the lexicon.
		'''
		return self.__lex
	
	def compile_train_graph(self, tree, transcription, LFile, outFile, lexicons=None):
		'''
		Compile training graph.

		Share Args:
			<tree>: file name or exkaldi DecisionTree object.
			<LFile>: file path.
			<lexicons>: None. If no any lexicons provided in DecisionTree, this is expected.

		Args:
			<transcription>: file path or exkaldi Transcription object with int format.
							Note that: int fotmat, not text format.
			<outFile>: graph output file path.

		Return:
			output file paths.
		'''
		declare.kaldi_existed()
		declare.not_void(type_name(self),self)
		declare.is_file("LFile", LFile)

		declare.is_potential_tree("tree", tree)
		if isinstance(tree, DecisionTree):
			treeLex = tree.lex
		else:
			treeLex = None
		if lexicons is None:
			if self.lex is None:
				assert treeLex is None, "<lexicons> is avaliable on this case."
				lexicons = treeLex
			else:
				lexicons = self.lex
		else:
			declare.is_lexicon_bank("lexicons", lexicons)

		with FileHandleManager() as fhm:

			hmmTemp = fhm.create("wb+", suffix=".mdl")
			self.save(hmmTemp)

			if isinstance(tree, DecisionTree):
				treeTemp = fhm.create("wb+", suffix=".tree")
				tree.save(treeTemp)
				tree = treeTemp.name
			
			disambigTemp = fhm.create("w+", suffix=".disambig")
			lexicons.dump_dict("disambig", disambigTemp, True)

			trees,models,transcriptions,LFiles,disambigs,outFiles = check_mutiple_resources( tree, hmmTemp.name, transcription, LFile, disambigTemp.name, outFile=outFile)

			for i, transcription in enumerate(transcriptions):
				transcription = load_transcription(transcription)
				sample = transcription.subset(nRandom=1)
				for uttID, txt in sample.items():
					txt = txt.split()
					try:
						[ int(w) for w in txt ]
					except ValueError:
						raise WrongDataFormat("Transcription should be int-value format. Please convert it firstly.")
				transcriptions[i] = transcription
		
			# define command pattern
			cmdPattern = 'compile-train-graphs --read-disambig-syms={dismbig} {tree} {model} {LFile} ark:{trans} ark:{outFile}'
			# define resources
			resources = {"model":models, "dismbig":disambigs, "tree":trees, "LFile":LFiles, "trans":transcriptions, "outFile":outFiles}
			# run commands parallely
			run_kaldi_commands_parallel(resources, cmdPattern)
			# analyze result
			if len(outFiles) == 1:
				return outFiles[0]
			else:
				return outFiles

	def update(self, statsFile, numgauss, power=0.25, minGaussianOccupancy=10):
		'''
		Update the parameters of HMM model.

		Share Args:
			<statsFile>: file name.
			<numgauss>: int value.
			<power>.
			<minGaussianOccupancy>.

		Parallel Args:
			Null
		'''
		declare.not_void(type_name(self), self)
		declare.is_file("statsFile", statsFile)

		gaussians = self.info.gaussians
		declare.larger("new number of gaussian", numgauss, "current number", gaussians)

		cmd = f'gmm-est --min-gaussian-occupancy={minGaussianOccupancy} --mix-up={numgauss} --power={power} '
		cmd += f'- {statsFile} -'

		out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=self.data)

		if (isinstance(cod,int) and cod != 0 ) or len(out) == 0:
			print(err.decode())
			raise KaldiProcessError("Failed to estimate new GMM parameters.")
		else:
			self.reset_data(out)
			return self

	def align(self, feat, trainGraphFile, transitionScale=1.0, acousticScale=0.1, 
				selfloopScale=0.1, beam=10, retryBeam=40, boostSilence=1.0, careful=False, name="ali", lexicons=None, outFile=None):
		'''
		Align acoustic feature with kaldi vertibi algorithm.

		Share Args:
			<lexicons>: None. If no any lexicons provided in DecisionTree, this is expected.
						In this step, we will use "context_indep" lexicon.
		
		Parallel Args:
			<feat>: exakldi feature or index table object.
			<trainGraphFile>: file path.
			<transitionScale>.
			<acousticScale>.
			<selfloopScale>.
			<beam>.
			<retryBeam>.
			<boostSilence>.
			<careful>.
			<name>.
			<outFile>.
		
		Return:
			exkaldi alignment object or index table object.
		'''
		declare.not_void(type_name(self), self)

		if lexicons is None:
			assert self.lex is not None, "No <lexicons> avaliable defaultly. Provide it please."
			lexicons = self.lex
		else:
			declare.is_lexicon_bank("lexicons", lexicons)

		results = check_mutiple_resources(feat, trainGraphFile, transitionScale, acousticScale, selfloopScale, 
										  beam, retryBeam, boostSilence, careful, name, outFile=outFile,
										)
		outFiles = results[-1]
		names = results[-2]
		with FileHandleManager() as fhm:

			modelTemp = fhm.create("wb+", suffix=".mdl")
			self.save(modelTemp)
			optionSilence = ":".join(lexicons("optional_silence", True))

			baseCmds = []
			for feat, graph, transScale, acScale, selfScale, beam, retryBeam, boostSilence, careful, _, _ in zip(*results):
				declare.is_feature("feat", feat)
				declare.is_file("trainGraphFile", graph)
				declare.is_positive_float("transitionScale", transScale)
				declare.is_positive_float("acousticScale", acScale)
				declare.is_positive_float("selfloopScale", selfScale)
				declare.is_positive_int("beam", beam)
				declare.is_positive_int("retryBeam", retryBeam)
				declare.is_non_negative_float("boostSilence", boostSilence)
				declare.is_bool("careful", careful)
				
				cmd = f'gmm-boost-silence --boost={boostSilence} {optionSilence} {modelTemp.name} - | '
				cmd += f'gmm-align-compiled --transition-scale={transScale} --acoustic-scale={acScale} --self-loop-scale={selfScale} '
				cmd += f'--beam={beam} --retry-beam={retryBeam} --careful={careful} '
				cmd += f'- ark:{graph}'
				baseCmds.append(cmd)
				
			cmdPattern ='{tool} {feat} ark:{outFile}'

			# define command pattern
			# define resources
			resources = {"feat":results[0], "tool":baseCmds, "outFile":outFiles}
			# run
			return run_kaldi_commands_parallel(resources, cmdPattern, analyzeResult=True, generateArchieve="ali", archieveNames=names)
	
	def accumulate_stats(self, feat, alignment, outFile):
		'''
		Accumulate GMM statistics in order to update GMM parameters.

		Share Args:
			Null
		
		Parallel Args:
			<feat>: exkaldi feature object.
			<alignment>: exkaldi transitionID alignment object or file path.
			<outFile>: file name.
		
		Return:
			output file paths.
		'''
		declare.not_void(type_name(self), self)
		feats, alignments, outFiles = check_mutiple_resources(feat, alignment, outFile=outFile)

		with FileHandleManager() as fhm:

			modelTemp = fhm.create("wb+", suffix=".mdl")
			self.save(modelTemp)

			models = []
			for feat, alignment, outFile in zip(feats, alignments, outFiles):
				declare.is_feature("feat", feat)
				declare.is_alignment("alignment", alignment)
				models.append(modelTemp.name)
				# check out file
				declare.is_valid_file_name("outFile", outFile)
				assert outFile != "-", f"Output file name is necessary."
			
			# define the command pattern
			cmdPattern = 'gmm-acc-stats-ali {model} {feat} {alignment} {outFile}'
			# define resources
			resources = {"feat":feats, "model":models, "alignment":alignments, "outFile":outFiles}
			# run
			run_kaldi_commands_parallel(resources, cmdPattern, analyzeResult=True)
			# analyze result
			if len(outFiles) == 1:
				return outFiles[0]
			else:
				return outFiles

	def align_equally(self, feat, trainGraphFile, name="equal_ali", outFile=None):
		'''
		Align feature averagely.

		Share Args:
			Null
		
		Parallel Args:
			<feat>: exkaldi feature object.
			<trainGraphFile>: file path.
			<name>: string.
			<outFile>: file name.
		
		Return:
			exakldi alignment or index table object.
		'''
		declare.not_void(type_name(self), self)

		feats, trainGraphFiles, names, outFiles = check_mutiple_resources(feat, trainGraphFile, name, outFile=outFile)
		for feat, trainGraphFile in zip(feats, trainGraphFiles):
			declare.is_feature("feat", feat)
			declare.is_file("trainGraphFile", trainGraphFile)
		# define command pattern
		cmdPattern = 'align-equal-compiled ark:{graph} {feat} ark:{outFile}'
		# define resources
		resources = {"feat":feats, "graph":trainGraphFiles, "outFile":outFiles}
		# run
		return run_kaldi_commands_parallel(resources, cmdPattern, generateArchieve="ali", archieveNames=names)
	
	def save(self, fileName):
		'''
		Save model to file.

		Args:
			<fileName>: a string.
		'''
		declare.not_void(type_name(self), self)

		declare.is_valid_file_name_or_handle("fileName", fileName)
		if isinstance(fileName, str):
			if not fileName.strip().endswith(".mdl"):
				fileName += ".mdl"
			make_dependent_dirs(fileName, pathIsFile=True)
			with open(fileName, "wb") as fw:
				fw.write(self.data)
			return fileName
		else:
			fileName.truncate()
			fileName.write(self.data)
			fileName.seek(0)
			
			return fileName
	
	def load(self, target):
		'''
		Reload a HMM-GMM model from file.
		The original data will be discarded.

		Args:
			<target>: file path.
		'''
		declare.is_file("target", target)
		
		cmd = f"gmm-info {target}"
		out, err, cod = run_shell_command(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		if isinstance(cod,int) and cod != 0:
			print(err.decode())
			raise WrongDataFormat("Failed to load HMM-GMM model.")
		else:
			with open(target, "rb") as fr:
				data = fr.read()
			self.reset_data(data)
			return self

	@property
	def info(self):
		'''
		Get the information of model.
		'''
		declare.not_void(type_name(self), self)
		
		cmd = f"gmm-info -"
		out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=self.data)
		if isinstance(cod, int) and cod != 0:
			print(err.decode())
			raise WrongDataFormat("Failed to get the infomation of model.")
		else:
			out = out.decode().strip().split("\n")
			names = []
			values = []
			for t1 in out:
				t1 = t1.strip().split()
				value = int(t1[-1])
				name = t1[-2].split("-")
				for index,t2 in enumerate(name[1:],start=1):
					name[index] = t2[0].upper() + t2[1:]
				name = "".join(name)

				names.append(name)
				values.append(value)
			return namedtuple("ModelInfo",names)(*values)

	def transform_gmm_means(self, matrixFile):
		'''
		Transform GMM means.

		Args:
			<matrixFile>: a trnsform matrix file.
		'''
		declare.not_void(type_name(self), self)
		declare.is_file("matrixFile", matrixFile)

		cmd = f'gmm-transform-means {matrixFile} - -'
		out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=self.data)

		if cod != 0:
			print(err.decode())
			raise KaldiProcessError(f"Failed to transform GMM means.")
		else:
			self.reset_data(out)

class MonophoneHMM(BaseHMM):

	def __init__(self, lexicons=None, name="mono"):
		super().__init__(data=None, name=name, lexicons=lexicons)
		self.__tempTree = None

	def initialize(self, feat, topoFile, lexicons=None):
		'''
		Initialize Monophone HMM-GMM model.

		Share Args:
			<feat>: exkaldi feature or index table object.
			<topoFile>: file path.
			<lexicons>: None. If no any lexicons provided in DecisionTree, this is expected.
						In this step, we will use "context_indep" lexicon.
		
		Parallel Args:
			Null
		'''
		declare.is_file("topoFile", topoFile)

		if lexicons is None:
			assert self.lex is not None, "No <lexicons> avaliable defaultly, so provide it please."
			lexicons = self.lex
		else:
			declare.is_lexicon_bank("lexicons", lexicons)

		declare.is_feature("feat", feat)
		feat = feat.subset(nHead=10)
		if type_name(feat) == "NumpyFeature":
			feat = feat.to_bytes()
		elif type_name(feat) == "ArkIndexTable":
			feat = feat.fetch(arkType="feat")
		
		with FileHandleManager() as fhm:

			setsTemp = fhm.create('w+', suffix=".int", encoding="utf-8")
			lexicons.dump_dict("sets", setsTemp, True)

			treeTemp = fhm.create('wb+')
			modelTemp = fhm.create('wb+')

			cmd = f'gmm-init-mono --shared-phones={setsTemp.name} --train-feats=ark:- {topoFile} {feat.dim} {modelTemp.name} {treeTemp.name}'
			out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=feat.data)

			if isinstance(cod,int) and cod != 0:
				print(err.decode())
				raise KaldiProcessError("Failed to initialize mono model.")

			treeTemp.seek(0)
			self.__tempTree = DecisionTree(lexicons=self.lex, data=treeTemp.read(), contextWidth=1, centralPosition=0, name="monoTree")
			modelTemp.seek(0)
			self.reset_data(modelTemp.read())
	
	@property
	def tree(self):
		'''
		Get the temp tree in monophone model.
		'''
		return self.__tempTree

	def train(self, feat, transcription, LFile, tempDir,
				numIters=40, maxIterInc=30, totgauss=1000, realignIter=None,
				transitionScale=1.0, acousticScale=0.1, selfloopScale=0.1,
				initialBeam=6, beam=10, retryBeam=40,
				boostSilence=1.0, careful=False, power=0.25, minGaussianOccupancy=10, lexicons=None):
		'''
		This is a high-level API to train the HMM-GMM model.

		Share Args:
			<LFile>: L.fst file path.
			<tempDir>: A directory to save intermidiate files.
			<numIters>: Int value, the max iteration times.
			<maxIterInc>: Int value, increase numbers of gaussian functions when iter is smaller than <numIters>.
			<totgauss>: Int value, the rough target numbers of gaussian functions.
			<realignIter>: None or list or tuple, the iter to realign.
			<transitionScale>:.
			<acousticScale>:.
			<selfloopScale>:.
			<initialBeam>:.
			<beam>:.
			<retryBeam>:.
			<boostSilence>.
			<careful>.
			<power>.
			<minGaussianOccupancy>.
			<lexicons>.
		
		Parallel Args:
			<feat>: exkaldi feature or index table object.
			<transcription>: exkaldi transcription object or file name (text format).

		Return:
			an index table object of final alignment.
		'''
		assert not self.is_void, f"Please initialize this model firstly."

		exNumgauss = self.info.gaussians
		declare.larger("Total number of gaussian", totgauss, "current number", exNumgauss)

		if realignIter is not None:
			declare.is_classes("realignIter", realignIter, (list,tuple))

		print("Start to train mono model.")
		make_dependent_dirs(tempDir, pathIsFile=False)

		starttime = datetime.datetime.now().strftime("%Y/%m/%d-%H:%M:%S")
		print(f"Start Time: {starttime}")

		if lexicons is None:
			assert self.lex is not None, "No <lexicons> avaliable defaultly, so provide it please."
			lexicons = self.lex
		else:
			declare.is_lexicon_bank("lexicons", lexicons)

		print("Convert transcription to int value format.")
		if isinstance(transcription, (list,tuple)):
			for i, trans in enumerate(transcription):
				transcription[i] = transcription_to_int(trans, symbolTable=lexicons, unkSymbol=lexicons("oov"))
				transcription[i].save( os.path.join(tempDir, f"{i}_train_text.int") )
		else:
			transcription = transcription_to_int(transcription, symbolTable=lexicons, unkSymbol=lexicons("oov"))
			transcription.save( os.path.join(tempDir, f"train_text.int") )

		print('Compiling training graph.')
		trainGraphFile = self.compile_train_graph(tree=self.tree, transcription=transcription, LFile=LFile,
													outFile=os.path.join(tempDir,"train_graph"), lexicons=lexicons
												)

		declare.is_positive_int("maxIterInc", maxIterInc)
		incgauss = (totgauss - exNumgauss)//maxIterInc
		search_beam = initialBeam
		
		for i in range(0, numIters+1, 1):
			
			print(f"Iter >> {i}")
			iterStartTime = time.time()
			# 1. align
			if i == 0:
				print('Aligning data equally')
				ali = self.align_equally(feat, trainGraphFile, outFile=os.path.join(tempDir,"train.ali"))
			elif (realignIter is None) or (i in realignIter):
				print("Aligning data")
				del ali
				ali = self.align(feat, trainGraphFile, transitionScale, acousticScale, selfloopScale, 
									search_beam, retryBeam, boostSilence, careful, lexicons=lexicons, 
									outFile=os.path.join(tempDir,"train.ali"),
								)
			else:
				print("Skip aligning")

			print("Accumulate GMM statistics")
			statsFile = os.path.join(tempDir,"stats.acc")
			_statsFiles = self.accumulate_stats(feat, alignment=ali, outFile=statsFile)
			if isinstance(_statsFiles, list):
				sum_gmm_stats(_statsFiles, statsFile)

			print("Update GMM parameters")
			gaussianOccupancy = 3 if i == 0 else minGaussianOccupancy
			self.update(statsFile, exNumgauss, power, gaussianOccupancy)

			if i >= 1:
				search_beam = beam
				exNumgauss += incgauss

			iterTimeCost = time.time() - iterStartTime
			print(f"Used time: {iterTimeCost:.4f} seconds")

		modeLFile = os.path.join(tempDir,"final.mdl")
		self.save(modeLFile)

		print('Align last time with final model.')
		del ali
		for tempAliFile in list_files(os.path.join(tempDir,f"*train.ali")):
			os.remove(tempAliFile)

		ali = self.align(feat, trainGraphFile, transitionScale, acousticScale, selfloopScale, 
							search_beam, retryBeam, boostSilence, careful, 
							outFile=os.path.join(tempDir,"final.ali"),
						)
		if isinstance(ali, list):
			ali = merge_archieves(ali)
		treeFile = os.path.join(tempDir, "tree")
		self.tree.save(treeFile)

		print('Done to train the monophone model.')
		print(f"Saved Final Model: {modeLFile}")
		print(f"Saved Alignments: ", ",".join(list_files(os.path.join(tempDir,"*final.ali"))))
		print(f"Saved tree: {treeFile}")
		endtime = datetime.datetime.now().strftime("%Y/%m/%d-%H:%M:%S")
		print(f"End Time: {endtime}")

		return ali

class TriphoneHMM(BaseHMM):

	def __init__(self, lexicons=None, name="tri"):
		super().__init__(data=None, name=name, lexicons=lexicons)
		self.__tree = None

	def initialize(self, tree, topoFile, feat=None, treeStatsFile=None):
		'''
		Initialize a Triphone Model.

		Share Args:
			<tree>: file path or exkaldi DecisionTree object.
			<topoFile>: file path.
			<numgauss>: int value.
			<feat>: exkaldi feature object.
			<treeStatsFile>: tree statistics file.
		
		Parallel Args:
			Null
		'''
		declare.is_file("topoFile", topoFile)
		declare.is_potential_tree("tree", tree)

		with FileHandleManager() as fhm:
			
			if feat is not None:

				if not isinstance(tree, str):
					treeTemp = fhm.create("wb+", suffix=".tree")
					tree.save(treeTemp)
					treeBackup = tree
					tree = treeTemp.name
				else:
					treeBackup = load_tree(tree)

				assert treeStatsFile is None, "Initialize model from example feature, so tree statistics file is invalid."
				declare.is_feature("feat", feat)
				feat = feat.subset(nRandom=10)
				if type_name(feat) == "NumpyFeature":
					feat = feat.to_bytes()
				elif type_name(feat) == "ArkIndexTable":
					feat = feat.read_record(arkType="feat")				

				cmd = f"gmm-init-model-flat {tree} {topoFile} - ark:- "
				out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=feat.data)
			else:
				declare.is_file("treeStatsFile", treeStatsFile)

				if isinstance(tree, str):
					tree = load_tree(tree)
				treeBackup = tree

				cmd = f"gmm-init-model - {treeStatsFile} {topoFile} -"
				out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=tree.data)

			if (isinstance(cod, int) and cod != 0) or len(out) == 0:
				print(err.decode())
				raise KaldiProcessError("Failed to initialize model.") 

			self.reset_data(out)
			self.__tree = treeBackup

	@property
	def tree(self):
		return self.__tree

	def train(self, feat, transcription, LFile, tree, tempDir, initialAli=None, 
				ldaMatFile=None, fmllrTransMat=None, spk2utt=None, utt2spk=None,
				numIters=40, maxIterInc=30, totgauss=1000, fmllrSilWt=0.0,
				realignIter=None, mlltIter=None, fmllrIter=None,
				transitionScale=1.0, acousticScale=0.1, selfloopScale=0.1,
				beam=10, retryBeam=40,
				boostSilence=1.0, careful=False, power=0.25, minGaussianOccupancy=10, lexicons=None):
		'''
		This is a high-level API to train the HMM-GMM model.
		
		Share Args:
			<LFile>: L.fst file path.
			<tree>: file path or exkaldi DecisionTree object.
			<tempDir>: A directory to save intermidiate files.
			<ldaMatFile>: If not None, do lda_mllt training.
			<fmllrTransMat>.
			<spk2utt>.
			<utt2spk>.
			<numIters>: Int value, the max iteration times.
			<maxIterInc>: Int value, increase numbers of gaussian functions when iter is smaller than <numIters>.
			<totgauss>: Int value, the rough target numbers of gaussian functions.
			<fmllrSilWt>.
			<realignIter>: None or list or tuple, the iter to realign.
			<mlltIter>.
			<transitionScale>:.
			<acousticScale>:.
			<selfloopScale>:.
			<initialBeam>:.
			<beam>:.
			<retryBeam>:.
			<boostSilence>.
			<careful>.
			<power>.
			<minGaussianOccupancy>.
			<lexicons>.
		
		Parallel Args:
			<feat>: exkaldi feature or index table object.
			<transcription>: exkaldi transcription object or file name (text format).
			<initialAli>: exakldi alignment or index table object.

		'''
		declare.not_void(type_name(self), self)
		
		if realignIter is not None:
			declare.is_classes("realignIter", realignIter, (list,tuple) )
		if mlltIter is not None:
			declare.is_classes("mlltIter", mlltIter, (list,tuple) )
		if fmllrIter is not None:
			declare.is_classes("fmllrIter", fmllrIter, (list,tuple) )

		if lexicons is None:
			assert self.lex is not None, "No <lexicons> avaliable defaultly, so provide it please."
			lexicons = self.lex
		else:
			declare.is_lexicon_bank("lexicons", lexicons)

		if ldaMatFile is not None:
			declare.is_file("ldaMatFile", ldaMatFile)
			print("Do LDA + MLLT training.")
			assert fmllrTransMat is None, "SAT training is not expected now."
			trainFeat = transform_feat(feat, ldaMatFile, outFile=os.path.join(tempDir,"lda_feat.ark"))
		elif fmllrTransMat is not None:
			print("Do SAT. Transform to fMLLR feature. <spk2utt> and <utt2spk> files are expected on this case.")
			declare.is_potential_list_table("spk2utt", spk2utt)
			declare.is_potential_list_table("utt2spk", utt2spk)
			trainFeat = use_fmllr(feat, fmllrTransMat, utt2spk, outFile=os.path.join(tempDir,"fmllr_feat.ark"))
		else:
			trainFeat = feat

		exNumgauss = self.info.gaussians
		declare.larger("total number of gaussian", totgauss, "current number", exNumgauss)

		print("Start to train triphone model.")
		make_dependent_dirs(tempDir, pathIsFile=False)

		starttime = datetime.datetime.now().strftime("%Y/%m/%d-%H:%M:%S")
		print(f"Start Time: {starttime}")

		print("Convert transcription to int value format.")
		if isinstance(transcription, (list,tuple)):
			for i, trans in enumerate(transcription):
				transcription[i] = transcription_to_int(trans, symbolTable=lexicons, unkSymbol=lexicons("oov"))
				transcription[i].save( os.path.join(tempDir, f"{i}_train_text.int") )
		else:
			transcription = transcription_to_int(transcription, symbolTable=lexicons, unkSymbol=lexicons("oov"))
			transcription.save( os.path.join(tempDir, f"train_text.int") )			

		print('Compiling training graph.')
		trainGraphFile = self.compile_train_graph(tree=tree, transcription=transcription, LFile=LFile, 
													outFile=os.path.join(tempDir,"train_graph"), lexicons=lexicons,
												)

		declare.is_positive_int("maxIterInc", maxIterInc)
		incgauss = (totgauss - exNumgauss)//maxIterInc

		statsFile = os.path.join(tempDir,"gmmStats.acc")
		for i in range(1, numIters+1, 1):
			
			print(f"Iter >> {i}")
			iterStartTime = time.time()
			# Align
			if  i == 1:
				if initialAli is None:
					print("Aligning data")
					ali = self.align(trainFeat, trainGraphFile, transitionScale, acousticScale, selfloopScale, 
										beam, retryBeam, boostSilence, careful, lexicons=lexicons,
										outFile=os.path.join(tempDir,"train.ali"),
									)
				else:
					print("Use the provided initial alignment")
					ali = initialAli
			elif (realignIter is None) or (i in realignIter):
				print("Aligning data")
				del ali
				ali = self.align(trainFeat, trainGraphFile, transitionScale, acousticScale, selfloopScale, 
									beam, retryBeam, boostSilence, careful, lexicons=lexicons,
									outFile=os.path.join(tempDir,"train.ali"),
								)
			else:
				print("Skip aligning")
			
			if ldaMatFile is not None:
				if mlltIter is None or (i in mlltIter):
					print("Accumulate MLLT statistics")
					accFile = accumulate_MLLT_stats(ali, lexicons, self, trainFeat, outFile=os.path.join(tempDir,"mllt.acc"))
					print("Estimate MLLT matrix")
					matFile = estimate_MLLT_matrix(accFile, outFile=os.path.join(tempDir,"mllt.mat"))
					print("Transform GMM means")
					self.transform_gmm_means(matFile)
					print("Compose new LDA-MLLT transform matrix")
					newTransMat = compose_transform_matrixs(ldaMatFile, matFile, outFile=os.path.join(tempDir,"trans.mat"))
					print("Transform feature")
					trainFeat = transform_feat(feat,newTransMat,outFile=os.path.join(tempDir,"lda_feat.ark"))
					ldaMatFile = newTransMat
				else:
					print("Skip tansform feature")
			elif fmllrTransMat is not None:
				if fmllrIter is None or (i in fmllrIter):
					print("Estimate fMLLR matrix")
					# If used parallel process, merge ali and feature
					if isinstance(ali,list):
						tempAli = merge_archieves(ali)
						tempFeat = merge_archieves(trainFeat)
						parallel = len(ali)
					else:
						tempAli = ali
						tempFeat = trainFeat
						parallel = 1
					fmllrTransMat = estimate_fMLLR_matrix(
														aliOrLat = tempAli, 
														lexicons = lexicons, 
														aliHmm = self, 
														feat = tempFeat, 
														spk2utt = spk2utt,
														silenceWeight = fmllrSilWt,
														outFile=os.path.join(tempDir,"trans.ark"),
													)
					# Then splice it.
					if parallel > 1:
						tempfmllrTrans = []
						for i in feat:
							spks = utt_to_spk(i.utts, utt2spk=utt2spk)
							tempfmllrTrans.append( fmllrTransMat.subset(uttIDs=spks) )
						fmllrTransMat = tempfmllrTrans
					print("Transform feature")
					trainFeat = use_fmllr(feat, fmllrTransMat, utt2spk, outFile=os.path.join(tempDir,"fmllr_feat.ark"))
				else:
					print("Skip tansform feature")

			print("Accumulate GMM statistics")
			_statsFiles = self.accumulate_stats(trainFeat, alignment=ali, outFile=statsFile)
			if isinstance(_statsFiles, list):
				sum_gmm_stats(_statsFiles, statsFile)

			print("Update GMM parameter")
			self.update(statsFile, exNumgauss, power, minGaussianOccupancy)
			os.remove(statsFile)

			exNumgauss += incgauss
			iterTimeCost = time.time() - iterStartTime
			print(f"Used time: {iterTimeCost:.4f} seconds")
		
		modeLFile = os.path.join(tempDir,"final.mdl")
		self.save(modeLFile)

		print('Align last time with final model')
		del ali
		for tempAliFile in list_files(os.path.join(tempDir,f"*train.ali")):
			os.remove(tempAliFile)

		ali = self.align(trainFeat, trainGraphFile, transitionScale, acousticScale, 
							selfloopScale, beam, retryBeam, boostSilence, careful,
							outFile=os.path.join(tempDir,f"final.ali"),
						)
		if isinstance(ali, list):
			ali = merge_archieves(ali)

		del self.__tree
		self.__tree = tree

		print('Done to train the triphone model')
		print(f"Saved Final Model: {modeLFile}")
		print(f"Saved Alignment: ", ",".join(list_files(os.path.join(tempDir,f"*final.ali"))) )
		if ldaMatFile is not None:
			print(f"Saved Feature Transform Matrix: {newTransMat}")
		elif fmllrTransMat is not None:
			print(f"Saved Feature Transform Matrix: ", ",".join(list_files(os.path.join(tempDir,f"*trans.ark"))) )
		endtime = datetime.datetime.now().strftime("%Y/%m/%d-%H:%M:%S")
		print(f"End Time: {endtime}")

		return ali

def load_tree(target, name="tree", lexicons=None):
	'''
	Reload a tree from file.
	The original data will be discarded.

	Args:
		<target>: file name.
		<name>: a string.
	
	Return:
		A exkaldi DecisionTree object.
	'''
	declare.is_file("target", target)
	
	cmd = f"tree-info {target}"
	out, err, cod = run_shell_command(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	if isinstance(cod,int) and cod != 0:
		print(err.decode())
		raise WrongDataFormat("Failed to load tree.")
	else:
		out = out.decode().strip().split("\n")
		contextWidth = int(out[1].strip().split()[-1])
		centralPosition = int(out[2].strip().split()[-1])		
		with open(target, "rb") as fr:
			data = fr.read()
		return DecisionTree(data=data, contextWidth=contextWidth, centralPosition=centralPosition, name=name, lexicons=lexicons)

def load_hmm(target, hmmType="triphone", name="hmm", lexicons=None):
	'''
	Reload a HMM-GMM model from file.
	The original data will be discarded.

	Args:
		<target>: file name.
		<hmmType>: "monophone" or "triphone".
		<name>: a string.
		<lexicons>: None or exkaldi LexiconBank object.

	Return:
		A MonophoneHMM object if hmmType is "monophone", else return TriphoneHMM object.
	'''
	declare.is_file("target", target)
	declare.is_instances("hmmType", hmmType, ["monophone", "triphone"])
	
	cmd = f"gmm-info {target}"
	out, err, cod = run_shell_command(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	if isinstance(cod,int) and cod != 0:
		print(err.decode())
		raise WrongDataFormat("Failed to load HMM-GMM model.")
	else:
		with open(target, "rb") as fr:
			data = fr.read()
		if hmmType == "monophone":
			hmm = MonophoneHMM(name=name, lexicons=lexicons)
		else:
			hmm = TriphoneHMM(name=name, lexicons=lexicons)
		hmm.reset_data(data)
		return hmm

def __sum_statistics_files(tool, statsFiles, outFile):
	'''
	Sum GMM or tree statistics.

	Args:
		<statsFiles>: a string, list or tuple of mutiple file paths.
		<outFile>: output file path.
	Return:
	 	file path.
	'''	
	declare.kaldi_existed()
	declare.is_valid_file_name("outFile", outFile)

	statsFiles = list_files(statsFiles)
	declare.members_are_files("statsFiles", statsFiles)
	statsFiles = " ".join(statsFiles)

	cmd = f'{tool} {outFile} {statsFiles}'

	out, err, cod = run_shell_command(cmd, stderr=subprocess.PIPE)

	if cod != 0:
		print(err.decode())
		raise KaldiProcessError(f"Failed to sum GMM statistics.")
	else:
		return os.path.abspath(outFile)

def sum_gmm_stats(statsFiles, outFile):
	'''
	Sum GMM statistics.

	Args:
		<statsFiles>: a string, list or tuple of mutiple file paths.
		<outFile>: output file path.
	Return:
	 	absolute path of accumulated file.
	'''
	tool = "gmm-sum-accs"
	return __sum_statistics_files(tool, statsFiles, outFile)

def sum_tree_stats(statsFiles, outFile):
	'''
	Sum tree statistics.

	Args:
		<statsFiles>: a string, list or tuple of mutiple file paths.
		<outFile>: output file path.
	Return:
	 	absolute path of accumulated file.
	'''
	tool = "sum-tree-stats"
	return __sum_statistics_files(tool, statsFiles, outFile)

def make_toponology(lexicons, outFile, numNonsilStates=3, numSilStates=5):
	'''
	Make toponology file.

	Args:
		<lexicons>: an LexiconBank object.
		<outFile>: output file path.
		<numNonsilStates>: the number of non-silence states.
		<numSilStates>: the number of silence states.
	Return:
	 	absolute path of generated file.
	'''
	declare.is_lexicon_bank("lexicons", lexicons)
	declare.is_valid_file_name("outFile", outFile)
	declare.is_positive_int("numNonsilStates", numNonsilStates)
	declare.is_positive_int("numSilStates", numSilStates)
	declare.kaldi_existed()

	nonsilPhones = lexicons("nonsilence", returnInt=True)
	silPhones = lexicons("silence", returnInt=True)

	nonsilphonelist = ":".join(nonsilPhones)
	silphonelist = ":".join(silPhones)

	make_dependent_dirs(outFile, pathIsFile=True)

	cmd = os.path.join(ExkaldiInfo.KALDI_ROOT,"egs","wsj","s5","utils","gen_topo.pl")
	cmd += f" {numNonsilStates} {numSilStates} {nonsilphonelist} {silphonelist} > {outFile}"
	out, err, cod = run_shell_command(cmd, stderr=subprocess.PIPE)

	if (isinstance(cod,int) and cod != 0) or (not os.path.isfile(outFile)) or (os.path.getsize(outFile) == 0):
		print(err.decode())
		if os.path.isfile(outFile):
			os.remove(outFile)
		raise KaldiProcessError("Failed to generate toponology file.")
	else:
		return os.path.abspath(outFile)

def convert_alignment(alignment, originHmm, targetHmm, tree, outFile=None):
	'''
	Convert alignment.

	Share Args:
		Null

	Parallel Args:
		<alignment>: file path or exkaldi transition-ID alignment object.
		<originHmm>: file path or exkaldi HMM object.
		<targetHmm>: file path or exkaldi HMM object.
		<tree>: file path or exkaldi DecisionTree object.
		<outFile>: file name.

	Return:
	 	exkaldi alignment or index table object.
	'''

	with FileHandleManager() as fhm:

		if isinstance(originHmm,BaseHMM):
			omTemp = fhm.create("wb+", suffix=".mdl")
			originHmm.save(omTemp)
			originHmm = omTemp.name

		if isinstance(targetHmm,BaseHMM):
			tmTemp = fhm.create("wb+", suffix=".mdl")
			targetHmm.save(tmTemp)
			targetHmm = tmTemp.name	
		
		if isinstance(tree, DecisionTree):
			treeTemp = fhm.create("wb+", suffix=".tree")
			tree.save(treeTemp)
			tree = treeTemp.name
		
		alignments, originHmms, targetHmms, trees, outFiles = check_mutiple_resources(alignment, originHmm, targetHmm, tree, outFile=outFile)
		names = []
		for alignment, originHmm, targetHmm, tree in zip(alignments, originHmms, targetHmms, trees):
			# check alignment
			declare.is_alignment("alignment", alignment)
			# check original model
			declare.is_potential_hmm("originHmm", originHmm)
			# check original model
			declare.is_potential_hmm("targetHmm", targetHmm)
			# check tree
			declare.is_potential_tree("tree", tree)	
			names.append( f"convert{alignment.name}" )			

		# define command pattern
		cmdPattern = "convert-ali {originHMM} {targetHmm} {tree} {alignment} ark:{outFile}"
		# define resources
		resources = {"alignment":alignments, "originHMM":originHmms, "targetHmm":targetHmms, "tree":trees, "outFile":outFiles}
		# run
		return run_kaldi_commands_parallel(resources, cmdPattern, generateArchieve="ali", archieveNames=names)

def transcription_to_int(transcription, symbolTable, unkSymbol):
	'''
	Transform text format transcrption to int format file.
	'''
	declare.is_potential_transcription("transcription", transcription)
	
	if isinstance(transcription, str):
		transcription = load_transcription(transcription)

	if isinstance(symbolTable, str):
		declare.is_file("symbolTable", symbolTable)
		symbolTable = load_list_table(symbolTable)
	elif isinstance(symbolTable, ListTable):
		pass
	elif type_name(symbolTable) == "LexiconBank":
		symbolTable = symbolTable("words")
	else:
		raise UnsupportedType(f"<symbolTable> should be a file, exkaldi ListTable or LexiconBank object but got {type_name(symbolTable)}.")	
	
	for utt, text in transcription.items():
		text = text.split()
		for index, word in enumerate(text):
			try:
				text[index] = str(symbolTable[word])
			except KeyError:
				try:
					text[index] = str(symbolTable[unkSymbol])
				except KeyError:
					raise WrongDataFormat(f"Word symbol table miss unknown-map symbol: {unkSymbol}")
	
		transcription[utt] = " ".join(text)
	
	return transcription

def transcription_from_int(transcription, symbolTable):
	'''
	Transform int format transcrption to text format file.
	'''
	declare.is_potential_transcription("transcription", transcription)
	
	if isinstance(transcription, str):
		transcription = load_transcription(transcription)

	if isinstance(symbolTable, str):
		declare.is_file("symbolTable", symbolTable)
		symbolWordTable = load_list_table(symbolTable).reverse()
	elif isinstance(symbolTable, ListTable):
		symbolWordTable = symbolTable.reverse()
	elif type_name(symbolTable) == "LexiconBank":
		symbolWordTable = symbolTable("words").reverse()
	else:
		raise UnsupportedType(f"<symbolTable> should be a file, exkaldi ListTable or LexiconBank object but got {type_name(symbolTable)}.")	
	
	for utt, text in transcription.items():
		text = text.split()
		for index, word in enumerate(text):
			try:
				text[index] = str(symbolWordTable[word])
			except KeyError:
				try:
					text[index] = str(symbolWordTable[int(word)])
				except KeyError:
					raise WrongDataFormat(f"Word symbol table miss symbol: {word}")
				except ValueError as e:
					print("Transcription may conlude non-int value.")
					raise e
					
		transcription[utt] = " ".join(text)
	
	return transcription

def __accumulate_LDA_MLLT_statistics(baseCmd, alignment, lexicons, hmm, feat, outFile, silenceWeight=0.0, randPrune=4):
	'''
	Accumulate LDA or MLLT statistics.
	'''
	declare.is_potential_hmm("hmm", hmm)
	declare.is_lexicon_bank("lexicons", lexicons)
	declare.is_non_negative("silenceWeight", silenceWeight)
	declare.is_non_negative_int("randPrune", randPrune)

	with FileHandleManager() as fhm:

		if isinstance(hmm, BaseHMM):
			hmmTemp = fhm.create("wb+", suffix=".mdl")
			hmm.save(hmmTemp)
			hmm = hmmTemp.name

		alignments,hmms,feats,silenceWeights,randPrunes,outFiles = check_mutiple_resources( alignment, hmm, feat, silenceWeight,
																							randPrune, outFile=outFile)

		for alignment,hmm,feat,silenceWeight,randPrune,outFile in zip(alignments, hmms, feats, silenceWeights, randPrunes, outFiles):
			declare.is_alignment("alignment", alignment)
			declare.is_feature("feat", feat)
			assert outFile != "-", f"<outFile> is necessary."

		# define command pattern
		silphonelist = ":".join(lexicons("silence", True))
		cmdPattern = "ali-to-post {alignment} ark:- | "
		cmdPattern += f"weight-silence-post {silenceWeight} {silphonelist} "+"{model} ark:- ark:- | "
		cmdPattern += f"{baseCmd} --rand-prune={randPrune} "
		cmdPattern += "{model} {feat} ark:- {outFile}"
		# define resources
		resources = {"alignment":alignments, "model":hmms, "feat":feats, "outFile":outFiles}
		# run
		run_kaldi_commands_parallel(resources, cmdPattern)

		if len(outFiles) == 1:
			return outFiles[0]
		else:
			return outFiles

def accumulate_LDA_stats(alignment, lexicons, hmm, feat, outFile, silenceWeight=0.0, randPrune=4):
	'''
	Acumulate LDA statistics to estimate LDA tansform matrix.

	Share Args:
		<lexicons>: exkaldi lexicons bank object.
		<hmm>: file name or exkaldi HMM object.
		<silenceWeight>.
		<randPrune>.
	
	Parallel Args:
		<alignment>: exkaldi alignment or index table object.
		<feat>: exkaldi feature or index object.
		<outFile>: output file name.
	'''
	Kalditool = 'acc-lda'

	return __accumulate_LDA_MLLT_statistics(Kalditool, alignment, lexicons, hmm, feat, outFile, silenceWeight, randPrune)

def accumulate_MLLT_stats(alignment, lexicons, hmm, feat, outFile, silenceWeight=0.0, randPrune=4):
	'''
	Acumulate MLLT statistics to estimate LDA+MLLT tansform matrix.
	
	Share Args:
		<lexicons>: exkaldi lexicons bank object.
		<hmm>: file name or exkaldi HMM object.
		<silenceWeight>.
		<randPrune>.
	
	Parallel Args:
		<alignment>: exkaldi alignment or index table object.
		<feat>: exkaldi feature or index object.
		<outFile>: output file name.
	'''
	Kalditool = 'gmm-acc-mllt'

	return __accumulate_LDA_MLLT_statistics(Kalditool, alignment, lexicons, hmm, feat, outFile, silenceWeight, randPrune)

def estimate_LDA_matrix(statsFiles, targetDim, outFile):
	'''
	Estimate the LDA transform matrix from LDA statistics.
	
	Share Args:
		<statsFiles>: str or list ot tuple of file paths.
		<targetDim>: int value.
		<outFile>: file name.
	
	Parallel Args:
		Null
	'''
	declare.kaldi_existed()
	declare.is_positive_int("targetDim", targetDim)
	declare.is_valid_file_name("outFile", outFile)

	statsFiles = list_files(statsFiles)
	declare.members_are_files("statsFiles", statsFiles)
	statsFiles = " ".join(statsFiles)

	cmd = f'est-lda --dim={targetDim} {outFile} {statsFiles}'
	out,err,cod = run_shell_command(cmd, stderr=subprocess.PIPE)
	if cod != 0:
		print(err.decode())
		raise KaldiProcessError("Failed to estimate LDA matrix.")
	else:
		return outFile

def estimate_MLLT_matrix(statsFiles, outFile):
	'''
	Estimate the MLLT transform matrix from MLLT statistics.
	
	Share Args:
		<statsFiles>: str or list ot tuple of file paths.
		<outFile>: file name.
	
	Parallel Args:
		Null
	'''
	declare.kaldi_existed()
	declare.is_valid_file_name("outFile", outFile)

	statsFiles = list_files(statsFiles)
	declare.members_are_files("statsFiles", statsFiles)
	statsFiles = " ".join(statsFiles)

	cmd = f'est-mllt {outFile} {statsFiles}'
	out,err,cod = run_shell_command(cmd, stderr=subprocess.PIPE)
	if cod != 0:
		print(err.decode())
		raise KaldiProcessError("Failed to estimate MLLT matrix.")
	else:
		return outFile

def compose_transform_matrixs(matA, matB, bIsAffine=False, utt2spk=None, outFile=None):
	'''
	The dot operator between two matrixes.

	Share Args:
		<matA>: matrix file or exkaldi fMLLR transform matrx object.
		<matB>: matrix file or exkaldi fMLLR transform matrx object.
		<bIsAffine>.
		<utt2spk>.
		<outFile>: None or file name.
	
	Parallel Args:
		Null.
	
	Return:
		If <outFile> is not None, return the absolute path of output file.
		Else, return Numpy Matrix object or BytesFmllrMatrix object.
	'''
	# prepare command pattern and resources
	cmdPattern = 'compose-transforms '
	resources = {"matB":None, "matA":None}

	if utt2spk is not None:
		declare.is_potential_list_table("utt2spk", utt2spk)
		cmdPattern += '--utt2spk=ark:{utt2spk} '
		resources["utt2spk"] = [utt2spk,]

	if bIsAffine:
		cmdPattern += '--b-is-affine=true '	

	bothAreFiles = False
	if isinstance(matB, str):
		try:
			load_mat(matB)
		except Exception:
			assert UnsupportedType(f"<matB> is not a valid matrix file: {matB}.")
		bothAreFiles = True
		name = ["",]
	else:
		declare.is_fmllr_matrix("matB", matB)
		name = [matB.name,]

	resources["matB"] = [matB,]
	cmdPattern += "{matB} "

	if isinstance(matA, str):
		try:
			load_mat(matA)
		except Exception:
			assert UnsupportedType(f"<matA> is not a valid matrix file: {matA}.")
		bothAreFiles = bothAreFiles and True
		name.append("")
	else:
		declare.is_fmllr_matrix("matA", matA)		
		name.append(matA.name)

	resources["matA"] = [matA,]
	cmdPattern += "{matA} "

	if outFile is None:
		assert bothAreFiles is False, "When compose two matrix files, <outFile> is necessary."
		resources["outFile"] = ["-",]
		cmdPattern += "ark:{outFile} "
	else:
		declare.is_valid_file_name("outFile", outFile)
		resources["outFile"] = [outFile,]
		if bothAreFiles is True:
			cmdPattern += "{outFile} "
		else:
			cmdPattern += "ark:{outFile} "
	# run
	results = run_kaldi_commands_parallel(resources, cmdPattern)
	if bothAreFiles:
		return outFile
	else:
		name = "compose({})".format(",".join(name))
		if outFile is None:
			return BytesFmllrMatrix(results[2], name=name)
		else:
			return load_index_table(outFile, name=name)

def load_mat(matrixFile):
	'''
	Read a matrix from file.

	Args:
		<matrixFile>: matrix file path.
	
	Return:
		Numpy Matrix Object.
	'''
	declare.is_file("matrixFile", matrixFile)
	declare.kaldi_existed()

	cmd = f'copy-matrix --binary=False {matrixFile} -'
	out, err, cod = run_shell_command(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	
	if cod != 0:
		print(err.decode())
		raise KaldiProcessError("Failed to compose matrixes.")
	else:
		out = out.decode().strip().strip("[]").strip().split("\n")
		results = []
		for line in out:
			results.append( np.asarray(line.strip().split(), dtype="float32") )
		return np.matrix(results).T

def estimate_fMLLR_matrix(aliOrLat, lexicons, aliHmm, feat, spk2utt, adaHmm=None, silenceWeight=0.0, acwt=1.0, name="fmllrMatrix", outFile=None):
	'''
	Estimate fMLLR transform matrix.

	Share Args:
		<lexicons>: exkaldi LexiconBank object.
		<aliHmm>: file or exkaldi HMM object.
		<adaHmm>: file or exkaldi HMM object.
		<silenceWeight>.
		<acwt>.

	Parallel Args:
		<aliOrLat>: exkaldi Transition alignment object or Lattice object.
		<feat>: exkaldi feature object.
		<spk2utt>.
		<name>.
		<outFile>: output file name.
	
	Return:
		exkaldi fMLLR transform matrix or index table object.
	'''
	declare.kaldi_existed()
	declare.is_potential_hmm("aliHmm", aliHmm)
	if adaHmm is not None:
		declare.is_potential_hmm("adaHmm", adaHmm)
	declare.is_non_negative("silenceWeight", silenceWeight)
	declare.is_non_negative("acwt", acwt)

	with FileHandleManager() as fhm:
		# modify two hmm model if they are public resources
		if isinstance(aliHmm, BaseHMM):
			aliHmmTemp = fhm.create("wb+",suffix=".mdl")
			aliHmm.save(aliHmmTemp)
			aliHmm = aliHmmTemp.name

		if adaHmm is not None and isinstance(adaHmm, BaseHMM):
			adaHmmTemp = fhm.create("wb+",suffix=".mdl")
			adaHmm.save(adaHmmTemp)
			adaHmm = adaHmmTemp.name
			adaptNewModel = True
		else:
			adaHmm = aliHmm
			adaptNewModel = False
		
		aliOrLats, feats, spk2utts, names, outFiles = check_mutiple_resources(aliOrLat, feat, spk2utt, name, outFile=outFile)
	
		fromAlignment = True
		for aliOrLat, feat, spk2utt, name in zip(aliOrLats, feats, spk2utts, names):
			# check alignment or lattice
			if type_name(aliOrLat) in ["ArkIndexTable","BytesAlignmentTrans","NumpyAlignmentTrans"]:
				fromAlignment = True
			elif type_name(aliOrLat) in ["Lattice","str"]:
				fromAlignment = False
			else:
				raise UnsupportedType(f"<aliOrLat> should exkaldi alignment object, index table, lattice object or lattice file but got: {type_name(aliOrLat)}.")
			# check feature
			declare.is_feature("feat", feat)
			# check utt2spk
			declare.is_potential_list_table("spk2utt", spk2utt)

		# define command pattern
		silphonelist = ":".join(lexicons("silence", True))
		if fromAlignment:
			cmdPattern = "ali-to-post {aliOrLat} ark:- | "
		else:
			cmdPattern = f"lattice-to-post --acoustic-scale={acwt} "+"ark:{aliOrLat} ark:- | "
		cmdPattern += f"weight-silence-post {silenceWeight} {silphonelist} {aliHmm} ark:- ark:- | "
		if adaptNewModel is True:
			cmdPattern += f"gmm-post-to-gpost {aliHmm}"+" ark:{feat} ark:- ark:- | gmm-est-fmllr-gpost "
		else:
			cmdPattern += "gmm-est-fmllr "

		cmdPattern += "--fmllr-update-type=full --spk2utt=ark:{spk2utt} "+f"{adaHmm} "+"{feat} ark:- ark:{outFile}"
		# define resources
		resources = {"aliOrLat":aliOrLats, "feat":feats, "spk2utt":spk2utts, "outFile":outFiles}
		# run 
		return run_kaldi_commands_parallel(resources, cmdPattern, generateArchieve="fmllrMat", archieveNames=names)


			