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
import tempfile
import copy
import time, datetime
from collections import namedtuple

from exkaldi.core.achivements import BytesAchievement, Transcription, ListTable, BytesAlignmentTrans
from exkaldi.core.load import load_ali
from exkaldi.utils import run_shell_command, check_config, make_dependent_dirs, type_name
from exkaldi.version import version as ExkaldiInfo
from exkaldi.version import WrongPath, KaldiProcessError, UnsupportedType, WrongOperation, WrongDataFormat

class DecisionTree(BytesAchievement):

	def __init__(self, data=b"", contextWidth=3, centralPosition=1, lexicons=None, name="tree"):
		super().__init__(data, name)

		if not lexicons is None:
			assert type_name(lexicons) == "LexiconBank", f"<lexicons> should be an exkaldi LexiconBank object but got {(type_name(lexicons))}."

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

		Args:
			<feat>: exkaldi feature object.
			<hmm>: exkaldi TriphoneHMM object or model file path.
			<alignment>: exkaldi alignment object and file path.
			<outFile>: file name.
			<lexicons>: None. If no any lexicons provided in DecisionTree, this is expected.
						In this step, we will use "context_indep" lexicon.
		
		Return:
			Absolute path of out file.
		'''
		if lexicons is None:
			assert self.lex is not None, "No <lexicons> avaliable defaultly, so provide it please."
			lexicons = self.lex
		else:
			assert type_name(lexicons) == "LexiconBank", f"<lexicons> should be an exkaldi LexiconBank object but got {(type_name(lexicons))}."

		assert isinstance(outFile, str), "<outFile> should be a string."
		make_dependent_dirs(outFile, pathIsFile=True)

		if type_name(feat) == "BytesFeature":
			feat = feat.sort(by="utt")
		elif type_name(feat) == "NumpyFeature":
			feat = feat.sort(by="utt").to_bytes()
		else:
			raise UnsupportedType(f"Expected exkaldi feature object but got {type_name(feat)}.")
				
		if isinstance(alignment, str):
			alignment = load_ali(alignment)
		
		if isinstance(alignment, BytesAlignmentTrans):
			alignment = alignment.sort(by="utt")
		elif isinstance(alignment, NumpyAlignmentTrans):
			alignment = alignment.sort(by="utt").to_bytes()
		else:
			raise UnsupportedType("<alignment> should be file name or exkaldi alignment object.")		

		ciphones = ":".join(lexicons("context_indep", True))

		modelTemp = tempfile.NamedTemporaryFile("wb+", suffix=".mdl")
		aliTemp = tempfile.NamedTemporaryFile("wb+", suffix=".ali")

		try:
			if isinstance(hmm, BaseHMM):
				modelTemp.write(hmm.data)
				modelTemp.seek(0)
				hmmFile = modelTemp.name
			elif isinstance(hmm, str):
				hmmFile = hmm
			else:
				raise UnsupportedType("<hmm> should be file name or exkaldi HMM object.")

			aliTemp.write(alignment.data)
			aliTemp.seek(0)

			cmd = f'acc-tree-stats --context-width={self.contextWidth} --central-position={self.centralPosition} --ci-phones={ciphones} '
			cmd += f'{hmmFile} ark:- ark:{aliTemp.name} {outFile}'

			out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, inputs=feat.data)

			if (isinstance(cod,int) and cod != 0) or (not os.path.isfile(outFile)) or (os.path.getsize(outFile)==0):
				print(err.decode())
				raise KaldiProcessError("Failed to initialize mono model.")
			
			return os.path.abspath(outFile)
	
		finally:
			modelTemp.close()
			aliTemp.close()
		
	def compile_questions(self, treeStatsFile, topoFile, outFile, lexicons=None):
		'''
		Compile questions.

		Args:
			<treeStatsFile>: file path.
			<topoFile>: topo file path.
			<outFile>: file name.
			<lexicons>: None. If no any lexicons provided in DecisionTree, this is expected.
						In this step, we will use "sets" and "extra_questions" lexicon.
		
		Return:
			Absolute path of out file.
		'''
		if lexicons is None:
			assert self.lex is not None, "No <lexicons> avaliable defaultly, so provide it please."
			lexicons = self.lex
		else:
			assert type_name(lexicons) == "LexiconBank", f"<lexicons> should be an exkaldi LexiconBank object but got {(type_name(lexicons))}."
		assert isinstance(outFile, str), "<outFile> should be a string."
		make_dependent_dirs(outFile, pathIsFile=True)

		for x in [treeStatsFile, topoFile]:
			assert isinstance(x, str), f"Expected a path-like string but got {type_name(x)}."
			if not os.path.isfile(x):
				raise WrongPath(f"No such file:{x}.")
		
		setsTemp = tempfile.NamedTemporaryFile("w+", suffix="_sets.int", encoding="utf-8")

		try:
			lexicons.dump_dict(name="sets", outFile=setsTemp, dumpInt=True)

			cmd = f'cluster-phones --context-width={self.contextWidth} --central-position={self.centralPosition} '
			cmd += f'{treeStatsFile} {setsTemp.name} -'

			out, err, cod = run_shell_command(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

			if (isinstance(cod,int) and cod != 0):
				print(err.decode())
				raise KaldiProcessError("Failed to cluster phones.")

			extra = lexicons.dump_dict("extra_questions", None, True)

			questions = "\n".join([out.decode().strip(), extra.strip()])

			cmd = f'compile-questions --context-width={self.contextWidth} --central-position={self.centralPosition} '
			cmd += f'{topoFile} - {outFile}'

			out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, inputs=questions)

			if (isinstance(cod, int) and cod != 0) or (not os.path.isfile(outFile)) or os.path.getsize(outFile) == 0:
				print(err.decode())
				raise KaldiProcessError("Failed to compile questions.")
			
			return os.path.abspath(outFile)

		finally:
			setsTemp.close()

	def build(self, treeStatsFile, questionsFile, numleaves, topoFile, cluster_thresh=-1, lexicons=None):
		'''
		Build tree.

		Args:
			<treeStatsFile>: file path.
			<questionsFile>: file path.
			<numleaves>: target numbers of leaves.
			<topoFile>: topo file path.
			<lexicons>: None. If no any lexicons provided in DecisionTree, this is expected.
						In this step, we will use "roots" lexicon.
		
		Return:
			Absolute path of out file.
		'''
		if lexicons is None:
			assert self.lex is not None, "No <lexicons> avaliable defaultly, so provide it please."
			lexicons = self.lex
		else:
			assert type_name(lexicons) == "LexiconBank", f"<lexicons> should be an exkaldi LexiconBank object but got {(type_name(lexicons))}."

		assert isinstance(questionsFile, str), f"Expected a path-like string but got {type_name(questionsFile)}."
		if not os.path.isfile(questionsFile):
			raise WrongPath(f"No such file:{questionsFile}.")

		rootsTemp = tempfile.NamedTemporaryFile("w+", suffix=".int", encoding="utf-8")

		try:
			lexicons.dump_dict("roots", rootsTemp, True)

			cmd = f'build-tree --context-width={self.contextWidth} --central-position={self.centralPosition} '
			cmd += f'--verbose=1 --max-leaves={numleaves} --cluster-thresh={cluster_thresh} '
			cmd += f'{treeStatsFile} {rootsTemp.name} {questionsFile} {topoFile} -'

			out, err, cod = run_shell_command(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

			if (isinstance(cod,int) and cod !=0 ) or len(out) == 0:
				print(err.decode())
				raise KaldiProcessError("Failed to build tree.")
			else:
				self.reset_data(out)
				return self

		finally:
			rootsTemp.close()			

	def train(self, feat, hmm, alignment, topoFile, numleaves, tempDir, cluster_thresh=-1, lexicons=None):
		'''
		This is a hign-level API to build a decision tree.

		Args:
			<feat>: exkaldi feature object.
			<hmm>: file path or exkaldi HMM object.
			<alignment>: file path or exkaldi transition-ID Alignment object.
			<numleaves>: target numbers of leaves.
			<tempDir>: a temp directory to storage some intermidiate files.
			<lexicons>: None. If no any lexicons provided in DecisionTree, this is expected.
						In this step, we will use "roots" lexicon.
		'''
		if lexicons is None:
			assert self.lex is not None, "No <lexicons> avaliable defaultly, so provide it please."
			lexicons = self.lex
		else:
			assert type_name(lexicons) == "LexiconBank", f"<lexicons> should be an exkaldi LexiconBank object but got {(type_name(lexicons))}."

		print("Start to build decision tree.")
		make_dependent_dirs(tempDir, pathIsFile=False)

		starttime = datetime.datetime.now().strftime("%Y/%m/%d-%H:%M:%S")
		print(f"Start Time: {starttime}")

		print(">> Accumulate tree statistics")
		statsFile = os.path.join(tempDir,"treeStats.acc")
		self.accumulate_stats(feat, hmm, alignment, outFile=statsFile, lexicons=lexicons)

		print(">> Cluster phones and compile questions")
		questionsFile = os.path.join(tempDir, "questions.qst" )
		self.compile_questions(statsFile, topoFile, outFile=questionsFile, lexicons=lexicons)

		print(">> Build tree")
		self.build(statsFile, questionsFile, numleaves, topoFile, cluster_thresh)

		treeFile = os.path.join(tempDir,"tree")
		self.save(treeFile)

		print('Done to build the decision tree.')
		print(f"Saved Final Tree: {treeFile}")
		endtime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
		print(f"End Time: {endtime}")

	def save(self, outFile="tree"):
		'''
		Save tree to file.
		'''
		if isinstance(outFile, str):
			make_dependent_dirs(outFile, pathIsFile=True)
			with open(outFile,"wb") as fw:
				fw.write(self.data)
			return os.path.abspath(outFile)
		elif isinstance(outFile, tempfile._TemporaryFileWrapper):
			outFile.read()
			outFile.seek(0)
			outFile.write(self.data)
			outFile.seek(0)
		else:
			raise UnsupportedType("<outFile> is unavaliable file name")

	def load(self, target):
		'''
		Reload a tree from file.
		The original data will be discarded.

		Args:
			<target>: file name.
		'''
		assert isinstance(target,str), "<target> should be a file path."
		if not os.path.isfile(target):
			raise WrongPath(f"No such file: {target}.")
		
		cmd = f"tree-info {target}"
		out, err, cod = run_shell_command(cmd, stderr=subprocess.PIPE)
		if isinstance(cod,int) and cod != 0:
			print(err.decode())
			raise WrongDataFormat("Failed to load tree.")
		else:
			with open(target, "rb") as fr:
				data = fr.read()
			self.reset_data(data)
			return self

	@property
	def info(self):
		
		'''
		Get the information of tree.
		'''
		if self.is_void:
			raise WrongOperation("Tree is void.")
		
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

class BaseHMM(BytesAchievement):

	def __init__(self, data=b"", name="hmm", lexicons=None):
		super().__init__(data, name)
		if not lexicons is None:
			assert type_name(lexicons) == "LexiconBank", f"<lexicons> should be an exkaldi LexiconBank object but got {(type_name(lexicons))}."
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

		Args:
			<tree>: file path or exkaldi DecisionTree object.
			<transcription>: file path or exkaldi Transcription object with int format.
							Note that: int fotmat, not text format.
			<Lfile>: L.fst file path.
			<outFile>: graph output file path.
			<lexicons>: None. If no any lexicons provided in DecisionTree, this is expected.
						In this step, we will use "context_indep" lexicon.
		
		Return:
			Absolute path of output file.
		'''
		assert isinstance(outFile, str), "<outFile> should be string."

		if self.is_void:
			raise WrongOperation("Model is void.")

		if lexicons is None:
			assert self.lex is not None, "No <lexicons> avaliable defaultly, so provide it please."
			lexicons = self.lex
		else:
			assert type_name(lexicons) == "LexiconBank", f"<lexicons> should be an exkaldi LexiconBank object but got {(type_name(lexicons))}."

		ExkaldiInfo.vertify_kaldi_existed()

		disambigTemp = tempfile.NamedTemporaryFile("w+", suffix="_disambig.int", encoding="utf-8")
		treeTemp = tempfile.NamedTemporaryFile("wb+", suffix="_tree")
		transTemp = tempfile.NamedTemporaryFile("w+", suffix="_trans.txt", encoding="utf-8")

		try:
			if isinstance(tree, str):
				if not os.path.isfile(tree):
					raise WrongPath(f"No such file: {tree}.")
				treeFile = tree
			elif isinstance(tree, DecisionTree):
				tree.save(outFile=treeTemp)
				treeFile = treeTemp.name
			else:
				raise UnsupportedType(f"<tree> should be file path or exkaldi DecisionTree object.")

			if isinstance(transcription, str):
				if not os.path.isfile(transcription):
					raise WrongPath(f"No such file: {transcription}.")
				transcription = Transcription().load(transcription).sort()			
			elif isinstance(transcription, Transcription):
				transcription = transcription.sort()
			else:
				raise UnsupportedType(f"<transcription> should be file path or exkaldi Transcription object.")
			transcription.save(transTemp)
			transFile = transTemp.name

			lexicons.dump_dict(name="disambig", outFile=disambigTemp, dumpInt=True)

			make_dependent_dirs(outFile, pathIsFile=True)

			cmd = f'compile-train-graphs --read-disambig-syms={disambigTemp.name} '
			cmd += f'{treeFile} - {LFile} '
			cmd += f'ark:{transFile} ark:{outFile}'

			out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, inputs=self.data)

			if (isinstance(cod, int) and cod != 0) or (not os.path.isfile(outFile)) or (os.path.getsize(outFile)==0):
				print(err.decode())
				if os.path.isfile(outFile):
					os.remove(outFile)
				raise KaldiProcessError("Failed to construct train graph.")
			else:
				return os.path.abspath(outFile)
		finally:
			disambigTemp.close()
			treeTemp.close()
			transTemp.close()

	def update(self, gmmStatsFile, numgauss, power=0.25, minGaussianOccupancy=10):

		if self.is_void:
			raise WrongOperation("Model is void.")

		assert isinstance(gmmStatsFile, str), f"Expected a path-like string but got {type_name(gmmStatsFile)}."
		if not os.path.isfile(gmmStatsFile):
			raise WrongPath(f"No such file:{gmmStatsFile}.")
		gaussians = self.info.gaussians
		assert isinstance(numgauss, int) and numgauss >= gaussians, f"<numgauss> should be a int value and no less than current numbers {gaussians}."

		cmd = f'gmm-est --min-gaussian-occupancy={minGaussianOccupancy} --mix-up={numgauss} --power={power} '
		cmd += f'- {gmmStatsFile} -'

		out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=self.data)

		if (isinstance(cod,int) and cod != 0 ) or len(out) == 0:
			print(err.decode())
			raise KaldiProcessError("Failed to estimate new GMM parameters.")
		else:
			self.reset_data(out)
			return self

	def align(self, feat, trainGraphFile, transitionScale=1.0, acousticScale=0.1, 
				selfloopScale=0.1, beam=10, retry_beam=40, boost_silence=1.0, careful=False, name="ali", lexicons=None):
		'''
		Align acoustic feature with kaldi vertibi algorithm.
			<lexicons>: None. If no any lexicons provided in DecisionTree, this is expected.
						In this step, we will use "context_indep" lexicon.
		'''
		if self.is_void:
			raise WrongOperation("Model is void.")

		if type_name(feat) == "BytesFeature":
			feat = feat.sort(by="utt")
		elif type_name(feat) == "NumpyFeature":
			feat = feat.sort(by="utt").to_bytes()
		else:
			raise UnsupportedType(f"Expected exkaldi feature object but got {type_name(feat)}.")

		assert isinstance(trainGraphFile, str), f"Expected a path-like string but got {type_name(trainGraphsFile)}."
		if not os.path.isfile(trainGraphFile):
			raise WrongPath(f"No such file:{trainGraphFile}.")		

		if lexicons is None:
			assert self.lex is not None, "No <lexicons> avaliable defaultly, so provide it please."
			lexicons = self.lex
		else:
			assert type_name(lexicons) == "LexiconBank", f"<lexicons> should be an exkaldi LexiconBank object but got {(type_name(lexicons))}."

		optionSilence = ":".join(lexicons("optional_silence", True))

		model = tempfile.NamedTemporaryFile("wb+", suffix=".mdl")

		try:
			cmd = f'gmm-boost-silence --boost={boost_silence} {optionSilence} - {model.name}'
			out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=self.data)

			if (isinstance(cod,int) and cod != 0 ) or os.path.getsize(model.name) == 0:
				print(err.decode())
				raise KaldiProcessError("Generate new HMM defeated.")

			cmd = f'gmm-align-compiled --transition-scale={transitionScale} --acoustic-scale={acousticScale} --self-loop-scale={selfloopScale} '
			cmd += f'--beam={beam} --retry-beam={retry_beam} --careful={careful} {model.name} '
			cmd += f'ark:{trainGraphFile} '
			cmd += f'ark:- ark:-'

			out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=feat.data)

			if (isinstance(cod,int) and cod != 0):
				print(err.decode())
				raise KaldiProcessError("Failed to align feature.")
			
			return BytesAlignmentTrans(out, name=name)
		
		finally:
			model.close()
	
	def accumulate_stats(self, feat, alignment, outFile):
		'''
		Accumulate GMM statistics in order to update GMM parameters.

		Args:
			<feat>: exkaldi feature object.
			<alignment>: exkaldi transitionID alignment object or file path.
			<outFile>: file name.
		'''
		if self.is_void:
			raise WrongOperation("Model is void.")

		if type_name(feat) == "BytesFeature":
			feat = feat.sort(by="utt")
		elif type_name(feat) == "NumpyFeature":
			feat = feat.sort(by="utt").to_bytes()
		else:
			raise UnsupportedType(f"Expected exkaldi feature object but got {type_name(feat)}.")

		alignTemp = tempfile.NamedTemporaryFile("wb+", suffix=".ali")
		modelTemp = tempfile.NamedTemporaryFile("wb+", suffix=".mdl")

		try:
			if isinstance(alignment, str):
				assert os.path.isfile(alignment), f"No such file: {alignment}."
				alignment = load_ali(alignment).sort(by="utt")
			elif isinstance(alignment, BytesAlignmentTrans):
				alignment = alignment.sort(by="utt")
			elif type_name(alignment) == "NumpyAlignmentTrans":
				alignment = alignment.sort(by="utt").to_bytes()
			else:
				raise UnsupportedType(f"<Alignment> should be file name or exkaldi transition alignment object but got: {type_name(alignment)}.")
			
			alignTemp.write(alignment.data)
			alignTemp.seek(0)
			modelTemp.write(self.data)
			modelTemp.seek(0)

			cmd = f'gmm-acc-stats-ali {modelTemp.name} ark:- ark:{alignTemp.name} {outFile}'
			out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, inputs=feat.data)

			if (isinstance(cod,int) and cod != 0) or (not os.path.isfile(outFile)) or os.path.getsize(outFile) == 0:
				print(err.decode())
				raise KaldiProcessError("Failed to accumulate GMM statistics.")
			else:
				return os.path.abspath(outFile)
		finally:
			alignTemp.close()
			modelTemp.close()

	def align_equally(self, feat, trainGraphFile, name="equal_ali"):
		'''
		Align feature averagely.

		Args:
			<feat>: exkaldi feature object.
			<trainGraphFile>: file path.
			<name>: string.
		'''
		if self.is_void:
			raise WrongOperation("Model is void.")

		if type_name(feat) == "BytesFeature":
			feat = feat.sort(by="utt")
		elif type_name(feat) == "NumpyFeature":
			feat = feat.sort(by="utt").to_bytes()
		else:
			raise UnsupportedType(f"Expected exkaldi feature object but got {type_name(feat)}.")

		assert isinstance(trainGraphFile, str), f"Expected a path-like string but got {type_name(trainGraphFile)}."
		if not os.path.isfile(trainGraphFile):
			raise WrongPath(f"No such file:{trainGraphFile}.")

		cmd = f'align-equal-compiled ark:{trainGraphFile} ark:- ark:-'

		out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=feat.data)

		if (isinstance(cod,int) and cod != 0):
			print(err.decode())
			raise KaldiProcessError("Failed to align feature equally.")
		
		return BytesAlignmentTrans(out, name=name)
	
	def save(self, outFile):
		'''
		Save model to file.
		'''
		if isinstance(outFile, str):
			if not outFile.strip().endswith(".mdl"):
				outFile += ".mdl"
			make_dependent_dirs(outFile, pathIsFile=True)
			with open(outFile, "wb") as fw:
				fw.write(self.data)
		elif isinstance(outFile, tempfile._TemporaryFileWrapper):
			outFile.read()
			outFile.seek(0)
			outFile.write(self.data)
			outFile.seek(0)
		else:
			raise UnsupportedType("<outFile> is unavaliable file name.")
	
	def load(self, target):
		'''
		Reload a HMM-GMM model from file.
		The original data will be discarded.

		Args:
			<target>: file name.
		'''
		assert isinstance(target,str), "<target> should be a file path."
		if not os.path.isfile(target):
			raise WrongPath(f"No such file: {target}.")
		
		cmd = f"gmm-info {target}"
		out, err, cod = run_shell_command(cmd, stderr=subprocess.PIPE)
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
		if self.is_void:
			raise WrongOperation("Model is void.")
		
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

class MonophoneHMM(BaseHMM):

	def __init__(self, lexicons=None, name="mono"):
		super().__init__(data=None,name=name,lexicons=lexicons)
		self.__tempTree = None

	def initialize(self, feat, topoFile, lexicons=None):
		'''
		Initialize Monophone HMM-GMM model.

		Args:
			<feat>: exkaldi DecisionTree object.
			<topoFile>: file path.
			<lexicons>: None. If no any lexicons provided in DecisionTree, this is expected.
						In this step, we will use "context_indep" lexicon.
		'''

		if type_name(feat) == "BytesFeature":
			feat = feat.subset(nHead=10)
		elif type_name(feat) == "NumpyFeature":
			feat = feat.subset(nHead=10).to_bytes()
		else:
			raise UnsupportedType(f"Expected exkaldi feature object but got {type_name(feat)}.")
		
		if isinstance(topoFile, str):
			if not os.path.isfile(topoFile):
				raise WrongPath(f"No such file:{topoFile}.")
		else:
			raise UnsupportedType("Expected toponology file path.")

		if lexicons is None:
			assert self.lex is not None, "No <lexicons> avaliable defaultly, so provide it please."
			lexicons = self.lex
		else:
			assert type_name(lexicons) == "LexiconBank", f"<lexicons> should be an exkaldi LexiconBank object but got {(type_name(lexicons))}."
	
		sets = tempfile.NamedTemporaryFile('w+', suffix=".int", encoding="utf-8")
		tree = tempfile.NamedTemporaryFile('wb+')
		model = tempfile.NamedTemporaryFile('wb+')

		try:
			lexicons.dump_dict(name="sets", outFile=sets, dumpInt=True)

			cmd = f'gmm-init-mono --shared-phones={sets.name} --train-feats=ark:- {topoFile} {feat.dim} {model.name} {tree.name}'
			out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=feat.data)

			if isinstance(cod,int) and cod != 0:
				print(err.decode())
				raise KaldiProcessError("Failed to initialize mono model.")

			model.seek(0)

			cmd = f'gmm-info --print-args=false {model.name} | grep gaussians'
			out, err, cod = run_shell_command(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

			if len(out) == 0:
				print(err.decode())
				raise KaldiProcessError("Cannot get the numbers of gaussians.")

			tree.seek(0)
			self.__tempTree = DecisionTree(lexicons=self.lex, data=tree.read(), name="monoTree")
			model.seek(0)
			self.reset_data(model.read())

		finally:
			sets.close()
			tree.close()
			model.close()
	
	@property
	def tree(self):
		'''
		Get the temp tree in monophone model.
		'''
		return self.__tempTree

	def train(self, feat, transcription, LFile, tempDir,
				num_iters=40, max_iter_inc=30, totgauss=1000, realign_iter=None,
				transitionScale=1.0, acousticScale=0.1, selfloopScale=0.1,
				initial_beam=6, beam=10, retry_beam=40,
				boost_silence=1.0, careful=False, power=0.25, minGaussianOccupancy=10, lexicons=None):
		'''
		This is a high-level APi to train the HMM-GMM model.

		Args:
			<feat>: exkaldi feature object.
			<transcription>: file or exkaldi transcription object.
							Note that, it should be text format, not int value format.
			<LFile>: L.fst file path.
			<tempDir>: A directory to save intermidiate files.
			<num_iters>: Int value, the max iteration times.
			<max_iter_inc>: Int value, increase numbers of gaussian functions when iter is smaller than <num_iters>.
			<totgauss>: Int value, the rough target numbers of gaussian functions.
			<realign_iter>: None or list or tuple, the iter to realign.
		'''
		ExkaldiInfo.vertify_kaldi_existed()

		if self.is_void:
			raise WrongOperation("Model must be initialized befor training.")
		
		exNumgauss = self.info.gaussians
		assert isinstance(totgauss,int) and totgauss >= exNumgauss, f"<totgauss> should be larger than current gaussians: {exNumgauss}."

		if lexicons is None:
			assert self.lex is not None, "No <lexicons> avaliable defaultly, so provide it please."
			lexicons = self.lex
		else:
			assert type_name(lexicons) == "LexiconBank", f"<lexicons> should be an exkaldi LexiconBank object but got {(type_name(lexicons))}."

		if realign_iter is not None:
			assert isinstance(realign_iter,(list,tuple)),"<realign_iter> should be a list or tuple of iter numbers."

		print("Start to train mono model.")
		make_dependent_dirs(tempDir, pathIsFile=False)

		starttime = datetime.datetime.now().strftime("%Y/%m/%d-%H:%M:%S")
		print(f"Start Time: {starttime}")

		print("Convert transcription to int value format.")
		trans = transcription_to_int(transcription, wordSymbolTable=lexicons, unkSymbol=lexicons("oov"))

		print('Compiling training graph.')
		trainGraphFile = os.path.join(tempDir, "train_graph")
		self.compile_train_graph(tree=self.tree, transcription=trans, LFile=LFile, outFile=trainGraphFile, lexicons=lexicons)

		print("\nIter 0")
		print('Aligning data equally', end=" ")
		ali = self.align_equally(feat, trainGraphFile=trainGraphFile)

		print('>> Accumulate GMM statistics', end=" ")
		statsFile = os.path.join(tempDir, "stats.acc")
		self.accumulate_stats(feat, alignment=ali, outFile=statsFile)

		print('>> Update GMM parameters')
		self.update(statsFile, exNumgauss, power=power, minGaussianOccupancy=3)

		assert isinstance(max_iter_inc, int) and max_iter_inc > 0, f"<max_iter_inc> must be positive int value but got: {max_iter_inc}."
		incgauss = (totgauss - exNumgauss)//max_iter_inc
		search_beam = initial_beam

		for i in range(1, num_iters+1, 1):
			
			print(f"\nIter {i}")

			if (realign_iter is None) or (i in realign_iter):
				print("Aligning data", end=" ")
				del ali
				ali = self.align(feat, trainGraphFile, transitionScale, acousticScale, selfloopScale, search_beam, retry_beam, boost_silence, careful, lexicons=lexicons)
			else:
				print("Skip aligning", end=" ")

			print(">> Accumulate GMM statistics", end=" ")
			os.remove(statsFile)
			self.accumulate_stats(feat, alignment=ali, outFile=statsFile)

			print(">> Update GMM parameter")
			self.update(statsFile, exNumgauss, power, minGaussianOccupancy)

			exNumgauss += incgauss
			search_beam = beam
		
		modelFile = os.path.join(tempDir,"final.mdl")
		self.save(modelFile)

		print('\nAlign last time with final model.')
		del ali
		ali = self.align(feat, trainGraphFile, transitionScale, acousticScale, selfloopScale, search_beam, retry_beam, boost_silence, careful)
		aliFile = os.path.join(tempDir,"final.ali")
		ali.save(aliFile)

		print('\nDone to train the monophone model.')
		print(f"Saved Final Model: {modelFile}")
		print(f"Saved Alignment Model: {aliFile}")
		endtime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
		print(f"End Time: {endtime}")

class TriphoneHMM(BaseHMM):

	def __init__(self, lexicons=None, name="tri"):
		super().__init__(data=None, name=name, lexicons=lexicons)
		self.__tree = None

	def initialize(self, tree, treeStatsFile, topoFile, numgauss):
		'''
		Initialize a Triphone Model.

		Args:
			<tree>: file path or exkaldi DecisionTree object.
			<treeStatsFile>: file path.
			<topoFile>: file path.
			<numgauss>: int value.
		'''
		if isinstance(tree, str):
			tree =  DecisionTree().load(tree)
		else:
			assert isinstance(tree, DecisionTree), "<tree> should be file name or exkaldi DecisionTree object."

		occs = tempfile.NamedTemporaryFile("wb+", suffix=".occs")
		try:
			cmd = f"gmm-init-model --write-occs={occs.name} - {treeStatsFile} {topoFile} -"
			out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=tree.data)
			if (isinstance(cod, int) and cod != 0) or len(out) == 0:
				print(err.decode())
				raise KaldiProcessError("Failed to initialize triphone model.") 
			
			occs.seek(0)
			cmd = f"gmm-mixup --mix-up={numgauss} - {occs.name} -"
			out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=out)
			if (isinstance(cod, int) and cod != 0) or len(out) == 0:
				print(err.decode())
				raise KaldiProcessError("Failed to initialize triphone model.")

			self.reset_data(out)
			self.__tree = tree

		finally:
			occs.close()

	@property
	def tree(self):
		return self.__tree

	def train(self, feat, transcription, LFile, tree, tempDir,
				num_iters=40, max_iter_inc=30, totgauss=1000, realign_iter=None,
				transitionScale=1.0, acousticScale=0.1, selfloopScale=0.1,
				initial_beam=6, beam=10, retry_beam=40,
				boost_silence=1.0, careful=False, power=0.25, minGaussianOccupancy=10, lexicons=None):
		'''
		This is a high-level API to train the HMM-GMM model.

		Args:
			<feat>: exkaldi feature object.
			<transcription>: file or exkaldi transcription object.
							 Note that, it should be text format, not int value format.
			<LFile>: L.fst file path.
			<tree>: file path or exkaldi DecisionTree object.
			<tempDir>: A directory to save intermidiate files.
			<num_iters>: Int value, the max iteration times.
			<max_iter_inc>: Int value, increase numbers of gaussian functions when iter is smaller than <num_iters>.
			<totgauss>: Int value, the rough target numbers of gaussian functions.
			<realign_iter>: None or list or tuple, the iter to realign.
		'''
		ExkaldiInfo.vertify_kaldi_existed()

		if self.is_void:
			raise WrongOperation("Model must be initialized befor training.")
		
		if realign_iter is not None:
			assert isinstance(realign_iter,(list,tuple)), "<realign_iter> should be a list or tuple of iter numbers."

		if lexicons is None:
			assert self.lex is not None, "No <lexicons> avaliable defaultly, so provide it please."
			lexicons = self.lex
		else:
			assert type_name(lexicons)=="LexiconBank", f"<lexicons> should be an exkaldi LexiconBank object but got {(type_name(lexicons))}."

		exNumgauss = self.info.gaussians
		assert isinstance(totgauss,int) and totgauss >= exNumgauss, f"<totgauss> should be larger than current gaussians: {exNumgauss}."

		print("Start to train triphone model.")
		make_dependent_dirs(tempDir, pathIsFile=False)

		starttime = datetime.datetime.now().strftime("%Y/%m/%d-%H:%M:%S")
		print(f"Start Time: {starttime}")

		print("Convert transcription to int value format.")
		trans = transcription_to_int(transcription, wordSymbolTable=lexicons, unkSymbol=lexicons("oov"))

		print('Compiling training graph.')
		trainGraphFile = os.path.join(tempDir, "train_graph")
		self.compile_train_graph(tree=tree, transcription=trans, LFile=LFile, outFile=trainGraphFile, lexicons=lexicons)

		assert isinstance(max_iter_inc, int) and max_iter_inc > 0, f"<max_iter_inc> must be positive int value but got: {max_iter_inc}."
		incgauss = (totgauss - exNumgauss)//max_iter_inc
		search_beam = initial_beam

		statsFile = os.path.join(tempDir, "gmmStats.acc")
		for i in range(1, num_iters+1, 1):
			
			print(f"\nIter {i}")

			if (realign_iter is None) or i==1 or (i in realign_iter):
				print("Aligning data", end=" ")
				ali = self.align(feat,trainGraphFile,transitionScale,acousticScale,selfloopScale,search_beam,retry_beam,boost_silence,careful,lexicons=lexicons)
			else:
				print("Skip aligning", end=" ")

			print(">> Accumulate GMM statistics", end=" ")

			self.accumulate_stats(feat, alignment=ali, outFile=statsFile)
			del ali

			print(">> Update GMM parameter")
			self.update(statsFile, exNumgauss, power, minGaussianOccupancy)
			os.remove(statsFile)

			exNumgauss += incgauss
			search_beam = beam
		
		modelFile = os.path.join(tempDir,"final.mdl")
		self.save(modelFile)

		print('\nAlign last time with final model.')
		ali = self.align(feat, trainGraphFile, transitionScale, acousticScale, selfloopScale, search_beam, retry_beam, boost_silence, careful)
		aliFile = os.path.join(tempDir,"final.ali")
		ali.save(aliFile)
		del ali

		del self.__tree
		self.__tree = tree

		print('\nDone to train the triphone model.')
		print(f"Saved Final Model: {modelFile}")
		print(f"Saved Alignment Model: {aliFile}")
		endtime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
		print(f"End Time: {endtime}")

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
	assert isinstance(target,str), "<target> should be a file path."
	if not os.path.isfile(target):
		raise WrongPath(f"No such file: {target}.")
	
	cmd = f"tree-info {target}"
	out, err, cod = run_shell_command(cmd, stderr=subprocess.PIPE)
	if isinstance(cod,int) and cod != 0:
		print(err.decode())
		raise WrongDataFormat("Failed to load tree.")
	else:
		with open(target, "rb") as fr:
			data = fr.read()
		return DecisionTree(data=out, name=name, lexicons=lexicons)

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
	assert isinstance(target, str), "<target> should be a file path."
	if not os.path.isfile(target):
		raise WrongPath(f"No such file: {target}.")
	
	cmd = f"gmm-info {target}"
	out, err, cod = run_shell_command(cmd, stderr=subprocess.PIPE)
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

def sum_gmm_accs(gmmStats, outFile):
	'''
	Sum GMM statistics.

	Args:
		<gmmStats>: a string, list or tuple of mutiple file paths.
		<outFile>: output file path.
	Return:
	 	absolute path of accumulated file.
	'''
	if isinstance(gmmStats, str):
		if len(glob.glob(gmmStats)) == 0:
			raise WrongPath(f"No such file:{gmmStats}.")
	elif isinstance(gmmStats, (list,tuple)):
		temp = []
		for fname in gmmStats:
			assert isinstance(fname, str), f"Expected path-like string but got {type_name(fname)}."
			temp.extend( glob.glob(fname) )
		gmmStats = " ".join(temp)
	else:
		raise UnsupportedType(f"Expected string, list or tuple object but got {type_name(gmmStats)}.")

	ExkaldiInfo.vertify_kaldi_existed()

	cmd = f'gmm-sum-accs {outFile} {gmmStats}'

	out, err, _ = run_shell_command(cmd, stderr=subprocess.PIPE)

	if (not os.path.isfile(outFile)) or os.path.getsize(outFile) == 0:
		print(err.decode())
		raise KaldiProcessError(f"Failed to sum GMM statistics.")
	else:
		return os.path.abspath(outFile)

def sum_tree_stats(treeStats, outFile):
	'''
	Sum tree statistics.

	Args:
		<treeStats>: a string, list or tuple of mutiple file paths.
		<outFile>: output file path.
	Return:
	 	absolute path of accumulated file.
	'''
	if isinstance(treeStats, str):
		if len(glob.glob(treeStats)) == 0:
			raise WrongPath(f"No such file:{treeStats}.")
	elif isinstance(treeStats, (list,tuple)):
		temp = []
		for fname in treeStats:
			assert isinstance(fname, str), f"Expected path-like string but got {type_name(fname)}."
			temp.extend(glob.glob(fname))
		treeStats = " ".join(temp)
	else:
		raise UnsupportedType(f"Expected string, list or tuple object but got {type_name(treeStats)}.")

	ExkaldiInfo.vertify_kaldi_existed()

	cmd = f'sum-tree-stats {outFile} {treeStats}'

	out, err, _ = run_shell_command(cmd, stderr=subprocess.PIPE)

	if (not os.path.isfile(outFile)) or os.path.getsize(outFile) == 0:
		print(err.decode())
		raise KaldiProcessError("Failed to sum tree statistics.")
	else:
		return os.path.abspath(outFile)

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
	assert type_name(lexicons) == "LexiconBank", f"Expected <lexicons> is exkaldi LexiconBank object but got {type_name(lexicons)}."
	assert isinstance(outFile, str), "Expected <outFile> is a name-like string."
	assert isinstance(numNonsilStates, int) and numNonsilStates > 0, "Expected <numNonsilStates> is a positive int number."
	assert isinstance(numSilStates, int) and numSilStates > 0, "Expected <numSilStates> is a positive int number."

	ExkaldiInfo.vertify_kaldi_existed()

	nonsilPhones = lexicons("nonsilence", returnInt=True)
	silPhones = lexicons("silence", returnInt=True)

	nonsilphonelist = ":".join(nonsilPhones)
	silphonelist = ":".join(silPhones)

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

def convert_alignment(alignment, monoHmm, triHmm, tree):
	'''
	Convert alignment from monophone level to triphone level.

	Args:
		<alignment>: file path or exkaldi transition-ID alignment object.
		<monoHmm>: file path or exkaldi MonophoneHMM object.
		<triHmm>: file path or exkaldi TriphoneHMM object.
		<tree>: file path or exkaldi DecisionTree object.
	Return:
	 	An exkaldi Transition-ID alignment object.
	'''
	if isinstance(alignment, str):
		alignment = load_ali(alignment)
	else:
		assert type_name(alignment) in ["BytesAlignmentTrans","NumpyAlignmentTrans"], f"<alignment> should be file name or exkaldi trans-ID alignment object but got: {(type_name(alignment))}." 

	if type_name(alignment) == "BytesAlignmentTrans":
		bytesFlag = True
	elif type_name(alignment) == "NumpyAlignmentTrans":
		alignment = alignment.to_bytes()
		bytesFlag = False
	else:
		raise UnsupportedType(f"Expected trainstion-ID alignment data but got: {type_name(alignment)}.")

	if isinstance(monoHmm, str):
		monoHmm = MonophoneHMM().load(monoHmm)
	else:
		assert isinstance(monoHmm, MonophoneHMM), f"<monoHmm> must be file or MonophoneHMm object but got: {type_name(monoHmm)}."

	if isinstance(triHmm, str):
		triHmm = TriphoneHMM().load(triHmm)
	else:
		assert isinstance(triHmm, TriphoneHMM), f"<triHmm> must be file or TriphoneHMM object but got: {type_name(triHmm)}."

	if isinstance(tree, str):
		tree = DecisionTree().load(tree)
	else:
		assert isinstance(tree, DecisionTree), f"<tree> must be file or DecisionTree object but got: {type_name(tree)}."

	monomodelTemp = tempfile.NamedTemporaryFile("wb+", suffix="_mono.mdl")
	trimodelTemp = tempfile.NamedTemporaryFile("wb+", suffix="_tri.mdl")
	treeTemp = tempfile.NamedTemporaryFile("wb+", suffix=".tree")

	try:
		monoHmm.save(monomodelTemp)
		triHmm.save(trimodelTemp)
		tree.save(treeTemp)

		cmd = f"convert-ali {monomodelTemp.name} {trimodelTemp.name} {treeTemp.name} ark:- ark:-"

		out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=alignment.data)
		if (isinstance(cod,int) and cod != 0) or len(out) == 0:
			print(err.decode())
			raise KaldiProcessError("Failed to convert alignment.")
		else:
			if bytesFlag:
				return BytesAlignmentTrans(out, name=alignment.name)
			else:
				return BytesAlignmentTrans(out, name=alignment.name).to_numpy()
	
	finally:
		monomodelTemp.close()
		trimodelTemp.close()
		treeTemp.close()

def transcription_to_int(transcription, wordSymbolTable, unkSymbol):
	'''
	Transform text format transcrption to int format file.
	'''
	if isinstance(transcription, str):
		if not os.path.isfile(transcription):
			raise WrongPath(f"No such file: {transcription}.")
		trans = Transcription().load(transcription)
	elif isinstance(transcription, Transcription):
		trans = copy.deepcopy(transcription)
	else:
		raise UnsupportedType(f"<transcription> should be a file or exkaldi Transcription object but got {type_name(transcription)}.")

	if isinstance(wordSymbolTable, str):
		if not os.path.isfile(wordSymbolTable):
			raise WrongPath(f"No such file: {wordSymbolTable}.")
		wordSymbolTable = ListTable(name="word2id").load(wordSymbolTable)
	elif isinstance(wordSymbolTable, ListTable):
		pass
	elif type_name(wordSymbolTable) == "LexiconBank":
		wordSymbolTable = wordSymbolTable("words")
	else:
		raise UnsupportedType(f"<wordSymbolTable> should be a file, exkaldi ListTable or LexiconBank object but got {type_name(wordSymbolTable)}.")	
	
	for utt, text in trans.items():
		assert isinstance(text, str), f"The value of Transcription table must be string but got {type_name(text)}."
		text = text.split()
		for index, word in enumerate(text):
			try:
				text[index] = str(wordSymbolTable[word])
			except KeyError:
				try:
					text[index] = str(wordSymbolTable[unkSymbol])
				except KeyError:
					raise WrongDataFormat(f"Word symbol table miss unknown-map symbol: {unkSymbol}")
	
		trans[utt] = " ".join(text)
	
	return trans

def transcription_from_int(transcription, wordSymbolTable):
	'''
	Transform int format transcrption to text format file.
	'''
	if isinstance(transcription, str):
		if not os.path.isfile(transcription):
			raise WrongPath(f"No such file: {transcription}.")
		trans = Transcription().load(transcription)
	elif isinstance(transcription, Transcription):
		trans = copy.deepcopy(transcription)
	else:
		raise UnsupportedType(f"<transcription> should be a file or exkaldi Transcription object but got {type_name(transcription)}.")

	if isinstance(wordSymbolTable, str):
		if not os.path.isfile(wordSymbolTable):
			raise WrongPath(f"No such file: {wordSymbolTable}.")
		symbolWordTable = ListTable(name="word2id").load(wordSymbolTable).reverse()
	elif isinstance(wordSymbolTable, ListTable):
		symbolWordTable = wordSymbolTable.reverse()
	elif type_name(wordSymbolTable) == "LexiconBank":
		items = map(lambda x:(str(x[1]),x[0]), wordSymbolTable("words").items())
		symbolWordTable = ListTable(items, name="id2word")
	else:
		raise UnsupportedType(f"<wordSymbolTable> should be a file, exkaldi ListTable or LexiconBank object but got {type_name(wordSymbolTable)}.")	
	
	for utt, text in trans.items():
		assert isinstance(text, str), f"The value of Transcription table must be string but got {type_name(text)}."
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
					
		trans[utt] = " ".join(text)
	
	return trans