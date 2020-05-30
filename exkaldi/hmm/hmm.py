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

from exkaldi.core.achivements import BytesAchievement, Transcription, ListTable, BytesAlignmentTrans
from exkaldi.core.load import load_ali
from exkaldi.utils import run_shell_command, check_config, make_dependent_dirs, type_name
from exkaldi.version import version as ExkaldiInfo
from exkaldi.version import WrongPath, KaldiProcessError, UnsupportedType, WrongOperation, WrongDataFormat

class DecisionTree(BytesAchievement):

	def __init__(self, lexicons, data=b"", contextWidth=3, centralPosition=1, name="tree"):

		assert type_name(lexicons) == "LexiconBank", f"<lexicons> should be exkaldi LexiconBank object but got {type_name(lexicons)}."
		super().__init__(data, name)

		self.__lex = lexicons
		self.__contextWidth = contextWidth
		self.__centralPosition = centralPosition
	
	@property
	def lex(self):
		return self.__lex

	@property
	def contextWidth(self):
		return self.__contextWidth

	@property
	def centralPosition(self):
		return self.__centralPosition

	def accumulate_tree_stats(self, feat, hmm, alignFile, outFile):

		if type_name(feat) == "BytesFeature":
			pass
		elif type_name(feat) == "NumpyFeature":
			feat = feat.to_bytes()
		else:
			raise UnsupportedType(f"Expected exkaldi feature object but got {type_name(feat)}.")
		
		if not isinstance(hmm, BaseHMM):
			raise UnsupportedType(f"Expected exkaldi HMM object but got {type_name(hmm)}.")	

		assert isinstance(alignFile, str), f"Expected a path-like string but got {type_name(alignFile)}."
		if not os.path.isfile(alignFile):
			raise WrongPath(f"No such file:{alignFile}.")

		ciphones = ":".join(self.__lex("context_indep",True))

		model = tempfile.NamedTemporaryFile("wb+", suffix=".mdl")

		try:
			model.write(hmm.data)
			model.seek(0)

			cmd = f'acc-tree-stats --context-width={self.contextWidth} --central-position={self.centralPosition} --ci-phones={ciphones} '
			cmd += f'{model.name} ark,c,cs:- ark:{alignFile} {outFile}'

			out, err, _ = run_shell_command(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, inputs=feat.data)

			if (not os.path.isfile(outFile)) or (os.path.getsize(outFile)==0):
				print(err.decode())
				raise KaldiProcessError("Failed to initialize mono model.")
	
		finally:
			model.close()
		
	def compile_questions(self, treeStatsFile, topoFile, outFile):

		for x in [treeStatsFile, topoFile]:
			assert isinstance(x, str), f"Expected a path-like string but got {type_name(x)}."
			if not os.path.isfile(x):
				raise WrongPath(f"No such file:{x}.")
		
		sets = tempfile.NamedTemporaryFile("w+", suffix=".int", encoding="utf-8")
		questions = tempfile.NamedTemporaryFile("a+", suffix=".int", encoding="utf-8")
	
		try:
			self.lex.dump_dict(name="sets", outFile=sets, dumpint=True)

			cmd = f'cluster-phones --context-width={self.contextWidth} --central-position={self.centralPosition} '
			cmd = f'{treeStatsFile} {sets.name} {questions.name}'

			out, err, _ = run_shell_command(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

			if os.path.getsize(questions.name) == 0:
				print(err.decode())
				raise KaldiProcessError("Failed to cluster phones.")
			
			self.lex.dump_dict("extra_questions", questions, True)

			cmd = f'compile-questions --context-width={self.contextWidth} --central-position={self.centralPosition} '
			cmd += f'{topoFile} {questions.name} {outFile}'

			out, err, _ = run_shell_command(cmd, stderr=subprocess.PIPE)

			if (not os.path.isfile(outFile)) or os.path.getsize(outFile) == 0:
				print(err.decode())
				raise KaldiProcessError("Failed to compile questions.")

		finally:
			sets.close()
			questions.close()	

	def build_tree(self, treeStatsFile, questionsFile, numleaves, topoFile, cluster_thresh=-1):

		assert isinstance(questionsFile, str), f"Expected a path-like string but got {type_name(questionsFile)}."
		if not os.path.isfile(questionsFile):
			raise WrongPath(f"No such file:{questionsFile}.")

		roots = tempfile.NamedTemporaryFile("w+", suffix=".int", encoding="utf-8")

		try:
			self.lex.dump_dict("roots", roots, True)

			cmd = f'build-tree --context-width={self.contextWidth} --central-position={self.centralPosition} '
			cmd += f'--verbose=1 --max-leaves={numleaves} --cluster-thresh={cluster_thresh} '
			cmd += f'{treeStatsFile} {roots.name} {questionsFile} {topoFile}'

			out, err, _ = run_shell_command(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

			if len(out) == 0:
				print(err.decode())
				raise KaldiProcessError("Failed to build tree.")
			else:
				self.__data = out

		finally:
			roots.close()			

	def train(self, feat, hmm, alignFile, topoFile, numleaves, tempDir, cluster_thresh=-1, treeaccOutFile=None):

		print("Accumulate tree statistics")
		if treeaccOutFile is None:
			treeaccOutFile = os.path.join(tempDir, "treeacc")
		self.accumulate_tree_stats(feat, hmm, alignFile, treeaccOutFile)

		print("Compile questions")
		questionsFile = os.path.join(tempDir, "questions.qst" )
		self.compile_questions(treeaccOutFile, topoFile, questionsFile)

		print("Build tree")
		self.build_tree(treeaccOutFile, questionsFile, numleaves, topoFile, cluster_thresh)

		print("Train tree done.")

	def save(self, outFile="tree"):
		if isinstance(outFile, str):
			make_dependent_dirs(outFile, pathIsFile=True)
			with open(outFile,"wb") as fw:
				fw.write(self.data)
		elif isinstance(outFile, tempfile._TemporaryFileWrapper):
			outFile.read()
			outFile.seek(0)
			outFile.write(self.data)
			outFile.seek(0)
		else:
			raise UnsupportedType("<outFile> is unavaliable file name")

	def load(self, path):
		'''
		Load tree from file.

		<Args>:
			file path.
		<Return>:
			An exkaldi DecisionTree object.
		'''
		#cmd = "tree-info"
		pass

class BaseHMM(BytesAchievement):

	def __init__(self, lexicons, data=b"", name="hmm"):
		assert type_name(lexicons) == "LexiconBank", f"Expected a exkaldi LexiconBank object but got {type_name(lexicons)}."
		self.__lex = lexicons
		
		super().__init__(data, name)
		self.__numgauss = 0

	@property
	def numgauss(self):
		if self.is_void:
			raise WrongOperation("Void HMM-GMM model.")
		cmd = "gmm-info --print-args=false - | grep gaussians"
		out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=self.data)
		if (isinstance(cod,int) and cod != 0):
			print(err.decode())
			raise KaldiProcessError("Get the number of gaussian functions.")

		return int(out.rstrip().split()[-1])

	@property
	def lex(self):
		return self.__lex
	
	def compile_train_graphs(self, tree, transcription, LFile, outFile):
		'''
		Compile training graph.

		Args:
			<tree>: file path or exkaldi DecisionTree object.
			<transcription>: file path or exkaldi Transcription object with int format.
							Note that: int fotmat, not text format.
			<Lfile>: L.fst file path.
			<outFile>: graph output file path.
		'''
		assert isinstance(outFile, str), "<outFile> should be string."

		if self.is_void:
			raise WrongOperation("Model is void.")

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
				raise UnsupportedType(f"<tree> should be file path or exkaldi DecisionTree objectc.")

			if isinstance(transcription, str):
				if not os.path.isfile(transcription):
					raise WrongPath(f"No such file: {transcription}.")
				transcription = Transcription().load(transcription).sort()			
			elif isinstance(transcription, Transcription):
				transcription = transcription.sort()
			else:
				raise UnsupportedType(f"<transcription> should be file path or exkaldi Transcription object.")
			transcription.save(outFile=transTemp)
			transFile = transTemp.name

			assert isinstance(tree, DecisionTree), f"<tree> should be a DecisionTree object but got {type_name(tree)}."

			self.lex.dump_dict(name="disambig", outFile=disambigTemp, dumpInt=True)

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
			
		finally:
			disambigTemp.close()
			treeTemp.close()
			transTemp.close()

	def estimate_gmm(self, gmmStatsFile, numgauss, power=0.25, minGaussianOccupancy=10):

		if self.is_void:
			raise WrongOperation("Model is void.")

		assert isinstance(gmmStatsFile, str), f"Expected a path-like string but got {type_name(gmmStatsFile)}."
		if not os.path.isfile(gmmStatsFile):
			raise WrongPath(f"No such file:{gmmStatsFile}.")
		assert isinstance(numgauss, int) and numgauss >= self.numgauss, f"<numgauss> should be a int value and no less than current numbers {self.numgauss}."

		cmd = f'gmm-est --min-gaussian-occupancy={minGaussianOccupancy} --mix-up={numgauss} --power={power} '
		cmd += f'- {gmmStatsFile} -'

		out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=self.data)

		if (isinstance(cod,int) and cod != 0 ) or len(out) == 0:
			print(err.decode())
			raise KaldiProcessError("Failed to estimate new GMM parameters.")
		else:
			self.reset_data(out)
			return self

	def align_compiled(self, feat, trainGraphsFile, outFile, transitionScale=1.0, acousticScale=0.1, 
						selfloopScale=0.1, beam=10, retry_beam=40, boost_silence=1.0, careful=False):

		if self.is_void:
			raise WrongOperation("Model is void.")

		if type_name(feat) == "BytesFeature":
			pass
		elif type_name(feat) == "NumpyFeature":
			feat = feat.to_bytes()
		else:
			raise UnsupportedType(f"Expected exkaldi feature object but got {type_name(feat)}.")

		assert isinstance(trainGraphsFile, str), f"Expected a path-like string but got {type_name(trainGraphsFile)}."
		if not os.path.isfile(trainGraphsFile):
			raise WrongPath(f"No such file:{trainGraphsFile}.")		
		
		optionSilence = ":".join(self.__lex("optional_silence", True))

		model = tempfile.NamedTemporaryFile("wb+", suffix=".mdl")

		try:
			cmd = f'gmm-boost-silence --boost={boost_silence} {optionSilence} - {model.name}'
			out, err, _ = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=self.data)

			if os.path.getsize(model.name) == 0:
				print(err.decode())
				raise KaldiProcessError("Generate new HMM defeated.")

			cmd = f'gmm-align-compiled --transition-scale={transitionScale} --acoustic-scale={acousticScale} --self-loop-scale={selfloopScale} '
			cmd += f'--beam={beam} --retry-beam={retry_beam} --careful={careful} {model.name} '
			cmd += f'ark:{trainGraphsFile} '
			cmd += f'ark,c,cs:- ark,t:{outFile}'

			out, err, _ = run_shell_command(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, inputs=feat.data)

			if (not os.path.isfile(outFile)) or os.path.getsize(outFile) == 0:
				print(err.decode())
				raise KaldiProcessError("Failed to align.")
		
		finally:
			model.close()
	
	def accumulate_gmm_stats(self, feat, alignment, outFile):
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

	def align_equal_compiled(self, feat, trainGraphFile, outFile=None):
		'''
		Align feature averagely.

		Args:
			<feat>: exkaldi feature object.
			<trainGraphFile>: file path.
			<outFile>: None or file name. If None, return exkaldi
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

		if outFile is None:
			cmd = f'align-equal-compiled ark:{trainGraphFile} ark:- ark:-'
		else:
			assert isinstance(outFile, str), f"<outFile> should be None or file name but got: {type_name(outFile)}."
			cmd = f'align-equal-compiled ark:{trainGraphFile} ark:- ark:{outFile}'

		out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=feat.data)

		if (isinstance(cod,int) and cod != 0):
			print(err.decode())
			raise KaldiProcessError("Failed to align feature equally.")
		else:
			os.path.abspath(outFile)
		
		if outFile is None:
			return BytesAlignmentTrans(out)
	
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

class MonophoneHMM(BaseHMM):

	def __init__(self, lexicons, name="mono"):
		super().__init__(lexicons, None, name)
		self.__tempTree = None

	def initialize(self, feat, topoFile):
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

		sets = tempfile.NamedTemporaryFile('w+', suffix=".int", encoding="utf-8")
		tree = tempfile.NamedTemporaryFile('wb+')
		model = tempfile.NamedTemporaryFile('wb+')

		try:
			self.lex.dump_dict(name="sets", outFile=sets, dumpInt=True)

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

			self.__numgauss = int(out.decode().strip().split()[-1])

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
		return self.__tempTree

	def train(self, feat, transcriptionFile, LFile, tempDir,
				num_iters=40, max_iter_inc=30, totgauss=1000, realign_iter=None,
				transitionScale=1.0, acousticScale=0.1, selfloopScale=0.1,
				initial_beam=6, beam=10, retry_beam=40,
				boost_silence=1.0, careful=False, power=0.25, minGaussianOccupancy=10):

		ExkaldiInfo.vertify_kaldi_existed()

		if self.is_void:
			raise WrongOperation("Model must be initialized befor training.")
		
		if realign_iter is not None:
			assert isinstance(realign_iter,(list,tuple)),"<realign_iter> should be a list or tuple of iter numbers."

		print("Start to train mono model.")
		print('Compiling training graphs.')
		trainGraphFile = os.path.join(tempDir, "trainGraph")
		self.compile_train_graphs(self.__tempTree, transcriptionFile, LFile, trainGraphFile)

		print("Iter 0")
		print('Aligning data equally')
		alignmentFile = os.path.join(tempDir, "ali")
		self.align_equal_compiled(feat, trainGraphFile, alignmentFile)

		print('Accumulate GMM statistics')
		gmmStatsFile = os.path.join(tempDir, "gmmStats.acc")
		self.accumulate_gmm_stats(feat, alignmentFile, gmmStatsFile)

		exNumgauss = self.numgauss
		print('Estimate GMM parameters')
		self.estimate_gmm(alignmentFile, exNumgauss, power, minGaussianOccupancy=3)

		incgauss = (totgauss - self.numgauss)//max_iter_inc
		search_beam = initial_beam

		for i in range(1, num_iters+1, 1):
			
			print(f"Iter {i}")

			if (realign_iter is None) or (i in realign_iter):
				print("Aligning data")
				self.align_compiled(feat, trainGraphFile, alignmentFile, transitionScale, acousticScale, selfloopScale, search_beam, retry_beam, boost_silence, careful)

			print("Accumulate GMM statistics")
			self.accumulate_gmm_stats(feat, alignmentFile, gmmStatsFile)

			print('Estimate GMM parameters')
			self.estimate_gmm(gmmStatsFile, exNumgauss, power, minGaussianOccupancy)

			exNumgauss += incgauss
			search_beam = beam
		
		print('Done training monophone system')

class TriphoneHMM(BaseHMM):

	def __init__(self, lexicons, name="tri"):
		super().__init__(lexicons, None, name)
		self.__tree = None

	def initialize(self, tree, treeAccFile, topoFile, numgauss):

		assert isinstance(tree, DecisionTree), "<tree> should be DecisionTree object."

		occs = tempfile.NamedTemporaryFile("wb+")
		try:
			cmd = f"gmm-init-model --write-occs={occs.name} - {treeAccFile} {topoFile} -"
			out, err, _ = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=tree.data)
			if len(out) == 0:
				print(err.decode())
				raise KaldiProcessError("Failed to initialize triphone model.") 
			
			occs.seek(0)
			cmd = f"gmm-mixup --mix-up={numgauss} - {occs.name} -"
			out, err, _ = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=out)
			if len(out) == 0:
				print(err.decode())
				raise KaldiProcessError("Failed to initialize triphone model.")

			self.__data = out
			self.__numgauss = numgauss
			self.__tree = tree
		finally:
			occs.close()

	@property
	def tree(self):
		return self.__tree

	def train(self, feat, transcriptionFile, LFile, tree, tempDir,
				num_iters=40, max_iter_inc=30, totgauss=1000, realign_iter=None,
				transitionScale=1.0, acousticScale=0.1, selfloopScale=0.1,
				initial_beam=6, beam=10, retry_beam=40,
				boost_silence=1.0, careful=False, power=0.25, minGaussianOccupancy=10):

		ExkaldiInfo.vertify_kaldi_existed()

		if self.is_void:
			raise WrongOperation("Model must be initialized befor training.")
		
		if realign_iter is not None:
			assert isinstance(realign_iter,(list,tuple)), "<realign_iter> should be a list or tuple of iter numbers."

		print("Start to train triphone model.")
		print('Compiling training graphs.')
		trainGraphFile = os.path.join( tempDir, "traingraph" )
		self.compile_train_graphs(transcriptionFile, LFile, tree, trainGraphFile)

		incgauss = (totgauss - self.numgauss)//max_iter_inc
		search_beam = initial_beam

		alignmentFile = os.path.join( tempDir, "ali" )
		gmmStatistics = os.path.join( tempDir, "gmmStats" )
		for i in range(1, num_iters+1, 1):
			
			print("Iter {}".format(i))

			if (realign_iter is None) or (i in realign_iter):
				print("Aligning data")
				self.align_compiled(feat, trainGraphFile, alignmentFile, transitionScale, acousticScale, selfloopScale, search_beam, retry_beam, boost_silence, careful)

			print("Accumulate GMM statistics")
			self.accumulate_gmm_stats(feat, alignmentFile, gmmStatistics)

			print('Estimate GMM parameters')
			self.estimate_gmm(gmmStatistics, exNumgauss, power, minGaussianOccupancy)

			exNumgauss += incgauss
			search_beam = beam
		
		print('Done training Triphone system')

def load_tree(path):
	pass

def load_hmm(path):
	pass

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

def make_toponology(lexicons, outFile="topo", numNonsilStates=3, numSilStates=5):
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

def convert_alignment(aliFile, monoHmm, triHmm, tree, outFile):
	'''
	Convert alignment from monophone level to triphone level.

	Args:
		<aliFile>: alignment file path.
		<monoHmm>: MonophoneHMM object.
		<triHmm>: TriphoneHMM object.
		<tree>: DecisionTree object.
		<outFile>: new alignment file path.
	Return:
	 	absolute path of generated file.
	'''
	assert isinstance(monoHmm, MonophoneHMM), "<monoHmm> should be MonophoneHMM object."
	assert isinstance(triHmm, TriphoneHMM), "<triHmm> should be TriphoneHMM object."
	assert isinstance(tree, DecisionTree), "<tree> should be DecisionTree object."

	monomodel = tempfile.NamedTemporaryFile("wb+")
	trimodel = tempfile.NamedTemporaryFile("wb+")

	try:
		monomodel.write(monoHmm.data)
		monomodel.seek(0)
		trimodel.write(triHmm.data)
		trimodel.seek(0)

		cmd = f"convert-ali {monomodel.name} {trimodel.name} - ark:{aliFile} ark:{outFile}"

		out, err, _ = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=tree.data)
		if not os.path.isfile(outFile) or os.path.getsize(outFile):
			print(err.decode())
			raise KaldiProcessError("Failed to convert alignment.")
	
	finally:
		monomodel.close()
		trimodel.close()

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
				raise WrongDataFormat(f"Word symbol table miss symbol: {word}")
		trans[utt] = " ".join(text)
	
	return trans