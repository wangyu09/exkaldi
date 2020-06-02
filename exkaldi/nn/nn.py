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

"""NN acoustic model training."""

import os
import sys
import random
import subprocess
import tempfile
import numpy as np
import threading
import math
import time
import shutil
from glob import glob

from exkaldi.version import WrongPath, WrongOperation, UnsupportedType
from exkaldi.utils.utils import make_dependent_dirs, type_name, run_shell_command, flatten
from exkaldi.core.load import load_feat
from collections import namedtuple, Iterable

class Supporter:
	'''
	Supporter is used to manage Neural Network training information.

	Args:
		<outDir>: the output directory of Log information.
	'''      
	def __init__(self, outDir='Result'):

		assert isinstance(outDir, str), "<outDir> should be a name-like string."
		make_dependent_dirs(outDir, pathIsFile=False)
		self.outDir = os.path.abspath(outDir)

		self.logFile = os.path.join(self.outDir,'log')
		with open(self.logFile, 'w', encoding='utf-8'): pass

		self.currentField = {}
		self.currentFieldIsFloat = {}
		self.globalField = []

		self.lastSavedArch = {}
		self.savedArchs = []
		self.savingThreshold = None

		self._allKeys = []

		self._iterSymbol = -1
		
	def send_report(self, info):
		'''
		Send information and these will be retained untill you do the statistics by using .collect_report().

		Args:
			<info>: a Python dict object includiing names and their values with int or float type.
					such as {"epoch":epoch,"train_loss":loss,"train_acc":acc}
					The value can be Python int, float object, Numpy int, float object or NUmpy ndarray with only one value.
		'''
		assert isinstance(info, dict), "Expected <info> is a Python dict object."
	
		for name,value in info.items(): 
			assert isinstance(name, str) and len(name) > 0, f"The name of info should be string avaliable but got {type_name(name)}."
			valueDtype = type_name(value)
			if valueDtype.startswith("int"): # Python int object, Numpy int object
				pass

			elif valueDtype.startswith("float"): # Python float object, Numpy float object
				self.currentFieldIsFloat[name] = True
			
			elif valueDtype == "ndarray" and value.shape == ():  # Numpy ndarray with only one value
				if value.dtype == "float":
					self.currentFieldIsFloat[name] = True
			else:
				raise UnsupportedType(f"Expected int or float value but got {type_name(value)}.")

			name = name.lower()
			if not name in self.currentField.keys():
				self.currentField[name] = []
			self.currentField[name].append(value)

	def collect_report(self, keys=None, plot=True):
		'''
		Do the statistics of received information. The result will be saved in outDir/log file. 

		Args:
			<keys>: Specify the data wanted to be collected. If "None", collect all data reported. 
			<plot>: If "True", print the statistics result to standard output.
		'''
		if keys is None:
			keys = list(self.currentField)
		elif isinstance(keys, str):
			keys = [keys,]
		elif isinstance(keys,(list,tuple)):
			pass
		else:
			raise WrongOperation("Expected <keys> is string, list or tuple.")
	
		self.globalField.append({})

		self._allKeys.extend( self.currentField.keys() )
		self._allKeys = list(set(self._allKeys))

		message = ''
		for name in keys:
			if name in self.currentField.keys():

				if len(self.currentField[name]) == 0:
					mn = 0.
				else:
					mn = sum( self.currentField[name] )/len( self.currentField[name] )

				if name in self.currentFieldIsFloat.keys():
					message += f'{name}:{mn:.5f}    '
				else:
					mn = int(mn)
					message += f'{name}:{mn}    '
					
				self.globalField[-1][name] = mn
			else:
				message += f'{name}:-----    '


		with open(self.logFile, 'a', encoding='utf-8') as fw:
			fw.write(message + '\n')
		# Print to screen
		if plot is True:
			print(message)
		# Clear
		self.currentField = {}
		self.currentFieldIsFloat = {}

	def save_arch(self, saveFunc, arch, addInfo=None, byKey=None, byMax=True, maxRetain=0):
		'''
		Usage:  obj.save_arch(saveFunc,archs={'model':model,'opt':optimizer})

		Save architecture such as models or optimizers when you use this function.
		Only collected information will be used to check the condition. So data collecting is expected beforehand.

		Args:
			<saveFunc>: a function to save archivements in <arch>. It need a parameter to reseive <arch>, for example:
						When use tensorflow 2.x
							def save_model(arch):
								for fileName, model in arch.items():
									model.save_weights(fileName+".h5")
			<arch>: a dict object whose keys are the name, and values are achivements' object. It will be
			<addInfo>: a reported name. If it is not None, will add the information to saving file name.
			<byKey>: a reported name. If it is not None, save achivements only this value is larger than last saved achivements.
			<byMax>: a bool value. Control the condition of <byKey>.
			<maxRetain>: the max numbers of saved achivements to retain. If 0, retain all.
		''' 
		assert isinstance(arch, dict), "Expected <arch> is dict whose keys are architecture-names and values are architecture-objects."
		assert callable(saveFunc), "<saveFunc> must be a callable object or function."
		assert isinstance(maxRetain,int) and maxRetain>=0, "<maxRetain> shoule be a non-negative int value."

		#if self.currentField != {}:
		#	self.collect_report(plot=False)
		
		suffix = f"_{self._iterSymbol}"
		self._iterSymbol += 1

		if not addInfo is None:
			assert isinstance(addInfo, (str, list, tuple)), 'Expected <addInfo> is string, list or tuple.'
			if isinstance(addInfo, str):
				addInfo = [addInfo,]
			for name in addInfo:
				if not name in self.globalField[-1].keys():
					suffix += f"{name}None"
				else:
					value = self.globalField[-1][name]
					if isinstance(value, float):
						suffix += f"_{name}{value:.4f}".replace(".","")
					else:
						suffix += f"_{name}{value}"

		saveFlag =False

		if byKey is None:
			self.lastSavedArch = {}
			newArchs = {}
			for name in arch.keys():
				fileName = os.path.join(self.outDir, name+suffix)
				newArchs[fileName] = arch[name]
				self.lastSavedArch[name] = fileName
			saveFlag = True

		else:
			byKey = byKey.lower()
			if not byKey in self.globalField[-1].keys():
				print(f"Warning: Failed to save architectures. Because the key {byKey} has not been reported last time.")
				return
			else:
				value = self.globalField[-1][byKey]

			if self.savingThreshold is None:
				self.savingThreshold = value
				save = True
			else:
				if byMax is True and value > self.savingThreshold:
					self.savingThreshold = value
					save = True
				elif byMax is False and value < self.savingThreshold:
					self.savingThreshold = value
					save = True

			if save is True:
				self.lastSavedArch = {}
				for name in arch.keys():
					if (addInfo is None) or (byKey not in addInfo):
						if isinstance(value, float):
							suffix += f"_{name}{value:.4f}".replace(".","")
						else:
							suffix += f"_{name}{value}"
					fileName = os.path.join(self.outDir, name+suffix)
					newArchs[fileName] = arch[name]
					self.lastSavedArch[name] = fileName
		
		if saveFlag is True:
			# Save
			saveFunc(newArchs)
			# Try to correct the file name
			for name, fileName in self.lastSavedArch.items():
				realFileName = glob( fileName + "*" )
				if len(realFileName) == 0:
					raise WrongOperation(f"Achivement whose name starts with {fileName} should have been saved done but not found.")
				elif len(realFileName) > 1:
					raise WrongOperation(f"More than one achivements whose name start with {fileName} were found.")
				else:
					self.lastSavedArch[name] = realFileName[0]

			self.savedArchs.append( self.lastSavedArch.items() )

			for items in self.savedArchs[0:-maxRetain]:
				for name, fileName in items:
					try:
						if os.path.exists(fileName):
							if os.path.isfile(fileName):
								os.remove(fileName)
							elif os.path.isdir(fileName):
								shutil.rmtree(fileName)
							else:
								raise UnsupportedType(f"Failed to remove {fileName}: It is not a file and directory path.")
					except Exception as e:
						print(f"Failed to remove saved achivements:{fileName}.")
						raise e
			
			self.savedArchs = self.savedArchs[-maxRetain:]

	@property
	def finalArch(self):
		'''
		Get the final saved achivements. 
		
		Return:
			A Python dict object whose key is architecture name and value is file path. 
		''' 
		return self.lastSavedArch
   
	def judge(self, key, condition, threshold, byDeltaRatio=False):
		'''
		Usage:  obj.judge('train_loss','<',0.0001,byDeltaRatio=True) or obj.judge('epoch','>=',10)
		
		Check if condition is true. 
		Only collected information will be used to check the condition. So data collecting is expected beforehand.

		Args:
			<key>: the name reported.
			<condition>: a string, condition operators such as ">" or "=="
			<threshold>: a int or float value.
			<byDeltaRatio>: bool value, if true, threshold should be a delta ratio value.
								deltaRatio = abs((value-value_pre)/value) 

		Return:
			True or False. 
		''' 
		assert condition in ['>','>=','<=','<','==','!='], '<condiction> is not a correct conditional operator.'
		assert isinstance(threshold,(int,float)), '<threshold> should be a float or int value.'

		#if self.currentField != {}:
		#	self.collect_report(plot=False)
		
		if byDeltaRatio is True:
			p = []
			for i in range(len(self.globalField)-1, -1, -1):
				if key in self.globalField[i].keys():
					p.append(self.globalField[i][key])
				if len(p) == 2:
					value = str(abs((p[0]-p[1])/p[0]))
					return eval( value + condition + str(threshold) )
			return False
		else:
			for i in range(len(self.globalField)-1, -1, -1):
				if key in self.globalField[i].keys():
					value = str(self.globalField[i][key])
					return eval(value + condition + str(threshold))
			return False

	def dump(self, keepItems=False):
		'''
		Usage:  product = obj.dump()
		Get all reported information.

		Args:
			<keepItems>: If True, return a dict object.
						 Else, return a list of dict objects. 
		
		Return:
			A dict object or list object.
		'''
		if self.currentField != {}:
			self.collect_report(plot=False)
		
		if self.globalField != []:
			allData = self.globalField
		else:
			raise WrongOperation('Not any information to dump.')

		if keepItems is True:
			items = {}
			for i in allData:
				for key in i.keys():
					if not key in items.keys():
						items[key] = []
					items[key].append(i[key])
			return items
		else:
			return allData

class DataIterator:
	'''
	Usage: obj = DataIterator('train.scp',64,chunks='auto',processFunc=function)

	If you give it a large scp file of train data, it will split it into N smaller chunks and load them into momery alternately with parallel thread. 
	It will shuffle the original scp file and split again while new epoch.
	'''

	def __init__(self, scpTable, processFunc, batchSize, chunks='auto', otherArgs=None, shuffle=False, retainData=0.0):
		
		assert type_name(scpTable) == "ScriptTable", "<scpTable> should be an exkaldi ScriptTable object."
		assert callable(processFunc), "<processFunc> should be a callable instance or function."

		self.fileProcessFunc = processFunc
		self._batchSize = batchSize
		self.otherArgs = otherArgs
		self._shuffle = shuffle
		self._chunks = chunks

		if isinstance(chunks, int):
			assert chunks>0, "Expected <chunks> is a positive int number but got {}.".format(chunks)
		elif chunks != 'auto':
			raise WrongOperation('Expected <chunks> is a positive int number or <auto> but got {}.'.format(chunks))
		
		totalDataNumber = len(scpTable)
		trainDataNumber = int(  totalDataNumber * (1-retainData) )
		evalDataNumber = totalDataNumber - trainDataNumber
		scpTable = scpTable.shuffle()

		self.trainTable = scpTable.subset(nHead=trainDataNumber)
		self.evalTable = scpTable.subset(nTail=evalDataNumber)
		del scpTable

		if chunks == 'auto':
			#Compute the chunks automatically
			sampleTable = self.trainTable.subset(nHead=10)
			with tempfile.NamedTemporaryFile('w', encoding='utf-8', suffix='.scp') as sampleFile:
				sampleTable.save(sampleFile)
				sampleFile.seek(0)
				sampleChunkData = load_feat(sampleFile.name, name="sampleFeature", useSuffix="scp")
			indexInfo = list(sampleChunkData.utt_index.values())[-1]
			meanSize = (indexInfo.startIndex + indexInfo.dataSize)/10 #Bytes
			autoChunkSize = math.ceil(104857600/meanSize)  # 100MB = 102400KB = 104857600 B
			self._chunks = trainDataNumber//autoChunkSize
			if self._chunks == 0: 
				self._chunks = 1

		self.make_dataset_bag(shuffle=False)
		self._epoch = 0
		
		self.load_dataset(0)
		self.currentDataset = self.nextDataset
		self.nextDataset = None

		self.epochSize = len(self.currentDataset)
		self.countEpochSizeFlag = True

		self.currentPosition = 0
		self.currentEpochPosition = 0
		self._isNewEpoch = False
		self._isNewChunk = False
		self.datasetIndex = 0

		if self._chunks > 1:
			self.datasetIndex = 1
			self.loadDatasetThread = threading.Thread(target=self.load_dataset,args=(1,))
			self.loadDatasetThread.start()

	def make_dataset_bag(self, shuffle=False):
		if shuffle:
			self.trainTable.shuffle()
		self.datasetBag = self.trainTable.subset(chunks=self._chunks)

	def load_dataset(self, datasetIndex):
		with tempfile.NamedTemporaryFile('w', suffix='.scp') as scpFile:
			self.datasetBag[datasetIndex].save(scpFile)
			scpFile.seek(0)
			chunkData = load_feat(scpFile.name, name="feat", useSuffix="scp")
		if self.otherArgs != None:
			self.nextDataset = self.fileProcessFunc(self, chunkData, self.otherArgs)
		else:
			self.nextDataset = self.fileProcessFunc(self, chunkData)

		self.nextDataset = [X for X in self.nextDataset]

		if self._batchSize > len(self.nextDataset):
			print(f"Warning: Batch Size <{self._batchSize}> is extremely large for this dataset, we hope you can use a more suitable value.")
		
	def next(self):
		i = self.currentPosition
		iEnd = i + self._batchSize
		N = len(self.currentDataset)

		batch = self.currentDataset[i:iEnd]

		if self._chunks == 1:
			if iEnd >= N:
				rest = iEnd - N
				if self._shuffle:
					random.shuffle(self.currentDataset)
				batch.extend(self.currentDataset[:rest])
				self.currentPosition = rest
				self.currentEpochPosition = self.currentPosition
				self._epoch += 1
				self._isNewEpoch = True
				self._isNewChunk = True
			else:
				self.currentPosition = iEnd
				self._isNewEpoch = False
				self._isNewChunk = False
		else:
			if iEnd >= N:
				rest = iEnd - N
				if self.loadDatasetThread.is_alive():
					self.loadDatasetThread.join()
				if self._shuffle:
					random.shuffle(self.nextDataset)
				batch.extend(self.nextDataset[:rest])
				self.currentPosition = rest
				self.currentDataset = self.nextDataset
				self._isNewChunk = True
				
				if self.countEpochSizeFlag:
					self.epochSize += len(self.currentDataset)

				self.datasetIndex = (self.datasetIndex+1)%self._chunks

				if self.datasetIndex == 1:
					self._epoch += 1
					self._isNewEpoch = True

				if self.datasetIndex == 0:
					self.countEpochSizeFlag = False
					self.make_dataset_bag(shuffle=True)

				self.loadDatasetThread = threading.Thread(target=self.load_dataset,args=(self.datasetIndex,))
				self.loadDatasetThread.start()

			else:
				self._isNewChunk = False
				self._isNewEpoch = False
				self.currentPosition = iEnd

			self.currentEpochPosition = (self.currentEpochPosition + self._batchSize)%self.epochSize

		return batch                            

	@property
	def batchSize(self):
		return self._batchSize

	@property
	def chunks(self):
		return self._chunks

	@property
	def chunk(self):
		if self.datasetIndex == 0:
			return self._chunks - 1
		else:
			return self.datasetIndex - 1

	@property
	def epoch(self):
		return self._epoch

	@property
	def isNewEpoch(self):
		return self._isNewEpoch

	@property
	def isNewChunk(self):
		return self._isNewChunk

	@property
	def epochProgress(self):
		if self._isNewEpoch is True:
			return 1.
		else:
			return round(self.currentEpochPosition/self.epochSize,2)
	
	@property
	def chunkProgress(self):
		if self._isNewChunk is True:
			return 1.
		else:
			return round(self.currentPosition/len(self.currentDataset),2)

	def get_retained_data(self, processFunc=None, batchSize=None, chunks='auto', otherArgs=None, shuffle=False, retainData=0.0):

		if self.evalTable.is_void:
			raise WrongOperation('No retained validation data.')   

		if processFunc is None:
			processFunc = self.fileProcessFunc
		
		if batchSize is None:
			batchSize = self._batchSize

		if isinstance(chunks, int):
			assert chunks > 0, "Expected <chunks> is a positive int number."
		elif chunks != 'auto':
			raise WrongOperation(f'Expected <chunks> is a positive int number or <auto> but got {chunks}.')

		if otherArgs is None:
			otherArgs = self.otherArgs

		reIterator = DataIterator(self.evalTable, processFunc, batchSize, chunks, otherArgs, shuffle, retainData)

		return reIterator

def pad_sequence(data, shuffle=False, pad=0):
	'''
	Usage:  data,lengths = pad_sequence(listData)

	Pad sequences with maximum length of one batch data. <data> should be a list object whose members have various sequence-lengths.
	If <shuffle> is "True", pad each sequence with random start-index and return padded data and length information of (startIndex,endIndex) of each sequence.
	If <shuffle> is "False", align the start index of all sequences then pad them rear. This will return length information of only endIndex.
	'''
	assert isinstance(data, list), f"Expected <data> is a list but got {type_name(data)}."
	assert isinstance(pad, (int, float)), f"Expected <pad> is an int or float value but got {type_name(pad)}."

	lengths = []
	for i in data:
		lengths.append(len(i))
	
	maxLen = int(max(lengths))
	batchSize = len(lengths)

	if len(data[0].shape) == 1:
		newData = pad * np.ones([maxLen,batchSize])
		for k in range(batchSize):
			snl = len(data[k])
			if shuffle:
				n = maxLen - snl
				stp = random.randint(0,n)
				newData[stp:stp+snl,k] = data[k]
				lengths[k] = (stp,stp+snl)
			else:
				newData[0:snl,k] = data[k]

	elif len(data[0].shape) == 2:
		dim = data[0].shape[1]
		newData = pad * np.ones([maxLen,batchSize,dim])
		for k in range(batchSize):
			snl = len(data[k])
			if shuffle:
				n = maxLen - snl
				stp = random.randint(0,n)
				newData[stp:stp+snl,k,:] = data[k]
				lengths[k] = (stp,stp+snl)
			else:
				newData[0:snl,k,:] = data[k]

	elif len(data[0].shape) >= 3:
		otherDims = data[0].shape[1:]
		allDims = 1
		for i in otherDims:
			allDims *= i
		newData = pad * np.ones([maxLen,batchSize,allDims])
		for k in range(batchSize):
			snl = len(data[k])
			if shuffle:
				n = maxLen - snl
				stp = random.randint(0,n)
				newData[stp:stp+snl,k,:] = data[k].reshape([snl,allDims])
				lengths[k] = (stp,stp+snl)
			else:
				newData[0:snl,k,:] = data[k].reshape([snl,allDims])
		newData = newData.reshape([maxLen,batchSize,*otherDims])

	return newData, lengths

def unpack_padded_sequence(data, lengths, batchSizeDim=1):

	'''
	Usage:  listData = unpack_padded_sequence(data,lengths)

	This is a reverse operation of .pad_sequence() function. Return a list whose members are sequences.
	We defaultly think the dimension 0 of <data> is sequence-length and the dimension 1 is batch-size.
	If the dimension of batch-size is not 1, assign the <batchSizeDim> please.
	'''   
	assert isinstance(data, np.ndarray), f"Expected <data> is NumPy array but got {type_name(data)}."
	assert isinstance(lengths, list), "Expected <lengths> is list whose members are padded start position ( and end position)."
	assert isinstance(batchSizeDim, int) and batchSizeDim >= 0, "<batchSizeDim> should be a non-negative int value."
	assert batchSizeDim < len(data.shape), "<batchSizeDim> is out of the dimensions of <data>."
	
	if batchSizeDim != 0:
		dims = [ d for d in range(len(data.shape))]
		dims.remove(batchSizeDim)
		newDim = [batchSizeDim,*dims]
		data = data.transpose(newDim)

	assert len(data) <= len(lengths), "<lengths> is shorter than batch size."

	new = []
	for i,j in enumerate(data):
		if isinstance(lengths[i],int):
			new.append(j[0:lengths[i]])
		elif isinstance(lengths[i],(list,tuple)) and len(lengths[i]) == 2:
			new.append(j[lengths[i][0]:lengths[i][1]])
		else:
			raise WrongOperation("<lengths> has wrong format.")
	
	return new

def softmax(data, axis=1):
	'''
	The softmax function.

	Args:
		<data>: a Numpy array.
		<axis>: the dimension to softmax.
	Return:
		A new array.
	'''
	assert isinstance(data, np.ndarray), f"<data> should be Numpy array but got {type_name(data)}."
	if len(data.shape) == 1:
		axis = 0
	assert isinstance(axis, int) and 0 <= axis < len(data.shape), "<axis> is out of range of the shape of data."
	
	maxValue = data.max(axis, keepdims=True)
	dataNor = data - maxValue

	dataExp = np.exp(dataNor)
	dataExpSum = np.sum(dataExp,axis, keepdims = True)

	return dataExp / dataExpSum

def log_softmax(data, axis=1):
	'''
	The log-softmax function.

	Args:
		<data>: a Numpy array.
		<axis>: the dimension to softmax.
	Return:
		A new array.
	'''
	assert isinstance(data, np.ndarray), f"Expected <data> is NumPy ndarray but got {type_name(data)}."
	if len(data.shape) == 1:
		axis = 0
	assert isinstance(axis, int) and 0 <= axis < len(data.shape), "<axis> is out of range."

	dataShape = list(data.shape)
	dataShape[axis] = 1
	maxValue = data.max(axis, keepdims=True)
	dataNor = data - maxValue
	
	dataExp = np.exp(dataNor)
	dataExpSum = np.sum(dataExp,axis)
	dataExpSumLog = np.log(dataExpSum) + maxValue.reshape(dataExpSum.shape)
	
	return data - dataExpSumLog.reshape(dataShape)

def accuracy(ref, hyp, ignore=None, mode='all'):
	'''
	Score one-2-one matching score between two items.

	Args:
		<ref>, <hyp>: iterable objects like list, tuple or NumPy array. It will be flattened before scoring.
		<ignore>: Ignoring specific symbols.
		<model>: If <mode> is "all", compute one-one matching score. For example, <ref> is (1,2,3,4), and <hyp> is (1,2,2,4), the score will be 0.75.
				 If <mode> is "present", only the members of <hyp> which appeared in <ref> will be scored no matter which position it is. 
	Return:
		a namedtuple object of score information.
	'''
	assert type_name(ref)!="Transcription" and type_name(hyp) != "Transcription", "Exkaldi Transcription objects are unsupported in this function."

	assert mode in ['all','present'], 'Expected <mode> to be "present" or "all".'

	x = flatten(ref)
	x = list( filter(lambda i:i!=ignore, x) ) 
	y = flatten(hyp)
	y = list( filter(lambda i:i!=ignore, y) ) 

	if mode == 'all':
		i = 0
		score = 0
		while True:
			if i >= len(x) or i >= len(y):
				break
			elif x[i] == y[i]:
				score += 1
			i += 1
		if i < len(x) or i < len(y):
			raise WrongOperation('<ref> and <hyp> have different length to score.')
		else:
			if len(x) == 0:
				accuracy = 1.0
			else:
				accuracy = score/len(x)

			return namedtuple("Score",["accuracy", "items", "rightItems"])(
						accuracy, len(x), score
					)
	else:
		x = sorted(x)
		score = 0
		for i in y:
			if i in x:
				score += 1
		if len(y) == 0:
			if len(x) == 0:
				accuracy = 1.0
			else:
				accuracy = 0.0
		else:
			accuracy = score/len(y)
		
		return namedtuple("Score", ["accuracy", "items", "rightItems"])(
					accuracy, len(y), score
				)

def pure_edit_distance(ref, hyp, ignore=None):
	'''
	Compute edit-distance score.

	Args:
		<ref>, <hyp>: iterable objects like list, tuple or NumPy array. It will be flattened before scoring.
		<ignore>: Ignoring specific symbols.	 
	Return:
		a namedtuple object including score information.	
	'''
	assert isinstance(ref, Iterable), "<ref> is not a iterable object."
	assert isinstance(hyp, Iterable), "<hyp> is not a iterable object."
	
	x = flatten(ref)
	x = list( filter(lambda i:i!=ignore, x) ) 
	y = flatten(hyp)
	y = list( filter(lambda i:i!=ignore, y) ) 

	lenX = len(x)
	lenY = len(y)

	mapping = np.zeros((lenX+1,lenY+1))

	for i in range(lenX+1):
		mapping[i][0] = i
	for j in range(lenY+1):
		mapping[0][j] = j
	for i in range(1,lenX+1):
		for j in range(1,lenY+1):
			if x[i-1] == y[j-1]:
				delta = 0
			else:
				delta = 1       
			mapping[i][j] = min(mapping[i-1][j-1]+delta, min(mapping[i-1][j]+1, mapping[i][j-1]+1))
	
	score = int(mapping[lenX][lenY])
	return namedtuple("Score",["editDistance", "items"])(
				score, len(x)
			)