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
import random
import subprocess
import tempfile
import numpy as np
import threading
import math
import time

from exkaldi.version import PathError
from exkaldi.utils.utils import make_dependent_dirs, type_name, run_shell_command
from exkaldi.utils.utils import WrongOperation, UnsupportedDataType
from exkaldi.core.load import load_feat

class Supporter:
	'''
	Usage:  supporter = Supporter(outDir='Result')

	Supporter is used to manage training information such as the change of loss and others.
	'''      
	def __init__(self, outDir='Result'):

		self.currentField = {}
		self.globalField = []

		assert isinstance(outDir, str), "<outDir> should be a name-like string."
		make_dependent_dirs(outDir, pathIsFile=False)
		self.outDir = os.path.abspath(outDir)
		self.logFile = os.path.join(self.outDir,'log')
		with open(self.logFile, 'w', encoding='utf-8'):
			pass
		
		self.lastSavedArch = {}
		self.savingThreshold = None

		self._allKeys = []

		self._iterSymbol = -1
		
	def send_report(self, info):
		'''
		Usage:  supporter = obj.send_report({"epoch":epoch,"train_loss":loss,"train_acc":acc})

		Send information and these will be retained untill you do the statistics by using .collect_report().
		<info> should be a dict of names and their values with int or float type. 
		'''
		assert isinstance(info, dict), "Expected <info> is a Python dict object."

		keys = list(info)
		allKeys = list(self.currentField)
	
		for i in keys: 
			assert isinstance(i, str), f"Expected name-like string but got {type_name(i)}."
			value = info[i]
			assert isinstance(value, (int,float)), f"Expected int or float value but got {type_name(value)}."
			i = i.lower()
			if not i in allKeys:
				self.currentField[i] = []
			self.currentField[i].append(value)

	def collect_report(self, keys=None, plot=True):
		'''
		Usage:  supporter = obj.collect_report(plot=True)

		Do the statistics of received information. The result will be saved in outDir/log file. 
		If <keys> is not "None", only collect the data in <keys>. 
		If <plot> is "True", print the statistics result to standard output.
		'''
		if keys is None:
			keys = list(self.currentField)
		elif isinstance(keys,str):
			keys = [keys,]
		elif isinstance(keys,(list,tuple)):
			pass
		else:
			raise WrongOperation("Expected <keys> is string, list or tuple.")
	
		self.globalField.append({})

		allKeys = list(self.currentField.keys())
		self._allKeys.extend(allKeys)
		self._allKeys = list(set(self._allKeys))

		message = ''
		for i in keys:
			if i in allKeys:
				mn = sum(self.currentField[i])/len(self.currentField[i])
				if type(self.currentField[i][0]) == int:
					mn = int(mn)
					message += (i + ':%d    '%(mn))
				else:
					message += (i + ':%.5f    '%(mn))
				self.globalField[-1][i] = mn
			else:
				message += (i + ':-----    ')

		with open(self.logFile, 'a', encoding='utf-8') as fw:
			fw.write(message + '\n')
		# Print to screen
		if plot is True:
			print(message)
		# Clear
		self.currentField = {}

	def save_arch(self, saveFunc, arch, addInfo=None, byKey=None, byMax=True):
		'''
		Usage:  obj.save_arch(saveFunc,archs={'model':model,'opt':optimizer})

		Save architecture such as models or optimizers when you use this function.
		If you use <byKey> and set <byMax>,  will be saved only while meeting the condition. 
		<archs> will be sent to <saveFunc> but with new name.
		''' 
		assert isinstance(arch, dict), "Expected <arch> is dict whose keys are architecture-names and values are architecture-objects."

		if self.currentField != {}:
			self.collect_report(plot=False)
		
		suffix = "_"+str(self._iterSymbol)
		self._iterSymbol += 1

		if not addInfo is None:
			assert isinstance(addInfo, (str,list,tuple)), 'Expected <addInfo> is string, list or tuple.'
			if isinstance(addInfo, str):
				addInfo = [addInfo,]
			for i in addInfo:
				if not i in self.globalField[-1].keys():
					continue
				value = self.globalField[-1][i]
				if isinstance(value, float):
					suffix += ( "_" + i + (f"{value:.4f}".replace(".","")))
				else:
					suffix += ( "_" + i + f'{value}')             

		if byKey == None:
			newArchs = []
			for name in arch.keys():
				fileName = os.path.join(self.outDir, name+suffix)
				newArchs.append((fileName, arch[name]))
				self.lastSavedArch[name] = fileName
			if len(newArchs) == 1:
				newArchs = newArchs[0]
			saveFunc(newArchs)
		else:
			byKey = byKey.lower()
			if not byKey in self.globalField[-1].keys():
				print(f"Warning: Save architectures defeat. Because the key {byKey} has not been reported.")
				return
			else:
				value = self.globalField[-1][byKey]

			save = False

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
				newArchs = []
				for name in arch.keys():
					if isinstance(value, float):
						suffix += ( "_" + byKey + (f'{value:.4f}'.replace('.','') ))
					else:
						suffix += ( "_" + byKey + f'{value}' )
					fileName = os.path.join(self.outDir, name+suffix)
					newArchs.append((fileName, arch[name]))
					if name in self.lastSavedArch.keys():
						os.remove(self.lastSavedArch[name])                    
					self.lastSavedArch[name] = fileName
				if len(newArchs) == 1:
					newArchs = newArchs[0]
				saveFunc(newArchs)

	@property
	def finalArch(self):
		'''
		Usage:  model = obj.finalArch

		Get the final saved model. Return a Python dict object whose key is architecture name and value is architecture path. 
		''' 
		return self.lastSavedArch
   
	def judge(self, key, condition, threshold, byDeltaRatio=False):
		'''
		Usage:  obj.judge('train_loss','<',0.0001,byDeltaRatio=True) or obj.judge('epoch','>=',10)

		Return "True" or "False". And If <key> is not reported before, return "False".
		if <byDeltaRate> is "True", we compute:
						   abs((value-value_pre)/value)  
		and compare it with threshold value.
		''' 
		assert condition in ['>','>=','<=','<','==','!='], '<condiction> is not a correct conditional operator.'
		assert isinstance(threshold,(int,float)), '<threshold> should be a float or int value.'

		if self.currentField != {}:
			self.collect_report(plot=False)
		
		if byDeltaRatio == True:
			p = []
			for i in range(len(self.globalField)-1,-1,-1):
				if key in self.globalField[i].keys():
					p.append(self.globalField[i][key])
				if len(p) == 2:
					value = str(abs((p[0]-p[1])/p[0]))
					return eval(value+condition+str(threshold))
			return False
		else:
			for i in range(len(self.globalField)-1,-1,-1):
				if key in self.globalField[i].keys():
					value = str(self.globalField[i][key])
					return eval(value+condition+str(threshold))
			return False

	def dump(self, keepItems=False, fromLogFile=None):
		'''
		Usage:  product = obj.dump()

		Get all reported information.
		If <fromLogFile> is not "None", read and return information from log file.
		If <keepItems> is True, return information by iterms name. 
		'''
		
		if fromLogFile != None:
			assert isinstance(fromLogFile, str), "Expected <fromLogFile> is file name-like string."
			if not os.path.isfile(fromLogFile):
				raise PathError('No such file:{}.'.format(fromLogFile))
			else:
				with open(fromLogFile, 'r', encoding='utf-8') as fr:
					lines = fr.readlines()
				allData = []
				for line in lines:
					line = line.strip()
					if len(line) != "":
						lineData = {}
						line = line.split()
						for i in line:
							i = i.split(":")
							if "-----" in i[1]:
								continue
							else: 
								try:
									v = int(i[1])
								except ValueError:
									v = float(i[1])
							lineData[i[0]] = v
						allData.append(lineData)
				if len(allData) == 0:
					raise WrongOperation('Not any information to dump.')
		else:
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

class DataIterator(object):
	'''
	Usage: obj = DataIterator('train.scp',64,chunks='auto',processFunc=function)

	If you give it a large scp file of train data, it will split it into N smaller chunks and load them into momery alternately with parallel thread. 
	It will shuffle the original scp file and split again while new epoch.
	'''

	def __init__(self,scpFiles,processFunc,batchSize,chunks='auto',otherArgs=None,shuffle=False,retainData=0.0):

		self.fileProcessFunc = processFunc
		self._batchSize = batchSize
		self.otherArgs = otherArgs
		self._shuffle = shuffle
		self._chunks = chunks

		if isinstance(scpFiles, str):
			out, err, _ = run_shell_command(f"ls {scpFiles}", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
			if out == b'':
				raise PathError(f"No such file:{scpFiles}")
			else:
				out = out.decode().strip().split('\n')
		elif isinstance(scpFiles, list):
			out = scpFiles
		else:
			raise UnsupportedDataType('Expected <scpFiles> is scp file-like string or list object.')

		if isinstance(chunks,int):
			assert chunks>0, "Expected <chunks> is a positive int number but got {}.".format(chunks)
		elif chunks != 'auto':
			raise WrongOperation('Expected <chunks> is a positive int number or <auto> but got {}.'.format(chunks))

		temp = []
		for scpFile in out:
			with open(scpFile, 'r', encoding='utf-8') as fr:
				temp.extend(fr.read().strip().split('\n'))
		K = int(len(temp)*(1-retainData))
		self.retainedFiles = temp[K:]
		self.allFiles = temp[0:K]

		if chunks == 'auto':
			#Compute the chunks automatically
			sampleChunk = random.sample(self.allFiles, 10)
			with tempfile.NamedTemporaryFile('w', encoding='utf-8', suffix='.scp') as sampleFile:
				sampleFile.write('\n'.join(sampleChunk))
				sampleFile.seek(0)
				sampleChunkData = load_feat(sampleFile.name)
			meanLength = int(np.mean(sampleChunkData.lens[1]))
			autoChunkSize = math.ceil(50000/meanLength)  # Use 50000 frames as threshold 
			self._chunks = len(self.allFiles)//autoChunkSize
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

	def make_dataset_bag(self,shuffle=False):
		self.datasetBag = []
		if shuffle:
			random.shuffle(self.allFiles)
		chunkSize = math.ceil(len(self.allFiles)/self._chunks)
		L = self._chunks -(chunkSize * self._chunks - len(self.allFiles))-1
		start = 0
		for i in range(self._chunks):
			if i > L:
				end = start + chunkSize - 1
			else:
				end = start + chunkSize
			chunkFiles = self.allFiles[start:end]
			start = end
			if len(chunkFiles) > 0:
				self.datasetBag.append(chunkFiles)

	def load_dataset(self,datasetIndex):
		with tempfile.NamedTemporaryFile('w', suffix='.scp') as scpFile:
			scpFile.write('\n'.join(self.datasetBag[datasetIndex]))
			scpFile.seek(0)
			chunkData = load_feat(scpFile.name)
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

		if len(self.retainedFiles) == 0:
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

		with tempfile.NamedTemporaryFile('w', encoding='utf-8', suffix='.scp') as reScpFile:
			reScpFile.write('\n'.join(self.retainedFiles))
			reScpFile.seek(0)  
			reIterator = DataIterator(reScpFile.name, processFunc, batchSize, chunks, otherArgs, shuffle, retainData)

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
	assert isinstance(axis, int) and 0 <= axis < len(data.shape), "<axis> is out of range."

	dataShape = list(data.shape)
	dataShape[axis] = 1
	maxValue = data.max(axis, keepdims=True)
	dataNor = data - maxValue
	
	dataExp = np.exp(dataNor)
	dataExpSum = np.sum(dataExp,axis)
	dataExpSumLog = np.log(dataExpSum) + maxValue.reshape(dataExpSum.shape)
	
	return data - dataExpSumLog.reshape(dataShape)