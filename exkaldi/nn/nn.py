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

from exkaldi.error import *
from exkaldi.utils.utils import make_dependent_dirs,type_name,run_shell_command,flatten
from exkaldi.utils import declare
from collections import namedtuple,Iterable

class Supporter:
	'''
	Supporter is used to manage Neural Network training information.

	Args:
		<outDir>: the output directory of Log information.
	'''      
	def __init__(self,outDir='Result'):
		
		declare.is_valid_dir_name("outDir",outDir)
		make_dependent_dirs(outDir,pathIsFile=False)
		self.outDir = os.path.abspath(outDir)

		self.logFile = os.path.join(self.outDir,'log')
		with open(self.logFile,'w',encoding='utf-8'): pass

		self.currentField = {}
		self.currentFieldIsFloat = {}
		self.globalField = []

		self.lastSavedArch = {}
		self.savedArchs = []
		self.savingThreshold = None

		self._allKeys = []

		self._iterSymbol = -1
		
	def send_report(self,info):
		'''
		Send information and these will be retained untill you do the statistics by using .collect_report().

		Args:
			<info>: a Python dict object includiing names and their values with int or float type.
					such as {"epoch":epoch,"train_loss":loss,"train_acc":acc}
					The value can be Python int,float object,Numpy int,float object or NUmpy ndarray with only one value.
		'''
		declare.is_classes("info",info,dict)

		for name,value in info.items(): 
			assert isinstance(name,str) and len(name) > 0,f"The name of info should be string avaliable but got {type_name(name)}."
			valueDtype = type_name(value)
			if valueDtype.startswith("int"): # Python int object,Numpy int object
				pass

			elif valueDtype.startswith("float"): # Python float object,Numpy float object
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

	def collect_report(self,keys=None,plot=True):
		'''
		Do the statistics of received information. The result will be saved in outDir/log file. 

		Args:
			<keys>: Specify the data wanted to be collected. If "None",collect all data reported. 
			<plot>: If "True",print the statistics result to standard output.
		'''
		if keys is None:
			keys = list(self.currentField)
		elif isinstance(keys,str):
			keys = [keys,]
		elif isinstance(keys,(list,tuple)):
			pass
		else:
			raise WrongOperation("Expected <keys> is string,list or tuple.")
	
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


		with open(self.logFile,'a',encoding='utf-8') as fw:
			fw.write(message + '\n')
		# Print to screen
		if plot is True:
			print(message)
		# Clear
		self.currentField = {}
		self.currentFieldIsFloat = {}

	def save_arch(self,saveFunc,arch,addInfo=None,byKey=None,byMax=True,maxRetain=0):
		'''
		Usage:  obj.save_arch(saveFunc,archs={'model':model,'opt':optimizer})

		Save architecture such as models or optimizers when you use this function.
		Only collected information will be used to check the condition. So data collecting is expected beforehand.

		Args:
			<saveFunc>: a function to save archivements in <arch>. It need a parameter to reseive <arch>,for example:
						When use tensorflow 2.x
							def save_model(arch):
								for fileName,model in arch.items():
									model.save_weights(fileName+".h5")
			<arch>: a dict object whose keys are the name,and values are achivements' object. It will be
			<addInfo>: a reported name. If it is not None,will add the information to saving file name.
			<byKey>: a reported name. If it is not None,save achivements only this value is larger than last saved achivements.
			<byMax>: a bool value. Control the condition of <byKey>.
			<maxRetain>: the max numbers of saved achivements to retain. If 0,retain all.
		''' 
		assert isinstance(arch,dict),"Expected <arch> is dict whose keys are architecture-names and values are architecture-objects."
		declare.is_callable("saveFunc",saveFunc)
		declare.is_non_negative_int("maxRetain",maxRetain)

		#if self.currentField != {}:
		#	self.collect_report(plot=False)
		
		suffix = f"_{self._iterSymbol}"
		self._iterSymbol += 1

		if not addInfo is None:
			assert isinstance(addInfo,(str,list,tuple)),'Expected <addInfo> is string,list or tuple.'
			if isinstance(addInfo,str):
				addInfo = [addInfo,]
			for name in addInfo:
				if not name in self.globalField[-1].keys():
					suffix += f"{name}None"
				else:
					value = self.globalField[-1][name]
					if isinstance(value,float):
						suffix += f"_{name}{value:.4f}".replace(".","")
					else:
						suffix += f"_{name}{value}"

		saveFlag =False

		if byKey is None:
			self.lastSavedArch = {}
			newArchs = {}
			for name in arch.keys():
				fileName = os.path.join(self.outDir,name+suffix)
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
						if isinstance(value,float):
							suffix += f"_{name}{value:.4f}".replace(".","")
						else:
							suffix += f"_{name}{value}"
					fileName = os.path.join(self.outDir,name+suffix)
					newArchs[fileName] = arch[name]
					self.lastSavedArch[name] = fileName
		
		if saveFlag is True:
			# Save
			saveFunc(newArchs)
			# Try to correct the file name
			for name,fileName in self.lastSavedArch.items():
				realFileName = glob( fileName + "*" )
				if len(realFileName) == 0:
					raise WrongOperation(f"Achivement whose name starts with {fileName} should have been saved done but not found.")
				elif len(realFileName) > 1:
					raise WrongOperation(f"More than one achivements whose name start with {fileName} were found.")
				else:
					self.lastSavedArch[name] = realFileName[0]

			self.savedArchs.append( self.lastSavedArch.items() )

			for items in self.savedArchs[0:-maxRetain]:
				for name,fileName in items:
					try:
						if os.path.exists(fileName):
							if os.path.isfile(fileName):
								os.remove(fileName)
							elif os.path.isdir(fileName):
								shutil.rmtree(fileName)
							else:
								raise UnsupportedType(f"Failed to remove {fileName}: It is not a file and directory path.")
					except Exception as e:
						e.args = (f"Failed to remove saved achivements:{fileName}."+"\n"+e.args[0],)
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
   
	def judge(self,key,condition,threshold,byDeltaRatio=False):
		'''
		Usage:  obj.judge('train_loss','<',0.0001,byDeltaRatio=True) or obj.judge('epoch','>=',10)
		
		Check if condition is true. 
		Only collected information will be used to check the condition. So data collecting is expected beforehand.

		Args:
			<key>: the name reported.
			<condition>: a string,condition operators such as ">" or "=="
			<threshold>: a int or float value.
			<byDeltaRatio>: bool value,if true,threshold should be a delta ratio value.
								deltaRatio = abs((value-value_pre)/value) 

		Return:
			True or False. 
		''' 
		declare.is_instance("condition operator",condition,['>','>=','<=','<','==','!='])
		declare.is_classes("threshold",threshold,(int,float))

		#if self.currentField != {}:
		#	self.collect_report(plot=False)
		
		if byDeltaRatio is True:
			p = []
			for i in range(len(self.globalField)-1,-1,-1):
				if key in self.globalField[i].keys():
					p.append(self.globalField[i][key])
				if len(p) == 2:
					value = str(abs((p[0]-p[1])/p[0]))
					return eval( value + condition + str(threshold) )
			return False
		else:
			for i in range(len(self.globalField)-1,-1,-1):
				if key in self.globalField[i].keys():
					value = str(self.globalField[i][key])
					return eval(value + condition + str(threshold))
			return False

	def dump(self,keepItems=False):
		'''
		Usage:  product = obj.dump()
		Get all reported information.

		Args:
			<keepItems>: If True,return a dict object.
						 Else,return a list of dict objects. 
		
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
	A data iterator to load and generate dataset with parallel threads for large-scale corpus.
	'''
	def __init__(self,indexTable,processFunc,batchSize,chunks='auto',otherArgs=None,shuffle=False,retainData=0.0):
		'''
		Args:
			_indexTable_: an ExKaldi IndexTable object whose <filePath> info is necessary.
			_processFunc_: a function receive a IndexTable object return return an iterable dataset object.
										It at least need two arguments to receive ( the data iteator itself, a IndexTable object of a chunk data ).
			_batchSize_: mini batch size.
			_chunks_: how many chunks to split.
			_otherArgs_: other arguments to send to <processFunc>.
			_shuffle_: If True, shuffle a batch data.
			_retainData_: a probability value. how much data to retained for evaluation.
		'''
		declare.is_index_table("indexTable",indexTable)
		declare.is_callable("processFunc",processFunc)	
		declare.is_positive_int("batchSize",batchSize)
		declare.is_bool("shuffle",shuffle)
		declare.in_boundary("retainData",retainData,minV=0.0,maxV=0.9)

		self.__processFunc = processFunc
		self.__batchSize = batchSize
		self.__otherArgs = otherArgs
		self.__shuffle = shuffle
		self.__chunks = chunks

		if chunks != 'auto':
			declare.is_positive_int("chunks",chunks)
		
		totalDataNumber = len(indexTable)
		trainDataNumber = int(  totalDataNumber * (1-retainData) )
		evalDataNumber = totalDataNumber - trainDataNumber
		scpTable = indexTable.shuffle()

		self.__trainTable = scpTable.subset(nHead=trainDataNumber)
		if evalDataNumber > 0:
			self.__evalTable = scpTable.subset(nTail=evalDataNumber)
		else:
			self.__evalTable = None

		if chunks == 'auto':
			#Compute the chunks automatically
			sampleTable = self.__trainTable.subset(nHead=10)
			meanSize = sum([ indexInfo.dataSize for indexInfo in sampleTable.values() ]) / 10
			autoChunkSize = math.ceil(104857600/meanSize)  # 100MB = 102400KB = 104857600 B
			self.__chunks = trainDataNumber//autoChunkSize
			if self.__chunks == 0: 
				self.__chunks = 1

		# split train dataset into N chunks
		self.__make_dataset_bag(shuffle=False)

		# initialize some parameters
		self.__epoch = 0
		self.__currentPosition = 0
		self.__currentEpochPosition = 0
		self.__isNewEpoch = False
		self.__isNewChunk = False
		self.__datasetIndex = 0

		# load the first chunk data
		self.__load_dataset(0)
		self.__currentDataset = self.__nextDataset
		self.__nextDataset = None

		# accumulate counts
		self.__epochSize = len(self.__currentDataset)
		self.__countEpochSizeFlag = True

		# try to load the next chunk
		if self.__chunks > 1:
			self.__datasetIndex = 1
			self.__loadDatasetThread = threading.Thread(target=self.__load_dataset,args=(1,))
			self.__loadDatasetThread.start()

	def __make_dataset_bag(self,shuffle=False):
		'''
		Split train index table into n chunks.
		'''
		if shuffle:
			self.__trainTable.shuffle()
		if self.__chunks > 1:
			self.__datasetBag = self.__trainTable.subset(chunks=self.__chunks)
		else:
			self.__datasetBag = [self.__trainTable,]

	def __load_dataset(self,datasetIndex):
		'''
		Read a chunk data into memory.
		'''
		if self.__otherArgs != None:
			self.__nextDataset = self.__processFunc(self, self.__datasetBag[datasetIndex], self.__otherArgs)
		else:
			self.__nextDataset = self.__processFunc(self, self.__datasetBag[datasetIndex])

		assert isinstance(self.__nextDataset,Iterable),"Process function should return an iterable objects."
		if not isinstance(self.__nextDataset,list):
			self.__nextDataset = [X for X in self.__nextDataset]

		if self.__batchSize > len(self.__nextDataset):
			print(f"Warning: Batch Size <{self.__batchSize}> is extremely large for this dataset, we hope you can use a more suitable value.")
		
	def next(self):
		'''
		Get the next batch data.
		'''
		i = self.__currentPosition
		iEnd = i + self.__batchSize
		N = len(self.__currentDataset)

		batch = self.__currentDataset[i:iEnd]

		if self.__chunks == 1:
			if iEnd >= N:
				rest = iEnd - N
				if self.__shuffle:
					random.shuffle(self.__currentDataset)
				batch.extend(self.__currentDataset[:rest])
				self.__currentPosition = rest
				self.__epoch += 1
				self.__isNewEpoch = True
				self.__isNewChunk = True
			else:
				self.__currentPosition = iEnd
				self.__isNewEpoch = False
				self.__isNewChunk = False
			self.__currentEpochPosition = self.__currentPosition
		else:
			if iEnd >= N:
				rest = iEnd - N
				if self.__loadDatasetThread.is_alive():
					self.__loadDatasetThread.join()
				if self.__shuffle:
					random.shuffle(self.__nextDataset)
				batch.extend(self.__nextDataset[:rest])
				self.__currentPosition = rest
				self.__currentDataset = self.__nextDataset
				self.__isNewChunk = True
				
				if self.__countEpochSizeFlag:
					self.__epochSize += len(self.__currentDataset)

				self.__datasetIndex = (self.__datasetIndex+1)%self.__chunks

				if self.__datasetIndex == 1:
					self.__epoch += 1
					self.__isNewEpoch = True

				if self.__datasetIndex == 0:
					self.__countEpochSizeFlag = False
					del self.__datasetBag
					self.__make_dataset_bag(shuffle=True)

				self.__loadDatasetThread = threading.Thread(target=self.__load_dataset,args=(self.__datasetIndex,))
				self.__loadDatasetThread.start()

			else:
				self.__isNewChunk = False
				self.__isNewEpoch = False
				self.__currentPosition = iEnd

			self.__currentEpochPosition = (self.__currentEpochPosition + self.__batchSize)%self.__epochSize

		return batch                          

	@property
	def batchSize(self):
		'''
		Get the batch size.

		Return:
			an int value.
		'''
		return self.__batchSize

	@property
	def chunks(self):
		'''
		Get the chunk size.

		Return:
			an int value.
		'''
		return self.__chunks

	@property
	def chunk(self):
		'''
		Get the current chunk ID.

		Return:
			an int value.
		'''		
		if self.__datasetIndex == 0:
			return self.__chunks - 1
		else:
			return self.__datasetIndex - 1

	@property
	def epoch(self):
		'''
		Get the current epoch ID.

		Return:
			an int value.
		'''	
		return self.__epoch

	@property
	def isNewEpoch(self):
		'''
		If it is a new epoch now, return True. 
		'''	
		return self.__isNewEpoch

	@property
	def isNewChunk(self):
		'''
		If it is a new chunk now, return True. 
		'''	
		return self.__isNewChunk

	@property
	def epochProgress(self):
		'''
		Get the epoch progress of current batch data.
		It is not precise at the first epoch because we will accumulate the counts gradually.

		Return:
			a float value within [0,1].
		'''	
		if self.__isNewEpoch is True:
			return 1.
		else:
			return round(self.__currentEpochPosition/self.__epochSize,4)
	
	@property
	def chunkProgress(self):
		'''
		Get the chunk progress of current batch data.

		Return:
			a float value within [0,1].
		'''	
		if self.__isNewChunk is True:
			return 1.
		else:
			return round(self.__currentPosition/len(self.__currentDataset),4)

	def get_retained_data(self,processFunc=None,batchSize=None,chunks='auto',otherArgs=None,shuffle=False,retainData=0.0):
		'''
		Get the retained data.

		Args:
			_processFunc_: if None, use the function of parent iterator.
			_batchSize_: if None, use the value of parent iterator.
			_batchSize_: 'auto' or an int value.
			_otherArgs_: if None, use the same values with other args.
			_shuffle_: whether shuffle a chunk data.
			_retainData_: a probability value. how much data to retained for evaluation.

		Return:
			A new DataIterator object.
		'''
		declare.not_void("retained data",self.__evalTable)

		if processFunc is None:
			processFunc = self.__processFunc
		
		if batchSize is None:
			batchSize = self.__batchSize

		if otherArgs is None:
			otherArgs = self.__otherArgs

		return DataIterator(self.__evalTable,processFunc,batchSize,chunks,otherArgs,shuffle,retainData)

def pad_sequence(data,dim=0,maxLength=None,dtype='float32',padding='tail',truncating='tail',value=0.0):
	'''
	Pad sequence.

	Args:
		<data>: a list of NumPy arrays.
		<dim>: which dimmension to pad. All other dimmensions should be the same size.
		<maxLength>: If larger than this theshold,truncate it.
		<dtype>: target dtype.
		<padding>: padding position,"head","tail" or "random".
		<truncating>: truncating position,"head","tail".
		<value>: padding value.
	
	Return:
		a two-tuple: (a Numpy array,a list of padding positions). 
	'''
	declare.is_classes("data",data,(list,tuple))
	declare.is_non_negative_int("dim",dim)
	declare.not_void("data",data)
	declare.is_classes("value",value,(int,float))
	declare.is_instances("padding",padding,["head","tail","random"])
	declare.is_instances("truncating",padding,["head","tail"])
	if maxLength is not None:
		declare.is_positive_int("maxLength",maxLength)

	lengths = []
	newData = []
	exRank = None
	exOtherDims = None
	for i in data:

		# verify
		declare.is_classes("data",i,np.ndarray)
		shape = i.shape
		if exRank is None:
			exRank = len(shape)
			assert dim < exRank,f"<dim> is out of range: {dim}>{exRank-1}."
		else:
			assert len(shape) == exRank,f"Arrays in <data> has different rank: {exRank}!={len(shape)}."

		if dim != 0:
			# transpose
			rank = [r for r in range(exRank)]
			rank[0] = dim
			rank[dim] = 0
			i = i.transpose(rank)

		if exOtherDims is None:
			exOtherDims = i.shape[1:]
		else:
			assert exOtherDims == i.shape[1:],f"Expect for sequential dimmension,All arrays in <data> has same shape but got: {exOtherDims}!={i.shape[1:]}."

		length = len(i)
		if maxLength is not None and length > maxLength:
			if truncating == "head":
				i = i[maxLength:,...]
			else:
				i = i[0:maxLength:,...]

		lengths.append(len(i))
		newData.append(i)

	maxLength = max(lengths)
	batchSize = len(newData)

	result = np.array(value,dtype=dtype) * np.ones([batchSize,maxLength,*exOtherDims],dtype=dtype)

	pos = []
	for i in range(batchSize):
		length = lengths[i]
		if padding == "tail":
			result[i][0:length] = newData[i]
			pos.append((0,length))
		elif padding == "head":
			start = maxLength - length
			result[i][start:] = newData[i]
			pos.append((start,maxLength))
		else:
			start = random.randint(0,maxLength-length)
			end = start + length
			result[i][start:end] = newData[i]
			pos.append((start,end))

	if dim != 0:
		exRank = len(result.shape)
		rank = [r for r in range(exRank)]
		rank[1] = dim+1
		rank[dim+1] = 1
		result = result.transpose(rank)

	return result,pos

def softmax(data,axis=1):
	'''
	The softmax function.

	Args:
		<data>: a Numpy array.
		<axis>: the dimension to softmax.
		
	Return:
		A new array.
	'''
	declare.is_classes("data",data,np.ndarray)
	if len(data.shape) == 1:
		axis = 0
	declare.in_boundary("axis",axis,0,len(data.shape)-1 )
	
	maxValue = data.max(axis,keepdims=True)
	dataNor = data - maxValue

	dataExp = np.exp(dataNor)
	dataExpSum = np.sum(dataExp,axis,keepdims = True)

	return dataExp / dataExpSum

def log_softmax(data,axis=1):
	'''
	The log-softmax function.

	Args:
		<data>: a Numpy array.
		<axis>: the dimension to softmax.
	Return:
		A new array.
	'''
	declare.is_classes("data",data,np.ndarray)
	if len(data.shape) == 1:
		axis = 0
	declare.in_boundary("axis",axis,0,len(data.shape)-1 )

	dataShape = list(data.shape)
	dataShape[axis] = 1
	maxValue = data.max(axis,keepdims=True)
	dataNor = data - maxValue
	
	dataExp = np.exp(dataNor)
	dataExpSum = np.sum(dataExp,axis)
	dataExpSumLog = np.log(dataExpSum) + maxValue.reshape(dataExpSum.shape)
	
	return data - dataExpSumLog.reshape(dataShape)

def accuracy(ref,hyp,ignore=None,mode='all'):
	'''
	Score one-2-one matching score between two items.

	Args:
		<ref>,<hyp>: iterable objects like list,tuple or NumPy array. It will be flattened before scoring.
		<ignore>: Ignoring specific symbols.
		<model>: If <mode> is "all",compute one-one matching score. For example,<ref> is (1,2,3,4),and <hyp> is (1,2,2,4),the score will be 0.75.
				 If <mode> is "present",only the members of <hyp> which appeared in <ref> will be scored no matter which position it is. 
	Return:
		a namedtuple object of score information.
	'''
	assert type_name(ref)!="Transcription" and type_name(hyp) != "Transcription","Exkaldi Transcription objects are unsupported in this function."

	assert mode in ['all','present'],'Expected <mode> to be "present" or "all".'

	x = flatten(ref)
	x = list( filter(lambda i:i!=ignore,x) ) 
	y = flatten(hyp)
	y = list( filter(lambda i:i!=ignore,y) ) 

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

			return namedtuple("Score",["accuracy","items","rightItems"])(
						accuracy,len(x),score
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
		
		return namedtuple("Score",["accuracy","items","rightItems"])(
					accuracy,len(y),score
				)

def pure_edit_distance(ref,hyp,ignore=None):
	'''
	Compute edit-distance score.

	Args:
		<ref>,<hyp>: iterable objects like list,tuple or NumPy array. It will be flattened before scoring.
		<ignore>: Ignoring specific symbols.	 
	Return:
		a namedtuple object including score information.	
	'''
	assert isinstance(ref,Iterable),"<ref> is not a iterable object."
	assert isinstance(hyp,Iterable),"<hyp> is not a iterable object."
	
	x = flatten(ref)
	x = list( filter(lambda i:i!=ignore,x) ) 
	y = flatten(hyp)
	y = list( filter(lambda i:i!=ignore,y) ) 

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
			mapping[i][j] = min(mapping[i-1][j-1]+delta,min(mapping[i-1][j]+1,mapping[i][j-1]+1))
	
	score = int(mapping[lenX][lenY])
	return namedtuple("Score",["editDistance","items"])(
				score,len(x)
			)

def compute_postprob_norm(ali,probDims):
	'''
	Compute alignment counts in order to normalize acoustic model posterior probability.
	For more help information,look at the Kaldi <analyze-counts> command.

	Args:
		<ali>: exkaldi NumpyAliTrans,NumpyAliPhone or NumpyAliPdf object.
		<probDims>: the dimensionality of posterior probability.
		
	Return:
		A numpy array of the normalization.
	''' 
	declare.kaldi_existed()
	declare.is_classes("ali",ali,["NumpyAliTrans","NumpyAliPhone","NumpyAliPdf"])
	declare.is_positive_int("probDims",probDims)

	txt = []
	for key,vlaue in ali.items():
		value = " ".join(map(str,vlaue.tolist()))
		txt.append( key+" "+value )
	txt = "\n".join(txt)

	cmd = f"analyze-counts --binary=false --counts-dim={probDims} ark:- -"
	out,err,cod = run_shell_command(cmd,stdin="PIPE",stdout="PIPE",stderr="PIPE",inputs=txt)
	if (isinstance(cod,int) and cod != 0) or out == b"":
		raise KaldiProcessError('Analyze counts defailed.',err.decode())
	else:
		out = out.decode().strip().strip("[]").strip().split()
		counts = np.array(out,dtype=np.float32)
		countBias = np.log(counts/np.sum(counts))
		return countBias