# coding=utf-8
#
# Yu Wang (University of Yamanashi)
# May, 2020
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

import copy
from io import BytesIO
import numpy as np
import random
import struct
import subprocess
import os
import tempfile
from collections import namedtuple
import sys

from exkaldi.version import info as ExkaldiInfo
from exkaldi.version import WrongPath, WrongOperation, WrongDataFormat, UnsupportedType, ShellProcessError, KaldiProcessError
from exkaldi.utils.utils import type_name, run_shell_command, make_dependent_dirs, list_files
from exkaldi.utils.utils import FileHandleManager
from exkaldi.utils import declare

''' ListTable class group''' 

class ListTable(dict):
	'''
	This is a subclass of Python dict.
	You can use it to hold kaldi text format tables, such as scp-files, utt2spk and so on. 
	'''
	def __init__(self, data={}, name="table"):
		super(ListTable, self).__init__(data)
		declare.is_valid_string("name", name)
		self.__name = name
		
	@property
	def is_void(self):
		'''
		If there is not any data, return True, or return False.
		'''
		if len(self.keys()) == 0:
			return True
		else:
			return False

	@property
	def name(self):
		'''
		Get its name.
		'''
		return self.__name

	@property
	def data(self):
		'''
		Return inner dict object.
		'''
		return dict(self)

	def rename(self, name):
		'''
		Rename.

		Args:
			<name>: a string.
		'''
		declare.is_valid_string("name",name)
		self.__name = name

	def reset_data(self, data=None):
		'''
		Reset the data.

		Args:
			<data>: a iterable object that can be converted to dict object.
		'''
		if data is None:
			self.clear()
		else:
			newData = dict(data)
			self.clear()
			self.update(newData)

	def sort(self, reverse=False):
		'''
		Sort by key.

		Args:
			<reverse>: If reverse, sort in descending order.
		Return:
			A new ListTable object.
		''' 
		items = sorted(self.items(), key=lambda x:x[0], reverse=reverse)
		newName = f"sort({self.name})"
		return ListTable(items, name=newName)

	def save(self, fileName=None, chunks=1, concatFunc=None):
		'''
		Save to file.

		Args:
			<fileName>: file name, opened file handle or None.
		
		Return:
			file name, None, or a string including all contents of this ListTable. 
		'''
		declare.not_void(type_name(self), self)
		if fileName is not None:
			declare.is_valid_file_name_or_handle("fileName", fileName)
		declare.in_boundary("chunks", chunks, minV=1)

		def purely_concat(item):
			try:
				return f"{item[0]} {item[1]}"
			except Exception:
				print(f"Utterance ID: {item[0]}")
				raise WrongDataFormat(f"Wrong key and value format: {type_name(item[0])} and {type_name(item[1])}. ")
		
		def save_chunk_data(chunkData, concatFunc, fileName):
			contents = "\n".join(map(concatFunc, chunkData.items())) + "\n"
			if fileName is None:
				return contents
			else:
				make_dependent_dirs(fileName, pathIsFile=True)
				with open(fileName, "w", encoding="utf-8") as fw:
					fw.write(contents)
				return fileName				

		if concatFunc is not None:
			declare.is_callable("concatFunc", concatFunc)
		else:
			concatFunc = purely_concat

		if fileName is None:
			return save_chunk_data(self, concatFunc, None)

		elif isinstance(fileName, str):

			if chunks == 1:
				return save_chunk_data(self, concatFunc, fileName)

			else:
				dirName = os.path.dirname(fileName)
				fileName = os.path.basename(fileName)
				savedFiles = []
				chunkDataList = self.subset(chunks=chunks)
				for i, chunkData in enumerate(chunkDataList):
					chunkFileName = os.path.join( dirName, f"ck{i}_{fileName}" )
					savedFiles.append( save_chunk_data(chunkData, concatFunc, chunkFileName) )
				
				return savedFiles		
				
		else:
			results = save_chunk_data(self, concatFunc, None)
			fileName.truncate()
			fileName.write(results)
			fileName.seek(0)
			
			return fileName

	def shuffle(self):
		'''
		Random shuffle the list table.

		Return:
			A new ListTable object.
		'''
		items = list(self.items())
		random.shuffle(items)
		newName = f"shuffle({self.name})"
		return ListTable(items, name=newName)
	
	def subset(self, nHead=0, nTail=0, nRandom=0, chunks=1, uttIDs=None):
		'''
		Subset.
		Only one mode will do when it is not the default value. 
		The priority order is: nHead > nTail > nRandom > chunks > uttIDs.
		
		Args:
			<nHead>: If it > 0, extract N head utterances.
			<nTail>: If it > 0, extract N tail utterances.
			<nRandom>: If it > 0, randomly sample N utterances.
			<chunks>: If it > 1, split data into N chunks.
			<uttIDs>: If it is not None, pick out these utterances whose ID in uttIDs.
		
		Return:
			a new ListTable object or a list of new ListTable objects.
		''' 
		declare.not_void(type_name(self), self)

		if nHead > 0:
			declare.is_positive_int("nHead",nHead)
			new = list(self.items())[0:nHead]
			newName = f"subset({self.name},head {nHead})"
			return ListTable(new, newName)
		
		elif nTail > 0:
			declare.is_positive_int("nTail",nTail)
			new = list(self.items())[-nTail:]
			newName = f"subset({self.name},tail {nTail})"
			return ListTable(new, newName)		

		elif nRandom > 0:
			declare.is_positive_int("nRandom",nRandom)
			new = random.choices(list(self.items()), k=nRandom)
			newName = f"subset({self.name},random {nRandom})"
			return ListTable(new, newName)	

		elif chunks > 1:
			declare.is_positive_int("chunks",chunks)
			datas = []
			allLens = len(self.keys())
			if allLens != 0:
				chunkUtts = allLens//chunks
				if chunkUtts == 0:
					chunks = allLens
					chunkUtts = 1
					t = 0
				else:
					t = allLens - chunkUtts * chunks

				items = list(self.items())
				for i in range(chunks):
					temp = {}
					if i < t:
						chunkuttIDs = items[i*(chunkUtts+1):(i+1)*(chunkUtts+1)]
					else:
						chunkuttIDs = items[i*chunkUtts:(i+1)*chunkUtts]
					temp.update(chunkuttIDs)
					newName = f"subset({self.name},chunk {chunks}-{i})"
					datas.append( ListTable(temp, newName) )
			return datas

		elif uttIDs != None:
			declare.is_classes("uttIDs", uttIDs, (str,list,tuple))
			
			if isinstance(uttIDs,str):
				newName = f"subset({self.name},uttIDs 1)"
				uttIDs = [uttIDs,]
			else:
				newName = f"subset({self.name},uttIDs {len(uttIDs)})"

			newDict = {}
			for utt in uttIDs:
				if utt in self.keys():
					newDict[utt] = self[utt]
				else:
					#print('Subset Warning: no data for utt {}'.format(utt))
					continue
			#if len(newDict) == 0:
			#	raise WrongDataFormat("Missed all utterances in <uttIDs>. We do not think it is an reasonable result.")

			return ListTable(newDict, newName)
		
		else:
			raise WrongOperation("Expected any of modes to subset.")

	def __add__(self, other):
		'''
		Integrate two ListTable objects. If key existed in both two objects, the former will be retained.

		Args:
			<other>: another ListTable object.
		Return:
			A new ListTable object.
		'''
		declare.belong_classes("other", other, ListTable)

		new = copy.deepcopy(other)
		new.update(self)

		newName = f"add({self.name},{other.name})"
		new.rename(newName)

		return new

	def reverse(self):
		'''
		Exchange the position of key and value. 

		Key and value must be one-one matching, or Error will be raised.
		'''

		newname = f"reverse({self.name})"
		new = ListTable(name=newname)

		for key,value in self.items():
			try:
				_ = new[value]
			except KeyError:
				new[value] = key
			else:
				raise WrongDataFormat(f"Only one-one matching table can be reversed but mutiple {value} have existed.")
		
		return new

## New subclass in version 1.3
class ArkIndexTable(ListTable):
	'''
	For accelerate to find utterance and reduce memory cost of intermidiate operation.
	This is used to hold the utterance index informat of Kaldi archive table (binary format). It just like the script-table file but is more useful.
	Its format like this:
	{ "utt0": namedtuple(frames=100, startIndex=1000, dataSize=10000, filePath="./feat.ark") }
	'''
	def __init__(self, data={}, name="indexTable"):
		super(ArkIndexTable, self).__init__(data, name)
		# Check the format of index.
		for key, value in self.items():
			declare.is_classes("value", value, [list,tuple,"Index"])
			if isinstance(value, (list,tuple)):
				assert len(value) in [3,4], f"Expected (frames, start index, data size[, file path]) but {value} does not match."
				self[key] = self.spec(*value)

	@property
	def spec(self):
		'''
		The index info spec.
		'''
		spec = namedtuple("Index",["frames", "startIndex", "dataSize", "filePath"])
		spec.__new__.__defaults__ = (None,)
		return spec

	@property
	def utts(self):
		return list(self.keys())

	def sort(self, by="utt", reverse=False):
		'''
		Sort utterances by frame length, utterance ID or start index.

		Args:
			<by>: "frame" or "utt" or "startIndex".
			<reverse>: If True, sort in descending order.
		
		Return:
			A new ArkIndexTable object.
		''' 
		declare.is_instances("by", by, ["utt", "frame", "startIndex"])

		if by == "utt":
			items = sorted(self.items(), key=lambda x:x[0], reverse=reverse)
		elif by == "frame":
			items = sorted(self.items(), key=lambda x:x[1].frames, reverse=reverse)
		else:
			items = sorted(self.items(), key=lambda x:x[1].startIndex, reverse=reverse)
		
		newName = f"sort({self.name},{by})"
		return ArkIndexTable(items, name=newName)

	def __add__(self, other):
		'''
		Integrate two ArkIndexTable objects. If utterance has existed in both two objects, the former will be retained.

		Args:
			<other>: another ArkIndexTable object.
		Return:
			A new ArkIndexTable object.
		'''
		declare.is_classes("other", other, ArkIndexTable)

		result = super().__add__(other)
		return ArkIndexTable(result, name=result.name)

	def shuffle(self):
		'''
		Random shuffle the index table.

		Return:
			A new ArkIndexTable object.
		'''
		result = super().shuffle()

		return ArkIndexTable(result, name=self.name)
	
	def subset(self, nHead=0, nTail=0, nRandom=0, chunks=1, uttIDs=None):
		'''
		Subset.
		Only one mode will work when it is not the default value. 
		The priority order is: nHead > nTail > nRandom > chunks > uttIDs.
		
		Args:
			<nHead>: If it > 0, extract N head utterances.
			<nTail>: If it > 0, extract N tail utterances.
			<nRandom>: If it > 0, randomly sample N utterances.
			<chunks>: If it > 1, split data into N chunks.
			<uttIDs>: If it is not None, pick out these utterances whose ID in uttIDs.
		Return:
			a new ArkIndexTable object or a list of new ArkIndexTable objects.
		''' 
		result = super().subset(nHead,nTail,nRandom,chunks,uttIDs)

		if isinstance(result, list):
			for index in range(len(result)):
				temp = result[index]
				result[index] = ArkIndexTable(temp, temp.name)
		else:
			result = ArkIndexTable(result, result.name)

		return result

	def save(self, fileName=None, chunks=1):
		'''
		Save this index informat to text file with kaidi script-file table format.
		Note that the frames informat will be discarded.

		Args:
			<fileName>: file name or file handle. 
		
		Return:
			file name or None or the contents of ListTable.
		'''
		declare.not_void(type_name(self), self)

		def concat(item):
			utt, indexInfo = item
			if indexInfo.filePath is None:
				raise WrongOperation("Cannot save to script file becase miss the archieve file path informat.")
			else:
				startIndex = indexInfo.startIndex + len(utt) + 1
				return f"{utt} {indexInfo.filePath}:{startIndex}"

		return super().save(fileName, chunks, concat)
	
	def fetch(self, arkType=None, uttIDs=None):
		"""
		Fetch records from file.

		Args:
			<uttID>: utterance ID or a list of utterance IDs.
			<arkType>: If None, return BytesMatrix or BytesVector object.
					   If "feat", return BytesFeature object.
					   If "cmvn", return BytesFeature object.
					   If "prob", return BytesFeature object.
					   If "ali", return BytesFeature object.
					   If "fmllrMat", return BytesFeature object.
					   If "mat", return BytesMatrix object.
					   If "vec", return BytesVector object.
		
		Return:
		    an exkaldi bytes achieve object. 
		"""
		declare.not_void(type_name(self), self)
		declare.is_instances("arkType", arkType, [None,"feat","cmvn","prob","ali","fmllrMat"])

		if uttIDs is None:
			uttIDs = self.keys()
		else:
			declare.is_classes("uttIDs", uttIDs, [str, list, tuple])
			if isinstance(uttIDs, str):
				uttIDs = [uttIDs,]
			declare.members_are_valid_strings("uttIDs", uttIDs)

		newTable = ArkIndexTable()

		with FileHandleManager() as fhm:

			startIndex = 0
			datas = []
			for uttID in uttIDs:
				try:
					indexInfo = self[uttID]
				except KeyError:
					continue
				else:
					if indexInfo.filePath is None:
						raise WrongDataFormat(f"Miss file path information in the index table: {uttID}.")
					
					fr = fhm.call(indexInfo.filePath)
					if fr is None:
						fr = fhm.open(indexInfo.filePath, mode="rb")
						
					fr.seek(indexInfo.startIndex)
					buf = fr.read(indexInfo.dataSize)
					newTable[uttID] = newTable.spec( indexInfo.frames, startIndex, indexInfo.dataSize, None )
					startIndex += indexInfo.dataSize
					datas.append(buf)
			
			if arkType is None:
				if matrixFlag is True:
					result = BytesMatrix( b"".join(datas), name=self.name, indexTable=newTable )
				else:
					result = BytesVector( b"".join(datas), name=self.name, indexTable=newTable )
			elif arkType == "mat":
				result = BytesMatrix( b"".join(datas), name=self.name, indexTable=newTable )
			elif arkType == "vec":
				result = BytesVector( b"".join(datas), name=self.name, indexTable=newTable )		
			elif arkType == "feat":
				result = BytesFeature( b"".join(datas), name=self.name, indexTable=newTable )
			elif arkType == "cmvn":
				result = BytesCMVNStatistics( b"".join(datas), name=self.name, indexTable=newTable )
			elif arkType == "prob":
				result = BytesProbability( b"".join(datas), name=self.name, indexTable=newTable )
			elif arkType == "ali":
				result = BytesAlignmentTrans( b"".join(datas), name=self.name, indexTable=newTable )
			else:
				result = BytesFmllrMatrix( b"".join(datas), name=self.name, indexTable=newTable )
			
			result.check_format()

			return result

## Subclass: for transcription, both ref and hyp
class Transcription(ListTable):
	'''
	This is used to hold transcription text, such as decoding n-best. 
	'''
	def __init__(self, data={}, name="transcription"):
		super(Transcription, self).__init__(data, name=name)

	def sort(self, reverse=False):
		'''
		Sort by utterance ID.

		Args:
			<reverse>: If reverse, sort in descending order.
		Return:
			A new Transcription object.
		''' 
		items = super().sort(reverse=reverse)		
		return Transcription(items, name=self.name)

	def __add__(self, other):
		'''
		Integrate two transcription objects. If utt-ID existed in both two objects, the former will be retained.

		Args:
			<other>: another Transcription object.
		Return:
			A new Transcription object.
		'''
		declare.is_classes("other", other, Transcription)

		result = super().__add__(other)
		return Transcription(result, name=result.name)

	def shuffle(self):
		'''
		Random shuffle the transcription.

		Return:
			A new Transcription object.
		'''
		result = super().shuffle()

		return Transcription(result, name=self.name)

	def subset(self, nHead=0, nTail=0, nRandom=0, chunks=1, uttIDs=None):
		'''
		Subset.
		Only one mode will work when it is not the default value. 
		The priority order is: nHead > nTail > nRandom > chunks > uttIDs.
		
		Args:
			<nHead>: If it > 0, extract N head utterances.
			<nTail>: If it > 0, extract N tail utterances.
			<nRandom>: If it > 0, randomly sample N utterances.
			<chunks>: If it > 1, split data into N chunks.
			<uttIDs>: If it is not None, pick out these utterances whose ID in uttIDs.
		Return:
			a new ArkIndexTable object or a list of new ArkIndexTable objects.
		''' 
		result = super().subset(nHead,nTail,nRandom,chunks,uttIDs)

		if isinstance(result, list):
			for index in range(len(result)):
				temp = result[index]
				result[index] = Transcription(temp, temp.name)
		else:
			result = Transcription(result, result.name)

		return result
	
	def convert(self, symbolTable, unkSymbol=None):
		'''
		Convert transcription between two types of symbol, typically text format and int format.

		Args:
			<symbolTable>: exkaldi ListTable object.
			<unkSymbol>: symbol of oov. If symbol is out of table, use this to replace.
		Return:
			A new Transcription object.
		'''
		declare.not_void(type_name(self), self)
		declare.is_classes("symbolTable", symbolTable, ListTable)
		
		symbolTable = dict( (str(k),str(v)) for k,v in symbolTable.items() )
		unkSymbol = str(unkSymbol)

		newTrans = Transcription(name=f"convert({self.name})")

		for uttID, text in self.items():
			declare.is_valid_string("transcription", text)
			text = text.split()
			for index, word in enumerate(text):
				try:
					text[index] = str(symbolTable[word])
				except KeyError:
					if unkSymbol is None:
						raise WrongDataFormat(f"Missed the corresponding target for symbol: {word}.")
					else:
						try:
							text[index] = str(symbolTable[unkSymbol])
						except KeyError:
							raise WrongDataFormat(f"Word symbol table missed unknown-map symbol: {unkSymbol}")
		
			newTrans[uttID] = " ".join(text)
	
		return newTrans

	def sentence_length(self):
		'''
		Count the length of each sentence ( It will count the numbers of inner space ).
		'''
		result = Metric(name=f"sentence_length({self.name})")
		for uttID, txt in self.items():
			declare.is_valid_string("transcription", txt)
			result[uttID] = txt.strip().count(" ") + 1
		return result

	def save(self, fileName=None, chunks=1, discardUttID=False):
		'''
		Save as text file.

		Args:
			<fileName>: None, file name or file handle.
			<discardUttID>: If True, discard the ifno of utterance IDs.
		
		Return:
			file path or the contents of ListTable.
		'''
		declare.is_bool("discardUttID", discardUttID)

		def concat(item, discardUttID):
			try:
				if discardUttID:
					return f"{item[1]}"
				else:
					return f"{item[0]} {item[1]}"
			except Exception:
				print(f"Utterance ID: {item[0]}")
				raise WrongDataFormat(f"Wrong key and value format: {type_name(item[0])} and {type_name(item[1])}. ")
		
		return super().save(fileName, chunks, lambda x:concat(x, discardUttID) )

	def count_word(self):
		'''
		Count the number of word.

		Return:
			Metric object.
		'''
		result = Metric(name=f"count_word({self.name})")
		for uttID, txt in self.items():
			declare.is_valid_string("transcription", txt)
			txt = txt.strip().split()
			for w in txt:
				try:
					_ = result[w]
				except KeyError:
					result[w] = 1
				else:
					result[w] += 1

		return result

## Subclass: for variable scores
class Metric(ListTable):
	'''
	This is used to hold the Metrics, such as AM or LM scores. 
	The data format in Metric is: { utterance ID : int or float score,  }
	'''
	def __init__(self, data={}, name="metric"):
		super(Metric, self).__init__(data, name=name)

	def sort(self, by="utt", reverse=False):
		'''
		Sort by utterance ID or score.

		Args:
			<by>: "utt" or "score".
			<reverse>: If reverse, sort in descending order.
		Return:
			A new Metric object.
		''' 
		declare.is_instances("by", by, ["utt", "score"])

		def filtering(x, i):
			declare.is_valid_string("uttID", x[0])
			declare.is_classes("score", x[1], (int,float))
			return x[i]

		if by == "utt":
			items = sorted(self.items(), key=lambda x:filtering(x,0), reverse=reverse)
		else:
			items = sorted(self.items(), key=lambda x:filtering(x,1), reverse=reverse)
		
		newName = f"sort({self.name},{by})"
		return Metric(items, name=newName)
	
	def __add__(self, other):
		'''
		Integrate two Metric objects. If utt is existed in both two objects, the former will be retained.

		Args:
			<other>: another Metric object.
		Return:
			A new Metric object.
		'''
		declare.is_classes("other", other, Metric)
		result = super().__add__(other)
		return Metric(result, name=result.name)

	def shuffle(self):
		'''
		Random shuffle the Metric table.

		Return:
			A new Metric object.
		'''
		results = super().shuffle()

		return Metric(results, name=self.name)

	def subset(self,nHead=0,nTail=0,nRandom=0,chunks=1,uttIDs=None):
		'''
		Subset feature.
		
		Args:
			<nHead>: If it > 0, extract N head utterances.
			<nTail>: If nHead=0 and nTail > 0, extract N tail utterances.
			<nRandom>: If nHead=0 and nTail=0 and nRandom > 0, randomly sample N utterances.
			<chunks>: If all of nHead, nTail, nRandom are 0 and chunks > 1, split data into N chunks.
			<uttIDs>: If nHead == 0 and chunks == 1 and uttIDs != None, pick out these utterances whose ID in uttIDs.
		Return:
			a new Metric object or a list of new Metric objects.
		'''
		result = super().subset(nHead,nTail,nRandom,chunks,uttIDs)

		if isinstance(result, list):
			for index in range(len(result)):
				temp = result[index]
				result[index] = Metric(temp, temp.name)
		else:
			result = Metric(result, result.name)

		return result

	def sum(self, weight=None):
		'''
		The weighted sum of all scores.
		'''
		if self.is_void:
			return 0.0

		if weight is None:
			return sum(self.values())
		else:
			declare.is_classes("weight", weight, ["dict","Metric"])

			totalSum = 0
			for uttID,value in self.items():
				try:
					W = weight[uttID]
				except KeyError:
					raise WrongOperation(f"Miss weight for: {uttID}.")
				else:
					declare.is_classes("weight", W, [int,float])
					totalSum += W*value
			
			return totalSum

	def mean(self, weight=None, epsilon=1e-8):
		'''
		The weighted average of all score.

		Args:
			<weigts>: the weight of each utterance.
		'''
		if self.is_void:
			return 0.0
		
		if weight is None:
			return self.sum()/len(self)
		else:
			declare.is_classes("weight",weight,["dict","Metric"])

			numerator = 0
			denominator = epsilon
			for key,value in self.items():
				try:
					W = weight[key]
				except KeyError:
					raise WrongOperation(f"Miss weight for: {key}.")
				else:
					declare.is_classes("weight", W, [int,float])
					numerator += W*value
					denominator += W

			return numerator/denominator

	def max(self):
		'''
		The maximum value.
		'''
		return max(self.values())
	
	def argmax(self):
		'''
		Get the uttID of the max score.
		'''
		return sorted(self.items(),key=lambda x:x[1], reverse=True)[0][0]

	def min(self):
		'''
		The minimum value.
		'''
		return min(self.values())
	
	def argmin(self):
		'''
		Get the uttID of the min score.
		'''
		return sorted(self.items(),key=lambda x:x[1])[0][0]

	def map(self, func):
		'''
		Map all arrays to a function.

		Args:
			<func>: callable function object.
		
		Return:
			A new Metric object.
		'''
		declare.is_callable(func)

		new = dict(map( lambda x:(x[0],func(x[1])), self.data.items()))

		return Metric(new, name=f"mapped({self.name})")

'''BytesArchieve class group'''
'''Designed for Kaldi binary archieve table. It also support other objects such as lattice, HMM-GMM and decision tree'''
## Base class
class BytesArchieve:

	def __init__(self, data=b'', name=None):

		if data != None:
			declare.is_classes("data", data, bytes)
		self.__data = data

		if name is None:
			self.__name = self.__class__.__name__
		else:
			declare.is_valid_string("name",name)
			self.__name = name
	
	@property
	def data(self):

		return self.__data
	
	def reset_data(self, newData=None):

		if newData != None:
			declare.is_classes("newData", newData, bytes)
		del self.__data
		self.__data = newData		

	@property
	def is_void(self):

		if self.__data is None or len(self.__data) == 0:
			return True
		else:
			return False

	@property
	def name(self):

		return self.__name

	def rename(self, newName=None):

		if newName is not None:
			declare.is_valid_string("newName",newName)
		else:
			newName = self.__class__.__name__
		self.__name = newName

## Base class: for Matrix Data archieves
class BytesMatrix(BytesArchieve):
	'''
	A base class for matrix data, such as feature, cmvn statistics, post probability.
	'''
	def __init__(self, data=b'', name="data", indexTable=None):
		'''
		Args:
			<data>: If it's BytesMatrix or ArkIndexTable object (or their subclasses), extra <indexTable> will not work.
					If it's NumpyMatrix or bytes object (or their subclasses), generate index table automatically if it is not provided.
		'''
		declare.belong_classes("data", data, [BytesMatrix,NumpyMatrix,ArkIndexTable,bytes])

		needIndexTableFlag = True

		if isinstance(data, BytesMatrix):
			self.__dataIndex = data.indexTable
			self.__dataIndex.rename(name)
			data = data.data
			needIndexTableFlag = False
		
		elif isinstance(data ,ArkIndexTable):
			data = data.fetch(arkType="mat", name=name)
			self.__dataIndex = data.indexTable
			data = data.data
			needIndexTableFlag = False

		elif isinstance(data, NumpyMatrix):
			data = (data.to_bytes()).data

		super().__init__(data, name)

		if needIndexTableFlag is True:
			if indexTable is None:
				self.__generate_index_table()
			else:
				declare.is_classes("indexTable", indexTable, ArkIndexTable)
				self.__verify_index_table(indexTable)
	
	def __verify_index_table(self, indexTable):
		'''
		Check the format of provided index table.
		'''
		newIndexTable = indexTable.sort("startIndex")
		start = 0
		for uttID, indexInfo in newIndexTable.items():
			if indexInfo.startIndex != start:
				raise WrongDataFormat(f"Start index of {uttID} dose not match: expected {start} but got {indexInfo.startIndex}.")
			if indexInfo.filePath is not None:
				newIndexTable[uttID] = indexInfo._replace(filePath=None)
			start += indexInfo.dataSize

		newIndexTable.rename(self.name)
		self.__dataIndex = newIndexTable

	def __generate_index_table(self):
		'''
		Generate a index table.
		'''
		if self.is_void:
			return None
		else:
			self.__dataIndex = ArkIndexTable(name=self.name)
			start = 0
			with BytesIO(self.data) as sp:
				while True:
					(utt, dataType, rows, cols, bufSize, buf) = self.__read_one_record(sp)
					if utt == None:
						break
					oneRecordLen = len(utt) + 16 + bufSize

					self.__dataIndex[utt] = self.__dataIndex.spec(rows, start, oneRecordLen)
					start += oneRecordLen

	def __read_one_record(self, fp):
		'''
		Read a utterance.
		'''
		utt = ''
		while True:
			char = fp.read(1).decode()
			if (char == '') or (char == ' '):break
			utt += char
		utt = utt.strip()
		if utt == '':
			if fp.read() == b'':
				return (None, None, None, None, None, None)
			else:
				fp.close()
				raise WrongDataFormat("Miss utterance ID before utterance.")
		binarySymbol = fp.read(2).decode()
		if binarySymbol == '\0B':
			sizeSymbol = fp.read(1).decode()
			if sizeSymbol not in ["C","F","D"]:
				fp.close()
				if sizeSymbol == '\4':
					raise WrongDataFormat(f"{type_name(self)} need matrix data but this seems like vector.")
				else:
					raise WrongDataFormat("This might not be kaldi archieve data.")
			dataType = sizeSymbol + fp.read(2).decode() 
			if dataType == 'CM ':
				fp.close()
				raise UnsupportedType("This is compressed binary data. Use load_feat() function to load ark file again or use decompress() function to decompress it firstly.")                    
			elif dataType == 'FM ':
				sampleSize = 4
			elif dataType == 'DM ':
				sampleSize = 8
			else:
				fp.close()
				raise WrongDataFormat(f"Expected data type FM(float32), DM(float64), CM(compressed data) but got {dataType}.")
			s1,rows,s2,cols = np.frombuffer(fp.read(10), dtype="int8, int32, int8, int32", count=1)[0]
			rows = int(rows)
			cols = int(cols)
			bufSize = rows * cols * sampleSize
			buf = fp.read(bufSize)
		else:
			fp.close()
			raise WrongDataFormat("Miss binary symbol before utterance.")
		return (utt, dataType, rows, cols, bufSize, buf)

	@property
	def indexTable(self):
		'''
		Get the index information of utterances.
		
		Return:
			A ArkIndexTable object.
		'''
		# Return deepcopied dict object.
		return copy.deepcopy(self.__dataIndex)

	@property
	def dtype(self):
		'''
		Get the data type of bytes data.
		
		Return:
			A string in 'float32', 'float64'.
		'''
		if self.is_void:
			_dtype = None
		else:
			with BytesIO(self.data) as sp:
				(utt, dataType, rows, cols, bufSize, buf) = self.__read_one_record(sp)
			if dataType == "FM ":
				_dtype = "float32"
			else:
				_dtype = "float64"
             
		return _dtype

	def to_dtype(self, dtype):
		'''
		Transform data type.

		Args:
			<dtype>: a string of "float", "float32" or "float64". IF "float", it will be treated as "float32".
		
		Return:
			A new BytesMatrix object.
		'''
		declare.is_instances("dtype", dtype, ["float", "float32", "float64"])
		declare.not_void(type_name(self), self)

		if dtype == "float":
			dtype = "float32"

		if self.dtype == dtype:
			return copy.deepcopy(self)
		else:
			if dtype == 'float32':
				newDataType = 'FM '
			else:
				newDataType = 'DM '
			
			result = []
			newDataIndex = ArkIndexTable(name=self.name)
			# Data size will be changed so generate a new index table.
			with BytesIO(self.data) as sp:
				start = 0
				while True:
					(utt, dataType, rows, cols, bufSize, buf) = self.__read_one_record(sp)
					if utt is None:
						break
					if dataType == 'FM ': 
						matrix = np.frombuffer(buf, dtype=np.float32)
					else:
						matrix = np.frombuffer(buf, dtype=np.float64)
					newMatrix = np.array(matrix, dtype=dtype).tobytes()
					data = (utt+' '+'\0B'+newDataType).encode()
					data += '\04'.encode()
					data += struct.pack(np.dtype('uint32').char, rows)
					data += '\04'.encode()
					data += struct.pack(np.dtype('uint32').char, cols)
					data += newMatrix
					result.append(data)

					oneRecordLength = len(data)
					newDataIndex[utt] = newDataIndex.spec(rows, start, oneRecordLength)
					start += oneRecordLength
					
			result = b''.join(result)

			return BytesMatrix(result, name=self.name, indexTable=newDataIndex)

	@property
	def dim(self):
		'''
		Get the data dimensions.
		
		Return:
			If data is void, return None, or return an int value.
		'''
		if self.is_void:
			return None
		else:
			with BytesIO(self.data) as sp:
				(utt, dataType, rows, cols, bufSize, buf) = self.__read_one_record(sp)
			
			return cols

	@property
	def utts(self):
		'''
		Get all utterance IDs.
		
		Return:
			a list including all utterance IDs.
		'''
		if self.is_void:
			return []
		else:
			return list(self.__dataIndex.keys())
			
	def check_format(self):
		'''
		Check if data has right kaldi format.
		
		Return:
			If data is void, return False.
			If data has right format, return True, or raise Error.
		'''
		if self.is_void:
			return False

		_dim = "unknown"
		_dataType = "unknown"
		with BytesIO(self.data) as sp:
			start = 0
			while True: 
				(utt, dataType, rows, cols, bufSize, buf) = self.__read_one_record(sp)
				if utt == None:
					break
				if _dim == "unknown":
					_dim = cols
					_dataType = dataType
				elif cols != _dim:
					raise WrongDataFormat(f"Expected dimension {_dim} but got {cols} at utterance {utt}.")
				elif _dataType != dataType:
					raise WrongDataFormat(f"Expected data type {_dataType} but got {dataType} at uttwerance {utt}.")                 
				else:
					try:
						if dataType == "FM ":
							mat = np.frombuffer(buf, dtype=np.float32)
						else:
							mat = np.frombuffer(buf, dtype=np.float64)
					except Exception as e:
						print(f"Wrong matrix data format at utterance {utt}.")
						raise e
				
				oneRecordLen = len(utt) + 16 + bufSize

				# Renew the index table.
				self.__dataIndex[utt] = self.__dataIndex.spec(rows, start, oneRecordLen)
				start += oneRecordLen			
					
		return True
	
	@property
	def lens(self):
		'''
		Get the numbers of utterances.
		If you want to get the frames of each utterance, try:
						obj.indexTable 
		attribute.
		
		Return:
			a int value.
		'''
		lengths = 0
		if not self.is_void:
			lengths = len(self.__dataIndex)
		
		return lengths

	def save(self, fileName, chunks=1, returnIndexTable=False):
		'''
		Save bytes data to file.

		Args:
			<fileName>: file name or file handle. If it7s a file name, suffix ".ark" will be add to the name defaultly.
			<chunks>: If larger than 1, data will be saved to mutiple files averagely. This would be invalid when <fileName> is a file handle.
			<returnIndexTable>: If True, return the index table containing the information of file path.
		
		Return:
			the path of saved files.
		'''
		declare.not_void(type_name(self), self)
		declare.is_valid_file_name_or_handle("fileName", fileName)
		declare.in_boundary("chunks", chunks, minV=1)
		declare.is_bool("returnIndexTable", returnIndexTable)

		if isinstance(fileName, str):

			def save_chunk_data(chunkData, arkFileName, returnIndexTable):

				make_dependent_dirs(arkFileName, pathIsFile=True)
				with open(arkFileName, "wb") as fw:
					fw.write(chunkData.data)
				
				if returnIndexTable is True:
					indexTable = chunkData.indexTable
					for uttID in indexTable.keys():
						indexTable[uttID] = indexTable[uttID]._replace(filePath=arkFileName)

					return indexTable
				else:
					return arkFileName

			fileName = fileName.strip()
			if not fileName.endswith('.ark'):
				fileName += '.ark'

			if chunks == 1:
				savedFiles = save_chunk_data(self, fileName, returnIndexTable)	
			else:
				dirName = os.path.dirname(fileName)
				fileName = os.path.basename(fileName)
				savedFiles = []
				chunkDataList = self.subset(chunks=chunks)
				for i, chunkData in enumerate(chunkDataList):
					chunkFileName = os.path.join( dirName, f"ck{i}_{fileName}" )
					savedFiles.append( save_chunk_data(chunkData, chunkFileName, returnIndexTable) )

			return savedFiles
		
		else:
			fileName.truncate()
			fileName.write(self.data)
			fileName.seek(0)

			return fileName

	def to_numpy(self):
		'''
		Transform bytes data to numpy data.
		
		Return:
			a NumpyMatrix object sorted by utterance ID.
		'''
		newDict = {}
		if not self.is_void:
			sortedIndex = self.indexTable.sort(by="utt", reverse=False)
			with BytesIO(self.data) as sp:
				for utt, indexInfo in sortedIndex.items():
					sp.seek(indexInfo.startIndex)
					(utt, dataType, rows, cols, bufSize, buf) = self.__read_one_record(sp)
					try:
						if dataType == 'FM ': 
							newMatrix = np.frombuffer(buf, dtype=np.float32)
						else:
							newMatrix = np.frombuffer(buf, dtype=np.float64)
					except Exception as e:
						print(f"Wrong matrix data format at utterance {utt}.")
						raise e	
					else:
						newDict[utt] = np.reshape(newMatrix, (rows,cols))

		return NumpyMatrix(newDict, name=self.name)

	def __add__(self, other):
		'''
		The plus operation between two objects.

		Args:
			<other>: a BytesMatrix or NumpyMatrix object (or their subclasses object).
		Return:
			a new BytesMatrix object.
		''' 
		declare.belong_classes("other", other, [BytesMatrix,NumpyMatrix,ArkIndexTable])

		if isinstance(other, NumpyMatrix):
			other = other.to_bytes()
		elif isinstance(other, ArkIndexTable):
			uttIDs = [ utt for utt in other.keys() if utt not in self.utts ]
			other = other.fecth(arkType="mat", uttIDs=uttIDs)
		
		newName = f"plus({self.name},{other.name})"
		if self.is_void:
			result = copy.deepcopy(other)
			result.rename(newName)
			return result
		elif other.is_void:
			result = copy.deepcopy(self)
			result.rename(newName)
			return result			
		elif self.dim != other.dim:
			raise WrongOperation(f"Data dimensions does not match: {self.dim}!={other.dim}.")

		selfUtts, selfDtype = self.utts, self.dtype
		otherDtype = other.dtype

		newDataIndex = self.indexTable
		#lastIndexInfo = list(newDataIndex.sort(by="startIndex", reverse=True).values())[0]
		start = len(self.data)

		newData = []
		with BytesIO(other.data) as op:
			for utt, indexInfo in other.indexTable.items():
				if not utt in selfUtts:
					op.seek( indexInfo.startIndex )
					if selfDtype == otherDtype:
						data = op.read( indexInfo.dataSize )
						data_size = indexInfo.dataSize

					else:
						(outt, odataType, orows, ocols, obufSize, obuf) = self.__read_one_record(op)
						obuf = np.array(np.frombuffer(obuf, dtype=otherDtype), dtype=selfDtype).tobytes()
						data = (outt+' '+'\0B'+odataType).encode()
						data += '\04'.encode()
						data += struct.pack(np.dtype('uint32').char, orows)
						data += '\04'.encode()
						data += struct.pack(np.dtype('uint32').char, ocols)
						data += obuf
						data_size = len(data)

					newDataIndex[utt] = newDataIndex.spec(indexInfo.frames, start, data_size)
					start += data_size

					newData.append(data)

		return BytesMatrix(b''.join([self.data, *newData]), name=newName, indexTable=newDataIndex)

	def subset(self, nHead=0, nTail=0, nRandom=0, chunks=1, uttIDs=None):
		'''
		Subset data.
		The priority of mode is nHead > nTail > nRandom > chunks > uttIDs.
		If you chose mutiple modes, only the prior one will work.
		
		Args:
			<nHead>: get N head utterances.
			<nTail>: get N tail utterances.
			<nRandom>: sample N utterances randomly.
			<chunks>: split data into N chunks averagely.
			<uttIDs>: pick out these utterances whose ID in uttIDs.
		Return:
			a new BytesMatrix object or a list of new BytesMatrix objects.
		''' 
		declare.not_void(type_name(self), self)

		if nHead > 0:
			declare.is_positive_int("nHead", nHead)
		
			newName = f"subset({self.name},head {nHead})"
			newDataIndex = ArkIndexTable(name=newName)
			totalSize = 0
			
			for utt, indexInfo in self.indexTable.items():
				newDataIndex[utt] = newDataIndex.spec(indexInfo.frames, totalSize, indexInfo.dataSize)
				totalSize += indexInfo.dataSize
				nHead -= 1
				if nHead <= 0:
					break
			
			with BytesIO(self.data) as sp:
				sp.seek(0)
				data = sp.read(totalSize)
	
			return BytesMatrix(data, name=newName, indexTable=newDataIndex)

		elif nTail > 0:
			declare.is_positive_int("nTail", nTail)
			
			newName = f"subset({self.name},tail {nTail})"
			newDataIndex = ArkIndexTable(name=newName)

			tailNRecord = list(self.indexTable.items())[-nTail:]
			start_index = tailNRecord[0][1].startIndex

			totalSize = 0
			for utt, indexInfo in tailNRecord:
				newDataIndex[utt] = newDataIndex.spec(indexInfo.frames, totalSize, indexInfo.dataSize)
				totalSize += indexInfo.dataSize

			with BytesIO(self.data) as sp:
				sp.seek(start_index)
				data = sp.read(totalSize)
	
			return BytesMatrix(data, name=newName, indexTable=newDataIndex)

		elif nRandom > 0:
			declare.is_positive_int("nRandom", nRandom)

			randomNRecord = random.choices(list(self.indexTable.items()), k=nRandom)
			newName = f"subset({self.name},random {nRandom})"

			newDataIndex = ArkIndexTable(name=newName)
			start_index = 0
			newData = []
			with BytesIO(self.data) as sp:
				for utt, indexInfo in randomNRecord:
					sp.seek(indexInfo.startIndex)
					newData.append( sp.read(indexInfo.dataSize) )

					newDataIndex[utt] = newDataIndex.spec(indexInfo.frames, start_index, indexInfo.dataSize)
					start_index += indexInfo.dataSize

			return BytesMatrix(b"".join(newData), name=newName, indexTable=newDataIndex)

		elif chunks > 1:
			declare.is_positive_int("chunks", chunks)

			uttLens = list(self.indexTable.items())
			allLens = len(uttLens)
			chunkUtts = allLens//chunks
			if chunkUtts == 0:
				chunks = allLens
				chunkUtts = 1
				t = 0
				print(f"Warning: utterances is fewer than <chunks> so only {chunks} files will be saved.")
			else:
				t = allLens - chunkUtts * chunks
			
			datas = []
			with BytesIO(self.data) as sp:                          
				sp.seek(0)
				for i in range(chunks):
					newName = f"subset({self.name},chunk {chunks}-{i})"
					newDataIndex = ArkIndexTable(name=newName)
					if i < t:
						chunkItems = uttLens[i*(chunkUtts+1) : (i+1)*(chunkUtts+1)]
					else:
						chunkItems = uttLens[i*chunkUtts : (i+1)*chunkUtts]
					chunkLen = 0
					for utt, indexInfo in chunkItems:
						newDataIndex[utt] = newDataIndex.spec(indexInfo.frames, chunkLen, indexInfo.dataSize)
						chunkLen += indexInfo.dataSize
					chunkData = sp.read(chunkLen)
					
					datas.append( BytesMatrix(chunkData, name=newName, indexTable=newDataIndex) )
			return datas

		elif uttIDs != None:
			declare.is_classes("uttIDs", uttIDs, [str,list,tuple])
			if isinstance(uttIDs, str):
				newName = f"subset({self.name},uttIDs 1)"
				uttIDs = [uttIDs,]
			else:
				declare.members_are_valid_strings("uttIDs", uttIDs)
				newName = f"subset({self.name},uttIDs {len(uttIDs)})"

			newData = []
			dataIndex = self.indexTable
			newDataIndex = ArkIndexTable(name=newName)
			start_index = 0
			with BytesIO(self.data) as sp:
				for utt in uttIDs:
					if utt in self.utts:
						indexInfo = dataIndex[utt]
						sp.seek( indexInfo.startIndex )
						newData.append( sp.read(indexInfo.dataSize) )

						newDataIndex[utt] = newDataIndex.spec(indexInfo.frames, start_index, indexInfo.dataSize)
						start_index += indexInfo.dataSize

			return BytesMatrix(b''.join(newData), name=newName, indexTable=newDataIndex)
		
		else:
			raise WrongOperation('Expected one of <nHead>, <nTail>, <nRandom>, <chunks> or <uttIDs> is avaliable but all got the default value.')

	def __call__(self, utt):
		'''
		Pick out the specified utterance.
		
		Args:
			<utt>: a string.
		Return:
			If existed, return a new BytesMatrix object.
			Or return None.
		'''
		declare.is_valid_string("utt", utt)
		if self.is_void:
			return None

		utt = utt.strip()

		if utt not in self.utts:
			return None
		else:
			indexInfo = self.indexTable[utt]
			newName = f"pick({self.name},{utt})"
			newDataIndex = ArkIndexTable(name=newName)
			with BytesIO(self.data) as sp:
				sp.seek( indexInfo.startIndex )
				data = sp.read( indexInfo.dataSize )

				newDataIndex[utt] =	indexInfo._replace(startIndex=0)
				result = BytesMatrix(data, name=newName, indexTable=newDataIndex)
			
			return result

	def sort(self, by="utt", reverse=False):
		'''
		Sort utterances by frames length or uttID

		Args:
			<by>: "frame" or "utt"
			<reverse>: If reverse, sort in descending order.
		Return:
			A new BytesMatrix object.
		''' 
		declare.is_instances("by", by, ["utt", "frame"])

		newDataIndex = self.indexTable.sort(by=by, reverse=reverse)
		ordered = True
		for i, j in zip(self.indexTable.items(), newDataIndex.items()):
			if i != j:
				ordered = False
				break
		if ordered:
			return copy.deepcopy(self)

		with BytesIO(self.data) as sp:
			if sys.getsizeof(self.data) > 10**9:
				## If the data size is large, divide it into N chunks and save it to intermidiate file.
				with FileHandleManager as fhm:
					temp = fhm.create("wb+")
					chunkdata = []
					chunkSize = 50
					count = 0
					start_index = 0
					for utt, indexInfo in newDataIndex.items():
						newDataIndex[utt] = indexInfo._replace(startIndex=start_index)
						start_index += indexInfo.dataSize

						sp.seek( indexInfo.startIndex )
						chunkdata.append( sp.read(indexInfo.dataSize) )
						count += 1
						if count >= chunkSize:
							temp.write( b"".join(chunkdata) )
							del chunkdata
							chunkdata = []
							count = 0
					if len(chunkdata) > 0:
						temp.write( b"".join(chunkdata) )
						del chunkdata
					temp.seek(0)
					newData = temp.read()
		
			else:
				newData = []
				start_index = 0
				for utt, indexInfo in newDataIndex.items():
					sp.seek( indexInfo.startIndex )
					newData.append( sp.read(indexInfo.dataSize) )

					newDataIndex[utt] = indexInfo._replace(startIndex=start_index)
					start_index += indexInfo.dataSize

				newData = b"".join(newData)

		return BytesMatrix(newData, name=self.name, indexTable=newDataIndex)			

## Subclass: for acoustic feature (in binary format)		
class BytesFeature(BytesMatrix):
	'''
	Hold the feature with kaldi binary format.
	'''
	def __init__(self, data=b"", name="feat", indexTable=None):
		'''
		Only allow BytesFeature, NumpyFeature, ArkIndexTable or bytes (do not extend to their subclasses and their parent-classes).
		'''
		declare.is_classes("data", data, [BytesFeature, NumpyFeature, ArkIndexTable, bytes])
		super().__init__(data, name, indexTable)
	
	def to_numpy(self):
		'''
		Transform feature to numpy format.

		Return:
			a NumpyFeature object.
		'''
		result = super().to_numpy()
		return NumpyFeature(result.data, name=result.name)

	def __add__(self, other):
		'''
		Plus operation between two feature objects.

		Args:
			<other>: a BytesFeature or NumpyFeature object.
		Return:
			a BytesFeature object.
		'''
		declare.is_feature("other", other)

		result = super().__add__(other)

		return BytesFeature(result.data, name=result.name, indexTable=result.indexTable)

	def splice(self, left=1, right=None):
		'''
		Splice front-behind N frames to generate new feature.

		Args:
			<left>: the left N-frames to splice.
			<right>: the right N-frames to splice. If None, right = left.
		Return:
			A new BytesFeature object whose dim became original-dim * (1 + left + right).
		''' 
		declare.kaldi_existed()
		declare.not_void(type_name(self), self )
		declare.is_non_negative_int("left", left)

		if right is None:
			right = left
		else:
			declare.is_non_negative_int("right", right)
		
		cmd = f"splice-feats --left-context={left} --right-context={right} ark:- ark:-"
		out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=self.data)

		if cod != 0 or out == b'':
			print(err.decode())
			raise KaldiProcessError("Failed to splice left-right frames.")
		else:
			newName = f"splice({self.name},{left},{right})"
			# New index table will be generated later.
			return BytesFeature(out, name=newName, indexTable=None)

	def select(self, dims, retain=False):
		'''
		Select specified dimensions of feature.

		Args:
			<dims>: A int value or string such as "1,2,5-10"
			<retain>: If True, return the rest dimensions of feature simultaneously.
		Return:
			A new BytesFeature object or two BytesFeature objects.
		''' 
		declare.kaldi_existed()
		declare.not_void( type_name(self), self )

		_dim = self.dim

		if isinstance(dims, int):
			declare.in_boundary("Selected index", dims, minV=0, maxV=_dim-1)
			selectFlag = str(dims)
			if retain:
				if dims == 0:
					retainFlag = f"1-{_dim-1}"
				elif dims == _dim-1:
					retainFlag = f"0-{_dim-2}"
				else:
					retainFlag = f"0-{dims-1},{dims+1}-{_dim-1}"
		
		elif isinstance(dims, str):
			declare.is_valid_string("dims", dims)
			if retain:
				retainFlag = [x for x in range(_dim)]
				for i in dims.strip().split(','):
					i = i.strip()
					if i == "":
						continue
					if not '-' in i:
						try:
							i = int(i)
						except ValueError:
							raise WrongOperation(f"Expected int value but got {i}.")
						else:
							declare.in_boundary("Selected index", i, minV=0, maxV=_dim-1)
							retainFlag[i] = -1 # flag
					else:
						i = i.split('-')
						assert len(i) == 2, "Index should has format like '1-2', '3-' or '-5'."
						if i[0].strip() == '':
							i[0] = 0
						if i[1].strip() == '':
							i[1] = _dim-1
						try:
							i[0] = int(i[0])
							i[1] = int(i[1])
						except ValueError:
							raise WrongOperation(f'Connot convert to int value: {i}.')
						else:
							if i[0] > i[1]:
								i[0], i[1] = i[1], i[0]
							declare.in_boundary("Selected index", i[1], minV=0, maxV=_dim-1)
						for j in range(i[0],i[1]+1,1):
							retainFlag[j] = -1 # flag
				temp = ''
				for x in retainFlag:
					if x != -1:
						temp += str(x) + ','
				retainFlag = temp[0:-1]
			selectFlag = dims
		
		else:
			raise WrongOperation(f"Expected int value or string like '1,4-9,12' but got {dims}.")

		cmdS = f'select-feats {selectFlag} ark:- ark:-'
		outS, errS, codS = run_shell_command(cmdS, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=self.data)
		
		if codS != 0 or outS == b'':
			print(errS.decode())
			raise KaldiProcessError("Failed to select data.")
		else:
			newName = f"select({self.name},{dims})"
			# New index table will be generated later.
			selectedResult = BytesFeature(outS, name=newName, indexTable=None)

		if retain:
			if retainFlag == "":
				newName = f"select({self.name}, void)"
				# New index table will be generated later.
				retainedResult = BytesFeature(name=newName, indexTable=None)
			else: 
				cmdR = f"select-feats {retainFlag} ark:- ark:-"
				outR, errR, codR = run_shell_command(cmdR, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=self.data)
				if codR != 0 or outR == b'':
					print(errR.decode())
					raise KaldiProcessError("Failed to select retained data.")
				else:
					newName = f"select({self.name},not {dims})"
					# New index table will be generated later.
					retainedResult = BytesFeature(outR, name=newName, indexTable=None)
		
			return selectedResult, retainedResult
		
		else:
			return selectedResult

	def subset(self, nHead=0, nTail=0, nRandom=0, chunks=1, uttIDs=None):
		'''
		Subset data.
		The priority of mode is nHead > nTail > nRandom > chunks > uttIDs.
		If you chose mutiple modes, only the prior one will work.
		
		Args:
			<nHead>: get N head utterances.
			<nTail>: get N tail utterances.
			<nRandom>: sample N utterances randomly.
			<chunks>: split data into N chunks averagely.
			<uttIDs>: pick out these utterances whose ID in uttIDs.
		Return:
			a new BytesFeature object or a list of new BytesFeature objects.
		''' 
		result = super().subset(nHead, nTail, nRandom, chunks, uttIDs)

		if isinstance(result, list):
			for index in range(len(result)):
				temp = result[index]
				result[index] = BytesFeature(temp.data, temp.name, temp.indexTable)
		else:
			result = BytesFeature(result.data, result.name, result.indexTable)

		return result
	
	def __call__(self, uttID):
		'''
		Pick out an utterance.
		
		Args:
			<uttID>: a string.
		Return:
			If existed, return a new BytesFeature object.
		''' 
		result = super().__call__(uttID)
		if result is not None:
			result = BytesFeature(result.data, result.name, result.indexTable)
		return result

	def add_delta(self, order=2):
		'''
		Add N orders delta informat to feature.

		Args:
			<order>: A positive int value.
		Return:
			A new BytesFeature object whose dimendion became original-dim * (1 + order). 
		''' 
		declare.kaldi_existed()
		declare.is_positive_int("order",order)
		declare.not_void( type_name(self), self )

		cmd = f"add-deltas --delta-order={order} ark:- ark:-"
		out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=self.data)
		if cod != 0 or out == b'':
			print(err.decode())
			raise KaldiProcessError('Failed to add delta feature.')
		else:
			newName = f"delta({self.name},{order})"
			# New index table need to be generated later.
			return BytesFeature(data=out, name=newName, indexTable=None)

	def paste(self, others):
		'''
		Paste feature in feature dimension.

		Args:
			<others>: a feature object or list of feature objects.
			<ordered>: If False, sort all objects.  
		Return:
			a new feature object.
		''' 
		declare.kaldi_existed()
		declare.not_void(type_name(self),self)
		declare.is_classes("others", others, [BytesFeature, NumpyFeature, ArkIndexTable, list, tuple])
		
		otherResp = []
		pastedName = [self.name,]
		
		with FileHandleManager() as fhm:
		
			if isinstance(others, BytesFeature):
				temp = fhm.create("wb+", suffix=".ark")
				others.sort(by="utt").save(temp)
				otherResp.append( f"ark:{temp.name}" )
				pastedName.append( others.name )

			elif isinstance(others, NumpyFeature):
				temp = fhm.create("wb+", suffix=".ark")
				others.sort(by="utt").to_bytes().save(temp)
				otherResp.append( f"ark:{temp.name}" )
				pastedName.append( others.name )
			
			elif isinstance(others, ArkIndexTable):
				temp = fhm.create("w+", suffix=".scp")
				others.sort(by="utt").save(temp)
				otherResp.append( f"scp:{temp.name}" )
				pastedName.append( others.name )

			else:
				for ot in others:
					declare.is_feature("others", ot)

					if isinstance(ot, BytesFeature):
						temp = fhm.create("wb+", suffix=".ark")
						ot.sort(by="utt").save(temp)
						otherResp.append( f"ark:{ot.name}" )

					elif isinstance(ot, NumpyFeature):
						temp = fhm.create("wb+", suffix=".ark")
						ot.sort(by="utt").to_bytes().save(temp)
						otherResp.append( f"ark:{ot.name}" )

					else:
						temp = fhm.create("w+", suffix=".scp")
						ot.sort(by="utt").save(temp)
						otherResp.append( f"scp:{ot.name}" )

					pastedName.append( ot.name )	
			
			selfData = fhm.create("wb+", suffix=".ark")
			self.sort(by="utt").save(selfData)

			otherResp = " ".join(otherResp)
			cmd = f"paste-feats ark:{selfData.name} {otherResp} ark:-"
			
			out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

			if cod != 0 or out == b'':
				print(err.decode())
				raise KaldiProcessError("Failed to paste feature.")
			else:
				pastedName = ",".join(pastedName)
				pastedName = f"paste({pastedName})"
				# New index table need to be generated later.
				return BytesFeature(out, name=pastedName, indexTable=None)

	def sort(self, by="utt", reverse=False):
		'''
		Sort utterances by frames length or uttID

		Args:
			<by>: "frame" or "utt"
			<reverse>: If reverse, sort in descending order.
		Return:
			A new BytesFeature object.
		''' 
		result = super().sort(by, reverse)
		return BytesFeature(result.data, name=result.name, indexTable=result.indexTable)

## Subclass: for CMVN statistics
class BytesCMVNStatistics(BytesMatrix):
	'''
	Hold the CMVN statistics with kaldi binary format.
	'''
	def __init__(self, data=b"", name="cmvn", indexTable=None):
		'''
		Only allow BytesCMVNStatistics, NumpyCMVNStatistics, ArkIndexTable or bytes (do not extend to their subclasses and their parent-classes).
		'''
		declare.is_classes("data", data, [BytesCMVNStatistics, NumpyCMVNStatistics, ArkIndexTable, bytes])

		super().__init__(data, name, indexTable)
	
	def to_numpy(self):
		'''
		Transform CMVN statistics to numpy format.

		Return:
			a NumpyCMVNStatistics object.
		'''
		result = super().to_numpy()
		return NumpyCMVNStatistics(result.data, name=result.name)

	def __add__(self, other):
		'''
		Plus operation between two CMVN statistics objects.

		Args:
			<other>: a BytesCMVNStatistics or NumpyCMVNStatistics object.
		Return:
			a BytesCMVNStatistics object.
		'''	
		declare.is_cmvn("other", other)

		result = super().__add__(other)

		return BytesCMVNStatistics(result.data, name=result.name, indexTable=result.indexTable)

	def subset(self, nHead=0, nTail=0, nRandom=0, chunks=1, uttIDs=None):
		'''
		Subset data.
		The priority of mode is nHead > nTail > nRandom > chunks > uttIDs.
		If you chose mutiple modes, only the prior one will work.
		
		Args:
			<nHead>: get N head utterances.
			<nTail>: get N tail utterances.
			<nRandom>: sample N utterances randomly.
			<chunks>: split data into N chunks averagely.
			<uttIDs>: pick out these utterances whose ID in uttIDs.
		Return:
			a new BytesCMVNStatistics object or a list of new BytesCMVNStatistics objects.
		''' 
		result = super().subset(nHead, nTail, nRandom, chunks, uttIDs)

		if isinstance(result, list):
			for index in range(len(result)):
				temp = result[index]
				result[index] = BytesCMVNStatistics(temp.data, temp.name, temp.indexTable)
		else:
			result = BytesCMVNStatistics(result.data, result.name, result.indexTable)

		return result

	def __call__(self, uttID):
		'''
		Pick out an utterance.
		
		Args:
			<uttID>: a string.
		Return:
			If existed, return a new BytesCMVNStatistics object.
		''' 
		result = super().__call__(uttID)
		if result is not None:
			result = BytesCMVNStatistics(result.data, result.name, result.indexTable)
		return result

	def sort(self, reverse=False):
		'''
		Sort utterances by utterance ID.

		Args:
			<reverse>: If reverse, sort in descending order.
		Return:
			A new BytesCMVNStatistics object.
		''' 
		result = super().sort(by="utt", reverse=reverse)
		return BytesCMVNStatistics(result.data, name=result.name, indexTable=result.indexTable)

## Subclass: for probability of neural network output
class BytesProbability(BytesMatrix):
	'''
	Hold the probalility with kaldi binary format.
	'''
	def __init__(self, data=b"", name="prob", indexTable=None):
		'''
		Only allow BytesProbability, NumpyProbability, ArkIndexTable or bytes (do not extend to their subclasses and their parent-classes).
		'''		
		declare.is_classes("data", data, [BytesProbability, NumpyProbability, ArkIndexTable, bytes])

		super().__init__(data, name, indexTable)
	
	def to_numpy(self):
		'''
		Transform post probability to numpy format.

		Return:
			a NumpyProbability object.
		'''
		result = super().to_numpy()
		return NumpyProbability(result.data, result.name)

	def __add__(self, other):
		'''
		Plus operation between two post probability objects.

		Args:
			<other>: a BytesProbability or NumpyProbability object.
		Return:
			a BytesProbability object.
		'''
		declare.is_probability("other", other)

		result = super().__add__(other)

		return BytesProbability(result.data, result.name, result.indexTable)

	def subset(self, nHead=0, nTail=0, nRandom=0, chunks=1, uttIDs=None):
		'''
		Subset data.
		The priority of mode is nHead > nTail > nRandom > chunks > uttIDs.
		If you chose mutiple modes, only the prior one will work.
		
		Args:
			<nHead>: get N head utterances.
			<nTail>: get N tail utterances.
			<nRandom>: sample N utterances randomly.
			<chunks>: split data into N chunks averagely.
			<uttIDs>: pick out these utterances whose ID in uttIDs.
		Return:
			a new BytesProbability object or a list of new BytesProbability objects.
		''' 
		result = super().subset(nHead, nTail, nRandom, chunks, uttIDs)

		if isinstance(result, list):
			for index in range(len(result)):
				temp = result[index]
				result[index] = BytesProbability(temp.data, temp.name, temp.indexTable)
		else:
			result = BytesProbability(result.data, result.name, result.indexTable)

		return result

	def __call__(self, uttID):
		'''
		Pick out an utterance.
		
		Args:
			<uttID>: a string.
		Return:
			If existed, return a new BytesProbability object.
		''' 
		result = super().__call__(uttID)
		if result is not None:
			result = BytesProbability(result.data, result.name, result.indexTable)
		return result

	def sort(self, by="utt", reverse=False):
		'''
		Sort utterances by frames length or uttID

		Args:
			<by>: "frame" or "utt"
			<reverse>: If reverse, sort in descending order.
		Return:
			A new BytesProbability object.
		''' 
		result = super().sort(by, reverse)

		return BytesProbability(result.data, name=result.name, indexTable=result.indexTable)

class BytesFmllrMatrix(BytesMatrix):
	'''
	Hold the fMLLR transform matrix with kaldi binary format.
	'''
	def __init__(self, data=b"", name="fmllrTrans", indexTable=None):
		'''
		Only allow BytesFmllrMatrix, NumpyFmllrMatrix, ArkIndexTable or bytes (do not extend to their subclasses and their parent-classes).
		'''		
		declare.is_classes("data", data, [BytesFmllrMatrix, NumpyFmllrMatrix, ArkIndexTable, bytes])

		super().__init__(data, name, indexTable)
	
	def to_numpy(self):
		'''
		Transform fMLLR transform matrix to numpy format.

		Return:
			a NumpyFmllrMatrix object.
		'''
		result = super().to_numpy()
		return NumpyFmllrMatrix(result.data, name=result.name)

	def __add__(self, other):
		'''
		Plus operation between two fMLLR transform matrix objects.

		Args:
			<other>: a BytesFmllrMatrix or NumpyFmllrMatrix object.
		Return:
			a BytesFmllrMatrix object.
		'''
		declare.is_fmllr_matrix("other", other)

		result = super().__add__(other)

		return BytesFmllrMatrix(result.data, name=result.name, indexTable=result.indexTable)

	def subset(self, nHead=0, nTail=0, nRandom=0, chunks=1, uttIDs=None):
		'''
		Subset data.
		The priority of mode is nHead > nTail > nRandom > chunks > uttIDs.
		If you chose mutiple modes, only the prior one will work.
		
		Args:
			<nHead>: get N head utterances.
			<nTail>: get N tail utterances.
			<nRandom>: sample N utterances randomly.
			<chunks>: split data into N chunks averagely.
			<uttIDs>: pick out these utterances whose ID in uttIDs.
		Return:
			a new BytesFmllrMatrix object or a list of new BytesFmllrMatrix objects.
		''' 
		result = super().subset(nHead, nTail, nRandom, chunks, uttIDs)

		if isinstance(result, list):
			for index in range(len(result)):
				temp = result[index]
				result[index] = BytesFmllrMatrix(temp.data, temp.name, temp.indexTable)
		else:
			result = BytesFmllrMatrix(result.data, result.name, result.indexTable)

		return result

	def __call__(self, uttID):
		'''
		Pick out an utterance.
		
		Args:
			<uttID>: a string.
		Return:
			If existed, return a new BytesFmllrMatrix object.
		''' 
		result = super().__call__(uttID)
		if result is not None:
			result = BytesFmllrMatrix(result.data, result.name, result.indexTable)
		return result
		
	def sort(self, by="utt", reverse=False):
		'''
		Sort utterances by frames length or uttID

		Args:
			<by>: "frame" or "utt"
			<reverse>: If reverse, sort in descending order.
		Return:
			A new BytesFmllrMatrix object.
		''' 
		result = super().sort(by, reverse)
		return BytesFmllrMatrix(result.data, name=result.name, indexTable=result.indexTable)

## Base class: for Vector Data archieves
class BytesVector(BytesArchieve):
	'''
	A base class to hold kaldi vector data such as alignment.  
	'''
	def __init__(self, data=b'', name="vec", indexTable=None):
		'''
		Args:
			<data>: If it's BytesMatrix or ArkIndexTable object (or their subclasses), extra <indexTable> will not work.
					If it's NumpyMatrix or bytes object (or their subclasses), generate index table automatically if it is not provided.
		'''
		declare.belong_classes("data", data, [BytesVector, NumpyVector, ArkIndexTable, bytes])

		needIndexTableFlag = True

		if isinstance(data, BytesVector):
			self.__dataIndex = data.indexTable
			self.__dataIndex.rename(name)
			data = data.data
			needIndexTableFlag = False
		
		elif isinstance(data ,ArkIndexTable):
			data = data.fetch(arkType="vec", name=name)
			self.__dataIndex = data.indexTable
			data = data.data
			needIndexTableFlag = False

		elif isinstance(data, NumpyVector):
			data = (data.to_bytes()).data

		super().__init__(data, name)

		if needIndexTableFlag is True:
			if indexTable is None:
				self.__generate_index_table()
			else:
				declare.is_classes("indexTable", indexTable, ArkIndexTable)
				self.__verify_index_table(indexTable)
	
	def __verify_index_table(self, indexTable):
		'''
		Check the format of provided index table.
		'''
		newIndexTable = indexTable.sort("startIndex")
		start = 0
		for uttID, indexInfo in newIndexTable.items():
			if indexInfo.startIndex != start:
				raise WrongDataFormat(f"Start index of {uttID} dose not match: expected {start} but got {indexInfo.startIndex}.")
			if indexInfo.filePath is not None:
				newIndexTable[uttID] = indexInfo._replace(filePath=None)
			start += indexInfo.dataSize
		
		newIndexTable.rename(self.name)
		self.__dataIndex = newIndexTable

	def __read_one_record(self, fp):
		'''
		Read a utterance.
		'''
		utt = ''
		while True:
			char = fp.read(1).decode()
			if (char == '') or (char == ' '):break
			utt += char
		utt = utt.strip()
		if utt == '':
			if fp.read() == b'':
				return (None, None, None, None, None)
			else:
				fp.close()
				raise WrongDataFormat("Miss utterance ID before utterance.")
		binarySymbol = fp.read(2).decode()
		if binarySymbol == '\0B':
			dataSize = fp.read(1).decode()
			if dataSize != '\4':
				fp.close()
				if sizeSymbol not in ["C","F","D"]:
					raise WrongDataFormat(f"{type_name(self)} need vector data but this seems like matrix.")
				else:
					raise WrongDataFormat(f"We only support read size 4 int vector but got {dataSize}.")
			frames = int(np.frombuffer(fp.read(4), dtype='int32', count=1)[0])
			if frames == 0:
				buf = b""
			else:
				buferSize = frames * 5
				buf = fp.read(buferSize)
		else:
			fp.close()
			raise WrongDataFormat("Miss binary symbol before utterance. We do not support read kaldi archieves with text format.")
		
		return (utt, 4, frames, buferSize, buf)

	def __generate_index_table(self):
		'''
		Genrate the index table.
		'''
		if self.is_void:
			return None
		else:
			# Index table will have the same name with BytesMatrix object.
			self.__dataIndex = ArkIndexTable(name=self.name)
			start_index = 0
			with BytesIO(self.data) as sp:
				while True:
					(utt, dataSize, frames, bufSize, buf) = self.__read_one_record(sp)
					if utt is None:
						break
					oneRecordLen = len(utt) + 8 + bufSize
					self.__dataIndex[utt] = self.__dataIndex.spec(frames, start_index, oneRecordLen)
					start_index += oneRecordLen

	@property
	def indexTable(self):
		'''
		Get the index informat of utterances.
		
		Return:
			A ArkIndexTable object.
		'''
		# Return deepcopied dict object.
		return copy.deepcopy(self.__dataIndex)

	@property
	def utts(self):
		'''
		Get all utts ID.
		
		Return:
			a list of all utterance IDs.
		'''
		if self.is_void:
			return []
		else:
			return list(self.__dataIndex.keys())

	@property
	def lens(self):
		'''
		Get the numbers of utterances.
		If you want to get the frames of each utterance, try:
						obj.indexTable 
		attribute.
		
		Return:
			a int value.
		'''
		lengths = 0
		if not self.is_void:
			lengths = len(self.indexTable)
		
		return lengths

	@property
	def dtype(self):
		return "int32"

	@property
	def dim(self):
		if self.is_void:
			return None
		else:
			return 0

	def to_numpy(self):
		'''
		Transform bytes data to numpy data.
		
		Return:
			a NumpyVector object sorted by utterance ID.
		'''
		newDict = {}
		if not self.is_void:
			sortedIndex = self.indexTable.sort(by="utt", reverse=False)
			with BytesIO(self.data) as sp:
				for utt, indexInfo in sortedIndex.items():
					sp.seek(indexInfo.startIndex)
					(utt, dataSize, frames, bufSize, buf) = self.__read_one_record(sp)
					vector = np.frombuffer(buf, dtype=[("size","int8"),("value","int32")], count=frames)
					vector = vector[:]["value"]
					newDict[utt] = vector

		return NumpyVector(newDict, name=self.name)
	
	def save(self, fileName, chunks=1, returnIndexTable=False):
		'''
		Save bytes data to file.

		Args:
			<fileName>: file name or file handle.
			<chunks>: If larger than 1, data will be saved to mutiple files averagely. This would be invalid when <fileName> is a file handle.
			<returnIndexTable>: If True, return the index table containing the information of file path.
		
		Return:
			the path of saved files.
		'''
		declare.not_void( type_name(self), self)
		declare.is_valid_file_name_or_handle("fileName", fileName)
		declare.in_boundary("chunks", chunks, minV=1)
		declare.is_bool("returnIndexTable", returnIndexTable)

		if isinstance(fileName, str):

			def save_chunk_data(chunkData, arkFileName, returnIndexTable):

				make_dependent_dirs(arkFileName, pathIsFile=True)
				with open(arkFileName, "wb") as fw:
					fw.write(chunkData.data)
				
				if returnIndexTable is True:
					indexTable = chunkData.indexTable
					for uttID in indexTable.keys():
						indexTable[uttID]._replace(filePath=arkFileName)

					return indexTable
				else:
					return arkFileName

			fileName = fileName.strip()
			if chunks == 1:
				savedFiles = save_chunk_data(self, fileName, returnIndexTable)	
			
			else:
				dirName = os.path.dirname(fileName)
				fileName = os.path.basename(fileName)
				savedFiles = []
				chunkDataList = self.subset(chunks=chunks)
				for i, chunkData in enumerate(chunkDataList):
					chunkFileName = os.path.join( dirName, f"ck{i}_{fileName}" )
					savedFiles.append( save_chunk_data(chunkData, chunkFileName, returnIndexTable) )

			return savedFiles
		
		else:
			fileName.truncate()
			fileName.write(self.data)
			fileName.seek(0)

			return fileName
		
	def __add__(self, other):
		'''
		The Plus operation between two objects.

		Args:
			<other>: a BytesVector or NumpyVector object.
		Return:
			a new BytesVector object.
		'''
		declare.belong_classes("other", other, [BytesVector, NumpyVector, ArkIndexTable])

		if isinstance(other, NumpyVector):
			other = other.to_bytes()
		elif isinstance(other, ArkIndexTable):
			uttIDs = [ utt for utt in other.keys() if utt not in self.utts ]
			other = other.fecth(arkType="vec", uttIDs=uttIDs)
		
		newName = f"plus({self.name},{other.name})"
		if self.is_void:
			result = copy.deepcopy(other)
			result.rename(newName)
			return result
		elif other.is_void:
			result = copy.deepcopy(self)
			result.rename(newName)
			return result

		selfUtts = self.utts
		newDataIndex = self.indexTable
		#lastIndexInfo = list(newDataIndex.sort(by="startIndex", reverse=True).values())[0]
		start = len(self.data)

		newData = []
		with BytesIO(other.data) as op:
			for utt, indexInfo in other.indexTable.items():
				if not utt in selfUtts:
					op.seek( indexInfo.startIndex )
					data = op.read( indexInfo.dataSize )
					data_size = indexInfo.dataSize
					newDataIndex[utt] = newDataIndex.spec(indexInfo.frames, start, data_size)
					start += data_size
					newData.append(data)

		return BytesVector(b''.join([self.data, *newData]), name=newName, indexTable=newDataIndex)

	def __call__(self, utt):
		'''
		Pick out one record.
		
		Args:
			<utt>: a string.
		Return:
			If existed, return a new BytesVector object.
		''' 
		declare.is_valid_string("utt", utt)
		if self.is_void:
			return None

		utt = utt.strip()

		if utt not in self.utts:
			return None
		else:
			indexInfo = self.indexTable[utt]
			newName = f"pick({self.name},{utt})"
			newDataIndex = ArkIndexTable(name=newName)
			with BytesIO(self.data) as sp:
				sp.seek( indexInfo.startIndex )
				data = sp.read( indexInfo.dataSize )

				newDataIndex[utt] =	indexInfo._replace(startIndex=0)
				result = BytesVector(data, name=newName, indexTable=newDataIndex)
			
			return result

	def subset(self, nHead=0, nTail=0, nRandom=0, chunks=1, uttIDs=None):
		'''
		Subset data.
		The priority of mode is nHead > nTail > nRandom > chunks > uttIDs.
		If you chose mutiple modes, only the prior one will work.
		
		Args:
			<nHead>: get N head utterances.
			<nTail>: get N tail utterances.
			<nRandom>: sample N utterances randomly.
			<chunks>: split data into N chunks averagely.
			<uttIDs>: pick out these utterances whose ID in uttIDs.
		Return:
			a new BytesVector object or a list of new BytesVector objects.
		''' 
		declare.not_void(type_name(self), self)

		if nHead > 0:
			declare.is_positive_int("nHead", nHead)
			newName = f"subset({self.name},head {nHead})"
			newDataIndex = ArkIndexTable(name=newName)
			totalSize = 0
			
			for utt, indexInfo in self.indexTable.items():
				newDataIndex[utt] = newDataIndex.spec(indexInfo.frames, totalSize, indexInfo.dataSize)
				totalSize += indexInfo.dataSize
				nHead -= 1
				if nHead <= 0:
					break
			
			with BytesIO(self.data) as sp:
				sp.seek(0)
				data = sp.read(totalSize)
	
			return BytesVector(data, name=newName, indexTable=newDataIndex)

		elif nTail > 0:
			declare.is_positive_int("nTail", nTail)
			newName = f"subset({self.name},tail {nTail})"
			newDataIndex = ArkIndexTable(name=newName)

			tailNRecord = list(self.indexTable.items())[-nTail:]
			start_index = tailNRecord[0][1].startIndex

			totalSize = 0
			for utt, indexInfo in tailNRecord:
				newDataIndex[utt] = newDataIndex.spec(indexInfo.frames, totalSize, indexInfo.dataSize)
				totalSize += indexInfo.dataSize

			with BytesIO(self.data) as sp:
				sp.seek(start_index)
				data = sp.read(totalSize)
	
			return BytesVector(data, name=newName, indexTable=newDataIndex)

		elif nRandom > 0:
			declare.is_positive_int("nRandom", nRandom)
			randomNRecord = random.choices(list(self.indexTable.items()), k=nRandom)
			newName = f"subset({self.name},random {nRandom})"

			newDataIndex = ArkIndexTable(name=newName)
			start_index = 0
			newData = []
			with BytesIO(self.data) as sp:
				for utt, indexInfo in randomNRecord:
					sp.seek(indexInfo.startIndex)
					newData.append( sp.read(indexInfo.dataSize) )

					newDataIndex[utt] = newDataIndex.spec(indexInfo.frames, start_index, indexInfo.dataSize)
					start_index += indexInfo.dataSize

			return BytesVector(b"".join(newData), name=newName, indexTable=newDataIndex)

		elif chunks > 1:
			declare.is_positive_int("chunks", chunks)
			uttLens = list(self.indexTable.items())
			allLens = len(uttLens)
			chunkUtts = allLens//chunks
			if chunkUtts == 0:
				chunks = allLens
				chunkUtts = 1
				t = 0
				print(f"Warning: utterances is fewer than <chunks> so only {chunks} files will be saved.")
			else:
				t = allLens - chunkUtts * chunks
			
			datas = []
			with BytesIO(self.data) as sp:                          
				sp.seek(0)
				for i in range(chunks):
					newName = f"subset({self.name},chunk {chunks}-{i})"
					newDataIndex = ArkIndexTable(name=newName)
					if i < t:
						chunkItems = uttLens[i*(chunkUtts+1) : (i+1)*(chunkUtts+1)]
					else:
						chunkItems = uttLens[i*chunkUtts : (i+1)*chunkUtts]
					chunkLen = 0
					for utt, indexInfo in chunkItems:
						newDataIndex[utt] = newDataIndex.spec(indexInfo.frames, chunkLen, indexInfo.dataSize)
						chunkLen += indexInfo.dataSize
					chunkData = sp.read(chunkLen)
					
					datas.append( BytesVector(chunkData, name=newName, indexTable=newDataIndex) )
			return datas

		elif uttIDs != None:
			declare.is_classes("uttIDs", uttIDs, [str,list,tuple])
			if isinstance(uttIDs, str):
				newName = f"subset({self.name},uttIDs 1)"
				uttIDs = [uttIDs,]
			else:
				declare.members_are_valid_strings("uttIDs", uttIDs)
				newName = f"subset({self.name},uttIDs {len(uttIDs)})"

			newData = []
			dataIndex = self.indexTable
			newDataIndex = ArkIndexTable(name=newName)
			start_index = 0
			with BytesIO(self.data) as sp:
				for utt in uttIDs:
					if utt in self.utts:
						indexInfo = dataIndex[utt]
						sp.seek( indexInfo.startIndex )
						newData.append( sp.read(indexInfo.dataSize) )

						newDataIndex[utt] = newDataIndex.spec(indexInfo.frames, start_index, indexInfo.dataSize)
						start_index += indexInfo.dataSize

			return BytesVector(b''.join(newData), name=newName, indexTable=newDataIndex)
		
		else:
			raise WrongOperation('Expected one of <nHead>, <nTail>, <nRandom>, <chunks> or <uttIDs> is avaliable but all got the default value.')

	def sort(self, by="utt", reverse=False):
		'''
		Sort utterances by frames length or uttID

		Args:
			<by>: "frame" or "utt"
			<reverse>: If reverse, sort in descending order.
		Return:
			A new BytesVector object.
		''' 
		declare.is_instances("by", by, ["utt", "frame"])

		newDataIndex = self.indexTable.sort(by=by, reverse=reverse)
		ordered = True
		for i, j in zip(self.indexTable.items(), newDataIndex.items()):
			if i != j:
				ordered = False
				break
		if ordered:
			return copy.deepcopy(self)

		with BytesIO(self.data) as sp:
			if sys.getsizeof(self.data) > 10**9:
				## If the data size is large, divide it into N chunks and save it to intermidiate file.
				with FileHandleManager as fhm:
					temp = fhm.create("wb+")
					chunkdata = []
					chunkSize = 50
					count = 0
					start_index = 0
					for utt, indexInfo in newDataIndex.items():
						newDataIndex[utt] = indexInfo._replace(startIndex=start_index)
						start_index += indexInfo.dataSize

						sp.seek( indexInfo.startIndex )
						chunkdata.append( sp.read(indexInfo.dataSize) )
						count += 1
						if count >= chunkSize:
							temp.write( b"".join(chunkdata) )
							del chunkdata
							chunkdata = []
							count = 0
					if len(chunkdata) > 0:
						temp.write( b"".join(chunkdata) )
						del chunkdata
					temp.seek(0)
					newData = temp.read()
		
			else:
				newData = []
				start_index = 0
				for utt, indexInfo in newDataIndex.items():
					sp.seek( indexInfo.startIndex )
					newData.append( sp.read(indexInfo.dataSize) )

					newDataIndex[utt] = indexInfo._replace(startIndex=start_index)
					start_index += indexInfo.dataSize

				newData = b"".join(newData)

		return BytesVector(newData, name=self.name, indexTable=newDataIndex)		

## Subclass: for transition-ID alignment
class BytesAlignmentTrans(BytesVector):
	'''
	Hold the alignment(transition ID) with kaldi binary format.
	'''
	def __init__(self, data=b"", name="transitionID", indexTable=None):
		'''
		Only allow BytesAlignmentTrans, NumpyAlignmentTrans, ArkIndexTable or bytes (do not extend to their subclasses and their parent-classes).
		'''
		declare.is_classes("data", data, [BytesAlignmentTrans, NumpyAlignmentTrans, ArkIndexTable, bytes])

		super().__init__(data, name, indexTable)

	def to_numpy(self, aliType="transitionID", hmm=None):
		'''
		Transform alignment to numpy format.

		Args:
			<aliType>: If it is "transitionID", transform to transition IDs.
					  If it is "phoneID", transform to phone IDs.
					  If it is "pdfID", transform to pdf IDs.
			<hmm>: None, or hmm file or exkaldi HMM object.

		Return:
			a NumpyAlignmentTrans or NumpyAlignmentPhone or NumpyAlignmentPdf object.
		'''
		declare.is_instances("aliType", aliType, ["transitionID", "pdfID", "phoneID"])
		if self.is_void:
			if aliType == "transitionID":
				return NumpyAlignmentTrans(name=self.name)
			elif aliType == "phoneID":
				return NumpyAlignmentPhone(name=f"to_phone({self.name})")
			else:
				return NumpyAlignmentPdf(name=f"to_pdf({self.name})")

		def transform(data, cmd):
			out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=data)
			if (isinstance(cod,int) and cod != 0) or out == b'':
				print(err.decode())
				raise KaldiProcessError('Failed to transform alignment.')
			else:
				result = {}
				sp = BytesIO(out)
				for line in sp.readlines():
					line = line.decode()
					line = line.strip().split()
					utt = line[0]
					matrix = np.array(line[1:], dtype=np.int32)
					result[utt] = matrix
				return result

		if aliType == "transitionID":
			result = super().to_numpy()
			return NumpyAlignmentTrans(result.data, self.name)
		
		else:
			declare.kaldi_existed()
			declare.is_potential_hmm("hmm", hmm)

			with FileHandleManager() as fhm:
				
				if not isinstance(hmm, str):
					temp = fhm.create("wb+", suffix=".mdl")
					hmm.save(temp)
					hmm = temp.name

				if aliType == "phoneID":
					cmd = f"ali-to-phones --per-frame=true {hmm} ark:- ark,t:-"
					result = transform(self.data, cmd)
					newName = f"to_phone({self.name})"
					return NumpyAlignmentPhone(result, newName)

				else:
					cmd = f"ali-to-pdf {hmm} ark:- ark,t:-"
					result = transform(self.data, cmd)
					newName = f"to_pdf({self.name})"
					return NumpyAlignmentPdf(result, newName)

	def __add__(self, other):
		'''
		The Plus operation between two objects.

		Args:
			<other>: a BytesAlignmentTrans or NumpyAlignmentTrans object.
		Return:
			a new BytesAlignmentTrans object.
		''' 
		declare.is_alignment("other", other)
		result = super().__add__(other)

		return BytesAlignmentTrans(result.data, result.name, result.indexTable)

	def __call__(self, uttID):
		'''
		Pick out a record.
		
		Args:
			<uttID>: a string.
		Return:
			If existed, return a new BytesAlignmentTrans object.
		''' 
		result = super().__call__(uttID)
		if result is not None:
			result = BytesAlignmentTrans(result.data, result.name, result.indexTable)
		return result

	def subset(self, nHead=0, nTail=0, nRandom=0, chunks=1, uttIDs=None):
		'''
		Subset data.
		The priority of mode is nHead > nTail > nRandom > chunks > uttIDs.
		If you chose mutiple modes, only the prior one will work.
		
		Args:
			<nHead>: get N head utterances.
			<nTail>: get N tail utterances.
			<nRandom>: sample N utterances randomly.
			<chunks>: split data into N chunks averagely.
			<uttIDs>: pick out these utterances whose ID in uttIDs.
		Return:
			a new BytesAlignmentTrans object or a list of new BytesAlignmentTrans objects.
		''' 
		result = super().subset(nHead, nTail, nRandom, chunks, uttIDs)

		if isinstance(result, list):
			for index in range(len(result)):
				temp = result[index]
				result[index] = BytesAlignmentTrans(temp.data, temp.name, temp.indexTable)
		else:
			result = BytesAlignmentTrans(result.data, result.name, result.indexTable)

		return result

	def sort(self, by="utt", reverse=False):
		'''
		Sort utterances by frames length or utterance ID.

		Args:
			<by>: "frame" or "utt".
			<reverse>: If reverse, sort in descending order.
		Return:
			A new BytesAlignmentTrans object.
		''' 
		result = super().sort(by, reverse)
		return BytesAlignmentTrans(result.data, name=result.name, indexTable=result.indexTable)

'''NumpyArchieve class group'''
'''Designed for Kaldi binary archieve table (in Numpy Format)'''
## Base Class
class NumpyArchieve:

	def __init__(self, data={}, name=None):
		if data is not None:
			declare.is_classes("data", data, dict)
		self.__data = data

		if name is None:
			self.__name = self.__class__.__name__
		else:
			declare.is_valid_string("name", name)
			self.__name = name	
	
	@property
	def data(self):
		return self.__data

	def reset_data(self, newData):
		if newData is not None:
			declare.is_classes("data", newData, dict)
		del self.__data
		self.__data = newData

	@property
	def is_void(self):
		if self.__data is None or len(self.__data) == 0:
			return True
		else:
			return False

	@property
	def name(self):
		return self.__name

	def rename(self, newName):
		declare.is_valid_string("name", newName)
		self.__name = newName

	def keys(self):
		return self.__data.keys()
	
	def values(self):
		return self.__data.values()

	def items(self):
		return self.__data.items()	

## Base Class: for Matrix Data Archieves 
class NumpyMatrix(NumpyArchieve):
	'''
	A base class for matrix data, such as feature, cmvn statistics, post probability.
	'''
	def __init__(self, data={}, name="mat"):
		'''
		Args:
			<data>: BytesMatrix or ArkIndexTable object or NumpyMatrix or dict object (or their subclasses)
		'''
		declare.belong_classes("data", data, [BytesMatrix, NumpyMatrix, ArkIndexTable, dict])

		if isinstance(data, BytesMatrix):
			data = data.to_Numpy().data
		elif isinstance(data, ArkIndexTable):
			data = data.fetch(arkType="mat").to_Numpy().data
		elif isinstance(data, NumpyMatrix):
			data = data.data

		super().__init__(data, name)

	@property
	def dtype(self):
		'''
		Get the data type of Numpy data.
		
		Return:
			A string, 'float32', 'float64'.
		'''  
		_dtype = None
		if not self.is_void:
			utts = self.utts
			_dtype = str(self.data[utts[0]].dtype)
		return _dtype
	
	@property
	def dim(self):
		'''
		Get the data dimensions.
		
		Return:
			If data is void, return None, or return an int value.
		'''		
		_dim = None
		if not self.is_void:
			utts = self.utts
			if len(self.data[utts[0]].shape) <= 1:
				_dim = 0
			else:
				_dim = self.data[utts[0]].shape[1]
		
		return _dim
		
	def to_dtype(self, dtype):
		'''
		Transform data type.

		Args:
			<dtype>: a string of "float", "float32" or "float64". IF "float", it will be treated as "float32".
		Return:
			A new NumpyMatrix object.
		'''
		declare.is_instances("dtype", dtype, ['float','float32','float64'])
		declare.not_void( type_name(self), self)

		if dtype == 'float': 
			dtype = 'float32'

		if self.dtype == dtype:
			newData = copy.deepcopy(self.data)
		else:
			newData = {}
			for utt in self.utts:
				newData[utt] = np.array(self.data[utt], dtype=dtype)
		
		return NumpyMatrix(newData, name=self.name)

	@property
	def utts(self):
		'''
		Get all utts ID.
		
		Return:
			a list of all utterance IDs.
		'''
		return list(self.data.keys())
	
	def check_format(self):
		'''
		Check if data has right kaldi format.
		
		Return:
			If data is void, return False.
			If data has right format, return True, or raise Error.
		'''
		if self.is_void:
			return False

		_dim = 'unknown'
		for utt in self.utts:

			declare.is_valid_string("key", utt)
			declare.is_classes("value", self.data[utt], np.ndarray)
			matrixShape = self.data[utt].shape

			if len(matrixShape) > 2:
				raise WrongDataFormat(f'Expected the shape of matrix is like [ frame length, dimension ] but got {matrixShape}.')
			elif len(matrixShape) == 2:
				if _dim == 'unknown':
					_dim = matrixShape[1]
				elif matrixShape[1] != _dim:
					raise WrongDataFormat(f"Expected uniform data dimension {_dim} but got {matrixShape[1]} at utt {utt}.")
			else:
				if _dim == "unknown":
					_dim = 0
				elif _dim != 0:
					raise WrongDataFormat(f"Expected data dimension {_dim} but got 0 at utt {utt}.")
		return True

	def to_bytes(self):
		'''
		Transform numpy data to bytes data.
		
		Return:
			a BytesMatrix object.
		'''	
		self.check_format()

		newDataIndex = ArkIndexTable(name=self.name)
		newData = []
		start_index = 0
		for utt in self.utts:
			matrix = self.data[utt]
			data = (utt+' ').encode()
			data += '\0B'.encode()
			if matrix.dtype == 'float32':
				data += 'FM '.encode()
			elif matrix.dtype == 'float64':
				data += 'DM '.encode()
			else:
				raise UnsupportedType(f'Expected "float32" or "float64" data, but got {matrix.dtype}.')
			
			data += '\04'.encode()
			data += struct.pack(np.dtype('uint32').char, matrix.shape[0])
			data += '\04'.encode()
			data += struct.pack(np.dtype('uint32').char, matrix.shape[1])
			data += matrix.tobytes()

			oneRecordLen = len(data)
			newDataIndex[utt] = newDataIndex.spec(matrix.shape[0], start_index, oneRecordLen)
			start_index += oneRecordLen

			newData.append(data)

		return BytesMatrix(b''.join(newData), self.name, newDataIndex)

	def save(self, fileName, chunks=1):
		'''
		Save numpy data to file.

		Args:
			<fileName>: file name. Defaultly suffix ".npy" will be add to the name.
			<chunks>: If larger than 1, data will be saved to mutiple files averagely.		
		Return:
			the path of saved files.
		'''
		declare.not_void( type_name(self), self)
		declare.is_valid_string("fileName", fileName)
		declare.in_boundary("chunks", chunks, minV=1)
		fileName = fileName.strip()

		if not fileName.endswith('.npy'):
			fileName += '.npy'

		make_dependent_dirs(fileName, pathIsFile=True)
		if chunks == 1:    
			allData = tuple(self.data.items())
			np.save(fileName, allData)
			return fileName
		else:
			chunkDataList = self.subset(chunks=chunks)

			dirName = os.path.dirname(fileName)
			fileName = os.path.basename(fileName)

			savedFiles = []
			for i,chunkData in enumerate(chunkDataList):
				chunkFileName = os.path.join( dirName, f"ck{i}_"+fileName )
				chunkData = tuple(self.data.items())
				np.save(chunkFileName, chunkData)
				savedFiles.append(chunkFileName)	
		
			return savedFiles

	def __add__(self, other):
		'''
		The Plus operation between two objects.

		Args:
			<other>: a BytesMatrix or NumpyMatrix object.
		Return:
			a new NumpyMatrix object.
		''' 
		declare.belong_classes("other", other, [BytesMatrix, NumpyMatrix, ArkIndexTable])

		if isinstance(other, BytesMatrix):
			other = other.to_numpy()
		elif isinstance(other, ArkIndexTable):
			uttIDs = [ utt for utt in other.keys() if utt not in self.utts ]
			other = other.fecth(arkType="mat", uttIDs=uttIDs).to_numpy()
		
		newName = f"plus({self.name},{other.name})"
		if self.is_void:
			result = copy.deepcopy(other)
			result.rename(newName)
			return result
		elif other.is_void:
			result = copy.deepcopy(self)
			result.rename(newName)
			return result
		elif self.dim != other.dim:
			raise WrongOperation(f"Data dimensions does not match: {self.dim}!={other.dim}.")

		temp = self.data.copy()
		selfUtts = list(self.utts)
		for utt in other.utts:
			if not utt in selfUtts:
				temp[utt] = other.data[utt]

		return NumpyMatrix(temp, newName)

	def __call__(self, uttID):
		'''
		Pick out the specified utterance.
		
		Args:
			<uttID>: a string.
		Return:
			If existed, return a new NumpyMatrix object.
		''' 
		declare.is_valid_string("uttID", uttID)
		if self.is_void:
			return None

		uttID = uttID.strip()

		if uttID not in self.utts:
			return None
		else:
			newName = f"pick({self.name},{uttID})"
			return NumpyMatrix({uttID:self.data[uttID]}, newName)

	@property
	def lens(self):
		'''
		Get the numbers of utterances.
		
		Return:
			a int value.
		'''
		if self.is_void:
			return 0
		else:
			return len(self.data)

	def subset(self, nHead=0, nTail=0, nRandom=0, chunks=1, uttIDs=None):
		'''
		Subset data.
		The priority of mode is nHead > nTail > nRandom > chunks > uttIDs.
		If you chose mutiple modes, only the prior one will work.
		
		Args:
			<nHead>: get N head utterances.
			<nTail>: get N tail utterances.
			<nRandom>: sample N utterances randomly.
			<chunks>: split data into N chunks averagely.
			<uttIDs>: pick out these utterances whose ID in uttIDs.
		Return:
			a new NumpyMatrix object or a list of new NumpyMatrix objects.
		''' 
		declare.not_void(type_name(self), self)

		if nHead > 0:
			declare.is_positive_int("nHead", nHead)
			newDict = {}
			for utt in self.utts[0:nHead]:
				newDict[utt]=self.data[utt]
			newName = f"subset({self.name},head {nHead})"
			return NumpyMatrix(newDict, newName)

		elif nTail > 0:
			declare.is_positive_int("nTail", nTail)
			newDict = {}
			for utt in self.utts[-nTail:]:
				newDict[utt]=self.data[utt]
			newName = f"subset({self.name},tail {nTail})"
			return NumpyMatrix(newDict, newName)

		elif nRandom > 0:
			declare.is_positive_int("nRandom", nRandom)
			newDict = dict(random.choices(self.items(), k=nRandom))
			newName = f"subset({self.name},tail {nRandom})"
			return NumpyMatrix(newDict, newName)

		elif chunks > 1:
			declare.is_positive_int("chunks", chunks)

			datas = []
			allLens = len(self.data)
			if allLens != 0:
				utts = self.utts
				chunkUtts = allLens//chunks
				if chunkUtts == 0:
					chunks = allLens
					chunkUtts = 1
					t = 0
				else:
					t = allLens - chunkUtts * chunks

				for i in range(chunks):
					temp = {}
					if i < t:
						chunkuttIDs = utts[i*(chunkUtts+1):(i+1)*(chunkUtts+1)]
					else:
						chunkuttIDs = utts[i*chunkUtts:(i+1)*chunkUtts]
					for utt in chunkuttIDs:
						temp[utt]=self.data[utt]
					newName = f"subset({self.name},chunk {chunks}-{i})"
					datas.append( NumpyMatrix(temp, newName) )
			return datas

		elif uttIDs != None:
			declare.is_classes("uttIDs", uttIDs, [str,list,tuple])
			if isinstance(uttIDs,str):
				newName = f"subset({self.name},uttIDs 1)"
				uttIDs = [uttIDs,]
			else:
				declare.members_are_valid_strings("uttIDs", uttIDs)
				newName = f"subset({self.name},uttIDs {len(uttIDs)})"

			newDict = {}
			selfKeys = self.utts
			for utt in uttIDs:
				if utt in selfKeys:
					newDict[utt] = self.data[utt]
				else:
					#print('Subset Warning: no data for utt {}'.format(utt))
					continue
			return NumpyMatrix(newDict, newName)
		
		else:
			raise WrongOperation('Expected one of <nHead>, <nTail>, <nRandom> or <chunks> is avaliable but all got default value.')

	def sort(self, by='frame', reverse=False):
		'''
		Sort utterances by frame length or uttID.

		Args:
			<by>: "frame" or "utt"
			<reverse>: If reverse, sort in descending order.
		Return:
			A new NumpyMatrix object.
		''' 
		declare.is_instances("by", by, ["utt","frame"])
		declare.is_bool("reverse", reverse)

		if by == "utt":
			items = sorted(self.items(), key=lambda x:x[0], reverse=reverse)
		else:
			items = sorted(self.items(), key=lambda x:len(x[1]), reverse=reverse)
		
		newName = "sort({},{})".format(self.name, by)
		return NumpyMatrix(dict(items), newName)

	def map(self, func):
		'''
		Map all arrays to a function.

		Args:
			<func>: callable function object.
		
		Return:
			A new NumpyMatrix object.
		'''
		declare.is_callable("func", func)

		new = dict(map( lambda x:(x[0],func(x[1])), self.data.items() ))

		return NumpyMatrix(new, name=f"mapped({self.name})")

## Subclass: for acoustic feature
class NumpyFeature(NumpyMatrix):
	'''
	Hold the feature with Numpy format.
	'''
	def __init__(self, data={}, name="feat"):
		'''
		Only allow BytesFeature, NumpyFeature, ArkIndexTable or dict (do not extend to their subclasses and their parent-classes).
		'''		
		declare.is_classes("data", data, [BytesFeature, NumpyFeature, ArkIndexTable, dict])

		super().__init__(data, name)

	def to_dtype(self, dtype):
		'''
		Transform data type.

		Args:
			<dtype>: a string of "float", "float32" or "float64". IF "float", it will be treated as "float32".
		Return:
			A new NumpyFeature object.
		'''
		result = super().to_dtype(dtype)

		return NumpyFeature(result.data, result.name)

	def to_bytes(self):
		'''
		Transform feature to bytes format.

		Return:
			a BytesFeature object.
		'''		
		result = super().to_bytes()
		return BytesFeature(result.data, self.name, result.indexTable)
	
	def __add__(self, other):
		'''
		Plus operation between two feature objects.

		Args:
			<other>: a BytesFeature or NumpyFeature object.
		Return:
			a NumpyFeature object.
		'''
		declare.is_feature("other", other)

		result = super().__add__(other)

		return NumpyFeature(result.data, result.name)

	def __call__(self, uttID):
		'''
		Pick out the specified utterance.
		
		Args:
			<uttID>: a string.
		Return:
			If existed, return a new NumpyFeature object.
		''' 
		result = super().__call__(uttID)
		if result is not None:
			result = NumpyFeature(data=result.data, name=result.name)
		return result

	def splice(self, left=4, right=None):
		'''
		Splice front-behind N frames to generate new feature data.

		Args:
			<left>: the left N-frames to splice.
			<right>: the right N-frames to splice. If None, right = left.
		Return:
			a new NumpyFeature object whose dim became original-dim * (1 + left + right).
		''' 
		declare.not_void(type_name(self), self)
		declare.is_non_negative_int("left", left)

		if right is None:
			right = left
		else:
			declare.is_non_negative_int("right", right)

		lengths = []
		matrixes = []
		for utt in self.utts:
			lengths.append((utt,len(self.data[utt])))
			matrixes.append(self.data[utt])

		leftSub = []
		rightSub = []
		for i in range(left):
			leftSub.append(matrixes[0][0])
		for j in range(right):
			rightSub.append(matrixes[-1][-1])
		matrixes = np.row_stack([*leftSub,*matrixes,*rightSub])
		
		N = matrixes.shape[0]
		dim = matrixes.shape[1]
		newMat=np.empty([N,dim*(left+right+1)])

		index = 0
		for lag in range(right,-left-1,-1):
			newMat[:,index:index+dim]=np.roll(matrixes,lag,axis=0)
			index += dim
		newMat = newMat[left:(0-right),:]

		newFea = {}
		index = 0
		for utt,length in lengths:
			newFea[utt] = newMat[index:index+length]
			index += length
		newName = f"splice({self.name},{left},{right})"
		return NumpyFeature(newFea, newName)
	
	def select(self, dims, retain=False):
		'''
		Select specified dimensions of feature.

		Args:
			<dims>: A int value or string such as "1,2,5-10"
			<retain>: If True, return the rest dimensions of feature simultaneously.
		Return:
			A new NumpyFeature object or two NumpyFeature objects.
		''' 
		declare.not_void(type_name(self), self)
		declare.is_bool("retain", retain)

		_dim = self.dim
		if isinstance(dims,int):
			declare.in_boundary("dims", dims, minV=0, maxV=_dim-1)
			selectFlag = [dims,]
		elif isinstance(dims,str):
			declare.is_valid_string("dims", dims)
			temp = dims.strip().split(',')
			selectFlag = []
			for i in temp:
				if not '-' in i:
					try:
						i = int(i)
					except ValueError:
						raise WrongOperation(f'Expected int value but got {i}.')
					else:
						declare.in_boundary("dims", i, minV=0, maxV=_dim-1)
						selectFlag.append( i )
				else:
					i = i.split('-')
					assert len(i) == 2, f"<dims> should has format like '1-3','4-','-5'."
					if i[0].strip() == '':
						i[0] = 0
					if i[1].strip() == '':
						i[1] = _dim-1
					try:
						i[0] = int(i[0])
						i[1] = int(i[1])
					except ValueError:
						raise WrongOperation('Expected selection index is int value.')
					else:
						if i[0] > i[1]:
							i[0], i[1] = i[1], i[0]
						declare.in_boundary("dims", i[1], maxV=_dim-1)
						selectFlag.extend([x for x in range(int(i[0]),int(i[1])+1)])
		else:
			raise WrongOperation(f'Expected <dims> is int value or string like 1,4-9,12 but got {type_name(dims)}.')

		retainFlag = sorted(list(set(selectFlag)))

		seleDict = {}
		if retain:
			reseDict = {}

		for utt in self.utts:
			newMat = []
			for index in selectFlag:
				newMat.append(self.data[utt][:,index][:,None])
			newMat = np.concatenate(newMat, axis=1)
			seleDict[utt] = newMat
			if retain:
				if len(retainFlag) == _dim:
					continue
				else:
					matrix = self.data[utt].copy()
					reseDict[utt] = np.delete(matrix, retainFlag, 1)
		newNameSele = f"select({self.name},{dims})"
		if retain:
			newNameRese = f"select({self.name},not {dims})"
			return NumpyFeature(seleDict, newNameSele), NumpyFeature(reseDict, newNameRese)
		else:
			return NumpyFeature(seleDict, newNameSele)

	def subset(self, nHead=0, nTail=0, nRandom=0, chunks=1, uttIDs=None):
		'''
		Subset data.
		The priority of mode is nHead > nTail > nRandom > chunks > uttIDs.
		If you chose mutiple modes, only the prior one will work.
		
		Args:
			<nHead>: get N head utterances.
			<nTail>: get N tail utterances.
			<nRandom>: sample N utterances randomly.
			<chunks>: split data into N chunks averagely.
			<uttIDs>: pick out these utterances whose ID in uttIDs.
		Return:
			a new NumpyFeature object or a list of new NumpyFeature objects.
		''' 
		result = super().subset(nHead, nTail, nRandom, chunks, uttIDs)

		if isinstance(result, list):
			for index in range(len(result)):
				temp = result[index]
				result[index] = NumpyFeature(temp.data, temp.name)
		else:
			result = NumpyFeature(result.data, result.name)

		return result

	def sort(self, by='utt', reverse=False):
		'''
		Sort utterances by frames length or uttID

		Args:
			<by>: "frame" or "utt"
			<reverse>: If reverse, sort in descending order.
		Return:
			A new NumpyFeature object.
		''' 		
		result = super().sort(by, reverse)

		return NumpyFeature(result.data, result.name)

	def normalize(self, std=True, alpha=1.0, beta=0.0, epsilon=1e-8, axis=0):
		'''
		Standerd normalize a feature at a file field.
		If std is True, Do: 
					alpha * (x-mean)/(stds + epsilon) + belta, 
		or do: 
					alpha * (x-mean) + belta.

		Args:
			<std>: True of False.
			<alpha>,<beta>: a float value.
			<epsilon>: a extremely small float value.
			<axis>: the dimension to normalize.
		Return:
			A new NumpyFeature object.
		'''
		declare.not_void(type_name(self), self)
		declare.is_bool("std", std)
		declare.is_positive("alpha", alpha)
		declare.is_classes("belta", beta, [float,int])
		declare.is_positive_float("epsilon", epsilon)
		declare.is_classes("axis", axis, int)

		utts = []
		lens = []
		data = []
		for uttID, matrix in self.data.items():
			utts.append(uttID)
			lens.append(len(matrix))
			data.append(matrix)

		data = np.row_stack(data)
		mean = np.mean(data, axis=axis)

		if std is True:  
			std = np.std(data,axis=axis)
			data = alpha*(data-mean)/(std+epsilon) + beta
		else:
			data = alpha*(data-mean) + beta

		newDict = {}
		start = 0
		for index,uttID in enumerate(utts):
			newDict[uttID] = data[start:(start+lens[index])]
			start += lens[index]

		newName = f"norm({self.name},std {std})"
		return NumpyFeature(newDict, newName) 

	def cut(self, maxFrames):
		'''
		Cut long utterance to mutiple shorter ones. 

		Args:
			<maxFrames>: a positive int value. Cut a utterance by the thresold value of 1.25*maxFrames into a maxFrames part and a rest part.
						If the rest part is still longer than 1.25*maxFrames, continue to cut it. 
		Return:
			A new NumpyFeature object.
		''' 
		declare.not_void(type_name(self), self)
		declare.is_positive_int("maxFrames", maxFrames)

		newData = {}
		cutThreshold = maxFrames + maxFrames//4

		for utt in self.utts:
			matrix = self.data[utt]
			if len(matrix) <= cutThreshold:
				newData[utt] = matrix
			else:
				i = 0 
				while True:
					newData[utt+"_"+str(i)] = matrix[i*maxFrames:(i+1)*maxFrames]
					i += 1
					if len(matrix[i*maxFrames:]) <= cutThreshold:
						break
				if len(matrix[i*maxFrames:]) != 0:
					newData[utt+"_"+str(i)] = matrix[i*maxFrames:]
		
		newName = f"cut({self.name},{maxFrames})"
		return NumpyFeature(newData, newName)

	def paste(self, others):
		'''
		Concatenate feature arrays of the same uttID from mutiple objects in feature dimendion.

		Args:
			<others>: an object or a list of objects of NumpyFeature or BytesFeature.
		Return:
			a new NumpyFeature objects.
		'''
		if not isinstance(others, (list,tuple)):
			others = [others,]

		for index, other in enumerate(others):
			declare.is_feature("others", other)
			if isinstance(other, BytesFeature):
				others[index] = other.to_numpy()    
			elif isinstance(other, ArkIndexTable):
				others[index] = other.fetch("feat").to_numpy()  

		newDict = {}
		for utt in self.utts:
			newMat=[]
			newMat.append(self.data[utt])
			length = self.data[utt].shape[0]
			dim = self.dim
			
			for index, other in enumerate(others, start=1):
				if utt in other.utts:
					if other.data[utt].shape[0] != length:
						raise WrongDataFormat(f"Data frames {length}!={other[utt].shape[0]} at utterance ID {utt}.")
					newMat.append(other.data[utt])
				else:
					#print("Concat Warning: Miss data of utt id {} in later dict".format(utt))
					break

			if len(newMat) < len(others) + 1:
				#If any member miss the data of current utt id, abandon data of this utt id of all menbers
				continue

			newDict[utt] = np.column_stack(newMat)
		
		newName = f"paste({self.name}"
		for other in others:
			newName += f",{other.name}"
		newName += ")"

		return NumpyFeature(newDict, newName)

	def map(self, func):
		'''
		Map all arrays to a function.

		Args:
			<func>: callable function object.
		
		Return:
			A new NumpyFeature object.
		'''
		result = super().map(func)
		return NumpyFeature(data=result.data, name=result.name)	

## Subclass: for fMLLR transform matrix
class NumpyFmllrMatrix(NumpyMatrix):
	'''
	Hold the fMLLR transform matrix with Numpy format.
	'''
	def __init__(self, data={}, name="fmllrMat"):
		'''
		Only allow BytesFmllrMatrix, NumpyFmllrMatrix, ArkIndexTable or dict (do not extend to their subclasses and their parent-classes).
		'''		
		declare.is_classes("data", data, [BytesFmllrMatrix, NumpyFmllrMatrix, ArkIndexTable, dict])

		super().__init__(data,name)

	def to_bytes(self):
		'''
		Transform feature to bytes format.

		Return:
			a BytesFmllrMatrix object.
		'''			
		result = super().to_bytes()
		return BytesFmllrMatrix(result.data, result.name)

	def to_dtype(self, dtype):
		'''
		Transform data type.

		Args:
			<dtype>: a string of "float", "float32" or "float64". IF "float", it will be treated as "float32".
		Return:
			A new NumpyFmllrMatrix object.
		'''
		result = super().to_dtype(dtype)

		return NumpyFmllrMatrix(result.data, result.name)

	def __add__(self, other):
		'''
		Plus operation between two fMLLR transform matrix objects.

		Args:
			<other>: a NumpyFmllrMatrix or BytesFmllrMatrix object.
		Return:
			a NumpyFmllrMatrix object.
		'''	
		declare.is_fmllr_matrix("other", other)

		result = super().__add__(other)

		return NumpyFmllrMatrix(result.data, result.name)

	def subset(self, nHead=0, nTail=0, nRandom=0, chunks=1, uttIDs=None):
		'''
		Subset data.
		The priority of mode is nHead > nTail > nRandom > chunks > uttIDs.
		If you chose mutiple modes, only the prior one will work.
		
		Args:
			<nHead>: get N head utterances.
			<nTail>: get N tail utterances.
			<nRandom>: sample N utterances randomly.
			<chunks>: split data into N chunks averagely.
			<uttIDs>: pick out these utterances whose ID in uttIDs.
		Return:
			a new NumpyFmllrMatrix object or a list of new NumpyFmllrMatrix objects.
		''' 
		result = super().subset(nHead, nTail, nRandom, chunks, uttIDs)

		if isinstance(result, list):
			for index in range(len(result)):
				temp = result[index]
				result[index] = NumpyFmllrMatrix(temp.data, temp.name)
		else:
			result = NumpyFmllrMatrix(result.data, result.name)

		return result

	def sort(self, by='utt', reverse=False):
		'''
		Sort utterances by frames length or uttID

		Args:
			<by>: "frame" or "utt"
			<reverse>: If reverse, sort in descending order.
		Return:
			A new NumpyFmllrMatrix object.
		''' 		
		result = super().sort(by,reverse)

		return NumpyFmllrMatrix(result.data, result.name)

	def __call__(self, uttID):
		'''
		Pick out the specified utterance.
		
		Args:
			<uttID>: a string.
		Return:
			If existed, return a new NumpyFmllrMatrix object.
		''' 
		result = super().__call__(uttID)
		if result is not None:
			result = NumpyFmllrMatrix(data=result.data, name=result.name)
		return result

	def map(self, func):
		'''
		Map all arrays to a function.

		Args:
			<func>: callable function object.
		
		Return:
			A new NumpyFmllrMatrix object.
		'''
		result = super().map(func)
		return NumpyFmllrMatrix(data=result.data, name=result.name)	

## Subclass: for probability of neural network output
class NumpyProbability(NumpyMatrix):
	'''
	Hold the probability with Numpy format.
	'''
	def __init__(self, data={}, name="prob"):
		'''
		Only allow BytesProbability, NumpyProbability, ArkIndexTable or dict (do not extend to their subclasses and their parent-classes).
		'''	
		declare.is_classes("data", data, [BytesProbability, NumpyProbability, ArkIndexTable, dict])
		
		super().__init__(data, name)

	def to_bytes(self):
		'''
		Transform post probability to bytes format.

		Return:
			a BytesProbability object.
		'''				
		result = super().to_bytes()
		return BytesProbability(result.data, result.name)

	def to_dtype(self, dtype):
		'''
		Transform data type.

		Args:
			<dtype>: a string of "float", "float32" or "float64". IF "float", it will be treated as "float32".
		Return:
			A new NumpyProbability object.
		'''
		result = super().to_dtype(dtype)

		return NumpyProbability(result.data, result.name)

	def __add__(self, other):
		'''
		Plus operation between two post probability objects.

		Args:
			<other>: a NumpyProbability or BytesProbability object.
		Return:
			a NumpyProbability object.
		'''	
		declare.is_probability("other", other)

		result = super().__add__(other)

		return NumpyProbability(result.data, result.name)

	def subset(self, nHead=0, nTail=0, nRandom=0, chunks=1, uttIDs=None):
		'''
		Subset data.
		The priority of mode is nHead > nTail > nRandom > chunks > uttIDs.
		If you chose mutiple modes, only the prior one will work.
		
		Args:
			<nHead>: get N head utterances.
			<nTail>: get N tail utterances.
			<nRandom>: sample N utterances randomly.
			<chunks>: split data into N chunks averagely.
			<uttIDs>: pick out these utterances whose ID in uttIDs.
		Return:
			a new NumpyProbability object or a list of new NumpyProbability objects.
		''' 
		result = super().subset(nHead, nTail, nRandom, chunks, uttIDs)

		if isinstance(result, list):
			for index in range(len(result)):
				temp = result[index]
				result[index] = NumpyProbability( temp.data, temp.name )
		else:
			result = NumpyProbability( result.data,result.name )

		return result

	def sort(self, by='utt', reverse=False):
		'''
		Sort utterances by frames length or uttID

		Args:
			<by>: "frame" or "utt"
			<reverse>: If reverse, sort in descending order.
		Return:
			A new NumpyProbability object.
		'''	
		result = super().sort(by,reverse)

		return NumpyProbability(result.data, result.name)

	def __call__(self, uttID):
		'''
		Pick out the specified utterance.
		
		Args:
			<uttID>: a string.
		Return:
			If existed, return a new NumpyProbability object.
		''' 
		result = super().__call__(uttID)
		if result is not None:
			result = NumpyProbability(data=result.data, name=result.name)
		return result

	def map(self, func):
		'''
		Map all arrays to a function.

		Args:
			<func>: callable function object.
		
		Return:
			A new NumpyProbability object.
		'''
		result = super().map(func)
		return NumpyProbability(data=result.data, name=result.name)	

## Subclass: for CMVN statistics
class NumpyCMVNStatistics(NumpyMatrix):
	'''
	Hold the CMVN statistics with Numpy format.
	'''
	def __init__(self, data={}, name="cmvn"):
		'''
		Only allow BytesCMVNStatistics, NumpyCMVNStatistics, ArkIndexTable or dict (do not extend to their subclasses and their parent-classes).
		'''	
		declare.is_classes("data", data, [BytesCMVNStatistics, NumpyCMVNStatistics, ArkIndexTable, dict])

		super().__init__(data, name)

	def to_bytes(self):
		'''
		Transform feature to bytes format.

		Return:
			a BytesCMVNStatistics object.
		'''			
		result = super().to_bytes()
		return BytesCMVNStatistics(result.data, result.name)

	def to_dtype(self, dtype):
		'''
		Transform data type.

		Args:
			<dtype>: a string of "float", "float32" or "float64". IF "float", it will be treated as "float32".
		Return:
			A new NumpyCMVNStatistics object.
		'''
		result = super().to_dtype(dtype)

		return NumpyCMVNStatistics(result.data, result.name)

	def __add__(self, other):
		'''
		Plus operation between two CMVN statistics objects.

		Args:
			<other>: a NumpyCMVNStatistics or BytesCMVNStatistics object.
		Return:
			a NumpyCMVNStatistics object.
		'''	
		declare.is_cmvn("other", other)

		result = super().__add__(other)

		return NumpyCMVNStatistics(result.data, result.name)

	def subset(self, nHead=0, nTail=0, nRandom=0, chunks=1, uttIDs=None):
		'''
		Subset data.
		The priority of mode is nHead > nTail > nRandom > chunks > uttIDs.
		If you chose mutiple modes, only the prior one will work.
		
		Args:
			<nHead>: get N head utterances.
			<nTail>: get N tail utterances.
			<nRandom>: sample N utterances randomly.
			<chunks>: split data into N chunks averagely.
			<uttIDs>: pick out these utterances whose ID in uttIDs.
		Return:
			a new NumpyCMVNStatistics object or a list of new NumpyCMVNStatistics objects.
		''' 
		result = super().subset(nHead, nTail, nRandom, chunks, uttIDs)

		if isinstance(result, list):
			for index in range(len(result)):
				temp = result[index]
				result[index] = NumpyCMVNStatistics(temp.data, temp.name)
		else:
			result = NumpyCMVNStatistics(result.data, result.name)

		return result

	def sort(self, by='utt', reverse=False):
		'''
		Sort utterances by frames length or uttID

		Args:
			<by>: "frame" or "utt"
			<reverse>: If reverse, sort in descending order.
		Return:
			A new NumpyCMVNStatistics object.
		''' 		
		result = super().sort(by,reverse)

		return NumpyCMVNStatistics(result.data, result.name)

	def __call__(self, uttID):
		'''
		Pick out the specified utterance.
		
		Args:
			<uttID>: a string.
		Return:
			If existed, return a new NumpyCMVNStatistics object.
		''' 
		result = super().__call__(uttID)
		if result is not None:
			result = NumpyCMVNStatistics(data=result.data, name=result.name)
		return result

	def map(self, func):
		'''
		Map all arrays to a function.

		Args:
			<func>: callable function object.
		
		Return:
			A new NumpyCMVNStatistics object.
		'''
		result = super().map(func)
		return NumpyCMVNStatistics(data=result.data, name=result.name)	

## Base Class: for Vector Data Archieves
class NumpyVector(NumpyMatrix):
	'''
	Hold the kaldi vector data with Numpy format.
	'''
	def __init__(self, data={}, name="ali"):
		'''
		Args:
			<data>: Bytesvector or ArkIndexTable object or NumpyVector or dict object (or their subclasses).
		'''
		declare.belong_classes("data", data, [BytesVector, NumpyVector, ArkIndexTable, dict])

		if isinstance(data, BytesVector):
			data = data.to_Numpy().data
		elif isinstance(data, ArkIndexTable):
			data = data.fetch(arkType="vec").to_Numpy().data
		elif isinstance(data, NumpyMatrix):
			data = data.data

		super().__init__(data, name)

	@property
	def dim(self):
		if self.is_void:
			return None
		else:
			return 0

	def to_dtype(self, dtype):
		'''
		Transform data type.

		Args:
			<dtype>: a string of "int", "int32" or "int64". IF "int", it will be treated as "int32".
		Return:
			A new NumpyMatrix object.
		'''
		declare.is_instances("dtype", dtype, ['int','int32','int64'])
		declare.not_void( type_name(self), self)

		if dtype == 'int': 
			dtype = 'int32'

		if self.dtype == dtype:
			newData = copy.deepcopy(self.data)
		else:
			newData = {}
			for utt in self.utts:
				newData[utt] = np.array(self.data[utt], dtype=dtype)
		
		return NumpyVector(newData, name=self.name)

	def check_format(self):
		'''
		Check if data has right kaldi format.
		
		Return:
			If data is void, return False.
			If data has right format, return True, or raise Error.
		'''
		if not self.is_void:
			_dim = 'unknown'
			for utt in self.utts:
				declare.is_valid_string("key", utt)
				declare.is_classes("value", self.data[utt], np.ndarray)
				vector = self.data[utt]
				assert len(vector.shape) == 1, f"Vector should be 1-dim data but got {vector.shape}."
				assert vector.dtype in ["int32","int64"], f"Only support int data format but got {vector.dtype}."

			return True
		else:
			return False

	def to_bytes(self):
		'''
		Transform vector to bytes format.

		Return:
			a BytesVector object.
		'''
		self.check_format()
		if self.dtype == "int64":
			raise WrongDataFormat(f"Only int32 vector can be convert to bytes object in current version but this is: {self.dtype}")

		newDataIndex = ArkIndexTable(name=self.name)
		newData = []
		start_index = 0
		for utt, vector in self.items():
			oneRecord = []
			oneRecord.append( ( utt + ' ' + '\0B' + '\4' ).encode() )
			oneRecord.append( struct.pack(np.dtype('int32').char, vector.shape[0]) ) 
			for f, v in vector:
				oneRecord.append( '\4'.encode() + struct.pack(np.dtype('int32').char, v) )
			oneRecord = b"".join(oneRecord)
			newData.append( oneRecord )

			oneRecordLen = len(oneRecord)
			newDataIndex[utt] = newDataIndex.spec(vector.shape[0], start_index, oneRecordLen)
			start_index += oneRecordLen

		return BytesVector(b''.join(newData), name=self.name, indexTable=newDataIndex)

	def __add__(self, other):
		'''
		Plus operation between two vector objects.

		Args:
			<other>: a BytesVector or NumpyVector object.
		Return:
			a NumpyVector object.
		'''	
		declare.belong_classes("other", other, [BytesVector, NumpyVector, ArkIndexTable])
		
		result = super().__add__(other)

		return NumpyVector(result.data, result.name)

	def subset(self, nHead=0, nTail=0, nRandom=0, chunks=1, uttIDs=None):
		'''
		Subset data.
		The priority of mode is nHead > nTail > nRandom > chunks > uttIDs.
		If you chose mutiple modes, only the prior one will work.
		
		Args:
			<nHead>: get N head utterances.
			<nTail>: get N tail utterances.
			<nRandom>: sample N utterances randomly.
			<chunks>: split data into N chunks averagely.
			<uttIDs>: pick out these utterances whose ID in uttIDs.
		Return:
			a new NumpyVector object or a list of new NumpyVector objects.
		''' 
		result = super().subset(nHead, nTail, nRandom, chunks, uttIDs)

		if isinstance(result, list):
			for index in range(len(result)):
				temp = result[index]
				result[index] = NumpyVector( temp.data, temp.name )
		else:
			result = NumpyVector( result.data,result.name )

		return result

	def sort(self, by='utt', reverse=False):
		'''
		Sort utterances by frames length or uttID

		Args:
			<by>: "frame" or "utt"
			<reverse>: If reverse, sort in descending order.
		Return:
			A new NumpyVector object.
		'''	
		result = super().sort(by,reverse)

		return NumpyVector(result.data, result.name)

	def __call__(self, uttID):
		'''
		Pick out the specified utterance.
		
		Args:
			<uttID>: a string.
		Return:
			If existed, return a new NumpyVector object.
		''' 
		result = super().__call__(uttID)
		if result is not None:
			result = NumpyVector(data=result.data, name=result.name)
		return result	

	def map(self, func):
		'''
		Map all arrays to a function.

		Args:
			<func>: callable function object.
		
		Return:
			A new NumpyVector object.
		'''
		result = super().map(func)
		return NumpyVector(data=result.data, name=result.name)			

## Subclass: for transition-ID alignment 			
class NumpyAlignmentTrans(NumpyVector):
	'''
	Hold the alignment(transition ID) with Numpy format.
	'''
	def __init__(self, data={}, name="transitionID"):
		'''
		Only allow BytesAlignmentTrans, NumpyAlignmentTrans, ArkIndexTable or dict (do not extend to their subclasses and their parent-classes).
		'''	
		declare.is_classes("data", data, [BytesAlignmentTrans, NumpyAlignmentTrans, ArkIndexTable, dict])
	
		super().__init__(data, name)

	def to_bytes(self):
		'''
		Tansform numpy alignment to bytes format.

		Return:
			A BytesAlignmentTrans object.
		'''
		result = super(NumpyAlignmentTrans, self.to_dtype("int32")).to_bytes()
		return BytesAlignmentTrans(data=result.data, name=self.name, indexTable=result.indexTable)

	def to_dtype(self, dtype):
		'''
		Transform data type.

		Args:
			<dtype>: a string of "int", "int32" or "int64". IF "int", it will be treated as "int32".
		Return:
			A new NumpyAlignment object.
		'''
		result = super().to_dtype(dtype)

		return NumpyAlignmentTrans(result.data, result.name)

	def __add__(self, other):
		'''
		The Plus operation between two transition ID alignment objects.

		Args:
			<other>: a NumpyAlignmentTrans or BytesAlignmentTrans object.
		Return:
			a new NumpyAlignmentTrans object.
		''' 
		declare.is_alignment("other", other)

		results = super().__add__(other)
		return NumpyAlignmentTrans(results.data, results.name)

	def subset(self, nHead=0, nTail=0, nRandom=0, chunks=1, uttIDs=None):
		'''
		Subset data.
		The priority of mode is nHead > nTail > nRandom > chunks > uttIDs.
		If you chose mutiple modes, only the prior one will work.
		
		Args:
			<nHead>: get N head utterances.
			<nTail>: get N tail utterances.
			<nRandom>: sample N utterances randomly.
			<chunks>: split data into N chunks averagely.
			<uttIDs>: pick out these utterances whose ID in uttIDs.
		Return:
			a new NumpyAlignmentTrans object or a list of new NumpyAlignmentTrans objects.
		''' 
		result = super().subset(nHead, nTail, nRandom, chunks, uttIDs)

		if isinstance(result, list):
			for index in range(len(result)):
				temp = result[index]
				result[index] = NumpyAlignmentTrans(temp.data, temp.name)
		else:
			result = NumpyAlignmentTrans(result.data,result.name)

		return result
	
	def sort(self, by='utt', reverse=False):
		'''
		Sort utterances by frames length or uttID

		Args:
			<by>: "frame" or "utt"
			<reverse>: If reverse, sort in descending order.
		Return:
			A new NumpyAlignmentTrans object.
		''' 			
		result = super().sort(by,reverse)

		return NumpyAlignmentTrans(result.data, result.name)

	def to_phoneID(self, hmm):
		'''
		Transform tansition ID alignment to phone ID format.

		Args:
			<hmm>: exkaldi HMM object or file path.
		Return:
			a NumpyAlignmentPhone object.
		'''		
		if self.is_void:
			return NumpyAlignmentPhone(result, name=f"to_phone({self.name})")
		declare.kaldi_existed()
		declare.is_potential_hmm("hmm", hmm)

		temp = []
		for utt,matrix in self.data.items():
			new = utt + " ".join(map(str,matrix.tolist()))
			temp.append( new )
		temp = ("\n".join(temp)).encode()

		with FileHandleManager() as fhm:
			
			if not isinstance(hmm, str):
				hmmTemp = fhm.create("wb+", suffix=".hmm")
				hmm.save(hmmTemp)
				hmm = hmmTemp.name

			cmd = f'copy-int-vector ark:- ark:- | ali-to-phones --per-frame=true {hmm} ark:- ark,t:-'
			out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=temp)
			if (isinstance(cod, int) and cod != 0) or out == b'':
				print(err.decode())
				raise KaldiProcessError('Failed to transform alignment to phone.')
			else:
				result = {}
				sp = BytesIO(out)
				for line in sp.readlines():
					line = line.decode()
					line = line.strip().split()
					utt = line[0]
					matrix = np.array(line[1:], dtype=np.int32)
					result[utt] = matrix
				return NumpyAlignmentPhone(result, name=f"to_phone({self.name})")

	def to_pdfID(self, hmm):
		'''
		Transform tansition ID alignment to pdf ID format.

		Args:
			<hmm>: exkaldi HMM object or file path.
		Return:
			a NumpyAlignmentPhone object.
		'''		
		if self.is_void:
			return NumpyAlignmentPdf(result, name=f"to_pdf({self.name})")	
		declare.kaldi_existed()
		declare.is_potential_hmm("hmm", hmm)	

		temp = []
		for utt,matrix in self.data.items():
			new = utt + " ".join(map(str,matrix.tolist()))
			temp.append( new )
		temp = ("\n".join(temp)).encode()

		with FileHandleManager() as fhm:
			
			if not isinstance(hmm, str):
				hmmTemp = fhm.create("wb+", suffix=".hmm")
				hmm.save(hmmTemp)
				hmm = hmmTemp.name

			cmd = f'copy-int-vector ark:- ark:- | ali-to-phones --per-frame=true {hmm} ark:- ark,t:-'
			out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=temp)
			if (isinstance(cod, int) and cod != 0) or out == b'':
				print(err.decode())
				raise KaldiProcessError('Failed to transform alignment to pdf.')
			else:
				result = {}
				sp = BytesIO(out)
				for line in sp.readlines():
					line = line.decode()
					line = line.strip().split()
					utt = line[0]
					matrix = np.array(line[1:], dtype=np.int32)
					result[utt] = matrix
				return NumpyAlignmentPhone(result, name=f"to_pdf({self.name})")

	def __call__(self, uttID):
		'''
		Pick out the specified utterance.
		
		Args:
			<uttID>: a string.
		Return:
			If existed, return a new NumpyAlignmentTrans object.
		''' 
		result = super().__call__(uttID)
		if result is not None:
			result = NumpyAlignmentTrans(data=result.data, name=result.name)
		return result	

	def map(self, func):
		'''
		Map all arrays to a function.

		Args:
			<func>: callable function object.
		
		Return:
			A new NumpyAlignmentTrans object.
		'''
		result = super().map(func)
		return NumpyAlignmentTrans(data=result.data, name=result.name)	

	def cut(self, maxFrames):
		'''
		Cut long utterance to mutiple shorter ones. 

		Args:
			<maxFrames>: a positive int value. Cut a utterance by the thresold value of 1.25*maxFrames into a maxFrames part and a rest part.
						If the rest part is still longer than 1.25*maxFrames, continue to cut it. 
		Return:
			A new NumpyAlignmentTrans object.
		''' 
		declare.not_void(type_name(self), self)	
		declare.is_positive_int("maxFrames", maxFrames)

		newData = {}
		cutThreshold = maxFrames + maxFrames//4

		for utt,vector in self.items():
			if len(vector) <= cutThreshold:
				newData[utt] = vector
			else:
				i = 0 
				while True:
					newData[utt+"_"+str(i)] = vector[i*maxFrames:(i+1)*maxFrames]
					i += 1
					if len(vector[i*maxFrames:]) <= cutThreshold:
						break
				if len(vector[i*maxFrames:]) != 0:
					newData[utt+"_"+str(i)] = vector[i*maxFrames:]
		
		newName = f"cut({self.name},{maxFrames})"
		return NumpyAlignmentTrans(newData, newName)

## Subclass: for user customized alignment 
class NumpyAlignment(NumpyVector):
	'''
	Hold the alignment with Numpy format.
	'''
	def __init__(self, data={}, name="ali"):
		'''
		Args:
			<data>: NumpyAlignment or dict object (or their subclasses).
		'''
		declare.belong_classes("data", data, [NumpyAlignment, dict])

		if isinstance(data, NumpyAlignment):
			data = data.data
				
		super().__init__(data, name)
	
	def to_dtype(self, dtype):
		'''
		Transform data type.

		Args:
			<dtype>: a string of "int", "int32" or "int64". IF "int", it will be treated as "int32".
		Return:
			A new NumpyAlignment object.
		'''
		result = super().to_dtype(dtype)

		return NumpyAlignment(result.data, result.name)

	def to_bytes(self):
		
		raise WrongOperation("Cannot convert this alignment to bytes.")
	
	def __add__(self, other):
		'''
		Plus operation between two alignment objects.

		Args:
			<other>: a NumpyAlignment object.
		Return:
			a NumpyAlignment object.
		'''	
		declare.belong_classes("other", other, NumpyAlignment)
		
		result = super().__add__(other)
		return NumpyAlignment(result.data, result.name)

	def subset(self, nHead=0, nTail=0, nRandom=0, chunks=1, uttIDs=None):
		'''
		Subset data.
		The priority of mode is nHead > nTail > nRandom > chunks > uttIDs.
		If you chose mutiple modes, only the prior one will work.
		
		Args:
			<nHead>: get N head utterances.
			<nTail>: get N tail utterances.
			<nRandom>: sample N utterances randomly.
			<chunks>: split data into N chunks averagely.
			<uttIDs>: pick out these utterances whose ID in uttIDs.
		Return:
			a new NumpyAlignment object or a list of new NumpyAlignment objects.
		''' 
		result = super().subset(nHead, nTail, nRandom, chunks, uttIDs)

		if isinstance(result, list):
			for index in range(len(result)):
				temp = result[index]
				result[index] = NumpyAlignment( temp.data, temp.name )
		else:
			result = NumpyAlignment( result.data,result.name )

		return result

	def sort(self, by='utt', reverse=False):
		'''
		Sort utterances by frames length or uttID

		Args:
			<by>: "frame" or "utt"
			<reverse>: If reverse, sort in descending order.
		Return:
			A new NumpyAlignment object.
		''' 			
		result = super().sort(by,reverse)

		return NumpyAlignment(result.data, result.name)

	def __call__(self, uttID):
		'''
		Pick out the specified utterance.
		
		Args:
			<uttID>: a string.
		Return:
			If existed, return a new NumpyAlignment object.
			Or return None.
		''' 
		result = super().__call__(uttID)
		if result is not None:
			result = NumpyAlignment(data=result.data, name=result.name)
		return result	

	def map(self, func):
		'''
		Map all arrays to a function.

		Args:
			<func>: callable function object.
		
		Return:
			A new NumpyAlignment object.
		'''
		result = super().map(func)
		return NumpyAlignment(data=result.data, name=result.name)	

	def cut(self, maxFrames):
		'''
		Cut long utterance to mutiple shorter ones. 

		Args:
			<maxFrames>: a positive int value. Cut a utterance by the thresold value of 1.25*maxFrames into a maxFrames part and a rest part.
						If the rest part is still longer than 1.25*maxFrames, continue to cut it. 
		Return:
			A new NumpyAlignment object.
		''' 
		declare.not_void(type_name(self), self)	
		declare.is_positive_int("maxFrames", maxFrames)

		newData = {}
		cutThreshold = maxFrames + maxFrames//4

		for utt,vector in self.items():
			if len(vector) <= cutThreshold:
				newData[utt] = vector
			else:
				i = 0 
				while True:
					newData[utt+"_"+str(i)] = vector[i*maxFrames:(i+1)*maxFrames]
					i += 1
					if len(vector[i*maxFrames:]) <= cutThreshold:
						break
				if len(vector[i*maxFrames:]) != 0:
					newData[utt+"_"+str(i)] = vector[i*maxFrames:]
		
		newName = f"cut({self.name},{maxFrames})"
		return NumpyAlignment(newData, newName)

## Subclass: for phone-ID alignment 
class NumpyAlignmentPhone(NumpyAlignment):
	'''
	Hold the alignment(phone ID) with Numpy format.
	'''
	def __init__(self, data={}, name="phoneID"):
		'''
		Only allow NumpyAlignmentPhone or dict (do not extend to their subclasses and their parent-classes).
		'''			
		declare.is_classes("data", data, [NumpyAlignmentPhone, dict])

		if isinstance(data, NumpyAlignmentPhone):
			data = data.data
				
		super().__init__(data, name)		

	def to_dtype(self, dtype):
		'''
		Transform data type.

		Args:
			<dtype>: a string of "int", "int32" or "int64". IF "int", it will be treated as "int32".
		Return:
			A new NumpyAlignmentPhone object.
		'''
		result = super().to_dtype(dtype)

		return NumpyAlignmentPhone(result.data, result.name)

	def __add__(self, other):
		'''
		The Plus operation between two phone ID alignment objects.

		Args:
			<other>: a NumpyAlignmentPhone object.
		Return:
			a new NumpyAlignmentPhone object.
		''' 
		declare.is_classes("other", other, NumpyAlignmentPhone)

		results = super().__add__(other)
		return NumpyAlignmentPhone(results.data, results.name)

	def subset(self, nHead=0, nTail=0, nRandom=0, chunks=1, uttIDs=None):
		'''
		Subset data.
		The priority of mode is nHead > nTail > nRandom > chunks > uttIDs.
		If you chose mutiple modes, only the prior one will work.
		
		Args:
			<nHead>: get N head utterances.
			<nTail>: get N tail utterances.
			<nRandom>: sample N utterances randomly.
			<chunks>: split data into N chunks averagely.
			<uttIDs>: pick out these utterances whose ID in uttIDs.
		Return:
			a new NumpyAlignmentPhone object or a list of new NumpyAlignmentPhone objects.
		''' 
		result = super().subset(nHead, nTail, nRandom, chunks, uttIDs)

		if isinstance(result, list):
			for index in range(len(result)):
				temp = result[index]
				result[index] = NumpyAlignmentPhone(temp.data, temp.name)
		else:
			result = NumpyAlignmentPhone(result.data,result.name)

		return result

	def __call__(self, uttID):
		'''
		Pick out the specified utterance.
		
		Args:
			<uttID>: a string.
		Return:
			If existed, return a new NumpyAlignmentPhone object.
		''' 
		result = super().__call__(uttID)
		if result is not None:
			result = NumpyAlignmentPhone(data=result.data, name=result.name)
		return result	

	def sort(self, by='utt', reverse=False):
		'''
		Sort utterances by frames length or uttID

		Args:
			<by>: "frame" or "utt"
			<reverse>: If reverse, sort in descending order.
		Return:
			A new NumpyAlignmentPhone object.
		''' 			
		result = super().sort(by, reverse)

		return NumpyAlignmentPhone(result.data, result.name)

	def map(self, func):
		'''
		Map all arrays to a function.

		Args:
			<func>: callable function object.
		
		Return:
			A new NumpyAlignmentPhone object.
		'''
		result = super().map(func)
		return NumpyAlignmentPhone(data=result.data, name=result.name)

	def cut(self, maxFrames):
		'''
		Cut long utterance to mutiple shorter ones. 

		Args:
			<maxFrames>: a positive int value. Cut a utterance by the thresold value of 1.25*maxFrames into a maxFrames part and a rest part.
						If the rest part is still longer than 1.25*maxFrames, continue to cut it. 
		Return:
			A new NumpyAlignmentPhone object.
		''' 
		result = super().cut(maxFrames)
		return NumpyAlignmentPhone(result.data, result.name)

## Subclass: for pdf-ID alignment 
class NumpyAlignmentPdf(NumpyAlignment):
	'''
	Hold the alignment(pdf ID) with Numpy format.
	'''
	def __init__(self, data={}, name="phoneID"):
		'''
		Only allow NumpyAlignmentPdf or dict (do not extend to their subclasses and their parent-classes).
		'''	
		declare.is_classes("data", data, [NumpyAlignmentPdf, dict])

		if isinstance(data, NumpyAlignmentPdf):
			data = data.data
				
		super().__init__(data, name)

	def to_dtype(self, dtype):
		'''
		Transform data type.

		Args:
			<dtype>: a string of "int", "int32" or "int64". IF "int", it will be treated as "int32".
		Return:
			A new NumpyAlignmentPdf object.
		'''
		result = super().to_dtype(dtype)

		return NumpyAlignmentPdf(result.data, result.name)

	def __add__(self, other):
		'''
		The Plus operation between two phone ID alignment objects.

		Args:
			<other>: a NumpyAlignmentPdf object.
		Return:
			a new NumpyAlignmentPdf object.
		''' 
		declare.is_classes("other", other, NumpyAlignmentPdf)

		results = super().__add__(other)
		return NumpyAlignmentPdf(results.data, results.name)

	def subset(self, nHead=0, nTail=0, nRandom=0, chunks=1, uttIDs=None):
		'''
		Subset data.
		The priority of mode is nHead > nTail > nRandom > chunks > uttIDs.
		If you chose mutiple modes, only the prior one will work.
		
		Args:
			<nHead>: get N head utterances.
			<nTail>: get N tail utterances.
			<nRandom>: sample N utterances randomly.
			<chunks>: split data into N chunks averagely.
			<uttIDs>: pick out these utterances whose ID in uttIDs.
		Return:
			a new NumpyAlignmentPdf object or a list of new NumpyAlignmentPdf objects.
		''' 
		result = super().subset(nHead, nTail, nRandom, chunks, uttIDs)

		if isinstance(result, list):
			for index in range(len(result)):
				temp = result[index]
				result[index] = NumpyAlignmentPdf(temp.data, temp.name)
		else:
			result = NumpyAlignmentPdf(result.data,result.name)

		return result

	def __call__(self, uttID):
		'''
		Pick out the specified utterance.
		
		Args:
			<uttID>: a string.
		Return:
			If existed, return a new NumpyAlignmentPdf object.
			Or return None.
		''' 
		result = super().__call__(uttID)
		if result is not None:
			result = NumpyAlignmentPdf(data=result.data, name=result.name)
		return result	

	def sort(self, by='frame', reverse=False):
		'''
		Sort utterances by frames length or uttID

		Args:
			<by>: "frame" or "utt"
			<reverse>: If reverse, sort in descending order.
		Return:
			A new NumpyAlignmentPdf object.
		''' 			
		result = super().sort(by,reverse)

		return NumpyAlignmentPdf(result.data, result.name)

	def map(self, func):
		'''
		Map all arrays to a function.

		Args:
			<func>: callable function object.
		
		Return:
			A new NumpyAlignmentPdf object.
		'''
		result = super().map(func)
		return NumpyAlignmentPdf(data=result.data, name=result.name)
	
	def cut(self, maxFrames):
		'''
		Cut long utterance to mutiple shorter ones. 

		Args:
			<maxFrames>: a positive int value. Cut a utterance by the thresold value of 1.25*maxFrames into a maxFrames part and a rest part.
						If the rest part is still longer than 1.25*maxFrames, continue to cut it. 
		Return:
			A new NumpyAlignmentPdf object.
		''' 
		result = super().cut(maxFrames)
		return NumpyAlignmentPdf(result.data, result.name)