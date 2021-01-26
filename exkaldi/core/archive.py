# coding=utf-8
#
# Yu Wang (University of Yamanashi)
# May,2020
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

import copy
from io import BytesIO
import numpy as np
import random
import struct
import os
from collections import namedtuple
import sys

from exkaldi.error import *
from exkaldi.utils.utils import type_name,run_shell_command,make_dependent_dirs,list_files
from exkaldi.utils.utils import FileHandleManager
from exkaldi.utils import declare

''' ListTable class group''' 

class ListTable(dict):
	'''
	This is a subclass of Python dict.
	You can use it to hold Kaldi text format tables,such as scp-files,utt2spk and so on. 
	'''
	def __init__(self,data={},name="table"):
		super().__init__(data)
		declare.is_valid_string("name",name)
		self.__name = name
		
	@property
	def is_void(self):
		'''
		Check whether this is a void object.

		Return:
			True or False.
		'''
		return len(self.keys()) == 0

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

	def rename(self,name):
		'''
		Rename.

		Args:
			<name>: a string.
		'''
		declare.is_valid_string("name",name)
		self.__name = name

	def reset_data(self,data=None):
		'''
		Reset the data.

		Args:
			<data>: an iterable object that can be converted to dict object.
		'''
		if data is None:
			self.clear()
		else:
			newData = dict(data) #check it
			self.clear()
			self.update(newData)

	def record(self,key,value):
		'''
		Enter a record. This is an universal API to add an item.

		Args:	
			_key_,_value_: python objects.
		'''
		super().__setitem__(key,value)

	def sort(self,by="key",reverse=False):
		'''
		Sort by key or value. 
		This is just a pseudo sort operation for the dict object but it works after python 3.6.

		Args:
			<by>: "key" or "value".
			<reverse>: If reverse, sort in descending order.
		
		Return:
			A new ListTable object.
		''' 
		declare.is_instances("by",by,["key","value"])

		if by == "key":
			items = sorted(self.items(),key=lambda x:x[0],reverse=reverse)
		else:
			items = sorted(self.items(),key=lambda x:x[1],reverse=reverse)

		newName = f"sort({self.name})"
		return ListTable(items,name=newName)

	def save(self,fileName=None,chunks=1,concatFunc=None):
		'''
		Save to file.

		Args:
			<fileName>: file name,opened file handle or None.  
			<chunks>: an int value. If > 1,split the table into N chunks and save them respectively. Work only when save to file.  
			<concatFunc>: a callable object or function to decide how to concatenate the key and value. 
										Depending on tasks,you can use a special function to concatenate key and value to be a string. 
										If None,defaultly: key + space + value.
		
		Return:
			file name,file handle or a string of contents.
		'''
		declare.greater_equal("chunks",chunks,None,1)
		if fileName:
			declare.is_valid_file_name_or_handle("file name",fileName)

		def purely_concat(item):
			try:
				return f"{item[0]} {item[1]}"
			except Exception:
				raise WrongDataFormat(f"Cannot convert to string and concatenate: {type_name(item[0])} and {type_name(item[1])}. ")
		
		def save_chunk_data(chunkData,concatFunc,fileName):
			contents = "\n".join(map(concatFunc,chunkData.items())) + "\n"
			if fileName is None:
				return contents
			else:
				make_dependent_dirs(fileName,pathIsFile=True)
				with open(fileName,"w",encoding="utf-8") as fw:
					fw.write(contents)
				return fileName

		if concatFunc:
			declare.is_callable("concat function",concatFunc)
		else:
			concatFunc = purely_concat

		if fileName is None:
			return save_chunk_data(self,concatFunc,None)

		elif isinstance(fileName,str):

			if chunks == 1:
				return save_chunk_data(self,concatFunc,fileName)

			else:
				dirName = os.path.dirname(fileName)
				fileName = os.path.basename(fileName)

				chunkDataList = self.subset(chunks=chunks)

				savedFiles = []
				newNamePattern = f"ck%0{len(str(chunks))}d_{fileName}"
				for i,chunkData in enumerate(chunkDataList):
					chunkFileName = os.path.join( dirName,newNamePattern%i )
					savedFiles.append( save_chunk_data(chunkData,concatFunc,chunkFileName) )
				return savedFiles		
				
		else:
			results = save_chunk_data(self,concatFunc,None)
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
		return ListTable(data=items,name=newName)
	
	def subset(self,nHead=0,nTail=0,nRandom=0,chunks=1,keys=None):
		'''
		Subset.
		Only one mode will do when it is not the default value. 
		The priority order is: nHead > nTail > nRandom > chunks > keys.
		
		Args:
			<nHead>: If it > 0,extract N head utterances.
			<nTail>: If it > 0,extract N tail utterances.
			<nRandom>: If it > 0,randomly sample N utterances.
			<chunks>: If it > 1,split data into N chunks.
			<keys>: If it is not None,pick out these utterances whose ID in keys.
		
		Return:
			a new ListTable object or a list of new ListTable objects.
		''' 
		declare.not_void(type_name(self),self)

		if nHead > 0:
			declare.is_positive_int("nHead",nHead)
			new = list(self.items())[0:nHead]
			newName = f"subset({self.name},head {nHead})"
			return ListTable(data=new,name=newName)
		
		elif nTail > 0:
			declare.is_positive_int("nTail",nTail)
			new = list(self.items())[-nTail:]
			newName = f"subset({self.name},tail {nTail})"
			return ListTable(data=new,name=newName)		

		elif nRandom > 0:
			declare.is_positive_int("nRandom",nRandom)
			if nRandom > self.lens:
				new = self
				nRandom = self.lens
			else:
				new = random.sample(list(self.items()),k=nRandom)
			newName = f"subset({self.name},random {nRandom})"
			return ListTable(data=new,name=newName)	

		elif chunks > 1:
			declare.is_positive_int("chunks",chunks)
			
			datas = []
			allLens = len(self.keys())
			if allLens != 0:
				chunkLens = allLens//chunks
				if chunkLens == 0:
					chunks = allLens
					chunkLens = 1
					t = 0
				else:
					t = allLens - chunkLens * chunks

				items = list(self.items())
				start = 0
				for i in range(chunks):
					if i < t:
						end = start + chunkLens + 1
					else:
						end = start + chunkLens
					chunkItems = items[start:end]
					newName = f"subset({self.name},chunk {chunks}-{i})"
					datas.append( ListTable(data=chunkItems,name=newName) )
					start = end

			return datas

		elif keys != None:
			declare.is_classes("keys",keys,(list,tuple))
			newName = f"subset({self.name},keys {len(keys)})"

			newDict = {}
			for k in keys:
				if k in self.keys():
					newDict[k] = self[k]

			return ListTable(data=newDict,name=newName)
		
		else:
			raise WrongOperation("When subset, at least one mode should be specified but all got default values.")

	def __add__(self,other):
		'''
		Integrate two ListTable objects. If key existed in both two objects, the former will be retained.

		Args:
			<other>: another ListTable object.
		
		Return:
			A new ListTable object.
		'''
		declare.belong_classes("other",other,ListTable)

		new = copy.deepcopy(self)
		for k in other.keys():
			if k not in new.keys():
				new[k] = other[k]
		newName = f"plus({self.name},{other.name})"
		new.rename(newName)
		return new

	def reverse(self):
		'''
		Exchange the position of key and value. 
		Key and value must be one-one matching, or Error will be raised.

		Return:
			a new ListTable object.
		'''
		newname = f"reverse({self.name})"
		new = ListTable(name=newname)
		for key,value in self.items():
			if value in new.keys():
				raise WrongDataFormat(f"Only one-one matching table can be reversed but multiple {value} have existed.")
			else:
				new[value] = key
				
		return new

	def key_existed(self,key):
		'''
		Query whether this key has existed. If existed, return True.

		Args:
			<key>: a key.

		Return:
			A bool value.
		'''
		return key in self.keys()

	@property
	def lens(self):
		'''
		Get the numbers os records.

		Return:
			an int value.
		'''
		return len(self)

class Transcription(ListTable):
	'''
	This is used to hold transcription text,such as decoding n-best. 
	'''
	def __init__(self,data={},name="transcription"):
		declare.is_classes("data",data,[dict,Transcription])
		super(Transcription,self).__init__(name=name)
		for key,value in data.items():
			self.record(key,value)

	def __setitem__(self,key,value):
		'''Overlap this method to avoid the wrong assignment.'''
		self.record(key,value)

	def setdefault(self,key,value=None):
		'''Overlap this method to avoid the wrong assignment.'''
		if self.key_existed(key):
			return self[key]
		else:
			self.record(key,value)
	
	def update(self,other):
		'''
		Args:
			<other>: exkaldi Transcription object.
		'''
		declare.is_transcription("other",other)

		super().update(other)

	def record(self,key,value):
		'''
		Add or modify a record:
		
		Args:
			<key>: a string. The utterance ID.
			<value>: a string. The spoken transcription.
		'''
		declare.is_valid_string("key",key)
		declare.is_valid_string("value",value)

		super().__setitem__(key,value)

	@property
	def utts(self):
		'''
		Get a list of all utterance IDs.
		'''
		return list(self.keys())

	def sort(self,by="utt",reverse=False):
		'''
		Sort.

		Args:
			<by>: "key"/"utt",or "value"/"sentence",or "sentenceLength".
			<reverse>: If reverse,sort in descending order.

		Return:
			A new Transcription object.
		''' 
		declare.is_instances("by",by,["utt","key","value","sentence","sentenceLength"])

		if by in ["utt","key"]:
			items = super().sort(by="key",reverse=reverse)
		elif by in ["value","sentence"]:
			items = super().sort(by="value",reverse=reverse)
		else:
			items = sorted(self.items(),key=lambda x:x[1].count(" ")+len(x[1]),reverse=reverse)

		items = dict(items)
		return Transcription(data=items,name=self.name)

	def __add__(self,other):
		'''
		Integrate two transcription objects. If key existed in both two objects,the former will be retained.

		Args:
			<other>: another Transcription object.

		Return:
			A new Transcription object.
		'''
		declare.is_classes("other",other,Transcription)

		result = super().__add__(other)
		return Transcription(data=result.data,name=result.name)

	def shuffle(self):
		'''
		Random shuffle the transcription.

		Return:
			A new Transcription object.
		'''
		result = super().shuffle()

		return Transcription(data=result.data,name=self.name)

	def subset(self,nHead=0,nTail=0,nRandom=0,chunks=1,keys=None):
		'''
		Subset.
		Only one mode will work when it is not the default value. 
		The priority order is: nHead > nTail > nRandom > chunks > keys.
		
		Args:
			<nHead>: If it > 0,extract N head utterances.
			<nTail>: If it > 0,extract N tail utterances.
			<nRandom>: If it > 0,randomly sample N utterances.
			<chunks>: If it > 1,split data into N chunks.
			<keys>: If it is not None,pick out these utterances whose ID in keys.
		Return:
			a new Transcription object or a list of new Transcription objects.
		''' 
		result = super().subset(nHead,nTail,nRandom,chunks,keys)

		if isinstance(result,list):
			for index in range(len(result)):
				temp = result[index]
				result[index] = Transcription(data=temp.data,name=temp.name)
		else:
			result = Transcription(data=result.data,name=result.name)

		return result
	
	def convert(self,symbolTable,unkSymbol=None,ignore=None):
		'''
		Convert transcription between two types of symbol, typically text format and int format.

		Args:
			<symbolTable>: exkaldi ListTable object.
			<unkSymbol>: symbol of oov. If symbol is out of table,use this to replace.
			<ignore>: ignore some specified symbol if necessary.

		Return:
			A new Transcription object.
		'''
		declare.not_void(type_name(self),self)
		declare.is_classes("symbolTable",symbolTable,ListTable)
		if ignore is not None:
			declare.is_classes("ignore",ignore,[str,list,tuple])
			if not isinstance(ignore,str):
				declare.members_are_valid_strings(ignore)
			else:
				ignore = [ignore,]
		else:
			ignore = []
		
		symbolTable = dict( (str(k),str(v)) for k,v in symbolTable.items() )
		unkSymbol = str(unkSymbol)

		newTrans = Transcription(name=f"convert({self.name})")

		for uttID,text in self.items():
			declare.is_valid_string("transcription",text)
			text = text.split()
			for index,word in enumerate(text):
				if word in ignore:
					continue
				try:
					text[index] = str(symbolTable[word])
				except KeyError:
					if unkSymbol is None:
						raise WrongDataFormat(f"Missed the corresponding target for symbol: {word}. You can specified the <unkSymbol> to replace it.")
					else:
						try:
							text[index] = str(symbolTable[unkSymbol])
						except KeyError:
							raise WrongDataFormat(f"Word symbol table has not the unknown symbol: {unkSymbol}")
		
			newTrans.record(uttID, " ".join(text))
	
		return newTrans

	def sentence_length(self):
		'''
		Count the length of each sentence ( It will count the numbers of inner space ).

		Return:
			a Metric object.
		'''
		result = Metric(name=f"sentence_length({self.name})")
		for uttID,txt in self.items():
			declare.is_valid_string("transcription",txt)
			result.record(uttID, txt.strip().count(" ")+1)
		return result

	def save(self,fileName=None,chunks=1,discardUttID=False):
		'''
		Save as text file.

		Args:
			<fileName>: None,file name or file handle.
			<chunks>: an int value. If greater than 1,split it and save them into N files.
			<discardUttID>: If True,discard the info of utterance IDs.
		
		Return:
			file name,file handle or the contents of ListTable.
		'''
		declare.is_bool("discardUttID",discardUttID)

		def concat(item,discardUtt):
			try:
				if discardUtt:
					return f"{item[1]}"
				else:
					return f"{item[0]} {item[1]}"
			except Exception:
				raise WrongDataFormat(f"Cannot cancatenate these key and value: {type_name(item[0])} and {type_name(item[1])}. ")
		
		return super().save(fileName=fileName,chunks=chunks,concatFunc=lambda x:concat(x,discardUttID) )

	def count_word(self):
		'''
		Count the number of words.

		Return:
			a Metric object.
		'''
		result = Metric(name=f"count_word({self.name})")
		for uttID,txt in self.items():
			declare.is_valid_string("transcription",txt)
			txt = txt.strip().split()
			for w in txt:
				try:
					_ = result[w]
				except KeyError:
					result.record(w, 1)
				else:
					result.record(w, result[w]+1)

		return result

class Metric(ListTable):
	'''
	This is used to hold the Metrics,such as AM or LM scores. 
	The data format in Metric is: { utterance ID : int or float score, }
	'''
	def __init__(self,data={},name="metric"):
		declare.is_classes("data",data,[dict,Metric])
		super().__init__(name=name)
		for key,value in data.items():
			self.record(key,value)		

	def __setitem__(self,key,value):
		'''Overlap this method to avoid the wrong assignment.'''
		self.record(key,value)

	def setdefault(self,key,value=None):
		'''Overlap this method to avoid the wrong assignment.'''
		if self.key_existed(key):
			return self[key]
		else:
			self.record(key,value)

	def update(self,other):
		'''
		Args:
			<other>: exkaldi Metric object.
		'''
		declare.is_classes("other",other,Metric)

		super().update(other)

	def record(self,key,value):
		'''
		Add or modify a record:
		
		Args:
			<key>: a string.
			<value>: a float or int value. The score.
		'''
		declare.is_valid_string("key",key)
		declare.is_classes("value",value,[int,float])

		super().__setitem__(key,value)

	def sort(self,by="key",reverse=False):
		'''
		Sort.

		Args:
			<by>: "key" or "score"/"value".
			<reverse>: If reverse,sort in descending order.

		Return:
			A new Metric object.
		''' 
		declare.is_instances("by",by,["key","value","score"])

		def filtering(x,i):
			declare.is_valid_string("key",x[0])
			declare.is_classes("score",x[1],(int,float))
			return x[i]

		if by == "key":
			items = sorted(self.items(),key=lambda x:filtering(x,0),reverse=reverse)
		else:
			items = sorted(self.items(),key=lambda x:filtering(x,1),reverse=reverse)
		items = dict(items)
		newName = f"sort({self.name},{by})"
		return Metric(items,name=newName)
	
	def __add__(self,other):
		'''
		Integrate two Metric objects. If utt is existed in both two objects,the former will be retained.

		Args:
			<other>: another Metric object.

		Return:
			A new Metric object.
		'''
		declare.is_classes("other",other,Metric)

		result = super().__add__(other)
		return Metric(result.data,name=result.name)

	def shuffle(self):
		'''
		Random shuffle the Metric table.

		Return:
			A new Metric object.
		'''
		result = super().shuffle()

		return Metric(result.data,name=self.name)

	def subset(self,nHead=0,nTail=0,nRandom=0,chunks=1,keys=None):
		'''
		Subset feature.
		
		Args:
			<nHead>: If it > 0,extract N head utterances.
			<nTail>: If nHead=0 and nTail > 0,extract N tail utterances.
			<nRandom>: If nHead=0 and nTail=0 and nRandom > 0,randomly sample N utterances.
			<chunks>: If all of nHead,nTail,nRandom are 0 and chunks > 1,split data into N chunks.
			<keys>: If nHead == 0 and chunks == 1 and keys != None,pick out these utterances whose ID in keys.
		
		Return:
			a new Metric object or a list of new Metric objects.
		'''
		result = super().subset(nHead,nTail,nRandom,chunks,keys)

		if isinstance(result,list):
			for index in range(len(result)):
				temp = result[index]
				result[index] = Metric(temp.data,temp.name)
		else:
			result = Metric(result.data,result.name)

		return result

	def sum(self,weight=None):
		'''
		The weighted sum of all scores.

		Args:
			_weight_: a dict object or Metric object. 

		Return:
			A float value.
		'''
		if self.is_void:
			return 0.0

		if weight is None:
			return sum(self.values())
		else:
			declare.is_classes("weight",weight,["dict","Metric"])

			totalSum = 0
			for key,value in self.items():
				if key not in weight.keys():
					raise WrongOperation(f"Miss weight for: {key}.")
				else:
					W = weight[key]
					declare.is_classes("weight",W,[int,float])
					totalSum += W*value
			
			return totalSum

	def mean(self,weight=None,epsilon=1e-8):
		'''
		The weighted average of all score.

		Args:
			<weigts>: the weight of each utterance.
		
		Return:
			a float value.
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
				if key not in weight.keys():
					raise WrongOperation(f"Miss weight for: {key}.")
				else:
					W = weight[key]
					declare.is_classes("weight",W,[int,float])
					numerator += W*value
					denominator += W

			return numerator/denominator

	def max(self):
		'''
		The maximum value.

		Return:
			a int or float value.
		'''
		return max(self.values())
	
	def argmax(self):
		'''
		Get the uttID of the max score.

		Return:
			a key value.
		'''
		return max(self.items(),key=lambda x:x[1])[0]

	def min(self):
		'''
		The minimum value.
		'''
		return min(self.values())
	
	def argmin(self):
		'''
		Get the uttID of the min score.
		'''
		return min(self.items(),key=lambda x:x[1])[0]

	def map(self,func):
		'''
		Map a function to all scores.

		Args:
			<func>: a callable function or object.
		
		Return:
			A new Metric object.
		'''
		declare.is_callable(func)

		new = dict(map(lambda x:(x[0],func(x[1])),self.items()))

		return Metric(new,name=f"mapped({self.name})")

class IndexTable(ListTable):
	'''
	For accelerate to find utterance and reduce memory cost of intermidiate operation.
	This is used to hold the utterance index informat of Kaldi archive table (binary format). It just like the script-table file but is more useful.
	Its format like this:
	{ "utt0": namedtuple(frames=100,startIndex=1000,dataSize=10000,filePath="./feat.ark") }
	'''
	def __init__(self,data={},name="indexTable"):
		declare.is_classes("data",data,[dict,IndexTable])
		super().__init__(name=name)
		for key,value in data.items():
			self[key] = value

	def __setitem__(self,key,value):
		'''Overlap this method to avoid the wrong assignment.'''
		if isinstance(value,(list,tuple)):
			assert len(value) in [3,4],f"Expected (frames,start index,data size[,file path]) but {value} does not match."
			self.record(key,*value)
		elif isinstance(value,IndexInfo):
			super().__setitem__(key,value)
		else:
			raise UnsupportedType(f"The value of index table shou be list, tuple or IndexInfo object but got: {type_name(value)}.")

	def setdefault(self,key,value=None):
		'''Overlap this method to avoid the wrong assignment.'''
		if self.key_existed(key):
			return self[key]
		else:
			self.__setitem__(key,value)
	
	def update(self,other):
		'''
		Args:
			<other>: exkaldi IndexTable object.
		'''
		declare.is_classes("other",other,IndexTable)

		super().update(other)

	def record(self,key,frames=None,startIndex=None,dataSize=None,filePath=None):
		'''
		Add or modify a record.
		
		Args:
			<key>: a string. The utterance ID.
			<frames>: an int value.
			<startIndex>: an int value. The start index of an archive record. Including the size of utterance ID.
			<dataSize>: an int value. The total size of an archive record. Including the size of utterance ID.
			<filePath>: a string. The total size of an archive record.
		'''
		declare.is_valid_string("key",key)

		if self.key_existed(key):
			value = self[key]
			if frames is not None:
				declare.is_positive_int("frames", frames)
				value = value._replace(frames=frames)
			if startIndex is not None:
				declare.is_non_negative_int("startIndex", startIndex)
				value = value._replace(startIndex=startIndex)
			if dataSize is not None:
				declare.is_positive_int("dataSize", dataSize)
				value = value._replace(dataSize=dataSize)		
			if filePath is not None:
				declare.is_file("filePath", filePath)
				value = value._replace(filePath=filePath)

		else:
			declare.is_positive_int("frames",frames)
			declare.is_non_negative_int("startIndex",startIndex)
			declare.is_positive_int("dataSize",dataSize)
			if filePath is not None:
				declare.is_file("filePath",filePath)
			value = self.spec(frames,startIndex,dataSize,filePath)

		super().__setitem__(key,value)

	@property
	def spec(self):
		'''
		The index info spec.

		Return:
			a namedtuple class.
		'''
		spec = namedtuple("IndexInfo",["frames","startIndex","dataSize","filePath"])
		spec.__new__.__defaults__ = (None,)
		return spec

	def sort(self,by="utt",reverse=False):
		'''
		Sort utterances by frame length,utterance ID or start index or file path name.

		Args:
			<by>: "key"/"utt","value"/"frame" or "startIndex" or "filePath".
			<reverse>: If True,sort in descending order.
		
		Return:
			A new IndexTable object.
		''' 
		declare.is_instances("by",by,["utt","key","frame","value","startIndex","filePath"])

		if by in ["utt","key"]:
			items = sorted(self.items(),key=lambda x:x[0],reverse=reverse)
		elif by in ["frame","value"]:
			items = sorted(self.items(),key=lambda x:x[1].frames,reverse=reverse)
		elif by == "startIndex":
			items = sorted(self.items(),key=lambda x:x[1].startIndex,reverse=reverse)
		else:
			try:
				items = sorted(self.items(),key=lambda x:x[1].filePath,reverse=reverse)
			except TypeError:
				items = self.items()
		
		newName = f"sort({self.name},{by})"
		return IndexTable(dict(items),name=newName)

	def __add__(self,other):
		'''
		Integrate two IndexTable objects. If utterance has existed in both two objects,the former will be retained.

		Args:
			<other>: another IndexTable object.

		Return:
			A new IndexTable object.
		'''
		declare.is_classes("other",other,IndexTable)

		result = super().__add__(other)
		return IndexTable(result.data,name=result.name)

	def shuffle(self):
		'''
		Random shuffle the index table.

		Return:
			A new IndexTable object.
		'''
		result = super().shuffle()

		return IndexTable(result.data,name=self.name)
	
	def subset(self,nHead=0,nTail=0,nRandom=0,chunks=1,keys=None):
		'''
		Subset.
		Only one mode will work when it is not the default value. 
		The priority order is: nHead > nTail > nRandom > chunks > keys.
		
		Args:
			<nHead>: If it > 0,extract N head utterances.
			<nTail>: If it > 0,extract N tail utterances.
			<nRandom>: If it > 0,randomly sample N utterances.
			<chunks>: If it > 1,split data into N chunks.
			<keys>: If it is not None,pick out these utterances whose ID in keys.

		Return:
			a new IndexTable object or a list of new IndexTable objects.
		''' 
		result = super().subset(nHead,nTail,nRandom,chunks,keys)

		if isinstance(result,list):
			for index in range(len(result)):
				temp = result[index]
				result[index] = IndexTable(temp.data,temp.name)
		else:
			result = IndexTable(result.data,result.name)

		return result

	def save(self,fileName=None,chunks=1):
		'''
		Save this index informat to text file with kaidi script-file table format.
		Note that the frames informat will be discarded.

		Args:
			<fileName>: file name or file handle.  
			<chunks>: an int value. If > 1,split it into N chunks and save them.  
								This option only work when _fileName_ is a file name.  
		
		Return:
			file name,file handle or a string.
		'''
		declare.not_void(type_name(self),self)

		def concat(item):
			utt,indexInfo = item
			if indexInfo.filePath is None:
				raise WrongOperation("Cannot save to script file becase miss the file path info.")
			else:
				startIndex = indexInfo.startIndex + len(utt) + 1
				return f"{utt} {indexInfo.filePath}:{startIndex}"

		return super().save(fileName,chunks,concat)
	
	def fetch(self,arkType="mat",keys=None):
		"""
		Fetch records from file.

		Args:
			<keys>: utterance ID or a list of utterance IDs.
			<arkType>: If None,return BytesMatrix or BytesVector object.
					   If "feat",return BytesFeat object.
					   If "cmvn",return BytesCMVN object.
					   If "prob",return BytesProb object.
					   If "ali",return BytesAliTrans object.
					   If "fmllr",return BytesFmllr object.
					   If "mat",return BytesMatrix object.
					   If "vec",return BytesVector object.
		
		Return:
		    an exkaldi bytes achieve object. 
		"""
		declare.not_void(type_name(self),self)
		declare.is_instances("arkType",arkType,[None,"mat","vec","feat","cmvn","prob","ali","fmllrMat"])

		if keys is None:
			keys = self.keys()
		else:
			declare.is_classes("keys",keys,[str,list,tuple])
			if isinstance(keys,str):
				keys = [keys,]
			else:
				declare.members_are_valid_strings("keys",keys)

		newTable = IndexTable(name=self.name)

		def is_matrix(oneRecord):
			with BytesIO(oneRecord) as sp:
				while True:
					char = sp.read(1).decode()
					if (char == '') or (char == ' '):
						break
				sp.read(2)
				sizeSymbol = sp.read(1).decode()
				if sizeSymbol == '\4':	
					return False
				else:
					return True

		with FileHandleManager() as fhm:

			startIndex = 0
			datas = []

			for k in keys:
				if k in self.keys():
					indexInfo = self[k]
					if indexInfo.filePath is None:
						raise WrongDataFormat(f"Miss file path information in the index table: {k}.")
					
					fr = fhm.call(indexInfo.filePath)
					if fr is None:
						fr = fhm.open(indexInfo.filePath,mode="rb")
						
					fr.seek(indexInfo.startIndex)
					buf = fr.read(indexInfo.dataSize)
					newTable[k] = newTable.spec( indexInfo.frames,startIndex,indexInfo.dataSize,None )
					startIndex += indexInfo.dataSize
					datas.append(buf)

			if len(datas) == 0:
				raise WrongOperation("Miss all utterance IDs. We don't think it's a reasonable result. Check the provided <keys> please.")

			matrixFlag = is_matrix(buf)
			
			if arkType is None:
				if matrixFlag is True:
					result = BytesMatrix( b"".join(datas),name=self.name,indexTable=newTable )
				else:
					result = BytesVector( b"".join(datas),name=self.name,indexTable=newTable )
			elif arkType == "mat":
				result = BytesMatrix( b"".join(datas),name=self.name,indexTable=newTable )
			elif arkType == "vec":
				result = BytesVector( b"".join(datas),name=self.name,indexTable=newTable )		
			elif arkType == "feat":
				result = BytesFeat( b"".join(datas),name=self.name,indexTable=newTable )
			elif arkType == "cmvn":
				result = BytesCMVN( b"".join(datas),name=self.name,indexTable=newTable )
			elif arkType == "prob":
				result = BytesProb( b"".join(datas),name=self.name,indexTable=newTable )
			elif arkType == "ali":
				result = BytesAliTrans( b"".join(datas),name=self.name,indexTable=newTable )
			else:
				result = BytesFmllr( b"".join(datas),name=self.name,indexTable=newTable )
			
			result.check_format()

			return result

	@property
	def utts(self):
		'''
		Get a list of all utterance IDs.
		'''
		return list(self.keys())

	@property
	def spks(self):
		'''
		The same as self.utts.
		'''
		return self.utts

class WavSegment(ListTable):
	'''
	It is a class to hold segment info.
	'''
	def __init__(self,data={},name="segment"):
		declare.is_classes("data",data,[dict,WavSegment])
		super().__init__(name=name)
		for key,value in data.items():
			self[key] = value

	def __setitem__(self,key,value):
		'''Overlap this method to avoid the wrong assignment.'''
		if isinstance(value,[list,tuple]):
			assert len(value) in [4,5],f"Expected (fileID,startTime,endTime,filePath[,text]) but {value} does not match."
			self.record(key,*value)
		elif isinstance(value,SegmentInfo):
			super().__setitem__(key,value)
		else:
			raise UnsupportedType(f"The value of index table should be list, tuple or SegmentInfo object but got: {type_name(value)}.")

	def setdefault(self,key,value=None):
		'''Overlap this method to avoid the wrong assignment.'''
		if self.key_existed(key):
			return self[key]
		else:
			self.__setitem__(key,value)

	def update(self,other):
		'''
		Args:
			<other>: exkaldi WavSegment object.
		'''
		declare.is_classes("other",other,WavSegment)

		super().update(other)

	def record(self,key=None,fileID=None,startTime=None,endTime=None,filePath=None,text=None):
		'''
		Add or modify a record.
		
		Args:
			<key>: a string. The utterance ID. If None, we will make it by: fileID-startTime-endTime.
			<fileID>: a string. the file ID.
			<startTime>: an float value. Seconds.
			<endTime>: an float value. Seconds.
			<filePath>: wav file path. a string.
			<dataSize>: an int value. The total size of an archive record. Including the utterance ID.
			<filePath>: a string. The total size of an archive record. Including the utterance ID.
			<text>: the transcription.
		
		'''
		if key is None:
			assert None not in [fileID,startTime,endTime], "When <key> has not been provided, we will make it from <fileID>, <startTime> and <endTime> automatically, so all or they are necessary."
			declare.is_valid_string("fileID", fileID)
			declare.is_non_negative_float("startTime",startTime)
			declare.is_non_negative_float("endTime",endTime)

			st = str("%.3f"%startTime).replace(".","")
			et = str("%.3f"%endTime).replace(".","")
			key = f"{fileID}-{st}-{et}" 

			if self.key_existed(key):
				value = self[key]
				value = value._replace(fileID=fileID,startTime=startTime,endTime=endTime)	
				if filePath is not None:
					declare.is_file("filePath",filePath)
					value = value._replace(filePath=filePath)				
				if text is not None:
					declare.is_valid_string("text",text)
					value = value._replace(text=text)		
			
			else:
				declare.is_file("filePath",filePath)
				if text is not None:
					declare.is_valid_string("text",text)

				value = self.spec(fileID,startTime,endTime,filePath,text)

		else:

			if self.key_existed(key):
				value = self[key]
				if fileID is not None:
					declare.is_valid_string("fileID", fileID)
					value = value._replace(fileID=fileID)
				if startTime is not None:
					declare.is_non_negative_float("startTime", startTime)
					value = value._replace(startTime=startTime)
				if endTime is not None:
					declare.is_non_negative_float("endTime", endTime)
					value = value._replace(endTime=endTime)		
				if filePath is not None:
					declare.is_file("filePath", filePath)
					value = value._replace(filePath=filePath)
				if text is not None:
					declare.is_valid_string("text", text)
					value = value._replace(text=text)					

			else:
				declare.is_valid_string("fileID", fileID)
				declare.is_non_negative_float("startTime", startTime)
				declare.is_non_negative_float("endTime", endTime)
				declare.is_file("filePath", filePath)
				if text is not None:
					declare.is_valid_string("text", text)

				value = self.spec(fileID,startTime,endTime,filePath,text)

		super().__setitem__(key,value)

	@property
	def spec(self):
		'''
		The segment info spec.

		Return:
			a namedtuple class.
		'''
		spec = namedtuple("SegmentInfo",["fileID","startTime","endTime","filePath","text"])
		spec.__new__.__defaults__ = (None,)
		return spec
	
	def sort(self,by="utt",reverse=False):
		'''
		Sort utterances utterance ID or start time or file path name.

		Args:
			<by>: "key"/"utt","value"/"startTime" or "filePath".
			<reverse>: If True,sort in descending order.
		
		Return:
			A new WavSegment object.
		''' 
		declare.is_instances("by",by,["utt","key","value","startTime","filePath"])

		if by in ["utt","key"]:
			items = sorted(self.items(),key=lambda x:x[0],reverse=reverse)
		elif by in ["startTime","value"]:
			items = sorted(self.items(),key=lambda x:x[1].startTime,reverse=reverse)
		else:
			items = sorted(self.items(),key=lambda x:x[1].filePath,reverse=reverse)
		
		newName = f"sort({self.name},{by})"
		return WavSegment(dict(items),name=newName)	

	def __add__(self,other):
		'''
		Integrate two WavSegment objects. If utterance has existed in both two objects,the former will be retained.

		Args:
			<other>: another WavSegment object.

		Return:
			A new WavSegment object.
		'''
		declare.is_classes("other",other,WavSegment)

		result = super().__add__(other)
		return WavSegment(result.data,name=result.name)	
	
	def shuffle(self):
		'''
		Random shuffle the segment.

		Return:
			A new WavSegment object.
		'''
		result = super().shuffle()

		return WavSegment(result.data,name=self.name)	
	
	def subset(self,nHead=0,nTail=0,nRandom=0,chunks=1,keys=None):
		'''
		Subset.
		Only one mode will work when it is not the default value. 
		The priority order is: nHead > nTail > nRandom > chunks > keys.
		
		Args:
			<nHead>: If it > 0,extract N head utterances.
			<nTail>: If it > 0,extract N tail utterances.
			<nRandom>: If it > 0,randomly sample N utterances.
			<chunks>: If it > 1,split data into N chunks.
			<keys>: If it is not None,pick out these utterances whose ID in keys.

		Return:
			a new WavSegment object or a list of new WavSegment objects.
		''' 
		result = super().subset(nHead,nTail,nRandom,chunks,keys)

		if isinstance(result,list):
			for index in range(len(result)):
				temp = result[index]
				result[index] = WavSegment(temp.data,temp.name)
		else:
			result = WavSegment(result.data,result.name)

		return result	

	def save(self,fileName=None,chunks=1):
		'''
		Save this segment to file with Kaldi segment file format.
		Note that the <filePath> and <text> information will be discarded. If you want to save them, please detach them respectively.

		Args:
			<fileName>: None,file name or file handle.  
			<chunks>: an int value. If > 1,split it into N chunks and save them.  
								This option only work when _fileName_ is a file name.  
		
		Return:
			file name,file handle or a string.
		'''
		declare.not_void(type_name(self),self)

		def concat(item):
			utt,segmentInfo = item
			return f"{utt} {segmentInfo.fileID} {segmentInfo.startTime} {segmentInfo.endTime}"

		return super().save(fileName,chunks,concat)

	def detach_wav(self):
		'''
		Detach file ID - wav file path information from segments.

		Return:
			an exkaldi ListTable object.
		'''
		declare.not_void(type_name(self),self)

		wavs = ListTable(name=f"wavs({self.name})")
		for key,segmentInfo in self.items():
			fileID,filePath = segmentInfo.fileID,segmentInfo.filePath
			if not wavs.key_existed(fileID):
				wavs.record(fileID,filePath)
		
		return wavs

	def extract_segment(self,outDir=None):
		'''
		Extract segment and save them to file.
		If outDir is None, save the segment wav file in the same as original wav file.

		Return:
			an exkaldi ListTable object. The generated wav.scp .
		'''
		declare.not_void(type_name(self),self)
		declare.kaldi_existed()
		if outDir is not None:
			declare.is_dir("outDir",outDir)

		with FileHandleManager() as fhm:

			segTemp = fhm.create("w+", suffix=".seg")
			self.save(segTemp)

			wavTemp = fhm.create("w+", suffix=".scp")
			self.detach_wav().save(wavTemp)

			cmd = f"extract-segments scp:{wavTemp.name} {segTemp.name} ark:-"
			out,err,cod = run_shell_command(cmd,stdout="PIPE",stderr="PIPE")
			if cod != 0:
				raise KaldiProcessError("Failed to extract segment.",err.decode())
			else:
				segWavs = ListTable(name=f"extract({self.name})")
				with BytesIO(out) as sp:
					while True:
						utt = ""
						while True:
							t = sp.read(1).decode()
							if t == " " or t == "":
								break
							else:
								utt += t

						header_ = sp.read(40)
						subchunk2size = sp.read(4)
						dataSize = struct.unpack("<L",subchunk2size)[0]
						buffer = sp.read(dataSize)

						SegmentInfo = self[utt]
						filePath = SegmentInfo.filePath
						if outDir is None:
							outFile = os.path.join(os.path.dirname(filePath),utt+".wav")
						else:
							outFile = os.path.join(outDir,utt+".wav")

						with open(outFile,"wb") as fw:
							fw.write( header_ + subchunk2size + buffer )
						
						segWavs.record(utt,outFile)

		return segWavs

	def detach_transcription(self):
		'''
		Detach utterance ID - text information from segments.

		Return:
			an exkaldi Transcription object.
		'''
		declare.not_void(type_name(self),self)

		trans = Transcription(name=f"transcription({self.name})")
		for key,segmentInfo in self.items():
			if segmentInfo.text is None:
				raise WrongOperation("Cannnot detach transcription because it did not existed.")
			trans.record(key,segmentInfo.text)
		
		return trans

	@property
	def utts(self):
		'''
		Get a list of all utterance IDs.
		'''
		return list(self.keys())

'''BytesArchive class group'''
'''Designed for Kaldi binary archive table. It also support other objects such as lattice,HMM-GMM and decision tree'''
## Base class
class BytesArchive:
	'''
	The base class of archive. 
	'''
	def __init__(self,data=b'',name=None):

		if data != None:
			declare.is_classes("data",data,bytes)
		self.__data = data

		if name is None:
			self.__name = self.__class__.__name__
		else:
			declare.is_valid_string("name",name)
			self.__name = name
	
	@property
	def data(self):
		'''
		Get the inner data.
		'''
		return self.__data
	
	def reset_data(self,data=None):
		'''
		Reset data. If None,clear the table.

		Return:
			<data>: bytes object.
		'''
		if data != None:
			declare.is_classes("data",data,bytes)
		del self.__data
		self.__data = data

	@property
	def is_void(self):
		'''
		Check whether this is a void object.
		'''
		if self.__data is None or len(self.__data) == 0:
			return True
		else:
			return False

	@property
	def name(self):
		'''
		Get the name.
		'''
		return self.__name

	def rename(self,name=None):
		'''
		Rename it.
		'''
		if name is not None:
			declare.is_valid_string("name",name)
		else:
			name = self.__class__.__name__
		self.__name = name

## Base class: for Matrix Data archives
class BytesMatrix(BytesArchive):
	'''
	A base class for matrix data,such as feature,cmvn statistics,post probability.
	'''
	def __init__(self,data=b'',name="data",indexTable=None):
		'''
		Args:
			<data>: If it's BytesMatrix or IndexTable or NumpyMatrix object (or their subclasses),extra <indexTable> will not work.
					If it's bytes object (or their subclasses),generate index table automatically if it is not provided.
			<name>: a string.
			<indexTable>: IndexTable object.
		'''
		declare.belong_classes("data",data,[BytesMatrix,NumpyMatrix,IndexTable,bytes])

		needIndexTableFlag = True

		if isinstance(data,BytesMatrix):
			self.__dataIndex = data.indexTable
			self.__dataIndex.rename(name)
			data = data.data
			needIndexTableFlag = False
		
		elif isinstance(data ,IndexTable):
			data = data.fetch(arkType="mat",name=name)
			self.__dataIndex = data.indexTable
			data = data.data
			needIndexTableFlag = False

		elif isinstance(data,NumpyMatrix):
			data = data.to_bytes()
			self.__dataIndex = data.indexTable
			data = data.data
			needIndexTableFlag = False

		super().__init__(data,name)

		if needIndexTableFlag is True:
			if indexTable is None:
				self.__generate_index_table()
			else:
				declare.is_classes("indexTable",indexTable,IndexTable)
				self.__verify_index_table(indexTable)
	
	def __verify_index_table(self,indexTable):
		'''
		Check the format of provided index table.
		'''
		newIndexTable = indexTable.sort("startIndex")
		start = 0
		for key,indexInfo in newIndexTable.items():
			if indexInfo.startIndex != start:
				raise WrongDataFormat(f"Start index of {key} dose not match: expected {start} but got {indexInfo.startIndex}.")
			if indexInfo.filePath is not None:
				newIndexTable[key] = indexInfo._replace(filePath=None)
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
			self.__dataIndex = IndexTable(name=self.name)
			start = 0
			with BytesIO(self.data) as sp:
				while True:
					(utt,dataType,rows,cols,bufSize,buf) = self.__read_one_record(sp)
					if utt is None:
						break
					oneRecordLen = len(utt) + 16 + bufSize

					self.__dataIndex[utt] = self.__dataIndex.spec(rows,start,oneRecordLen)
					start += oneRecordLen

	def __read_one_record(self,fp):
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
				return (None,None,None,None,None,None)
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
					raise WrongDataFormat("This might not be Kaldi archive data.")
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
				raise WrongDataFormat(f"Expected data type FM(float32),DM(float64),CM(compressed data) but got {dataType}.")
			s1,rows,s2,cols = np.frombuffer(fp.read(10),dtype="int8,int32,int8,int32",count=1)[0]
			rows = int(rows)
			cols = int(cols)
			bufSize = rows * cols * sampleSize
			buf = fp.read(bufSize)
		else:
			fp.close()
			raise WrongDataFormat("Miss binary symbol before utterance.")
		return (utt,dataType,rows,cols,bufSize,buf)

	@property
	def indexTable(self):
		'''
		Get the index information of utterances.
		
		Return:
			A IndexTable object.
		'''
		# Return a dict object.
		return copy.deepcopy(self.__dataIndex)

	@property
	def dtype(self):
		'''
		Get the data type of bytes data.
		
		Return:
			None,or a string in 'float32','float64'.
		'''
		if self.is_void:
			_dtype = None
		else:
			with BytesIO(self.data) as sp:
				(utt,dataType,rows,cols,bufSize,buf) = self.__read_one_record(sp)
			if dataType == "FM ":
				_dtype = "float32"
			else:
				_dtype = "float64"
             
		return _dtype

	def to_dtype(self,dtype):
		'''
		Transform data type.

		Args:
			<dtype>: a string of "float","float32" or "float64". If "float",it will be treated as "float32".
		
		Return:
			A new BytesMatrix object.
		'''
		declare.is_instances("dtype",dtype,["float","float32","float64"])
		declare.not_void(type_name(self),self)

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
			newDataIndex = IndexTable(name=self.name)
			# Data size will be changed so generate a new index table.
			with BytesIO(self.data) as sp:
				start = 0
				while True:
					(utt,dataType,rows,cols,bufSize,buf) = self.__read_one_record(sp)
					if utt is None:
						break
					if dataType == 'FM ': 
						matrix = np.frombuffer(buf,dtype=np.float32)
					else:
						matrix = np.frombuffer(buf,dtype=np.float64)
					newMatrix = np.array(matrix,dtype=dtype).tobytes()
					data = (utt+' '+'\0B'+newDataType).encode()
					data += '\04'.encode()
					data += struct.pack(np.dtype('uint32').char,rows)
					data += '\04'.encode()
					data += struct.pack(np.dtype('uint32').char,cols)
					data += newMatrix
					result.append(data)

					oneRecordLength = len(data)
					newDataIndex[utt] = newDataIndex.spec(rows,start,oneRecordLength)
					start += oneRecordLength
					
			result = b''.join(result)

			return BytesMatrix(result,name=self.name,indexTable=newDataIndex)

	@property
	def dim(self):
		'''
		Get the data dimensions.
		
		Return:
			If data is void,return None,or return an int value.
		'''
		if self.is_void:
			return None
		else:
			with BytesIO(self.data) as sp:
				(utt,dataType,rows,cols,bufSize,buf) = self.__read_one_record(sp)
			
			return cols

	def keys(self):
		'''
		Get a iterator of all keys (utterance IDs or speaker IDs).
		'''
		if  self.__dataIndex is None:
			return dict().keys()
		else:
			return self.__dataIndex.keys()

	def check_format(self):
		'''
		Check whether data has right Kaldi format.
		
		Return:
			If data is void,return False.
			If data has right format,return True,or raise Error.
		'''
		if self.is_void:
			return False

		_dim = "unknown"
		_dataType = "unknown"
		with BytesIO(self.data) as sp:
			start = 0
			while True: 
				(utt,dataType,rows,cols,bufSize,buf) = self.__read_one_record(sp)
				if utt is None:
					break
				if _dim == "unknown":
					_dim = cols
					_dataType = dataType
				elif cols != _dim:
					raise WrongDataFormat(f"Expected dimension {_dim} but got {cols} at utterance {utt}.")
				elif _dataType != dataType:
					raise WrongDataFormat(f"Expected data type {_dataType} but got {dataType} at utterance {utt}.")                 
				else:
					try:
						if dataType == "FM ":
							mat = np.frombuffer(buf,dtype=np.float32)
						else:
							mat = np.frombuffer(buf,dtype=np.float64)
					except Exception as e:
						e.args = ( f"Wrong matrix data format at utterance {utt}." + "\n" + e.args[0],)
						raise e
				
				oneRecordLen = len(utt) + 16 + bufSize

				# Renew the index table.
				self.__dataIndex[utt] = self.__dataIndex.spec(rows,start,oneRecordLen)
				start += oneRecordLen			
					
		return True
	
	@property
	def lens(self):
		'''
		Get the numbers of utterances.
		If you want to get the frames of each utterance,try:
						obj.indexTable 
		attribute.
		
		Return:
			a int value.
		'''
		lengths = 0
		if not self.is_void:
			lengths = len(self.__dataIndex)
		
		return lengths

	def save(self,fileName,chunks=1,returnIndexTable=False):
		'''
		Save bytes data to file.

		Args:
			<fileName>: file name or file handle. If it's a file name,suffix ".ark" will be add to the name defaultly.
			<chunks>: If larger than 1,data will be saved to multiple files averagely. This would be invalid when <fileName> is a file handle.
			<returnIndexTable>: If True,return the index table containing the information of file path.
		
		Return:
			the path of saved files.
		'''
		declare.not_void(type_name(self),self)
		declare.is_valid_file_name_or_handle("fileName",fileName)
		declare.greater_equal("chunks",chunks,None,1)
		declare.is_bool("returnIndexTable",returnIndexTable)

		if isinstance(fileName,str):

			def save_chunk_data(chunkData,arkFileName,returnIndexTable):

				make_dependent_dirs(arkFileName,pathIsFile=True)
				with open(arkFileName,"wb") as fw:
					fw.write(chunkData.data)
				
				if returnIndexTable is True:
					indexTable = chunkData.indexTable
					for key in indexTable.keys():
						indexTable[key] = indexTable[key]._replace(filePath=arkFileName)

					return indexTable
				else:
					return arkFileName

			fileName = fileName.strip()
			if not fileName.endswith('.ark'):
				fileName += '.ark'

			if chunks == 1:
				savedFiles = save_chunk_data(self,fileName,returnIndexTable)	
			
			else:
				dirName = os.path.dirname(fileName)
				fileName = os.path.basename(fileName)
				savedFiles = []
				chunkDataList = self.subset(chunks=chunks)
				newFileNamePattern = f"ck%0{len(str(chunks))}d_{fileName}"
				for i,chunkData in enumerate(chunkDataList):
					chunkFileName = os.path.join( dirName,newFileNamePattern%i )
					savedFiles.append( save_chunk_data(chunkData,chunkFileName,returnIndexTable) )

			return savedFiles
		
		else:
			fileName.truncate()
			fileName.write(self.data)
			fileName.seek(0)

			return fileName

	def to_numpy(self):
		'''
		Transform bytes data to NumPy format.
		
		Return:
			a NumpyMatrix object sorted by key.
		'''
		newDict = {}
		if not self.is_void:
			sortedIndex = self.indexTable.sort(by="key",reverse=False)
			with BytesIO(self.data) as sp:
				for key,indexInfo in sortedIndex.items():
					sp.seek(indexInfo.startIndex)
					(utt,dataType,rows,cols,bufSize,buf) = self.__read_one_record(sp)
					try:
						if dataType == 'FM ': 
							newMatrix = np.frombuffer(buf,dtype=np.float32)
						else:
							newMatrix = np.frombuffer(buf,dtype=np.float64)
					except Exception as e:
						e.args = ( f"Wrong matrix data format at utterance {key}." + "\n" + e.args[0],)
						raise e	
					else:
						newDict[key] = np.reshape(newMatrix,(rows,cols))

		return NumpyMatrix(newDict,name=self.name)

	def __add__(self,other):
		'''
		The plus operation between two objects.

		Args:
			<other>: a BytesMatrix,NumpyMatrix,IndexTable object (or their subclasses object).
		
		Return:
			a new BytesMatrix object.
		''' 
		declare.belong_classes("other",other,[BytesMatrix,NumpyMatrix,IndexTable])

		if isinstance(other,NumpyMatrix):
			other = other.to_bytes()
		elif isinstance(other,IndexTable):
			keys = [ key for key in other.keys() if key not in self.keys() ]
			other = other.fecth(arkType="mat",keys=keys)
		
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

		selfDtype = self.dtype
		otherDtype = other.dtype

		newDataIndex = self.indexTable
		start = len(self.data)

		newData = []
		with BytesIO(other.data) as op:
			for utt,indexInfo in other.indexTable.items():
				if not utt in self.keys():
					op.seek( indexInfo.startIndex )
					if selfDtype == otherDtype:
						data = op.read( indexInfo.dataSize )
						data_size = indexInfo.dataSize

					else:
						(outt,odataType,orows,ocols,obufSize,obuf) = self.__read_one_record(op)
						obuf = np.array(np.frombuffer(obuf,dtype=otherDtype),dtype=selfDtype).tobytes()
						data = (outt+' '+'\0B'+odataType).encode()
						data += '\04'.encode()
						data += struct.pack(np.dtype('uint32').char,orows)
						data += '\04'.encode()
						data += struct.pack(np.dtype('uint32').char,ocols)
						data += obuf
						data_size = len(data)

					newDataIndex[utt] = newDataIndex.spec(indexInfo.frames,start,data_size)
					start += data_size

					newData.append(data)

		return BytesMatrix(b''.join([self.data,*newData]),name=newName,indexTable=newDataIndex)

	def subset(self,nHead=0,nTail=0,nRandom=0,chunks=1,keys=None):
		'''
		Subset data.
		The priority of mode is nHead > nTail > nRandom > chunks > keys.
		If you chose multiple modes,only the prior one will work.
		
		Args:
			<nHead>: get N head utterances.
			<nTail>: get N tail utterances.
			<nRandom>: sample N utterances randomly.
			<chunks>: split data into N chunks averagely.
			<keys>: pick out these utterances whose ID in keys.
		
		Return:
			a new BytesMatrix object or a list of new BytesMatrix objects.
		''' 
		declare.not_void(type_name(self),self)

		if nHead > 0:
			declare.is_positive_int("nHead",nHead)
			newName = f"subset({self.name},head {nHead})"
			newDataIndex = IndexTable(name=newName)
			totalSize = 0
			
			for utt,indexInfo in self.indexTable.items():
				newDataIndex[utt] = newDataIndex.spec(indexInfo.frames,totalSize,indexInfo.dataSize)
				totalSize += indexInfo.dataSize
				
				nHead -= 1
				if nHead <= 0:
					break
			
			with BytesIO(self.data) as sp:
				sp.seek(0)
				data = sp.read(totalSize)
	
			return BytesMatrix(data,name=newName,indexTable=newDataIndex)

		elif nTail > 0:
			declare.is_positive_int("nTail",nTail)
			newName = f"subset({self.name},tail {nTail})"
			newDataIndex = IndexTable(name=newName)

			tailNRecord = list(self.indexTable.items())[-nTail:]
			start_index = tailNRecord[0][1].startIndex

			totalSize = 0
			for utt,indexInfo in tailNRecord:
				newDataIndex[utt] = newDataIndex.spec(indexInfo.frames,totalSize,indexInfo.dataSize)
				totalSize += indexInfo.dataSize

			with BytesIO(self.data) as sp:
				sp.seek(start_index)
				data = sp.read(totalSize)
	
			return BytesMatrix(data,name=newName,indexTable=newDataIndex)

		elif nRandom > 0:
			declare.is_positive_int("nRandom",nRandom)

			if nRandom >= self.lens:
				newName = f"subset({self.name},random {self.lens})"
				return BytesMatrix(self,name=newName)

			randomNRecord = random.sample(list(self.indexTable.items()),k=nRandom)
			newName = f"subset({self.name},random {nRandom})"

			newDataIndex = IndexTable(name=newName)
			start_index = 0
			newData = []
			with BytesIO(self.data) as sp:
				for utt,indexInfo in randomNRecord:
					sp.seek(indexInfo.startIndex)
					newData.append( sp.read(indexInfo.dataSize) )
					newDataIndex[utt] = newDataIndex.spec(indexInfo.frames,start_index,indexInfo.dataSize)
					start_index += indexInfo.dataSize
					
			return BytesMatrix(b"".join(newData),name=newName,indexTable=newDataIndex)

		elif chunks > 1:
			declare.is_positive_int("chunks",chunks)

			uttLens = list(self.indexTable.items())
			allLens = len(uttLens)
			chunkLens = allLens//chunks
			if chunkLens == 0:
				chunks = allLens
				chunkLens = 1
				t = 0
				print(f"Warning: utterances is fewer than <chunks> so only {chunks} files will be saved.")
			else:
				t = allLens - chunkLens * chunks
			
			datas = []
			with BytesIO(self.data) as sp:                          
				sp.seek(0)
				start = 0
				for i in range(chunks):
					newName = f"subset({self.name},chunk {chunks}-{i})"
					newDataIndex = IndexTable(name=newName)
					if i < t:
						end = start + chunkLens + 1
					else:
						end = start + chunkLens
					chunkItems = uttLens[start:end]

					chunkDataSize = 0
					for utt,indexInfo in chunkItems:
						newDataIndex[utt] = newDataIndex.spec(indexInfo.frames,chunkDataSize,indexInfo.dataSize)
						chunkDataSize += indexInfo.dataSize
					chunkData = sp.read(chunkDataSize)
					
					datas.append( BytesMatrix(chunkData,name=newName,indexTable=newDataIndex) )
					start = end
			return datas

		elif keys != None:
			declare.is_classes("keys",keys,[list,tuple])
			declare.members_are_valid_strings("keys",keys)
			newName = f"subset({self.name},keys {len(keys)})"

			newData = []
			dataIndex = self.indexTable
			newDataIndex = IndexTable(name=newName)
			start_index = 0
			with BytesIO(self.data) as sp:
				for utt in keys:
					if utt in self.keys():
						indexInfo = dataIndex[utt]
						sp.seek( indexInfo.startIndex )
						newData.append( sp.read(indexInfo.dataSize) )

						newDataIndex[utt] = newDataIndex.spec(indexInfo.frames,start_index,indexInfo.dataSize)
						start_index += indexInfo.dataSize

			return BytesMatrix(b''.join(newData),name=newName,indexTable=newDataIndex)
		
		else:
			raise WrongOperation('Expected one of <nHead>,<nTail>,<nRandom>,<chunks> or <keys> is avaliable but all got the default value.')

	def __call__(self,utt):
		'''
		Pick out the specified utterance.
		
		Args:
			<utt>: a string.
		Return:
			If existed,return a new BytesMatrix object.
			Or return None.
		'''
		declare.is_valid_string("utt",utt)
		if self.is_void:
			return None

		utt = utt.strip()

		if utt not in self.keys():
			return None
		else:
			indexInfo = self.indexTable[utt]
			newName = f"pick({self.name},{utt})"
			newDataIndex = IndexTable(name=newName)
			with BytesIO(self.data) as sp:
				sp.seek( indexInfo.startIndex )
				data = sp.read( indexInfo.dataSize )

				newDataIndex[utt] =	indexInfo._replace(startIndex=0)
				result = BytesMatrix(data,name=newName,indexTable=newDataIndex)
			
			return result

	def sort(self,by="utt",reverse=False):
		'''
		Sort.

		Args:
			<by>: "frame"/"value" or "utt"/"key"/"spk".
			<reverse>: If reverse,sort in descending order.

		Return:
			A new BytesMatrix object.
		''' 
		declare.is_instances("by",by,["utt","key","spk","frame","value"])

		if by in ["utt","key","spk"]:
			newDataIndex = self.indexTable.sort(by="key",reverse=reverse)
		else:
			newDataIndex = self.indexTable.sort(by="value",reverse=reverse)
		ordered = True
		for i,j in zip(self.indexTable.items(),newDataIndex.items()):
			if i != j:
				ordered = False
				break
		if ordered:
			return copy.deepcopy(self)

		with BytesIO(self.data) as sp:
			if sys.getsizeof(self.data) > 10**9:
				## If the data size is large,divide it into N chunks and save it to intermidiate file.
				with FileHandleManager as fhm:
					temp = fhm.create("wb+")
					chunkdata = []
					chunkSize = 50
					count = 0
					start_index = 0

					for utt,indexInfo in newDataIndex.items():
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
				for utt,indexInfo in newDataIndex.items():
					sp.seek( indexInfo.startIndex )
					newData.append( sp.read(indexInfo.dataSize) )

					newDataIndex[utt] = indexInfo._replace(startIndex=start_index)
					start_index += indexInfo.dataSize

				newData = b"".join(newData)

		return BytesMatrix(newData,name=self.name,indexTable=newDataIndex)			

## Subclass: for acoustic feature (in binary format)		
class BytesFeat(BytesMatrix):
	'''
	Hold the feature with Kaldi binary format.
	'''
	def __init__(self,data=b"",name="feat",indexTable=None):
		'''
		Only allow BytesFeat,NumpyFeat,IndexTable or bytes (do not extend to their subclasses and their parent-classes).
		'''
		declare.is_classes("data",data,[BytesFeat,NumpyFeat,IndexTable,bytes])
		super().__init__(data,name,indexTable)
	
	@property
	def utts(self):
		'''
		Get all utterance IDs.

		Return:
		  a list. 
		'''
		return list(self.keys())

	def to_numpy(self):
		'''
		Transform feature to numpy format.

		Return:
			a NumpyFeat object.
		'''
		result = super().to_numpy()
		return NumpyFeat(result.data,name=result.name)

	def __add__(self,other):
		'''
		Plus operation between two feature objects.

		Args:
			<other>: a BytesFeat or NumpyFeat object.
		Return:
			a BytesFeat object.
		'''
		declare.is_feature("other",other)

		result = super().__add__(other)

		return BytesFeat(result.data,name=result.name,indexTable=result.indexTable)

	def splice(self,left=1,right=None):
		'''
		Splice front-behind N frames to generate new feature.

		Args:
			<left>: the left N-frames to splice.
			<right>: the right N-frames to splice. If None,right = left.

		Return:
			A new BytesFeat object whose dim became original-dim * (1 + left + right).
		''' 
		declare.kaldi_existed()
		declare.not_void(type_name(self),self)
		declare.is_non_negative_int("left",left)

		if right is None:
			right = left
		else:
			declare.is_non_negative_int("right",right)
		
		cmd = f"splice-feats --left-context={left} --right-context={right} ark:- ark:-"
		out,err,cod = run_shell_command(cmd,stdin="PIPE",stdout="PIPE",stderr="PIPE",inputs=self.data)

		if cod != 0 or out == b'':
			raise KaldiProcessError("Failed to splice left-right frames.",err.decode())
		else:
			newName = f"splice({self.name},{left},{right})"
			# New index table will be generated later.
			return BytesFeat(out,name=newName,indexTable=None)

	def select(self,dims,retain=False):
		'''
		Select specified dimensions of feature.

		Args:
			<dims>: A int value or string such as "1,2,5-10"
			<retain>: If True,return the rest dimensions of feature simultaneously.

		Return:
			A new BytesFeat object or two BytesFeat objects.
		''' 
		declare.kaldi_existed()
		declare.not_void( type_name(self),self )

		_dim = self.dim

		if isinstance(dims,int):
			declare.in_boundary("Selected index",dims,minV=0,maxV=_dim-1)
			selectFlag = str(dims)
			if retain:
				if dims == 0:
					retainFlag = f"1-{_dim-1}"
				elif dims == _dim-1:
					retainFlag = f"0-{_dim-2}"
				else:
					retainFlag = f"0-{dims-1},{dims+1}-{_dim-1}"
		
		elif isinstance(dims,str):
			declare.is_valid_string("dims",dims)
			selectFlag = dims

			if retain:
				retainFlag = list(range(_dim))
				for i in dims.strip().split(','):
					i = i.strip()
					if i == "":
						continue
					if not '-' in i:
						try:
							i = int(i)
						except ValueError:
							raise WrongOperation(f"Selected index should be an int value but got {i}.")
						else:
							declare.in_boundary("Selected index",i,minV=0,maxV=_dim-1)
							retainFlag[i] = -1 # flag
					else:
						i = i.split('-')
						assert len(i) == 2,"Selected indexes should has format like '1-2','3-' or '-5'."
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
								i[0],i[1] = i[1],i[0]
							declare.in_boundary("Selected index",i[1],minV=0,maxV=_dim-1)
						for j in range(i[0],i[1]+1,1):
							retainFlag[j] = -1 # flag
				temp = ''
				for x in retainFlag:
					if x != -1:
						temp += str(x) + ','
				retainFlag = temp[:-1]
		
		else:
			raise WrongOperation(f"Expected int value or string like '1,4-9,12' but got {dims}.")

		cmdS = f'select-feats {selectFlag} ark:- ark:-'
		outS,errS,codS = run_shell_command(cmdS,stdin="PIPE",stdout="PIPE",stderr="PIPE",inputs=self.data)
		
		if codS != 0 or outS == b'':
			raise KaldiProcessError("Failed to select data.",errS.decode())
		else:
			newName = f"select({self.name},{dims})"
			# New index table will be generated later.
			selectedResult = BytesFeat(outS,name=newName,indexTable=None)

		if retain:
			if retainFlag == "":
				newName = f"select({self.name},void)"
				# New index table will be generated later.
				retainedResult = BytesFeat(name=newName,indexTable=None)
			else: 
				cmdR = f"select-feats {retainFlag} ark:- ark:-"
				outR,errR,codR = run_shell_command(cmdR,stdin="PIPE",stdout="PIPE",stderr="PIPE",inputs=self.data)
				if codR != 0 or outR == b'':
					raise KaldiProcessError("Failed to select retained data.",errR.decode())
				else:
					newName = f"select({self.name},not {dims})"
					# New index table will be generated later.
					retainedResult = BytesFeat(outR,name=newName,indexTable=None)
		
			return selectedResult,retainedResult
		
		else:
			return selectedResult

	def subset(self,nHead=0,nTail=0,nRandom=0,chunks=1,keys=None):
		'''
		Subset data.
		The priority of mode is nHead > nTail > nRandom > chunks > keys.
		If you chose multiple modes,only the prior one will work.
		
		Args:
			<nHead>: get N head utterances.
			<nTail>: get N tail utterances.
			<nRandom>: sample N utterances randomly.
			<chunks>: split data into N chunks averagely.
			<keys>: pick out these utterances whose ID in keys.
		Return:
			a new BytesFeat object or a list of new BytesFeat objects.
		''' 
		result = super().subset(nHead,nTail,nRandom,chunks,keys)

		if isinstance(result,list):
			for index in range(len(result)):
				temp = result[index]
				result[index] = BytesFeat(temp.data,temp.name,temp.indexTable)
		else:
			result = BytesFeat(result.data,result.name,result.indexTable)

		return result
	
	def __call__(self,uttID):
		'''
		Pick out an utterance.
		
		Args:
			<uttID>: a string.
		Return:
			If existed,return a new BytesFeat object.
		''' 
		result = super().__call__(uttID)
		if result is not None:
			result = BytesFeat(result.data,result.name,result.indexTable)
		return result

	def add_delta(self,order=2):
		'''
		Add N orders delta informat to feature.

		Args:
			<order>: A positive int value.

		Return:
			A new BytesFeat object whose dimendion became original-dim * (1 + order). 
		''' 
		declare.kaldi_existed()
		declare.is_positive_int("order",order)
		declare.not_void( type_name(self),self )

		cmd = f"add-deltas --delta-order={order} ark:- ark:-"
		out,err,cod = run_shell_command(cmd,stdin="PIPE",stdout="PIPE",stderr="PIPE",inputs=self.data)
		if cod != 0 or out == b'':
			raise KaldiProcessError('Failed to add delta feature.',err.decode())
		else:
			newName = f"delta({self.name},{order})"
			# New index table need to be generated later.
			return BytesFeat(data=out,name=newName,indexTable=None)

	def paste(self,others):
		'''
		Paste feature in feature dimension.

		Args:
			<others>: a feature object or list of feature objects.

		Return:
			a new feature object.
		''' 
		declare.kaldi_existed()
		declare.not_void(type_name(self),self)
		if isinstance(others,(list,tuple)):
			for fe in others:
				declare.is_feature("others",fe)
		else:
			declare.is_feature("others",others)

		otherResp = []
		pastedName = [self.name,]
		
		with FileHandleManager() as fhm:
		
			if isinstance(others,BytesFeat):
				temp = fhm.create("wb+",suffix=".ark")
				others.sort(by="utt").save(temp)
				otherResp.append( f"ark:{temp.name}" )
				pastedName.append( others.name )

			elif isinstance(others,NumpyFeat):
				temp = fhm.create("wb+",suffix=".ark")
				others.sort(by="utt").to_bytes().save(temp)
				otherResp.append( f"ark:{temp.name}" )
				pastedName.append( others.name )
			
			elif isinstance(others,IndexTable):
				temp = fhm.create("w+",suffix=".scp")
				others.sort(by="utt").save(temp)
				otherResp.append( f"scp:{temp.name}" )
				pastedName.append( others.name )

			else:
				for ot in others:

					if isinstance(ot,BytesFeat):
						temp = fhm.create("wb+",suffix=".ark")
						ot.sort(by="utt").save(temp)
						otherResp.append( f"ark:{ot.name}" )

					elif isinstance(ot,NumpyFeat):
						temp = fhm.create("wb+",suffix=".ark")
						ot.sort(by="utt").to_bytes().save(temp)
						otherResp.append( f"ark:{ot.name}" )

					else:
						temp = fhm.create("w+",suffix=".scp")
						ot.sort(by="utt").save(temp)
						otherResp.append( f"scp:{ot.name}" )

					pastedName.append( ot.name )	
			
			selfData = fhm.create("wb+",suffix=".ark")
			self.sort(by="utt").save(selfData)

			otherResp = " ".join(otherResp)
			cmd = f"paste-feats ark:{selfData.name} {otherResp} ark:-"
			
			out,err,cod = run_shell_command(cmd,stdin="PIPE",stdout="PIPE",stderr="PIPE")

			if cod != 0 or out == b'':
				raise KaldiProcessError("Failed to paste feature.",err.decode())
			else:
				pastedName = ",".join(pastedName)
				pastedName = f"paste({pastedName})"
				# New index table need to be generated later.
				return BytesFeat(out,name=pastedName,indexTable=None)

	def sort(self,by="utt",reverse=False):
		'''
		Sort utterances by frames length or uttID

		Args:
			<by>: "frame"/"value" or "utt"/"key"
			<reverse>: If reverse,sort in descending order.

		Return:
			A new BytesFeat object.
		''' 
		result = super().sort(by,reverse)
		return BytesFeat(result.data,name=result.name,indexTable=result.indexTable)

## Subclass: for CMVN statistics
class BytesCMVN(BytesMatrix):
	'''
	Hold the CMVN statistics with Kaldi binary format.
	'''
	def __init__(self,data=b"",name="cmvn",indexTable=None):
		'''
		Only allow BytesCMVN,NumpyCMVN,IndexTable or bytes (do not extend to their subclasses and their parent-classes).
		'''
		declare.is_classes("data",data,[BytesCMVN,NumpyCMVN,IndexTable,bytes])

		super().__init__(data,name,indexTable)

	@property
	def spks(self):
		'''
		Get all speaker IDs.

		Return:
		  a list. 
		'''
		return list(self.keys())

	def to_numpy(self):
		'''
		Transform CMVN statistics to numpy format.

		Return:
			a NumpyCMVN object.
		'''
		result = super().to_numpy()
		return NumpyCMVN(result.data,name=result.name)

	def __add__(self,other):
		'''
		Plus operation between two CMVN statistics objects.

		Args:
			<other>: a BytesCMVN or NumpyCMVN object.
		Return:
			a BytesCMVN object.
		'''	
		declare.is_cmvn("other",other)

		result = super().__add__(other)

		return BytesCMVN(result.data,name=result.name,indexTable=result.indexTable)

	def subset(self,nHead=0,nTail=0,nRandom=0,chunks=1,keys=None):
		'''
		Subset data.
		The priority of mode is nHead > nTail > nRandom > chunks > keys.
		If you chose multiple modes,only the prior one will work.
		
		Args:
			<nHead>: get N head utterances.
			<nTail>: get N tail utterances.
			<nRandom>: sample N utterances randomly.
			<chunks>: split data into N chunks averagely.
			<keys>: pick out these utterances whose ID in keys.
		Return:
			a new BytesCMVN object or a list of new BytesCMVN objects.
		''' 
		result = super().subset(nHead,nTail,nRandom,chunks,keys)

		if isinstance(result,list):
			for index in range(len(result)):
				temp = result[index]
				result[index] = BytesCMVN(temp.data,temp.name,temp.indexTable)
		else:
			result = BytesCMVN(result.data,result.name,result.indexTable)

		return result

	def __call__(self,spk):
		'''
		Pick out an utterance.
		
		Args:
			<spk>: a string. the spkeaker ID

		Return:
			If existed,return a new BytesCMVN object.
		''' 
		result = super().__call__(spk)
		if result is not None:
			result = BytesCMVN(result.data,result.name,result.indexTable)
		return result

	def sort(self,by="spk",reverse=False):
		'''
		Sort utterances by utterance ID.

		Args:
			<by>: only one mode,"spk"/"key".
			<reverse>: If reverse,sort in descending order.
		Return:
			A new BytesCMVN object.
		''' 
		declare.is_instances("by",by,["key","spk"])

		result = super().sort(by="key",reverse=reverse)
		return BytesCMVN(result.data,name=result.name,indexTable=result.indexTable)

## Subclass: for probability of neural network output
class BytesProb(BytesMatrix):
	'''
	Hold the probalility with Kaldi binary format.
	'''
	def __init__(self,data=b"",name="prob",indexTable=None):
		'''
		Only allow BytesProb,NumpyProb,IndexTable or bytes (do not extend to their subclasses and their parent-classes).
		'''		
		declare.is_classes("data",data,[BytesProb,NumpyProb,IndexTable,bytes])

		super().__init__(data,name,indexTable)

	@property
	def utts(self):
		'''
		Get all utterance IDs.

		Return:
		  a list. 
		'''
		return list(self.keys())

	def to_numpy(self):
		'''
		Transform post probability to numpy format.

		Return:
			a NumpyProb object.
		'''
		result = super().to_numpy()
		return NumpyProb(result.data,result.name)

	def __add__(self,other):
		'''
		Plus operation between two post probability objects.

		Args:
			<other>: a BytesProb or NumpyProb object.
		Return:
			a BytesProb object.
		'''
		declare.is_probability("other",other)

		result = super().__add__(other)

		return BytesProb(result.data,result.name,result.indexTable)

	def subset(self,nHead=0,nTail=0,nRandom=0,chunks=1,keys=None):
		'''
		Subset data.
		The priority of mode is nHead > nTail > nRandom > chunks > keys.
		If you chose multiple modes,only the prior one will work.
		
		Args:
			<nHead>: get N head utterances.
			<nTail>: get N tail utterances.
			<nRandom>: sample N utterances randomly.
			<chunks>: split data into N chunks averagely.
			<keys>: pick out these utterances whose ID in keys.
		Return:
			a new BytesProb object or a list of new BytesProb objects.
		''' 
		result = super().subset(nHead,nTail,nRandom,chunks,keys)

		if isinstance(result,list):
			for index in range(len(result)):
				temp = result[index]
				result[index] = BytesProb(temp.data,temp.name,temp.indexTable)
		else:
			result = BytesProb(result.data,result.name,result.indexTable)

		return result

	def __call__(self,utt):
		'''
		Pick out an utterance.
		
		Args:
			<utt>: a string.
		Return:
			If existed,return a new BytesProb object.
		''' 
		result = super().__call__(utt)
		if result is not None:
			result = BytesProb(result.data,result.name,result.indexTable)
		return result

	def sort(self,by="utt",reverse=False):
		'''
		Sort utterances by frames length or uttID

		Args:
			<by>: "frame"/"value" or "utt"/"key"
			<reverse>: If reverse,sort in descending order.

		Return:
			A new BytesProb object.
		''' 
		result = super().sort(by,reverse)

		return BytesProb(result.data,name=result.name,indexTable=result.indexTable)

class BytesFmllr(BytesMatrix):
	'''
	Hold the fMLLR transform matrix with Kaldi binary format.
	'''
	def __init__(self,data=b"",name="fmllrTrans",indexTable=None):
		'''
		Only allow BytesFmllr,NumpyFmllr,IndexTable or bytes (do not extend to their subclasses and their parent-classes).
		'''		
		declare.is_classes("data",data,[BytesFmllr,NumpyFmllr,IndexTable,bytes])

		super().__init__(data,name,indexTable)

	@property
	def spks(self):
		'''
		Get all speaker IDs.

		Return:
		  a list. 
		'''
		return list(self.keys())

	def to_numpy(self):
		'''
		Transform fMLLR transform matrix to numpy format.

		Return:
			a NumpyFmllr object.
		'''
		result = super().to_numpy()
		return NumpyFmllr(result.data,name=result.name)

	def __add__(self,other):
		'''
		Plus operation between two fMLLR transform matrix objects.

		Args:
			<other>: a BytesFmllr or NumpyFmllr object.
		Return:
			a BytesFmllr object.
		'''
		declare.is_fmllr_matrix("other",other)

		result = super().__add__(other)

		return BytesFmllr(result.data,name=result.name,indexTable=result.indexTable)

	def subset(self,nHead=0,nTail=0,nRandom=0,chunks=1,keys=None):
		'''
		Subset data.
		The priority of mode is nHead > nTail > nRandom > chunks > keys.
		If you chose multiple modes,only the prior one will work.
		
		Args:
			<nHead>: get N head utterances.
			<nTail>: get N tail utterances.
			<nRandom>: sample N utterances randomly.
			<chunks>: split data into N chunks averagely.
			<keys>: pick out these utterances whose ID in keys.
		Return:
			a new BytesFmllr object or a list of new BytesFmllr objects.
		''' 
		result = super().subset(nHead,nTail,nRandom,chunks,keys)

		if isinstance(result,list):
			for index in range(len(result)):
				temp = result[index]
				result[index] = BytesFmllr(temp.data,temp.name,temp.indexTable)
		else:
			result = BytesFmllr(result.data,result.name,result.indexTable)

		return result

	def __call__(self,spk):
		'''
		Pick out an utterance.
		
		Args:
			<spk>: a string.
		Return:
			If existed,return a new BytesFmllr object.
		''' 
		result = super().__call__(spk)
		if result is not None:
			result = BytesFmllr(result.data,result.name,result.indexTable)
		return result
		
	def sort(self,by="spk",reverse=False):
		'''
		Sort utterances by speaker ID.

		Args:
			<by>: only one mode, "key"/"spk".
			<reverse>: If reverse,sort in descending order.

		Return:
			A new BytesFmllr object.
		''' 
		result = super().sort(by,reverse)
		return BytesFmllr(result.data,name=result.name,indexTable=result.indexTable)

## Base class: for Vector Data archives
class BytesVector(BytesArchive):
	'''
	A base class to hold Kaldi vector data such as alignment.  
	'''
	def __init__(self,data=b'',name="vec",indexTable=None):
		'''
		Args:
			<data>: If it's BytesMatrix or IndexTable object (or their subclasses),extra <indexTable> will not work.
					If it's NumpyMatrix or bytes object (or their subclasses),generate index table automatically if it is not provided.
		'''
		declare.belong_classes("data",data,[BytesVector,NumpyVector,IndexTable,bytes])

		needIndexTableFlag = True

		if isinstance(data,BytesVector):
			self.__dataIndex = data.indexTable
			self.__dataIndex.rename(name)
			data = data.data
			needIndexTableFlag = False
		
		elif isinstance(data ,IndexTable):
			data = data.fetch(arkType="vec",name=name)
			self.__dataIndex = data.indexTable
			data = data.data
			needIndexTableFlag = False

		elif isinstance(data,NumpyVector):
			data = (data.to_bytes()).data

		super().__init__(data,name)

		if needIndexTableFlag is True:
			if indexTable is None:
				self.__generate_index_table()
			else:
				declare.is_classes("indexTable",indexTable,IndexTable)
				self.__verify_index_table(indexTable)
	
	def __verify_index_table(self,indexTable):
		'''
		Check the format of provided index table.
		'''
		newIndexTable = indexTable.sort("startIndex")
		start = 0
		for uttID,indexInfo in newIndexTable.items():
			if indexInfo.startIndex != start:
				raise WrongDataFormat(f"Start index of {uttID} dose not match: expected {start} but got {indexInfo.startIndex}.")
			if indexInfo.filePath is not None:
				newIndexTable[uttID] = indexInfo._replace(filePath=None)
			start += indexInfo.dataSize
		
		newIndexTable.rename(self.name)
		self.__dataIndex = newIndexTable

	def __read_one_record(self,fp):
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
				return (None,None,None,None,None)
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
			frames = int(np.frombuffer(fp.read(4),dtype='int32',count=1)[0])
			if frames == 0:
				buf = b""
			else:
				bufferSize = frames * 5
				buf = fp.read(bufferSize)
		else:
			fp.close()
			raise WrongDataFormat("Miss binary symbol before utterance. We do not support read Kaldi archives with text format.")
		
		return (utt,4,frames,bufferSize,buf)

	def __generate_index_table(self):
		'''
		Genrate the index table.
		'''
		if self.is_void:
			return None
		else:
			# Index table will have the same name with BytesMatrix object.
			self.__dataIndex = IndexTable(name=self.name)
			start_index = 0
			with BytesIO(self.data) as sp:
				while True:
					(utt,dataSize,frames,bufSize,buf) = self.__read_one_record(sp)
					if utt is None:
						break
					oneRecordLen = len(utt) + 8 + bufSize
					self.__dataIndex[utt] = self.__dataIndex.spec(frames,start_index,oneRecordLen)
					start_index += oneRecordLen

	@property
	def indexTable(self):
		'''
		Get the index informat of utterances.
		
		Return:
			A IndexTable object.
		'''
		# Return deepcopied dict object.
		return copy.deepcopy(self.__dataIndex)

	def keys(self):
		'''
		Get all keys.
		
		Return:
			a list of all utterance IDs.
		'''
		if self.__dataIndex is None:
			return dict().keys()
		else:
			return self.__dataIndex.keys()

	@property
	def lens(self):
		'''
		Get the numbers of utterances.
		If you want to get the frames of each utterance,try:
						obj.indexTable 
		attribute.
		
		Return:
			a int value.
		'''
		lengths = 0
		if not self.is_void:
			lengths = len(self.indexTable)
		
		return lengths

	def check_format(self):
		'''
		Check if data has right Kaldi format.
		
		Return:
			If data is void,return False.
			If data has right format,return True,or raise Error.
		'''
		if self.is_void:
			return False

		with BytesIO(self.data) as sp:
			start = 0
			while True: 
				(utt,typeSize,frames,bufSize,buf) = self.__read_one_record(sp)
				if utt == None:
					break
				
				oneRecordLen = len(utt) + 8 + frames * 5
				# Update the index table.
				self.__dataIndex[utt] = self.__dataIndex.spec(frames,start,oneRecordLen)
				start += oneRecordLen			
					
		return True

	@property
	def dtype(self):
		'''
		Get the dtype of vector. In current ExKaldi,we only use vector is int32.
		'''
		if self.is_void:
			return None
		else:		
			return "int32"

	@property
	def dim(self):
		'''
		Get the dimensionality of this vector. Defaultly,it should be 1.
		'''
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
			sortedIndex = self.indexTable.sort(by="utt",reverse=False)
			with BytesIO(self.data) as sp:
				for utt,indexInfo in sortedIndex.items():
					sp.seek(indexInfo.startIndex)
					(utt,dataSize,frames,bufSize,buf) = self.__read_one_record(sp)
					vector = np.frombuffer(buf,dtype=[("size","int8"),("value","int32")],count=frames)
					vector = vector[:]["value"]
					newDict[utt] = vector

		return NumpyVector(newDict,name=self.name)
	
	def save(self,fileName,chunks=1,returnIndexTable=False):
		'''
		Save bytes data to file.

		Args:
			<fileName>: file name or file handle.
			<chunks>: If larger than 1,data will be saved to multiple files averagely. This would be invalid when <fileName> is a file handle.
			<returnIndexTable>: If True,return the index table containing the information of file path.
		
		Return:
			the path of saved files.
		'''
		declare.not_void( type_name(self),self)
		declare.is_valid_file_name_or_handle("fileName",fileName)
		declare.greater_equal("chunks",chunks,"minimum chunk",1)
		declare.is_bool("returnIndexTable",returnIndexTable)

		if isinstance(fileName,str):

			def save_chunk_data(chunkData,arkFileName,returnIndexTable):

				make_dependent_dirs(arkFileName,pathIsFile=True)
				with open(arkFileName,"wb") as fw:
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
				savedFiles = save_chunk_data(self,fileName,returnIndexTable)	
			
			else:
				dirName = os.path.dirname(fileName)
				fileName = os.path.basename(fileName)
				savedFiles = []
				chunkDataList = self.subset(chunks=chunks)
				newFileNamePattern = f"ck%0{len(str(chunks))}d_{fileName}"
				for i,chunkData in enumerate(chunkDataList):
					chunkFileName = os.path.join( dirName,newFileNamePattern%i )
					savedFiles.append( save_chunk_data(chunkData,chunkFileName,returnIndexTable) )

			return savedFiles
		
		else:
			fileName.truncate()
			fileName.write(self.data)
			fileName.seek(0)

			return fileName
		
	def __add__(self,other):
		'''
		The Plus operation between two objects.

		Args:
			<other>: a BytesVector or NumpyVector object.
		Return:
			a new BytesVector object.
		'''
		declare.belong_classes("other",other,[BytesVector,NumpyVector,IndexTable])

		if isinstance(other,NumpyVector):
			other = other.to_bytes()
		elif isinstance(other,IndexTable):
			keys = [ utt for utt in other.keys() if utt not in self.keys() ]
			other = other.fecth(arkType="vec",keys=keys)
		
		newName = f"plus({self.name},{other.name})"
		if self.is_void:
			result = copy.deepcopy(other)
			result.rename(newName)
			return result
		elif other.is_void:
			result = copy.deepcopy(self)
			result.rename(newName)
			return result

		newDataIndex = self.indexTable
		#lastIndexInfo = list(newDataIndex.sort(by="startIndex",reverse=True).values())[0]
		start = len(self.data)

		newData = []
		with BytesIO(other.data) as op:
			for utt,indexInfo in other.indexTable.items():
				if not utt in self.keys():
					op.seek( indexInfo.startIndex )
					data = op.read( indexInfo.dataSize )
					data_size = indexInfo.dataSize
					newDataIndex[utt] = newDataIndex.spec(indexInfo.frames,start,data_size)
					start += data_size
					newData.append(data)

		return BytesVector(b''.join([self.data,*newData]),name=newName,indexTable=newDataIndex)

	def __call__(self,utt):
		'''
		Pick out one record.
		
		Args:
			<utt>: a string.
		Return:
			If existed,return a new BytesVector object.
		''' 
		declare.is_valid_string("utt",utt)
		if self.is_void:
			return None

		utt = utt.strip()

		if utt not in self.keys():
			return None
		else:
			indexInfo = self.indexTable[utt]
			newName = f"pick({self.name},{utt})"
			newDataIndex = IndexTable(name=newName)
			with BytesIO(self.data) as sp:
				sp.seek( indexInfo.startIndex )
				data = sp.read( indexInfo.dataSize )

				newDataIndex[utt] =	indexInfo._replace(startIndex=0)
				result = BytesVector(data,name=newName,indexTable=newDataIndex)
			
			return result

	def subset(self,nHead=0,nTail=0,nRandom=0,chunks=1,keys=None):
		'''
		Subset data.
		The priority of mode is nHead > nTail > nRandom > chunks > keys.
		If you chose multiple modes,only the prior one will work.
		
		Args:
			<nHead>: get N head utterances.
			<nTail>: get N tail utterances.
			<nRandom>: sample N utterances randomly.
			<chunks>: split data into N chunks averagely.
			<keys>: pick out these utterances whose ID in keys.
		Return:
			a new BytesVector object or a list of new BytesVector objects.
		''' 
		declare.not_void(type_name(self),self)

		if nHead > 0:
			declare.is_positive_int("nHead",nHead)
			newName = f"subset({self.name},head {nHead})"
			newDataIndex = IndexTable(name=newName)
			totalSize = 0
			
			for utt,indexInfo in self.indexTable.items():
				newDataIndex[utt] = newDataIndex.spec(indexInfo.frames,totalSize,indexInfo.dataSize)
				totalSize += indexInfo.dataSize
				nHead -= 1
				if nHead <= 0:
					break
			
			with BytesIO(self.data) as sp:
				sp.seek(0)
				data = sp.read(totalSize)
	
			return BytesVector(data,name=newName,indexTable=newDataIndex)

		elif nTail > 0:
			declare.is_positive_int("nTail",nTail)
			newName = f"subset({self.name},tail {nTail})"
			newDataIndex = IndexTable(name=newName)

			tailNRecord = list(self.indexTable.items())[-nTail:]
			start_index = tailNRecord[0][1].startIndex

			totalSize = 0
			for utt,indexInfo in tailNRecord:
				newDataIndex[utt] = newDataIndex.spec(indexInfo.frames,totalSize,indexInfo.dataSize)
				totalSize += indexInfo.dataSize

			with BytesIO(self.data) as sp:
				sp.seek(start_index)
				data = sp.read(totalSize)
	
			return BytesVector(data,name=newName,indexTable=newDataIndex)

		elif nRandom > 0:
			declare.is_positive_int("nRandom",nRandom)

			if nRandom >= self.lens:
				newName = f"subset({self.name},random {self.lens})"
				return BytesVector(self,name=newName)

			randomNRecord = random.sample(list(self.indexTable.items()),k=nRandom)
			newName = f"subset({self.name},random {nRandom})"

			newDataIndex = IndexTable(name=newName)
			start_index = 0
			newData = []
			with BytesIO(self.data) as sp:
				for utt,indexInfo in randomNRecord:
					sp.seek(indexInfo.startIndex)
					newData.append( sp.read(indexInfo.dataSize) )

					newDataIndex[utt] = newDataIndex.spec(indexInfo.frames,start_index,indexInfo.dataSize)
					start_index += indexInfo.dataSize

			return BytesVector(b"".join(newData),name=newName,indexTable=newDataIndex)

		elif chunks > 1:
			declare.is_positive_int("chunks",chunks)
			uttLens = list(self.indexTable.items())
			allLens = len(uttLens)
			chunkLens = allLens//chunks
			if chunkLens == 0:
				chunks = allLens
				chunkLens = 1
				t = 0
				print(f"Warning: utterances is fewer than <chunks> so only {chunks} files will be saved.")
			else:
				t = allLens - chunkLens * chunks
			
			datas = []
			with BytesIO(self.data) as sp:                          
				sp.seek(0)
				start = 0
				for i in range(chunks):
					newName = f"subset({self.name},chunk {chunks}-{i})"
					newDataIndex = IndexTable(name=newName)
					if i < t:
						end = start + chunkLens + 1
					else:
						end = start + chunkLens
					chunkItems = uttLens[start:end]
					chunkLen = 0
					for utt,indexInfo in chunkItems:
						newDataIndex[utt] = newDataIndex.spec(indexInfo.frames,chunkLen,indexInfo.dataSize)
						chunkLen += indexInfo.dataSize
					chunkData = sp.read(chunkLen)
					
					datas.append( BytesVector(chunkData,name=newName,indexTable=newDataIndex) )
					start = end
			return datas

		elif keys != None:
			declare.is_classes("keys",keys,[list,tuple])
			declare.members_are_valid_strings("keys",keys)
			newName = f"subset({self.name},keys {len(keys)})"

			newData = []
			dataIndex = self.indexTable
			newDataIndex = IndexTable(name=newName)
			start_index = 0
			with BytesIO(self.data) as sp:
				for utt in keys:
					if utt in self.keys():
						indexInfo = dataIndex[utt]
						sp.seek( indexInfo.startIndex )
						newData.append( sp.read(indexInfo.dataSize) )

						newDataIndex[utt] = newDataIndex.spec(indexInfo.frames,start_index,indexInfo.dataSize)
						start_index += indexInfo.dataSize

			return BytesVector(b''.join(newData),name=newName,indexTable=newDataIndex)
		
		else:
			raise WrongOperation('Expected one of <nHead>,<nTail>,<nRandom>,<chunks> or <keys> is avaliable but all got the default value.')

	def sort(self,by="utt",reverse=False):
		'''
		Sort utterances by frames length or uttID

		Args:
			<by>: "frame"/"value" or "utt"/"key".
			<reverse>: If reverse,sort in descending order.

		Return:
			A new BytesVector object.
		''' 
		declare.is_instances("by",by,["utt","key","frame","value"])

		if by in ["key","utt"]:
			newDataIndex = self.indexTable.sort(by="key",reverse=reverse)
		else:
			newDataIndex = self.indexTable.sort(by="value",reverse=reverse)
		ordered = True
		for i,j in zip(self.indexTable.items(),newDataIndex.items()):
			if i != j:
				ordered = False
				break
		if ordered:
			return copy.deepcopy(self)

		with BytesIO(self.data) as sp:
			if sys.getsizeof(self.data) > 10**9:
				## If the data size is large,divide it into N chunks and save it to intermidiate file.
				with FileHandleManager as fhm:
					temp = fhm.create("wb+")
					chunkdata = []
					chunkSize = 50
					count = 0
					start_index = 0
					for utt,indexInfo in newDataIndex.items():
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
				for utt,indexInfo in newDataIndex.items():
					sp.seek( indexInfo.startIndex )
					newData.append( sp.read(indexInfo.dataSize) )

					newDataIndex[utt] = indexInfo._replace(startIndex=start_index)
					start_index += indexInfo.dataSize

				newData = b"".join(newData)

		return BytesVector(newData,name=self.name,indexTable=newDataIndex)		

## Subclass: for transition-ID alignment
class BytesAliTrans(BytesVector):
	'''
	Hold the alignment(transition ID) with Kaldi binary format.
	'''
	def __init__(self,data=b"",name="transitionID",indexTable=None):
		'''
		Only allow BytesAliTrans,NumpyAliTrans,IndexTable or bytes (do not extend to their subclasses and their parent-classes).
		'''
		declare.is_classes("data",data,[BytesAliTrans,NumpyAliTrans,IndexTable,bytes])

		super().__init__(data,name,indexTable)

	@property
	def utts(self):
		'''
		Get all utterance IDs.

		Return:
		  a list. 
		'''
		return list(self.keys())

	def to_numpy(self,aliType="transitionID",hmm=None):
		'''
		Transform alignment to numpy format.

		Args:
			<aliType>: If it is "transitionID",transform to transition IDs.
					  If it is "phoneID",transform to phone IDs.
					  If it is "pdfID",transform to pdf IDs.
			<hmm>: None,or hmm file or exkaldi HMM object.

		Return:
			a NumpyAliTrans or NumpyAliPhone or NumpyAliPdf object.
		'''
		declare.is_instances("aliType",aliType,["transitionID","pdfID","phoneID"])
		
		if self.is_void:
			if aliType == "transitionID":
				return NumpyAliTrans(name=self.name)
			elif aliType == "phoneID":
				return NumpyAliPhone(name=f"to_phone({self.name})")
			else:
				return NumpyAliPdf(name=f"to_pdf({self.name})")

		def transform(data,cmd):
			out,err,cod = run_shell_command(cmd,stdin="PIPE",stdout="PIPE",stderr="PIPE",inputs=data)
			if (isinstance(cod,int) and cod != 0) or out == b'':
				raise KaldiProcessError('Failed to transform alignment.',err.decode())
			else:
				result = {}
				sp = BytesIO(out)
				for line in sp.readlines():
					line = line.decode()
					line = line.strip().split()
					utt = line[0]
					matrix = np.array(line[1:],dtype=np.int32)
					result[utt] = matrix
				return result

		if aliType == "transitionID":
			result = super().to_numpy()
			return NumpyAliTrans(result.data,self.name)
		
		else:
			declare.kaldi_existed()
			declare.is_potential_hmm("hmm",hmm)

			with FileHandleManager() as fhm:
				
				if not isinstance(hmm,str):
					temp = fhm.create("wb+",suffix=".mdl")
					hmm.save(temp)
					hmm = temp.name

				if aliType == "phoneID":
					cmd = f"ali-to-phones --per-frame=true {hmm} ark:- ark,t:-"
					result = transform(self.data,cmd)
					newName = f"to_phone({self.name})"
					return NumpyAliPhone(result,newName)

				else:
					cmd = f"ali-to-pdf {hmm} ark:- ark,t:-"
					result = transform(self.data,cmd)
					newName = f"to_pdf({self.name})"
					return NumpyAliPdf(result,newName)

	def __add__(self,other):
		'''
		The Plus operation between two objects.

		Args:
			<other>: a BytesAliTrans or NumpyAliTrans object.
		Return:
			a new BytesAliTrans object.
		''' 
		declare.is_alignment("other",other)
		result = super().__add__(other)

		return BytesAliTrans(result.data,result.name,result.indexTable)

	def __call__(self,uttID):
		'''
		Pick out a record.
		
		Args:
			<uttID>: a string.
		Return:
			If existed,return a new BytesAliTrans object.
		''' 
		result = super().__call__(uttID)
		if result is not None:
			result = BytesAliTrans(result.data,result.name,result.indexTable)
		return result

	def subset(self,nHead=0,nTail=0,nRandom=0,chunks=1,keys=None):
		'''
		Subset data.
		The priority of mode is nHead > nTail > nRandom > chunks > keys.
		If you chose multiple modes,only the prior one will work.
		
		Args:
			<nHead>: get N head utterances.
			<nTail>: get N tail utterances.
			<nRandom>: sample N utterances randomly.
			<chunks>: split data into N chunks averagely.
			<keys>: pick out these utterances whose ID in keys.
			
		Return:
			a new BytesAliTrans object or a list of new BytesAliTrans objects.
		''' 
		result = super().subset(nHead,nTail,nRandom,chunks,keys)

		if isinstance(result,list):
			for index in range(len(result)):
				temp = result[index]
				result[index] = BytesAliTrans(temp.data,temp.name,temp.indexTable)
		else:
			result = BytesAliTrans(result.data,result.name,result.indexTable)

		return result

	def sort(self,by="utt",reverse=False):
		'''
		Sort utterances by frames length or utterance ID.

		Args:
			<by>: "frame"/"value" or "utt"/"key".
			<reverse>: If reverse,sort in descending order.

		Return:
			A new BytesAliTrans object.
		''' 
		result = super().sort(by,reverse)
		return BytesAliTrans(result.data,name=result.name,indexTable=result.indexTable)

'''NumpyArchive class group'''
'''Designed for Kaldi binary archive table (in Numpy Format)'''
## Base Class
class NumpyArchive:
	'''
	The base class of NumPy archives.
	'''
	def __init__(self,data={},name=None):
		if data is not None:
			declare.is_classes("data",data,dict)
		self.__data = data

		if name is None:
			self.__name = self.__class__.__name__
		else:
			declare.is_valid_string("name",name)
			self.__name = name	

	@property
	def is_void(self):
		'''
		Check whether this is a void object.

		Return:
			True or False
		'''
		if self.__data is None or len(self.__data) == 0:
			return True
		else:
			return False

	@property
	def data(self):
		'''
		Get it's inner data.

		Return:
			a dict object which each key is a string and value is a NumPy array.
		'''
		return self.__data.copy()

	def reset_data(self,data=None):
		'''
		Reset the data.

		Args:
			_data_: a dict object. If None,clear it.
		'''		
		if data is not None:
			declare.is_classes("data",data,dict)
		del self.__data
		self.__data = data

	@property
	def name(self):
		'''
		Get the name

		Return:
			a string.
		'''
		return self.__name

	def rename(self,newName):
		'''
		Rename it.

		Args:
			_name_: a string. 
		'''
		declare.is_valid_string("name",newName)
		self.__name = newName

	def items(self):
		'''
		Get a iterator of the items.

		Return:
			a iterator.
		'''
		return self.__data.items()	

	def keys(self):
		'''
		Get a iterator of the keys.

		Return:
			a iterator.
		'''
		return self.__data.keys()	

	def values(self):
		'''
		Get a iterator of the values.

		Return:
			a iterator.
		'''
		return self.__data.values()	

	@property
	def array(self):
		'''
		Get all arrays.

		Return:
			a list of arrays.
		'''
		return list(self.values())

	def __getitem__(self,key):
		'''
		Get an item.

		Args:
			key: a string.
		'''
		if key not in self.__data.keys():
			raise WrongOperation(f"No such key: {key}.")
		else:
			return copy.deepcopy(self.__data[key])
	
	def __setitem__(self,key,value):
		'''
		Set the items directly.

		Args:
			key: a string, the utterance or skpeaker ID.
			value: a 1-d or 2-d Numpy array.
		'''
		self.__data[key] = value

## Base Class: for Matrix Data Archives 
class NumpyMatrix(NumpyArchive):
	'''
	A base class for matrix data,such as feature,cmvn statistics,post probability.
	'''
	def __init__(self,data={},name="mat"):
		'''
		Args:
			<data>: BytesMatrix or IndexTable object or NumpyMatrix or dict object (or their subclasses)
		'''
		declare.belong_classes("data",data,[BytesMatrix,NumpyMatrix,IndexTable,dict])

		if isinstance(data,BytesMatrix):
			data = data.to_Numpy().data
		elif isinstance(data,IndexTable):
			data = data.fetch(arkType="mat").to_Numpy().data
		elif isinstance(data,NumpyMatrix):
			data = data.data

		super().__init__(data,name)

	@property
	def dtype(self):
		'''
		Get the data type of Numpy data.
		
		Return:
			A string,'float32','float64'.
		'''  
		_dtype = None
		if not self.is_void:
			_dtype = str(list(self.values())[0].dtype)
		return _dtype
	
	@property
	def dim(self):
		'''
		Get the data dimensions.
		
		Return:
			If data is void,return None,or return an int value.
		'''		
		_dim = None
		if not self.is_void:
			if len(list(self.values())[0].shape) <= 1:
				_dim = 0
			else:
				_dim = list(self.values())[0].shape[1]
		
		return _dim
		
	def to_dtype(self,dtype):
		'''
		Transform data type.

		Args:
			<dtype>: a string of "float","float32" or "float64". IF "float",it will be treated as "float32".

		Return:
			A new NumpyMatrix object.
		'''
		declare.is_instances("dtype",dtype,['float','float32','float64'])
		declare.not_void( type_name(self),self)

		if dtype == 'float': 
			dtype = 'float32'

		if self.dtype == dtype:
			newData = copy.deepcopy(self.data)
		else:
			newData = {}
			for key in self.keys():
				newData[key] = np.array(self.data[key],dtype=dtype)
		
		return NumpyMatrix(newData,name=self.name)
	
	def check_format(self):
		'''
		Check if data has right Kaldi format.
		
		Return:
			If data is void,return False.
			If data has right format,return True,or raise Error.
		'''
		if self.is_void:
			return False

		_dim = 'unknown'
		for utt in self.keys():

			declare.is_valid_string("key",utt)
			declare.is_classes("value",self.data[utt],np.ndarray)
			matrixShape = self.data[utt].shape

			if len(matrixShape) > 2:
				raise WrongDataFormat(f'Expected the shape of matrix is like [ frame length,dimension ] but got {matrixShape}.')
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

		newDataIndex = IndexTable(name=self.name)
		newData = []
		start_index = 0
		for utt in self.keys():
			matrix = self.data[utt]
			data = (utt+' ').encode()
			data += '\0B'.encode()
			if matrix.dtype == 'float32':
				data += 'FM '.encode()
			elif matrix.dtype == 'float64':
				data += 'DM '.encode()
			else:
				raise UnsupportedType(f'Expected "float32" or "float64" data,but got {matrix.dtype}.')
			
			data += '\04'.encode()
			data += struct.pack(np.dtype('uint32').char,matrix.shape[0])
			data += '\04'.encode()
			data += struct.pack(np.dtype('uint32').char,matrix.shape[1])
			data += matrix.tobytes()

			oneRecordLen = len(data)
			newDataIndex[utt] = newDataIndex.spec(matrix.shape[0],start_index,oneRecordLen)
			start_index += oneRecordLen

			newData.append(data)

		return BytesMatrix(b''.join(newData),name=self.name,indexTable=newDataIndex)

	def save(self,fileName,chunks=1):
		'''
		Save numpy data to file.

		Args:
			<fileName>: file name. Defaultly suffix ".npy" will be add to the name.
			<chunks>: If larger than 1,data will be saved to multiple files averagely.		

		Return:
			the path of saved files.
		'''
		declare.not_void( type_name(self),self)
		declare.is_valid_string("fileName",fileName)
		declare.greater_equal("chunks",chunks,"minimum chunk",1)
		fileName = fileName.strip()

		if not fileName.endswith('.npy'):
			fileName += '.npy'

		make_dependent_dirs(fileName,pathIsFile=True)
		if chunks == 1:    
			allData = tuple(self.data.items())
			np.save(fileName,allData)
			return fileName
		else:
			chunkDataList = self.subset(chunks=chunks)

			dirName = os.path.dirname(fileName)
			fileName = os.path.basename(fileName)

			savedFiles = []
			newFileNamePattern = f"ck%0{len(str(chunks))}d_"+fileName
			for i,chunkData in enumerate(chunkDataList):
				chunkFileName = os.path.join(dirName,newFileNamePattern%i)
				chunkData = tuple(self.data.items())
				np.save(chunkFileName,chunkData)
				savedFiles.append(chunkFileName)	
		
			return savedFiles

	def __add__(self,other):
		'''
		The Plus operation between two objects.

		Args:
			<other>: a BytesMatrix,NumpyMatrix or IndexTable (or their subclassed) object.

		Return:
			a new NumpyMatrix object.
		''' 
		declare.belong_classes("other",other,[BytesMatrix,NumpyMatrix,IndexTable])

		if isinstance(other,BytesMatrix):
			other = other.to_numpy()
		elif isinstance(other,IndexTable):
			keys = [ utt for utt in other.keys() if utt not in self.keys() ]
			other = other.fecth(arkType="mat",keys=keys).to_numpy()
		
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
		for utt in other.keys:
			try:
				temp[utt]
			except KeyError:
				temp[utt] = other.data[utt]

		return NumpyMatrix(data=temp,name=newName)

	def __call__(self,utt):
		'''
		Pick out the specified utterance.
		
		Args:
			<utt>: a string.

		Return:
			If existed,return a new NumpyMatrix object.
		''' 
		declare.is_valid_string("utt",utt)
		if self.is_void:
			return None

		utt = utt.strip()

		try:
			t = self.data[utt]
		except KeyError:
			return None
		else:
			newName = f"pick({self.name},{utt})"
			return NumpyMatrix({utt:t},newName)

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

	def subset(self,nHead=0,nTail=0,nRandom=0,chunks=1,keys=None):
		'''
		Subset data.
		The priority of mode is nHead > nTail > nRandom > chunks > keys.
		If you chose multiple modes,only the prior one will work.
		
		Args:
			<nHead>: get N head utterances.
			<nTail>: get N tail utterances.
			<nRandom>: sample N utterances randomly.
			<chunks>: split data into N chunks averagely.
			<keys>: pick out these utterances whose ID in keys.

		Return:
			a new NumpyMatrix object or a list of new NumpyMatrix objects.
		''' 
		declare.not_void(type_name(self),self)

		if nHead > 0:
			declare.is_positive_int("nHead",nHead)
			newDict = {}
			for utt in list(self.keys())[0:nHead]:
				newDict[utt]=self.data[utt]
			newName = f"subset({self.name},head {nHead})"
			return NumpyMatrix(newDict,newName)

		elif nTail > 0:
			declare.is_positive_int("nTail",nTail)
			newDict = {}
			for utt in list(self.keys())[-nTail:]:
				newDict[utt]=self.data[utt]
			newName = f"subset({self.name},tail {nTail})"
			return NumpyMatrix(newDict,newName)

		elif nRandom > 0:
			declare.is_positive_int("nRandom",nRandom)
			if nRandom >= self.lens:
				newName = f"subset({self.name},tail {self.lens})"
				newDict = self
			else:
				newDict = dict(random.sample(self.items,k=nRandom))
				newName = f"subset({self.name},tail {nRandom})"
			return NumpyMatrix(newDict,newName)

		elif chunks > 1:
			declare.is_positive_int("chunks",chunks)

			datas = []
			allLens = len(self.data)
			if allLens != 0:
				utts = list(self.keys())
				chunkLens = allLens//chunks
				if chunkLens == 0:
					chunks = allLens
					chunkLens = 1
					t = 0
				else:
					t = allLens - chunkLens * chunks

				start = 0
				for i in range(chunks):
					temp = {}
					if i < t:
						end = start + chunkLens + 1
					else:
						end = start + chunkLens
					chunkkeys = utts[start:end]
					for utt in chunkkeys:
						temp[utt]=self.data[utt]
					newName = f"subset({self.name},chunk {chunks}-{i})"
					datas.append( NumpyMatrix(temp,newName) )
					start = end
			return datas

		elif keys != None:
			declare.is_classes("keys",keys,[list,tuple])
			declare.members_are_valid_strings("keys",keys)
			newName = f"subset({self.name},keys {len(keys)})"

			newDict = {}
			for utt in keys:
				try:
					t = self.data[utt]
				except KeyError:
					continue
				else:
					newDict[utt] = t
			return NumpyMatrix(newDict,newName)
		
		else:
			raise WrongOperation('Expected one of <nHead>,<nTail>,<nRandom> or <chunks> is avaliable but all got default value.')

	def sort(self,by='key',reverse=False):
		'''
		Sort.

		Args:
			<by>: "utt"/"key"/"spk",or "frame"/"value". 
			<reverse>: If reverse,sort in descending order.

		Return:
			A new NumpyMatrix object.
		''' 
		declare.is_instances("by",by,["utt","key","spk","frame","value"])
		declare.is_bool("reverse",reverse)

		if by in ["utt","spk","key"]:
			items = sorted(self.items(),key=lambda x:x[0],reverse=reverse)
		else:
			items = sorted(self.items(),key=lambda x:len(x[1]),reverse=reverse)
		
		newName = "sort({},{})".format(self.name,by)
		return NumpyMatrix(dict(items),newName)

	def map(self,func):
		'''
		Map all arrays to a function.

		Args:
			<func>: callable function object.
		
		Return:
			A new NumpyMatrix object.
		'''
		declare.is_callable("func",func)

		new = dict(map(lambda x:(x[0],func(x[1])),self.data.items()))

		return NumpyMatrix(new,name=f"mapped({self.name})")

	def __setitem__(self,key,value):
		'''
		Set an item.

		Args:
			key: a string.
			value: a Numpy array.
		'''
		declare.is_valid_string("key",key,debug=f"The key of NumpyMatrix must be a string such as utterance ID or speaker ID but got: {key}.")
		assert isinstance(value,np.ndarray) and len(value.shape) == 2, f"The value of NumpyMatrix must be a 2-d Numpy array."
		if not self.is_void:
			assert value.shape[1] == self.dim, f"Cannot set item because it has a unexpected dimmension: {value.shape[1]} != {self.dim}."

		if value.dtype != self.dtype:
			value = np.array(value,dtype=self.dtype)
		
		super().__setitem__(key,value)

## Subclass: for acoustic feature
class NumpyFeat(NumpyMatrix):
	'''
	Hold the feature with Numpy format.
	'''
	def __init__(self,data={},name="feat"):
		'''
		Only allow BytesFeat,NumpyFeat,IndexTable or dict (do not extend to their subclasses and their parent-classes).
		'''		
		declare.is_classes("data",data,[BytesFeat,NumpyFeat,IndexTable,dict])

		super().__init__(data,name)

	@property
	def utts(self):
		'''
		Get all utterance IDs.

		Return:
		  a list. 
		'''
		return list(self.keys())	

	def to_dtype(self,dtype):
		'''
		Transform data type.

		Args:
			<dtype>: a string of "float","float32" or "float64". IF "float",it will be treated as "float32".

		Return:
			A new NumpyFeat object.
		'''
		result = super().to_dtype(dtype)

		return NumpyFeat(result.data,result.name)

	def to_bytes(self):
		'''
		Transform feature to bytes format.

		Return:
			a BytesFeat object.
		'''		
		result = super().to_bytes()
		return BytesFeat(result.data,self.name,result.indexTable)
	
	def __add__(self,other):
		'''
		Plus operation between two feature objects.

		Args:
			<other>: a BytesFeat or NumpyFeat object.

		Return:
			a NumpyFeat object.
		'''
		declare.is_feature("other",other)

		result = super().__add__(other)

		return NumpyFeat(result.data,result.name)

	def __call__(self,key):
		'''
		Pick out the specified record.
		
		Args:
			<key>: a string.

		Return:
			If existed,return a new NumpyFeat object.
		''' 
		result = super().__call__(key)
		if result is not None:
			result = NumpyFeat(data=result.data,name=result.name)
		return result

	def splice(self,left=4,right=None):
		'''
		Splice front-behind N frames to generate new feature data.

		Args:
			<left>: the left N-frames to splice.
			<right>: the right N-frames to splice. If None,right = left.

		Return:
			a new NumpyFeat object whose dim became original-dim * (1 + left + right).
		''' 
		declare.not_void(type_name(self),self)
		declare.is_non_negative_int("left",left)

		if right is None:
			right = left
		else:
			declare.is_non_negative_int("right",right)

		lengths = []
		matrixes = []
		for utt in self.keys():
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
		return NumpyFeat(newFea,newName)
	
	def select(self,dims,retain=False):
		'''
		Select specified dimensions of feature.

		Args:
			<dims>: An int value or string such as "1,2,5-10"
			<retain>: If True,return the rest dimensions of feature simultaneously.

		Return:
			A new NumpyFeat object or two NumpyFeat objects.
		''' 
		declare.not_void(type_name(self),self)
		declare.is_bool("retain",retain)

		_dim = self.dim
		if isinstance(dims,int):
			declare.in_boundary("dims",dims,minV=0,maxV=_dim-1)
			selectFlag = [dims,]

		elif isinstance(dims,str):
			declare.is_valid_string("dims",dims)
			temp = dims.strip().split(',')
			selectFlag = []
			for i in temp:
				if not '-' in i:
					try:
						i = int(i)
					except ValueError:
						raise WrongOperation(f'Expected int value but got {i}.')
					else:
						declare.in_boundary("dims",i,minV=0,maxV=_dim-1)
						selectFlag.append( i )
				else:
					i = i.split('-')
					assert len(i) == 2,f"<dims> should has format like '1-3','4-','-5'."
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
							i[0],i[1] = i[1],i[0]
						declare.less_than("dims",i[1],"maximum dimension",_dim-1)
						selectFlag.extend([x for x in range(int(i[0]),int(i[1])+1)])
		else:
			raise WrongOperation(f'Expected <dims> is int value or string like 1,4-9,12 but got {type_name(dims)}.')

		retainFlag = sorted(list(set(selectFlag)))

		seleDict = {}
		if retain:
			reseDict = {}

		for utt in self.keys():
			newMat = []
			for index in selectFlag:
				newMat.append(self.data[utt][:,index][:,None])
			newMat = np.concatenate(newMat,axis=1)
			seleDict[utt] = newMat
			if retain:
				if len(retainFlag) == _dim:
					continue
				else:
					matrix = self.data[utt].copy()
					reseDict[utt] = np.delete(matrix,retainFlag,1)
		newNameSele = f"select({self.name},{dims})"
		if retain:
			newNameRese = f"select({self.name},not {dims})"
			return NumpyFeat(seleDict,newNameSele),NumpyFeat(reseDict,newNameRese)
		else:
			return NumpyFeat(seleDict,newNameSele)

	def subset(self,nHead=0,nTail=0,nRandom=0,chunks=1,keys=None):
		'''
		Subset data.
		The priority of mode is nHead > nTail > nRandom > chunks > keys.
		If you chose multiple modes,only the prior one will work.
		
		Args:
			<nHead>: get N head utterances.
			<nTail>: get N tail utterances.
			<nRandom>: sample N utterances randomly.
			<chunks>: split data into N chunks averagely.
			<keys>: pick out these utterances whose ID in keys.

		Return:
			a new NumpyFeat object or a list of new NumpyFeat objects.
		''' 
		result = super().subset(nHead,nTail,nRandom,chunks,keys)

		if isinstance(result,list):
			for index in range(len(result)):
				temp = result[index]
				result[index] = NumpyFeat(temp.data,temp.name)
		else:
			result = NumpyFeat(result.data,result.name)

		return result

	def sort(self,by='utt',reverse=False):
		'''
		Sort.

		Args:
			<by>: "frame"/"value" or "utt"/"key".
			<reverse>: If reverse,sort in descending order.

		Return:
			A new NumpyFeat object.
		''' 		
		result = super().sort(by,reverse)

		return NumpyFeat(result.data,result.name)

	def normalize(self,std=True,alpha=1.0,beta=0.0,epsilon=1e-8,axis=0):
		'''
		Standerd normalize a feature at a file field.
		If std is True,Do: 
					alpha * (x-mean)/(stds + epsilon) + belta,
		or do: 
					alpha * (x-mean) + belta.

		Args:
			<std>: True of False.
			<alpha>,<beta>: a float value.
			<epsilon>: a extremely small float value.
			<axis>: the dimension to normalize.
		
		Return:
			A new NumpyFeat object.
		'''
		declare.not_void(type_name(self),self)
		declare.is_bool("std",std)
		declare.is_positive("alpha",alpha)
		declare.is_classes("belta",beta,[float,int])
		declare.is_positive_float("epsilon",epsilon)
		declare.is_classes("axis",axis,int)

		utts = []
		lens = []
		data = []
		for uttID,matrix in self.data.items():
			utts.append(uttID)
			lens.append(len(matrix))
			data.append(matrix)

		data = np.row_stack(data)
		mean = np.mean(data,axis=axis)

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
		return NumpyFeat(newDict,newName) 

	def cut(self,maxFrames):
		'''
		Cut long utterance to multiple shorter ones. 

		Args:
			<maxFrames>: a positive int value. Cut a utterance by the thresold value of 1.25*maxFrames into a maxFrames part and a rest part.
						If the rest part is still longer than 1.25*maxFrames,continue to cut it. 
			
		Return:
			A new NumpyFeat object.
		''' 
		declare.not_void(type_name(self),self)
		declare.is_positive_int("maxFrames",maxFrames)

		newData = {}
		cutThreshold = maxFrames + maxFrames//4

		for utt in self.keys():
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
		return NumpyFeat(newData,newName)

	def paste(self,others):
		'''
		Concatenate feature arrays of the same utterance ID from multiple objects in feature dimention.

		Args:
			<others>: an object or a list of objects of NumpyFeat or BytesFeat or IndexTable.

		Return:
			a new NumpyFeat objects.
		'''
		if not isinstance(others,(list,tuple)):
			others = [others,]

		for index,other in enumerate(others):
			declare.is_feature("others",other)
			if isinstance(other,BytesFeat):
				others[index] = other.to_numpy()    
			elif isinstance(other,IndexTable):
				others[index] = other.fetch("feat").to_numpy()  

		newDict = {}
		for utt in self.keys():
			newMat=[]
			newMat.append(self.data[utt])
			frames = self.data[utt].shape[0]
			dim = self.dim
			
			for index,other in enumerate(others,start=1):
				if utt in other.keys:
					if other.data[utt].shape[0] != frames:
						raise WrongDataFormat(f"Data frames {frames}!={other[utt].shape[0]} at utterance ID {utt}.")
					newMat.append(other.data[utt])
				else:
					#print("Concat Warning: Miss data of utt id {} in later dict".format(utt))
					break

			if len(newMat) < len(others) + 1:
				#If any member miss the data of current utt id,abandon data of this utt id of all menbers
				continue

			newDict[utt] = np.column_stack(newMat)
		
		newName = f"paste({self.name}"
		for other in others:
			newName += f",{other.name}"
		newName += ")"

		return NumpyFeat(newDict,newName)

	def map(self,func):
		'''
		Map all arrays to a function.

		Args:
			<func>: callable function object.
		
		Return:
			A new NumpyFeat object.
		'''
		result = super().map(func)
		return NumpyFeat(data=result.data,name=result.name)	

## Subclass: for CMVN statistics
class NumpyCMVN(NumpyMatrix):
	'''
	Hold the CMVN statistics with Numpy format.
	'''
	def __init__(self,data={},name="cmvn"):
		'''
		Only allow BytesCMVN,NumpyCMVN,IndexTable or dict (do not extend to their subclasses and their parent-classes).
		'''	
		declare.is_classes("data",data,[BytesCMVN,NumpyCMVN,IndexTable,dict])

		super().__init__(data,name)

	@property
	def spks(self):
		'''
		Get all speakers IDs.

		Return:
		  a list. 
		'''
		return list(self.keys())

	def to_bytes(self):
		'''
		Transform feature to bytes format.

		Return:
			a BytesCMVN object.
		'''			
		result = super().to_bytes()
		return BytesCMVN(result.data,result.name)

	def to_dtype(self,dtype):
		'''
		Transform data type.

		Args:
			<dtype>: a string of "float","float32" or "float64". IF "float",it will be treated as "float32".

		Return:
			A new NumpyCMVN object.
		'''
		result = super().to_dtype(dtype)

		return NumpyCMVN(result.data,result.name)

	def __add__(self,other):
		'''
		Plus operation between two CMVN statistics objects.

		Args:
			<other>: a NumpyCMVN,BytesCMVN or IndexTable object.

		Return:
			a NumpyCMVN object.
		'''	
		declare.is_cmvn("other",other)

		result = super().__add__(other)

		return NumpyCMVN(result.data,result.name)

	def subset(self,nHead=0,nTail=0,nRandom=0,chunks=1,keys=None):
		'''
		Subset data.
		The priority of mode is nHead > nTail > nRandom > chunks > keys.
		If you chose multiple modes,only the prior one will work.
		
		Args:
			<nHead>: get N head utterances.
			<nTail>: get N tail utterances.
			<nRandom>: sample N utterances randomly.
			<chunks>: split data into N chunks averagely.
			<keys>: pick out these utterances whose ID in keys.

		Return:
			a new NumpyCMVN object or a list of new NumpyCMVN objects.
		''' 
		result = super().subset(nHead,nTail,nRandom,chunks,keys)

		if isinstance(result,list):
			for index in range(len(result)):
				temp = result[index]
				result[index] = NumpyCMVN(temp.data,temp.name)
		else:
			result = NumpyCMVN(result.data,result.name)

		return result

	def sort(self,by='spk',reverse=False):
		'''
		Sort.

		Args:
			<by>: "spk"/"key".
			<reverse>: If reverse,sort in descending order.

		Return:
			A new NumpyCMVN object.
		'''
		declare.is_instances("by",by,["key","spk"])

		result = super().sort(by,reverse)

		return NumpyCMVN(result.data,result.name)

	def __call__(self,key):
		'''
		Pick out the specified utterance.
		
		Args:
			<key>: a string.
		
		Return:
			If existed,return a new NumpyCMVN object.
		''' 
		result = super().__call__(key)
		if result is not None:
			result = NumpyCMVN(data=result.data,name=result.name)
		return result

	def map(self,func):
		'''
		Map all arrays to a function.

		Args:
			<func>: callable function object.
		
		Return:
			A new NumpyCMVN object.
		'''
		result = super().map(func)
		return NumpyCMVN(data=result.data,name=result.name)	

## Subclass: for fMLLR transform matrix
class NumpyFmllr(NumpyMatrix):
	'''
	Hold the fMLLR transform matrix with Numpy format.
	'''
	def __init__(self,data={},name="fmllrMat"):
		'''
		Only allow BytesFmllr,NumpyFmllr,IndexTable or dict (do not extend to their subclasses and their parent-classes).
		'''		
		declare.is_classes("data",data,[BytesFmllr,NumpyFmllr,IndexTable,dict])

		super().__init__(data,name)

	@property
	def spks(self):
		'''
		Get all speakers IDs.

		Return:
		  a list. 
		'''
		return list(self.keys())

	def to_bytes(self):
		'''
		Transform feature to bytes format.

		Return:
			a BytesFmllr object.
		'''			
		result = super().to_bytes()
		return BytesFmllr(result.data,result.name)

	def to_dtype(self,dtype):
		'''
		Transform data type.

		Args:
			<dtype>: a string of "float","float32" or "float64". IF "float",it will be treated as "float32".

		Return:
			A new NumpyFmllr object.
		'''
		result = super().to_dtype(dtype)

		return NumpyFmllr(result.data,result.name)

	def __add__(self,other):
		'''
		Plus operation between two fMLLR transform matrix objects.

		Args:
			<other>: a NumpyFmllr,BytesFmllr or IndexTable object.

		Return:
			a NumpyFmllr object.
		'''	
		declare.is_fmllr_matrix("other",other)

		result = super().__add__(other)

		return NumpyFmllr(result.data,result.name)

	def subset(self,nHead=0,nTail=0,nRandom=0,chunks=1,keys=None):
		'''
		Subset data.
		The priority of mode is nHead > nTail > nRandom > chunks > keys.
		If you chose multiple modes,only the prior one will work.
		
		Args:
			<nHead>: get N head utterances.
			<nTail>: get N tail utterances.
			<nRandom>: sample N utterances randomly.
			<chunks>: split data into N chunks averagely.
			<keys>: pick out these utterances whose ID in keys.

		Return:
			a new NumpyFmllr object or a list of new NumpyFmllr objects.
		''' 
		result = super().subset(nHead,nTail,nRandom,chunks,keys)

		if isinstance(result,list):
			for index in range(len(result)):
				temp = result[index]
				result[index] = NumpyFmllr(temp.data,temp.name)
		else:
			result = NumpyFmllr(result.data,result.name)

		return result

	def sort(self,by='spk',reverse=False):
		'''
		Sort.

		Args:
			<by>: "spk"/"key" or "frame"/"value".
			<reverse>: If reverse,sort in descending order.

		Return:
			A new NumpyFmllr object.
		''' 		
		result = super().sort(by,reverse)

		return NumpyFmllr(result.data,result.name)

	def __call__(self,key):
		'''
		Pick out the specified utterance.
		
		Args:
			<key>: a string.

		Return:
			None or a NumpyFmllr object.
		''' 
		result = super().__call__(key)
		if result is not None:
			result = NumpyFmllr(data=result.data,name=result.name)
		return result

	def map(self,func):
		'''
		Map all arrays to a function.

		Args:
			<func>: callable function object.
		
		Return:
			A new NumpyFmllr object.
		'''
		result = super().map(func)
		return NumpyFmllr(data=result.data,name=result.name)	

## Subclass: for probability of neural network output
class NumpyProb(NumpyMatrix):
	'''
	Hold the probability with Numpy format.
	'''
	def __init__(self,data={},name="prob"):
		'''
		Only allow BytesProb,NumpyProb,IndexTable or dict (do not extend to their subclasses and their parent-classes).
		'''	
		declare.is_classes("data",data,[BytesProb,NumpyProb,IndexTable,dict])
		
		super().__init__(data,name)

	@property
	def utts(self):
		'''
		Get all utterance IDs.

		Return:
		  a list. 
		'''
		return list(self.keys())

	def to_bytes(self):
		'''
		Transform post probability to bytes format.

		Return:
			a BytesProb object.
		'''				
		result = super().to_bytes()
		return BytesProb(result.data,result.name)

	def to_dtype(self,dtype):
		'''
		Transform data type.

		Args:
			<dtype>: a string of "float","float32" or "float64". IF "float",it will be treated as "float32".

		Return:
			A new NumpyProb object.
		'''
		result = super().to_dtype(dtype)

		return NumpyProb(result.data,result.name)

	def __add__(self,other):
		'''
		Plus operation between two post probability objects.

		Args:
			<other>: a NumpyProb,BytesProb or IndexTable object.

		Return:
			a NumpyProb object.
		'''	
		declare.is_probability("other",other)

		result = super().__add__(other)

		return NumpyProb(result.data,result.name)

	def subset(self,nHead=0,nTail=0,nRandom=0,chunks=1,keys=None):
		'''
		Subset data.
		The priority of mode is nHead > nTail > nRandom > chunks > keys.
		If you chose multiple modes,only the prior one will work.
		
		Args:
			<nHead>: get N head utterances.
			<nTail>: get N tail utterances.
			<nRandom>: sample N utterances randomly.
			<chunks>: split data into N chunks averagely.
			<keys>: pick out these utterances whose ID in keys.

		Return:
			a new NumpyProb object or a list of new NumpyProb objects.
		''' 
		result = super().subset(nHead,nTail,nRandom,chunks,keys)

		if isinstance(result,list):
			for index in range(len(result)):
				temp = result[index]
				result[index] = NumpyProb(temp.data,temp.name)
		else:
			result = NumpyProb(result.data,result.name)

		return result

	def sort(self,by='utt',reverse=False):
		'''
		Sort utterances by frames length or uttID

		Args:
			<by>: "key"/"utt" or "value"/"frame".
			<reverse>: If reverse,sort in descending order.

		Return:
			A new NumpyProb object.
		'''	
		result = super().sort(by,reverse)

		return NumpyProb(result.data,result.name)

	def __call__(self,key):
		'''
		Pick out the specified utterance.
		
		Args:
			<key>: a string.

		Return:
			None or a NumpyProb object.
		''' 
		result = super().__call__(key)
		if result is not None:
			result = NumpyProb(data=result.data,name=result.name)
		return result

	def map(self,func):
		'''
		Map all arrays to a function.

		Args:
			<func>: callable function object.
		
		Return:
			A new NumpyProb object.
		'''
		result = super().map(func)
		return NumpyProb(data=result.data,name=result.name)	

## Base Class: for Vector Data Archives
class NumpyVector(NumpyMatrix):
	'''
	Hold the Kaldi vector data with Numpy format.
	'''
	def __init__(self,data={},name="vec"):
		'''
		Args:
			<data>: Bytesvector or IndexTable object or NumpyVector or dict object (or their subclasses).
			<name>: a string.
		'''
		declare.belong_classes("data",data,[BytesVector,NumpyVector,IndexTable,dict])

		if isinstance(data,BytesVector):
			data = data.to_Numpy().data
		elif isinstance(data,IndexTable):
			data = data.fetch(arkType="vec").to_Numpy().data
		elif isinstance(data,NumpyMatrix):
			data = data.data

		super().__init__(data,name)

	@property
	def dtype(self):
		'''
		Get the data type of Numpy data.
		
		Return:
			A string,like 'int32'.
		'''  
		_dtype = None
		if not self.is_void:
			_dtype = str(list(self.values())[0].dtype)
		return _dtype

	@property
	def dim(self):
		'''
		Get the dimmension.

		Return:
			None or 0.
		'''
		if self.is_void:
			return None
		else:
			return 0

	def to_dtype(self,dtype):
		'''
		Transform data type.

		Args:
			<dtype>: a string of "int","int16","int32" or "int64". IF "int",it will be treated as "int32".

		Return:
			A new NumpyMatrix object.
		'''
		declare.is_instances("dtype",dtype,['int','int16','int32','int64'])
		declare.not_void( type_name(self),self)

		if dtype == 'int': 
			dtype = 'int32'

		if self.dtype == dtype:
			newData = copy.deepcopy(self.data)
		else:
			newData = {}
			for utt in self.keys():
				newData[utt] = np.array(self.data[utt],dtype=dtype)
		
		return NumpyVector(data=newData,name=self.name)

	def check_format(self):
		'''
		Check if data has right Kaldi format.
		
		Return:
			If data is void,return False.
			If data has right format,return True,or raise Error.
		'''
		if not self.is_void:

			for key in self.keys():
				declare.is_valid_string("key",key)
				declare.is_classes("value",self.data[key],np.ndarray)
				vector = self.data[key]
				assert len(vector.shape) == 1,f"Vector should be 1-dim data but got {vector.shape}."
				assert vector.dtype in ["int16","int32","int64"],f"Only support int 16/32/64 data format but got {vector.dtype}."

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
			raise WrongDataFormat(f"Only int32 vector can be convert to bytes object in current ExKaldi but this is: {self.dtype}")

		newDataIndex = IndexTable(name=self.name)
		newData = []
		start_index = 0
		for utt,vector in self.items():
			oneRecord = []
			oneRecord.append( ( utt + ' ' + '\0B' + '\4' ).encode() )
			oneRecord.append( struct.pack(np.dtype('int32').char,vector.shape[0]) ) 
			for f,v in vector:
				oneRecord.append( '\4'.encode() + struct.pack(np.dtype('int32').char,v) )
			oneRecord = b"".join(oneRecord)
			newData.append( oneRecord )

			oneRecordLen = len(oneRecord)
			newDataIndex[utt] = newDataIndex.spec(vector.shape[0],start_index,oneRecordLen)
			start_index += oneRecordLen

		return BytesVector(b''.join(newData),name=self.name,indexTable=newDataIndex)

	def __add__(self,other):
		'''
		Plus operation between two vector objects.

		Args:
			<other>: a BytesVector,NumpyVector or IndexTable (or their subclass) object.

		Return:
			a new NumpyVector object.
		'''	
		declare.belong_classes("other",other,[BytesVector,NumpyVector,IndexTable])
		
		result = super().__add__(other)

		return NumpyVector(result.data,result.name)

	def subset(self,nHead=0,nTail=0,nRandom=0,chunks=1,keys=None):
		'''
		Subset data.
		The priority of mode is nHead > nTail > nRandom > chunks > keys.
		If you chose multiple modes,only the prior one will work.
		
		Args:
			<nHead>: get N head utterances.
			<nTail>: get N tail utterances.
			<nRandom>: sample N utterances randomly.
			<chunks>: split data into N chunks averagely.
			<keys>: pick out these utterances whose ID in keys.

		Return:
			a new NumpyVector object or a list of new NumpyVector objects.
		''' 
		result = super().subset(nHead,nTail,nRandom,chunks,keys)

		if isinstance(result,list):
			for index in range(len(result)):
				temp = result[index]
				result[index] = NumpyVector( temp.data,temp.name )
		else:
			result = NumpyVector( result.data,result.name )

		return result

	def sort(self,by='key',reverse=False):
		'''
		Sort utterances by frames length or uttID

		Args:
			<by>: "frame"/"value" or "utt"/"key".
			<reverse>: If reverse,sort in descending order.

		Return:
			A new NumpyVector object.
		'''	
		result = super().sort(by,reverse)

		return NumpyVector(result.data,result.name)

	def __call__(self,key):
		'''
		Pick out the specified utterance.
		
		Args:
			<key>: a string.

		Return:
			If existed,return a new NumpyVector object.
		''' 
		result = super().__call__(key)
		if result is not None:
			result = NumpyVector(data=result.data,name=result.name)
		return result	

	def map(self,func):
		'''
		Map all arrays to a function.

		Args:
			<func>: callable function object.
		
		Return:
			A new NumpyVector object.
		'''
		result = super().map(func)
		return NumpyVector(data=result.data,name=result.name)			

	def __setitem__(self,key,value):
		'''
		Set an item.

		Args:
			key: a string.
			value: a 1-d Numpy array.
		'''
		declare.is_valid_string("key",key,debug=f"The key of NumpyVector must be a string such as utterance ID or speaker ID but got: {key}.")
		assert isinstance(value,np.ndarray) and len(value.shape) == 1, f"The value of NumpyVector must be a 1-d Numpy array."

		assert value.dtype == int, "NumpyVector only accept int array."
		value = np.array(value,dtype=self.dtype)
		
		super().__setitem__(key,value)

## Subclass: for transition-ID alignment 			
class NumpyAliTrans(NumpyVector):
	'''
	Hold the alignment(transition ID) with Numpy format.
	'''
	def __init__(self,data={},name="transitionID"):
		'''
		Only allow BytesAliTrans,NumpyAliTrans,IndexTable or dict (do not extend to their subclasses and their parent-classes).
		'''	
		declare.is_classes("data",data,[BytesAliTrans,NumpyAliTrans,IndexTable,dict])
	
		super().__init__(data,name)

	@property
	def utts(self):
		'''
		Get all utterance IDs.

		Return:
		  a list. 
		'''
		return list(self.keys())	

	def to_bytes(self):
		'''
		Tansform numpy alignment to bytes format.

		Return:
			A BytesAliTrans object.
		'''
		result = super(NumpyAliTrans,self.to_dtype("int32")).to_bytes()

		return BytesAliTrans(data=result.data,name=self.name,indexTable=result.indexTable)

	def to_dtype(self,dtype):
		'''
		Transform data type.

		Args:
			<dtype>: a string of "int","int16","int32" or "int64". IF "int",it will be treated as "int32".
		Return:
			A new NumpyAli object.
		'''
		result = super().to_dtype(dtype)

		return NumpyAliTrans(result.data,result.name)

	def __add__(self,other):
		'''
		The Plus operation between two transition ID alignment objects.

		Args:
			<other>: a NumpyAliTrans or BytesAliTrans object.

		Return:
			a new NumpyAliTrans object.
		''' 
		declare.is_alignment("other",other)

		results = super().__add__(other)
		return NumpyAliTrans(results.data,results.name)

	def subset(self,nHead=0,nTail=0,nRandom=0,chunks=1,keys=None):
		'''
		Subset data.
		The priority of mode is nHead > nTail > nRandom > chunks > keys.
		If you chose multiple modes,only the prior one will work.
		
		Args:
			<nHead>: get N head utterances.
			<nTail>: get N tail utterances.
			<nRandom>: sample N utterances randomly.
			<chunks>: split data into N chunks averagely.
			<keys>: pick out these utterances whose ID in keys.

		Return:
			a new NumpyAliTrans object or a list of new NumpyAliTrans objects.
		''' 
		result = super().subset(nHead,nTail,nRandom,chunks,keys)

		if isinstance(result,list):
			for index in range(len(result)):
				temp = result[index]
				result[index] = NumpyAliTrans(temp.data,temp.name)
		else:
			result = NumpyAliTrans(result.data,result.name)

		return result
	
	def sort(self,by='utt',reverse=False):
		'''
		Sort utterances by frames length or uttID

		Args:
			<by>: "frame"/"value" or "utt"/"key".
			<reverse>: If reverse,sort in descending order.
		
		Return:
			A new NumpyAliTrans object.
		''' 			
		result = super().sort(by,reverse)

		return NumpyAliTrans(result.data,result.name)

	def __to_phone_or_pdf(self,tool,hmm):
		'''
		Transform to phone ID alignment or pdf ID alignment.
		'''
		if self.is_void:
			return {}

		declare.kaldi_existed()
		declare.is_potential_hmm("hmm",hmm)

		temp = []
		for utt,matrix in self.data.items():
			new = utt + " ".join(map(str,matrix.tolist()))
			temp.append( new )
		temp = ("\n".join(temp)).encode()

		with FileHandleManager() as fhm:
			
			if not isinstance(hmm,str):
				hmmTemp = fhm.create("wb+",suffix=".hmm")
				hmm.save(hmmTemp)
				hmm = hmmTemp.name

			cmd = f'copy-int-vector ark:- ark:- | {tool} {hmm} ark:- ark,t:-'
			out,err,cod = run_shell_command(cmd,stdin="PIPE",stdout="PIPE",stderr="PIPE",inputs=temp)
			if (isinstance(cod,int) and cod != 0) or out == b'':
				raise KaldiProcessError('Failed to transform alignment.',err.decode())
			else:
				result = {}
				sp = BytesIO(out)
				for line in sp.readlines():
					line = line.decode()
					line = line.strip().split()
					utt = line[0]
					matrix = np.array(line[1:],dtype=np.int32)
					result[utt] = matrix
				
				return result

	def to_phoneID(self,hmm):
		'''
		Transform tansition ID alignment to phone ID alignment.

		Args:
			<hmm>: exkaldi HMM object or file path.
		Return:
			a NumpyAliPhone object.
		'''		
		result = self.__to_phone_or_pdf(tool="ali-to-phones --per-frame=true",hmm=hmm)

		return NumpyAliPhone(result,name=f"to_phone({self.name})")

	def to_pdfID(self,hmm):
		'''
		Transform tansition ID alignment to pdf ID alignment.

		Args:
			<hmm>: exkaldi HMM object or file path.

		Return:
			a NumpyAliPdf object.
		'''		
		result = self.__to_phone_or_pdf(tool="ali-to-pdf",hmm=hmm)

		return NumpyAliPdf(result,name=f"to_pdf({self.name})")

	def __call__(self,key):
		'''
		Pick out the specified record.
		
		Args:
			<key>: a string.

		Return:
			If existed,return a new NumpyAliTrans object.
		''' 
		result = super().__call__(key)
		if result is not None:
			result = NumpyAliTrans(data=result.data,name=result.name)
		return result	

	def map(self,func):
		'''
		Map a function to all arrays.

		Args:
			<func>: a callable object or function.
		
		Return:
			A new NumpyAliTrans object.
		'''
		result = super().map(func)
		return NumpyAliTrans(data=result.data,name=result.name)	

	def cut(self,maxFrames):
		'''
		Cut long utterance to multiple shorter ones. 

		Args:
			<maxFrames>: a positive int value. Cut a utterance by the thresold value of 1.25*maxFrames into a maxFrames part and a rest part.
						If the rest part is still longer than 1.25*maxFrames,continue to cut it. 

		Return:
			A new NumpyAliTrans object.
		''' 
		declare.not_void(type_name(self),self)	
		declare.is_positive_int("maxFrames",maxFrames)

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
		return NumpyAliTrans(newData,newName)

## Subclass: for user customized alignment 
class NumpyAli(NumpyVector):
	'''
	Hold the alignment with Numpy format.
	'''
	def __init__(self,data={},name="ali"):
		'''
		Args:
			<data>: NumpyAli or dict object (or their subclasses).
		'''
		declare.belong_classes("data",data,[NumpyAli,dict])

		if isinstance(data,NumpyAli):
			data = data.data
				
		super().__init__(data,name)

	@property
	def utts(self):
		'''
		Get all utterance IDs.

		Return:
		  a list. 
		'''
		return list(self.keys())	

	def to_dtype(self,dtype):
		'''
		Transform data type.

		Args:
			<dtype>: a string of "int","int16","int32" or "int64". IF "int",it will be treated as "int32".

		Return:
			A new NumpyAli object.
		'''
		result = super().to_dtype(dtype)

		return NumpyAli(result.data,result.name)

	def to_bytes(self):
		'''
		Customized alignment can not be converted to bytes format.
		'''
		raise WrongOperation("Cannot convert this alignment to bytes.")
	
	def __add__(self,other):
		'''
		Plus operation between two alignment objects.

		Args:
			<other>: a NumpyAli object.

		Return:
			a NumpyAli object.
		'''	
		declare.belong_classes("other",other,NumpyAli)
		
		result = super().__add__(other)
		return NumpyAli(result.data,result.name)

	def subset(self,nHead=0,nTail=0,nRandom=0,chunks=1,keys=None):
		'''
		Subset data.
		The priority of mode is nHead > nTail > nRandom > chunks > keys.
		If you chose multiple modes,only the prior one will work.
		
		Args:
			<nHead>: get N head utterances.
			<nTail>: get N tail utterances.
			<nRandom>: sample N utterances randomly.
			<chunks>: split data into N chunks averagely.
			<keys>: pick out these utterances whose ID in keys.

		Return:
			a new NumpyAli object or a list of new NumpyAli objects.
		''' 
		result = super().subset(nHead,nTail,nRandom,chunks,keys)

		if isinstance(result,list):
			for index in range(len(result)):
				temp = result[index]
				result[index] = NumpyAli( temp.data,temp.name )
		else:
			result = NumpyAli( result.data,result.name )

		return result

	def sort(self,by='utt',reverse=False):
		'''
		Sort utterances by frames length or uttID

		Args:
			<by>: "key"/"utt" or "value"/"frame"
			<reverse>: If reverse,sort in descending order.

		Return:
			A new NumpyAli object.
		''' 			
		result = super().sort(by,reverse)

		return NumpyAli(result.data,result.name)

	def __call__(self,key):
		'''
		Pick out the specified utterance.
		
		Args:
			<key>: a string.

		Return:
			If existed,return a new NumpyAli object.
			Or return None.
		''' 
		result = super().__call__(key)
		if result is not None:
			result = NumpyAli(data=result.data,name=result.name)
		return result	

	def map(self,func):
		'''
		Map all arrays to a function.

		Args:
			<func>: callable function object.
		
		Return:
			A new NumpyAli object.
		'''
		result = super().map(func)
		return NumpyAli(data=result.data,name=result.name)	

	def cut(self,maxFrames):
		'''
		Cut long utterance to multiple shorter ones. 

		Args:
			<maxFrames>: a positive int value. Cut a utterance by the thresold value of 1.25*maxFrames into a maxFrames part and a rest part.
						If the rest part is still longer than 1.25*maxFrames,continue to cut it. 

		Return:
			A new NumpyAli object.
		''' 
		declare.not_void(type_name(self),self)	
		declare.is_positive_int("maxFrames",maxFrames)

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
		return NumpyAli(newData,newName)

## Subclass: for phone-ID alignment 
class NumpyAliPhone(NumpyAli):
	'''
	Hold the alignment(phone ID) with Numpy format.
	'''
	def __init__(self,data={},name="phoneID"):
		'''
		Only allow NumpyAliPhone or dict (do not extend to their subclasses and their parent-classes).
		'''			
		declare.is_classes("data",data,[NumpyAliPhone,dict])

		if isinstance(data,NumpyAliPhone):
			data = data.data
				
		super().__init__(data,name)		

	def to_dtype(self,dtype):
		'''
		Transform data type.

		Args:
			<dtype>: a string of "int","int16","int32" or "int64". If "int",it will be treated as "int32".
		Return:
			A new NumpyAliPhone object.
		'''
		result = super().to_dtype(dtype)

		return NumpyAliPhone(result.data,result.name)

	def __add__(self,other):
		'''
		The plus operation between two phone ID alignment objects.

		Args:
			<other>: a NumpyAliPhone object.

		Return:
			a new NumpyAliPhone object.
		''' 
		declare.is_classes("other",other,NumpyAliPhone)

		results = super().__add__(other)
		return NumpyAliPhone(results.data,results.name)

	def subset(self,nHead=0,nTail=0,nRandom=0,chunks=1,keys=None):
		'''
		Subset data.
		The priority of mode is nHead > nTail > nRandom > chunks > keys.
		If you chose multiple modes,only the prior one will work.
		
		Args:
			<nHead>: get N head utterances.
			<nTail>: get N tail utterances.
			<nRandom>: sample N utterances randomly.
			<chunks>: split data into N chunks averagely.
			<keys>: pick out these utterances whose ID in keys.

		Return:
			a new NumpyAliPhone object or a list of new NumpyAliPhone objects.
		''' 
		result = super().subset(nHead,nTail,nRandom,chunks,keys)

		if isinstance(result,list):
			for index in range(len(result)):
				temp = result[index]
				result[index] = NumpyAliPhone(temp.data,temp.name)
		else:
			result = NumpyAliPhone(result.data,result.name)

		return result

	def __call__(self,key):
		'''
		Pick out the specified utterance.
		
		Args:
			<key>: a string.

		Return:
			If existed,return a new NumpyAliPhone object.
		''' 
		result = super().__call__(key)
		if result is not None:
			result = NumpyAliPhone(data=result.data,name=result.name)
		return result	

	def sort(self,by='utt',reverse=False):
		'''
		Sort alignment.

		Args:
			<by>: "key"/"utt" or "value"/"frame" 
			<reverse>: If reverse,sort in descending order.
		
		Return:
			A new NumpyAliPhone object.
		''' 			
		result = super().sort(by,reverse)

		return NumpyAliPhone(result.data,result.name)

	def map(self,func):
		'''
		Map all arrays to a function.

		Args:
			<func>: callable function object.
		
		Return:
			A new NumpyAliPhone object.
		'''
		result = super().map(func)
		return NumpyAliPhone(data=result.data,name=result.name)

	def cut(self,maxFrames):
		'''
		Cut long utterance to multiple shorter ones. 

		Args:
			<maxFrames>: a positive int value. Cut a utterance by the thresold value of 1.25*maxFrames into a maxFrames part and a rest part.
						If the rest part is still longer than 1.25*maxFrames,continue to cut it. 

		Return:
			A new NumpyAliPhone object.
		''' 
		result = super().cut(maxFrames)
		return NumpyAliPhone(result.data,result.name)

## Subclass: for pdf-ID alignment 
class NumpyAliPdf(NumpyAli):
	'''
	Hold the alignment(pdf ID) with Numpy format.
	'''
	def __init__(self,data={},name="phoneID"):
		'''
		Only allow NumpyAliPdf or dict (do not extend to their subclasses and their parent-classes).
		'''	
		declare.is_classes("data",data,[NumpyAliPdf,dict])

		if isinstance(data,NumpyAliPdf):
			data = data.data
				
		super().__init__(data,name)

	def to_dtype(self,dtype):
		'''
		Transform data type.

		Args:
			<dtype>: a string of "int","int16","int32" or "int64". IF "int",it will be treated as "int32".

		Return:
			A new NumpyAliPdf object.
		'''
		result = super().to_dtype(dtype)

		return NumpyAliPdf(result.data,result.name)

	def __add__(self,other):
		'''
		The Plus operation between two phone ID alignment objects.

		Args:
			<other>: a NumpyAliPdf object.
		Return:
			a new NumpyAliPdf object.
		''' 
		declare.is_classes("other",other,NumpyAliPdf)

		results = super().__add__(other)
		return NumpyAliPdf(results.data,results.name)

	def subset(self,nHead=0,nTail=0,nRandom=0,chunks=1,keys=None):
		'''
		Subset data.
		The priority of mode is nHead > nTail > nRandom > chunks > keys.
		If you chose multiple modes,only the prior one will work.
		
		Args:
			<nHead>: get N head utterances.
			<nTail>: get N tail utterances.
			<nRandom>: sample N utterances randomly.
			<chunks>: split data into N chunks averagely.
			<keys>: pick out these utterances whose ID in keys.

		Return:
			a new NumpyAliPdf object or a list of new NumpyAliPdf objects.
		''' 
		result = super().subset(nHead,nTail,nRandom,chunks,keys)

		if isinstance(result,list):
			for index in range(len(result)):
				temp = result[index]
				result[index] = NumpyAliPdf(temp.data,temp.name)
		else:
			result = NumpyAliPdf(result.data,result.name)

		return result

	def __call__(self,key):
		'''
		Pick out the specified utterance.
		
		Args:
			<key>: a string.

		Return:
			If existed,return a new NumpyAliPdf object.
			Or return None.
		''' 
		result = super().__call__(key)
		if result is not None:
			result = NumpyAliPdf(data=result.data,name=result.name)
		return result	

	def sort(self,by='utt',reverse=False):
		'''
		Sort alignment.

		Args:
			<by>: "key"/"utt" or "value"/"frame".
			<reverse>: If reverse,sort in descending order.
		
		Return:
			A new NumpyAliPdf object.
		''' 			
		result = super().sort(by,reverse)
		return NumpyAliPdf(result.data,result.name)

	def map(self,func):
		'''
		Map all arrays to a function.

		Args:
			<func>: callable function object.
		
		Return:
			A new NumpyAliPdf object.
		'''
		result = super().map(func)
		return NumpyAliPdf(data=result.data,name=result.name)
	
	def cut(self,maxFrames):
		'''
		Cut long utterances to shorter ones. 

		Args:
			<maxFrames>: a positive int value. Cut a utterance by the thresold value of 1.25*maxFrames into a maxFrames part and a rest part.
						If the rest part is still longer than 1.25*maxFrames,continue to cut it. 
		
		Return:
			A new NumpyAliPdf object.
		''' 
		result = super().cut(maxFrames)
		return NumpyAliPdf(result.data,result.name)