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
import struct
import subprocess
import os
import tempfile

import sys
sys.path.append('/mnt/c/Users/sanja/Documents/github/exkaldi')

from exkaldi.utils.utils import type_name, run_shell_command, make_dependent_dirs
from exkaldi.version import WrongPath, WrongOperation, WrongDataFormat, UnsupportedType, ShellProcessError, KaldiProcessError
from exkaldi.version import version as ExkaldiInfo

''' Kaldi script table '''
class ScripTable(dict):
	'''
	This is a subclass of Python dict and used to hold all kaldi list tables such as feature scp file, utt2spk file, transcription file and so on. 
	'''
	def __init__(self, data={}, name="table"):
		assert isinstance(name, str) and len(name) >0, "Name is not a string avaliable."
		self.__name = name
		super(ScripTable, self).__init__(data)
	
	@property
	def is_void(self):
		if len(self.keys()) == 0:
			return True
		else:
			return False

	@property
	def name(self):
		return copy.deepcopy(self.__name)

	def rename(self, name):
		'''
		Rename.

		Args:
			<name>: a string.
		'''
		assert isinstance(name, str) and len(name) >0, "Name is not a string avaliable."
		self.__name = name

	def sort(self, reverse=False):
		'''
		Sort by utterance ID.

		Args:
			<reverse>: If reverse, sort in descending order.
		Return:
			A new ListTable object.
		''' 
		if self.is_void:
			raise WrongOperation("No data to sort.")

		items = sorted(self.items(), key=lambda x:x[0], reverse=reverse)
		
		return ScripTable(items, name=self.name)

	def save(self, fileName=None):
		'''
		Save as text formation.

		Args:
			<fileName>: If None, return a string of text formation.
		'''
		def concat(utterance):
			try:
				return f"{utterance[0]} {utterance[1]}"
			except Exception:
				print(f"Utterance ID: {utterance[0]}")
				raise WrongDataFormat(f"Wrong key and value format: {type_name(utterance[0])} and {type_name(utterance[1])}. ")

		results = "\n".join( map(concat, self.items()) )

		if fileName is None:
			return results
		elif isinstance(fileName, tempfile._TemporaryFileWrapper):
			fileName.write(results)
		else:
			assert isinstance(fileName, str) and len(fileName) > 0, "file name is unavaliable."
			make_dependent_dirs(fileName, pathIsFile=True)
			with open(fileName, "w", encoding="utf-8") as fw:
				fw.write(results)

	def load(self, fileName):
		'''
		Load a transcription from file. If utt is existed, it will be overlaped.

		Args:
			<fileName>: the txt file path.
		'''
		assert isinstance(fileName, str) and len(fileName)>0, "file name is unavaliable."
		if not os.path.isfile(fileName):
			raise WrongPath(f"No such file:{fileName}.")

		with open(fileName, "r", encoding="utf-8") as fr:
			lines = fr.readlines()
		for index, line in enumerate(lines, start=1):
			le = line.strip().split(maxsplit=1)
			if len(le) < 2:
				print(f"Line Number:{index}")
				print(f"Line Content:{line}")
				raise WrongDataFormat("Missing entire uttID and utterance information.")
			else:
				self[le[0]] = le[1]

	def __add__(self, other):
		'''
		Integrate two ScripTable objects. If utt is existed in both two objects, the former will be retained.

		Args:
			<other>: another ScripTable object.
		Return:
			A new ScripTable object.
		'''
		assert isinstance(other, ScripTable), f"Cannot add {type_name(other)}."
		new = copy.deepcopy(other)
		new.update(self)

		newName = f"add({self.name},{other.name})"
		new.rename(newName)

		return new

class Transcription(ScripTable):
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
		items = sorted(self.items(), key=lambda x:x[0], reverse=reverse)		
		return Transcription(items, name=self.name)

	def __add__(self, other):
		'''
		Integrate two transcription objects. If utt is existed in both two objects, the former will be retained.

		Args:
			<other>: another Transcription object.
		Return:
			A new Transcription object.
		'''
		assert isinstance(other, Transcription), f"Cannot add {type_name(other)}."
		result = super().__add__(other)
		return Transcription(result, name=result.name)

class Cost(ScripTable):
	'''
	This is used to hold the AM or LM cost of N-Best. 
	'''
	def __init__(self, data={}, name="cost"):
		super(Cost, self).__init__(data, name=name)

	def sort(self, reverse=False):
		'''
		Sort by utterance ID.

		Args:
			<reverse>: If reverse, sort in descending order.
		Return:
			A new Cost object.
		''' 
		results = super().sort(reverse=False)	
		return Cost(results, name=self.name)
	
	def __add__(self, other):
		'''
		Integrate two Cost objects. If utt is existed in both two objects, the former will be retained.

		Args:
			<other>: another Cost object.
		Return:
			A new Cost object.
		'''
		assert isinstance(other, Cost), f"Cannot add {type_name(other)}."
		result = super().__add__(other)
		return Cost(result, name=result.name)

class BytesDataIndex(ScripTable):
	'''
	For accelerate to find utterance and reduce Memory cost of intermidiate operation.
	This is used to hold utterance index information of bytes data. It likes the script-file of a feature binary achivements in Kaldi.
	Every BytesData object will carry with one BytesDataIndex object.
	{ "utt": (frames, start_index, length) }
	'''	
	def __init__(self, data, name="index"):
		super(BytesDataIndex, self).__init__(data, name=name)
	
	def sort(self, by="utt", reverse=False):
		'''
		Sort utterances by frames length or uttID

		Args:
			<by>: "frame" or "utt" or "startIndex"
			<reverse>: If reverse, sort in descending order.
		Return:
			A new NumpyData object.
		''' 
		if self.is_void:
			raise WrongOperation('No data to sort.')
		assert by in ["utt","frame"], "We only support sorting by 'name' or 'frame'."

		if by == "utt":
			items = sorted(self.items(), lambda x:x[0], reverse=reverse)
		elif by == "frame":
			items = sorted(self.items(), lambda x:x[1][0], reverse=reverse)
		else:
			items = sorted(self.items(), lambda x:x[1][1], reverse=reverse)
		
		return BytesDataIndex(items)

	def __add__(self, other):
		'''
		Integrate two BytesDataIndex objects. If utt is existed in both two objects, the former will be retained.

		Args:
			<other>: another BytesDataIndex object.
		Return:
			A new BytesDataIndex object.
		'''
		assert isinstance(other, BytesDataIndex), f"Cannot add {type_name(other)}."
		result = super().__add__(other)
		return BytesDataIndex(result, name=result.name)

	def save(self):
		raise WrongOperation("BytesDataIndex is unsupported to be saved as file.")

	def load(self):
		raise WrongOperation("BytesDataIndex is unsupported to be loaded from file.")

''' Kaldi archive table '''

class BytesAchievement:

	def __init__(self, data=b'', name=None):
		if data != None:
			assert isinstance(data, bytes), f"Expected Python bytes object, but got {type_name(data)}."
		self.__data = data

		if name is None:
			self.__name = self.__class__.__name__
		else:
			assert isinstance(name, str) and len(name) > 0, "Name must be a string avaliable."
			self.__name = name
	
	@property
	def data(self):
		return self.__data
	
	@property
	def is_void(self):
		if self.__data is None or self.__data == b'':
			return True
		else:
			return False

	@property
	def name(self):
		return self.__name

	def rename(self, newName):
		assert isinstance(newName, str) and len(newName) > 0, "New name must be a string avaliable."
		self.__name = newName

class BytesData(BytesAchievement):
	'''
	A base class of bytes feature, cmvn statistics, post probability data.  
	'''
	def __init__(self, data=b'', name="data", indexTable=None):
		if isinstance(data, BytesData):
			data = data.data
		elif isinstance(data, NumpyData):
			data = (data.to_bytes()).data
		elif isinstance(data, bytes):
			pass
		else:
			raise UnsupportedDataType(f"Expected exkaldi BytesData, NumpyData or Python bytes object but got {type_name(data)}.")
		super().__init__(data, name)

		if indexTable is None:
			self.__generate_index_table()
		else:
			assert isinstance(indexTable, BytesDataIndex), f"<indexTable> should be a BytesDataIndex object bur got {type_name(indexTable)}."
			self.__dataIndex = indexTable

	def __generate_index_table(self):
		'''
		Genrate the index table.
		'''
		if self.is_void:
			self.__dataIndex = None
		else:
			self.__dataIndex = BytesDataIndex(name=self.name + "_index")
			start_index = 0
			with BytesIO(self.data) as sp:
				while True:
					(utt, dataType, rows, cols, buf) = self.__read_one_record(sp)
					if utt == None:
						break
					if dataType == 'DM ':
						sampleSize = 8
					else:
						sampleSize = 4
					oneRecordLen = len(utt) + 16 + rows * cols * sampleSize

					self.__dataIndex[utt] = (rows, start_index, oneRecordLen)
					start_index += oneRecordLen            

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
				return (None,None,None,None,None)
			else:
				fp.close()
				raise WrongDataFormat("Miss utterance ID before utterance.")
		binarySymbol = fp.read(2).decode()
		if binarySymbol == '\0B':
			dataType = fp.read(3).decode() 
			if dataType == 'CM ':
				fp.close()
				raise UnsupportedDataType("This is compressed ark data. Use load() function to load ark file again or use decompress() function to decompress it firstly.")                    
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
			buf = fp.read(rows * cols * sampleSize)
		else:
			fp.close()
			raise WrongDataFormat("Miss binary symbol before utterance.")
		return (utt, dataType, rows, cols, buf)

	@property
	def utt_index(self):
		'''
		Get the index information of utterances.
		
		Return:
			A BytesDataIndex object.
		'''
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
				(utt, dataType, rows, cols, buf) = self.__read_one_record(sp)
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
			A new BytesData object.
		'''
		assert isinstance(dtype, str) and (dtype in ["float", "float32", "float64"]), f"Expected <dtype> is string from 'float', 'float32' or 'float64' but got '{dtype}'."

		if self.is_void or self.dtype == dtype:
			result = copy.deepcopy(self.data)
		else:
			if dtype == 'float32' or dtype == 'float':
				newDataType = 'FM '
			else:
				newDataType = 'DM '
			
			result = []
			with BytesIO(self.data) as sp:
				start_index = 0
				while True:
					(utt, dataType, rows, cols, buf) = self.__read_one_record(sp)
					if utt == None:
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
					self.__dataIndex[utt] = (cols, start_index, oneRecordLength)
					start_index += oneRecordLength
					
			result = b''.join(result)

		return BytesData(result, name=self.name)

	@property
	def dim(self):
		'''
		Get the data dimensionality.
		
		Return:
			If data is void, return None, or return an int value.
		'''
		if self.is_void:
			dimension = None
		else:
			with BytesIO(self.data) as sp:
				(utt, dataType, rows, cols, buf) = self.__read_one_record(sp)
				dimension = cols
		
		return dimension

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
			
	def check_format(self):
		'''
		Check if data has right kaldi formation.
		
		Return:
			If data is void, return False.
			If data has right formation, return True, or raise Error.
		'''
		if not self.is_void:
			_dim = "unknown"
			_dataType = "unknown"
			with BytesIO(self.data) as sp:
				start_index = 0
				while True: 
					(utt,dataType,rows,cols,buf) = self.__read_one_record(sp)
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
								vec = np.frombuffer(buf, dtype=np.float32)
							else:
								vec = np.frombuffer(buf, dtype=np.float64)
						except Exception as e:
							print(f"Wrong matrix data format at utterance {utt}.")
							raise e
					
					if dataType == 'DM ':
						sampleSize = 8
					else:
						sampleSize = 4
					oneRecordLen = len(utt) + 16 + rows * cols * sampleSize

					self.__dataIndex[utt] = (rows, start_index, oneRecordLen)
					start_index += oneRecordLen					
					
			return True
		else:
			return False

	@property
	def lens(self):
		'''
		Get the frames of all utterances.
		
		Return:
			a tuple: (the numbers of all utterances, the utterance ID and frames of each utterance).
			If there is not any data, return (0, None).
		'''
		lengths = (0, None)
		if not self.is_void:
			_lens = dict( (key, value[0] ) for key, value in self.__dataIndex.items() )
			lengths = (len(_lens), _lens)
		
		return lengths

	def save(self, fileName, chunks=1, outScpFile=False):
		'''
		Save bytes data to file.

		Args:
			<fileName>: file name. Defaultly suffix ".ark" will be add to the name.
			<chunks>: If larger than 1, data will be saved to mutiple files averagely.
			<outScpFile>: If True, ".scp" file will be saved simultaneously.
		
		Return:
			the path of saved files.
		'''        
		assert isinstance(fileName, str), "file name must be a string."
		if self.is_void:
			raise WrongOperation('No data to save.')

		ExkaldiInfo.vertify_kaldi_existed()

		def save_chunk_data(chunkData, arkFileName, outScpFile):
			if outScpFile is True:
				scpFilename = arkFileName[-4:] + "scp"
				cmd = f"copy-feats ark:- ark,scp:{arkFileName},{scpFilename}"
			else:
				cmd = f"copy-feats ark:- ark:{arkFileName}"

			out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=chunkData)

			if (isinstance(cod, int) and cod != 0) or (not os.path.isfile(arkFileName)) or (os.path.getsize(arkFileName) == 0):
				print(err.decode())
				if os.path.isfile(arkFileName):
					os.remove(arkFileName)
				raise KaldiProcessError("Failed to save bytes data.")
			else:
				if outScpFile is True:
					return (arkFileName, scpFilename)
				else:
					return arkFileName

		if chunks == 1:
			if not fileName.strip().endswith('.ark'):
				fileName += '.ark'
			savedFilesName = save_chunk_data(self.data, fileName, outScpFile)	
		else:
			if fileName.strip().endswith('.ark'):
				fileName = fileName[0:-4]
			
			uttLens = [ value[2] for vlaue in self.__dataIndex.values() ]
			allLens = len(uttLens)
			chunkUtts = allLens//chunks
			if chunkUtts == 0:
				chunks = allLens
				chunkUtts = 1
				t = 0
				print(f"Warning: utterances is fewer than <chunks> so only {chunks} files will be saved.")
			else:
				t = allLens - chunkUtts * chunks

			savedFilesName = []

			with BytesIO(self.data) as sp:
				for i in range(chunks):
					if i < t:
						chunkLen = sum(uttLens[i*(chunkUtts+1):(i+1)*(chunkUtts+1)])
					else:
						chunkLen = sum(uttLens[i*chunkUtts:(i+1)*chunkUtts])
					chunkData = sp.read(chunkLen)
					savedFilesName.append(save_chunk_data(chunkData, fileName + f'_ck{i}.ark', outScpFile))

		return savedFilesName

	def to_numpy(self):
		'''
		Transform bytes data to numpy data.
		
		Return:
			a NumpyData object.
		'''
		newDict = {}
		if not self.is_void:
			with BytesIO(self.data) as sp:
				while True:
					(utt, dataType, rows, cols, buf) = self.__read_one_record(sp)
					if utt == None:
						break
					try:
						if dataType == 'FM ': 
							newMatrix = np.frombuffer(buf, dtype=np.float32)
						else:
							newMatrix = np.frombuffer(buf, dtype=np.float64)
					except Exception as e:
						print(f"Wrong matrix data format at utterance {utt}.")
						raise e	
					else:
						newDict[utt] = np.reshape(newMatrix,(rows,cols))
		return NumpyData(newDict, name=self.name)

	def __add__(self, other):
		'''
		The Plus operation between two objects.

		Args:
			<other>: a BytesData or NumpyData object.
		Return:
			a new BytesData object.
		''' 
		if isinstance(other, BytesData):
			pass
		elif type_name(other) == "NumpyData":
			other = other.to_bytes()
		else:
			raise UnsupportedDataType(f"Expected exkaldi BytesData or NumpyData object but got {type_name(other)}.")
		
		if self.is_void:
			return copy.deepcopy(other)
		elif other.is_void:
			return copy.deepcopy(self)
		elif self.dim != other.dim:
			raise WrongOperation(f"Data dimensonality does not match: {self.dim}!={other.dim}.")

		selfUtts, selfDtype = self.utts, self.dtype
		otherDtype = other.dtype

		newDataIndex = self.utt_index
		lastIndexInfo = list(newDataIndex.sort(by="startIndex", reverse=True).values())[0]
		start_index = lastIndexInfo[1] + lastIndexInfo[2]

		newData = []
		with BytesIO(other.data) as op:
			for utt, indexInfo in other.utt_index.items():
				if not utt in selfUtts:
					ocols, si, dataSize = indexInfo
					if selfDtype == otherDtype:
						op.seek(si)
						data = op.read(dataSize)
					else:
						(outt, odataType, orows, ocols, obuf) = self.__read_one_record(op)
						obuf = np.array(np.frombuffer(obuf, dtype=otherDtype), dtype=selfDtype).tobytes()
						data = (outt+' ').encode()
						data += '\0B'.encode()
						data += odataType.encode()
						data += '\04'.encode()
						data += struct.pack(np.dtype('uint32').char, orows)
						data += '\04'.encode()
						data += struct.pack(np.dtype('uint32').char, ocols)
						data += obuf
						dataSize = len(data)

					newData.append(data)
					newDataIndex[utt] = (ocols, start_index, dataSize)
					start_index +=	dataSize

		newName = f"plus({self.name},{other.name})"
		return BytesData(b''.join([self.data, *newData]), name=newName, indexTable=newDataIndex)

	def subset(self, nHead=0, chunks=1, uttList=None):
		'''
		Subset data.
		
		Args:
			<nHead>: If it > 0, extract N head utterances.
			<chunks>: If nHead == 0 and chunks > 1, split data into N chunks.
			<uttList>: If nHead == 0 and chunks == 1 and uttList != None, pick out these utterances whose ID in uttList.
		Return:
			a new BytesData object or a list of new BytesData objects.
		''' 
		if self.is_void:
			raise WrongOperation("Cannot subset a void data.")

		if nHead > 0:
			assert isinstance(nHead, int), f"Expected <nHead> is an int number but got {nHead}."			
			newName = f"subset({self.name},head {nHead})"

			newDataIndex = BytesDataIndex(name=newName)
			start_index = 0
			
			for utt, indexInfo in self.utt_index.items():
				fs, si, ds = indexInfo
				newDataIndex[utt] = (fs, start_index, ds)
				start_index += ds
				nHead -= 1
				if nHead <= 0:
					break
			
			with BytesIO(self.data) as sp:
				sp.seek(0)
				data = sp.read(start_index)
	
			return BytesData(data, name=newName, indexTable=newDataIndex)
		
		elif chunks > 1:
			assert isinstance(chunks, int), f"Expected <chunks> is an int number but got {chunks}."

			uttLens = list(self.__dataIndex.items())
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
					newDataIndex = BytesDataIndex(name=newName)
					if i < t:
						chunkItems = uttLens[i*(chunkUtts+1) : (i+1)*(chunkUtts+1)]
					else:
						chunkItems = uttLens[i*chunkUtts : (i+1)*chunkUtts]
					chunkLen = 0
					for utt, indexInfo in chunkItems:
						fs, si, ds = indexInfo
						newDataIndex[utt] = (fs, chunkLen, ds)
						chunkLen += ds
					chunkData = sp.read(chunkLen)
					
					datas.append(BytesData(chunkData, name=newName, indexTable=newDataIndex))
			return datas

		elif uttList != None:
			if isinstance(uttList, str):
				newName = f"subset({self.name},uttList 1)"
				uttList = [uttList,]
			elif isinstance(uttList, (list,tuple)):
				newName = f"subset({self.name},uttList {len(uttList)})"
				pass
			else:
				raise UnsupportedDataType(f"Expected <uttList> is string, list or tuple but got {type_name(uttList)}.")

			newData = []
			dataIndex = self.utt_index
			newDataIndex = BytesDataIndex(name=newName)
			start_index = 0
			with BytesIO(self.data) as sp:
				for utt in uttList:
					if utt in self.utts:
						fr, si, ds = dataIndex[utt]
						newDataIndex[utt] = (fr, start_index, ds)
						start_index += ds
						sp.seek(si)
						newData.append( sp.read(ds) )

			return BytesData(b''.join(newData), name=newName, indexTable=newDataIndex)

		else:
			raise WrongOperation('Expected <nHead> is larger than "0", or <chunks> is larger than "1", or <uttList> is not None.')

	def __call__(self, utt):
		'''
		Pick out the specified utterance.
		
		Args:
			<utt>: a string.
		Return:
			If existed, return a new BytesData object.
			Or return None.
		''' 
		assert isinstance(utt, str), "utterance ID should be a name-like string."
		utt = utt.strip()

		if self.is_void:
			raise WrongOperation("Cannot get any data from a void object.")
		else:
			if utt not in self.utts:
				raise WrongOperation(f"No such utterance:{utt}.")
			else:
				fr, si, ds = self.utt_index[utt]
				newName = f"pick({self.name},{uttID})"
				newDataIndex = BytesDataIndex(name=newName)
				with BytesIO(self.data) as sp:
					sp.seek(si)
					data = sp.read(ds)
					result = BytesData(data, name=newName, indexTable=newDataIndex)
				
				return result

	def sort(self, by="utt", reverse=False):
		'''
		Sort utterances by frames length or uttID

		Args:
			<by>: "frame" or "utt"
			<reverse>: If reverse, sort in descending order.
		Return:
			A new BytesData object.
		''' 
		assert by in ["utt", "frames"], f"<by> should be 'utt' or 'frame'."

		newDataIndex = self.utt_index.sort(by=by, reverse=reverse)

		with BytesIO(self.data) as sp:
			if sys.getsizeof(self.data) > 1e-8:
				## If the data size is extremely large, divide it into N chunks and save it to intermidiate file.
				temp = tempfile.NamedTemporaryFile("wb+")
				try:
					chunkdata = []
					chunkSize = 50
					count = 0
					for utt, indexInfo in newDataIndex.items():
						fs, si, ds = indexInfo
						sp.seek(si)
						chunkdata.append( sp.read(ds) )
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
				finally:
					temp.close()
			else:
				chunkdata = []
				start_index = 0
				for utt, indexInfo in newDataIndex.items():
					fs, si, ds = indexInfo
					sp.seek(si)
					chunkdata.append( sp.read(ds) )
					newDataIndex[utt] = (fs, start_index, ds)
					start_index += ds

				newData = b"".join(chunkdata)
				del chunkdata

		return BytesData(newData, name=self.name, indexTable=newDataIndex)			
						
class BytesFeature(BytesData):
	'''
	Hold the feature with kaldi binary format.
	'''
	def __init__(self, data=b"", name="feat", indexTable=None):
		if isinstance(data, BytesFeature):
			data = data.data
		elif isinstance(data, NumpyFeature):
			data = (data.to_bytes()).data
		elif isinstance(data, bytes):
			pass
		else:
			raise UnsupportedDataType(f"Expected BytesFeature, NumpyFeature or Python bytes object but got {type_name(data)}.")
		
		super().__init__(data, name, indexTable)
	
	def to_numpy(self):
		'''
		Transform feature to numpy formation.

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
		if isinstance(other, BytesFeature):
			pass
		elif isinstance(other, NumpyFeature):
			other = other.to_bytes()
		else:
			raise UnsupportedDataType(f"Excepted a BytesFeature or NumpyFeature object but got {type_name(other)}.")

		result = super().__add__(other)
		return NumpyFeature(result.data, name=result.name, indexTable=result.utt_index)

	def splice(self, left=1, right=None):
		'''
		Splice front-behind N frames to generate new feature data.

		Args:
			<left>: the left N-frames to splice.
			<right>: the right N-frames to splice. If None, right = left.
		Return:
			a new BytesFeature object whose dim became original-dim * (1 + left + right).
		''' 
		ExkaldiInfo.vertify_kaldi_existed()
		if self.is_void:
			raise WrongOperation("Cannot operate a void feature data.")

		assert isinstance(left, int) and left >= 0, f"Expected <left> is a positive int number but got {left}."
		if right is None:
			right = left
		else:
			assert isinstance(right, int) and right >= 0, f"Expected <right> is a positive int number but got {right}."

		cmd = f"splice-feats --left-context={left} --right-context={right} ark:- ark:-"
		out, err, _ = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=self.data)

		if out == b'':
			print(err.decode())
			raise KaldiProcessError("Failed to splice left-right frames.")
		else:
			newName = f"splice({self.name},{left},{right})"
			return BytesFeature(out, name=newName)

	def select(self, dims, retain=False):
		'''
		Select specified dimensions of feature.

		Args:
			<dims>: A int value or string such as "1,2,5-10"
			<retain>: If True, return the rest dimensions of feature simultaneously.
		Return:
			A new BytesFeature object or two BytesFeature objects.
		''' 
		ExkaldiInfo.vertify_kaldi_existed()
		if self.is_void:
			raise WrongOperation("Cannot operate a void feature data.")

		_dim = self.dim

		if isinstance(dims, int):
			assert 0 <= dims < _dim, f"Selection index should be smaller than data dimension {_dim} but got {dims}."
			selectFlag = str(dims)
			if retain:
				if dims == 0:
					retainFlag = f"1-{_dim-1}"
				elif dims == _dim-1:
					retainFlag = f"0-{_dim-2}"
				else:
					retainFlag = f"0-{dims-1},{dims+1}-{_dim-1}"
		
		elif isinstance(dims, str):
			if dims.strip() == "":
				raise WrongOperation("<dims> is not a value avaliable.")

			if retain:
				retainFlag = [x for x in range(_dim)]
				for i in dims.strip().split(','):
					if i.strip() == "":
						continue
					if not '-' in i:
						try:
							i = int(i)
						except ValueError:
							raise WrongOperation(f"Expected int value but got {i}.")
						else:
							assert 0 <= i < _dim, f"Selection index should be smaller than data dimension {_dim} but got {i}."
							retainFlag[i]=-1
					else:
						i = i.split('-')
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
							assert i[1] < _dim, f"Selection index should be smaller than data dimension {_dim} but got {i[1]}."
						for j in range(i[0],i[1]+1,1):
							retainFlag[j] = -1
				temp = ''
				for x in retainFlag:
					if x != -1:
						temp += str(x)+','
				retainFlag = temp[0:-1]
			selectFlag = dims
		
		else:
			raise WrongOperation(f"Expected int value or string like '1,4-9,12' but got {dims}.")

		cmdS = f'select-feats {selectFlag} ark:- ark:-'
		outS, errS, _ = run_shell_command(cmdS, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=self.data)
		
		if outS == b'':
			print(errS.decode())
			raise KaldiProcessError("Failed to select data.")
		else:
			newName = f"select({self.name},{dims})"
			selectedResult = BytesFeature(outS, name=newName)

		if retain:
			if retainFlag == "":
				newName = f"select({self.name}, void)"
				retainedResult = BytesFeature(name=newName)
			else: 
				cmdR = f"select-feats {retainFlag} ark:- ark:-"
				outR, errR, _ = run_shell_command(cmdR, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=self.data)
				if outR == b'':
					print(errR.decode())
					raise KaldiProcessError("Failed to select retained data.")
				else:
					newName = f"select({self.name},not {dims})"
					retainedResult = BytesFeature(outR, name=newName)
		
			return selectedResult, retainedResult
		
		else:
			return selectedResult

	def subset(self, nHead=0, chunks=1, uttList=None):
		'''
		Subset feature.
		
		Args:
			<nHead>: If it > 0, extract N head utterances.
			<chunks>: If nHead == 0 and chunks > 1, split data into N chunks.
			<uttList>: If nHead == 0 and chunks == 1 and uttList != None, pick out these utterances whose ID in uttList.
		Return:
			a new BytesFeature object or a list of new BytesFeature objects.
		'''
		result = super().subset(nHead,chunks,uttList)

		if isinstance(result, list):
			for index in range(len(result)):
				temp = result[index]
				result[index] = BytesFeature(temp.data, temp.name)
		else:
			result = BytesFeature(result.data, result.name)

		return result
	
	def __call__(self, uttID):
		'''
		Pick out the specified utterance.
		
		Args:
			<uttID>: a string.
		Return:
			If existed, return a new BytesFeature object.
			Or return None.
		''' 
		result = super().__call__(uttID)
		if result is None:
			return None
		else:
			return BytesFeature(result.data, result.name)

	def add_delta(self, order=2):
		'''
		Add N orders delta information to feature.

		Args:
			<order>: A positive int value.
		Return:
			A new BytesFeature object whose dimendion became original-dim * (1 + order). 
		''' 
		ExkaldiInfo.vertify_kaldi_existed()
		assert isinstance(order, int) and order > 0, f"<order> should be a positive int value but got {order}."

		if self.is_void:
			raise WrongOperation("Cannot operate a void feature data.")

		cmd = f"add-deltas --delta-order={order} ark:- ark:-"
		out, err, _ = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=self.data)
		if out == b'':
			print(err.decode())
			raise KaldiProcessError('Failed to add delta feature.')
		else:
			newName = f"delta({self.name},{order})"
			return BytesFeature(data=out, name=newName)

	def paste(self, others, ordered=False):
		'''
		Paste feature in feature dimension.

		Args:
			<others>: a feature object or list of feature objects.
			<ordered>: If False, sort all objects.  
		Return:
			a new feature object.
		''' 
		ExkaldiInfo.vertify_kaldi_existed()
		if self.is_void:
			raise WrongOperation("Cannot operate a void feature data.")
		
		selfData = tempfile.NamedTemporaryFile("wb+")
		otherData = []
		otherName = []
		try:
			if isinstance(others, BytesFeature):
				temp = tempfile.NamedTemporaryFile("wb+")
				if ordered:
					temp.write(others.data)
				else:
					temp.write(others.sort(by="utt").data)
				otherData.append(temp)
				otherName.append(others.name)
			elif isinstance(others, NumpyFeature):
				temp = tempfile.NamedTemporaryFile("wb+")
				if ordered:
					temp.write(others.to_bytes().data)
				else:
					temp.write(others.sort(by="utt").to_bytes().data)
				otherData.append(temp)
				otherName.append(others.name)
			elif isinstance(others, (list,tuple)):
				for ot in others:
					if isinstance(ot, BytesFeature):
						if ordered:
							da = ot.data
						else:
							da = ot.sort(by="utt").data
					elif isinstance(ot, NumpyFeature):
						if ordered:
							da = ot.to_bytes().data
						else:
							da = ot.sort(by="utt").to_bytes().data
					else:
						raise UnsupportedDataType(f"Expected exkaldi feature object but got {type_name(ot)}.")
					temp = tempfile.NamedTemporaryFile("wb+")
					temp.write(da)
					otherData.append(temp)
					otherName.append(ot.name)		
			else:
				raise UnsupportedDataType(f"Expected exkaldi feature object but got {type_name(others)}.")
			
			if ordered:
				selfData.write(self.data)
			else:
				selfData.write(self.sort(by="utt").data)
			cmd = f"paste-feats ark:{selfData.name} "
			for ot in otherData:
				cmd += f"ark:{ot.name} "
			cmd += "ark:-"
			
			out, err, _ = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

			if out == b'':
				print(err.decode())
				raise KaldiProcessError("Failed to paste feature.")
			else:
				newName = f"paste({self.name}"
				for on in otherName:
					newName += ",{on}"
				newName += ")"
			
				return BytesFeature(out, name=newName)
		finally:
			selfData.close()
			for ot in otherData:
				ot.close()

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
		return BytesFeature(result.data, name=result.name)

class BytesCMVNStatistics(BytesData):
	'''
	Hold the CMVN statistics with kaldi binary format.
	'''
	def __init__(self, data=b"", name="cmvn"):
		if isinstance(data, BytesCMVNStatistics):
			data = data.data
		elif isinstance(data, NumpyCMVNStatistics):
			data = (data.to_bytes()).data
		elif isinstance(data, bytes):
			pass
		else:
			raise UnsupportedDataType(f"Expected BytesCMVNStatistics, NumpyCMVNStatistics or Python bytes object but got {type_name(data)}.")

		super().__init__(data, name)
	
	def to_numpy(self):
		'''
		Transform CMVN statistics to numpy formation.

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
		if isinstance(other, BytesCMVNStatistics):
			pass
		elif isinstance(other, NumpyCMVNStatistics):
			other = other.to_bytes()
		else:
			raise UnsupportedDataType(f"Excepted a BytesCMVNStatistics or NumpyCMVNStatistics object but got {type_name(other)}.")

		result = super().__add__(other)

		return BytesCMVNStatistics(result.data, name=result.name)

	def subset(self, nHead=0, chunks=1, uttList=None):
		'''
		Subset CMVN statistics.
		
		Args:
			<nHead>: If it > 0, extract N head utterances.
			<chunks>: If nHead == 0 and chunks > 1, split data into N chunks.
			<uttList>: If nHead == 0 and chunks == 1 and uttList != None, pick out these utterances whose ID in uttList.
		Return:
			a new BytesCMVNStatistics object or a list of new BytesCMVNStatistics objects.
		'''
		result = super().subset(nHead, chunks, uttList)

		if isinstance(result, list):
			for index in range(len(result)):
				temp = result[index]
				result[index] = BytesCMVNStatistics(temp.data, temp.name)
		else:
			result = BytesCMVNStatistics(result.data, result.name)

		return result

	def __call__(self, uttID):
		'''
		Pick out the specified utterance.
		
		Args:
			<uttID>: a string.
		Return:
			If existed, return a new BytesCMVNStatistics object.
			Or return None.
		''' 
		result = super().__call__(uttID)
		if result is None:
			return None
		else:
			return BytesCMVNStatistics(result.data, result.name)

	def sort(self, by="utt", reverse=False):
		'''
		Sort utterances by frames length or uttID

		Args:
			<by>: "frame" or "utt"
			<reverse>: If reverse, sort in descending order.
		Return:
			A new BytesCMVNStatistics object.
		''' 
		result = super().sort(by, reverse)
		return BytesCMVNStatistics(result.data, name=result.name)

class BytesProbability(BytesData):
	'''
	Hold the probalility with kaldi binary format.
	'''
	def __init__(self, data=b"", name="postprob"):
		if isinstance(data, BytesProbability):
			data = data.data
		elif isinstance(data, NumpyProbability):
			data = (data.to_bytes()).data
		elif isinstance(data, bytes):
			pass
		else:
			raise UnsupportedDataType(f"Expected BytesProbability, NumpyProbability or Python bytes object but got {type_name(data)}.")

		super().__init__(data, name)
	
	def to_numpy(self):
		'''
		Transform post probability to numpy formation.

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
		if isinstance(other, BytesProbability):
			other = other.data
		elif isinstance(other, NumpyProbability):
			other = (other.to_bytes()).data
		else:
			raise UnsupportedDataType(f"Expected BytesProbability or NumpyProbability object but got {type_name(other)}.")

		result = super().__add__(other)

		return BytesProbability(result.data, result.name)

	def subset(self,nHead=0,chunks=1,uttList=None):
		'''
		Subset post probability.
		
		Args:
			<nHead>: If it > 0, extract N head utterances.
			<chunks>: If nHead == 0 and chunks > 1, split data into N chunks.
			<uttList>: If nHead == 0 and chunks == 1 and uttList != None, pick out these utterances whose ID in uttList.
		Return:
			a new BytesProbability object or a list of new BytesProbability objects.
		'''
		result = super().subset(nHead,chunks,uttList)

		if isinstance(result, list):
			for index in range(len(result)):
				temp = result[index]
				result[index] = BytesProbability(temp.data, temp.name)
		else:
			result = BytesProbability(result.data, result.name)

		return result

	def __call__(self, uttID):
		'''
		Pick out the specified utterance.
		
		Args:
			<uttID>: a string.
		Return:
			If existed, return a new BytesProbability object.
			Or return None.
		''' 
		result = super().__call__(uttID)
		if result is None:
			return None
		else:
			return BytesProbability(result.data, result.name)

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
		return BytesProbability(result.data, name=result.name)

class BytesAlignmentTrans(BytesAchievement):
	'''
	Hold the alignment(transition ID) with kaldi binary format.
	'''
	def __init__(self, data=b"", name="transitionID"):
		if isinstance(data, BytesProbability):
			data = data.data
		elif isinstance(data, NumpyProbability):
			data = (data.to_bytes()).data
		elif isinstance(data, bytes):
			pass
		else:
			raise UnsupportedDataType(f"Expected BytesAlignment, NumpyAlignment or Python bytes object but got {type_name(data)}.")		
		super().__init__(data, name)
	
	def to_numpy(self, aliType="transitionID", hmm=None):
		'''
		Transform alignment to numpy formation.

		Args:
			<aliType>: If it is "transitionID", transform to transition IDs.
					  If it is "phoneID", transform to phone IDs.
					  If it is "pdfID", transform to pdf IDs.
		Return:
			a NumpyAlignmentTrans or NumpyAlignmentPhone or NumpyAlignmentPdf object.
		'''
		if self.is_void:
			return NumpyAlignmentTrans({}, self.name)

		ExkaldiInfo.vertify_kaldi_existed()

		def transform(data, cmd):
			out, err, _ = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=data)
			if out == b'':
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
				return results

		if aliType == "transitionID":
			cmd = "copy-int-vector ark:- ark,t:-"
			result = transform(self.data, cmd)
			return NumpyAlignmentTrans(result, self.name)
		
		else:
			temp = tempfile.NamedTemporaryFile("wb+")
			try:
				if type_name(hmm) in ("HMM", "MonophoneHMM", "TriphoneHMM"):
					hmm.save(temp)
					hmmFileName = temp.name
				elif isinstance(hmm, str):
					if not os.path.isfile(hmm):
						raise WrongPath(f"No such file:{hmm}.")
					hmmFileName = hmm
				else:
					raise UnsupportedDataType(f"<hmm> should be a filePath or exkaldi HMM and its sub-class object. but got {type_name(hmm)}.") 

				if aliType == "phoneID":
					cmd = f"ali-to-phones --per-frame=true {hmmFileName} ark:- ark,t:-"
					result = transform(self.data, cmd)
					return NumpyAlignmentPhone(result, self.name)

				elif target == "pdfID":
					cmd = f"ali-to-pdf {hmmFileName} ark:- ark,t:-"
					result = transform(self.data, cmd)
					return NumpyAlignmentPdf(result, self.name)
				else:
					raise WrongOperation(f"<target> should be 'trainsitionID', 'phoneID' or 'pdfID' but got {target}.")
			finally:
				temp.close()

	def save(self, fileName, chunks=1):
		'''
		Save bytes alignment to file.

		Args:
			<fileName>: file name. Defaultly suffix ".ali" will be add to the name.
			<chunks>: If larger than 1, data will be saved to mutiple files averagely.
		
		Return:
			the path of saved files.
		'''
		ExkaldiInfo.vertify_kaldi_existed()
		assert isinstance(fileName, str), "file name should be a string."
		assert isinstance(chunks, int) and chunks > 0, "<chunks> should be a positive int value."

		if self.is_void:
			raise WrongOperation('No data to save.')

		#if sys.getsizeof(self)/chunks > 10000000000:
		#   print("Warning: Data size is extremely large. Try to save it with a long time.")

		def save_chunk_data(chunkData, fileName):

			cmd = f"copy-int-vector ark:- ark:{fileName}"
			out, err, _ = run_shell_command(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, inputs=chunkData)

			if not os.path.isfile(fileName) or os.path.getsize(fileName) == 0:
				print(err.decode())
				if os.path.isfile(fileName):
					os.remove(fileName)
				raise KaldiProcessError('Failed to save alignment.')
			else:
				return os.path.abspath(fileName)

		if chunks == 1:
			if not fileName.rstrip().endswith(".ali"):
				fileName += ".ali"
			savedFilesName = save_chunk_data(self.data, fileName)
		else:
			if fileName.rstrip().endswith(".ali"):
				fileName = fileName.rstrip()[:-4]

			cmd = "copy-int-vector ark:- ark,t:-"
			out, err, _ = run_shell_command(cmd, stdin=subprocess.PIPE,stderr=subprocess.PIPE, inputs=self.data)
			utts = []
			with BytesIO(out) as sp:
				utts.extend(sp.readlines())
			allLens = len(utts)
			chunkUtts = allLens//chunks
			if chunkUtts == 0:
				chunks = allLens
				chunkUtts = 1
				t = 0
				print(f"Warning: utterances is fewer than <chunks> so only {chunks} files will be saved.")
			else:
				t = allLens - chunkUtts * chunks

			savedFilesName = []
			start = 0
			for i in range(chunks):
				if i < t:
					chunkLen = chunkUtts + 1
				else:
					chunkLen = chunkUtts
				chunkData = utts[ start:start+chunkLen ]
				start += chunkLen
				savedFilesName.append(save_chunk_data(chunkData, fileName + f"_ck{i}.ali"))

		return savedFilesName

class NumpyAchievement:

	def __init__(self, data={}, name=None):
		if data is not None:
			assert isinstance(data,dict), f"Expected Python dict object, but got {type_name(data)}."
		self.__data = data

		if name is None:
			self.__name = self.__class__.__name__
		else:
			assert isinstance(name, str) and len(name) > 0, "<name> must be a string."
			self.__name = name	
	
	@property
	def data(self):
		return self.__data

	@property
	def is_void(self):
		if self.__data is None or len(self.__data) == 0:
			return True
		else:
			return False

	@property
	def name(self):
		return self.__name

	def rename(self,newName):
		assert isinstance(newName,str) and len(newName) > 0, "new name must be a string avaliable."
		self.__name = newName

class NumpyData(NumpyAchievement):

	def __init__(self, data={}, name=None):
		if isinstance(data, NumpyData):
			data = data.data
		elif isinstance(data, BytesData):
			data = (data.to_Numpy()).data
		elif isinstance(data, dict):
			pass
		else:
			raise UnsupportedDataType(f"Expected NumpyData, BytesData or Python dict object but got {type_name(data)}.")
		super().__init__(data, name)

	@property
	def arrays(self):
		'''
		Get all arrays.

		Return:
			a list of arrays.
		'''
		return list(self.data.values())

	@property
	def items(self):
		'''
		Return an iterable object of (utt, matrix).
		'''
		return self.data.items()

	@property
	def dtype(self):
		'''
		Get the data type of Numpy data.
		
		Return:
			A string, 'float32', 'float64', or 'int32'.
		'''  
		_dtype = None
		if not self.is_void:
			utts = self.utts
			_dtype = str(self.data[utts[0]].dtype)
		return _dtype
	
	@property
	def dim(self):
		'''
		Get the data dimensionality.
		
		Return:
			If data is void, return None, or return an int value.
		'''		
		_dim = None
		if not self.is_void:
			utts = self.utts
			if self.data[utts[0]].shape <= 1:
				_dim = 0
			else:
				_dim = self.data[utts[0]].shape[1]
		
		return _dim
		
	def to_dtype(self, dtype):
		'''
		Transform data type.

		Args:
			<dtype>: a string of "float", "float32" or "float64". IF "float", it will be treated as "float32".
					or a string of "int", "int32" or "int64". IF "int", it will be treated as "int32".
		Return:
			A new BytesData object.
		'''
		if self.is_void or self.dtype == dtype:
			newData = copy.deepcopy(self.data)
		else:
			assert dtype in ['int','int32','int64','float','float32','float64'], f'Expected <dtype> is "int", "int32", "int64", "float", "float32" or "float64" but got {dtype}.'
			if dtype == 'int':
				dtype = 'int32'
			elif dtype == 'float': 
				dtype = 'float32'
			newData = {}
			for utt in self.utts:
				newData[utt] = np.array(self.data[utt], dtype=dtype)
		
		return NumpyData(newData, name=self.name)

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
		Check if data has right kaldi formation.
		
		Return:
			If data is void, return False.
			If data has right formation, return True, or raise Error.
		'''
		if not self.is_void:
			_dim = 'unknown'
			for utt in self.utts:
				if not isinstance(utt,str):
					raise WrongDataFormat(f'Expected utterance ID is a string but got {type_name(utt)}.')
				if not isinstance(self.data[utt],np.ndarray):
					raise WrongDataFormat(f'Expected value is NumPy ndarray but got {type_name(self.data[utt])}.')
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
		else:
			return False

	def to_bytes(self):
		'''
		Transform numpy data to bytes data.
		
		Return:
			a BytesData object.
		'''
		#totalSize = 0
		#for u in self.keys():
		#    totalSize += sys.getsizeof(self[u])
		#if totalSize > 10000000000:
		#    print('Warning: Data is extramely large. Try to transform it but it maybe result in MemoryError.')

		if self.dim == 0:
			raise WrongOperation("1-dimension data cannot be changed into bytes object.")
		
		self.check_format()

		newData = []
		for utt in self.utts:
			matrix = self.data[utt]
			data = (utt+' ').encode()
			data += '\0B'.encode()
			if matrix.dtype == 'float32':
				data += 'FM '.encode()
			elif matrix.dtype == 'float64':
				data += 'DM '.encode()
			else:
				raise UnsupportedDataType(f'Expected "float32" or "float64" data, but got {matrix.dtype}.')
			
			data += '\04'.encode()
			data += struct.pack(np.dtype('uint32').char, matrix.shape[0])
			data += '\04'.encode()
			data += struct.pack(np.dtype('uint32').char, matrix.shape[1])
			data += matrix.tobytes()
			newData.append(data)

		return BytesData(b''.join(newData), self.name)

	def save(self, fileName, chunks=1):
		'''
		Save numpy data to file.

		Args:
			<fileName>: file name. Defaultly suffix ".npy" will be add to the name.
			<chunks>: If larger than 1, data will be saved to mutiple files averagely.		
		Return:
			the path of saved files.
		'''      
		if self.is_void:
			raise WrongOperation('No data to save.')

		#totalSize = 0
		#for u in self.keys():
		#    totalSize += sys.getsizeof(self[u])
		#if totalSize > 10000000000:
		#    print('Warning: Data size is extremely large. Try to save it with a long time.')
		
		if fileName.rstrip().endswith('.npy'):
			fileName = fileName[0:-4]

		if chunks == 1:          
			allData = tuple(self.data.items())
			np.save(fileName, allData)
			savedFilesName = fileName + '.npy'
		else:
			allData = tuple(self.data.items())
			allLens = len(allData)
			chunkUtts = allLens//chunks
			if chunkUtts == 0:
				chunks = allLens
				chunkUtts = 1
				t = 0
				print(f"Warning: utterances is fewer than <chunks> so only {chunks} files will be saved.")
			else:
				t = allLens - chunkUtts * chunks
			savedFilesName = []
			for i in range(chunks):
				if i < t:
					chunkData = allData[i*(chunkUtts+1):(i+1)*(chunkUtts+1)]
				else:
					chunkData = allData[i*chunkUtts:(i+1)*chunkUtts]
				np.save(fileName + f'_ck{i}.npy',chunkData)
				savedFilesName.append(fileName + f'_ck{i}.npy')
		
		return savedFilesName

	def __add__(self, other):
		'''
		The Plus operation between two objects.

		Args:
			<other>: a BytesData or NumpyData object.
		Return:
			a new NumpyData object.
		''' 
		if isinstance(other, NumpyData):
			pass
		elif isinstance(other, BytesData):
			other = other.to_numpy()
		else:
			raise UnsupportedDataType(f"Expected exkaldi BytesData or NumpyData object but got {type_name(other)}.")

		if self.is_void:
			return copy.deepcopy(other)
		elif other.is_void:
			return copy.deepcopy(self)
		elif self.dim != other.dim:
			raise WrongOperation(f"Data dimensonality does not match: {self.dim}!={other.dim}.")

		temp = self.data.copy()
		selfUtts = list(self.utts)
		for utt in other.utts:
			if not utt in selfUtts:
				temp[utt] = other.data[utt]

		newName = f"plus({self.name},{other.name})"
		return NumpyData(temp, newName)

	def __call__(self, uttID):
		'''
		Pick out the specified utterance.
		
		Args:
			<uttID>: a string.
		Return:
			If existed, return a new NumpyData object.
			Or return None.
		''' 
		assert isinstance(uttID, str), "utterance ID should be a name-like string."
		uttID = uttID.strip()

		if self.is_void:
			raise WrongOperation("Cannot get any data from void object.")
		else:
			if uttID not in self.utts:
				return None
			else:
				newName = f"pick({self.name},{uttID})"
				return NumpyData({uttID:self.data[uttID]}, newName)

	@property
	def lens(self):
		'''
		Get the frames of all utterances.
		
		Return:
			a tuple: (the numbers of all utterances, the utterance ID and frames of each utterance).
			If there is not any data, return (0, None).
		'''
		if self.is_void:
			return (0, None)
		else:
			_lens = {}
			for utt, matrix in self.data.items():
				_lens[utt] = len(matrix)
			return (len(_lens) ,_lens)

	def subset(self, nHead=0, chunks=1, uttList=None):
		'''
		Subset data.
		
		Args:
			<nHead>: If it > 0, extract N head utterances.
			<chunks>: If nHead == 0 and chunks > 1, split data into N chunks.
			<uttList>: If nHead == 0 and chunks == 1 and uttList != None, pick out these utterances whose ID in uttList.
		Return:
			a new NumpyData object or a list of new NumpyData objects.
		''' 
		if self.is_void:
			raise WrongOperation("Cannot subset a void data.")

		if nHead > 0:
			assert isinstance(nHead,int), f"Expected <nHead> is an int number but got {nHead}."
			newDict = {}
			for utt in self.utts[0:nHead]:
				newDict[utt]=self.data[utt]
			newName = f"subset({self.name},head {nHead})"
			return NumpyData(newDict, newName)

		elif chunks > 1:
			assert isinstance(chunks, int), f"Expected <chunks> is an int number but got {chunks}."
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
						chunkUttList = utts[i*(chunkUtts+1):(i+1)*(chunkUtts+1)]
					else:
						chunkUttList = utts[i*chunkUtts:(i+1)*chunkUtts]
					for utt in chunkUttList:
						temp[utt]=self.data[utt]
					newName = f"subset({self.name},chunk {chunks}-{i})"
					datas.append( NumpyData(temp, newName) )
			return datas

		elif uttList != None:
			
			if isinstance(uttList,str):
				newName = f"subset({self.name},uttList 1)"
				uttList = [uttList,]
			elif isinstance(uttList,(list,tuple)):
				newName = f"subset({self.name},uttList {len(uttList)})"
				pass
			else:
				raise UnsupportedDataType(f'Expected <uttList> is a string,list or tuple but got {type_name(uttList)}.')

			newDict = {}
			selfKeys = self.utts
			for utt in uttList:
				if utt in selfKeys:
					newDict[utt] = self.data[utt]
				else:
					#print('Subset Warning: no data for utt {}'.format(utt))
					continue
			return NumpyData(newDict, newName)
		
		else:
			raise WrongOperation('Expected <nHead> is larger than "0", or <chunks> is larger than "1", or <uttList> is not None.')

	def sort(self, by='frame', reverse=False):
		'''
		Sort utterances by frames length or uttID

		Args:
			<by>: "frame" or "utt"
			<reverse>: If reverse, sort in descending order.
		Return:
			A new NumpyData object.
		''' 
		if self.is_void:
			raise WrongOperation('No data to sort.')
		assert by in ["utt","frame"], "We only support sorting by 'name' or 'frame'."

		items = self.data.items()

		if by == "utt":
			items = sorted(items, key=lambda x:x[0], reverse=reverse)
		else:
			items = sorted(items, key=lambda x:len(x[1]), reverse=reverse)
		
		newData = {}
		for key, value in items:
			newData[key] = value
		
		newName = "sort({},{})".format(self.name, by)
		return NumpyData(newData, newName)

	def map(self, func):
		'''
		Map all arrays to a function.

		Args:
			<func>: callable function object.
		
		Return:
			A new NumpyData object.
		'''
		new = {}
		for utt, matrx in self.data.items():
			new[utt] = func(matrix)
		return NumpyData(new, name=f"mapped({self.name})")

class NumpyFeature(NumpyData):
	'''
	Hold the feature with Numpy format.
	'''
	def __init__(self, data={}, name="feat"):
		if isinstance(data, BytesFeature):
			data = (data.to_numpy()).data
		elif isinstance(data, NumpyFeature):
			data = data.data
		elif isinstance(data, dict):
			pass
		else:
			raise UnsupportedDataType(f"Expected BytesFeature, NumpyFeature or Python dict object but got {type_name(data)}.")	
		super().__init__(data,name)

	def to_dtype(self, dtype):
		'''
		Transform data type.

		Args:
			<dtype>: a string of "float", "float32" or "float64". IF "float", it will be treated as "float32".
		Return:
			A new NumpyFeature object.
		'''
		assert dtype in ["float", "float32", "float64"], '<dtype> must be a string of "float", "float32" or "float64".'

		result = super().to_dtype(dtype)

		return NumpyFeature(result.data, result.name)

	def to_bytes(self):
		'''
		Transform feature to bytes formation.

		Return:
			a BytesFeature object.
		'''		
		result = super().to_bytes()
		return BinaryFeature(result, self.name)
	
	def __add__(self, other):
		'''
		Plus operation between two feature objects.

		Args:
			<other>: a BytesFeature or NumpyFeature object.
		Return:
			a NumpyFeature object.
		'''		
		if isinstance(other, NumpyFeature):
			pass
		elif isinstance(other, BytesFeature):
			other = other.to_numpy()
		else:
			raise UnsupportedDataType(f'Excepted a BytesFeature or NumpyFeature object but got {type_name(other)}.')
		
		result = super().__add__(other)

		return NumpyFeature(result.data, result.name)

	def __call__(self, uttID):
		'''
		Pick out the specified utterance.
		
		Args:
			<uttID>: a string.
		Return:
			If existed, return a new NumpyFeature object.
			Or return None.
		''' 
		result = super().__call__(uttID)
		if result is None:
			return None
		else:
			return NumpyFeature(result.data, result.name)

	def splice(self, left=4, right=None):
		'''
		Splice front-behind N frames to generate new feature data.

		Args:
			<left>: the left N-frames to splice.
			<right>: the right N-frames to splice. If None, right = left.
		Return:
			a new NumpyFeature object whose dim became original-dim * (1 + left + right).
		''' 
		assert isinstance(left,int) and left >= 0, 'Expected <left> is non-negative int value.'
		if right == None:
			right = left
		else:
			assert isinstance(right,int) and right >= 0, 'Expected <right> is non-negative int value.'

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
		_dim = self.dim
		if isinstance(dims,int):
			assert dims >= 0, '<dims> should be a non-negative value.'
			assert dims < _dim, f"Selection index should be smaller than data dimension {_dim} but got {dims}."
			selectFlag = [dims,]
		elif isinstance(dims,str):
			if dims.strip() == "":
				raise WrongOperation("<dims> is not a dmensional value avaliable.")
			temp = dims.split(',')
			selectFlag = []
			for i in temp:
				if not '-' in i:
					try:
						i = int(i)
					except ValueError:
						raise WrongOperation(f'Expected int value but got {i}.')
					else:
						assert i >= 0, '<dims> should be a non-negative value.'
						assert i < _dim, f"Selection index should be smaller than data dimension {_dim} but got {i}."
						selectFlag.append(i)
				else:
					i = i.split('-')
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
						assert i[1] < _dim, f"Selection index should be smaller than data dimension {_dim} but got {i[1]}."
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
			newMat = np.concatenate(newMat,axis=1)
			seleDict[utt] = newMat
			if retain:
				if len(retainFlag) == _dim:
					continue
				else:
					matrix = selfdata.data[utt].copy()
					reseDict[utt] = np.delete(matrix,retainFlag,1)
		newNameSele = f"select({self.name},{dims})"
		if retain:
			newNameRese = f"select({self.name},not {dims})"
			return NumpyFeature(seleDict, newNameSele), NumpyFeature(reseDict, newNameRese)
		else:
			return NumpyFeature(seleDict, newNameSele)

	def subset(self, nHead=0, chunks=1, uttList=None):
		'''
		Subset feature.
		
		Args:
			<nHead>: If it > 0, extract N head utterances.
			<chunks>: If nHead == 0 and chunks > 1, split data into N chunks.
			<uttList>: If nHead == 0 and chunks == 1 and uttList != None, pick out these utterances whose ID in uttList.
		Return:
			a new NumpyFeature object or a list of new NumpyFeature objects.
		'''
		result = super().subset(nHead,chunks,uttList)

		if isinstance(result, list):
			for index in range(len(result)):
				temp = result[index]
				result[index] = NumpyFeature(temp.data, temp.name)
		else:
			result = NumpyFeature(result.data, result.name)

		return result

	def sort(self, by='frame', reverse=False):
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

	def normalize(self, std=True, alpha=1.0, beta=0.0, epsilon=1e-6, axis=0):
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
		if self.is_void:
			return NumpyFeature({})

		assert isinstance(epsilon,(float,int)) and epsilon > 0, "Expected <epsilon> is positive value."
		assert isinstance(alpha,(float,int)) and alpha > 0, "Expected <alpha> is positive value."
		assert isinstance(beta,(float,int)), "Expected <beta> is an int or float value."
		assert isinstance(axis,int), "Expected <axis> is an int value."

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
		if self.is_void:
			raise WrongOperation('No data to cut.')

		assert isinstance(maxFrames ,int) and maxFrames > 0, f"Expected <maxFrames> is positive int number but got {maxFrames}."

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
			if isinstance(other, NumpyFeature):                
				pass
			elif isinstance(other, BytesFeature):
				others[index] = other.to_numpy()    
			else:
				raise UnsupportedDataType(f'Excepted a NumpyFeature or BytesFeature object but got {type_name(other)}.')

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

class NumpyCMVNStatistics(NumpyData):
	'''
	Hold the CMVN statistics with Numpy format.
	'''
	def __init__(self, data={}, name="cmvn"):
		if isinstance(data, BytesCMVNStatistics):			
			data = (data.to_numpy()).data
		elif isinstance(data, NumpyCMVNStatistics):
			data = data.data
		elif isinstance(data, dict):
			pass
		else:
			raise UnsupportedDataType(f"Expected BytesCMVNStatistics, NumpyCMVNStatistics or Python dict object but got {type_name(data)}.")		
		super().__init__(data,name)

	def to_bytes(self):
		'''
		Transform feature to bytes formation.

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
		assert dtype in ["float", "float32", "float64"], '<dtype> must be a string of "float", "float32" or "float64".'

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
		if isinstance(other, NumpyCMVNStatistics):
			pass
		elif isinstance(other, BytesCMVNStatistics):
			other = other.to_numpy()
		else:
			raise UnsupportedDataType(f'Excepted a BytesCMVNStatistics or NumpyCMVNStatistics object but got {type_name(other)}.')
		
		result = super().__add__(other)

		return NumpyCMVNStatistics(result.data, result.name)

	def subset(self, nHead=0, chunks=1, uttList=None):
		'''
		Subset CMVN statistics.
		
		Args:
			<nHead>: If it > 0, extract N head utterances.
			<chunks>: If nHead == 0 and chunks > 1, split data into N chunks.
			<uttList>: If nHead == 0 and chunks == 1 and uttList != None, pick out these utterances whose ID in uttList.
		Return:
			a new NumpyCMVNStatistics object or a list of new NumpyCMVNStatistics objects.
		'''
		result = super().subset(nHead,chunks,uttList)

		if isinstance(result, list):
			for index in range(len(result)):
				temp = result[index]
				result[index] = NumpyCMVNStatistics(temp.data, temp.name)
		else:
			result = NumpyCMVNStatistics(result.data, result.name)

		return result

	def sort(self, by='frame', reverse=False):
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
			If existed, return a new NumpyFeature object.
			Or return None.
		''' 
		result = super().__call__(uttID)
		if result is None:
			return None
		else:
			return NumpyCMVNStatistics(result.data, result.name)

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

class NumpyProbability(NumpyData):
	'''
	Hold the probability with Numpy format.
	'''
	def __init__(self, data={}, name="postprob"):
		if isinstance(data, BytesProbability):
			data = (data.to_numpy()).data
		elif isinstance(data, NumpyProbability):
			data = data.data
		elif isinstance(data, dict):
			pass
		else:
			raise UnsupportedDataType(f"Expected BytesProbability, NumpyProbability or Python dict object but got {type_name(data)}.")		
		super().__init__(data,name)

	def to_bytes(self):
		'''
		Transform post probability to bytes formation.

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
		assert dtype in ["float", "float32", "float64"], '<dtype> must be a string of "float", "float32" or "float64".'

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
		if isinstance(other, NumpyProbability):
			pass
		elif isinstance(other, BytesProbability):
			other = other.to_numpy()
		else:
			raise UnsupportedDataType(f'Excepted a BytesProbability or NumpyProbability object but got {type_name(other)}.')
		
		result = super().__add__(other)

		return NumpyProbability(result.data, result.name)

	def subset(self, nHead=0, chunks=1, uttList=None):
		'''
		Subset post probability.
		
		Args:
			<nHead>: If it > 0, extract N head utterances.
			<chunks>: If nHead == 0 and chunks > 1, split data into N chunks.
			<uttList>: If nHead == 0 and chunks == 1 and uttList != None, pick out these utterances whose ID in uttList.
		Return:
			a new NumpyProbability object or a list of new NumpyProbability objects.
		'''
		result = super().subset(nHead,chunks,uttList)

		if isinstance(result, list):
			for index in range(len(result)):
				temp = result[index]
				result[index] = NumpyProbability( temp.data, temp.name )
		else:
			result = NumpyProbability( result.data,result.name )

		return result

	def sort(self, by='frame', reverse=False):
		'''
		Sort utterances by frames length or uttID

		Args:
			<by>: "frame" or "utt"
			<reverse>: If reverse, sort in descending order.
		Return:
			A new NumpyCMVNStatistics object.
		''' 			
		result = super().sort(by,reverse)

		return NumpyProbability(result.data, result.name)

	def __call__(self, uttID):
		'''
		Pick out the specified utterance.
		
		Args:
			<uttID>: a string.
		Return:
			If existed, return a new NumpyFeature object.
			Or return None.
		''' 
		result = super().__call__(uttID)
		if result is None:
			return None
		else:
			return NumpyProbability(result.data, result.name)

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

class NumpyAlignment(NumpyData):
	'''
	Hold the alignment with Numpy format.
	'''
	def __init__(self, data={}, name="ali"):
		if isinstance(data, NumpyAlignment):
			data = data.data
		elif isinstance(data, dict):
			pass
		else:
			raise UnsupportedDataType(f"Expected NumpyAlignment or Python dict object but got {type_name(data)}.")		
		super().__init__(data, name)
	
	def to_dtype(self, dtype):
		'''
		Transform data type.

		Args:
			<dtype>: a string of "int", "int32" or "int64". IF "int", it will be treated as "int32".
		Return:
			A new NumpyAlignment object.
		'''
		assert dtype in ["int", "int32", "int64"], '<dtype> must be a string of "int", "int32" or "int64".'

		result = super().to_dtype(dtype)

		return NumpyAlignment(result.data, result.name)

	def to_bytes(self):
		
		raise WrongOperation("Transforming to bytes is unavaliable").
	
	def __add__(self, other):
		'''
		Plus operation between two post probability objects.

		Args:
			<other>: a NumpyProbability or BytesProbability object.
		Return:
			a NumpyProbability object.
		'''	
		if isinstance(other, NumpyAlignment):
			pass
		else:
			raise UnsupportedDataType(f'Excepted a NumpyAlignment object but got {type_name(other)}.')
		
		result = super().__add__(other)

		return NumpyAlignment(result.data, result.name)

	def subset(self, nHead=0, chunks=1, uttList=None):
		'''
		Subset alignment data.
		
		Args:
			<nHead>: If it > 0, extract N head utterances.
			<chunks>: If nHead == 0 and chunks > 1, split data into N chunks.
			<uttList>: If nHead == 0 and chunks == 1 and uttList != None, pick out these utterances whose ID in uttList.
		Return:
			a new NumpyAlignment object or a list of new NumpyAlignment objects.
		'''
		result = super().subset(nHead,chunks,uttList)

		if isinstance(result, list):
			for index in range(len(result)):
				temp = result[index]
				result[index] = NumpyAlignment( temp.data, temp.name )
		else:
			result = NumpyAlignment( result.data,result.name )

		return result

	def sort(self, by='frame', reverse=False):
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
			If existed, return a new NumpyFeature object.
			Or return None.
		''' 
		result = super().__call__(uttID)
		if result is None:
			return None
		else:
			return NumpyAlignment(result.data, result.name)

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

class NumpyAlignmentTrans(NumpyAlignment):
	'''
	Hold the alignment(transition ID) with Numpy format.
	'''
	def __init__(self, data={}, name="transitionID"):
		if isinstance(data, NumpyAlignmentTrans):
			data = data.data
		elif isinstance(data, BytesAlignmentTrans):
			data = (data.to_numpy()).data
		elif isinstance(data, dict):
			pass
		else:
			raise UnsupportedDataType(f"Expected NumpyAlignmentTrans, BytesAlignmentTrans or Python dict object but got {type_name(data)}.")		
		super().__init__(data, name)

	def to_bytes(self):
		'''
		Tansform numpy alignment to bytes formation.

		Return:
			A BytesAlignmentTrans object.
		'''
		if self.is_void:
			return BytesAlignment(b"", self.name)

		ExkaldiInfo.vertify_kaldi_existed()
		
		temp = []
		for utt,matrix in self.data.items():
			new = utt + " ".join(map(str,matrix.tolist()))
			temp.append( new )
		temp = ("\n".join(temp)).encode()

		cmd = 'copy-int-vector ark:- ark:-'
		out, err, _ = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=temp)
		if out == b'':
			print(err.decode())
			raise KaldiProcessError('Failed to transform alignment to bytes formation.')
		else:
			return BytesAlignmentTrans(out, self.name)		

	def to_dtype(self):
		'''
		Transform data type.

		Args:
			<dtype>: a string of "int", "int32" or "int64". IF "int", it will be treated as "int32".
		Return:
			A new NumpyAlignment object.
		'''
		assert dtype in ["int", "int32", "int64"], '<dtype> must be a string of "int", "int32" or "int64".'

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
		if isinstance(other, NumpyAlignmentTrans):
			pass
		elif isinstance(other, BytesAlignmentTrans):
			other = other.to_numpy()
		else:
			raise UnsupportedDataType(f"Expected exkaldi NumpyAlignmentTrans or BytesAlignmentTrans object but got {type_name(other)}.")

		results = super().__add__(other)
		return NumpyAlignmentTrans(results.data, results.name)

	def subset(self, nHead=0, chunks=1, uttList=None):
		'''
		Subset alignment data.
		
		Args:
			<nHead>: If it > 0, extract N head utterances.
			<chunks>: If nHead == 0 and chunks > 1, split data into N chunks.
			<uttList>: If nHead == 0 and chunks == 1 and uttList != None, pick out these utterances whose ID in uttList.
		Return:
			a new NumpyAlignmentTrans object or a list of new NumpyAlignmentTrans objects.
		'''
		result = super().subset(nHead, chunks, uttList)

		if isinstance(result, list):
			for index in range(len(result)):
				temp = result[index]
				result[index] = NumpyAlignmentTrans(temp.data, temp.name)
		else:
			result = NumpyAlignmentTrans(result.data,result.name)

		return result
	
	def sort(self, by='frame', reverse=False):
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
		Transform tansition ID alignment to phone ID formation.

		Args:
			<hmm>: exkaldi HMM object or file path.
		Return:
			a NumpyAlignmentPhone object.
		'''		
		if self.is_void:
			return BytesAlignment(b"", self.name)		
		
		ExkaldiInfo.vertify_kaldi_existed()
		temp = []
		for utt,matrix in self.data.items():
			new = utt + " ".join(map(str,matrix.tolist()))
			temp.append( new )
		temp = ("\n".join(temp)).encode()

		model = tempfile.NamedTemporaryFile("wb+")
		try:
			if type_name(hmm) in ("HMM", "MonophoneHMM", "TriphoneHMM"):
				hmm.save(model)
				hmmFileName = model.name
			elif isinstance(hmm, str):
				if not os.path.isfile(hmm):
					raise WrongPath(f"No such file:{hmm}.")
				hmmFileName = model
			else:
				raise UnsupportedDataType(f"<hmm> should be a filePath or exkaldi HMM and its sub-class object. but got {type_name(hmm)}.") 

			cmd = f'copy-int-vector ark:- ark:- | ali-to-phones --per-frame=true {hmmFileName} ark:- ark,t:-'
			out, err, _ = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=temp)
			if out == b'':
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
				return NumpyAlignmentPhone(result, name="phoneID")

		finally:
			model.close()

	def to_pdfID(self, hmm):
		'''
		Transform tansition ID alignment to pdf ID formation.

		Args:
			<hmm>: exkaldi HMM object or file path.
		Return:
			a NumpyAlignmentPhone object.
		'''		
		if self.is_void:
			return BytesAlignment(b"", self.name)		
		
		ExkaldiInfo.vertify_kaldi_existed()
		temp = []
		for utt,matrix in self.data.items():
			new = utt + " ".join(map(str,matrix.tolist()))
			temp.append( new )
		temp = ("\n".join(temp)).encode()

		model = tempfile.NamedTemporaryFile("wb+")
		try:
			if type_name(hmm) in ("HMM", "MonophoneHMM", "TriphoneHMM"):
				hmm.save(model)
				hmmFileName = model.name
			elif isinstance(hmm, str):
				if not os.path.isfile(hmm):
					raise WrongPath(f"No such file:{hmm}.")
				hmmFileName = model
			else:
				raise UnsupportedDataType(f"<hmm> should be a filePath or exkaldi HMM and its sub-class object. but got {type_name(hmm)}.") 

			cmd = f'copy-int-vector ark:- ark:- | ali-to-phones --per-frame=true {hmmFileName} ark:- ark,t:-'
			out, err, _ = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=temp)
			if out == b'':
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
				return NumpyAlignmentPhone(result, name="pdfID")

		finally:
			model.close()

	def __call__(self, uttID):
		'''
		Pick out the specified utterance.
		
		Args:
			<uttID>: a string.
		Return:
			If existed, return a new NumpyFeature object.
			Or return None.
		''' 
		result = super().__call__(uttID)
		if result is None:
			return None
		else:
			return NumpyAlignmentTrans(result.data, result.name)

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

class NumpyAlignmentPhone(NumpyAchievement):
	'''
	Hold the alignment(phone ID) with Numpy format.
	'''
	def __init__(self, data={}, name="phoneID"):
		if isinstance(data, NumpyAlignmentPhone):
			data = data.data
		elif isinstance(data, dict):
			pass
		else:
			raise UnsupportedDataType(f"Expected NumpyAlignmentPhone or Python dict object but got {type_name(data)}.")		
		super().__init__(data, name)

	def __add__(self, other):
		'''
		The Plus operation between two phone ID alignment objects.

		Args:
			<other>: a NumpyAlignmentPhone object.
		Return:
			a new NumpyAlignmentPhone object.
		''' 
		if isinstance(other, NumpyAlignmentPhone):
			pass
		else:
			raise UnsupportedDataType(f"Expected exkaldi NumpyAlignmentPhone object but got {type_name(other)}.")

		results = super().__add__(other)
		return NumpyAlignmentPhone(results.data, results.name)

	def subset(self, nHead=0, chunks=1, uttList=None):
		'''
		Subset alignment data.
		
		Args:
			<nHead>: If it > 0, extract N head utterances.
			<chunks>: If nHead == 0 and chunks > 1, split data into N chunks.
			<uttList>: If nHead == 0 and chunks == 1 and uttList != None, pick out these utterances whose ID in uttList.
		Return:
			a new NumpyAlignmentPhone object or a list of new NumpyAlignmentPhone objects.
		'''
		result = super().subset(nHead, chunks, uttList)

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
			Or return None.
		''' 
		result = super().__call__(uttID)
		if result is None:
			return None
		else:
			return NumpyAlignmentPhone(result.data, result.name)

	def sort(self, by='frame', reverse=False):
		'''
		Sort utterances by frames length or uttID

		Args:
			<by>: "frame" or "utt"
			<reverse>: If reverse, sort in descending order.
		Return:
			A new NumpyAlignmentPhone object.
		''' 			
		result = super().sort(by,reverse)

		return NumpyAlignmentPhone(result.data, result.name)

	def to_bytes(self):
		
		raise WrongOperation("Transforming to bytes is unavaliable").

	def to_dtype(self):
		'''
		Transform data type.

		Args:
			<dtype>: a string of "int", "int32" or "int64". IF "int", it will be treated as "int32".
		Return:
			A new NumpyAlignmentPhone object.
		'''
		assert dtype in ["int", "int32", "int64"], '<dtype> must be a string of "int", "int32" or "int64".'

		result = super().to_dtype(dtype)

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

class NumpyAlignmentPdf(NumpyAchievement):
	'''
	Hold the alignment(pdf ID) with Numpy format.
	'''
	def __init__(self, data={}, name="phoneID"):
		if isinstance(data, NumpyAlignmentPdf):
			data = data.data
		elif isinstance(data, dict):
			pass
		else:
			raise UnsupportedDataType(f"Expected NumpyAlignmentPdf or Python dict object but got {type_name(data)}.")		
		super().__init__(data, name)

	def __add__(self, other):
		'''
		The Plus operation between two phone ID alignment objects.

		Args:
			<other>: a NumpyAlignmentPdf object.
		Return:
			a new NumpyAlignmentPdf object.
		''' 
		if isinstance(other, NumpyAlignmentPdf):
			pass
		else:
			raise UnsupportedDataType(f"Expected exkaldi NumpyAlignmentPdf object but got {type_name(other)}.")

		results = super().__add__(other)
		return NumpyAlignmentPdf(results.data, results.name)

	def subset(self, nHead=0, chunks=1, uttList=None):
		'''
		Subset alignment data.
		
		Args:
			<nHead>: If it > 0, extract N head utterances.
			<chunks>: If nHead == 0 and chunks > 1, split data into N chunks.
			<uttList>: If nHead == 0 and chunks == 1 and uttList != None, pick out these utterances whose ID in uttList.
		Return:
			a new NumpyAlignmentPdf object or a list of new NumpyAlignmentPdf objects.
		'''
		result = super().subset(nHead, chunks, uttList)

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
		if result is None:
			return None
		else:
			return NumpyAlignmentPdf(result.data, result.name)

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

	def to_bytes(self):
		
		raise WrongOperation("Transforming to bytes is unavaliable").

	def to_dtype(self):
		'''
		Transform data type.

		Args:
			<dtype>: a string of "int", "int32" or "int64". IF "int", it will be treated as "int32".
		Return:
			A new NumpyAlignmentPdf object.
		'''
		assert dtype in ["int", "int32", "int64"], '<dtype> must be a string of "int", "int32" or "int64".'

		result = super().to_dtype(dtype)

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

"""