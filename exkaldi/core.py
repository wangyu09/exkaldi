
# -*- coding: UTF-8 -*-
################# Version Information ################
# ExKaldi, version 0.2.1
# Yu Wang, Chee Siang Leow, Hiromitsu Nishizaki (University of Yamanashi)
# Akio Kobayashi (Tsukuba University of Technology)
# Takehito Utsuro (University of Tsukuba)
# Nov, 19, 2019
#
# ExKaldi Automatic Speech Recognition tookit is designed to build a interface between Kaldi and Deep Learning frameworks with Python Language.
# The main functions are implemented by Kaldi command, and based on this, we developed some extension tools:
# 1, Transform and deal with feature and label data of both Kaldi data format and NumPy format.
# 2, Design and train a neural network acoustic model.
# 3, Build a customized ASR system.
# More information in https://github.com/wangyu09/exkaldi
######################################################

import os,sys
import tempfile
import importlib
import math,random
import numpy as np
from io import BytesIO
import struct,copy,re,time
import subprocess,threading
from collections import Iterable

class PathError(Exception):pass
class WrongOperation(Exception):pass
class WrongDataFormat(Exception):pass
class KaldiProcessError(Exception):pass
class UnsupportedDataType(Exception):pass

ENV = None
KALDIROOT = None
kaidiNotFoundError = PathError('Kaldi ASR toolkit has not been found.')

def get_env():
	'''
	Usage:  ENV = get_env()

	Return the current environment which ExKaldi is running at.
	'''
	global ENV

	if ENV is None:
		ENV = os.environ.copy()

	return ENV

def get_kaldi_path():
	'''
	Usage:  kaldiRoot = get_kaldi_path() 

	Return the root directory of Kaldi toolkit. If the Kaldi has not been found, return None.
	'''
	global KALDIROOT

	if KALDIROOT == None:
		p = subprocess.Popen('which copy-feats',shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
		(out,err) = p.communicate()
		if out == b'':
			print("Warning: Kaldi has not been found. You can set it with .set_kaldi_path() function. If not, ERROR will occur when implementing part of core functions.")
		else:
			KALDIROOT = out.decode().strip()[0:-23]
	
	return KALDIROOT

_ = get_kaldi_path()

def set_kaldi_path(path):
	'''
	Usage: set_kaldi_path(path='/kaldi')

	Set the root directory of Kaldi toolkit manually.
	'''
	assert isinstance(path,str), '<path> should be a directory name-like string.'

	if not os.path.isdir(path):
		raise PathError('No such directory:{}.'.format(path))
	else:
		path = os.path.abspath(path.strip())

	p = subprocess.Popen('ls {}/src/featbin/copy-feats'.format(path),shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
	(out,err) = p.communicate()
	if out == b'':
		raise WrongOperation("{} is not a Kaldi root directory avaliable.".format(path))
	
	global KALDIROOT, ENV

	KALDIROOT = path
	ENV = os.environ.copy()

	systemPATH = []
	for i in ENV['PATH'].split(':'):
		if i.endswith('/tools/openfst'):
			continue
		elif i.endswith('/src/featbin'):
			continue
		elif i.endswith('/src/Gambian'):
			continue
		elif i.endswith('/src/nnetbin'):
			continue
		elif i.endswith('/src/bin'):
			continue
		else:
			systemPATH.append(i)
	systemPATH.append(path+'/src/bin')
	systemPATH.append(path+'/tools/openfst')
	systemPATH.append(path+'/src/featbin')
	systemPATH.append(path+'/src/Gambian')
	systemPATH.append(path+'/src/nnetbin')
	ENV['PATH'] = ":".join(systemPATH)

# ------------ Basic Classes ------------

class KaldiArk(bytes):
	'''
	Usage: obj = KaldiArk(binaryData) or obj = KaldiArk()
	
	KaldiArk is a subclass of bytes. It holds the Kaldi's ark data in binary type. 
	KaldiArk object can be transformed to visible form, KaldiDict object.
	'''
	def __init__(self,*args):
		super(KaldiArk,self).__init__()
	
	def _read_one_record(self,fp):
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
				raise WrongDataFormat('Miss utterance ID before utterance.')
		binarySymbol = fp.read(2).decode()
		if binarySymbol == '\0B':
			dataType = fp.read(3).decode() 
			if dataType == 'CM ':
				fp.close()
				raise UnsupportedDataType('This is compressed ark data. Use load(<arkFile>) function to load ark file again or \
											use decompress(<KaldiArk>) function to decompress it firstly.')                    
			elif dataType == 'FM ' or dataType == 'IM ':
				sampleSize = 4
			elif dataType == 'DM ' or dataType == 'UM ':
				sampleSize = 8
			else:
				fp.close()
				raise WrongDataFormat('Expected data type FM(float32),DM(float64),IM(int32),UM(int64),CM(compressed ark data) but got {}.'.format(dataType))
			s1,rows,s2,cols = np.frombuffer(fp.read(10), dtype='int8,int32,int8,int32', count=1)[0]
			rows = int(rows)
			cols = int(cols)
			buf = fp.read(rows * cols * sampleSize)
		else:
			fp.close()
			raise WrongDataFormat('Miss binary symbol before utterance.')
		return (utt,dataType,rows,cols,buf)
	
	def __str__(self):
		return "This is a KaldiArk object with unviewable binary data. To looking its content, please use .array method."

	@property
	def lens(self):
		'''
		Usage:  lengths = obj.lens
		
		Return a tuple: (the numbers of all utterances, the utterance ID and frames of each utterance).
		If there is not any data, return (0, None).
		'''
		if self == b"":
			lengths = (0, None)
		else:
			with BytesIO(self) as sp:
				_lens = {}
				while True:
					(utt,dataType,rows,cols,buf) = self._read_one_record(sp)
					if utt == None:
						break
					else:
						_lens[utt] = rows
			lengths = (len(_lens), _lens)
		
		return lengths
	
	@property
	def dim(self):
		'''
		Usage:  dimension = obj.dim
		
		Return an int value: data dimension.
		'''
		if self == b"":
			dimension = None
		else:
			with BytesIO(self) as sp:
				(utt,dataType,rows,cols,buf) = self._read_one_record(sp)
			dimension = cols
		
		return dimension
	
	@property
	def dtype(self):
		'''
		Usage:  dataType = obj.dtype
		
		Return a string: data type. We only use 'float32', 'float64', 'int32', 'int64'.
		'''
		if self == b"":
			_dtype = None
		else:
			with BytesIO(self) as sp:
				(utt,dataType,rows,cols,buf) = self._read_one_record(sp)
			if dataType == 'FM ':
				_dtype = 'float32'
			elif dataType == 'DM ':
				_dtype = 'float64'
			elif dataType == 'IM ':
				_dtype = 'int32'
			else:
				_dtype = 'int64'                
		return _dtype

	def to_dtype(self,dtype):
		'''
		Usage:  newObj = obj.to_dtype('float64')
		
		Return a new KaldiArk object. 'float' will be treated as 'float32' and 'int' will be 'int32'.
		'''
		assert isinstance(dtype,str) and (dtype in ["int", "int32", "int64", "float", "float32", "float64"]), 'Expected \
						<dtype> is string from "int", "int32", "int64", "float", "float32" or "float64" but got "{}"'.format(dtype)

		if self == b"" or self.dtype == dtype:
			result = copy.deepcopy(self)
		else:
			if dtype == 'float32' or dtype == 'float':
				newDataType = 'FM '
			elif dtype == 'float64':
				newDataType = 'DM '
			elif dtype == 'int32' or dtype == 'int':
				newDataType = 'IM '
			else:
				newDataType = 'UM '
			
			result = []
			with BytesIO(self) as sp:
				while True:
					(utt,dataType,rows,cols,buf) = self._read_one_record(sp)
					if utt == None:
						break
					if dataType == 'FM ': 
						matrix = np.frombuffer(buf, dtype=np.float32)
					elif dataType == 'DM ':
						matrix = np.frombuffer(buf, dtype=np.float64)
					elif dataType == 'IM ':
						matrix = np.frombuffer(buf, dtype=np.int32)
					else:
						matrix = np.frombuffer(buf, dtype=np.int64)
					newMatrix = np.array(matrix,dtype=dtype).tobytes()
					data = (utt+' '+'\0B'+newDataType).encode()
					data += '\04'.encode()
					data += struct.pack(np.dtype('uint32').char, rows)
					data += '\04'.encode()
					data += struct.pack(np.dtype('uint32').char, cols)
					data += newMatrix
					result.append(data)
			result = KaldiArk(b''.join(result))

		return result

	@property
	def utts(self):
		'''
		Usage:  utteranceIDs = obj.utts
		
		Return a list: including all utterance IDs.
		'''
		allUtts = []
		with BytesIO(self) as sp:
			while True:
				(utt,dataType,rows,cols,buf) = self._read_one_record(sp)
				if utt == None:
					break
				else:
					allUtts.append(utt)
		return allUtts
	
	def check_format(self):
		'''
		Usage: obj.check_format()
		
		Check whether data has a correct format of Kaldi ark data. If having, return True, or raise ERROR.
		'''
		if self != b'':
			_dim = 'unknown'
			_dataType = 'unknown'
			with BytesIO(self) as sp:
				while True: 
					(utt,dataType,rows,cols,buf) = self._read_one_record(sp)
					if utt == None:
						break
					if _dim == 'unknown':
						_dim = cols
						_dataType = dataType
					elif cols != _dim:
						raise WrongDataFormat("Expected dimension {} but got {} at utterance {}.".format(_dim,cols,utt))
					elif _dataType != dataType:
						raise WrongDataFormat("Expected data type {} but got {} at uttwerance {}.".format(_dataType,dataType,utt))                    
					else:
						try:
							if dataType == 'FM ':
								vec = np.frombuffer(buf, dtype=np.float32)
							elif dataType == 'DM ':
								vec = np.frombuffer(buf, dtype=np.float64)
							elif dataType == 'IM ':
								vec = np.frombuffer(buf, dtype=np.int32)
							else:
								vec = np.frombuffer(buf, dtype=np.int64)
						except Exception as e:
							print("Wrong matrix data format at utterance {}.".format(utt))
							raise e
			return True
		else:
			return False

	def concat(self,others,axis=1):
		raise WrongOperation('KaldiArk.concat() function has been removed in current version. Try to use KaldiDict.concat() please.')

	@property
	def array(self):
		'''
		Usage:  newObj = obj.array
		
		Return a KaldiDict object. Transform ark data into NumPy array data.
		'''
		newDict = KaldiDict()
		if self != b'':
			with BytesIO(self) as sp:
				while True:
					(utt,dataType,rows,cols,buf) = self._read_one_record(sp)
					if utt == None:
						break
					try:
						if dataType == 'FM ': 
							newMatrix = np.frombuffer(buf, dtype=np.float32)
						elif dataType == 'DM ':
							newMatrix = np.frombuffer(buf, dtype=np.float64)
						elif dataType == 'IM ':
							newMatrix = np.frombuffer(buf, dtype=np.int32)
						else:
							newMatrix = np.frombuffer(buf, dtype=np.int64)
					except Exception as e:
						print("Wrong matrix data format at utterance {}.".format(utt))
						raise e	
					else:				
						newDict[utt] = np.reshape(newMatrix,(rows,cols))
		return newDict
	
	def save(self,fileName,chunks=1,outScpFile=False):
		'''
		Usage: obj.save('feat.ark')
		
		Save as .ark (and .scp) file. If <chunks> is larger than "1", split it averagely and save them.
		'''        
		if self == b'':
			raise WrongOperation('No data to save.')

		#if sys.getsizeof(self)/chunks > 10000000000:
		#   print("Warning: Data size is extremely large. Try to save it with a long time.")

		global KALDIROOT,kaidiNotFoundError,ENV
		if KALDIROOT is None:
			raise kaidiNotFoundError

		def save_chunk_data(chunkData,fileName,outScpFile):
			if outScpFile is True:
				cmd = "copy-feats ark:- ark,scp:{},{}".format(fileName,fileName[0:-3]+"scp")
			else:
				cmd = "copy-feats ark:- ark:{}".format(fileName)
			p = subprocess.Popen(cmd,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=ENV)
			(out,err) = p.communicate(input=chunkData)
			if not os.path.isfile(fileName) or os.path.getsize(fileName) == 0:
				err = err.decode()
				print(err)
				if os.path.isfile(fileName):
					os.remove(fileName)
				raise KaldiProcessError('Save ark data defeated.')
			else:
				if outScpFile is True:
					return (fileName,fileName[0:-3]+"scp")
				else:
					return fileName
		
		if self.dtype == 'int32':
			savingData = self.to_dtype('float32')
		elif self.dtype == 'int64':
			savingData = self.to_dtype('float64')
		else:
			savingData = self

		if chunks == 1:
			if not fileName.strip().endswith('.ark'):
				fileName += '.ark'
			savedFilesName = save_chunk_data(savingData,fileName,outScpFile)		
		else:
			if fileName.strip().endswith('.ark'):
				fileName = fileName[0:-4]
			with BytesIO(savingData) as sp:
				uttLens = []
				while True:
					(utt,dataType,rows,cols,buf) = self._read_one_record(sp)
					if utt == None:
						break
					if dataType == 'DM ' or dataType == 'UM ':
						sampleSize = 8
					else:
						sampleSize = 4
					oneRecordLen = len(utt) + 16 + rows * cols * sampleSize
					uttLens.append(oneRecordLen)
				sp.seek(0)
				allLens = len(uttLens)
				chunkUtts = allLens//chunks
				if chunkUtts == 0:
					chunks = allLens
					chunkUtts = 1
					t = 0
					print("Warning: utterances is fewer than <chunks> so only {} files will be saved.".format(chunks))
				else:
					t = allLens - chunkUtts * chunks

				savedFilesName = []
				for i in range(chunks):
					if i < t:
						chunkLen = sum(uttLens[i*(chunkUtts+1):(i+1)*(chunkUtts+1)])
					else:
						chunkLen = sum(uttLens[i*chunkUtts:(i+1)*chunkUtts])
					chunkData = sp.read(chunkLen)
					savedFilesName.append(save_chunk_data(chunkData,fileName+'_ck{}.ark'.format(i),outScpFile))

		return savedFilesName
		
	def __add__(self,other):
		'''
		Usage:  obj3 = obj1 + obj2
		
		Return a new KaldiArk object. obj2 can be KaldiArk or KaldiDict object.
		Note that if there are the same utterance ID in both obj1 and obj2, data only in the formar will be retained.
		''' 
		if isinstance(other,KaldiArk):
			pass
		elif isinstance(other,KaldiDict):
			other = other.ark
		else:
			raise UnsupportedDataType('Excepted a KaldiArk or KaldiDict object but got {}.'.format(type(other)))
		
		if self.dim != None and other.dim != None and self.dim != other.dim:
			raise WrongOperation('Expected unified dimenson but {}!={}.'.format(self.dim,other.dim))        

		selfUtts = self.utts
		selfDtype = self.dtype
		newData = []
		with BytesIO(other) as op:
			while True:
				(outt,odataType,orows,ocols,obuf) = self._read_one_record(op)
				if outt == None:
					break
				elif not outt in selfUtts:
					if odataType == 'FM ': 
						temp = 'float32'
					elif odataType == 'DM ':
						temp = 'float64'
					elif odataType == 'IM ':
						temp = 'int32'
					else:
						temp = 'int64'
					if selfDtype != None and odataType != None and selfDtype != temp:
						obuf = np.array(np.frombuffer(obuf,dtype=temp),dtype=selfDtype).tobytes()
					data = (outt+' ').encode()
					data += '\0B'.encode()        
					data += odataType.encode()
					data += '\04'.encode()
					data += struct.pack(np.dtype('uint32').char, orows)
					data += '\04'.encode()
					data += struct.pack(np.dtype('uint32').char, ocols)
					data += obuf
					newData.append(data)

		return KaldiArk(b''.join([self,*newData]))

	def splice(self,left=1,right=None):
		'''
		Usage:  newObj = obj.splice(4) or newObj = obj.splice(4,3)
		
		Return a new KaldiArk object. If <right> is None, we define: right = left. If you don't want to splice the right frames, set it "0".
		''' 
		assert isinstance(left,int) and left >= 0, "Expected <left> is a positive int number."

		global KALDIROOT,kaidiNotFoundError,ENV
		if KALDIROOT is None:
			raise kaidiNotFoundError
		
		if right == None:
			right = left
		else:
			assert isinstance(right,int) and right >= 0, "Expected <right> is a positive int number."
		
		if self.dtype == 'int32':
			temp = self.to_dtype('float32')
		elif self.dtype == 'int64':
			temp = self.to_dtype('float64')
		else:
			temp = self

		cmd = 'splice-feats --left-context={} --right-context={} ark:- ark:-'.format(left,right)
		p = subprocess.Popen(cmd,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=ENV)
		(out,err) = p.communicate(input=temp)

		if out == b'':
			err = err.decode()
			print(err)
			raise KaldiProcessError("Splice left-right frames defeated.")
		else:
			result = KaldiArk(out)
			if self.dtype == 'int32':
				result = result.to_dtype('int32')
			elif self.dtype == 'int64':
				result = result.to_dtype('int64')
			return result

	def select(self,dims,retain=False):
		'''
		Usage:  newObj = obj.select(4) or newObj1,newObj2 = obj.select('5,10-15',retain=True)
		
		Select data by pointing dimensions. <dims> should be an int value or string like "1,5-20".
		If <retain> is "True", return two KaldiArk objects of both selected data and non-selected data.
		''' 
		global KALDIROOT,kaidiNotFoundError,ENV
		if KALDIROOT is None:
			raise kaidiNotFoundError

		_dim = self.dim
		if _dim == 1:
			raise WrongOperation('Cannot select any data from 1-dim data.')

		elif isinstance(dims,int):
			assert dims >= 0, "Expected <dims> is a non-negative int number."
			assert dims < _dim, "Selection index should be smaller than data dimension {} but got {}.".format(_dim,dims)
			selectFlag = str(dims)
			if retain:
				if dims == 0:
					retainFlag = '1-{}'.format(_dim-1)
				elif dims == _dim-1:
					retainFlag = '0-{}'.format(_dim-2)
				else:
					retainFlag = '0-{},{}-{}'.format(dims-1,dims+1,_dim-1)
		elif isinstance(dims,str):
			
			if dims.strip() == "":
				raise WrongOperation("<dims> is not a dmensional value avaliable.")

			if retain:
				retainFlag = [x for x in range(_dim)]
				for i in dims.strip().split(','):
					if i.strip() == "":
						continue
					if not '-' in i:
						try:
							i = int(i)
						except ValueError:
							raise WrongOperation('Expected int value but got {}.'.format(i))
						else:
							assert i >= 0, "Expected <dims> is a non-negative int number."
							assert i < _dim, "Selection index should be smaller than data dimension {} but got {}.".format(_dim,i)
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
							assert i[1] < _dim, "Selection index should be smaller than data dimension {} but got {}.".format(_dim, i[1])
						for j in range(i[0],i[1]+1,1):
							retainFlag[j] = -1
				temp = ''
				for x in retainFlag:
					if x != -1:
						temp += str(x)+','
				retainFlag = temp[0:-1]
			selectFlag = dims
		else:
			raise WrongOperation('Expected int value or string like "1,4-9,12" but got {}.'.format(type(dims)))
		
		if self.dtype == 'int32':
			temp = self.to_dtype('float32')
		elif self.dtype == 'int64':
			temp = self.to_dtype('float64')
		else:
			temp = self

		cmdS = 'select-feats {} ark:- ark:-'.format(selectFlag)
		pS = subprocess.Popen(cmdS,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=ENV)
		(outS,errS) = pS.communicate(input=temp)
		if outS == b'':
			errS = errS.decode()
			print(errS)
			raise KaldiProcessError("Select data defeated.")
		else:
			selectedResult = KaldiArk(outS)

		if retain:
			if retainFlag == "":
				retainedResult = KaldiArk()
			else: 
				cmdR = 'select-feats {} ark:- ark:-'.format(retainFlag)
				pR = subprocess.Popen(cmdR,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=ENV)
				(outR,errR) = pR.communicate(input=temp)
				if outR == b'':
					errR = errR.decode()
					print(errS)
					raise KaldiProcessError("Select retained data defeated.")
				else:
					retainedResult = KaldiArk(outR)
		
			if self.dtype == 'int32':
				return selectedResult.to_dtype('int32'),retainedResult.to_dtype('int32')
			elif self.dtype == 'int64':
				return selectedResult.to_dtype('int64'),retainedResult.to_dtype('int64')
			else:
				return selectedResult,retainedResult
		else:
			if self.dtype == 'int32':
				return selectedResult.to_dtype('int32')
			elif self.dtype == 'int64':
				return selectedResult.to_dtype('int64')
			else:
				return selectedResult

	def subset(self,nHead=0,chunks=1,uttList=None):
		'''
		Usage:  newObj = obj.subset(nHead=10) or newObj = obj.subset(chunks=10) or newObj = obj.subset(uttList=uttList)
		
		If <nHead> is larger than "0", return a new KaldiArk object whose content is start <nHead> pieces of data. 
		Or if chunks is larger than "1", split all data averagely as N KaidiArk objects. Return a list.
		Or if <uttList> is not "None", select utterances if they appeared in obj. Return a KaldiArk object of selected data.
		''' 
		if nHead > 0:
			assert isinstance(nHead,int), "Expected <nHead> is an int number but got {}.".format(nHead)
			
			if len(self) == 0:
				return KaldiArk()

			with BytesIO(self) as sp:
				uttLens = []
				while nHead > 0:
					(utt,dataType,rows,cols,buf) = self._read_one_record(sp)
					if utt == None:
						break
					if dataType == 'DM ' or dataType == 'UM ':
						sampleSize = 8
					else:
						sampleSize = 4
					oneRecordLen = len(utt) + 16 + rows * cols * sampleSize
					uttLens.append(oneRecordLen) 
					nHead -= 1       
				sp.seek(0)
				data = sp.read(sum(uttLens))
			return KaldiArk(data)
		
		elif chunks > 1:
			assert isinstance(chunks,int), "Expected <chunks> is an int number but got {}.".format(chunks)

			if len(self) == 0:
				return []

			datas = []
			with BytesIO(self) as sp:
				uttLens = []
				while True:
					(utt,dataType,rows,cols,buf) = self._read_one_record(sp)
					if utt == None:
						break
					if dataType == 'DM ' or dataType == 'UM ':
						sampleSize = 8
					else:
						sampleSize = 4
					oneRecordLen = len(utt) + 16 + rows * cols * sampleSize
					uttLens.append(oneRecordLen)                                
				sp.seek(0)
				allLens = len(uttLens)
				chunkUtts = allLens//chunks
				if chunkUtts == 0:
					chunks = allLens
					chunkUtts = 1
					t = 0
				else:
					t = allLens - chunkUtts * chunks
				for i in range(chunks):
					if i < t:
						chunkLen = sum(uttLens[i*(chunkUtts+1):(i+1)*(chunkUtts+1)])
					else:
						chunkLen = sum(uttLens[i*chunkUtts:(i+1)*chunkUtts])
					chunkData = sp.read(chunkLen)
					datas.append(KaldiArk(chunkData))
			return datas

		elif uttList != None:
			if len(self) == 0:
				return KaldiArk()
			if isinstance(uttList,str):
				uttList = [uttList,]
			elif isinstance(uttList,(list,tuple)):
				pass
			else:
				raise UnsupportedDataType('Expected <uttList> is string, list or tuple but got {}.'.format(type(uttList)))

			newData = []
			with BytesIO(self) as sp:
				while True:
					(utt,dataType,rows,cols,buf) = self._read_one_record(sp)
					if utt == None:
						break
					elif utt in uttList:
						data = b''
						data = (utt+' \0B'+dataType).encode()
						data += '\04'.encode()
						data += struct.pack(np.dtype('uint32').char, rows)
						data += '\04'.encode()
						data += struct.pack(np.dtype('uint32').char, cols)
						data += buf
						newData.append(data)
			return KaldiArk(b''.join(newData))

		else:
			raise WrongOperation('Expected <nHead> is larger than "0", or <chunks> is larger than "1", or <uttList> is not None.')

class KaldiDict(dict):
	'''
	Usage:  obj = KaldiDict(binaryData) or obj = KaldiDict()

	KaldiDict is a subclass of Python dict. It is visible form of KaldiArk and holds the feature data and aligment data with NumPy array type. 
	Its keys are the names of all utterances and the values are the data. KaldiDict object can also implement some mixed operations with KaldiArk such as "+" and so on.
	Note that KaldiDict has some specific functions which KaldiArk does not have.
	'''
	def __init__(self,*args):
		super(KaldiDict,self).__init__(*args)
		self.check_format()

	def __add_dim_to_1dimData(self):
		for utt in self.keys():
			if len(self[utt].shape) == 1:
				self[utt] = self[utt][None,:]

	@property
	def dim(self):
		'''
		Usage:  dimension = obj.dim
		
		Return an int value: data dimension.
		'''
		_dim = None
		if len(self.keys()) != 0:
			utt = list(self.keys())[0]
			if len(self[utt].shape) == 1:
				self.__add_dim_to_1dimData()
			_dim = self[utt].shape[1]
		return _dim

	@property
	def target(self):
		'''
		Usage:  label_classes = obj.target
		
		Return an int value: the classes of alignment data.
		'''
		if len(self.keys()) == 0:
			return None        
		maxValue = None
		for utt in self.keys():
			if not (self[utt].dtype in ["int8","int16","int32","int64"]):
				raise WrongOperation('Cannot obtain target from float data.')
			if len(self[utt].shape) == 1:
				self[utt] = self[utt][None,:]
			t = np.max(self[utt])
			if maxValue == None or t > maxValue:
				maxValue = t
		return int(maxValue) + 1

	@property
	def lens(self):
		'''
		Usage: length = obj.lens
		Return a tuple: (the numbers of all utterances, the utterance IDs and frames of each utterance). 
		If there is not any data, return (0, None).
		'''
		_lens = None
		allUtts = self.keys()
		if len(allUtts) != 0:
			_lens = {}
			for utt in allUtts:
				if len(self[utt].shape) == 1:
					self[utt] = self[utt][None,:]
				_lens[utt] = len(self[utt])
		if _lens == None:
			return (0,None)
		else:
			return (len(_lens),_lens)

	@property
	def dtype(self):
		'''
		Usage:  dataType = obj.dtype
		
		Return a string: data type. We only use 'float32', 'float64', 'int32', 'int64'.
		'''        
		_dtype = None
		if len(self.keys()) != 0:
			utt = list(self.keys())[0]
			_dtype = str(self[utt].dtype) 
			if len(self[utt].shape) == 1:
				self.__add_dim_to_1dimData() 
		return _dtype
	
	def to_dtype(self,dtype):
		'''
		Usage:  newObj = obj.to_dtype('float')

		Return a new KaldiArk object. 'float' will be treated as 'float32' and 'int' will be 'int32'.
		'''
		self.__add_dim_to_1dimData() 
		if self.dtype != dtype:
			assert dtype in ['int','int32','int64','float','float32','float64'],'Expected <dtype> is "int", "int32", "int64", "float", "float32" or "float64" but got {}.'.format(dtype)
			if dtype == 'int': 
				dtype = 'int32'
			elif dtype == 'float': 
				dtype = 'float32'
			newData = KaldiDict()
			for utt in self.keys():
				newData[utt] = np.array(self[utt],dtype=dtype)
			return newData
		else:
			return self

	@property
	def utts(self):
		'''
		Usage:  utteranceIDs = obj.utts
		
		Return a list object: including all utterance IDs.
		'''
		self.__add_dim_to_1dimData()

		return list(self.keys())
	
	def check_format(self):
		'''
		Usage:  obj.check_format()
		
		Check whether data has a correct format of Kaldi ark data. If having, return True, or raise ERROR.
		'''
		if len(self.keys()) != 0:
			_dim = 'unknown'
			for utt in self.keys():
				if not isinstance(utt,str):
					raise WrongDataFormat('Expected utterance ID is a string but got {}.'.format(type(utt)))
				if not isinstance(self[utt],np.ndarray):
					raise WrongDataFormat('Expected value is NumPy ndarray but got {}.'.format(type(self[utt])))
				if len(self[utt].shape) == 1:
					self[utt] = self[utt][None,:]
				matrixShape = self[utt].shape
				if len(matrixShape) > 2:
					raise WrongDataFormat('Expected the shape of matrix is like [ frame length, dimension ] but got {}.'.format(matrixShape))
				if len(matrixShape) == 2:
					if _dim == 'unknown':
						_dim = matrixShape[1]
					elif matrixShape[1] != _dim:
						raise WrongDataFormat("Expected uniform data dimension {} but got {} at utt {}.".format(_dim,matrixShape[1],utt))
			return True
		else:
			return False
	
	@property
	def ark(self):
		'''
		Usage:  newObj = obj.ark
		
		Return a KaldiArk object. Transform NumPy array data into ark binary data.
		'''

		#totalSize = 0
		#for u in self.keys():
		#    totalSize += sys.getsizeof(self[u])
		#if totalSize > 10000000000:
		#    print('Warning: Data is extramely large. Try to transform it but it maybe result in MemoryError.')

		newData = []
		for utt in self.keys():
			if len(self[utt].shape) == 1:
				self[utt] = self[utt][None,:]
			matrix = self[utt]
			data = (utt+' ').encode()
			data += '\0B'.encode()
			if matrix.dtype == 'float32':
				data += 'FM '.encode()
			elif matrix.dtype == 'float64':
				data += 'DM '.encode()
			elif matrix.dtype == 'int32':
				data += 'IM '.encode()
			elif matrix.dtype == 'int64':
				data += 'UM '.encode()
			else:
				raise UnsupportedDataType('Expected "int32", "int64", "float32" or "float64" data, but got {}.'.format(matrix.dtype))
			data += '\04'.encode()
			data += struct.pack(np.dtype('uint32').char, matrix.shape[0])
			data += '\04'.encode()
			data += struct.pack(np.dtype('uint32').char, matrix.shape[1])
			data += matrix.tobytes()
			newData.append(data)

		return KaldiArk(b''.join(newData))

	def save(self,fileName,chunks=1):
		'''
		Usage:  obj.save('feat.npy') or obj.save('feat.npy',chunks=2)
		
		Save as .npy file. If <chunks> is larger than 1, split it averagely and save them.
		'''      
		if len(self.keys()) == 0:
			raise WrongOperation('No data to save.')

		#totalSize = 0
		#for u in self.keys():
		#    totalSize += sys.getsizeof(self[u])
		#if totalSize > 10000000000:
		#    print('Warning: Data size is extremely large. Try to save it with a long time.')
		
		if fileName.strip().endswith('.npy'):
			fileName = fileName[0:-4]

		if chunks == 1:          
			allData = tuple(self.items())
			np.save(fileName,allData)
			savedFilesName = fileName + '.npy'
		else:
			allData = tuple(self.items())
			allLens = len(allData)
			chunkUtts = allLens//chunks
			if chunkUtts == 0:
				chunks = allLens
				chunkUtts = 1
				t = 0
				print("Warning: utterances is fewer than <chunks> so only {} files will be saved.".format(chunks))
			else:
				t = allLens - chunkUtts * chunks
			savedFilesName = []
			for i in range(chunks):
				if i < t:
					chunkData = allData[i*(chunkUtts+1):(i+1)*(chunkUtts+1)]
				else:
					chunkData = allData[i*chunkUtts:(i+1)*chunkUtts]
				np.save(fileName+'_ck{}.npy'.format(i),chunkData)
				savedFilesName.append(fileName+'_ck{}.npy'.format(i))
		
		return savedFilesName

	def __add__(self,other):
		'''
		Usage:  obj3 = obj1 + obj2
		
		Return a new KaldiDict object. obj2 can be KaldiArk or KaldiDict object.
		Note that if there are the same utterance ID in both obj1 and obj2, data only in the formar will be retained.
		''' 
		if isinstance(other,KaldiDict):
			pass         
		elif isinstance(other,KaldiArk):
			other = other.array
		else:
			raise UnsupportedDataType('Expected a KaldiArk or KaldiDict object but got {}.'.format(type(other)))
	
		if self.dim != None and other.dim != None and self.dim != other.dim:
			raise WrongDataFormat('Expected unified dimension but {}!={}.'.format(self.dim,other.dim))

		self.__add_dim_to_1dimData()

		temp = self.copy()
		selfUtts = list(self.keys())
		for utt in other.keys():
			if len(other[utt].shape) == 1:
				other[utt] = other[utt][None,:]
			if not utt in selfUtts:
				temp[utt] = other[utt]
		return KaldiDict(temp)
	
	def concat(self,others,axis=1):
		'''
		Usage:  obj3 = obj1.concat(obj2) or newObj = obj1.concat([obj2,obj3....])
		
		Return a KaldiDict object. obj2,obj3... can be KaldiArk or KaldiDict objects.
		Note that only these utterance IDs which appeared in all objects can be retained in concatenated result. 
		When one member of the data only has a dim or only have one frames and axis is 1, it will be concatenated to all frames.
		''' 
		if axis != 1 and axis != 0:
			raise WrongOperation('Expected <axis> is "1" or "0" but got {}.'.format(axis))

		if not isinstance(others,(list,tuple)):
			others = [others,]

		for index,other in enumerate(others):
			if isinstance(other,KaldiDict):                   
				pass
			elif isinstance(other,KaldiArk):
				others[index] = other.array       
			else:
				raise UnsupportedDataType('Excepted a KaldiArk or KaldiDict object but got {}.'.format(type(other))) 

		newDict = KaldiDict()

		for utt in self.keys():

			newMat=[]
			
			if len(self[utt].shape) == 1:
				self[utt] = self[utt][None,:]

			if axis == 1:
				adjustFrameIndexs = []
				if self[utt].shape[0] == 1:
					adjustFrameIndexs.append(0)
				maxFrames = self[utt].shape[0]

			newMat.append(self[utt])
			length = self[utt].shape[0]
			dim = self[utt].shape[1]
			
			for index,other in enumerate(others,start=1):
				if utt in other.keys():
					if len(other[utt].shape) == 1:
						other[utt] = other[utt][None,:]
					if axis == 1:
						if other[utt].shape[0] > maxFrames:
							maxFrames = other[utt].shape[0]
						if other[utt].shape[0] == 1:
							adjustFrameIndexs.append(index)
						elif other[utt].shape[0] != length:
							raise WrongDataFormat("Data frames {}!={} at utterance ID {}.".format(length,other[utt].shape[0],utt))
					elif axis == 0 and other[utt].shape[1] != dim:
						raise WrongDataFormat("Data dims {}!={} at utterance ID {}.".format(dim,other[utt].shape[1],utt))
					newMat.append(other[utt])
				else:
					#print("Concat Warning: Miss data of utt id {} in later dict".format(utt))
					break
			if len(newMat) < len(others) + 1:
				#If any member miss the data of current utt id, abandon data of this utt id of all menbers
				continue
			if axis == 0:
				newDict[utt] = np.concatenate(newMat,axis=0)
			else:
				if len(adjustFrameIndexs) > 0:
					for index in adjustFrameIndexs:
						new = []
						for i in range(maxFrames):
							new.append(newMat[index][0])
					newMat[index] = np.row_stack(new)
				newDict[utt] = np.concatenate(newMat,axis=1)
		return newDict

	def splice(self,left=4,right=None):
		'''
		Usage:  newObj = obj.splice(4) or newObj = obj.splice(4,3)
		
		Return a new KaldiDict object. If <right> is None, we define: right = left. If you don't want to splice, set the value zero.
		''' 
		assert isinstance(left,int) and left >= 0, 'Expected <left> is non-negative int value.'
		if right == None:
			right = left
		else:
			assert isinstance(right,int) and right >= 0, 'Expected <right> is non-negative int value.'

		lengths = []
		matrixes = []
		for utt in self.keys():
			if len(self[utt].shape) == 1:
				self[utt] = self[utt][None,:]
			lengths.append((utt,len(self[utt])))
			matrixes.append(self[utt])

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

		newFea = KaldiDict()
		index = 0
		for utt,length in lengths:
			newFea[utt] = newMat[index:index+length]
			index += length
		return newFea
	
	def select(self,dims,retain=False):
		'''
		Usage:  newObj = obj.select(4) or newObj1,newObj2 = obj.select('5,10-15',True)
		
		Select dimensions data. <dims> should be an int value or string like "1,5-20".
		If <retain> is True, return two new KaldiDict objects concluding both slected data and non-selected data.
		'''
		_dim = self.dim
		if _dim == 1:
			raise WrongOperation('Cannot select any data from 1-dim data.')
		elif isinstance(dims,int):
			assert dims >= 0, '<dims> should be a non-negative value.'
			assert dims < _dim, "Selection index should be smaller than data dimension {} but got {}.".format(_dim,dims)
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
						raise WrongOperation('Expected int value but got {}.'.format(i))
					else:
						assert i >= 0, '<dims> should be a non-negative value.'
						assert i < _dim, "Selection index should be smaller than data dimension {} but got {}.".format(_dim,i)
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
						assert i[1] < _dim, "Selection index should be smaller than data dimension {} but got {}.".format(_dim, i[1])
						selectFlag.extend([x for x in range(int(i[0]),int(i[1])+1)])
		else:
			raise WrongOperation('Expected <dims> is int value or string like 1,4-9,12 but got {}.'.format(type(dims)))

		retainFlag = sorted(list(set(selectFlag)))

		seleDict = KaldiDict()
		if retain:
			reseDict = KaldiDict()
		for utt in self.keys():
			if len(self[utt].shape) == 1:
				self[utt] = self[utt][None,:]
			newMat = []
			for index in selectFlag:
				newMat.append(self[utt][:,index][:,None])
			newMat = np.concatenate(newMat,axis=1)
			seleDict[utt] = newMat
			if retain:
				if len(retainFlag) == _dim:
					continue
				else:
					matrix = self[utt].copy()
					reseDict[utt] = np.delete(matrix,retainFlag,1)
		if retain:
			return seleDict,reseDict
		else:
			return seleDict

	def subset(self,nHead=0,chunks=1,uttList=None):
		'''
		Usage:  newObj = obj.subset(nHead=10) or newObj = obj.subset(chunks=10) or newObj = obj.subset(uttList=uttList)
		
		Subset data.
		If nHead > 0, return a new KaldiDict object whose content is front nHead pieces of data. 
		Or If chunks > 1, split data averagely as chunks KaidiDict objects. Return a list.
		Or If uttList != None, select utterances if appeared in obj. Return selected data.
		
		''' 
		self.__add_dim_to_1dimData()

		if nHead > 0:
			assert isinstance(nHead,int), "Expected <nHead> is an int number but got {}.".format(nHead)
			newDict = KaldiDict()
			utts = list(self.keys())
			for utt in utts[0:nHead]:
				newDict[utt]=self[utt]
			return newDict

		elif chunks > 1:
			assert isinstance(chunks,int), "Expected <chunks> is an int number but got {}.".format(chunks)
			datas = []
			allLens = len(self)
			if allLens != 0:
				utts = list(self.keys())
				chunkUtts = allLens//chunks
				if chunkUtts == 0:
					chunks = allLens
					chunkUtts = 1
					t = 0
				else:
					t = allLens - chunkUtts * chunks

				for i in range(chunks):
					datas.append(KaldiDict())
					if i < t:
						chunkUttList = utts[i*(chunkUtts+1):(i+1)*(chunkUtts+1)]
					else:
						chunkUttList = utts[i*chunkUtts:(i+1)*chunkUtts]
					for utt in chunkUttList:
						datas[i][utt]=self[utt]
			return datas

		elif uttList != None:

			if len(self.keys()) == 0:
				return KaldiDict()
			
			if isinstance(uttList,str):
				uttList = [uttList,]
			elif isinstance(uttList,(list,tuple)):
				pass
			else:
				raise UnsupportedDataType('Expected <uttList> is a string,list or tuple but got {}.'.format(type(uttList)))

			newDict = KaldiDict()
			selfKeys = list(self.keys())
			for utt in uttList:
				if utt in selfKeys:
					newDict[utt] = self[utt]
				else:
					#print('Subset Warning: no data for utt {}'.format(utt))
					continue
			return newDict
		else:
			raise WrongOperation('Expected <nHead> is larger than "0", or <chunks> is larger than "1", or <uttList> is not None.')

	def sort(self,by='frame',reverse=False):
		'''
		Usage:  newObj = obj.sort()
		
		Sort data by frame "length" or "name". Return a new KaldiDict object.
		''' 
		if len(self.keys()) == 0:
			raise WrongOperation('No data to sort.')

		assert by in ['frame','name'], 'We only support sorting by "name" or "frame".'

		self.__add_dim_to_1dimData()
		items = self.items()

		if by == 'name':
			items = sorted(items,key=lambda x:x[0],reverse=reverse)
		else:
			items = sorted(items,key=lambda x:len(x[1]),reverse=reverse)
		
		newData = KaldiDict()
		for key, value in items:
			newData[key] = value
		
		return newData 

	def merge(self,keepDim=False,sortFrame=False):
		'''
		Usage:  data,uttlength = obj.merge() or data,uttlength = obj.merge(keepDim=True)
		
		Return two value.
		If <keepDim> is "False", the first one is 2-dimensional NumPy array, the second one is a list consists of utterance ID and frames of each utterance. 
		If <keepDim> is "True", the first one will be a list whose members are sequences of all utterances.
		If <sortFrame> is "True", it will sort by length of matrix before merge.
		''' 
		if len(self.keys()) == 0:
			raise WrongOperation('Not any data to merge.') 

		uttLens = {}
		matrixs = []

		self.__add_dim_to_1dimData()

		if sortFrame is True:          
			items = sorted(self.items(), key=lambda x:len(x[1]))
		else:
			items = self.items()

		for utt,mat in items:
			uttLens[utt] = len(mat)
			matrixs.append(mat)
		if keepDim is False:
			matrixs = np.row_stack(matrixs)
		return matrixs, uttLens

	def remerge(self,data,uttLens):
		'''
		Usage:  obj = obj.merge(data,utterancelengths)
		
		Return a KaldiDict object. This is a inverse operation of .merge() function.
		''' 
		assert isinstance(uttLens,(list,tuple,dict)), "Expected <uttLens> is a list, tuple or dict object."
		if isinstance(uttLens,dict):
			uttLens = uttLens.items()
		for i in uttLens:
			assert isinstance(i,(list,tuple)) and len(i) == 2, "<uttLens> has not enough information of utterance ID and its length."
			assert isinstance(i[0],str), "Expected utterance ID is string but got {}.".format(i[0])
			assert isinstance(i[1],int) and i[1] > 0, "Expected length is positive int value but got {}.".format(i[1])

		if len(self.keys()) == 0:
			newDict = self
			returnFlag = False
		else:
			newDict = KaldiDict()
			returnFlag = True

		if isinstance(data,list):
			if len(uttLens) < len(data):
				raise WrongOperation("The length information in <uttLens> was not enough.")
			for i,(utt,lens) in enumerate(uttLens):
				matix = data[i]
				if len(matix) != lens:
					raise WrongDataFormat('The frames {}!={} at utterance ID "{}".'.format(len(matix),lens,utt))
				newDict[utt] = matix
				if i >= len(data):
					break

		elif isinstance(data,np.ndarray):
			if len(data.shape) > 2:
				raise WrongOperation("Expected the dimension of <data> is smaller than 2 but got {}.".format(len(data.sahpe)))
			elif len(data.shape) == 1:
				data = data[:,None]
			start = 0
			for utt,lens in uttLens:
				matix = data[start:start+lens]
				if len(matix) < lens:
					raise WrongDataFormat('The frames {}<{} at utterance ID "{}".'.format(len(matix),lens,utt))
				newDict[utt] = data[start:start+lens]
				start += lens
				if start >= len(data):
					break
			if len(data[start:]) > 0:
				raise WrongOperation("The length information in <uttLens> was not enough.")
		else:
			raise UnsupportedDataType('It is not merged KaldiDict data.')

		if returnFlag is True:
			return newDict

	def normalize(self,std=True,alpha=1.0,beta=0.0,epsilon=1e-6,axis=0):
		'''
		Usage:  newObj = obj.normalize()
		
		Return a KaldiDict object. If <std> is True, do: 
					alpha * (x-mean)/(std+epsilon) + belta, 
		or do: 
					alpha * (x-mean) + belta.
		'''
		if len(self.keys()) == 0:
			return KaldiDict()

		assert isinstance(epsilon,(float,int)) and epsilon > 0, "Expected <epsilon> is positive value."
		assert isinstance(alpha,(float,int)) and alpha > 0, "Expected <alpha> is positive value."
		assert isinstance(beta,(float,int)), "Expected <beta> is an int or float value."
		assert isinstance(axis,int), "Expected <axis> is an int value."

		data,uttLens = self.merge()
		mean = np.mean(data,axis=axis)

		if std is True:  
			std = np.std(data,axis=axis)
			data = alpha*(data-mean)/(std+epsilon) + beta
		else:
			data = alpha*(data-mean) + beta

		newDict = KaldiDict()
		start = 0
		for utt,lens in uttLens.items():
			newDict[utt] = data[start:(start+lens)]
			start += lens

		return newDict 

	def cut(self,maxFrames):
		'''
		Usage:  newObj = obj.cut(100)
		
		Cut data by <maxFrames> if its frame length is larger than 1.25*<maxFrames>.
		'''
		if len(self.keys()) == 0:
			raise WrongOperation('No data to cut.')

		assert isinstance(maxFrames,int) and maxFrames > 0, "Expected <maxFrames> is positive int number but got {}.".format(type(maxFrames))

		newData = {}

		cutThreshold = maxFrames + maxFrames//4

		for key in self.keys():
			if len(self[key].shape) == 1:
				self[key] = self[key][None,:]
			matrix = self[key]
			if len(matrix) <= cutThreshold:
				newData[key] = matrix
			else:
				i = 0 
				while True:
					newData[key+"_"+str(i)] = matrix[i*maxFrames:(i+1)*maxFrames]
					i += 1
					if len(matrix[i*maxFrames:]) <= cutThreshold:
						break
				if len(matrix[i*maxFrames:]) != 0:
					newData[key+"_"+str(i)] = matrix[i*maxFrames:]
		
		return KaldiDict(newData)

	def tuple_value(self,others,sort=False):
		'''
		Usage:  data = obj1.tuple_value(obj1)
		
		Return a list.
		Tuple the utterance ID and data of the same utterance ID from different Kaldidict or KaldiArk objects.
		If <sort> is True, sort the data by frame length of current object.
		'''         

		if not isinstance(others,(list,tuple)):
			others = [others,]

		for index,other in enumerate(others):
			if isinstance(other,KaldiDict):                   
				pass
			elif isinstance(other,KaldiArk):
				others[index] = other.array       
			else:
				raise UnsupportedDataType('Excepted KaldiArk or KaldiDict object but got {}.'.format(type(other)))

		new = []
		for utt in self.keys():
			if len(self[utt].shape) == 1:
				self[utt] = self[utt][None,:]
			temp = (utt,self[utt],)
			for other in others:
				if len(other[utt].shape) == 1:
					other[utt] = other[utt][None,:]
				if not utt in other.keys():
					break
				else:
					temp += (other[utt],)
			if len(temp) < len(others) + 2:
				continue
			else:
				new.append(temp)
		
		if sort is True:
			new = sorted(new, key=lambda x:len(x[1]))
		
		return new

class KaldiLattice(object):
	'''
	Usage:  obj = KaldiLattice() or obj = KaldiLattice(lattice,hmm,wordSymbol)

	KaldiLattice holds the lattice and its related file path: HMM file and WordSymbol file. 
	The <lattice> can be lattice binary data or file path. Both <hmm> and <wordSymbol> are expected to be file path.
	decode_lattice() function will return a KaldiLattice object. Aslo, you can define a empty KaldiLattice object and load its data later.
	'''    
	def __init__(self,lat=None,hmm=None,wordSymbol=None):

		self._lat = lat
		self._hmm = hmm
		self._wordSymbol = wordSymbol

		if lat != None:
			assert hmm != None and wordSymbol != None, "Expected both HMM file and word-to-id file."
			if isinstance(lat,str):
				self.load(lat,hmm,wordSymbol)
			elif isinstance(lat,bytes):
				pass
			else:
				raise UnsupportedDataType("<lat> is not a correct lattice format: file path or byte data.")

	def load(self,latFile,hmm,wordSymbol):
		'''
		Usage:  obj.load('graph/lat.gz','graph/final.mdl','graph/words.txt')

		Load lattice data to memory. <latFile> should be file path. <hmm> and <wordSymbol> are expected as file path.
		Note that the new data will coverage original data in current object.
		We don't check whether it is really a lattice data at the beginning.
		'''
		assert isinstance(latFile,str), "Expected <latFile> is path-like string."
		assert isinstance(hmm,str), "Expected <hmm> is path-like string."
		assert isinstance(wordSymbol,str), "Expected <wordSymbol> is path-like string."

		for x in [latFile,hmm,wordSymbol]:
			if not os.path.isfile(x):
				raise PathError('No such file:{}.'.format(x))

		if latFile.endswith('.gz'):
			p = subprocess.Popen('gunzip -c {}'.format(latFile),shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
			(out,err)=p.communicate()
			if out == b'':
				print(err.decode())
				raise WrongDataFormat('Lattice load defeat!')
			else:
				self._lat = out
				self._hmm = hmm
				self._wordSymbol = wordSymbol
		else:
			try :
				with open(latFile,'rb') as fr:
					out = fr.read()
				if out == b'':
					raise WrongDataFormat('It seems a null file.')
				else:
					self._lat = out
					self._hmm = hmm
					self._wordSymbol = wordSymbol
			except Exception as e:
				print("Load lattice file defeated. Please make sure it is a lattice file avaliable.")
				raise e
			
	def save(self,fileName,copyFile=False):
		'''
		Usage:  obj.save("lat.gz")

		Save lattice as .gz file. If <copyFile> is True, HMM file and word-to-symbol file will be copied to the same directory as lattice file. 
		''' 
		if self._lat == None:
			raise WrongOperation('No any data to save.')   

		if not fileName.endswith('.gz'):
			fileName += '.gz'

		cmd1 = 'gzip -c > {}'.format(fileName)
		p1 = subprocess.Popen(cmd1,shell=True,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
		(out1,err1) = p1.communicate(input=self._lat)
		if not os.path.isfile(fileName):
			err1 = err1.decode()
			print(err1)
			raise Exception('Save lattice defeat.')
		if copyFile == True:
			for x in [self._hmm,self._wordSymbol]:
				if not os.path.isfile(x):
					raise PathError('No such file:{}.'.format(x))
			i = fileName.rfind('/')
			if i > 0:
				latDir = fileName[0:i+1]
			else:
				latDir = './'
			cmd2 = 'cp -f {} {}; cp -f {} {}'.format(self._hmm, latDir, self._wordSymbol, latDir)
			p2 = subprocess.Popen(cmd2,shell=True,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
			(out2,err2) = p2.communicate(input=self._lat)

	@property
	def value(self):
		'''
		Usage:  lat = obj.value

		Return a lattice binary data. 
		''' 
		return self._lat
	
	@property
	def model(self):
		'''
		Usage:  HMM = obj.hmm

		Return a HMM file path. 
		''' 
		return self._hmm

	@property
	def lexicon(self):
		'''
		Usage:  lexicon = obj.lexicon

		Return a word-to-id file path. 
		''' 
		return self._wordSymbol

	def get_1best_words(self,minLmwt=1,maxLmwt=None,Acwt=1.0,outFile=None):

		raise WrongOperation('get_1best_words() has been removed in current version. Try to use get_1best() please.')
	
	def get_1best(self,lmwt=1,maxLmwt=None,acwt=1.0,outFile=None,phoneSymbol=None):
		'''
		Usage:  out = obj.get_1best(acwt=0.2)

		If <outFile> is not "None", return file path, or return decoding text with Python list format. 
		If <maxLmwt> is "None", return only a list. Or, return a Python dict object. Its keys are language model weight from <lmwt> to <maxLmwt>.
		If <phoneSymbol> is not "None", return phones of 1-best words.
		'''
		global KALDIROOT,kaidiNotFoundError,ENV
		if KALDIROOT is None:
			raise kaidiNotFoundError

		if self._lat == None:
			raise WrongOperation('No any data in lattice.')

		for x in [self._hmm,self._wordSymbol]:
			if not os.path.isfile(x):
				raise PathError('Missing such file:{}.'.format(x))

		assert isinstance(lmwt,int) and lmwt >=0, "Expected <lmwt> is a non-negative int number."
		if maxLmwt != None:
			if maxLmwt < lmwt:
				raise WrongOperation('<maxLmwt> must larger than <lmwt>.')
			else:
				assert isinstance(maxLmwt,int), "Expected <maxLmwt> is a non-negative int number."
				maxLmwt += 1
		else:
			maxLmwt = lmwt + 1
		
		if phoneSymbol != None:
			assert isinstance(phoneSymbol,str), "Expected <phoneSymbol> is a file path-like string."
			if not os.path.isfile(phoneSymbol):
				raise PathError("No such file: {}.".format(phoneSymbol))
			useLexicon = phoneSymbol
		else:
			useLexicon = self._wordSymbol

		result = {}
		if outFile != None:
			assert isinstance(outFile,str), "Expected <outFile> is a file name-like string."
			for LMWT in range(lmwt,maxLmwt,1):
				outFileLMWT = '{}.{}'.format(outFile,LMWT)
				if phoneSymbol != None:
					cmd0 = KALDIROOT+'/src/latbin/lattice-align-phones --replace-output-symbols=true {} ark:- ark:- | '.format(self._hmm)
				else:
					cmd0 = ''
				cmd1 = KALDIROOT+'/src/latbin/lattice-best-path --lm-scale={} --acoustic-scale={} --word-symbol-table={} --verbose=2 ark:- ark,t:- | '.format(LMWT,acwt,useLexicon)
				cmd2 = KALDIROOT+'/egs/wsj/s5/utils/int2sym.pl -f 2- {} > {}'.format(useLexicon,outFileLMWT)
				cmd = cmd0 + cmd1 + cmd2
				p = subprocess.Popen(cmd,shell=True,stdin=subprocess.PIPE,stderr=subprocess.PIPE, env=ENV)
				(out,err) = p.communicate(input=self._lat)
				if not os.path.isfile(outFileLMWT) or os.path.getsize(outFileLMWT) == 0:
					print(err.decode())
					if os.path.isfile(outFileLMWT):
						os.remove(outFileLMWT)
					raise KaldiProcessError('Lattice to 1-best Defeated.')
				else:
					result[LMWT] = outFileLMWT
		else:
			for LMWT in range(lmwt,maxLmwt,1):
				if phoneSymbol != None:
					cmd0 = KALDIROOT+'/src/latbin/lattice-align-phones --replace-output-symbols=true {} ark:- ark:- | '.format(self._hmm)
					#cmd0 = KALDIROOT+'/src/latbin/lattice-to-phone-lattice {} ark:- ark:- | '.format(self._hmm)
				else:
					cmd0 = ''
				cmd1 = KALDIROOT+'/src/latbin/lattice-best-path --lm-scale={} --acoustic-scale={} --word-symbol-table={} --verbose=2 ark:- ark,t:- |'.format(LMWT,acwt,useLexicon)
				cmd2 = KALDIROOT+'/egs/wsj/s5/utils/int2sym.pl -f 2- {} '.format(useLexicon)
				cmd = cmd0 + cmd1 + cmd2
				p = subprocess.Popen(cmd,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE, env=ENV)
				(out,err) = p.communicate(input=self._lat)
				if out == b'':
					print(err.decode())
					raise KaldiProcessError('Lattice to 1-best Defeated.')
				else:
					out = out.decode().split("\n")
					result[LMWT] = out[0:-1]
		if maxLmwt - lmwt == 1:
			result = result[lmwt]
		return result
	
	def scale(self,acwt=1,invAcwt=1,ac2lm=0,lmwt=1,lm2ac=0):
		'''
		Usage:  newObj = obj.sacle(inAcwt=0.2)

		Scale lattice. Return a new KaldiLattice object.
		''' 

		global KALDIROOT,kaidiNotFoundError,ENV
		if KALDIROOT is None:
			raise kaidiNotFoundError

		if self._lat == None:
			raise WrongOperation('No any lattice to scale.')

		for x in [self._hmm,self._wordSymbol]:
			if not os.path.isfile(x):
				raise PathError('Missing file:{}.'.format(x))                

		for x in [acwt,invAcwt,ac2lm,lmwt,lm2ac]:
			assert isinstance(x,int) and x>= 0, "Expected scale is positive int value."
		
		cmd = KALDIROOT+'/src/latbin/lattice-scale'
		cmd += ' --acoustic-scale={}'.format(acwt)
		cmd += ' --acoustic2lm-scale={}'.format(ac2lm)
		cmd += ' --inv-acoustic-scale={}'.format(invAcwt)
		cmd += ' --lm-scale={}'.format(lmwt)
		cmd += ' --lm2acoustic-scale={}'.format(lm2ac)
		cmd += ' ark:- ark:-'

		p = subprocess.Popen(cmd,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=ENV)
		(out,err) = p.communicate(input=self._lat)

		if out == b'':
			print(err.decode())
			raise KaldiProcessError("Scale lattice defeated.")
		else:
			return KaldiLattice(out,self._hmm,self._wordSymbol)

	def add_penalty(self,penalty=0):
		'''
		Usage:  newObj = obj.add_penalty(0.5)

		Add word insertion penalty. Return a new KaldiLattice object.
		''' 
		global KALDIROOT,kaidiNotFoundError,ENV
		if KALDIROOT is None:
			raise kaidiNotFoundError

		if self._lat == None:
			raise WrongOperation('No any lattice to scale.')

		for x in [self._hmm,self._wordSymbol]:
			if not os.path.isfile(x):
				raise PathError('No such file:{}.'.format(x))     

		assert isinstance(penalty,(int,float)) and penalty>= 0, "Expected <penalty> is positive int or float value."
		
		cmd = KALDIROOT+'/src/latbin/lattice-add-penalty'
		cmd += ' --word-ins-penalty={}'.format(penalty)
		cmd += ' ark:- ark:-'

		p = subprocess.Popen(cmd,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE, env=ENV)
		(out,err) = p.communicate(input=self._lat)

		if out == b'':
			print(err.decode())
			raise KaldiProcessError("Add penalty defeated.")
		else:
			return KaldiLattice(out,self._hmm,self._wordSymbol)

	def get_nbest(self,n,acwt=1,outFile=None,outAliFile=None,requireCost=False):
		'''
		Usage:  out = obj.get_nbest(minLmwt=1)

		Return a dict object. Its keys are the weight of language, and value will be result-list if <outFile> is "False" or result-file-path if <outFile> is "True". 
		''' 
		assert isinstance(n,int) and n > 0, "Expected <n> is a positive int value."
		assert isinstance(acwt,(int,float)) and acwt > 0, "Expected <acwt> is a positive int or float value."

		global KALDIROOT,kaidiNotFoundError,ENV
		if KALDIROOT is None:
			raise kaidiNotFoundError
	
		if self._lat == None:
			raise WrongOperation('No any data in lattice.')

		for x in [self._hmm,self._wordSymbol]:
			if not os.path.isfile(x):
				raise PathError('Missed file:{}.'.format(x))

		cmd = KALDIROOT+'/src/latbin/lattice-to-nbest --acoustic-scale={} --n={} ark:- ark:- |'.format(acwt,n)
		if outFile != None:
			assert isinstance(outFile,str), 'Expected <outFile> is file name-like string but got {}.'.format(outFile)
			if outAliFile != None:
				assert isinstance(outAliFile,str), 'Expected <outAliFile> is file name-like string but got {}.'.format(outAliFile)
				if not outAliFile.endswith('.gz'):
					outAliFile += '.gz'
				cmd += KALDIROOT+'/src/latbin/nbest-to-linear ark:- "ark,t:|gzip -c > {}" "ark,t:|{}/egs/wsj/s5/utils/int2sym.pl -f 2- {} > {}"'.format(outAliFile,KALDIROOT,self._wordSymbol,outFile)
				if requireCost:
					outCostFile = outFile+'.cost'
					cmd += ' ark,t:{}.lm ark,t:{}.ac'.format(outCostFile,outCostFile)
				p = subprocess.Popen(cmd,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=ENV)
				(out,err) = p.communicate(input=self._lat)
				if not os.path.isfile(outFile) or os.path.getsize(outFile) == 0:
					print(err.decode())
					if os.path.isfile(outFile):
						os.remove(outFile)
					raise KaldiProcessError('Get N best words defeated.')
				else:
					if requireCost:
						return (outFile, outAliFile, outCostFile+'.lm', outCostFile+'.ac')
					else:
						return (outFile, outAliFile)
			else:
				with tempfile.NamedTemporaryFile('w+',encoding='utf-8') as outAliFile:
					cmd += KALDIROOT+'/src/latbin/nbest-to-linear ark:- ark:{} "ark,t:|{}/egs/wsj/s5/utils/int2sym.pl -f 2- {} > {}"'.format(outAliFile.name,KALDIROOT,self._wordSymbol,outFile)
					if requireCost:
						outCostFile = outFile+'.cost'
						cmd += ' ark,t:{}.lm ark,t:{}.ac'.format(outCostFile,outCostFile)
					p = subprocess.Popen(cmd,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=ENV)
					(out,err) = p.communicate(input=self._lat)
				if not os.path.isfile(outFile) or os.path.getsize(outFile) == 0:
					print(err.decode())
					if os.path.isfile(outFile):
						os.remove(outFile)
					raise KaldiProcessError('Get N best words defeated.')
				else:
					if requireCost:
						return (outFile, outCostFile+'.lm', outCostFile+'.ac')
					else:
						return outFile
		else:
			with tempfile.NamedTemporaryFile('w+',encoding='utf-8') as outCostFile_lm:  
				with tempfile.NamedTemporaryFile('w+',encoding='utf-8') as outCostFile_ac:
					if outAliFile != None:
						assert isinstance(outAliFile,str), 'Expected <outAliFile> is file name-like string but got {}.'.format(outAliFile)
						if not outAliFile.endswith('.gz'):
							outAliFile += '.gz'
						cmd += KALDIROOT+'/src/latbin/nbest-to-linear ark:- "ark,t:|gzip -c > {}" "ark,t:|{}/egs/wsj/s5/utils/int2sym.pl -f 2- {}"'.format(outAliFile,KALDIROOT,self._wordSymbol)    
						if requireCost:
							cmd += ' ark,t:{} ark,t:{}'.format(outCostFile_lm.name,outCostFile_ac.name)
						p = subprocess.Popen(cmd,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE, env=ENV)
						(out,err) = p.communicate(input=self._lat)
					else:
						with tempfile.NamedTemporaryFile('w+',encoding='utf-8') as outAliFile:
							cmd += KALDIROOT+'/src/latbin/nbest-to-linear ark:- ark:{} "ark,t:|{}/egs/wsj/s5/utils/int2sym.pl -f 2- {}"'.format(outAliFile.name,KALDIROOT,self._wordSymbol)    
							if requireCost:
								cmd += ' ark,t:{} ark,t:{}'.format(outCostFile_lm.name,outCostFile_ac.name)
							p = subprocess.Popen(cmd,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE, env=ENV)
							(out,err) = p.communicate(input=self._lat)

					if out == b'':
						print(err.decode())
						raise KaldiProcessError('Get N best defeated.')
					else:
						if requireCost:
							out = out.decode().split("\n")[:-1]
							allResult = []
							outCostFile_lm.seek(0)
							outCostFile_ac.seek(0)
							lines_ac = outCostFile_ac.read().split("\n")[:-1]
							lines_lm = outCostFile_lm.read().split("\n")[:-1]
							for result, ac_score, lm_score in zip(out,lines_ac,lines_lm):
								allResult.append([result,float(ac_score.split()[1]),float(lm_score.split()[1])])
							out = allResult
						else:
							out = out.decode().split("\n")[:-1]

						exUtt = None
						new = []
						temp = []
						index = 1
						for i in out:
							if isinstance(i,list):
								utterance = i[0]
							else:
								utterance = i
							if exUtt == None:
								t = utterance.find("-"+str(index)+" ")
								exUtt = utterance[0:t]
							else:
								if utterance.startswith(exUtt):
									t = utterance.find("-"+str(index)+" ")
								else:
									new.append(temp)
									temp = []	
									index = 1
									t = utterance.find("-"+str(index)+" ")
									exUtt = utterance[0:t]
							utterance = exUtt + utterance[t+2:]
							if isinstance(i,list):
								temp.append((utterance,i[1],i[2]))
							else:
								temp.append(utterance)
							index += 1	
						if len(temp) > 0:
							new.append(temp)
						return new                         

	def __add__(self,other):
		'''
		Usage:  lat3 = lat1 + lat2

		Return a new KaldiLattice object. lat2 must be KaldiLattice object.
		Note that this is only a simple additional operation to make two lattices being one.
		'''

		assert isinstance(other,KaldiLattice), "Expected KaldiLattice but got {}.".format(type(other))

		if self._lat == None:
			return copy.deepcopy(other)
		elif other._lat == None:
			return copy.deepcopy(self)
		elif self._hmm != other._hmm or self._wordSymbol != other._wordSymbol:
			raise WrongOperation("Both two members must use the same HMM and word-symbol file.")
		
		newLat = self._lat + other._lat 

		for x in [self._hmm,self._wordSymbol]:
			if not os.path.isfile(x):
				raise PathError('Missing file:{}.'.format(x))     

		return KaldiLattice(newLat, self._hmm, self._wordSymbol)

# ------------ Other Classes -----------

class Supporter(object):
	'''
	Usage:  supporter = Supporter(outDir='Result')

	Supporter is used to manage training information such as the change of loss and others.
	'''      
	def __init__(self,outDir='Result'):

		self.currentField = {}
		self.globalField = []

		assert isinstance(outDir,str), "<outDir> should be a name-like string."
		if not os.path.isdir(outDir):
			os.mkdir(outDir)
		self.outDir = os.path.abspath(outDir)
		self.logFile = self.outDir+'/log'
		with open(self.logFile,'w',encoding='utf-8'):
			pass
		
		self.lastSavedArch = {}
		self.savingThreshold = None

		self._allKeys = []

		self._iterSymbol = -1
		
	def send_report(self,info):
		'''
		Usage:  supporter = obj.send_report({"epoch":epoch,"train_loss":loss,"train_acc":acc})

		Send information and these will be retained untill you do the statistics by using .collect_report().
		<info> should be a dict of names and their values with int or float type. 
		'''
		assert isinstance(info,dict), "Expected <info> is a Python dict object."

		keys = list(info)
		allKeys = list(self.currentField)
	
		for i in keys: 
			assert isinstance(i,str), "Expected name-like string but got {}.".format(type(i))
			value = info[i]
			assert isinstance(value,(int,float)), "Expected int or float value but got {}.".format(type(value))
			i = i.lower()
			if not i in allKeys:
				self.currentField[i] = []
			self.currentField[i].append(value)

	def collect_report(self,keys=None,plot=True):
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

		with open(self.logFile,'a',encoding='utf-8') as fw:
			fw.write(message + '\n')
		# Print to screen
		if plot is True:
			print(message)
		# Clear
		self.currentField = {}

	def save_arch(self,saveFunc,arch,addInfo=None,byKey=None,byMax=True):
		'''
		Usage:  obj.save_arch(saveFunc,archs={'model':model,'opt':optimizer})

		Save architecture such as models or optimizers when you use this function.
		If you use <byKey> and set <byMax>,  will be saved only while meeting the condition. 
		<archs> will be sent to <saveFunc> but with new name.
		''' 
		assert isinstance(arch,dict), "Expected <arch> is dict whose keys are architecture-names and values are architecture-objects."

		if self.currentField != {}:
			self.collect_report(plot=False)
		
		suffix = "_"+str(self._iterSymbol)
		self._iterSymbol += 1

		if not addInfo is None:
			assert isinstance(addInfo,(str,list,tuple)), 'Expected <addInfo> is string, list or tuple.'
			if isinstance(addInfo,str):
				addInfo = [addInfo,]
			for i in addInfo:
				if not i in self.globalField[-1].keys():
					continue
				value = self.globalField[-1][i]
				if isinstance(value,float):
					suffix += ( "_" + i + ("%.4f"%(value)).replace(".",""))
				else:
					suffix += ( "_" + i + '%d'%(value))             

		if byKey == None:
			newArchs = []
			for name in arch.keys():
				fileName = self.outDir+'/'+name+suffix
				newArchs.append((fileName, arch[name]))
				self.lastSavedArch[name] = fileName
			if len(newArchs) == 1:
				newArchs = newArchs[0]
			saveFunc(newArchs)
		else:
			byKey = byKey.lower()
			if not byKey in self.globalField[-1].keys():
				print("Warning: Save architectures defeat. Because the key {} has not been reported.".format(byKey))
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
					if isinstance(value,float):
						suffix += ( "_" + byKey + ('%.4f'%(value)).replace('.','') )
					else:
						suffix += ( "_" + byKey + '%d'%(value) )
					fileName = self.outDir+'/'+name+suffix
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
   
	def judge(self,key,condition,threshold,byDeltaRatio=False):
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

	def dump(self,keepItems=False,fromLogFile=None):
		'''
		Usage:  product = obj.dump()

		Get all reported information.
		If <fromLogFile> is not "None", read and return information from log file.
		If <keepItems> is True, return information by iterms name. 
		'''
		
		if fromLogFile != None:
			assert isinstance(fromLogFile,str), "Expected <fromLogFile> is file name-like string."
			if not os.path.isfile(fromLogFile):
				raise PathError('No such file:{}.'.format(fromLogFile))
			else:
				with open(fromLogFile,'r',encoding='utf-8') as fr:
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

		if isinstance(scpFiles,str):
			p = subprocess.Popen('ls {}'.format(scpFiles),shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
			(out,err) = p.communicate()
			if out == b'':
				raise PathError("No such file:{}".format(scpFiles))
			else:
				out = out.decode().strip().split('\n')
		elif isinstance(scpFiles,list):
			out = scpFiles
		else:
			raise UnsupportedDataType('Expected <scpFiles> is scp file-like string or list object.')

		if isinstance(chunks,int):
			assert chunks>0, "Expected <chunks> is a positive int number but got {}.".format(chunks)
		elif chunks != 'auto':
			raise WrongOperation('Expected <chunks> is a positive int number or <auto> but got {}.'.format(chunks))

		temp = []
		for scpFile in out:
			with open(scpFile,'r',encoding='utf-8') as fr:
				temp.extend(fr.read().strip().split('\n'))
		K = int(len(temp)*(1-retainData))
		self.retainedFiles = temp[K:]
		self.allFiles = temp[0:K]

		if chunks == 'auto':
			#Compute the chunks automatically
			sampleChunk = random.sample(self.allFiles,10)
			with tempfile.NamedTemporaryFile('w',encoding='utf-8',suffix='.scp') as sampleFile:
				sampleFile.write('\n'.join(sampleChunk))
				sampleFile.seek(0)
				sampleChunkData = load(sampleFile.name)
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
		with tempfile.NamedTemporaryFile('w',suffix='.scp') as scpFile:
			scpFile.write('\n'.join(self.datasetBag[datasetIndex]))
			scpFile.seek(0)
			chunkData = load(scpFile.name)
		if self.otherArgs != None:
			self.nextDataset = self.fileProcessFunc(self,chunkData,self.otherArgs)
		else:
			self.nextDataset = self.fileProcessFunc(self,chunkData)

		self.nextDataset = [X for X in self.nextDataset]

		if self._batchSize > len(self.nextDataset):
			print("Warning: Batch Size < {} > is extremely large for this dataset, we hope you can use a more suitable value.".format(self._batchSize))
		
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
				while self.loadDatasetThread.is_alive():
					time.sleep(0.5)
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
			return self.currentEpochPosition/self.epochSize
	
	@property
	def chunkProgress(self):
		if self._isNewChunk is True:
			return 1.
		else:
			return self.currentPosition/len(self.currentDataset)

	def get_retained_data(self,processFunc=None,batchSize=None,chunks='auto',otherArgs=None,shuffle=False,retainData=0.0):

		if len(self.retainedFiles) == 0:
			raise WrongOperation('No retained validation data.')   

		if processFunc == None:
			processFunc = self.fileProcessFunc
		
		if batchSize == None:
			batchSize = self._batchSize

		if isinstance(chunks,int):
			assert chunks > 0,"Expected <chunks> is a positive int number."
		elif chunks != 'auto':
			raise WrongOperation('Expected <chunks> is a positive int number or <auto> but got {}.'.format(chunks))

		if otherArgs == None:
			otherArgs = self.otherArgs

		with tempfile.NamedTemporaryFile('w',encoding='utf-8',suffix='.scp') as reScpFile:
			reScpFile.write('\n'.join(self.retainedFiles))
			reScpFile.seek(0)  
			reIterator = DataIterator(reScpFile.name,processFunc,batchSize,chunks,otherArgs,shuffle,retainData)

		return reIterator

# ---------- Basic Class Functions ------- 

def save(data,*params):
	'''
	Usage:  save(obj, 'data.ark')
	
	The same as KaldiArk().save() or KaldiDict().save() or KaldiLattice().save() method.
	'''  
	if isinstance(data,(KaldiDict,KaldiArk,KaldiLattice)):
		data.save(*params)
	else:
		raise UnsupportedDataType("Expected <data> is KaldiDict, KaldiArk or KaldiLattice object but got {}.".format(type(data)))

def concat(data,axis):
	'''
	Usage:  newObj = concat([obj1,obj2],axis=1)
	
	Return a KaldDict object. The same as KaldiDict().concat() fucntion.
	'''  
	assert isinstance(data,(list,tuple)), "Expected <data> is a list or tuple object but got {}.".format(type(data))

	if len(data) > 0:
		if not isinstance(data[0],(KaldiArk,KaldiDict)):
			raise UnsupportedDataType('Expected KaldiDict or KaldiArk object but got {}.'.format(data[0]))
		if isinstance(data[0],KaldiArk):
			data[0] = data[0].array
		if len(data) > 1:
			return data[0].concat(data[1:],axis)
		else:
			return copy.deepcopy(data[0])
	else:
		return KaldiDict()

def cut(data,maxFrames):
	'''
	Usage:  result = cut(100)
	
	Return a KaldDict object. The same as KaldiDict().cut() fucntion.
	'''  
	if not isinstance(data,(KaldiArk,KaldiDict)):
		raise UnsupportedDataType('Expected KaldiDict or KaldiArk object but got {}.'.format(data))
	elif isinstance(data,KaldiArk):
		data = data.array
	
	return data.cut(maxFrames)

def normalize(data,std=True,alpha=1.0,beta=0.0,epsilon=1e-6,axis=0):
	'''
	Usage:  newObj = normalize(obj)
	
	Return a KaldiDict object. The same as KaldiDict().normalize() fucntion.
	'''

	if not isinstance(data,(KaldiArk,KaldiDict)):
		raise UnsupportedDataType('Expected KaldiDict or KaldiArk object but got {}.'.format(data))
	elif isinstance(data,KaldiArk):
		data = data.array

	return data.normalize(std,alpha,beta,epsilon,axis)

def merge(data,keepDim=False,sortFrame=False):
	'''
	Usage:  data, lengths = merge(obj)
	
	Return a KaldiDict object. The same as KaldiDict().merge() fucntion.
	'''   
	if not isinstance(data,(KaldiArk,KaldiDict)):
		raise UnsupportedDataType('Expected KaldiDict or KaldiArk object but got {}.'.format(data))
	elif isinstance(data,KaldiArk):
		data = data.array

	return data.merge(keepDim,sortFrame)

def remerge(matrix,uttLens):
	'''
	Usage:  newObj = remerge(data,lengths)
	
	Return a KaldiDict object. The same as KaldiDict().remerge() fucntion.
	'''
	newData = KaldiDict()
	newData.remerge(matrix,uttLens)

	return newData

def sort(data,by='frame',reverse=False):
	'''
	Usage:  newObj = sort(obj)
	
	Return a KaldiDict object. The same as KaldiDict().sort() fucntion.
	'''
	if not isinstance(data,(KaldiArk,KaldiDict)):
		raise UnsupportedDataType('Expected KaldiDict or KaldiArk object but got {}.'.format(data))
	elif isinstance(data,KaldiArk):
		data = data.array

	return data.sort(by,reverse)

def splice(data,left=4,right=None):
	'''
	Usage:  newObj = splice(chunks=5)
	
	Return KaldiArk or kaldiDict object. The same as KaldiArk().splice() or KaldiDict().splice() fucntion.
	'''
	if isinstance(data,(KaldiDict,KaldiArk)):
		return data.splice(left,right)
	else:
		raise UnsupportedDataType("Expected KaldiDict or KaldiArk data but got {}.".format(type(data)))     

# ---------- Feature and Label Process Fucntions -----------

def compute_mfcc(wavFile,rate=16000,frameWidth=25,frameShift=10,melBins=23,featDim=13,windowType='povey',useSuffix=None,config=None,asFile=False):
	'''
	Usage:  obj = compute_mfcc("test.wav") or compute_mfcc("test.scp")

	Compute MFCC feature. If <asFile> is False, return a KaldiArk object. Or return file-path.
	Some usual options can be assigned directly. If you want use more, set <config> = your-configure, but if you do this, these usual configures we provided will be ignored.
	You can use .check_config('compute_mfcc') function to get configure information that you can set.
	Also you can run shell command "compute-mfcc-feats" to look their meaning.
	'''
	if useSuffix != None:
		assert isinstance(useSuffix,str), "Expected <useSuffix> is a string."
		useSuffix = useSuffix.strip().lower()[-3:]
	else:
		useSuffix = ""
	assert useSuffix in ["","scp","wav"], 'Expected <useSuffix> is "scp" or "wav".'

	global KALDIROOT,kaidiNotFoundError,ENV
	if KALDIROOT is None:
		raise kaidiNotFoundError

	if isinstance(wavFile,str):
		if os.path.isdir(wavFile):
			raise WrongOperation('Expected <wavFile> is file path but got a directory:{}.'.format(wavFile))
		else:
			p = subprocess.Popen('ls {}'.format(wavFile),shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=ENV)
			(out,err) = p.communicate()
			if out == b'':
				raise PathError("No such file:{}".format(wavFile))
			else:
				allFiles = out.decode().strip().split('\n')
	else:
		raise UnsupportedDataType('Expected filename-like string but got a {}.'.format(type(wavFile)))

	kaldiTool = 'compute-mfcc-feats'
	if config == None:    
		config = {}
		config["--allow-downsample"] = "true"
		config["--allow-upsample"] = "true"
		config["--sample-frequency"] = rate
		config["--frame-length"] = frameWidth
		config["--frame-shift"] = frameShift
		config["--num-mel-bins"] = melBins
		config["--num-ceps"] = featDim
		config["--window-type"] = windowType
	if check_config(name='compute_mfcc',config=config):
		for key in config.keys():
			kaldiTool += ' {}={}'.format(key,config[key])
	
	results = []

	if asFile is True:
		for wavFile in allFiles:
			wavFile = os.path.abspath(wavFile)
			dirIndex = wavFile.rfind('/')
			dirName = wavFile[:dirIndex+1]
			fileName = wavFile[dirIndex+1:]
			fileName = "".join(fileName.split("."))
			outFile = dirName + fileName + ".mfcc.ark"

			if fileName[-3:].lower() == 'wav':
				cmd = 'echo {} {} | {} scp:- ark:{}'.format(fileName,wavFile,kaldiTool,outFile)
			elif fileName[-3:].lower() == 'scp':
				cmd = '{} scp:{} ark:{}'.format(kaldiTool,wavFile,outFile)
			elif useSuffix == "wav":
				cmd = 'echo {} {} | {} scp:- ark:{}'.format(fileName,wavFile,kaldiTool,outFile)
			elif useSuffix == "scp":
				cmd = '{} scp:{} ark:{}'.format(kaldiTool,wavFile,outFile)
			else:
				raise UnsupportedDataType('Unknown file suffix. You can declare it by making <useSuffix> "wav" or "scp".')
			
			p = subprocess.Popen(cmd,shell=True,stderr=subprocess.PIPE,env=ENV)
			(out,err) = p.communicate()
			if not os.path.isfile(outFile) or os.path.getsize(outFile) == 0:
				err = err.decode()
				print(err)
				if os.path.isfile(outFile):
					os.remove(outFile)
				raise KaldiProcessError('Compute MFCC defeated.')
			else:
				results.append(outFile)
	
	else:
		for wavFile in allFiles:
			wavFile = os.path.abspath(wavFile)

			if wavFile[-3:].lower() == "wav":
				dirIndex = wavFile.rfind('/')
				fileName = wavFile[dirIndex+1:]
				fileName = "".join(fileName.split("."))
				cmd = 'echo {} {} | {} scp:- ark:-'.format(fileName,wavFile,kaldiTool)
			elif wavFile[-3:].lower() == 'scp':
				cmd = '{} scp:{} ark:-'.format(kaldiTool,wavFile)
			elif "wav" in useSuffix:
				dirIndex = wavFile.rfind('/')
				fileName = wavFile[dirIndex+1:]
				fileName = "".join(fileName.split("."))
				cmd = 'echo {} {} | {} scp:- ark:-'.format(fileName,wavFile,kaldiTool)
			elif "scp" in useSuffix:        
				cmd = '{} scp:{} ark:-'.format(kaldiTool,wavFile)    
			else:
				raise UnsupportedDataType('Unknown file suffix. You can declare it by making <useSuffix> "wav" or "scp".')

			p = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=ENV)
			(out,err) = p.communicate()   
			if out == b'':
				err = err.decode()
				print(err)
				raise KaldiProcessError('Compute MFCC defeated.')
			else:
				results.append(KaldiArk(out))
	
	if len(results) == 1:
		results = results[0]
	return results

def compute_fbank(wavFile,rate=16000,frameWidth=25,frameShift=10,melBins=23,windowType='povey',useSuffix=None,config=None,asFile=False):
	'''
	Usage:  obj = compute_fbank("test.wav") or compute_fbank("test.scp")

	Compute fbank feature. If <asFile> is False, return a KaldiArk object. Or return file-path.
	Some usual options can be assigned directly. If you want use more, set <config> = your-configure, but if you do this, these usual configures we provided will be ignored.
	You can use check_config('compute_fbank') function to get configure information that you can set.
	Also you can run shell command "compute-fbank-feats" to look their meaning.
	'''
	if useSuffix != None:
		assert isinstance(useSuffix,str), "Expected <useSuffix> is a string."
		useSuffix = useSuffix.strip().lower()[-3:]
	else:
		useSuffix = ""
	assert useSuffix in ["","scp","wav"], 'Expected <useSuffix> is "scp" or "wav".'

	global KALDIROOT,kaidiNotFoundError,ENV
	if KALDIROOT is None:
		raise kaidiNotFoundError

	if isinstance(wavFile,str):
		if os.path.isdir(wavFile):
			raise WrongOperation('Expected <wavFile> is file path but got a directory:{}.'.format(wavFile))
		else:
			p = subprocess.Popen('ls {}'.format(wavFile),shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=ENV)
			(out,err) = p.communicate()
			if out == b'':
				raise PathError("No such file:{}".format(wavFile))
			else:
				allFiles = out.decode().strip().split('\n')
	else:
		raise UnsupportedDataType('Expected filename-like string but got a {}.'.format(type(wavFile)))

	kaldiTool = 'compute-fbank-feats'
	if config == None:    
		config = {}
		config["--allow-downsample"] = "true"
		config["--allow-upsample"] = "true"
		config["--sample-frequency"] = rate
		config["--frame-length"] = frameWidth
		config["--frame-shift"] = frameShift
		config["--num-mel-bins"] = melBins
		config["--window-type"] = windowType
	if check_config(name='compute_fbank',config=config):
		for key in config.keys():
			kaldiTool += ' {}={}'.format(key,config[key])
	
	results = []

	if asFile is True:

		for wavFile in allFiles:
			wavFile = os.path.abspath(wavFile)
			dirIndex = wavFile.rfind('/')
			dirName = wavFile[:dirIndex+1]
			fileName = wavFile[dirIndex+1:]
			fileName = "".join(fileName.split("."))
			outFile = dirName + fileName + ".fbank.ark"

			if fileName[-3:].lower() == 'wav':
				cmd = 'echo {} {} | {} scp:- ark:{}'.format(fileName,wavFile,kaldiTool,outFile)
			elif fileName[-3:].lower() == 'scp':
				cmd = '{} scp:{} ark:{}'.format(kaldiTool,wavFile,outFile)
			elif useSuffix == "wav":
				cmd = 'echo {} {} | {} scp:- ark:{}'.format(fileName,wavFile,kaldiTool,outFile)
			elif useSuffix == "scp":
				cmd = '{} scp:{} ark:{}'.format(kaldiTool,wavFile,outFile)
			else:
				raise UnsupportedDataType('Unknown file suffix. You can declare it by making <useSuffix> "wav" or "scp".')
	
			p = subprocess.Popen(cmd,shell=True,stderr=subprocess.PIPE,env=ENV)
			(out,err) = p.communicate()
			if not os.path.isfile(outFile) or os.path.getsize(outFile) == 0:
				err = err.decode()
				print(err)
				if os.path.isfile(outFile):
					os.remove(outFile)
				raise KaldiProcessError('Compute fbank defeated.')
			else:
				results.append(outFile)
	
	else:
		for wavFile in allFiles:
			wavFile = os.path.abspath(wavFile)
			if wavFile[-3:].lower() == "wav":
				dirIndex = wavFile.rfind('/')
				fileName = wavFile[dirIndex+1:]
				fileName = "".join(fileName.split("."))
				cmd = 'echo {} {} | {} scp:- ark:-'.format(fileName,wavFile,kaldiTool)
			elif wavFile[-3:].lower() == 'scp':
				cmd = '{} scp:{} ark:-'.format(kaldiTool,wavFile)
			elif "wav" in useSuffix:
				dirIndex = wavFile.rfind('/')
				fileName = wavFile[dirIndex+1:]
				fileName = "".join(fileName.split("."))
				cmd = 'echo {} {} | {} scp:- ark:-'.format(fileName,wavFile,kaldiTool)
			elif "scp" in useSuffix:        
				cmd = '{} scp:{} ark:-'.format(kaldiTool,wavFile)    
			else:
				raise UnsupportedDataType('Unknown file suffix. You can declare it by making <useSuffix> "wav" or "scp".')

			p = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=ENV)
			(out,err) = p.communicate()   
			if out == b'':
				err = err.decode()
				print(err)
				raise KaldiProcessError('Compute fbank defeated.')
			else:
				results.append(KaldiArk(out))
	
	if len(results) == 1:
		results = results[0]
	return results

def compute_plp(wavFile,rate=16000,frameWidth=25,frameShift=10,melBins=23,featDim=13,windowType='povey',useSuffix=None,config=None,asFile=False):
	'''
	Usage:  obj = compute_plp("test.wav") or compute_plp("test.scp")

	Compute plp feature. If <asFile> is False, return a KaldiArk object. Or return file-path.
	Some usual options can be assigned directly. If you want use more, set <config> = your-configure, but if you do this, these usual configures we provided will be ignored.
	You can use check_config('compute_plp') function to get configure information that you can set.
	Also you can run shell command "compute-plp-feats" to look their meaning.
	'''
	if useSuffix != None:
		assert isinstance(useSuffix,str), "Expected <useSuffix> is a string."
		useSuffix = useSuffix.strip().lower()[-3:]
	else:
		useSuffix = ""
	assert useSuffix in ["","scp","wav"], 'Expected <useSuffix> is "scp" or "wav".'

	global KALDIROOT,kaidiNotFoundError,ENV
	if KALDIROOT is None:
		raise kaidiNotFoundError

	if isinstance(wavFile,str):
		if os.path.isdir(wavFile):
			raise WrongOperation('Expected <wavFile> file path but got a directory:{}.'.format(wavFile))
		else:
			p = subprocess.Popen('ls {}'.format(wavFile),shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=ENV)
			(out,err) = p.communicate()
			if out == b'':
				raise PathError("No such file:{}".format(wavFile))
			else:
				allFiles = out.decode().strip().split('\n')
	else:
		raise UnsupportedDataType('Expected filename-like string but got a {}.'.format(type(wavFile)))

	kaldiTool = 'compute-plp-feats'
	if config == None:    
		config = {}
		config["--allow-downsample"] = "true"
		config["--allow-upsample"] = "true"
		config["--sample-frequency"] = rate
		config["--frame-length"] = frameWidth
		config["--frame-shift"] = frameShift
		config["--num-mel-bins"] = melBins
		config["--num-ceps"] = featDim
		config["--window-type"] = windowType
	if check_config(name='compute_plp',config=config):
		for key in config.keys():
			kaldiTool += ' {}={}'.format(key,config[key])
	
	results = []

	if asFile is True:

		for wavFile in allFiles:
			wavFile = os.path.abspath(wavFile)
			dirIndex = wavFile.rfind('/')
			dirName = wavFile[:dirIndex+1]
			fileName = wavFile[dirIndex+1:]
			fileName = "".join(fileName.split("."))
			outFile = dirName + fileName + ".plp.ark"

			if fileName[-3:].lower() == 'wav':
				cmd = 'echo {} {} | {} scp:- ark:{}'.format(fileName,wavFile,kaldiTool,outFile)
			elif fileName[-3:].lower() == 'scp':
				cmd = '{} scp:{} ark:{}'.format(kaldiTool,wavFile,outFile)
			elif useSuffix == "wav":
				cmd = 'echo {} {} | {} scp:- ark:{}'.format(fileName,wavFile,kaldiTool,outFile)
			elif useSuffix == "scp":
				cmd = '{} scp:{} ark:{}'.format(kaldiTool,wavFile,outFile)
			else:
				raise UnsupportedDataType('Unknown file suffix. You can declare it by making <useSuffix> "wav" or "scp".')
			
			p = subprocess.Popen(cmd,shell=True,stderr=subprocess.PIPE,env=ENV)
			(out,err) = p.communicate()
			if not os.path.isfile(outFile) or os.path.getsize(outFile) == 0:
				err = err.decode()
				print(err)
				if os.path.isfile(outFile):
					os.remove(outFile)
				raise KaldiProcessError('Compute plp defeated.')
			else:
				results.append(outFile)
	
	else:

		for wavFile in allFiles:
			wavFile = os.path.abspath(wavFile)

			if wavFile[-3:].lower() == "wav":
				dirIndex = wavFile.rfind('/')
				fileName = wavFile[dirIndex+1:]
				fileName = "".join(fileName.split("."))
				cmd = 'echo {} {} | {} scp:- ark:-'.format(fileName,wavFile,kaldiTool)
			elif wavFile[-3:].lower() == 'scp':
				cmd = '{} scp:{} ark:-'.format(kaldiTool,wavFile)
			elif useSuffix == "wav":
				dirIndex = wavFile.rfind('/')
				fileName = wavFile[dirIndex+1:]
				fileName = "".join(fileName.split("."))
				cmd = 'echo {} {} | {} scp:- ark:-'.format(fileName,wavFile,kaldiTool)
			elif useSuffix == "scp":        
				cmd = '{} scp:{} ark:-'.format(kaldiTool,wavFile)    
			else:
				raise UnsupportedDataType('Unknown file suffix. You can declare it by making <useSuffix> "wav" or "scp".')

			p = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=ENV)
			(out,err) = p.communicate()   
			if out == b'':
				err = err.decode()
				print(err)
				raise KaldiProcessError('Compute plp defeated.')
			else:
				results.append(KaldiArk(out))
	
	if len(results) == 1:
		results = results[0]
	return results

def compute_spectrogram(wavFile,rate=16000,frameWidth=25,frameShift=10,windowType='povey',useSuffix=None,config=None,asFile=False):
	'''
	Usage:  obj = compute_spetrogram("test.wav") or compute_spetrogram("test.scp")

	Compute spetrogram feature. If <asFile> is "False", return a KaldiArk object. Or return file-path.
	Some usual options can be assigned directly. If you want use more, set <config>=your-configure, but if you do this, these usual configures we provided will be ignored.
	You can use .check_config('compute_spetrogram') function to get configure information that you can set.
	Also you can run shell command "compute-spetrogram-feats" to look their meaning.
	'''
	if useSuffix != None:
		assert isinstance(useSuffix,str), "Expected <useSuffix> is a string."
		useSuffix = useSuffix.strip().lower()[-3:]
	else:
		useSuffix = ""
	assert useSuffix in ["","scp","wav"], 'Expected <useSuffix> is "scp" or "wav".'

	global KALDIROOT,kaidiNotFoundError,ENV
	if KALDIROOT is None:
		raise kaidiNotFoundError

	if isinstance(wavFile,str):
		if os.path.isdir(wavFile):
			raise WrongOperation('Expected <wavFile> is file path but got a directory:{}.'.format(wavFile))
		else:
			p = subprocess.Popen('ls {}'.format(wavFile),shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=ENV)
			(out,err) = p.communicate()
			if out == b'':
				raise PathError("No such file:{}".format(wavFile))
			else:
				allFiles = out.decode().strip().split('\n')
	else:
		raise UnsupportedDataType('Expected filename-like string but got a {}.'.format(type(wavFile)))

	kaldiTool = 'compute-spetrogram-feats'
	if config == None: 
		config = {}
		config["--allow-downsample"] = "true"
		config["--allow-upsample"] = "true"
		config["--sample-frequency"] = rate
		config["--frame-length"] = frameWidth
		config["--frame-shift"] = frameShift
		config["--window-type"] = windowType
	if check_config(name='compute_spetrogram',config=config):
		for key in config.keys():
			kaldiTool += ' {}={}'.format(key,config[key])
	
	results = []

	if asFile is True:
		for wavFile in allFiles:
			wavFile = os.path.abspath(wavFile)
			dirIndex = wavFile.rfind('/')
			dirName = wavFile[:dirIndex+1]
			fileName = wavFile[dirIndex+1:]
			fileName = "".join(fileName.split("."))
			outFile = dirName + fileName + ".spetrogram.ark"

			if fileName[-3:].lower() == 'wav':
				cmd = 'echo {} {} | {} scp:- ark:{}'.format(fileName,wavFile,kaldiTool,outFile)
			elif fileName[-3:].lower() == 'scp':
				cmd = '{} scp:{} ark:{}'.format(kaldiTool,wavFile,outFile)
			elif useSuffix == "wav":
				cmd = 'echo {} {} | {} scp:- ark:{}'.format(fileName,wavFile,kaldiTool,outFile)
			elif useSuffix == "scp":
				cmd = '{} scp:{} ark:{}'.format(kaldiTool,wavFile,outFile)
			else:
				raise UnsupportedDataType('Unknown file suffix. You can declare it by making <useSuffix> "wav" or "scp".')

			p = subprocess.Popen(cmd,shell=True,stderr=subprocess.PIPE,env=ENV)
			(out,err) = p.communicate()
			if not os.path.isfile(outFile) or os.path.getsize(outFile) == 0:
				err = err.decode()
				print(err)
				if os.path.isfile(outFile):
					os.remove(outFile)
				raise KaldiProcessError('Compute spetrogram defeated.')
			else:
				results.append(outFile)
	else:
		for wavFile in allFiles:
			wavFile = os.path.abspath(wavFile)
			if wavFile[-3:].lower() == "wav":
				dirIndex = wavFile.rfind('/')
				fileName = wavFile[dirIndex+1:]
				fileName = "".join(fileName.split("."))
				cmd = 'echo {} {} | {} scp:- ark:-'.format(fileName,wavFile,kaldiTool)
			elif wavFile[-3:].lower() == 'scp':
				cmd = '{} scp:{} ark:-'.format(kaldiTool,wavFile)
			elif useSuffix == "wav":
				dirIndex = wavFile.rfind('/')
				fileName = wavFile[dirIndex+1:]
				fileName = "".join(fileName.split("."))
				cmd = 'echo {} {} | {} scp:- ark:-'.format(fileName,wavFile,kaldiTool)
			elif useSuffix == "scp":
				cmd = '{} scp:{} ark:-'.format(kaldiTool,wavFile)    
			else:
				raise UnsupportedDataType('Unknown file suffix. You can declare it by making <useSuffix> "wav" or "scp".')

			p = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=ENV)
			(out,err) = p.communicate()   
			if out == b'':
				err = err.decode()
				print(err)
				raise KaldiProcessError('Compute spetrogram defeated.')
			else:
				results.append(KaldiArk(out))
	
	if len(results) == 1:
		results = results[0]
	return results

def use_cmvn(feat,cmvnStatFile=None,utt2spkFile=None,spk2uttFile=None,outFile=None):
	'''
	Usage:  obj = use_cmvn(feat) or obj = use_cmvn(feat,cmvnStatFile,utt2spkFile) or obj = use_cmvn(feat,utt2spkFile,spk2uttFile)

	Apply CMVN to feature. Return a KaldiArk object or file path if <outFile> is not None. 
	If <cmvnStatFile> is None, it will firstly compute the CMVN state. But <utt2spkFile> and <spk2uttFile> are expected given at the same time if they are not None.
	''' 

	global KALDIROOT,kaidiNotFoundError,ENV
	if KALDIROOT is None:
		raise kaidiNotFoundError

	if isinstance(feat,KaldiArk):
		pass
	elif isinstance(feat,KaldiDict):
		feat = feat.ark
	else:
		raise UnsupportedDataType("Expected KaldiArk KaldiDict but got {}.".format(type(feat)))

	fw = tempfile.NamedTemporaryFile('w+b',suffix='ark')

	try:
		if cmvnStatFile == None:
			if spk2uttFile != None:
				if utt2spkFile == None:
					raise WrongOperation('Miss utt2spk file.')
				else:
					cmd1 = 'compute-cmvn-stats --spk2utt=ark:{} ark:- ark:-'.format(spk2uttFile)
			else:
				cmd1 = 'compute-cmvn-stats ark:- ark:-'
			p1 = subprocess.Popen(cmd1,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=ENV)
			(out1,err1) = p1.communicate(input=feat)
			if out1 == b'':
				err1 = err1.decode()
				print(err1)
				raise KaldiProcessError("Compute cmvn stats defeated")
			else:
				fw.write(out1)
				cmvnStatFile = fw.name

		if cmvnStatFile.endswith('ark'):
			cmvnStatFileOption = 'ark:'+cmvnStatFile
		else:
			cmvnStatFileOption = 'scp:'+cmvnStatFile

		if outFile != None:
			if not outFile.endswith('.ark'):
				outFile += '.ark'
			if utt2spkFile != None:
				cmd2 = 'apply-cmvn --utt2spk=ark:{} {} ark:- ark:{}'.format(utt2spkFile,cmvnStatFileOption,outFile)
			else:
				cmd2 = 'apply-cmvn {} ark:- ark:{}'.format(cmvnStatFileOption,outFile)

			p2 = subprocess.Popen(cmd2,shell=True,stdin=subprocess.PIPE,stderr=subprocess.PIPE,env=ENV)
			(out2,err2) = p2.communicate(input=feat)

			if not os.path.isfile(outFile) or os.path.getsize(outFile) == 0:
				err2 = err2.decode()
				print(err2)
				if os.path.isfile(outFile):
					os.remove(outFile)
				raise KaldiProcessError('Use cmvn defeated.')
			else:
				return outFile
		else:
			if utt2spkFile != None:
				cmd3 = 'apply-cmvn --utt2spk=ark:{} {} ark:- ark:-'.format(utt2spkFile,cmvnStatFileOption)
			else:
				cmd3 = 'apply-cmvn {} ark:- ark:-'.format(cmvnStatFileOption)

			p3 = subprocess.Popen(cmd3,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=ENV)
			(out3,err3) = p3.communicate(input=feat)

			if out3 == b'':
				err3 = err3.decode()
				print(err3)
				raise KaldiProcessError('Use cmvn defeated.')
			else:
				return KaldiArk(out3)  
	finally:
		fw.close()

def compute_cmvn_stats(feat,outFile,spk2uttFile=None):
	'''
	Usage:  obj = compute_cmvn_stats(feat,'train_cmvn.ark') or obj = compute_cmvn_stats(feat,'train_cmvn.ark','train/spk2utt')

	Compute CMVN state and save it as file. Return CMVN file path. 
	''' 
	global KALDIROOT,kaidiNotFoundError,ENV
	if KALDIROOT is None:
		raise kaidiNotFoundError

	if isinstance(feat,KaldiArk):
		pass
	elif isinstance(feat,KaldiDict):
		feat = feat.ark
	else:
		raise UnsupportedDataType("Expected <feat> is a KaldiArk or KaldiDict object but got {}.".format(type(feat)))
	
	if spk2uttFile != None:
		cmd = 'compute-cmvn-stats --spk2utt=ark:{} ark:-'.format(spk2uttFile)
	else:
		cmd = 'compute-cmvn-stats ark:-'

	if outFile.endswith('.scp'):
		cmd += ' ark,scp:{},{}'.format(outFile[0:-4]+'.ark',outFile)
	else:
		if not outFile.endswith('.ark'):
			outFile = outFile + '.ark'
		cmd += ' ark:{}'.format(outFile)

	p = subprocess.Popen(cmd,shell=True,stdin=subprocess.PIPE,stderr=subprocess.PIPE,env=ENV)
	(_,err) = p.communicate(input=feat)

	if not os.path.isfile(outFile):
		err = err.decode()
		print(err)
		raise KaldiProcessError('Compute CMVN stats defeated.')
	else:
		return outFile    

def use_cmvn_sliding(feat,windowsSize=None,std=False):
	'''
	Usage:  obj = use_cmvn_sliding(feat) or obj = use_cmvn_sliding(feat,windows=200)

	Apply sliding CMVN to feature. Return KaldiArk object. If <windowsSize> is None, the window width will be set larger than frames of <feat>.
	If <std> is "False", only apply "mean", or apply both "mean" and "std".
	''' 
	global KALDIROOT,kaidiNotFoundError,ENV
	if KALDIROOT is None:
		raise kaidiNotFoundError
	if isinstance(feat,KaldiArk):
		pass
	elif isinstance(feat,KaldiDict):
		feat = feat.ark
	else:
		raise UnsupportedDataType("Expected <feat> is a KaldiArk or KaldiDict object but got {}.".format(type(feat)))
	if windowsSize==None:
		featLen = feat.lens[1]
		maxLen = max([length for utt,length in featLen])
		windowsSize = math.ceil(maxLen/100)*100
	else:
		assert isinstance(windowsSize,int), "Expected <windowsSize> is an int value."

	if std==True:
		std='true'
	else:
		std='false'

	cmd = 'apply-cmvn-sliding --cmn-window={} --min-cmn-window=100 --norm-vars={} ark:- ark:-'.format(windowsSize,std)
	p = subprocess.Popen(cmd,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=ENV)
	(out,err) = p.communicate(input=feat)
	if out == b'':
		err = err.decode()
		print(err)
		raise KaldiProcessError('Use sliding cmvn defeated.')
	else:
		return KaldiArk(out)

def add_delta(feat,order=2,outFile=None):
	'''
	Usage:  newObj = add_delta(feat)

	Add N orders delta to data. If <outFile> is not "None", return file path. Or return a KaldiArk object.
	''' 
	global KALDIROOT,kaidiNotFoundError,ENV
	if KALDIROOT is None:
		raise kaidiNotFoundError
	if isinstance(feat,KaldiArk):
		pass
	elif isinstance(feat,KaldiDict):
		feat = feat.ark
	else:
		raise UnsupportedDataType("Expected <feat> is a KaldiArk or KaldiDict object but got {}.".format(type(feat)))
	
	if outFile != None:
		assert isinstance(outFile,str), "Expected <outFile> is a name-like string."
		outFile = os.path.abspath(outFile)
		if not outFile.endswith('.ark'):
			outFile += '.ark'
		cmd1 = 'add-deltas --delta-order={} ark:- ark:{}'.format(order,outFile)
		p1 = subprocess.Popen(cmd1,shell=True,stdin=subprocess.PIPE,stderr=subprocess.PIPE,env=ENV)
		(out,err) = p1.communicate(input=feat)
		if not os.path.isfile(outFile) or os.path.getsize(outFile) == 0:
			err = err.decode()
			print(err)
			if os.path.isfile(outFile):
				os.remove(outFile)
			raise KaldiProcessError('Add delta defeated.')
		else:
			return outFile
	else:
		cmd2 = 'add-deltas --delta-order={} ark:- ark:-'.format(order)
		p2 = subprocess.Popen(cmd2,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE, env=ENV)
		(out,err) = p2.communicate(input=feat)
		if out == b'':
			err = err.decode()
			print(err)
			raise KaldiProcessError('Add delta defeated.')
		else:
			return KaldiArk(out)      

def get_ali(aliFile,hmm=None,returnPhone=False):
	raise WrongOperation(" .get_ali() function has been removed in current version. Please use .load_ali().")

def load(fileName,useSuffix=None):
	'''
	Usage:  obj = load('feat.npy') or obj = load('feat.ark') or obj = load('feat.scp') or obj = load('feat.lst', useSuffix='scp')

	Load Kaldi ark feat file, kaldi scp feat file, KaldiArk file, or KaldiDict file. Return a KaldiArk or KaldiDict object.
	'''
	if useSuffix != None:
		assert isinstance(useSuffix,str), "Expected <useSuffix> is a string."
		useSuffix = useSuffix.strip().lower()[-3:]
	else:
		useSuffix = ""
	assert useSuffix in ["","scp","ark","npy"], 'Expected <useSuffix> is "ark", "scp" or "npy" but got "{}".'.format(useSuffix)

	if isinstance(fileName,str):
		if os.path.isdir(fileName):
			raise WrongOperation('Expected file name but got a directory:{}.'.format(fileName))
		else:
			p = subprocess.Popen('ls {}'.format(fileName),shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=ENV)
			(out,err) = p.communicate()
			if out == b'':
				raise PathError("No such file:{}".format(fileName))
			else:
				allFiles = out.decode().strip().split('\n')
	else:
		raise UnsupportedDataType('Expected <fileName> is file name-like string but got a {}.'.format(type(fileName)))

	allData_ark = KaldiArk()
	allData_dict = KaldiDict()

	def loadNpyFile(fileName,allData):
		try:
			temp = np.load(fileName,allow_pickle=True)
			data = KaldiDict()
			#totalSize = 0
			for utt_mat in temp:
				data[utt_mat[0]] = utt_mat[1]
				#totalSize += sys.getsizeof(utt_mat[1])
			#if totalSize > 10000000000:
			#    print('Warning: Data is extramely large. It could not be used correctly sometimes.')                
		except:
			raise UnsupportedDataType('Expected "npy" data with KaldiDict format but got {}.'.format(fileName))
		else:
			allData += data
		return allData
	
	def loadArkScpFile(fileName,allData,suffix):

		global KALDIROOT,kaidiNotFoundError,ENV
		if KALDIROOT is None:
			raise kaidiNotFoundError

		if suffix == "ark":
			cmd = 'copy-feats ark:'
		else:
			cmd = 'copy-feats scp:'

		cmd1 = cmd + '{} ark:-'.format(fileName)
		p = subprocess.Popen(cmd1,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=ENV)
		(out,err) = p.communicate()
		if out == b'':
			err = err.decode()
			print(err)
			raise KaldiProcessError('Copy feat defeated.')
		else:
			#if sys.getsizeof(out) > 10000000000:
			#    print('Warning: Data is extramely large. It could not be used correctly sometimes.') 
			allData += KaldiArk(out)
		return allData

	for fileName in allFiles:
		if fileName[-3:].lower() == "npy":
			allData_dict = loadNpyFile(fileName,allData_dict)
		elif fileName[-3:].lower() in ["ark","scp"]:
			allData_ark = loadArkScpFile(fileName,allData_ark,fileName[-3:].lower())
		elif useSuffix == "npy":
			allData_dict = loadNpyFile(fileName,allData_dict)
		elif useSuffix in ["ark","scp"]:
			allData_ark = loadArkScpFile(fileName,allData_ark,useSuffix)
		else:
			raise UnsupportedDataType('Unknown file format. You can assign the <useSuffix> with "scp", "ark" or "npy".')

	if useSuffix == None:
		if allFiles[0][-3:].lower() == "npy":
			return allData_dict + allData_ark.array
		else:
			return allData_ark + allData_dict.ark 
	elif useSuffix == "npy":
		return  allData_dict + allData_ark.array
	else:
		return allData_ark + allData_dict.ark

def load_ali(aliFile,hmm=None,returnPhone=False):
	'''
	Usage:  obj = load_ali("ali.1.gz","graph/final.mdl")

	Load force-alignment data from alignment file. Return a KaldiDict object. If <returnPhone> is True, return phone IDs, or return pdf IDs.
	If <hmm> is None, find the final.mdl automaticlly at the same path with <aliFile>.
	''' 
	global KALDIROOT,kaidiNotFoundError,ENV
	if KALDIROOT is None:
		raise kaidiNotFoundError

	if isinstance(aliFile,str):
		p = subprocess.Popen('ls {}'.format(aliFile),shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=ENV)
		(out,err) = p.communicate()
		if out == b'':
			raise PathError("No such file or dir:{}".format(aliFile))
		else:
			allFiles = out.decode().strip().split('\n')
	else:
		raise UnsupportedDataType('Expected filename-like string but got a {}.'.format(type(aliFile)))

	if hmm == None:
		i = aliFile.rfind('/')
		if i > 0:
			hmm = aliFile[0:i] +'/final.mdl'
		else:
			hmm = './final.mdl'
		if not os.path.isfile(hmm):
			raise PathError("HMM file was not found. Please assign <hmm>.")      
	elif isinstance(hmm,str):
		if not os.path.isfile(hmm):
			raise PathError("No such file:{}".format(hmm))
	else:
		raise UnsupportedDataType("Expected <hmm> is a path-like string but got {}.".format(type(hmm)))
	hmm = os.path.abspath(hmm)

	results = KaldiDict()

	for aliFile in allFiles:
		if not aliFile.endswith('.gz'):
			continue
		else:
			aliFile = os.path.abspath(aliFile)
		
		if returnPhone:
			cmd = 'gunzip -c {} | ali-to-phones --per-frame=true {} ark:- ark,t:-'.format(aliFile,hmm)
		else:
			cmd = 'gunzip -c {} | ali-to-pdf {} ark:- ark,t:-'.format(aliFile,hmm)

		p = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=ENV)
		(out,err) = p.communicate()

		if out == b'':
			err = err.decode()
			print(err)
			raise KaldiProcessError('Load alignment data defeated.')
		else:
			sp = BytesIO(out)
			for line in sp.readlines():
				line = line.decode()
				line = line.strip().split()
				utt = line[0]
				matrix = np.array(line[1:],dtype=np.int32)[:,None]
				results[utt] = matrix
		
	if len(results.utts) == 0:
		raise WrongOperation("Not any .gz file has been loaded.")
	else:
		return results

def load_lat(latFile,hmm,wordSymbol):
	'''
	Usage:  obj = load_lat("lat.gz","graph/final.mdl","graph/words.txt")

	Load lattice data to memory from file. Return a KaldiLattice object.
	''' 	
	new = KaldiLattice()
	new.load(latFile,hmm,wordSymbol)

	return new

def analyze_counts(aliFile,outFile,countPhone=False,hmm=None,dim=None):
	'''
	Usage:  obj = analyze_counts(aliFile,outFile)

	Compute alignment counts in order to normalize acoustic model posterior probability.
	We defaultly compute pdf IDs counts but if <countPhone> is True, compute phone IDs counts.   
	For more help information, look at the Kaldi <analyze-counts> command.
	''' 

	global KALDIROOT,kaidiNotFoundError,ENV
	if KALDIROOT is None:
		raise kaidiNotFoundError

	if isinstance(aliFile,str):
		if os.path.isdir(aliFile):
			raise WrongOperation('Expected file path but got a directory:{}.'.format(aliFile))
		else:
			p = subprocess.Popen('ls {}'.format(aliFile),shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=ENV)
			(out,err) = p.communicate()
			if out == b'':
				raise PathError("No such file:{}".format(aliFile))
	else:
		raise UnsupportedDataType('Expected filename-like string but got a {}.'.format(type(aliFile)))    
	
	if hmm == None:
		i = aliFile.rfind('/')
		if i > 0:
			hmm = aliFile[0:i] +'/final.mdl'
		else:
			hmm = './final.mdl'
		if not os.path.isfile(hmm):
			raise WrongOperation('Did not find HMM file. Please assign <hmm>.')
	elif not os.path.isfile(hmm):
		raise PathError('No such file:{}.'.format(hmm))
	
	if dim == None:
		if countPhone:
			cmd = 'hmm-info {} | grep -i "phones"'.format(hmm)
		else:
			cmd = 'hmm-info {} | grep -i "pdfs"'.format(hmm)
		p = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=ENV)
		(out,err) = p.communicate()
		out = out.decode().strip()
		if out == '':
			print(err.decode())
			raise KaldiProcessError('Acquire HMM information defailed.')
		else:
			dim = out.split()[-1]

	options = '--print-args=False --verbose=0 --binary=false --counts-dim={} '.format(dim)
	if countPhone:
		getAliOption = 'ali-to-phones --per-frame=true'
	else:
		getAliOption = 'ali-to-pdf'
	cmd = "analyze-counts {}\"ark:{} {} \\\"ark:gunzip -c {} |\\\" ark:- |\" {}".format(options,getAliOption,hmm,aliFile,outFile)
	p = subprocess.Popen(cmd,shell=True,stderr=subprocess.PIPE,env=ENV)
	(out,err) = p.communicate()     
	if not os.path.isfile(outFile) or os.path.getsize(outFile) == 0:
		print(err.decode())
		if os.path.isfile(outFile):
			os.remove(outFile)
		raise KaldiProcessError('Analyze counts defailed.')
	else:
		return outFile

def decompress(data):
	'''
	Usage:  obj = decompress(feat)

	Expected <data> is a KaldiArk object whose data-type is "CM", kaldi compressed ark data. Return a KaldiArk object.
	This function is a cover of kaldi-io-for-python tools. For more information about it, please access to https://github.com/vesis84/kaldi-io-for-python/blob/master/kaldi_io/kaldi_io.py 
	'''     
	def _read_compressed_mat(fd):

		# Format of header 'struct',
		global_header = np.dtype([('minvalue','float32'),('range','float32'),('num_rows','int32'),('num_cols','int32')]) # member '.format' is not written,
		per_col_header = np.dtype([('percentile_0','uint16'),('percentile_25','uint16'),('percentile_75','uint16'),('percentile_100','uint16')])

		# Read global header,
		globmin, globrange, rows, cols = np.frombuffer(fd.read(16), dtype=global_header, count=1)[0]
		cols = int(cols)
		rows = int(rows)

		# The data is structed as [Colheader, ... , Colheader, Data, Data , .... ]
		#                         {           cols           }{     size         }
		col_headers = np.frombuffer(fd.read(cols*8), dtype=per_col_header, count=cols)
		col_headers = np.array([np.array([x for x in y]) * globrange * 1.52590218966964e-05 + globmin for y in col_headers], dtype=np.float32)
		data = np.reshape(np.frombuffer(fd.read(cols*rows), dtype='uint8', count=cols*rows), newshape=(cols,rows)) # stored as col-major,

		mat = np.zeros((cols,rows), dtype='float32')
		p0 = col_headers[:, 0].reshape(-1, 1)
		p25 = col_headers[:, 1].reshape(-1, 1)
		p75 = col_headers[:, 2].reshape(-1, 1)
		p100 = col_headers[:, 3].reshape(-1, 1)
		mask_0_64 = (data <= 64)
		mask_193_255 = (data > 192)
		mask_65_192 = (~(mask_0_64 | mask_193_255))

		mat += (p0  + (p25 - p0) / 64. * data) * mask_0_64.astype(np.float32)
		mat += (p25 + (p75 - p25) / 128. * (data - 64)) * mask_65_192.astype(np.float32)
		mat += (p75 + (p100 - p75) / 63. * (data - 192)) * mask_193_255.astype(np.float32)

		return mat.T,rows,cols        
	
	with BytesIO(data) as sp:
		newData = []

		while True:
			data = b''
			utt = ''
			while True:
				char = sp.read(1)
				data += char
				char = char.decode()
				if (char == '') or (char == ' '):break
				utt += char
			utt = utt.strip()
			if utt == '':break
			binarySymbol = sp.read(2)
			data += binarySymbol
			binarySymbol = binarySymbol.decode()
			if binarySymbol == '\0B':
				dataType = sp.read(3).decode()
				if dataType == 'CM ':
					data += 'FM '.encode()
					matrix,rows,cols = _read_compressed_mat(sp)
					data += '\04'.encode()
					data += struct.pack(np.dtype('uint32').char, rows)
					data += '\04'.encode()
					data += struct.pack(np.dtype('uint32').char, cols)
					data += matrix.tobytes()
					newData.append(data)
				else:
					raise UnsupportedDataType("This is not a compressed ark data.")
			else:
				raise WrongDataFormat('Miss right binary symbol.')    
	return KaldiArk(b''.join(newData))

# ---------- Decode Funtions -----------

def decode_lattice(amp,hmm,hclg,wordSymbol,minActive=200,maxActive=7000,maxMem=50000000,beam=10,latBeam=8,acwt=1,config=None,maxThreads=1,outFile=None):
	'''
	Usage: kaldiLatticeObj = decode_lattice(amp,'graph/final.mdl','graph/hclg')

	Decode by generating lattice from acoustic probability. Return a KaldiLattice object or file path if <outFile> is not None.
	Some usual options can be assigned directly. If you want use more, set <config> = your-configure, but if you do this, these usual configures we provided will be ignored.
	You can use .check_config('decode_lattice') function to get configure information you could set.
	Also run shell command "latgen-faster-mapped" to look their meaning.
	''' 
	global KALDIROOT,kaidiNotFoundError,ENV
	if KALDIROOT is None:
		raise kaidiNotFoundError

	if isinstance(amp,KaldiArk):
		pass
	elif isinstance(amp,KaldiDict):
		amp = amp.ark
	else:
		raise UnsupportedDataType("Expected <amp> is a KaldiArk or KaldiDict object.")
	
	for i in [hmm,hclg,wordSymbol]:
		assert isinstance(i,str), "Expected all of <hmm>, <hclg>, <wordSymbol> are path-like string."
		if not os.path.isfile(i):
			raise PathError("No such file:{}".format(i))

	if maxThreads > 1:
		kaldiTool = "latgen-faster-mapped-parallel --num-threads={}".format(maxThreads)
	else:
		kaldiTool = "latgen-faster-mapped" 

	if config == None:    
		config = {}
		config["--allow-partial"] = "true"
		config["--min-active"] = minActive
		config["--max-active"] = maxActive
		config["--max_mem"] = maxMem
		config["--beam"] = beam
		config["--lattice-beam"] = latBeam
		config["--acoustic-scale"] = acwt
	config["--word-symbol-table"] = wordSymbol

	if check_config(name='decode_lattice',config=config):
		for key in config.keys():
			kaldiTool += ' {}={}'.format(key,config[key])

	if outFile != None:
		assert isinstance(outFile,str), "Expected <outFile> is name-like string."
		if not outFile.endswith('.gz'):
			outFile += '.gz'
		cmd1 = '{} {} {} ark:- ark:| gzip -c > {}'.format(kaldiTool,hmm,hclg,outFile)
		p = subprocess.Popen(cmd1,shell=True,stdin=subprocess.PIPE,stderr=subprocess.PIPE,env=ENV)
		(out1,err1) = p.communicate(input=amp)
		if not os.path.isfile(outFile) or os.path.getsize(outFile) == 0:
			err1 = err1.decode()
			print(err1)
			if os.path.isfile(outFile):
				os.remove(outFile)
			raise KaldiProcessError('Generate lattice defeat.')
		else:
			return outFile
	else:
		cmd2 = '{} {} {} ark:- ark:-'.format(kaldiTool,hmm,hclg)
		p = subprocess.Popen(cmd2,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=ENV)
		(out2,err2) = p.communicate(input=amp)
		if out2 == b'':
			err2 = err2.decode()
			print(err2)
			raise KaldiProcessError('Generate lattice defeat.')
		else:
			return KaldiLattice(out2,hmm,wordSymbol)

# ---------- Other Funtions -----------

def check_config(name,config=None):
	'''
	Usage:  configure = check_config(name='compute_mfcc')  or  check_config(name='compute_mfcc',config=your_configure)
	
	Get default configure if <config> is None, or check if given <config> has a right format.
	'''

	assert isinstance(name,str), "<name> should be a name-like string."

	module = importlib.import_module('exkaldi.function_config')
	c = module.configure(name)

	if c is None:
		print('Warning: no default configure for name "{}".'.format(name))
		return None

	if config == None:
		new = {}
		for key,value in c.items():
			new[key] = value[0]
		return new
	else:
		if not isinstance(config,dict):
			raise WrongOperation('<config> has a wrong format. You can use check_config("{}") to look expected configure format.'.format(name))
		for k in config.keys():
			if not k in c.keys():
				raise WrongOperation('No such key: < {} > in {}.'.format(k,name))
			else:
				proto = c[k][1]
				if isinstance(config[k],bool):
					raise WrongOperation('configure <{}> is bool value "{}", but we expected str value like "true" or "false".'.format(k,config[k]))
				elif not isinstance(config[k],proto):
					raise WrongDataFormat("configure <{}> is expected {} but got {}.".format(k,proto,type(config[k])))
			return True

def run_shell_cmd(cmd,inputs=None):
	'''
	Usage:  out,err = run_shell_cmd('ls -lh')

	This is a simple way to run shell command. Return binary string (out,err). 
	Expected <cmd> is command string.
	<input> can be string or bytes.
	'''

	if isinstance(cmd,str):
		shell = True
	elif isinstance(cmd, list):
		shell = False
	else:
		raise WrongOperation("Expected <cmd> is string or list whose menbers are a command and its options.")

	global ENV

	if inputs != None:
		if isinstance(inputs,str):
			inputs = inputs.encode()
		elif isinstance(inputs,bytes):
			pass
		else:
			raise UnsupportedDataType("Expected <inputs> is str or bytes but got {}.".format(type(inputs)))

	p = subprocess.Popen(cmd,shell=shell,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=ENV)
	(out,err) = p.communicate(input=inputs)

	return out,err

def split_file(filePath,chunks=2):
	'''
	Usage:  score = split_file('eval.scp',5)
	Split a large text file into N smaller files by lines. 
	The splited files will be put at the same folder as original file and return their paths as a list.
	'''    
	assert isinstance(chunks,int) and chunks > 1, "Expected <chunks> is int value and larger than 1."

	if not os.path.isfile(filePath):
		raise PathError("No such file:{}.".format(filePath))

	with open(filePath,'r',encoding='utf-8') as fr:
		data = fr.readlines()

	lines = len(data)
	chunkLines = lines//chunks

	if chunkLines == 0:
		chunkLines = 1
		chunks = lines
		t = 0
	else:
		t = lines - chunkLines * chunks

	a = len(str(chunks))
	files = []

	filePath = os.path.abspath(filePath)
	dirIndex = filePath.rfind('/')
	if dirIndex == -1:
		dirName = ""
		fileName = filePath
	else:
		dirName = filePath[:dirIndex+1]
		fileName = filePath[dirIndex+1:]

	suffixIndex = fileName.rfind('.')
	if suffixIndex != -1:
		newFile = dirName + fileName[0:suffixIndex] + "_%0{}d".format(a) + fileName[suffixIndex:]
	else:
		newFile = dirName + fileName + "_%0{}d".format(a)

	for i in range(chunks):
		if i < t:
			chunkData = data[i*(chunkLines+1):(i+1)*(chunkLines+1)]
		else:
			chunkData = data[i*chunkLines:(i+1)*chunkLines]
		with open(newFile%(i),'w',encoding='utf-8') as fw:
			fw.write(''.join(chunkData))
		files.append(newFile%(i))
	
	return files

def pad_sequence(data,shuffle=False,pad=0):
	'''
	Usage:  data,lengths = pad_sequence(listData)

	Pad sequences with maximum length of one batch data. <data> should be a list object whose members have various sequence-lengths.
	If <shuffle> is "True", pad each sequence with random start-index and return padded data and length information of (startIndex,endIndex) of each sequence.
	If <shuffle> is "False", align the start index of all sequences then pad them rear. This will return length information of only endIndex.
	'''
	assert isinstance(data,list), "Expected <data> is a list but got {}.".format(type(data))
	assert isinstance(pad,(int,float)), "Expected <pad> is an int or float value but got {}.".format(type(data))

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

def unpack_padded_sequence(data,lengths,batchSizeDim=1):
	'''
	Usage:  listData = unpack_padded_sequence(data,lengths)

	This is a reverse operation of .pad_sequence() function. Return a list whose members are sequences.
	We defaultly think the dimension 0 of <data> is sequence-length and the dimension 1 is batch-size.
	If the dimension of batch-size is not 1, assign the <batchSizeDim> please.
	'''   
	assert isinstance(data,np.ndarray), "Expected <data> is NumPy array but got {}.".format(type(data))
	assert isinstance(lengths,list), "Expected <lengths> is list whose members are padded start position ( and end position)."
	assert isinstance(batchSizeDim,int) and batchSizeDim >= 0, "<batchSizeDim> should be a non-negative int value."
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

def wer(ref,hyp,ignore=None,mode='all'):
	'''
	Usage:  score = wer('text','1_best',ignore='<sil>')

	Compute WER (word error rate) between <ref> and <hyp>. 
	Return a dict object with score information like: {'WER':0,'allWords':10,'ins':0,'del':0,'sub':0,'SER':0,'wrongSentences':0,'allSentences':1,'missedSentences':0}
	Both <hyp> and <ref> can be text files or list objects.
	'''
	assert mode in ['all','present'], 'Expected <mode> to be "present" or "all".'

	global KALDIROOT,kaidiNotFoundError,ENV
	if KALDIROOT is None:
		raise kaidiNotFoundError

	if ignore == None:

		if isinstance(hyp,list):
			out1 = "\n".join(hyp)
		elif isinstance(hyp,str) and os.path.isfile(hyp):
			with open(hyp,'r',encoding='utf-8') as fr:
				out1 = fr.read()
		else:
			raise UnsupportedDataType('<hyp> is not a result-list or file avalible.')

		if out1 == '':
			raise WrongDataFormat("<hyp> has not data to score.")
		else:
			out1 = out1.encode()

		if isinstance(ref,list):
			out2 = "\n".join(ref)
		elif isinstance(ref,str) and os.path.isfile(ref):
			with open(ref,'r',encoding='utf-8') as fr:
				out2 = fr.read()
		else:
			raise UnsupportedDataType('<ref> is not a result-list or file avalible.')

		if out2 == '':
			raise WrongDataFormat("<ref> has not data to score.")
		
	else:
		assert isinstance(ignore,str), "Expected <ignore> to be a string."
		assert len(ignore.strip()) > 0, "<ignore> is not a string avaliable."

		if isinstance(hyp,list):
			hyp = ("\n".join(hyp)).encode()
			p1 = subprocess.Popen('sed "s/{} //g"'.format(ignore),shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
			(out1,err1) = p1.communicate(input=hyp)
		elif isinstance(hyp,str) and os.path.isfile(hyp):
			p1 = subprocess.Popen('sed "s/{} //g" <{}'.format(ignore,hyp),shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
			(out1,err1) = p1.communicate()
		else:
			raise UnsupportedDataType('<hyp> is not a result-list or file avalible.')

		if out1 == b'':
			raise WrongDataFormat("<hyp> has not data to score.")

		if isinstance(ref,list):
			ref = ("\n".join(ref)).encode()
			p2 = subprocess.Popen('sed "s/{} //g"'.format(ignore),shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
			(out2,err2) = p2.communicate(input=ref)
		elif isinstance(ref,str) and os.path.isfile(ref):
			p2 = subprocess.Popen('sed "s/{} //g" <{}'.format(ignore,ref),shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
			(out2,err2) = p2.communicate()
		else:
			raise UnsupportedDataType('<ref> is not a result-list or file avalible.') 

		if out2 == b'':
			raise WrongDataFormat("<ref> has not data to score.")

		out2 = out2.decode()

	with tempfile.NamedTemporaryFile('w+',encoding='utf-8') as fw:
		fw.write(out2)
		fw.seek(0)
		cmd = 'compute-wer --text --mode={} ark:{} ark,p:-'.format(mode,fw.name)
		p3 = subprocess.Popen(cmd,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=ENV)
		(out3,err3) = p3.communicate(input=out1)
	
	if out3 == b'':
		err3 = err3.decode()
		print(err3)
		raise KaldiProcessError('Compute WER defeated.')
	else:
		out = out3.decode().split("\n")
		score = {}
		pattern1 = '%WER (.*) \[ (.*) \/ (.*), (.*) ins, (.*) del, (.*) sub \]'
		pattern2 = "%SER (.*) \[ (.*) \/ (.*) \]"
		pattern3 = "Scored (.*) sentences, (.*) not present in hyp."
		s1 = re.findall(pattern1,out[0])[0]
		s2 = re.findall(pattern2,out[1])[0]
		s3 = re.findall(pattern3,out[2])[0]    
		score['WER']=float(s1[0])
		score['allWords']=int(s1[2])
		score['ins']=int(s1[3])
		score['del']=int(s1[4])
		score['sub']=int(s1[5])
		score['SER']=float(s2[0])
		score['wrongSentences']=int(s2[1])        
		score['allSentences']=int(s2[2])
		score['missedSentences']=int(s3[1])

		return score

def accuracy(ref,hyp,ignore=None,mode='all'):
	'''
	Usage:  score = accuracy(label,prediction,ignore=0)

	If <mode> is "all", compute one-one matching score. For example, <ref> is (1,2,3,4), and <hyp> is (1,2,2,4), the score will be 0.75.
	If <mode> is "present", only the members of <hyp> which appeared in <ref> will be scored no matter which position it is. 
	For example, <ref> is (1,2,3,4), and <hyp> is (5,4,7,1,9), only "4" and "1" are right results so the score will be 0.4.
	Both <ref> and <hyp> are expected to be iterable objects like list, tuple or NumPy array. They will be flattened before scoring.
	Ignoring specific values if <ignore> is not None.
	'''
	assert mode in ['all','present'], 'Expected <mode> to be "present" or "all".'

	def flatten(iterableObj):
		new = []
		for i in iterableObj:
			if ignore != None:
				if isinstance(i,np.ndarray):
					if i.all() == ignore:
						continue
				elif i == ignore:
					continue
			if isinstance(i,str) and len(i) <= 1:
				new.append(i)
			elif not isinstance(i,Iterable):
				new.append(i)
			else:
				new.extend(flatten(i))
		return new

	x = flatten(ref)
	y = flatten(hyp)

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
				return 1.0
			else:
				return score/len(x)
	else:
		x = sorted(x)
		score = 0
		for i in y:
			if i in x:
				score += 1
		if len(y) == 0:
			if len(x) == 0:
				return 1.0
			else:
				return 0.0
		else:
			return score/len(y)

def edit_distance(ref,hyp,ignore=None):
	'''
	Usage: score = edit_distance(predict,target,ignore=0)

	Compute edit-distance score. 
	Both <ref> and <hyp> should be iterable objects such as string, list, tuple, or NumPy array.
	'''
	assert isinstance(ref,Iterable), "<ref> is not a iterable object."
	assert isinstance(hyp,Iterable), "<hyp> is not a iterable object."
	
	def flatten(iterableObj):
		new = []
		for i in iterableObj:
			if ignore != None:
				if isinstance(i,np.ndarray):
					if i.all() == ignore:
						continue
				elif i == ignore:
					continue
			if isinstance(i,str) and len(i) <= 1:
				new.append(i)
			elif not isinstance(i,Iterable):
				new.append(i)
			else:
				new.extend(flatten(i))
		return new

	x = flatten(ref)
	y = flatten(hyp)

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
	
	return mapping[lenX][lenY]

def log_softmax(data,axis=1):
	'''
	Usage: out = log_softmax(data,1)

	Compute log softmax output of <data> in dimension <axis>: 
					data - log(sum(exp(data))).
	<data> should be a NumPy array.
	'''
	assert isinstance(data,np.ndarray), "Expected <data> is NumPy ndarray but got {}.".format(type(data))
	assert isinstance(axis,int) and axis >= 0, "<axis> should be a positive int value."
	assert axis < len(data.shape), '<axis> {} is out of the dimensions of <data>.'.format(axis)

	dataShape = list(data.shape)
	dataShape[axis] = 1
	dataExp = np.exp(data)
	dataExpLog = np.log(np.sum(dataExp,axis)).reshape(dataShape)

	return data - dataExpLog
