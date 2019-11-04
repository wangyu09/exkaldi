################# Version Information ################
# exkaldi V0.1
# University of Yamanashi (Yu Wang, Chee Siang Leow, Hiromitsu Nishizaki, University of Yamanashi)
# Tsukuba University of Technology (Akio Kobayashi)
# University of Tsukuba (Takehito Utsuro)
# Oct, 31, 2019
#
# ExKaldi Automatic Speech Recognition tookit is designed to build a interface between Kaldi and Deep Learning frameworks with Python Language.
# The main functions are implemented by Kaldi command, and based on this, we developed some extension tools:
# 1, Transform and deal with feature and label data of both Kaldi data format and NumPy format.
# 2, Design and train a neural network acoustic model.
# 3, Build a customized ASR system.
# 4, Recognize you voice from microphone. 
######################################################

import os,sys
import struct,copy,re,time
import math,socket,random
import subprocess,threading
import wave
import queue,tempfile
import numpy as np
from io import BytesIO
import configparser
from collections import Iterable

class PathError(Exception):pass
class UnsupportedDataType(Exception):pass
class WrongDataFormat(Exception):pass
class KaldiProcessError(Exception):pass
class WrongOperation(Exception):pass
class NetworkError(Exception):pass    

def get_kaldi_path():
    '''
    Useage:  KALDIROOT = get_kaldi_path() 
    Return Kaldi root path. If the Kaldi toolkit is not found, it will raise Error.
    '''
    p = subprocess.Popen('which copy-feats',shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    (out,err) = p.communicate()
    if out == b'':
        raise PathError('Kaldi was not found. Make sure it has been installed correctly.')
    else:
        return out.decode().strip()[0:-23]

KALDIROOT = get_kaldi_path()

# ------------ Basic Class ------------
#DONE
class KaldiArk(bytes):
    '''
    Useage: obj = KaldiArk(binaryData) or obj = KaldiArk()
    
    KaldiArk is a subclass of bytes. It holds the Kaldi ark data in binary type. 
    KaldiArk and KaldiDict object have almost the same attributes and methods, and they can do some mixed operations such as "+" and "concat" and so on.
    Moreover, forced-alignment can also be held by KaldiArk and KaldiDict object, and we defined it as int32 data type that is new of tranditional Kaldi data format.

    '''
    def __init__(self,*args):
        super(KaldiArk,self).__init__()
    
    def _read_one_record(self,fp):
        '''
        Useage:  (utt,dataType,rows,cols,buf) = _read_one_record(binaryPointer)
        
        Read one piece of record from binary ark data. Return (utterance id, dtype of data, rows of data, clos of data, object of binary data)
        We don't support to use it in external way.

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
                raise WrongDataFormat('Miss <utt id> in front of utterance.')
        binarySymbol = fp.read(2).decode()
        if binarySymbol == '\0B':
            dataType = fp.read(3).decode() 
            if dataType == 'CM ':
                fp.close()
                raise UnsupportedDataType('This is compressed ark data. Use load(<arkFile>) function to load ark file again or \
                                            use decompress(<KaldiArk>) function to decompress it firstly.')                    
            elif dataType == 'FM ' or dataType == 'IM ':
                sampleSize = 4
            elif dataType == 'DM ':
                sampleSize = 8
            else:
                fp.close()
                raise WrongDataFormat('Expected data type FM(float32),DM(float64),IM(int32),CM(compressed ark data) but got {}.'.format(dataType))
            s1, rows, s2, cols = np.frombuffer(fp.read(10), dtype='int8,int32,int8,int32', count=1)[0]
            rows = int(rows)
            cols = int(cols)
            buf = fp.read(rows * cols * sampleSize)
        else:
            fp.close()
            raise WrongDataFormat('Miss <binary symbol> in front of utterance.')
        return (utt,dataType,rows,cols,buf)
    
    def __str__(self):
        return "KaldiArk object with unviewable binary data. To looking its content, please use .array method."

    @property
    def lens(self):
        '''
        Useage:  lens = obj.lens
        
        Return a tuple: ( the numbers of all utterances, the frames of each utterance ). The first one is an int, and second one is a list.
        If there is not any data, return (0,None)

        '''
        _lens = None
        if self != b'':
            sp = BytesIO(self)
            _lens =[]
            while True:
                (utt,dataType,rows,cols,buf) = self._read_one_record(sp)
                if utt == None:
                    break
                _lens.append(rows)
            sp.close()
        if _lens == None:
            return (0,_lens)
        else:
            return (len(_lens),_lens)
    
    @property
    def dim(self):
        '''
        Useage:  dim = obj.dim
        
        Return an int: data dimension.
        If it is alignment data, dim will be 1.

        '''
        _dim = None
        if self != b'':
            sp = BytesIO(self)
            (utt,dataType,rows,cols,buf) = self._read_one_record(sp)
            _dim = cols
            sp.close()
        return _dim
    
    @property
    def dtype(self):
        '''
        Useage:  dtype = obj.dtype
        
        Return an str: data type. We only use 'float32','float64' and 'int32'.

        '''
        _dtype = None
        if self != b'':
            sp = BytesIO(self)
            (utt,dataType,rows,cols,buf) = self._read_one_record(sp)
            sp.close()
            if dataType == 'FM ':
                _dtype = 'float32'
            elif dataType == 'DM ':
                _dtype = 'float64'
            else:
                _dtype = 'int32'                
        return _dtype

    def to_dtype(self,dtype):
        '''
        Useage:  newObj = obj.to_dtype('float')
        
        Return a new KaldiArk object. 'float' will be treated as 'float32' and 'int' will be 'int32'.

        '''

        if len(self) == 0 or self.dtype == dtype:
            return copy.deepcopy(self)
        else:
            if dtype == 'float32' or dtype == 'float':
                newDataType = 'FM '
            elif dtype == 'float64':
                newDataType = 'DM '
            elif dtype == 'int32' or dtype == 'int':
                newDataType = 'IM '
            else:
                raise WrongOperation('Expected dtype <int><int32><float><float32><float64> but got {}'.format(dtype))
            
            newData = []
            sp = BytesIO(self)
            while True:
                (utt,dataType,rows,cols,buf) = self._read_one_record(sp)
                if utt == None:
                    break
                if dataType == 'FM ': 
                    matrix = np.frombuffer(buf, dtype=np.float32)
                elif dataType == 'IM ':
                    matrix = np.frombuffer(buf, dtype=np.int32)
                else:
                    matrix = np.frombuffer(buf, dtype=np.float64)
                newMatrix = np.array(matrix,dtype=dtype).tobytes()
                data = (utt+' '+'\0B'+newDataType).encode()
                data += '\04'.encode()
                data += struct.pack(np.dtype('uint32').char, rows)
                data += '\04'.encode()
                data += struct.pack(np.dtype('uint32').char, cols)
                data += newMatrix
                newData.append(data)
            sp.close()
            return KaldiArk(b''.join(newData))

    @property
    def utts(self):
        '''
        Useage:  utts = obj.utts
        
        Return a list: including all utterance id.

        '''
        allUtts = []
        sp = BytesIO(self)
        while True:
            (utt,dataType,rows,cols,buf) = self._read_one_record(sp)
            if utt == None:
                break
            allUtts.append(utt)
        sp.close()
        return allUtts
    
    def check_format(self):
        '''
        Useage:  obj.check_format()
        
        Check if data has a correct kaldi ark data format. If had, return True, or raise error.

        '''
        if self != b'':
            _dim = 'unknown'
            _dataType = 'unknown'
            sp = BytesIO(self)
            while True: 
                (utt,dataType,rows,cols,buf) = self._read_one_record(sp)
                if utt == None:
                    break
                if _dim == 'unknown':
                    _dim = cols
                    _dataType = dataType
                elif cols != _dim:
                    sp.close()
                    raise WrongDataFormat("Expect dim {} but got{} at utt id {}".format(_dim,cols,utt))
                elif _dataType != dataType:
                    sp.close()
                    raise WrongDataFormat("Expect type {} but got{} at utt id {}".format(_dataType,dataType,utt))                    
                else:
                    try:
                        if dataType == 'FM ':
                            vec = np.frombuffer(buf, dtype=np.float32)
                        elif dataType == 'IM ':
                            vec = np.frombuffer(buf, dtype=np.int32)
                        else:
                            vec = np.frombuffer(buf, dtype=np.float64)
                    except Exception as e:
                        sp.close()
                        print("Wrong data matrix format at utt id {}".format(utt))
                        raise e
            return True
        else:
            return False

    @property
    def array(self):
        '''
        Useage:  newObj = obj.array
        
        Return a KaldiDict object. Transform ark data into numpy array data.

        '''    
        newDict = KaldiDict()
        if self == b'':
            return newDict
        sp = BytesIO(self)
        while True:
            (utt,dataType,rows,cols,buf) = self._read_one_record(sp)
            if utt == None:
                break
            if dataType == 'FM ': 
                newMatrix = np.frombuffer(buf, dtype=np.float32)
            elif dataType == 'IM ':
                newMatrix = np.frombuffer(buf, dtype=np.int32)
            else:
                newMatrix = np.frombuffer(buf, dtype=np.float64)
            if cols > 1:
                newMatrix = np.reshape(newMatrix,(rows,cols))
            newDict[utt] = newMatrix
        sp.close()
        return newDict
    
    def save(self,fileName,chunks=1):
        '''
        Useage:  obj.save('feat.ark') or obj.save('feat.ark',chunks=2)
        
        Save as .ark file. If chunks is larger than 1, split it averagely and save them.

        '''        
        if self == b'':
            raise WrongOperation('No data to save.')

        if sys.getsizeof(self)/chunks > 10000000000:
           print("Warning: Data size is extremely large. Try to save it with a long time.")
        
        if chunks == 1:
            if not fileName.strip().endswith('.ark'):
                fileName += '.ark'
            with open(fileName,'wb') as fw:
                fw.write(self)
        else:
            if fileName.strip().endswith('.ark'):
                fileName = fileName[0:-4]
            sp = BytesIO(self)
            uttLens = []
            while True:
                (utt,dataType,rows,cols,buf) = self._read_one_record(sp)
                if utt == None:
                    break
                if dataType == 'DM ':
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
                with open(fileName+'_ck{}.ark'.format(i),'wb') as fw:
                    fw.write(chunkData)
            sp.close()
    
    def __add__(self,other):
        '''
        Useage:  obj3 = obj1 + obj2
        
        Return a new KaldiArk object. obj2 can be KaldiArk or KaldiDict object.
        Note that if there are the same utt id in both obj1 and obj2, data in the formar will be retained.

        ''' 

        if isinstance(other,KaldiArk):
            pass
        elif isinstance(other,KaldiDict):          
            other = other.ark
        else:
            raise UnsupportedDataType('Excepted KaldiArk or KaldiDict but got {}.'.format(type(other)))
        
        if self.dim != other.dim:
            raise WrongOperation('Expect unified dim but {}!={}.'.format(self.dim,other.dim))        

        selfUtts = self.utts
        newData = []
        op = BytesIO(other)
        while True:
            (outt,odataType,orows,ocols,obuf) = self._read_one_record(op)
            if outt == None:
                break
            elif not outt in selfUtts:
                data = b''
                data += (outt+' ').encode()
                data += '\0B'.encode()        
                data += odataType.encode()
                data += '\04'.encode()
                data += struct.pack(np.dtype('uint32').char, orows)
                data += '\04'.encode()
                data += struct.pack(np.dtype('uint32').char, ocols)
                data += obuf
                newData.append(data)
        op.close()
        return KaldiArk(b''.join([self,*newData]))

    def concat(self,others,axis=1):
        '''
        Useage:  obj3 = obj1.concat(obj2) or newObj = obj1.concat([obj2,obj3....])
        
        Return a new KaldiArk object. obj2,obj3... can be KaldiArk or KaldiDict objects.
        Note that only these utterance ids which appeared in all objects can be retained in concat result. 

        ''' 
        if axis != 1 and axis != 0:
            raise WrongOperation("Expect axis==1 or 0 but got {}.".format(axis))
        if not isinstance(others,(list,tuple)):
            others = [others,]

        for index,other in enumerate(others):
            if isinstance(other,KaldiArk):                 
                continue
            elif isinstance(other,KaldiDict):
                others[index] = other.ark
            else:
                raise UnsupportedDataType('Expect KaldiArk or KaldiDict but got {}.'.format(type(other))) 

        if axis == 1:
            dataType = self.dtype
            if dataType == 'int32':
                newData = self.to_dtype('float32')
            else:
                newData = self
            for other in others:
                with tempfile.NamedTemporaryFile(mode='w+b') as fn:
                    otherDtype = other.dtype
                    if otherDtype == 'int32':
                        other = other.to_dtype('float32')
                    fn.write(other)
                    fn.seek(0)
                    cmd = 'paste-feats ark:- ark:{} ark:-'.format(fn.name)
                    p = subprocess.Popen(cmd,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                    (newData,err) = p.communicate(input=newData)
                    if newData == b'':
                        err = err.decode()
                        raise KaldiProcessError(err)
                    elif dataType == 'int32' and otherDtype == 'int32':
                        newData = bytes(KaldiArk(newData).to_dtype('int32'))
        else:
            sp = BytesIO(self)
            op = [BytesIO(x) for x in others]
            newData = []
            dataBackup = []
            for j in range(len(others)):
                dataBackup.append(dict())
            while True:
                data = b''
                (sutt,sdataType,srows,scols,sbuf) = self._read_one_record(sp)
                if sutt == None:
                    break
                for i,opi in enumerate(op):
                    if sutt in list(dataBackup[i].keys()):
                        sbuf += dataBackup[i][sutt]
                    else:
                        while True:
                            (outt,odataType,orows,ocols,obuf) = self._read_one_record(opi)
                            if outt == None:
                                sp.close()
                                [x.close() for x in op]
                                raise WrongDataFormat('Miss data to concat at {} of {}th member.'.format(sutt,i+1))
                            elif outt == sutt:
                                if ocols != scols:
                                    sp.close()
                                    [x.close() for x in op]
                                    raise WrongDataFormat('Data dim {}!={} at {}.'.format(scols,ocols,sutt))
                                srows += orows
                                sbuf += obuf
                                break
                            else:
                                dataBackup[i][outt] = obuf
                data += (sutt+' '+'\0B'+sdataType).encode()
                data += '\04'.encode()
                data += struct.pack(np.dtype('uint32').char, srows)
                data += '\04'.encode()
                data += struct.pack(np.dtype('uint32').char, scols)
                data += sbuf
                newData.append(data)
            sp.close()
            [x.close() for x in op]
            newData = b''.join(newData)
        return KaldiArk(newData)

    def splice(self,left=4,right=None):
        '''
        Useage:  newObj = obj.splice(4) or newObj = obj.splice(4,3)
        
        Return a new KaldiArk object. If right is None, we define right = left. So if you don't want to splice, set the value = 0.

        ''' 
        if right == None:
            right = left
        
        cmd = 'splice-feats --left-context={} --right-context={} ark:- ark:-'.format(left,right)
        p = subprocess.Popen(cmd,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        (out,err) = p.communicate(input=self)

        if out == b'':
            err = err.decode()
            raise KaldiProcessError(err)
        else:
            return KaldiArk(out)

    def select(self,dims,reserve=False):
        '''
        Useage:  newObj = obj.select(4) or newObj = obj.select('5,10-15') or newObj1,newObj2 = obj.select('5,10-15',True)
        
        Select data according dims. < dims > should be an int or string like "1,5-20".
        If < reserve > is True, return 2 new KaldiArk objects. Or only return selected data.

        ''' 
        _dim = self.dim
        if _dim == 1:
            raise WrongOperation('Cannot select any data from 1-dim data.')

        elif isinstance(dims,int):
            assert dims >= 0, "Expected dims >= 0."
            selectFlag = str(dims)
            if reserve:
                if dims == 0:
                    reserveFlag = '1-{}'.format(_dim-1)
                elif dims == _dim-1:
                    reserveFlag = '0-{}'.format(_dim-2)
                else:
                    reserveFlag = '0-{},{}-{}'.format(dims-1,dims+1,_dim-1)
        elif isinstance(dims,str):
            if reserve:
                reserveFlag = [x for x in range(_dim)]
                for i in dims.strip().split(','):
                    if not '-' in i:
                        reserveFlag[int(i)]=-1    
                    else:
                        i = i.split('-')
                        for j in range(int(i[0]),int(i[1])+1,1):
                            reserveFlag[j]=-1
                temp = ''
                for x in reserveFlag:
                    if x != -1:
                        temp += str(x)+','
                reserveFlag = temp[0:-1]
            selectFlag = dims
        else:
            raise WrongOperation('Expect int or string like 1,4-9,12 but got {}.'.format(type(dims)))
        
        cmdS = 'select-feats {} ark:- ark:-'.format(selectFlag)
        pS = subprocess.Popen(cmdS,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        (outS,errS) = pS.communicate(input=self)
        if outS == b'':
            errS = errS.decode()
            raise KaldiProcessError(errS)
        elif reserve:
            cmdR = 'select-feats {} ark:- ark:-'.format(reserveFlag)
            pR = subprocess.Popen(cmdR,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            (outR,errR) = pR.communicate(input=self)
            if outR == b'':
                errR = errR.decode()
                raise KaldiProcessError(errS)
            else:
                return KaldiArk(outS),KaldiArk(outR)
        else:
            return KaldiArk(outS)

    def subset(self,nHead=0,chunks=1,uttList=None):
        '''
        Useage:  newObj = obj.subset(nHead=10) or newObj = obj.subset(chunks=10) or newObj = obj.subset(uttList=uttList)
        
        Subset data.
        If nHead > 0, return a new KaldiArk object whose content is front nHead pieces of data. 
        Or If chunks > 1, split data averagely as chunks KaidiArk objects. Return a list.
        Or If uttList != None, select utterances if appeared in obj. Return selected data.
        
        ''' 

        if nHead > 0:
            sp = BytesIO(self)
            uttLens = []
            while len(uttLens) < nHead:
                (utt,dataType,rows,cols,buf) = self._read_one_record(sp)
                if utt == None:
                    break
                if dataType == 'DM ':
                    sampleSize = 8
                else:
                    sampleSize = 4
                oneRecordLen = len(utt) + 16 + rows * cols * sampleSize
                uttLens.append(oneRecordLen)                
            sp.seek(0)
            data = sp.read(sum(uttLens))
            sp.close()
            return KaldiArk(data)
        elif chunks > 1:
            datas = []
            sp = BytesIO(self)
            uttLens = []
            while True:
                (utt,dataType,rows,cols,buf) = self._read_one_record(sp)
                if utt == None:
                    break
                if dataType == 'DM ':
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
            sp.close()
            return datas

        elif uttList != None:
            if isinstance(uttList,str):
                uttList = [uttList,]
            elif isinstance(uttList,(list,tuple)):
                pass
            else:
                raise UnsupportedDataType('Expected <uttList> is str,list or tuple but got {}.'.format(type(uttList)))

            newData = []
            sp = BytesIO(self)
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
            sp.close()
            return KaldiArk(b''.join(newData))          
        else:
            raise WrongOperation("Expected one value of <nHead>, <chunks> or <uttList>.")
#DONE
class KaldiDict(dict):
    '''
    Useage:  obj = KaldiDict(binaryData)  or   obj = KaldiDict()

    KaldiDict is a subclass of dict. It is visible form of KaldiArk and holds the feature data and aligment data with NumPy array type. 
    Its keys are the names of all utterances and the values are the data. KaldiDict object can also implement some mixed operations with KaldiArk such as "+" and "concat" and so on.
    Note that KaldiDict has some specific functions which KaldiArk dosen't have.

    '''
    def __init__(self,*args):
        super(KaldiDict,self).__init__(*args)

    @property
    def dim(self):
        '''
        Useage:  dim = obj.dim
        
        Return an int: feature dimensional. If it is alignment data, dim will be 1.

        '''        
        _dim = None
        if len(self.keys()) != 0:
            utt = list(self.keys())[0]
            shape = self[utt].shape
            if len(shape) == 1:
                _dim = 1
            else:
                _dim = shape[1]
        return _dim

    @property
    def lens(self):
        '''
        Useage:  lens = obj.lens
        
        Return a tuple: ( the numbers of all utterances, the frames of each utterance ). The first one is an int, and second one is a list.
        If there is not any data, return (0,None)

        '''
        _lens = None
        allUtts = self.keys()
        if len(allUtts) != 0:
            _lens = []
            for utt in allUtts:
                _lens.append(len(self[utt]))
        if _lens == None:
            return (0,None)
        else:
            return (len(_lens),_lens)

    @property
    def dtype(self):
        '''
        Useage:  dtype = obj.dtype
        
        Return an str: data type. We only use 'float32','float64' and 'int32'.

        '''        
        _dtype = None
        if len(self.keys()) != 0:
            utt = list(self.keys())[0]
            _dtype = str(self[utt].dtype)            
        return _dtype
    
    def to_dtype(self,dtype):
        '''
        Useage:  newObj = obj.to_dtype('float')
        
        Return a new KaldiArk object. 'float' will be treated as 'float32' and 'int' will be 'int32'.

        '''        
        if self.dtype != dtype:
            assert dtype in ['int','int32','float','float32','float64'],'Expected dtype==<int><int32><float><float32><float64> but got {}.'.format(dtype)
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
        Useage:  utts = obj.utts
        
        Return a list: including all utterance id.

        '''        
        return list(self.keys())
    
    def check_format(self):
        '''
        Useage:  obj.check_format()
        
        Check if data has a correct kaldi ark data format. If had, return True, or raise error.

        '''        
        if len(self.keys()) != 0:
            _dim = 'unknown'
            for utt in self.keys():
                if not isinstance(utt,str):
                    raise WrongDataFormat('Expected <utt id> is str but got {}.'.format(type(utt)))
                if not isinstance(self[utt],np.ndarray):
                    raise WrongDataFormat('Exprcted numpy ndarray but got {}.'.format(type(self[utt])))
                matrixShape = self[utt].shape
                if len(matrixShape) > 2:
                    raise WrongDataFormat('Expected matrix shape=[Frames,Feature-dims] or [Frames] but got {}.'.format(matrixShape))
                if len(matrixShape) == 2:
                    if _dim == 'unknown':
                        _dim = matrixShape[1]
                    elif matrixShape[1] != _dim:
                        raise WrongDataFormat("Expected uniform data dim {} but got {} at utt {}.".format(_dim,matrixShape[1],utt))
                else:
                    if _dim == 'unknown':
                        _dim = 1
                    elif _dim != 1:
                        raise WrongDataFormat("Expected uniform data dim {} but got 1-dim data at utt {}.".format(_dim,utt))
            return True
        else:
            return False
    
    @property
    def ark(self):
        '''
        Useage:  newObj = obj.ark
        
        Return a KaldiArk object. Transform numpy array data into ark binary data.

        '''    
        totalSize = 0
        for u in self.keys():
            totalSize += sys.getsizeof(self[u])
        if totalSize > 10000000000:
            print('Warning: Data is extramely large. Try to transform it but it maybe result in MemoryError.')
        
        newData = []
        for utt in self.keys():
            data = b''
            matrix = self[utt]
            if len(matrix.shape) == 1:
                matrix = matrix[:,np.newaxis]
            data += (utt+' ').encode()
            data += '\0B'.encode()
            if matrix.dtype == 'float32':
                data += 'FM '.encode()
            elif matrix.dtype == 'float64':
                data += 'DM '.encode()
            elif matrix.dtype == 'int32':
                data += 'IM '.encode()
            else:
                raise UnsupportedDataType("Expected int32 float32 float64 data, but got {}.".format(matrix.dtype))
            data += '\04'.encode()
            data += struct.pack(np.dtype('uint32').char, matrix.shape[0])
            data += '\04'.encode()
            data += struct.pack(np.dtype('uint32').char, matrix.shape[1])
            data += matrix.tobytes()
            newData.append(data)

        return KaldiArk(b''.join(newData))

    def save(self,fileName,chunks=1):
        '''
        Useage:  obj.save('feat.npy') or obj.save('feat.npy',chunks=2)
        
        Save as .npy file. If chunks is larger than 1, split it averagely and save them.

        '''          
        if len(self.keys()) == 0:
            raise WrongOperation('No data to save.')

        totalSize = 0
        for u in self.keys():
            totalSize += sys.getsizeof(self[u])
        if totalSize > 10000000000:
            print('Warning: Data size is extremely large. Try to save it with a long time.')
        
        if fileName.strip().endswith('.npy'):
            fileName = fileName[0:-4]

        if chunks == 1:          
            datas = tuple(self.items())
            np.save(fileName,datas)
        else:
            if fileName.strip().endswith('.npy'):
                fileName = fileName[0:-4]
            datas = tuple(self.items())
            allLens = len(datas)
            chunkUtts = allLens//chunks
            if chunkUtts == 0:
                chunks = allLens
                chunkUtts = 1
                t = 0
            else:
                t = allLens - chunkUtts * chunks
            for i in range(chunks):
                if i < t:
                    chunkData = datas[i*(chunkUtts+1):(i+1)*(chunkUtts+1)]
                else:
                    chunkData = datas[i*chunkUtts:(i+1)*chunkUtts]
                np.save(fileName+'_ck{}.npy'.format(i),chunkData)         

    def __add__(self,other):
        '''
        Useage:  obj3 = obj1 + obj2
        
        Return a new KaldiDict object. obj2 can be KaldiArk or KaldiDict object.
        Note that if there are the same utt id in both obj1 and obj2, data in the formar will be retained.

        ''' 

        if isinstance(other,KaldiDict):
            pass         
        elif isinstance(other,KaldiArk):
            other = other.array
        else:
            raise UnsupportedDataType('Excepted KaldiArk KaldiDict but got {}.'.format(type(other)))
    
        if self.dim != other.dim:
            raise WrongDataFormat('Expected unified dim but {}!={}.'.format(self.dim,other.dim))

        temp = self.copy()
        selfUtts = list(self.keys())
        for utt in other.keys():
            if not utt in selfUtts:
                temp[utt] = other[utt]
        return KaldiDict(temp)
    
    def concat(self,others,axis=1):
        '''
        Useage:  obj3 = obj1.concat(obj2) or newObj = obj1.concat([obj2,obj3....])
        
        Return a new KaldiDict object. obj2,obj3... can be KaldiArk or KaldiDict objects.
        Note that only these utterance ids which appeared in all objects can be retained in concat result. 

        ''' 
        if axis != 1 and axis != 0:
            raise WrongOperation("Expected axis ==1 or 0 but got {}.".format(axis))

        if not isinstance(others,(list,tuple)):
            others = [others,]

        for index,other in enumerate(others):
            if isinstance(other,KaldiDict):                   
                pass
            elif isinstance(other,KaldiArk):
                others[index] = other.array       
            else:
                raise UnsupportedDataType('Excepted KaldiArk KaldiDict but got {}.'.format(type(other))) 
        
        newDict = KaldiDict()
        for utt in self.keys():
            newMat=[]
            if len(self[utt].shape) == 1:
                newMat.append(self[utt][:,np.newaxis])
            else:
                newMat.append(self[utt])
            length = self[utt].shape[0]
            dim = self[utt].shape[1]
            for other in others:
                if utt in other.keys():
                    temp = other[utt]
                    if len(temp.shape) == 1:
                        temp = temp[:,np.newaxis]
                    if axis == 1 and temp.shape[0] != length:
                        raise WrongDataFormat("Feature frames {}!={} at utt {}.".format(length,temp.shape[0],utt))
                    elif axis == 0 and temp.shape[1] != dim:
                        raise WrongDataFormat("Feature dims {}!={} on utt {}.".format(dim,temp.shape[1],utt))
                    else:
                        newMat.append(temp)                 
                else:
                    #print("Concat Warning: Miss data of utt id {} in later dict".format(utt))
                    break
            if len(newMat) < len(others) + 1:
                #If any member miss the data of current utt id, abandon data of this utt id of all menbers
                continue 
            newMat = np.concatenate(newMat,axis=axis)
            if newMat.shape[1] == 1:
                newMat = newMat.reshape(-1)
            newDict[utt] = newMat
        return newDict

    def splice(self,left=4,right=None):
        '''
        Useage:  newObj = obj.splice(4) or newObj = obj.splice(4,3)
        
        Return a new KaldiDict object. If right is None, we define right = left. So if you don't want to splice, set the value = 0.

        ''' 
        if right == None:
            right = left
        lengths = []
        matrixes = []
        utts = self.keys()

        matrixes.append(utts[0][0:left,:])
        for utt in utts:
            lengths.append((utt,len(self[utt])))
            matrixes.append(self[utt])
        matrixes.append(utts[-1][(0-right):,:])

        matrixes = np.concatenate(matrixes,axis=0)
        N = matrixes.shape[0]
        dim = matrixes.shape[1]

        newMat=np.empty([N,dim*(left+right+1)])
        index = 0
        for lag in range(-left,right+1):
            newMat[:,index:index+dim]=np.roll(matrixes,lag,axis=0)
            index += dim
        newMat = newMat[left:(0-right),:]

        newFea = KaldiDict()
        index = 0
        for utt,length in lengths:
            newFea[utt] = newMat[index:index+length]
            index += length
        return newFea
    
    def select(self,dims,reserve=False):
        '''
        Useage:  newObj = obj.select(4) or newObj = obj.select('5,10-15') or newObj1,newObj2 = obj.select('5,10-15',True)
        
        Select data according dims. < dims > should be an int or string like "1,5-20".
        If < reserve > is True, return 2 new KaldiDict objects. Or only return selected data.

        '''         
        _dim = self.dim
        if _dim == 1:
            raise WrongOperation('Cannot select any data from 1-dim data.')
        elif isinstance(dims,int):
            selectFlag = [dims]
        elif isinstance(dims,str):
            temp = dims.split(',')
            selectFlag = []
            for i in temp:
                if not '-' in i:
                    i = int(i)
                    selectFlag.append(i)
                else:
                    i = i.split('-')
                    selectFlag.extend([x for x in range(int(i[0]),int(i[1])+1)])
        else:
            raise WrongOperation('Expected int or string like 1,4-9,12 but got {}.'.format(type(indexs)))
        
        reserveFlag = list(set(selectFlag))
        seleDict = KaldiDict()
        if reserve:
            reseDict = KaldiDict()
        for utt in self.keys():
            newMat = []
            for index in selectFlag:
                newMat.append(self[utt][:,index][:,np.newaxis])
            newMat = np.concatenate(newMat,axis=1)
            seleDict[utt] = newMat
            if reserve:
                reseDict[utt] = np.delete(self[utt],reserveFlag,1)

        if reserve:
            return seleDict,reseDict
        else:
            return seleDict

    def subset(self,nHead=0,chunks=1,uttList=None):
        '''
        Useage:  newObj = obj.subset(nHead=10) or newObj = obj.subset(chunks=10) or newObj = obj.subset(uttList=uttList)
        
        Subset data.
        If nHead > 0, return a new KaldiDict object whose content is front nHead pieces of data. 
        Or If chunks > 1, split data averagely as chunks KaidiDict objects. Return a list.
        Or If uttList != None, select utterances if appeared in obj. Return selected data.
        
        ''' 
        if nHead > 0:
            newDict = KaldiDict()
            utts = list(self.keys())
            for utt in utts[0:nHead]:
                newDict[utt]=self[utt]
            return newDict

        elif chunks > 1:
            datas = []
            utts = list(self.keys())
            allLens = len(self)
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

            if isinstance(uttList,str):
                uttList = [uttList,]
            elif isinstance(uttList,(list,tuple)):
                pass
            else:
                raise UnsupportedDataType('Expected <uttList> is str,list or tuple but got {}.'.format(type(uttList)))

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
            raise WrongOperation("nHead, chunks and uttList are not allowed be None at the same time.")

    def sort(self,by='frame',reverse=False):
        '''
        Useage:  newObj = obj.sort()
        
        Sort data by frame length or name. Return a new KaldiDict object.

        ''' 
        if len(self.keys()) == 0:
            raise WrongOperation('No data to sort.')

        assert by=='frame' or by=='name', 'We only support sorting by <name> or <frame>.'

        items = self.items()

        if by == 'name':
            sorted(items,key=lambda x:x[0],reverse=reverse)
        else:
            sorted(items,key=lambda x:len(x[1]),reverse=reverse)
        
        newData = KaldiDict()
        for key, value in items:
            newData[key] = value
        
        return newData 

    def merge(self,keepDim=False,sort=False):
        '''
        Useage:  data,uttlength = obj.merge() or data,uttlength = obj.merge(keepDim=True)
        
        Return two value.
        If < keepDim > is False, the first one is 2-dim numpy array, the second one is a list consists of id and frames of each utterance. 
        If < keepDim > is True, the first one will be a list.
        if < sort > is True, it will sort by length of matrix before merge.
        ''' 
        uttLens = []
        matrixs = []
        if sort:
            items = sorted(self.items(), key=lambda x:len(x[1]))
        else:
            items = self.items()
        for utt,mat in items:
            uttLens.append((utt,len(mat)))
            matrixs.append(mat)
        if not keepDim:
            matrixs = np.concatenate(matrixs,axis=0)
        return matrixs,uttLens

    def remerge(self,matrix,uttLens):
        '''
        Useage:  obj = obj.merge(data,uttlength)
        
        Return KaldiDict object. This is a inverse operation of .merge() function.
        
        ''' 
        if len(self.keys()) == 0:
            if isinstance(matrix,list):
                for i,(utt,lens) in enumerate(uttLens):
                    self[utt] = matrix[i]
            elif isinstance(matrix,np.ndarray):
                start = 0
                for utt,lens in uttLens:
                    self[utt] = matrix[start:start+lens]
                    start += lens
            else:
                raise UnsupportedDataType('It is not KaldiDict merged data.')
        else:
            newDict = KaldiDict()
            if isinstance(matrix,list):
                for i,(utt,lens) in enumerate(uttLens):
                    newDict[utt] = matrix[i]
            elif isinstance(matrix,np.ndarray):
                start = 0
                for utt,lens in uttLens:
                    newDict[utt] = matrix[start:start+lens]
                    start += lens
            else:
                raise UnsupportedDataType('It is not KaldiDict merged data.')
            return newDict

    def normalize(self,std=True,alpha=1.0,beta=0.0,epsilon=1e-6,axis=0):
        '''
        Useage:  newObj = obj.normalize()
        
        Return a KaldiDict object. if < std > is True, do alpha*(x-mean)/(std+epsilon)+belta, or do alpha*(x-mean)+belta.
        
        '''
        data,uttLens = self.merge()
        mean = np.mean(data,axis=axis)

        if std == True:  
            std = np.std(data,axis=axis)
            data = alpha*(data-mean)/(std+epsilon)+beta
        else:
            data = alpha*(data-mean)+beta

        newDict = KaldiDict()
        start = 0
        for utt,lens in uttLens:
            newDict[utt] = data[start:(start+lens)]
            start += lens

        return newDict 

    def cut(self,maxFrames):
        '''
        Useage:  newObj = obj.cut(100)
        
        Cut data every maxFrames if its frame length is larger than 1.25 * <maxFrames>.

        '''      
        if len(self.keys()) == 0:
            raise WrongOperation('No data to cut.')

        assert isinstance(maxFrames,int), "Expected < maxFrames > is int but got {}.".format(type(maxFrames))
        assert maxFrames > 0, "Expected < maxFrames > is positive number but got {}.".format(type(maxFrames))

        newData = {}

        cutThreshold = maxFrames + maxFrames//4

        for key in self.keys():
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
#DONE
class KaldiLattice(object):
    '''
    Useage:  obj = KaldiLattice()  or   obj = KaldiLattice(lattice,hmm,wordSymbol)

    KaldiLattice holds the lattice and its related file path: hmm file and WordSymbol file. 
    The <lattice> can be lattice binary data or file path. Both < hmm > and < wordSymbol > are expected file path.
    pythonkaldi.decode_lattice function will return a KaldiLattice object. Aslo, you can define a empty KaldiLattice object and load its data later.

    '''    
    def __init__(self,lat=None,hmm=None,wordSymbol=None):

        self._lat = lat
        self._hmm = hmm
        self._word2id = wordSymbol

        if lat != None:
            assert hmm != None and wordSymbol != None, "Expected HMM file and word-to-id file."
            if isinstance(lat,str):
                self.load(lat,hmm,wordSymbol)
            elif isinstance(lat,bytes):
                pass
            else:
                raise UnsupportedDataType("<lat> is not a correct lattice format: path or byte data.")

    def load(self,latFile,hmm,wordSymbol):
        '''
        Useage:  obj.load('graph/lat.gz','graph/final.mdl','graph/words.txt')

        Load lattice to memory. <latFile> should be file path. <hmm> and <wordSymbol> are expected as file path.
        Note that the new data will coverage original data in current obj.
        We don't check whether it is really a lattice data.

        '''    
        for x in [latFile,hmm,wordSymbol]:
            if not os.path.isfile(x):
                raise PathError('No such file:{}.'.format(x))

        if latFile.endswith('.gz'):
            p = subprocess.Popen('gunzip -c {}'.format(latFile),shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            (out,err)=p.communicate()
            if out == b'':
                print('Lattice load defeat!')
                raise Exception(err.decode())
            else:
                self._lat = out
                self._hmm = hmm
                self._word2id = wordSymbol
        else:
            with open('latFile','rb') as fr:
                out = fr.read()
            if out == b'':
                raise WrongDataFormat('It seems a null file.')
            else:
                self._lat = out
                self._hmm = hmm
                self._word2id = wordSymbol
            
    def save(self,fileName,copyFile=False):
        '''
        Useage:  obj.save("lat.gz")

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
            for x in [self._hmm,self._word2id]:
                if not os.path.isfile(x):
                    raise PathError('No such file:{}.'.format(x))
            i = fileName.rfind('/')
            if i > 0:
                latDir = fileName[0:i+1]
            else:
                latDir = './'
            cmd2 = 'cp -f {} {}; cp -f {} {}'.format(self._hmm, latDir, self._word2id, latDir)
            p2 = subprocess.Popen(cmd2,shell=True,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
            (out2,err2) = p2.communicate(input=self._lat)

    @property
    def value(self):
        '''
        Useage:  lat = obj.value

        Return a tuple:(lattice-data, hmm-file, wordSymbol-file). 
        ''' 
        return (self._lat,self._hmm,self._word2id)

    def get_1best_words(self,minLmwt=1,maxLmwt=None,Acwt=1.0,outFile=None):

        raise WrongOperation('get_1best_words() has been removed in current version. Try to use get_1best() please.')
    
    def get_1best(self,lmwt=1,maxLmwt=None,acwt=1.0,outFile=None,phoneSymbol=None):
        '''
        Useage:  out = obj.get_1best(minLmwt=1)

        Return a dict object. Its key is lm weight, and value will be result-list if <outFile> is False or result-file-path if <outFile> is not None. 
        If <phoneSymbol> is not True, return phones of 1best words.

        ''' 
        if self._lat == None:
            raise WrongOperation('No any data in lattice.')

        for x in [self._hmm,self._word2id]:
            if not os.path.isfile(x):
                raise PathError('No such file:{}.'.format(x))

        KALDIROOT = get_kaldi_path()

        if maxLmwt != None:
            if maxLmwt < lmwt:
                raise WrongOperation('<maxLmwt> must larger than <minLmwt>.')
            else:
                maxLmwt += 1
        else:
            maxLmwt = lmwt + 1
        
        if phoneSymbol != None:
            useLexicon = phoneSymbol
        else:
            useLexicon = self._word2id

        result = {}
        if outFile != None:
            for LMWT in range(lmwt,maxLmwt,1):
                if phoneSymbol != None:
                    cmd0 = KALDIROOT+'/src/latbin/lattice-align-phones --replace-output-symbols=true {} ark:- ark:- | '.format(self._hmm)
                else:
                    cmd0 = ''
                cmd1 = cmd0 + KALDIROOT+'/src/latbin/lattice-best-path --lm-scale={} --acoustic-scale={} --word-symbol-table={} --verbose=2 ark:- ark,t:- | '.format(LMWT,acwt,useLexicon)
                cmd2 = KALDIROOT+'/egs/wsj/s5/utils/int2sym.pl -f 2- {} > {}.{}'.format(useLexicon,outFile,LMWT)
                cmd = cmd1 + cmd2
                p = subprocess.Popen(cmd,shell=True,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
                (out,err) = p.communicate(input=self._lat)
                if not os.path.isfile('{}.{}'.format(outFile,LMWT)):
                    err = err.decode()
                    logFile = '{}.{}.log'.format(outFile,LMWT)
                    with open(logFile,'w') as fw:
                        fw.write(err)
                    raise KaldiProcessError('Lattice to 1-best Defeated. Look the log file {}.'.format(logFile))
                else:
                    result[LMWT] = '{}.{}'.format(outFile,LMWT)
        else:
            for LMWT in range(lmwt,maxLmwt,1):
                if phoneSymbol != None:
                    cmd0 = KALDIROOT+'/src/latbin/lattice-align-phones --replace-output-symbols=true {} ark:- ark:- | '.format(self._hmm)
                    #cmd0 = KALDIROOT+'/src/latbin/lattice-to-phone-lattice {} ark:- ark:- | '.format(self._hmm)
                else:
                    cmd0 = ''
                cmd1 = cmd0 + KALDIROOT+'/src/latbin/lattice-best-path --lm-scale={} --acoustic-scale={} --word-symbol-table={} --verbose=2 ark:- ark,t:- |'.format(LMWT,acwt,useLexicon)
                cmd2 = KALDIROOT+'/egs/wsj/s5/utils/int2sym.pl -f 2- {} '.format(useLexicon)
                cmd = cmd1+cmd2
                p = subprocess.Popen(cmd,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                (out,err) = p.communicate(input=self._lat)
                if out == b'':
                    err = err.decode()
                    logFile = '{}.{}.log'.format(outFile,LMWT)
                    with open(logFile,'w') as fw:
                        fw.write(err)
                    raise KaldiProcessError('Lattice to 1-best Defeated. Look the log file {}.'.format(logFile))
                else:
                    out = out.decode().split("\n")
                    result[LMWT] = out[0:-1]
        if maxLmwt == None:
            result = result[lmwt]
        return result
    
    def scale(self,acwt=1,invAcwt=1,ac2lm=0,lmwt=1,lm2ac=0):
        '''
        Useage:  newObj = obj.sacle(inAcwt=0.2)

        Scale lattice. Return a new KaldiLattice object.

        ''' 
        if self._lat == None:
            raise WrongOperation('No any lattice to scale.')

        for x in [self._hmm,self._word2id]:
            if not os.path.isfile(x):
                raise PathError('Missing file:{}.'.format(x))                

        for x in [acwt,invAcwt,ac2lm,lmwt,lm2ac]:
            assert isinstance(x,int) and x>= 0,"Expected scale is positive int value."
        
        cmd = KALDIROOT+'/src/latbin/lattice-scale'
        cmd += ' --acoustic-scale={}'.format(acwt)
        cmd += ' --acoustic2lm-scale={}'.format(ac2lm)
        cmd += ' --inv-acoustic-scale={}'.format(invAcwt)
        cmd += ' --lm-scale={}'.format(lmwt)
        cmd += ' --lm2acoustic-scale={}'.format(lm2ac)
        cmd += ' ark:- ark:-'

        p = subprocess.Popen(cmd,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        (out,err) = p.communicate(input=self._lat)

        if out == b'':
            raise KaldiProcessError(err.decode())
        else:
            return KaldiLattice(out,self._hmm,self._word2id)

    def add_penalty(self,penalty=0):
        '''
        Useage:  newObj = obj.add_penalty(0.5)

        Add penalty. Return a new KaldiLattice object.

        ''' 
        if self._lat == None:
            raise WrongOperation('No any lattice to scale.')
        for x in [self._hmm,self._word2id]:
            if not os.path.isfile(x):
                raise PathError('No such file:{}.'.format(x))     

        assert isinstance(penalty,(int,float)) and penalty>= 0, "Expected <penalty> is positive int or float value."
        
        cmd = KALDIROOT+'/src/latbin/lattice-add-penalty'
        cmd += ' --word-ins-penalty={}'.format(penalty)
        cmd += ' ark:- ark:-'

        p = subprocess.Popen(cmd,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        (out,err) = p.communicate(input=self._lat)

        if out == b'':
            raise KaldiProcessError(err.decode())
        else:
            return KaldiLattice(out,self._hmm,self._word2id)

    def get_nbest(self,n,acwt=1,outFile=None,outAliFile=None,requreCost=False):
        '''
        Useage:  out = obj.get_nbest(minLmwt=1)

        Return a dict object. Its key is lm weight, and value will be result-list if <outFile> is False or result-file-path if <outFile> is True. 

        ''' 
        if self._lat == None:
            raise WrongOperation('No any data in lattice.')

        for x in [self._hmm,self._word2id]:
            if not os.path.isfile(x):
                raise PathError('Missed file:{}.'.format(x))

        cmd = KALDIROOT+'/src/latbin/lattice-to-nbest --acoustic-scale={} --n={} ark:- ark:- |'.format(acwt,n)
        if outFile != None:
            assert isinstance(outFile,str), 'Expected string-like file name but got {}.'.format(outFile)
            if outAliFile != None:
                if not outAliFile.endswith('.gz'):
                    outAliFile += '.gz'
                cmd += KALDIROOT+'/src/latbin/nbest-to-linear ark:- "ark,t:|gzip -c > {}" "ark,t:|{}/egs/wsj/s5/utils/int2sym.pl -f 2- {} > {}"'.format(outAliFile,KALDIROOT,self._word2id,outFile)
                if requreCost:
                    outCostFile = outFile+'.cost'
                    cmd += ' ark,t:{}.lm ark,t:{}.ac'.format(outCostFile,outCostFile)
                p = subprocess.Popen(cmd,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                (out,err) = p.communicate(input=self._lat)
                if not os.path.isfile(outFile):
                    print(err.decode())
                    raise KaldiProcessError('Get n best defeat.')
                else:
                    if requreCost:
                        return (outFile, outAliFile, outCostFile+'.lm', outCostFile+'.ac')
                    else:
                        return (outFile, outAliFile)
            else:
                with tempfile.NamedTemporaryFile('w+') as outAliFile:
                    cmd += KALDIROOT+'/src/latbin/nbest-to-linear ark:- ark:{} "ark,t:|{}/egs/wsj/s5/utils/int2sym.pl -f 2- {} > {}"'.format(outAliFile.name,KALDIROOT,self._word2id,outFile)
                    if requreCost:
                        outCostFile = outFile+'.cost'
                        cmd += ' ark,t:{}.lm ark,t:{}.ac'.format(outCostFile,outCostFile)
                    p = subprocess.Popen(cmd,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                    (out,err) = p.communicate(input=self._lat)
                if not os.path.isfile(outFile):
                    print(err.decode())
                    raise KaldiProcessError('Get n best defeat.')
                else:
                    if requreCost:
                        return (outFile, outCostFile+'.lm', outCostFile+'.ac')
                    else:
                        return outFile
        else:
            with tempfile.NamedTemporaryFile('w+') as outCostFile_lm:  
                with tempfile.NamedTemporaryFile('w+') as outCostFile_ac:
                    if outAliFile != None:
                        assert isinstance(outAliFile,str), 'Expected string-like alignment file name but got {}.'.format(outAliFile)
                        if not outAliFile.endswith('.gz'):
                            outAliFile += '.gz'
                        cmd += KALDIROOT+'/src/latbin/nbest-to-linear ark:- "ark,t:|gzip -c > {}" "ark,t:|{}/egs/wsj/s5/utils/int2sym.pl -f 2- {}"'.format(outAliFile,KALDIROOT,self._word2id)    
                        if requreCost:
                            cmd += ' ark,t:{} ark,t:{}'.format(outCostFile_lm.name,outCostFile_ac.name)
                        p = subprocess.Popen(cmd,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                        (out,err) = p.communicate(input=self._lat)
                    else:
                        with tempfile.NamedTemporaryFile('w+') as outAliFile:
                            cmd += KALDIROOT+'/src/latbin/nbest-to-linear ark:- ark:{} "ark,t:|{}/egs/wsj/s5/utils/int2sym.pl -f 2- {}"'.format(outAliFile.name,KALDIROOT,self._word2id)    
                            if requreCost:
                                cmd += ' ark,t:{} ark,t:{}'.format(outCostFile_lm.name,outCostFile_ac.name)
                            p = subprocess.Popen(cmd,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                            (out,err) = p.communicate(input=self._lat)

                    if out == b'':
                        print(err.decode())
                        raise KaldiProcessError('Get n best defeat.')
                    else:
                        if requreCost:
                            out = out.decode().split("\n")[:-1]
                            allResult = []
                            outCostFile_lm.seek(0)
                            outCostFile_ac.seek(0)
                            lines_ac = outCostFile_ac.read().split("\n")[:-1]
                            lines_lm = outCostFile_lm.read().split("\n")[:-1]
                            for result, ac_score, lm_score in zip(out,lines_ac,lines_lm):
                                    allResult.append((result,float(ac_score.split()[1]),float(lm_score.split()[1])))
                            out = allResult
                        else:
                            out = out.decode().split("\n")[:-1]

                        exUtt = None
                        new = []
                        temp = []
                        for i in out:
                            if isinstance(i,tuple):
                                utt = i[0][0:i[0].find('-')]
                            else:
                                utt = i[0:i.find('-')]
                            if exUtt == None:
                                exUtt = utt
                                temp.append(i)
                            elif utt == exUtt:
                                temp.append(i)
                            else:
                                new.append(temp)
                                temp = []
                                exUtt = utt
                        if len(temp) > 0:
                            new.append(temp)
                        return new                            

    def __add__(self,other):
        '''
        Useage:  lat3 = lat1 + lat2

        Return a new KaldiLattice object. lat2 must be KaldiLattice object.
        Note that this is only a simple additional operation to make two lattices being one.

        '''
        assert isinstance(other,KaldiLattice), "Expected KaldiLattice but got {}.".format(type(other))

        if self._lat == None:
            return other
        elif other._lat == None:
            return self
        elif self._hmm != other._hmm or self._word2id != other._word2id:
            raise WrongOperation("Both two members must use the same HMM model and word-symbol file.")
        
        newLat = self._lat + other._lat 

        for x in [self._hmm,self._word2id]:
            if not os.path.isfile(x):
                raise PathError('Missing file:{}.'.format(x))     

        return KaldiLattice(newLat, self._hmm, self._word2id)

# ------------ Other Class -----------

class Supporter(object):
    '''
    Useage:  supporter = Supporter(outDir='Result')

    Supporter is a class to be similar to chainer report. But we designed some useful functions such as save model by maximum accuracy and adjust learning rate.
    '''      
    def __init__(self,outDir='Result'):

        self.currentField = {}

        self.globalField = []

        if outDir.endswith('/'):
            outDir = outDir[:-1]
        self.outDir = outDir

        self.count = 0

        if not os.path.isdir(self.outDir):
            os.mkdir(self.outDir)

        self.logFile = self.outDir+'/log'

        with open(self.logFile,'w'):
            pass
        
        self.lastSavedModel = {}
        self.savingThreshold = None

        self._allKeys = []

        self.startTime = None
        
    def send_report(self,x,*args):
        '''
        Useage:  supporter = obj.send_report({"epoch":epoch,"train_loss":loss,"train_acc":acc})

        Send information and thses info will be retained untill you do the statistics by using obj.collect_report().

        '''           
        keys = list(x)

        allKeys = list(self.currentField)
    
        for i in keys: 
            value = x[i]
            try:
                value=float(value.data)
            except:
                pass
            i = i.lower()
            if not i in allKeys:
                self.currentField[i] = []
            self.currentField[i].append(value)

    def collect_report(self,keys=None,plot=True):
        '''
        Useage:  supporter = obj.collect_report(plot=True)

        Do the statistics of received information. The result will be saved in outDir/log file. If < keys > is not None, only collect the data in keys. 
        If < plot > is True, print the statistics result to standard output.
        
        '''   
        if keys == None:
            keys = list(self.currentField)
    
        self.globalField.append({})

        allKeys = list(self.currentField)
        self._allKeys.extend(allKeys)
        self._allKeys = list(set(self._allKeys))

        message = ''
        for i in keys:
            if i in allKeys:
                mn = float(np.mean(self.currentField[i]))
                if type(self.currentField[i][0]) == int:
                    mn = int(mn)
                    message += (i + ':%d    '%(mn))
                else:
                    message += (i + ':%.5f    '%(mn))
                self.globalField[-1][i] = mn
            else:
                message += (i + ':-----    ')

        # Print to log file
        #if self.log[-2] != '[':
        #    self.log[-2] += ','
        #self.log[-1] = '    {'
        #allKeys = list(self.globalField[-1].keys())
        #for i in allKeys[:-1]:
        #    self.log.append('        "{}": {},'.format(i,self.globalField[-1][i]))
        #self.log.append('        "{}": {}'.format(allKeys[-1],self.globalField[-1][allKeys[-1]]))
        #self.log.append('    }')
        #self.log.append(']')

        with open(self.logFile,'a') as fw:
            fw.write(message + '\n')
        
        # Print to screen
        if plot:
            print(message)
        # Clear
        self.currentField = {}

    def save_model(self,saveFunc,models,byKey=None,maxValue=True):
        '''
        Useage:  obj.save_model(saveFunc,models)

        Save model when you use this function. Your can give <iterSymbol> and it will be add to the end of file name.
        If you use < byKey > and set < maxValue >, model will be saved only while meeting the condition.
        We use chainer as default framework, but if you don't, give the < saveFunc > specially please. 
        
        ''' 
        assert isinstance(models,dict), "Expected <models> is dict whose key is model-name and value is model-object."

        if self.currentField != {}:
            self.collect_report(plot=False)

        if 'epoch' in self.globalField[-1].keys():
            suffix = '_'+str(self.globalField[-1]['epoch'])+'_' 
        else:
            suffix = "_"

        if byKey == None:
            for name in models.keys():
                fileName = self.outDir+'/'+name+suffix[:-1]+'.model'
                saveFunc(fileName,models[name])
                self.lastSavedModel[name] = fileName
        else:
            byKey = byKey.lower()
            if not byKey in self.globalField[-1].keys():
                print("Warning: Cannot save model, Because key <{}> has not been reported.".format(byKey))
                return
            else:
                value = self.globalField[-1][byKey]

            save = False

            if self.savingThreshold == None:
                self.savingThreshold = value
                save = True
            else:
                if maxValue == True and value > self.savingThreshold:
                    self.savingThreshold = value
                    save = True
                elif maxValue == False and value < self.savingThreshold:
                    self.savingThreshold = value
                    save = True

            if save:
                for name in models.keys():
                    if isinstance(value,float):
                        value = ('%.5f'%(value)).replace('.','')
                    else:
                        value = str(value)
                    fileName = self.outDir+'/'+ name + suffix + value + '.model'
                    saveFunc(fileName,models[name])
                    if name in self.lastSavedModel.keys():
                        os.remove(self.lastSavedModel[name])
                    self.lastSavedModel[name] = fileName

    @property
    def finalModel(self):
        '''
        Useage:  model = obj.finalModel

        Get the final saved model. Return a dict whose key is model name and value is model path. 
        
        ''' 
        return self.lastSavedModel
   
    def judge(self,key,condition,threshold,byDeltaRate=False):
        '''
        Useage:  newLR = obj.judge('train_loss','<',0.1)

        Return True or False. And If <key> is not reported before, return False.

        if <byDeltaRate> is True, we compute:
                           abs( value - value_pre / value )  
        and compare it with threshold value.
        
        ''' 
        assert condition in ['>','>=','<=','<','==','!='], '<condiction> is not a correct conditional operator.'
        assert isinstance(threshold,(int,float)), '<threshold> should be float or int value.'

        if self.currentField != {}:
            self.collect_report(plot=False)
        
        if byDeltaRate == True:
            p = []
            for i in range(len(self.globalField)-1,-1,-1):
                if key in self.globalField[i].keys():
                    p.append(self.globalField[i][key])
                if len(p) == 2:
                    value = str(abs(p[0]-p[1]/p[0]))
                    return eval(value+condition+str(threshold))
            return False
        else:
            for i in range(len(self.globalField)-1,-1,-1):
                if key in self.globalField[i].keys():
                    value = str(self.globalField[i][key])
                    return eval(value+condition+str(threshold))
            return False

    def dump(self,logFile=None):
        if logFile != None:
            if not os.path.isfile(logFile):
                raise PathError('No such file:{}.'.format(logFile))
            else:
                with open(logFile) as fr:
                    lines = fr.readlines()
                allData = []
                for line in lines:
                    line = line.strip()
                    if len(line) != "":
                        lineData = {}
                        line = line.split()
                        for i in line:
                            i = i.split(":")
                            try:
                                v = int(i[1])
                            except ValueError:
                                v = float(i[1])
                            lineData[i[0]] = int(i[1])
                        allData.append(lineData)
                return allData
        else:
            if self.currentField != {}:
                self.collect_report(plot=False)
            
            if self.globalField != []:
                return self.globalField
            else:
                raise WrongOperation('Not any information to dump.')

class DataIterator(object):
    '''
    Usage: obj = DataIterator(data,64) or obj = DataIterator('train.scp',64,chunks='auto',processFunc=function)

    This is a imporved data interator. You try its distinctive ability. 
    If you give it a large scp file of train data, it will split it into n smaller chunks and load them into momery alternately with parallel thread. 
    It will shuffle the original scp file and split again while new epoch.

    '''
    def __init__(self,scpFiles,processFunc,batchSize,chunks='auto',otherArgs=None,shuffle=False,validDataRatio=0.0):

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
            raise UnsupportedDataType('Expected scp file-like str or list.')

        if isinstance(chunks,int):
            assert chunks>0, "Expected chunks is a positive int number but got {}.".format(chunks)
        elif chunks != 'auto':
            raise WrongOperation('Expected chunks is a positive int number or <auto> but got {}.'.format(chunks))

        temp = []
        for scpFile in out:
            with open(scpFile,'r') as fr:
                temp.extend(fr.read().strip().split('\n'))
        K = int(len(temp)*(1-validDataRatio))
        self.validFiles = temp[K:]
        self.allFiles = temp[0:K]

        if chunks == 'auto':
            #Compute the chunks automatically
            sampleChunk = random.sample(self.allFiles,10)
            with tempfile.NamedTemporaryFile('w',suffix='.scp') as sampleFile:
                sampleFile.write('\n'.join(sampleChunk))
                sampleFile.seek(0)
                sampleChunkData = load(sampleFile.name)
            meanLength = int(np.mean(sampleChunkData.lens[1]))
            autoChunkSize = math.ceil(50000/meanLength)  # Use 30000 frames as threshold 
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
            else:
                self.currentPosition = iEnd
                self._isNewEpoch = False
        else:
            if iEnd >= N:
                rest = iEnd - N
                while self.loadDatasetThread.is_alive():
                    time.sleep(0.1)
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
    def epoch(self):
        return self._epoch

    @property
    def isNewEpoch(self):
        return self._isNewEpoch

    @property
    def isNewChunk(self):
        return self._isNewChunk

    def getValiData(self,processFunc=None,batchSize=None,chunks='auto',otherArgs=None,shuffle=False):

        if len(self.validFiles) == 0:
            raise WrongOperation('No reserved validation data.')   

        if processFunc == None:
            processFunc = self.fileProcessFunc
        
        if batchSize == None:
            batchSize = self._batchSize

        if isinstance(chunks,int):
            assert chunks > 0,"Expected chunks is a positive int number."
        elif chunks != 'auto':
            raise WrongOperation('Expected chunks is a positive int number or <auto> but got {}.'.format(chunks))

        if otherArgs == None:
            otherArgs = self.otherArgs

        with tempfile.NamedTemporaryFile('w',suffix='.scp') as validScpFile:
            validScpFile.write('\n'.join(self.validFiles))
            validScpFile.seek(0)                
            validIterator = DataIterator(validScpFile.name,processFunc,batchSize,chunks,otherArgs,shuffle,0)

        return validIterator

import pyaudio

## Not receive but can send
## not send but receive wait

class SpeakClient(object):

    def __init__(self):

        self.p = pyaudio.PyAudio()
        self.client = None

        self.threadManager = {}
        self._counter = 0

        self.dataQueue = queue.Queue()
        self.resultQueue = queue.Queue()
        
        self.localErrFlag = False
        self.remoteErrFlag = False
        self.endFlag = False
        self.safeFlag = False

        self.config_wave_format()

    def __enter__(self,*args):
        self.safeFlag = True
        return self
    
    def __exit__(self,errType,errValue,errTrace):
        if errType == KeyboardInterrupt:
            self.endFlag = True
        self.wait()
        if self.client != None:
            self.client.close()
        self.p.terminate()

        #self._counter = 0
        #self.dataQueue.queue.clear()
        #self.resultQueue.queue.clear()
        #self.client=None
        #for name in self.threadManager.keys():
        #   self.threadManager[name] = None
        #self.endFlag = False
        #self.errFlag = False
    
    def wait(self):
        for name,thread in self.threadManager.items():
            if thread.is_alive():
                thread.join()

    def close(self):
        self.endFlag = True
        self.__exit__(None,None,None)

    def config_wave_format(self,Format=None,Width=None,Channels=1,Rate=16000,ChunkFrames=1024):

        assert Channels==1 or Channels==2, "Expected <Channels> is 1 or 2 but got {}.".format(Channels)

        if Format != None:
            assert Format in ['int8','int16','int32'], "Expected <Format> is int8, int16 or int32 but got{}.".format(Format)
            assert Width == None, 'Only one of <Format> and <Width> is expected to be assigned but both two are gotten.'
            if Format == 'int8':
                self.width = 1
            elif Format == 'int16':
                self.width = 2
            else:
                self.width = 4           
        else:
            assert Width != None, 'Expected to assign one value of <Format> aor <Width> but got two None.'
            assert Width in [1,2,4], "Expected <Width> is 1, 2 or 4 but got{}.".format(Width)
            self.width = Width
            if Width == 1:
                self.formats = 'int8'
            elif Width == 2:
                self.formats = 'int16'
            else:
                self.formats = 'int32'
    
        self.formats = Format
        self.channels = Channels
        self.rate = Rate
        self.chunkFrames = ChunkFrames
        self.chunkSize = self.width*Channels*ChunkFrames

    def read(self,wavFile):

        if self.safeFlag == False:
            raise WrongOperation('We only allow user to run speak client by using <with> grammar.')

        if not os.path.isfile(wavFile):
            raise PathError('No such wav file: {}.'.format(wavFile))
        
        if 'read' in self.threadManager.keys() and self.threadManager['read'].is_alive():
            raise WrongOperation('Another read task is running now.')

        if 'record' in self.threadManager.keys() and self.threadManager['record'].is_alive():
            raise WrongOperation('Record and Read are not allowed to run concurrently.')

        def readWave(wavFile,dataQueue):
            try:
                self._counter = 0

                wf = wave.open(wavFile,'rb')
                wfRate = wf.getframerate()
                wfChannels = wf.getnchannels()
                wfWidth = wf.getsampwidth()
                if not wfWidth in [1,2,4]:
                    raise WrongOperation("Only int8, int16 or int32 wav data can be accepted.")
                
                self.config_wave_format(None,wfWidth,wfChannels,wfRate,1024)

                secPerRead = self.chunkFrames/self.rate

                firstMessage = "{},{},{},{}".format(self.formats,self.channels,self.rate,self.chunkFrames)
                firstMessage = firstMessage + " "*(32-len(firstMessage))
                dataQueue.put(firstMessage.encode())

                data = wf.readframes(self.chunkFrames)
                while len(data) == self.chunkSize:
                    self._counter += secPerRead
                    dataQueue.put(data)
                    if True in [self.localErrFlag, self.remoteErrFlag, self.endFlag]:
                        data = b""
                        break
                    time.sleep(secPerRead)
                    data = wf.readframes(self.chunkFrames)
                if data != b"":
                    self._counter += len(data)/self.width/self.channels/self.rate
                    lastChunkData = data + b" "*(self.chunkSize-len(data))
                    dataQueue.put(lastChunkData)
            except Exception as e:
                self.localErrFlag = True
                raise e
            else:
                if True in [self.remoteErrFlag, self.localErrFlag]:
                    pass
                else:
                    dataQueue.put('endFlag')

        self.threadManager['read'] = threading.Thread(target=readWave,args=(wavFile,self.dataQueue,))
        self.threadManager['read'].start()

    def record(self,seconds=None):

        if self.safeFlag == False:
            raise WrongOperation('We only allow user to run speak client by using <with> grammar.')

        if 'record' in self.threadManager.keys() and self.threadManager['record'].is_alive():
            raise WrongOperation('Another record task is running now.')

        if 'read' in self.threadManager.keys() and self.threadManager['read'].is_alive():
            raise WrongOperation('Record and Read are not allowed to run concurrently.')       

        if seconds != None:
            assert isinstance(seconds,(int,float)) and seconds > 0,'Expected <seconds> is positive int or float number.'

        def recordWave(seconds,dataQueue):
            try:
                self._counter = 0

                secPerRecord = self.chunkFrames/self.rate

                firstMessage = "{},{},{},{}".format(self.formats,self.channels,self.rate,self.chunkFrames)
                firstMessage = firstMessage + " "*(32-len(firstMessage))
                dataQueue.put(firstMessage.encode())

                if self.formats == 'int8':
                    ft = pyaudio.paInt8
                elif self.formats == 'int16':
                    ft = pyaudio.paInt16
                else:
                    ft = pyaudio.paInt32
                stream = self.p.open(format=ft,channels=self.channels,rate=self.rate,input=True,output=False)

                if seconds != None:
                    recordLast = True
                    while self._counter <= (seconds-secPerRecord):
                        data = stream.read(self.chunkFrames)
                        dataQueue.put(data)
                        self._counter += secPerRecord
                        if True in [self.localErrFlag, self.endFlag, self.remoteErrFlag]:
                            recordLast = False
                            break
                    if recordLast:
                        lastRecordFrames = int((seconds-self._counter)*self.rate)
                        data = stream.read(lastRecordFrames)
                        data += b" "*(self.chunkSize-len(data))
                        dataQueue.put(data)
                        self._counter = seconds
                else:
                    while True:
                        stream.read(self.chunkFrames)
                        dataQueue.put(data)
                        self._counter += secPerRecord
                        if True in [self.localErrFlag, self.endFlag, self.remoteErrFlag]:
                            break
            except Exception as e:
                self.localErrFlag = True
                raise e
            else:
                if True in [self.localErrFlag, self.remoteErrFlag]:
                    pass
                else:
                    dataQueue.put('endFlag')
            finally:
                stream.stop_stream()
                stream.close()

        self.threadManager['record'] = threading.Thread(target=recordWave,args=(seconds,self.dataQueue,))
        self.threadManager['record'].start()

    def recognize(self,func,args=None,interval=0.3):

        if not self.safeFlag:
            raise WrongOperation('We only allow user to run speak client by using <with> grammar.')
        
        if 'recognize' in self.threadManager.keys() and self.threadManager['recognize'].is_alive():
            raise WrongOperation('Another recognition task is running now.')

        if ('send' in self.threadManager.keys() and self.threadManager['send'].is_alive()) or ('receive' in self.threadManager.keys() and self.threadManager['receive'].is_alive()):
            raise WrongOperation('<local> mode and <remote> mode are not expected to run meanwhile.')

        def recognizeWave(dataQueue,func,args,resultQueue,interval):
            
            class VOD(object):
                def __init__(self):
                    self.lastRe = None
                    self.c = 0
                def __call__(self,re):
                    if re == self.lastRe:
                        self.c  += 1
                        if self.c == 3:
                            self.c = 1
                            return True
                        else:
                            return False
                    self.lastRe = re
                    self.c = 1
                    return False
                    
            vod = VOD()

            dataPerReco = []
            timesPerReco = None
            count = 0

            try:
                while True:
                    if self.localErrFlag == True:
                        break
                    if dataQueue.empty():
                        if ('read' in self.threadManager.keys() and self.threadManager['read'].is_alive()) or ('record' in self.threadManager.keys() and self.threadManager['record'].is_alive()):
                            time.sleep(0.01)
                        else:
                            raise WrongOperation('Excepted data input from Read(file) or Record(microphone).')
                    else:
                        chunkData = dataQueue.get()
                        if timesPerReco == None:
                            # Compute timesPerReco and Throw the first message
                            timesPerReco = math.ceil(self.rate*interval/self.chunkFrames)
                            continue
                        if chunkData == 'endFlag':
                            count = timesPerReco + 1
                        else:
                            dataPerReco.append(chunkData)
                            count += 1
                        if count >= timesPerReco:
                            if len(dataPerReco) > 0:
                                with tempfile.NamedTemporaryFile('w+b',suffix='.wav') as waveFile:
                                    wf = wave.open(waveFile.name, 'wb')
                                    wf.setsampwidth(self.width)
                                    wf.setnchannels(self.channels)
                                    wf.setframerate(self.rate)
                                    wf.writeframes(b''.join(dataPerReco))
                                    wf.close()
                                    if args != None:
                                        result = func(waveFile.name,args)
                                    else:
                                        result = func(waveFile.name)        
                            if count > timesPerReco:
                                resultQueue.put((True,result))
                                break
                            else:
                                sof = vod(result)
                                resultQueue.put((sof,result))
                                if sof:
                                    dataPerReco = []
                                count = 0
            except Exception as e:
                self.localErrFlag = True
                raise e
            else:
                if self.localErrFlag == True:
                    pass
                else:
                    resultQueue.append('endFlag')

        self.threadManager['recognize'] = threading.Thread(target=recognizeWave,args=(self.dataQueue,func,args,self.resultQueue,interval,))
        self.threadManager['record'].start()

    def connect_to(self, proto='TCP', targetHost=None, targetPort=9509, timeout=10):

        if not self.safeFlag:
            raise WrongOperation('Please run with safe mode by using <with> grammar.')

        if self.client != None:
            raise WrongOperation('Another local client is working.')

        assert proto in ['TCP','UDP'], "Expected <proto> is TCP or UDP but got {}.".format(proto)

        self.proto = proto
        self.targetHost = targetHost
        self.targetPort = targetPort

        if timeout != None:
            assert isinstance(time,int),'Expected <timeout> seconds is positive int number but got {}.'.format(timeout)
            socket.setdefaulttimeout(timeout)
        
        if proto == 'TCP':
            try:
                self.client = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
                self.client.connect((targetHost,targetPort))
                verification = self.client.recv(32)
                if verification != b'hello world':
                    raise NetworkError('Connection anomaly.')
            except ConnectionRefusedError:
                raise NetworkError('Target server has not been activated.')
            finally:
                self.client.close()
                self.client = None
                self.localErrFlag = True
        else:
            self.client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.client.sendto(b'hello world',(targetHost,targetPort))
            try:
                i = 0
                while i < 5:
                    verification,addr = self.client.recvfrom(32)
                    if verification == b'hello world' and addr == (targetHost,targetPort):
                        break
                    i += 1
                if i == 5:
                    raise NetworkError('Cannot communicate with target server.')
            except TimeoutError:
                raise NetworkError('Target server seems has not been activated.')
            finally:
                self.client.close()
                self.client = None
                self.localErrFlag = True

        return True

    def send(self):

        if not self.safeFlag:
            raise WrongOperation('Please run with safe mode by using <with> grammar.')
        
        if 'send' in self.threadManager.keys() and self.threadManager['send'].is_alive():
            raise WrongOperation('Another send task is running now.')      

        if self.client == None:
            raise WrongOperation('No activated local client.')
        
        if 'recognize' in self.threadManager.keys() and self.threadManager['recognize'].is_alive():
            raise WrongOperation('<local> mode and <remote> mode are not expected to run meanwhile.')        

        def sendWave(dataQueue):
            try:
                while True:
                    if True in [self.localErrFlag,self.remoteErrFlag]:
                        break
                    if dataQueue.empty():
                        if ('read' in self.threadManager.keys() and self.threadManager['read'].is_alive()) or ('record' in self.threadManager.keys() and self.threadManager['record'].is_alive()):
                            time.sleep(0.01)
                        else:
                            raise WrongOperation('Excepted data input from Read(file) or Record(microphone).')
                    else:
                        message = dataQueue.get()
                        if message == 'endFlag':
                            break
                        elif self.proto == 'TCP':
                            self.client.send(message)
                        else:
                            self.client.sendto(message,(self.targetHost,self.targetPort))
            except Exception as e:
                self.localErrFlag = True
                raise e
            finally:
                if self.remoteErrFlag == True:
                    pass
                else:
                    if self.localErrFlag == True:
                        lastMessage = 'errFlag'.encode()
                    else:
                        lastMessage = 'endFlag'.encode()

                    if self.proto == 'TCP':
                        self.client.send(lastMessage)
                    else:
                        self.client.sendto(lastMessage,(self.targetHost,self.targetPort))
                 
        self.threadManager['send'] = threading.Thread(target=sendWave,args=(self.dataQueue,))
        self.threadManager['send'].start()
    
    def receive(self):

        if not self.safeFlag:
            raise WrongOperation('Please run with safe mode by using <with> grammar.')
        
        if 'receive' in self.threadManager.keys() and self.threadManager['receive'].is_alive():
            raise WrongOperation('Another receive task is running now.')

        if 'recognize' in self.threadManager.keys() and self.threadManager['recognize'].is_alive():
            raise WrongOperation('Local mode and Remote mode are not expected to run meanwhile.')     

        if self.client == None:
            raise WrongOperation('No activated network client.')

        def recvResult(resultQueue):
            try:
                while True:
                    if self.localErrFlag == True:
                        break               
                    if self.proto == 'TCP':
                        message = self.client.recv(self.chunkSize)
                    else:
                        i = 0
                        while i < 5:
                            message, addr = self.client.recvfrom(self.chunkSize)
                            if addr == (self.targetHost,self.targetPort):
                                break
                        if i == 5:
                            raise NetworkError('Communication between local client and remote server worked anomaly.')
                    message = message.decode.strip()
                    if message in ['endFlag','errFlag']:
                        break
            except TimeoutError:
                raise NetworkError('Please ensure server has been activated.')
            except Exception as e:
                self.localErrFlag = True
                raise e
            else:
                if message == 'errFlag':
                    self.remoteErrFlag == True
                else:
                    resultQueue.put('endFlag')

        self.threadManager['receive'] = threading.Thread(target=recvResult,args=(self.resultQueue,))
        self.threadManager['receive'].start()

    def get(self):
        while True:
            if self.resultQueue.empty():
                if ('recognize' in self.threadManager.keys() and self.threadManager['recognize'].is_alive()) or ('receive' in self.threadManager.keys() and self.threadManager['receive'].is_alive()):
                    time.sleep(0.01)
                else:
                    return None
            else:
                message = self.resultQueue.get()
                if message == 'endFlag':
                    return None
                else:
                    return message

    @property
    def timer(self):
        return self._counter

class RemoteServer(object):

    def __init__(self, proto='TCP', bindHost=None, bindPort=9509):

        self.client = None
        self.threadManager = {}

        self.dataQueue = queue.Queue()
        self.resultQueue = queue.Queue()

        self.localErrFlag = False
        self.remoteErrFlag = False
        self.safeFlag = False

        socket.setdefaulttimeout(20)
        
        if bindHost != None:
            self.bind(proto,bindHost,bindPort)
    
    def __enter__(self):
        self.safeFlag = True
        return self
    
    def __exit__(self,errType,errValue,errTrace):
        self.wait()
        time.sleep(1)
        if self.client != None:
            self.client
        

    def wait(self):
        for name,thread in self.threadManager.items():
            if thread.is_alive():
                thread.join()

    def _config_wave_format(self,Format=None,Width=None,Channels=1,Rate=16000,ChunkFrames=1024):

        assert Channels==1 or Channels==2,"Expected <Channels> is 1 or 2 but got {}.".format(Channels)

        if Format != None:
            assert Format in ['int8','int16','int32'], "Expected <Format> is int8, int16 or int32 but got{}.".format(Format)
            assert Width == None, 'Only one of <Format> and <Width> is expected to assigned but both two.'
            if Format == 'int8':
                self.width = 1
            elif Format == 'int16':
                self.width = 2
            else:
                self.width = 4           
        else:
            assert Width != None, 'Expected one value in <Format> and <Width> but got two None.'
            assert Width in [1,2,4], "Expected <Width> is 1, 2 or 4 but got{}.".format(Format)
            self.width = Width
            if Width == 1:
                self.formats = 'int8'
            elif Width == 2:
                self.formats = 'int16'
            else:
                self.formats = 'int32'
    
        self.formats = Format
        self.channels = Channels
        self.rate = Rate
        self.chunkFrames = ChunkFrames
        self.chunkSize = self.width*Channels*ChunkFrames
    
    def bind(self, proto='TCP', bindHost=None, bindPort=9509):

        assert proto in ['TCP','UDP'],'Expected proto TCP or UDP but got {}'.format(proto)

        if self.bindHost != None:
            raise WrongOperation('Server has already bound to {}.'.format((self.bindHost,self.bindPort)))
        
        assert host != None, 'Expected <host> is not None.'

        if proto == 'TCP':
            self.server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        else:
            self.server = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

        self.server.bind((bindHost,bindHort))

        self.proto = proto
        self.bindHost = bindHost
        self.bindPort = bindPort

    def connect_from(self,targetHost):

        if not self.safeFlag:
            raise WrongOperation('Please run with safe mode by using <with> grammar.')

        if self.bindHost == None:
            raise WrongOperation('Bind Host IP and Port firstly by using .bind() method.')

        if self.client != None:
            raise WrongOperation('Another connection is running right now.')

        try:
            i = 0
            while i < 5:
                if self.proto == 'TCP':
                    self.listen(1)
                    self.client, addr = self.server.accept()
                    if addr[0] == targetHost:
                        self.client.send(b'hello world')
                        self.targetAddr = addr
                        break
                    else:
                        self.client.close()
                else:
                    vertification,addr = self.server.recvfrom(32)
                    if vertification == b'hello world' and addr[0] == targetHost:
                        self.client = self.server
                        self.client.sendto(b'hello world',addr)
                        self.targetAddr = addr
                        break
                i += 1
        except Exception as e:
            self.localErrFlag = True
            raise e
        except socket.timeout:
            raise NetworkError('No connect-application from any remote client.')
        else:
            if i >= 5:
                self.client = None
                self.localErrFlag = True
                raise NetworkError("Cannot connect from {}".format(targetHost))
            else:
                return True

    def receive(self):

        if not self.safeFlag:
            raise WrongOperation('Please run with safe mode by using <with> grammar.')

        if 'receive' in self.threadManager.keys() and self.threadManager['receive'].is_alive():
            raise WrongOperation('Another receive thread is running now')
        
        if self.client == None:
            raise WrongOperation('Did not connect from any remote client.')

        def recvWave(dataQueue):
            try:
                if self.proto == 'TCP':
                    vertification = self.client.recv(32)
                else:
                    i = 0
                    while i < 5:
                        vertification,addr = self.client.recvfrom(32)
                        if addr == self.targetAddr:
                            break                            
                    if i == 5:
                        raise NetworkError('Communicate between server and remote client worked anomaly.')
                vertification = vertification.decode().strip().split(',')
                self._config_wave_format(idMe[0],None,int(idMe[1]),int(idMe[2]),int(idMe[3]))
            except Exception as e:
                self.localErrFlag = True
                raise e                    
            except socket.timeout:
                raise NetworkError('Did not received any data from remote client.')
            else:
                while True:
                    if self.localErrFlag == True:
                        break
                    try:
                        if self.proto == 'TCP':
                            message = self.client.recv(self.chunkSize)
                        else:
                            i = 0
                            while i < 5:
                                vertification,addr = self.client.recvfrom(32)
                                if addr == self.targetAddr:
                                    break                            
                            if i == 5:
                                raise NetworkError('Communicate between server and remote client worked anomaly.')
                    except Exception as e:
                        self.localErrFlag = True
                        raise e 
                    except socket.timeout:
                        raise NetworkError('Did not received any data from remote client.')
                    else:
                        if message == b'errFlag':
                            self.remoteErrFlag = True
                            break
                        elif message == b'endFlag':
                            dataQueue.put('endFlag')
                            break
                        else:
                            dataQueue.put(message)
            
        self.threadManager['receive'] = threading.Thread(target=recvWave,args=(self.dataQueue,))
        self.threadManager['receive'].start()

    def recognize(self,func,args=None,interval=0.3):

        if not self.safeFlag:
            raise WrongOperation('We only allow user to run speak client by using <with> grammar.')
        
        if 'recognize' in self.threadManager.keys() and self.threadManager['recognize'].is_alive():
            raise WrongOperation('Another recognize task is running now.')

        def recognizeWave(dataQueue,func,args,resultQueue,interval):

            class VOD(object):
                def __init__(self):
                    self.lastRe = None
                    self.c = 0
                def __call__(self,re):
                    if re == self.lastRe:
                        self.c  += 1
                        if self.c == 3:
                            self.c = 1
                            return True
                        else:
                            return False
                    self.lastRe = re
                    self.c = 1
                    return False
                    
            vod = VOD()

            dataPerReco = []
            timesPerReco = None
            count = 0

            try:
                while True:
                    if True in [self.localErrFlag,self.remoteErrFlag]:
                        break
                    elif dataQueue.empty():
                        if 'receive' in self.threadManager.keys() and self.threadManager['receive'].is_alive():
                            time.sleep(0.01)
                        else:
                            raise WrongOperation('Excepted data input from Receive().')
                    else:
                        chunkData = dataQueue.get()
                        if timesPerReco == None:
                            # Compute timesPerReco and Throw the first message
                            timesPerReco = math.ceil(self.rate*interval/self.chunkFrames)
                            continue
                        if chunkData == 'endFlag':
                            count = timesPerReco + 1
                        else:
                            dataPerReco.append(chunkData)
                            count += 1

                        if count >= timesPerReco:
                            if len(dataPerReco) > 0:
                                with tempfile.NamedTemporaryFile('w+b',suffix='.wav') as waveFile:
                                    wf = wave.open(waveFile.name, 'wb')
                                    wf.setsampwidth(self.width)
                                    wf.setnchannels(self.channels)
                                    wf.setframerate(self.rate)
                                    wf.writeframes(b''.join(dataPerReco))
                                    wf.close()
                                    if args != None:
                                        result = func(waveFile.name,args)
                                    else:
                                        result = func(waveFile.name)        
                            if count > timesPerReco:
                                resultQueue.put((True,result))
                                break
                            else:
                                sof = vod(result)
                                resultQueue.put((sof,result))
                                if sof:
                                    dataPerReco = []
                                count = 0
            except Exception as e:
                self.localErrFlag = True
                raise e
            else:
                if self.localErrFlag == True or self.remoteErrFlag == True:
                    pass
                else:
                    resultQueue.append('endFlag')

        self.threadManager['recognize'] = threading.Thread(target=recognizeWave,args=(self.dataQueue,func,args,self.resultQueue,interval,))
        self.threadManager['record'].start()

    def send(self):

        if not self.safeFlag:
            raise WrongOperation('Please run under safe state by using <with> grammar')
            #print('Running without safe mode. Please use exit() function finally to ensure tasks safely')

        if 'send' in self.threadManager.keys() and self.threadManager['send'].is_alive():
            raise WrongOperation('Another send thread is running now')

        if self.client == None:
            raise WrongOperation('No activated server client')

        if resultQueue == None:
            resultQueue = self.resultQueue

        def sendResult(resultQueue):
            try:
                while True:
                    if True in [self.localErrFlag,self.remoteErrFlag]:
                        break
                    if resultQueue.empty():
                        if ('receive' in self.threadManager.keys() and self.threadManager['receive'].is_alive()) or ('recognize' in self.threadManager.keys() and self.threadManager['recognize'].is_alive()):
                            time.sleep(0.01)
                        else:
                            raise WrongOperation('Excepted production from Receive() and Recognize() task.')
                    else:
                        message = resultQueue.get()
                        if 'endFlag' == message:
                            break
                        else:
                            #message has a format: (sectionOverFlag,result)
                            data = []
                            #we will send data with a format: sectionOverFlag(Y/N/T) + ' ' + resultData + placeHolderSymbol(' ')
                            rSize = self.chunkSize - 2  # -len("Y ")
                            #result is likely longer than rSize, so cut it
                            #Get the N front rSize part, give them all 'Y' sign   
                            for i in range(len(message[1])//rSize):
                                data.append('T ' + message[1][i*rSize:(i+1)*rSize])
                            if len(message[1][i*rSize:]) > 0:
                                lastData = ''
                                if message[0]:
                                    lastData = 'Y ' + message[1][i*rSize:]
                                else:
                                    lastData = 'N ' + message[1][i*rSize:]
                                lastData += " "*(self.chunkSize-len(lastData))
                                data.append(lastData)
                            else:
                                if message[0]:
                                    data[-1] = 'Y ' + data[-1][2:]
                                else:
                                    data[-1] = 'N ' + data[-1][2:]                           

                            data = ''.join(data)
    
                            if self.proto == 'TCP':
                                self.client.send(data.encode())
                            else:
                                self.client.sendto(data.encode(),self.targetAddr)
            except Exception as e:
                self.localErrFlag = True
                raise e
            finally:
                if self.remoteErrFlag == True:
                    pass
                elif self.localErrFlag == True:
                    if self.proto == 'TCP':
                        self.client.send('errFlag'.encode())
                    else:
                        self.client.sendto('errFlag'.encode(),self.targetAddr)
                else:
                    if self.proto == 'TCP':
                        self.client.send('endFlag'.encode())
                    else:
                        self.client.sendto('endFlag'.encode(),self.targetAddr)
            
        self.threadManager['send'] = threading.Thread(target=sendResult,args=(self.resultQueue,))
        self.threadManager['send'].start()                

    def get(self):
        while True:
            if self.resultQueue.empty():
                if ('recognize' in self.threadManager.keys() and self.threadManager['recognize'].is_alive()) or ('receive' in self.threadManager.keys() and self.threadManager['receive'].is_alive()):
                    time.sleep(0.01)
                else:
                    return None
            else:
                message = self.resultQueue.get()
                if message == 'endFlag':
                    return None
                else:
                    return message

# ---------- Basic Class Functions ------- 

def save(data,fileName,chunks=1):
    '''
    Useage:  exkaldi.save(obj, 'data.ark')
    
    The same as KaldiArk().save() or KaldiDict().save() method.

    '''  
    if isinstance(data,(KaldiDict,KaldiArk)):
        data.save(fileName,chunks)
    else:
        raise UnsupportedDataType("Expected KaldiDict or KaldiArk data but got {}.".format(type(data)))

def concat(datas,axis):
    '''
    Useage:  newObj = exkaldi.concat([obj1,obj2],axis=1)
    
    The same as KaldiArk().concat() or KaldiDict().concat() fucntion.

    '''  
    assert isinstance(datas,list), "Expected <datas> is a list object but got {}.".format(type(datas))

    if len(datas) > 0:
        if not isinstance(datas[0],(KaldiArk,KaldiDict)):
            raise UnsupportedDataType('Expected KaldiDict or KaldiArk object but got {}.'.format(datas[0]))
        if len(datas) > 1:
            return datas[0].concat(datas[1:],axis)
        else:
            return datas[0]
    else:
        return None

def cut(data,maxFrames):
    '''
    Useage:  exkaldi.cut(100)
    
    Whatever <data> is KaldiArk or KaldiDict object, return KaldiDict object.
    
    The same as KaldiDict().cut() fucntion.

    '''  
    if not isinstance(data,(KaldiArk,KaldiDict)):
        raise UnsupportedDataType('Expected KaldiDict or KaldiArk object but got {}.'.format(data))
    elif isinstance(data,KaldiArk):
        data = data.array
    
    return data.cut(maxFrames)

def normalize(data,std=True,alpha=1.0,beta=0.0,epsilon=1e-6,axis=0):
    '''
    Useage:  newObj = exkaldi.normalize(obj)
    
    Whatever <data> is KaldiArk or KaldiDict object, return KaldiDict object.
    
    The same as KaldiDict().normalize() fucntion.

    '''     
    if not isinstance(data,(KaldiArk,KaldiDict)):
        raise UnsupportedDataType('Expected KaldiDict or KaldiArk object but got {}.'.format(data))
    elif isinstance(data,KaldiArk):
        data = data.array

    return data.normalize(std,alpha,beta,epsilon,axis)

def subset(data,nHead=0,chunks=1,uttList=None):
    '''
    Useage:  newObj = exkaldi.subset(chunks=5)
    
    The same as KaldiArk().subset() or KaldiDict().subset() fucntion.

    '''     
    if isinstance(data,(KaldiDict,KaldiArk)):
        data.subset(nHead,chunks,uttList)
    else:
        raise UnsupportedDataType("Expected KaldiDict or KaldiArk data but got {}.".format(type(data)))   

def merge(data,keepDim=False,sort=False):
    '''
    Useage:  newObj = exkaldi.merge(obj)
    
    Whatever <data> is KaldiArk or KaldiDict object, return KaldiDict object.
    
    The same as KaldiDict().merge() fucntion.

    '''   
    if not isinstance(data,(KaldiArk,KaldiDict)):
        raise UnsupportedDataType('Expected KaldiDict or KaldiArk object but got {}.'.format(datas[0]))
    elif isinstance(data,KaldiArk):
        data = data.array
    return data.merge(keepDim,sort)

def remerge(matrix,uttLens):
    '''
    Useage:  newObj = exkaldi.remerge(obj)
    
    Whatever <data> is KaldiArk or KaldiDict object, return KaldiDict object.
    
    The same as KaldiDict().remerge() fucntion.

    '''   
    newData = KaldiDict()
    newData.remerge(matrix,uttLens)
    return newData

def sort(data,by='frame',reverse=False):
    '''
    Useage:  newObj = exkaldi.sort(obj)
    
    Whatever <data> is KaldiArk or KaldiDict object, return KaldiDict object.
    
    The same as KaldiDict().sort() fucntion.

    '''   
    if not isinstance(data,(KaldiArk,KaldiDict)):
        raise UnsupportedDataType('Expected KaldiDict or KaldiArk object but got {}.'.format(datas[0]))
    elif isinstance(data,KaldiArk):
        data = data.array

    return data.sort(by,reverse)

def select(data, dims,reserve=False):
    '''
    Useage:  newObj = exkaldi.select(chunks=5)
    
    The same as KaldiArk().select() or KaldiDict().select() fucntion.

    '''    
    if isinstance(data,(KaldiDict,KaldiArk)):
        data.select(data, dims,reserve)
    else:
        raise UnsupportedDataType("Expected KaldiDict or KaldiArk data but got {}.".format(type(data)))    

def splice(data,left=4,right=None):
    '''
    Useage:  newObj = exkaldi.splice(chunks=5)
    
    The same as KaldiArk().splice() or KaldiDict().splice() fucntion.

    '''      
    if isinstance(data,(KaldiDict,KaldiArk)):
        data.splice(data,left,right)
    else:
        raise UnsupportedDataType("Expected KaldiDict or KaldiArk data but got {}.".format(type(data)))     

def to_dtype(data,dtype):
    '''
    Useage:  newObj = exkaldi.to_dtype(‘float64’)
    
    The same as KaldiArk().to_dtype() or KaldiDict().to_dtype() fucntion.

    '''      
    if isinstance(data,(KaldiDict,KaldiArk)):
        data.to_dtype(data,dtype)
    else:
        raise UnsupportedDataType("Expected KaldiDict or KaldiArk data but got {}.".format(type(data)))     

# ---------- Feature and Label Process Fucntions -----------

def compute_mfcc(wavFile,rate=16000,frameWidth=25,frameShift=10,melBins=23,featDim=13,windowType='povey',useUtt='MAIN',useSuffix=None,config=None,outFile=None):
    '''
    Useage:  obj = compute_mfcc("test.wav") or compute_mfcc("test.scp")

    Compute mfcc feature. Return KaldiArk object or file path if <outFile> is not None.
    We provide some usual options, but if you want use more, set < config > = your-configure. Note that if you do this, these usual configures we provided will be ignored.
    You can use pythonkaldi.check_config('compute_mfcc') function to get configure information you could set.
    Also run shell command "compute-mfcc-feats" to look their meaning.

    '''  
    kaldiTool = 'compute-mfcc-feats'

    if config != None:
        
        if check_config(name='compute_mfcc',config=config):
            for key in config.keys():
                kaldiTool = ' {}={}'.format(key,config[key])

    else:
        kaldiTool += ' --allow-downsample=true'
        kaldiTool += ' --allow-upsample=true'
        kaldiTool += ' --sample-frequency={}'.format(rate)
        kaldiTool += ' --frame-length={}'.format(frameWidth)
        kaldiTool += ' --frame-shift={}'.format(frameShift)
        kaldiTool += ' --num-mel-bins={}'.format(melBins)
        kaldiTool += ' --num-ceps={}'.format(featDim)
        kaldiTool += ' --window-type={}'.format(windowType)

    if useSuffix == None:
        useSuffix = ''
    elif isinstance(useSuffix,str):
        pass
    else:
        raise WrongOperation('Wrong suffix type.')

    if outFile != None:

        if not outFile.endswith('.ark'):
            outFile += '.ark'
        
        if wavFile.endswith('wav') or "wav" in useSuffix:
            cmd = 'echo {} {} | {} scp:- ark:{}'.format(useUtt,wavFile,kaldiTool,outFile)
        elif wavFile.endswith('scp') or "scp" in useSuffix:
            cmd = '{} scp:{} ark:{}'.format(kaldiTool,wavFile,outFile)
        else:
            raise UnsupportedDataType('Unknown file suffix. You can declare it by using <useSuffix>="wav" or "scp".')

        p = subprocess.Popen(cmd,shell=True,stderr=subprocess.PIPE)
        (out,err) = p.communicate()
        if not os.path.isfile(outFile):
            err = err.decode()
            print(err)
            raise KaldiProcessError('Compute mfcc defeated.')
        else:
            return outFile
    else:

        if wavFile.endswith('wav') or "wav" in useSuffix:
            cmd = 'echo {} {} | {} scp:- ark:-'.format(useUtt,wavFile,kaldiTool)
        elif wavFile.endswith('scp') or "scp" in useSuffix:
            cmd = '{} scp:{} ark:-'.format(kaldiTool,wavFile)
        else:
            raise UnsupportedDataType('Unknown file suffix. You can declare it by using <useSuffix>="wav" or "scp".')

        p = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        (out,err) = p.communicate()   

        if out == b'':
            err = err.decode()
            print(err)
            raise KaldiProcessError('Compute mfcc defeated.')
        else:
            return KaldiArk(out)

def compute_fbank(wavFile,rate=16000,frameWidth=25,frameShift=10,melBins=23,windowType='povey',useUtt='MAIN',useSuffix=None,config=None,outFile=None):
    '''
    Useage:  obj = compute_fbank("test.wav") or compute_mfcc("test.scp")

    Compute fbank feature. Return KaldiArk object or file path if <outFile> is not None.
    We provide some usual options, but if you want use more, set < config > = your-configure. Note that if you do this, these usual configures we provided will be ignored.
    You can use pythonkaldi.check_config('compute_fbank') function to get configure information you could set.
    Also run shell command "compute-fbank-feats" to look their meaning.

    '''  
    kaldiTool = 'compute-fbank-feats'

    if config != None:
        
        if check_config(name='compute_fbank',config=config):
            for key in config.keys():
                kaldiTool = ' {}={}'.format(key,config[key])

    else:
        kaldiTool += ' --allow-downsample=true'
        kaldiTool += ' --allow-upsample=true'
        kaldiTool += ' --sample-frequency={}'.format(rate)
        kaldiTool += ' --frame-length={}'.format(frameWidth)
        kaldiTool += ' --frame-shift={}'.format(frameShift)
        kaldiTool += ' --num-mel-bins={}'.format(melBins)
        kaldiTool += ' --window-type={}'.format(windowType)

    if useSuffix == None:
        useSuffix = ''
    elif isinstance(useSuffix,str):
        pass
    else:
        raise WrongOperation('Wrong suffix type.')

    if outFile != none:

        if not outFile.endswith('.ark'):
            outFile += '.ark'
        
        if wavFile.endswith('wav') or "wav" in useSuffix:
            cmd = 'echo {} {} | {} scp:- ark:{}'.format(useUtt,wavFile,kaldiTool,outFile)
        elif wavFile.endswith('scp') or "scp" in useSuffix:
            cmd = '{} scp:{} ark:{}'.format(kaldiTool,wavFile,outFile)
        else:
            raise UnsupportedDataType('Unknown file suffix. You can declare it by using <useSuffix>="wav" or "scp".')

        p = subprocess.Popen(cmd,shell=True,stderr=subprocess.PIPE)
        (out,err) = p.communicate()
        if not os.path.isfile(outFile):
            err = err.decode()
            print(err)
            raise KaldiProcessError('Compute fbank defeated.')
        else:
            return outFile
    else:

        if wavFile.endswith('wav') or "wav" in useSuffix:
            cmd = 'echo {} {} | {} scp:- ark:-'.format(useUtt,wavFile,kaldiTool)
        elif wavFile.endswith('scp') or "scp" in useSuffix:
            cmd = '{} scp:{} ark:-'.format(kaldiTool,wavFile)
        else:
            raise UnsupportedDataType('Unknown file suffix. You can declare it by using <useSuffix>="wav" or "scp".')

        p = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        (out,err) = p.communicate()   

        if out == b'':
            err = err.decode()
            print(err)
            raise KaldiProcessError('Compute fbank defeated.')
        else:
            return KaldiArk(out)

def compute_plp(wavFile,rate=16000,frameWidth=25,frameShift=10,melBins=23,featDim=13,windowType='povey',useUtt='MAIN',useSuffix=None,config=None,outFile=None):
    '''
    Useage:  obj = compute_plp("test.wav") or compute_mfcc("test.lst",useSuffix='scp')

    Compute plp feature. Return KaldiArk object or file path if <outFile> is not None.
    We provide some usual options, but if you want use more, set < config > = your-configure. Note that if you do this, these usual configures we provided will be ignored.
    You can use pythonkaldi.check_config('compute_plp') function to get configure information you could set.
    Also run shell command "compute-plp-feats" to look their meaning.

    '''  
    kaldiTool = 'compute-plp-feats'

    if config != None:
        
        if check_config(name='compute_plp',config=config):
            for key in config.keys():
                kaldiTool = ' {}={}'.format(key,config[key])

    else:
        kaldiTool += ' --allow-downsample=true'
        kaldiTool += ' --allow-upsample=true'
        kaldiTool += ' --sample-frequency={}'.format(rate)
        kaldiTool += ' --frame-length={}'.format(frameWidth)
        kaldiTool += ' --frame-shift={}'.format(frameShift)
        kaldiTool += ' --num-mel-bins={}'.format(melBins)
        kaldiTool += ' --num-ceps={}'.format(featDim)
        kaldiTool += ' --window-type={}'.format(windowType)

    if useSuffix == None:
        useSuffix = ''
    elif isinstance(useSuffix,str):
        pass
    else:
        raise WrongOperation('Wrong suffix type.')

    if outFile != None:

        if not outFile.endswith('.ark'):
            outFile += '.ark'
        
        if wavFile.endswith('wav') or "wav" in useSuffix:
            cmd = 'echo {} {} | {} scp:- ark:{}'.format(useUtt,wavFile,kaldiTool,outFile)
        elif wavFile.endswith('scp') or "scp" in useSuffix:
            cmd = '{} scp:{} ark:{}'.format(kaldiTool,wavFile,outFile)
        else:
            raise UnsupportedDataType('Unknown file suffix. You can declare it by using <useSuffix>="wav" or "scp".')

        p = subprocess.Popen(cmd,shell=True,stderr=subprocess.PIPE)
        (out,err) = p.communicate()
        if not os.path.isfile(outFile):
            err = err.decode()
            print(err)
            raise KaldiProcessError('Compute plp defeated.')
        else:
            return outFile
    else:

        if wavFile.endswith('wav') or "wav" in useSuffix:
            cmd = 'echo {} {} | {} scp:- ark:-'.format(useUtt,wavFile,kaldiTool)
        elif wavFile.endswith('scp') or "scp" in useSuffix:
            cmd = '{} scp:{} ark:-'.format(kaldiTool,wavFile)
        else:
            raise UnsupportedDataType('Unknown file suffix. You can declare it by using <useSuffix>="wav" or "scp".')

        p = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        (out,err) = p.communicate()   

        if out == b'':
            err = err.decode()
            print(err)
            raise KaldiProcessError('Compute plp defeated.')
        else:
            return KaldiArk(out)

def compute_spectrogram(wavFile,rate=16000,frameWidth=25,frameShift=10,windowType='povey',useUtt='MAIN',useSuffix=None,config=None,outFile=None):
    '''
    Useage:  obj = compute_spetrogram("test.wav") or compute_mfcc("test.lst",useSuffix='scp')

    Compute spectrogram feature. Return KaldiArk object or file path if <outFile> is not None.
    We provide some usual options, but if you want use more, set < config > = your-configure. Note that if you do this, these usual configures we provided will be ignored.
    You can use pythonkaldi.check_config('compute_spetrogram') function to get configure information you could set.
    Also run shell command "compute-spetrogram-feats" to look their meaning.

    ''' 
    kaldiTool = 'compute-spectrogram-feats'

    if config != None:
        
        if check_config(name='compute_spetrogram',config=config):
            for key in config.keys():
                kaldiTool = ' {}={}'.format(key,config[key])

    else:
        kaldiTool += ' --allow-downsample=true'
        kaldiTool += ' --allow-upsample=true'
        kaldiTool += ' --sample-frequency={}'.format(rate)
        kaldiTool += ' --frame-length={}'.format(frameWidth)
        kaldiTool += ' --frame-shift={}'.format(frameShift)
        kaldiTool += ' --window-type={}'.format(windowType)

    if useSuffix == None:
        useSuffix = ''
    elif isinstance(useSuffix,str):
        pass
    else:
        raise WrongOperation('Wrong suffix type.')

    if outFile != None:

        if not outFile.endswith('.ark'):
            outFile += '.ark'
        
        if wavFile.endswith('wav') or "wav" in useSuffix:
            cmd = 'echo {} {} | {} scp:- ark:{}'.format(useUtt,wavFile,kaldiTool,outFile)
        elif wavFile.endswith('scp') or "scp" in useSuffix:
            cmd = '{} scp:{} ark:{}'.format(kaldiTool,wavFile,outFile)
        else:
            raise UnsupportedDataType('Unknown file suffix. You can declare it by using <useSuffix>="wav" or "scp".')

        p = subprocess.Popen(cmd,shell=True,stderr=subprocess.PIPE)
        (out,err) = p.communicate()
        if not os.path.isfile(outFile):
            err = err.decode()
            print(err)
            raise KaldiProcessError('Compute spectrogram defeated.')
        else:
            return outFile
    else:

        if wavFile.endswith('wav') or "wav" in useSuffix:
            cmd = 'echo {} {} | {} scp:- ark:-'.format(useUtt,wavFile,kaldiTool)
        elif wavFile.endswith('scp') or "scp" in useSuffix:
            cmd = '{} scp:{} ark:-'.format(kaldiTool,wavFile)
        else:
            raise UnsupportedDataType('Unknown file suffix. You can declare it by using <useSuffix>="wav" or "scp".')

        p = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        (out,err) = p.communicate()   

        if out == b'':
            err = err.decode()
            print(err)
            raise KaldiProcessError('Compute spectrogram defeated.')
        else:
            return KaldiArk(out)

def use_cmvn(feat,cmvnStatFile=None,utt2spkFile=None,spk2uttFile=None,outFile=None):
    '''
    Useage:  obj = use_cmvn(feat) or obj = use_cmvn(feat,cmvnStatFile,utt2spkFile) or obj = use_cmvn(feat,utt2spkFile,spk2uttFile)

    Apply CMVN to feature. Return KaldiArk object or file path if <outFile> is true. 
    If < cmvnStatFile >  are None, first compute the CMVN state. But <utt2spkFile> and <spk2uttFile> are expected given at the same time if they were not None.

    ''' 
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
            p1 = subprocess.Popen(cmd1,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
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

            p2 = subprocess.Popen(cmd2,shell=True,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
            (out2,err2) = p2.communicate(input=feat)

            if not os.path.isfile(outFile):
                err2 = err2.decode()
                print(err2)
                raise KaldiProcessError('Use cmvn defeated.')
            else:
                return outFile
        else:
            if utt2spkFile != None:
                cmd3 = 'apply-cmvn --utt2spk=ark:{} {} ark:- ark:-'.format(utt2spkFile,cmvnStatFileOption)
            else:
                cmd3 = 'apply-cmvn {} ark:- ark:-'.format(cmvnStatFileOption)

            p3 = subprocess.Popen(cmd3,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
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
    Useage:  obj = compute_cmvn_stats(feat,'train_cmvn.ark') or obj = compute_cmvn_stats(feat,'train_cmvn.ark','train/spk2utt')

    Compute CMVN state and save it as file. Return cmvn file path. 

    ''' 
    if isinstance(feat,KaldiArk):
        pass
    elif isinstance(feat,KaldiDict):
        feat = feat.ark
    else:
        raise UnsupportedDataType("Expected KaldiArk KaldiDict but got {}.".format(type(feat)))
    
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

    p = subprocess.Popen(cmd,shell=True,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    (_,err) = p.communicate(input=feat)

    if not os.path.isfile(outFile):
        err = err.decode()
        print(err)
        raise KaldiProcessError('Compute cmvn stats defeated.')
    else:
        return outFile    

def use_cmvn_sliding(feat,windowsSize=None,std=False):
    '''
    Useage:  obj = use_cmvn_sliding(feat) or obj = use_cmvn_sliding(feat,windows=200)

    Apply sliding CMVN to feature. Return KaldiArk object. If <windowsSize> is None, the window width will be set larger than frames of <feat>.
    If < std > is False, only apply mean, or apply both mean and std.

    ''' 
    if isinstance(feat,KaldiArk):
        pass
    elif isinstance(feat,KaldiDict):
        feat = feat.ark
    else:
        raise UnsupportedDataType("Expected KaldiArk KaldiDict but got {}.".format(type(feat)))
            
    if windowsSize==None:
        featLen = feat.lens[1][0]
        windowsSize = math.ceil(featLen/100)*100
    else:
        assert isinstance(windowsSize,int), "Expected windows size is int."

    if std==True:
        std='true'
    else:
        std='false'

    cmd = 'apply-cmvn-sliding --cmn-window={} --min-cmn-window=100 --norm-vars={} ark:- ark:-'.format(windowsSize,std)
    p = subprocess.Popen(cmd,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    (out,err) = p.communicate(input=feat)
    if out == b'':
        err = err.decode()
        print(err)
        raise KaldiProcessError('Use sliding cmvn defeated.')
    else:
        return KaldiArk(out)  

def add_delta(feat,order=2,outFile=None):
    '''
    Useage:  newObj = add_delta(feat)

    Add n-orders delta to feature. Return KaldiArk object or file path if <outFile> is True.

    ''' 
    if isinstance(feat,KaldiArk):
        pass
    elif isinstance(feat,KaldiDict):
        feat = feat.ark
    else:
        raise UnsupportedDataType("Expected KaldiArk KaldiDict but got {}.".format(type(feat)))

    if outFile != None:

        if not outFile.endswith('.ark'):
            outFile += '.ark'
            
        cmd1 = 'add-deltas --delta-order={} ark:- ark:{}'.format(order,outFile)

        p1 = subprocess.Popen(cmd1,shell=True,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
        (out,err) = p1.communicate(input=feat)

        if not os.path.isfile(outFile):
            err = err.decode()
            print(err)
            raise KaldiProcessError('Add delta defeated.')
        else:
            return outFile 
    else:
        cmd2 = 'add-deltas --delta-order={} ark:- ark:-'.format(order)
        p2 = subprocess.Popen(cmd2,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        (out,err) = p2.communicate(input=feat)
        if out == b'':
            err = err.decode()
            print(err)
            raise KaldiProcessError('Add delta defeated.')
        else:
            return KaldiArk(out)      

def get_ali(aliFile,hmm=None,returnPhone=False):
    '''
    Useage:  obj = get_ali('ali.1.gz','graph/final.mdl') or obj = get_ali('ali.1.gz',returnPhone=True)

    Get alignment from ali file. Return a KaldiDict object. If <returnPhone> is True, return phone id, or return pdf id.
    If <hmm> is None, find the final.mdl automaticlly at the same path with <aliFile>
    ''' 
    if isinstance(aliFile,str):
        if aliFile.endswith('.gz'):
            pass
        elif os.path.isdir(aliFile):
            if aliFile.endswith('/'):
                aliFile = aliFile[:-1]
            aliFile += '/*.gz'
        else:
            raise PathError('{}: No such file or directory.'.format(aliFile))
    else:
        raise UnsupportedDataType("Expected gzip file or folder but got {}.".format(type(aliFile)))

    if hmm == None:
        i = aliFile.rfind('/')
        if i > 0:
            hmm = aliFile[0:i] +'/final.mdl'
        else:
            hmm = './final.mdl'
        if not os.path.isfile(hmm):
            raise PathError("HMM file was not found. Please assign it by <hmm>")        
    elif isinstance(hmm,str):
        if not os.path.isfile(hmm):
            raise PathError("No such file:{}".format(hmm))
    else:
        raise UnsupportedDataType("Expected <hmm> is a path-like string but got {}.".format(type(hmm)))

    if returnPhone:
        cmd = 'gunzip -c {} | ali-to-phones --per-frame=true {} ark:- ark,t:-'.format(aliFile,hmm)
    else:
        cmd = 'gunzip -c {} | ali-to-pdf {} ark:- ark,t:-'.format(aliFile,hmm)

    p = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    (out,err) = p.communicate()

    if out == b'':
        err = err.decode()
        print(err)
        raise KaldiProcessError('Get ali data defeated.')
    else:
        ali_dict = {}
        sp = BytesIO(out)
        for line in sp.readlines():
            line = line.decode()
            line = line.strip().split()
            utt = line[0]
            matrix = np.array(line[1:],dtype=np.int32)
            ali_dict[utt] = matrix
        return KaldiDict(ali_dict)

def analyze_counts(aliFile,outFile,countPhone=False,hmm=None,dim=None):
    '''
    Useage:  obj = analyze_counts(aliFile,outFile)

    Compute label counts in order to normalize acoustic model posterior probability.
    We defaultly compute pdf counts but if <countPhone> is True, compute phone counts.   
    For more help information, look kaldi <analyze-counts> command.

    ''' 
    if hmm == None:
        i = aliFile.rfind('/')
        if i > 0:
            hmm = aliFile[0:i] +'/final.mdl'
        else:
            hmm = './final.mdl'
        if not os.path.isfile(hmm):
            raise WrongOperation('Did not find hmm model file. Please assign it.')
    elif not os.path.isfile(hmm):
        raise PathError('No such file:{}.'.format(hmm))
    
    if dim == None:
        if countPhone:
            cmd = 'hmm-info {} | grep -i "phones"'.format(hmm)
        else:
            cmd = 'hmm-info {} | grep -i "pdfs"'.format(hmm)
        p = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        (out,err) = p.communicate()
        out = out.decode().strip()
        if out == '':
            print(err.decode())
            raise KaldiProcessError('Acquire hmm model information defailed.')
        else:
            dim = out.split()[-1]

    options = '--print-args=False --verbose=0 --binary=false --counts-dim={} '.format(dim)
    if countPhone:
        getAliOption = 'ali-to-phones --per-frame=true'
    else:
        getAliOption = 'ali-to-pdf'
    cmd = "analyze-counts {}\"ark:{} {} \\\"ark:gunzip -c {} |\\\" ark:- |\" {}".format(options,getAliOption,hmm,aliFile,outFile)
    p = subprocess.Popen(cmd,shell=True,stderr=subprocess.PIPE)
    (out,err) = p.communicate()     
    if not os.path.isfile(outFile):
        print(err.decode())
        raise KaldiProcessError('Analyze counts defailed.')
    else:
        return outFile

def decompress(data):
    '''
    Useage:  obj = decompress(feat)

    Feat are expected KaldiArk object whose data type is "CM", that is kaldi compressed ark data. Return a KaldiArk object.
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
    
    sp = BytesIO(data)
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
            sp.close()
            raise WrongDataFormat('Miss right binary symbol.')    
    sp.close()
    return KaldiArk(b''.join(newData))

def load(fileName,useSuffix=None):
    '''
    Useage:  obj = load('feat.npy') or obj = load('feat.ark') or obj = load('feat.scp') or obj = load('feat.lst', useSuffix='scp')

    Load kaldi ark feat file, kaldi scp feat file, KaldiArk file, or KaldiDict file. Return KaldiArk or KaldiDict object.

    '''      
    if isinstance(fileName,str):
        if not os.path.isfile(fileName):
            raise PathError("No such file:{}.".format(fileName))
    else:
        raise UnsupportedDataType('Expected feature file.')

    if useSuffix == 'npy' or fileName.endswith('.npy'):

        temp = np.load(fileName)
        datas = KaldiDict()
        try:
            totalSize = 0
            for utt_mat in temp:
                datas[utt_mat[0]] = utt_mat[1]
                totalSize += sys.getsizeof(utt_mat[1])
            if totalSize > 10000000000:
                print('Warning: Data is extramely large. It could not be used correctly sometimes.')                
        except:
            raise UnsupportedDataType("It is not KaldiDict data.")
        else:
            return datas
    else:
        if useSuffix == 'ark' or fileName.endswith('.ark'):
            cmd = 'copy-feats ark:{} ark:-'.format(fileName)
        elif useSuffix == 'scp' or fileName.endswith('.scp'):
            cmd = 'copy-feats scp:{} ark:-'.format(fileName)
        else:
            raise UnsupportedDataType('Unknown suffix. You can assign useSuffix=<scp> <ark> or <npy>.')

        p = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        (out,err) = p.communicate()
        if out == b'':
            err = err.decode()
            print(err)
            raise KaldiProcessError('Copy feat defeated.')
        else:
            if sys.getsizeof(out) > 10000000000:
                print('Warning: Data is extramely large. It could not be used correctly sometimes.') 
            return KaldiArk(out)

# ---------- Decode Funtions -----------

def decode_lattice(amp,hmm,hclg,wordSymbol,minActive=200,maxActive=7000,maxMem=50000000,beam=10,latBeam=8,acwt=1,config=None,maxThreads=1,outFile=None):
    '''
    Useage:  kaldiLatticeObj = decode_lattice(amp,'graph/final.mdl','graph/hclg')

    Decode by generating lattice from acoustic probability. Return KaldiLattice object or file path if <outFile> is True.
    We provide some usual options, but if you want use more, set < config > = your-configure. Note that if you do this, these usual configures we provided will be ignored.
    You can use pythonkaldi.check_config('decode-lattice') function to get configure information you could set.
    Also run shell command "latgen-faster-mapped" to look their meaning.

    '''  
    if isinstance(amp,KaldiArk):
        pass
    elif isinstance(amp,KaldiDict):
        amp = amp.ark
    else:
        raise UnsupportedDataType("Expected KaldiArk KaldiDict file")
    
    for i in [hmm,hclg,wordSymbol]:
        if not os.path.isfile(i):
            raise PathError("No such file:{}".format(i))

    if maxThreads > 1:
        kaldiTool = "latgen-faster-mapped-parallel --num-threads={}".format(maxThreads)
    else:
        kaldiTool = "latgen-faster-mapped" 

    if config != None:
        
        if check_config(name='decode_lattice',config=config):
            config['--word-symbol-table'] = wordSymbol
            for key in config.keys():
                kaldiTool = ' {}={}'.format(key,config[key])

    else:
        kaldiTool += ' --allow-partial=true'
        kaldiTool += ' --min-active={}'.format(minActive)
        kaldiTool += ' --max-active={}'.format(maxActive)
        kaldiTool += ' --max_mem={}'.format(maxMem)
        kaldiTool += ' --beam={}'.format(beam)
        kaldiTool += ' --lattice-beam={}'.format(latBeam)
        kaldiTool += ' --acoustic-scale={}'.format(acwt)
        kaldiTool += ' --word-symbol-table={}'.format(wordSymbol)

    if outFile != None:

        if not outFile.endswith('.gz'):
            outFile += '.gz'

        cmd1 = '{} {} {} ark:- ark:| gzip -c > {}'.format(kaldiTool,hmm,hclg,outFile)
        p = subprocess.Popen(cmd1,shell=True,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
        (out1,err1) = p.communicate(input=amp)
        if not os.path.isfile(outFile):
            err1 = err1.decode()
            print(err1)
            raise KaldiProcessError('Generate lattice defeat.')
        else:
            return outFile
    else:
        cmd2 = '{} {} {} ark:- ark:-'.format(kaldiTool,hmm,hclg)
        p = subprocess.Popen(cmd2,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
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
    Useage:  configure = check_config(name='compute_mfcc')  or  check_config(name='compute_mfcc',config=configure)
    
    Get default configure if < config > is None, or check if given < config > has a right format.
    This function will read "conf" file which is placed in "./", so if there is not, will raise error.
    Also you can change the content of "conf" file.

    '''

    assert isinstance(name,str), "<name> should be a name-like string."

    cFile = './conf'

    if not os.path.isfile(cFile):
        raise PathError("Miss the global configure file. Please download it again from https://github.com/wangyu09/pythonkaldi.")

    c = configparser.ConfigParser()
    c.read(cFile)

    if not name in c:
        print("Warning: no default configure for name {}.".format(name))
        return None 

    def transType(proto,value=''):
        value = value.strip().lower()
        if value == 'none':
            return None
        
        proto = proto.strip().lower()
        if proto == 'float':
            if value != '':
                return float(value)
            else:
                return float
        elif proto == 'int':
            if value != '':
                return int(value)
            else:
                return int
        elif proto == 'bool':
            if value == 'false':
                return False
            elif value != '':
                return True
            else:
                return bool
        elif proto == 'str':
            if value != '':
                return value
            else:
                return str
        else:
            raise UnsupportedDataType('{} is unsupported type.')       

    if config == None:
        new = {}
        for key,values in c.items(name):
            values = values.split(',')
            proto = values[-1]
            if len(values) == 2:
                new[key] = transType(proto,values[0])
            else:
                for index,value in enumerate(values[0:-1]):
                    values[index] = transType(proto,value)
                new[key] = values[0:-1]
        return new
    else:
        if not isinstance(config,dict):
            raise WrongOperation("<config> has a wrong format. You can use PythonKaldi.check_config({}) to look expected configure format.".format(name))

        if len(config.keys()) < len(c.items(name)):
            reKeys = []
            for k,v in c.items(name):
                if not k in config.keys():
                    reKeys.append(k)
            raise WrongOperation('Missing configure of keys: {}.'.format(','.join(reKeys)))
        else:

            for k in config.keys():
                if k in c[name]:
                    value = c.get(name,k)
                else:
                    raise WrongOperation('No such configure value: < {} > in {}.'.format(k,name))
                
                proto = value.split(',')[-1]

                if isinstance(config[k],(list,tuple)):
                    for v in config[k]:
                        if v != None and not isinstance(v,transType(proto)):
                            raise WrongDataFormat("configure < {} > is expected {} but got {}.".format(k,proto,type(v)))
                else:
                    if config[k] != None and not isinstance(config[k],transType(proto)):
                        raise WrongDataFormat("configure < {} > is expected {} but got {}.".format(k,proto,type(config[k])))

            return True

def run_shell_cmd(cmd,inputs=None):
    '''
    Useage:  out,err = run_shell_cmd('ls -lh')

    We provided a basic way to run shell command. Return binary string (out,err). 

    '''

    if inputs != None:
        if isinstance(inputs,str):
            inputs = inputs.encode()
        elif isinstance(inputs,bytes):
            pass
        else:
            raise UnsupportedDataType("Expected <inputs> is str or bytes but got {}.".format(type(inputs)))

    p = subprocess.Popen(cmd,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    (out,err) = p.communicate(input=inputs)
    return out,err

def split_file(filePath,chunks=2):
    '''
    Useage:  score = split_file('eval.scp',5)

    Split a large scp file into n smaller files. The splited files will be put at the same folder as original file and return their paths as a list.
    '''    
    assert isinstance(chunks,int) and chunks > 1, "Expected <chunks> is int and bigger than 1."

    if not os.path.isfile(filePath):
        raise PathError("No such file:{}.".format(filePath))

    with open(filePath) as fr:
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
    suffixIndex = filePath.rfind('.')
    if suffixIndex != -1:
        newFile = filePath[0:suffixIndex] + "_%0{}d".format(a) + filePath[suffixIndex:]
    else:
        newFile = filePath + "_%0{}d".format(a)

    for i in range(chunks):
        if i < t:
            chunkData = data[i*(chunkLines+1):(i+1)*(chunkLines+1)]
        else:
            chunkData = data[i*chunkLines:(i+1)*chunkLines]
        with open(newFile%(i),'w') as fw:
            fw.write(''.join(chunkData))
        files.append(newFile%(i))
    
    return files

def pad_sequence(data,shuffle=False,pad=0):
    '''
    Useage:  data,lengths = pad_sequence(dataList)

    Pad sequences with max length. < data > is expected as a list whose members are sequence data with a shape of [ frames,featureDim ].

    If <shuffle> is True, pad each sequence with random start index and return padded data and length information of (startIndex,endIndex) of each sequence.

    If <shuffle> is False, align the start index of all sequences then pad them rear. This will return length information of only endIndex.
    '''    
    assert isinstance(data,list), "Expected < data > is a list but got {}.".format(type(data))
    assert isinstance(pad,(int,float)), "Expected < pad > is an int or float but got {}.".format(type(data))

    lengths = []
    for i in data:
        lengths.append(len(i))
    
    maxLen = int(max(lengths))
    batchSize = len(lengths)

    if len(data[0].shape) == 1:
        data = pad * np.ones([maxLen,batchSize])
        for k in range(batchSize):
            snt = len(data[k])
            if shuffle:
                n = maxLen - snt
                n = random.randint(0,n)
                data[n:n+snt,k] = data[k]
                lengths[k] = (n,n+snt)
            else:
                data[0:snt,k] = data[k]
    elif len(data[0].shape) == 2:
        dim = data[0].shape[1]
        data = pad * np.ones([maxLen,batchSize,dim])
        for k in range(batchSize):
            snt = len(data[k])
            if shuffle:
                n = maxLen - snt
                n = random.randint(0,n)
                data[n:n+snt,k,:] = data[k]
                lengths[k] = (n,n+snt)
            else:
                data[0:snt,k,:] = data[k]
    elif len(data[0].shape) >= 3:
        otherDims = data[0].shape[2:]
        allDims = 1
        for i in otherDims:
            allDims *= i
        data = pad * np.ones([maxLen,batchSize,allDims])
        for k in range(batchSize):
            snt = len(data[k])
            if shuffle:
                n = maxLen - snt
                n = random.randint(0,n)
                data[n:n+snt,k,:] = data[k].reshape([snt,allDims])
                lengths[k] = (n,n+snt)
            else:
                data[0:snt,k,:] = data[k].reshape([snt,allDims])
        data = data.reshape([maxLen,batchSize,*otherDims])
    return data, lengths

def unpack_padded_sequence(data,lengths,batchSizeFirst=False):

    '''
    Useage:  listData = pad_sequence(data,lengths)

    This is a reverse operation of pad_sequence() function. Return a list whose members are sequences.
    '''   

    assert isinstance(data,np.ndarray), "Expected <data> is numpy array but got {}.".format(type(data))
    
    if not batchSizeFirst:
        s = data.shape[2:]
        data = data.transpose([1,0,*s])

    new = []
    for i,j in enumerate(data):
        if isinstance(lengths[i],int):
            new.append(j[0:lengths[i]])
        else:
            new.append(j[lengths[i][0]:lengths[i][1]])
    
    return new

def wer(ref,hyp,mode='present',ignore=None, p=True):
    '''
    Useage:  score = compute_wer('ref.txt','pre.txt',ignore='<sil>') or score = compute_wer(out[1],'ref.txt')

    Compute wer between prediction result and reference text. Return a dict object with score information like:
    {'WER':0,'allWords':10,'ins':0,'del':0,'sub':0,'SER':0,'wrongSentences':0,'allSentences':1,'missedSentences':0}
    Both <hyp> and <ref> can be text file or result which obtained from KaldiLattice.get_1best_word().  

    '''
    if ignore == None:

        if isinstance(hyp,list):
            out1 = "\n".join(hyp)
        elif isinstance(hyp,str) and os.path.isfile(hyp):
            with open(hyp,'r') as fr:
                out1 = fr.read()
        else:
            raise UnsupportedDataType('Hyp is not a result-list or file avalible.')

        if out1 == '':
            raise WrongDataFormat("Hyp has not correct data.")
        else:
            out1 = out1.encode()

        if isinstance(ref,list):
            out2 = "\n".join(ref)
        elif isinstance(ref,str) and os.path.isfile(ref):
            with open(ref,'r') as fr:
                out2 = fr.read()
        else:
            raise UnsupportedDataType('Ref is not a result-list or file avalible.')

        if out2 == '':
            raise WrongDataFormat("Ref has not correct data.")
        
    else:
        if not (isinstance(ignore,str) and len(ignore) > 0):
            raise WrongOperation('<ignore> must be a string avaliable.')

        if isinstance(hyp,list):
            hyp = ("\n".join(hyp)).encode()
            p1 = subprocess.Popen('sed "s/{} //g"'.format(ignore),shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            (out1,err1) = p1.communicate(input=hyp)
        elif isinstance(hyp,str) and os.path.isfile(hyp):
            p1 = subprocess.Popen('sed "s/{} //g" <{}'.format(ignore,hyp),shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            (out1,err1) = p1.communicate()     
        else:
            raise UnsupportedDataType('Hyp is not a result-list or file avalible.')

        if out1 == b'':
            raise WrongDataFormat("Hyp has not correct data.")

        if isinstance(ref,list):
            ref = ("\n".join(ref)).encode()
            p2 = subprocess.Popen('sed "s/{} //g"'.format(ignore),shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            (out2,err2) = p2.communicate(input=ref)
        elif isinstance(ref,str) and os.path.isfile(ref):
            p2 = subprocess.Popen('sed "s/{} //g" <{}'.format(ignore,ref),shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            (out2,err2) = p2.communicate()
        else:
            raise UnsupportedDataType('Ref is not a result-list or file avalible.') 

        if out2 == b'':
            raise WrongDataFormat("Ref has not correct data.")

        out2 = out2.decode()

    with tempfile.NamedTemporaryFile('w+') as fw:
        fw.write(out2)
        fw.seek(0)
        cmd = 'compute-wer --text --mode={} ark:{} ark,p:-'.format(mode,fw.name)
        p3 = subprocess.Popen(cmd,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        (out3,err3) = p3.communicate(input=out1)
    
    if out3 == b'':
        err3 = err3.decode()
        print(err3)
        raise KaldiProcessError('Compute wer defeated.')
    else:
        out = out3.decode()
        if not p:
            print(out)
        score = {}
        out = out.split("\n")
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

def accuracy(predict,label,ignore=None,mode='all'):
    '''
    Useage:  score = accuracy(predict,label,ignore=0)

    Compute one-one match score. for example predict is (1,2,3,4), and label is (1,2,2,4), the score will be 0.75.
    Both <predict> and <label> are expected list, tuple or NumPy array with int members. They will be flatten before score.
    Ignoring value can be assigned with <ignore>.
    If <mode> is all, it will raise ERROR when the length of predict and label is different.
    If <mode> is present, compare depending on the shorter one.
    '''
    assert mode in ['all','present'], 'Expected <mode> is present or all.'

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

    x = flatten(predict)
    y = flatten(label)

    i = 0
    score = []
    while True:
        if i >= len(x) or i >= len(y):
            break
        elif x[i] == y[i]:
            score.append(1)
        else:
            score.append(0)
        i += 1
   
    if mode == 'present':
        return float(np.mean(score))
    else:
        if i < len(predict) or i < len(label):
            raise WrongOperation('<present> and <label> have different length to score.')
        else:
            return float(np.mean(score))

def edit_distance(predict,label,ignore=None):
    '''
    Useage:  score = edit_distance(predict,label,ignore=0)

    Compute edit distance score. 
    Both <predict> and <label> can be string, list, tuple, or NumPy array.
    '''
    #assert isinstance(x,str) and isinstance(y,str), "Expected both <x> and <y> are string."
    
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

    x = flatten(predict)
    y = flatten(label)

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

    assert isinstance(data,np.ndarray), "Expected numpy ndarray data but got {}.".format(type(data))
    assert axis >=0 and axis < len(data.shape), '{} is out of the dimensions of data.'

    tShape = list(data.shape)
    tShape[axis] = 1
    data = np.array(data,dtype='float32')
    dataExp = np.exp(data)
    dataExpLog = np.log(np.sum(dataExp,axis)).reshape(tShape)

    return data - dataExpLog
