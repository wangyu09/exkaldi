############# Version Information #############
# PythonKaldi V1.6
# WangYu, University of Yamanashi 
# Sep, 9, 2019
###############################################

import os,sys
import struct,copy,re,time
import math,socket,random
import subprocess,threading
import wave#,pyaudio
import queue,tempfile
import numpy as np
from io import BytesIO
import configparser

class PathError(Exception):pass
class UnsupportedDataType(Exception):pass
class WrongDataFormat(Exception):pass
class KaldiProcessError(Exception):pass
class WrongOperation(Exception):pass

def get_kaldi_path():
    '''
    Useage:  KALDIROOT = get_kaldi_path() 
    
    Return kaldi path. If the kaldi are not found, will raise error.

    '''
    p = subprocess.Popen('which copy-feats',shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    (out,err) = p.communicate()
    if out == b'':
        #print('Kaldi not found. If you really have installed it, set the global parameter KALDIROOT=<your path> manually.')
        #return './'
        raise PathError('Kaldi not found.')
    else:
        return out.decode().strip()[0:-23]

KALDIROOT = get_kaldi_path()

def check_config(name,config=None):
    '''
    Useage:  configure = check_config(name='compute_mfcc')  or   check_config(name='compute_mfcc',config=configure)
    
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
            raise WrongOperation("<config> has a wrong format. Try to use PythonKaldi.check_config({}) to look expected configure format.".format(name))

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

class KaldiArk(bytes):
    '''
    Useage:  obj = KaldiArk(binaryData)  or   obj = KaldiArk()
    
    KaldiArk is a subclass of bytes. It maks a object who holds the kaldi ark data in a binary type. 
    KaldiArk and KaldiDict object have almost the same attributes and functions, and they can do some mixed operations such as "+" and "concat" and so on.
    Moreover, alignment can also be held by KaldiArk and KaldiDict in Pythonkaldi tool, and we defined it as int32 data type.

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
        return "KaldiArk object with unviemable binary data"

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
        
        Return an int: feature dimensional.
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

    def toDtype(self,dtype):
        '''
        Useage:  newObj = obj.toDtype('float')
        
        Return a new KaldiArk object. 'float' will be treated as 'float32' and 'int' will be 'int32'.

        '''
        if self.dtype != dtype:
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
        else:
            return self

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
                newData = self.toDtype('float32')
            else:
                newData = self
            for other in others:
                with tempfile.NamedTemporaryFile(mode='w+b') as fn:
                    otherDtype = other.dtype
                    if otherDtype == 'int32':
                        other = other.toDtype('float32')
                    fn.write(other)
                    fn.seek(0)
                    cmd = 'paste-feats ark:- ark:{} ark:-'.format(fn.name)
                    p = subprocess.Popen(cmd,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                    (newData,err) = p.communicate(input=newData)
                    if newData == b'':
                        err = err.decode()
                        raise KaldiProcessError(err)
                    elif dataType == 'int32' and otherDtype == 'int32':
                        newData = bytes(KaldiArk(newData).toDtype('int32'))
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

class KaldiDict(dict):
    '''
    Useage:  obj = KaldiDict(binaryData)  or   obj = KaldiDict()

    KaldiDict is a subclass of dict. It is a object who holds the kaldi ark data in numpy array type. 
    Its key are the utterance id and the value is the numpy array data. KaldiDict can also do some mixed operations with KaldiArk such as "+" and "concat" and so on.
    Note that KaldiDict has some functions which KaldiArk dosen't have.

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
    
    def toDtype(self,dtype):
        '''
        Useage:  newObj = obj.toDtype('float')
        
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
                    
    def merge(self,keepDim=False):
        '''
        Useage:  data,uttlength = obj.merge() or data,uttlength = obj.merge(keepDim=True)
        
        If < keepDim > is False, the first one is returned result is 2-dim numpy array, the second one is a list consists of id and frames of each utterance. 
        If < keepDim > is True, only the first one will be a list.

        ''' 
        uttLens = []
        matrixs = []
        for utt in self.keys():
            mat = self[utt]
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

class KaldiLattice(object):
    '''
    Useage:  obj = KaldiLattice()  or   obj = KaldiLattice(lattice,hmmgmm,wordSymbol)

    KaldiLattice holds the lattice and its related file path: HmmGmm file and WordSymbol file. 
    The <lattice> can be lattice binary data or file path. Both < HmmGmm > and < wordSymbol > are expected file path.
    pythonkaldi.decode_lattice function will return a KaldiLattice object. Aslo, you can define a empty KaldiLattice object and load its data later.

    '''    
    def __init__(self,lattice=None,HmmGmm=None,wordSymbol=None):

        self._lat = lattice
        self._model = HmmGmm
        self._word2id = wordSymbol

        if lattice != None:
            assert HmmGmm != None and wordSymbol != None, "Expected HmmGmm file and wordSymbol file."
            if isinstance(lattice,str):
                self.load(lattice,HmmGmm,wordSymbol)
            elif isinstance(lattice,bytes):
                pass
            else:
                raise UnsupportedDataType("<lattice> is not a correct lattice data.")

    def load(self,latFile,HmmGmm,wordSymbol):
        '''
        Useage:  obj.load(lattice,hmmgmm,wordSymbol)

        Load lattice to memory. < latFile > can be file path or binary data. < HmmGmm > and < wordSymbol > are expected as file path.
        Note that the original data in obj will be abandoned. 

        '''    
        for x in [latFile,HmmGmm,wordSymbol]:
            if not os.path.isfile(x):
                raise PathError('No such file:{}.'.format(latFile))
        if latFile.endswith('.gz'):
            p = subprocess.Popen('gunzip -c {}'.format(latFile),shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            (out,err)=p.communicate()
            if out == b'':
                print('Lattice load defeat!')
                raise Exception(err.decode())
            else:
                self._lat = out
                self._model = HmmGmm
                self._word2id = wordSymbol
        else:
            raise UnsupportedDataType('Expected gz file but got {}.'.format(latFile))

    def get_1best_words(self,minLmwt=1,maxLmwt=None,Acwt=1.0,outDir='.',asFile=False):
        '''
        Useage:  out = obj.get_1best_words(minLmwt=1)

        Return a dict object. Its key is lm weight, and value will be result-list if <asFile> is False or result-file-path if <asFile> is True. 

        ''' 
        if self._lat == None:
            raise WrongOperation('No any data in lattice.')

        if outDir.endswith('/'):
            outDir=outDir[:-1]

        KALDIROOT = get_kaldi_path()
        wordSymbol = self._word2id

        if maxLmwt != None:
            if maxLmwt < minLmwt:
                raise WrongOperation('<maxLmwt> must larger than <minLmwt>.')
            else:
                maxLmwt += 1
        else:
            maxLmwt = minLmwt + 1

        result = {}
        if asFile != False:
            if not os.path.isdir(outDir):
                os.mkdir(outDir)
            if asFile == True:
                asFile = outDir+'/1best'
            elif '/' in asFile:
                asFile = outDir+'/'+ asFile.split('/')[-1]
            else:
                asFile = outDir +'/'+ asFile

            for LMWT in range(minLmwt,maxLmwt,1):
                cmd1 = KALDIROOT+'/src/latbin/lattice-best-path --lm-scale={} --acoustic-scale={} --word-symbol-table={} --verbose=2 ark:- ark,t:- |'.format(LMWT,Acwt,wordSymbol)
                cmd2 = KALDIROOT+'/egs/wsj/s5/utils/int2sym.pl -f 2- {} > {}_{}'.format(wordSymbol,asFile,LMWT)
                cmd = cmd1+cmd2
                p = subprocess.Popen(cmd,shell=True,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
                (out,err) = p.communicate(input=self._lat)
                if not os.path.isfile('{}_{}'.format(asFile,LMWT)):
                    err = err.decode()
                    logFile = outDir+'/lattice_1best.{}.log'.format(LMWT)
                    with open(logFile,'w') as fw:
                        fw.write(err)
                    raise KaldiProcessError('Lattice to 1-best Defeated. Look the log file {}.'.format(logFile))
                else:
                    result[LMWT] = '{}_{}'.format(asFile,LMWT)
        else:
            for LMWT in range(minLmwt,maxLmwt,1):
                cmd1 = KALDIROOT+'/src/latbin/lattice-best-path --lm-scale={} --acoustic-scale={} --word-symbol-table={} --verbose=2 ark:- ark,t:- |'.format(LMWT,Acwt,wordSymbol)
                cmd2 = KALDIROOT+'/egs/wsj/s5/utils/int2sym.pl -f 2- {} '.format(wordSymbol)
                cmd = cmd1+cmd2
                p = subprocess.Popen(cmd,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                (out,err) = p.communicate(input=self._lat)
                if out == b'':
                    err = err.decode()
                    if not os.path.isdir(outDir):
                        os.mkdir(outDir)
                    logFile = outDir+'/lattice_1best.{}.log'.format(LMWT)
                    with open(logFile,'w') as fw:
                        fw.write(err)
                    raise KaldiProcessError('Lattice to 1-best Defeated. Look the log file {}.'.format(logFile))
                else:
                    out = out.decode().split("\n")
                    result[LMWT] = out[0:-1]
        if maxLmwt == None:
            result = result[minLmwt]
        return result

    def scale(self,Acwt=1,inAcwt=1,Ac2Lm=0,Lmwt=1,Lm2Ac=0):
        '''
        Useage:  newObj = obj.sacle(inAcwt=0.2)

        Scale lattice. Return a new KaldiLattice object.

        ''' 
        if self._lat == None:
            raise WrongOperation('No any lattice to scale.')

        for x in [Acwt,inAcwt,Ac2Lm,Lmwt,Lm2Ac]:
            assert isinstance(x,int) and x>= 0,"Expected each scale is int and >=0."
        
        cmd = KALDIROOT+'/src/latbin/lattice-scale'
        cmd += ' --acoustic-scale={}'.format(Acwt)
        cmd += ' --acoustic2lm-scale={}'.format(Ac2Lm)
        cmd += ' --inv-acoustic-scale={}'.format(inAcwt)
        cmd += ' --lm-scale={}'.format(Lmwt)
        cmd += ' --lm2acoustic-scale={}'.format(Lm2Ac)
        cmd += ' ark:- ark:-'

        p = subprocess.Popen(cmd,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        (out,err) = p.communicate(input=self._lat)

        if out == b'':
            raise KaldiProcessError(err.decode())
        else:
            return KaldiLattice(out,self._model,self._word2id)

    def add_penalty(self,penalty=0):
        '''
        Useage:  newObj = obj.add_penalty(0.5)

        Add penalty. Return a new KaldiLattice object.

        ''' 
        if self._lat == None:
            raise WrongOperation('No any lattice to scale.')

        assert isinstance(penalty,(int,float)) and penalty>= 0, "Expected <penalty> is int or float and >=0."
        
        cmd = KALDIROOT+'/src/latbin/lattice-add-penalty'
        cmd += ' --word-ins-penalty={}'.format(penalty)
        cmd += ' ark:- ark:-'

        p = subprocess.Popen(cmd,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        (out,err) = p.communicate(input=self._lat)

        if out == b'':
            raise KaldiProcessError(err.decode())
        else:
            return KaldiLattice(out,self._model,self._word2id)

    def save(self,fileName):
        '''
        Useage:  obj.save("lat.gz")

        Save lattice as .gz file. 

        ''' 
        if self._lat == None:
            raise WrongOperation('No any data to save.')   

        if not fileName.endswith('.gz'):
            fileName += '.gz'

        cmd = 'gzip -c > {}'.format(fileName)
        p = subprocess.Popen(cmd,shell=True,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
        (out,err) = p.communicate(input=self._lat)
        if not os.path.isfile(fileName):
            err = err.decode()
            print(err)
            exit(1)

    @property
    def value(self):
        '''
        Useage:  lat = obj.value

        Return binary lattice data. 

        ''' 
        return self._lat

    def __add__(self,other):
        '''
        Useage:  lat3 = lat1 + lat2

        Return a new KaldiLattice object. lat2 must be KaldiLattice object.
        Note that this is only a simple additional operation to make two lattices be one.

        '''         
        assert isinstance(other,KaldiLattice), "Expected KaldiLattice but got {}.".format(type(other))

        if self._lat == None:
            return other
        elif other._lat == None:
            return self
        elif self._model != other._model or self._word2id != other._word2id:
            raise WrongOperation("Both two members must use the same hmm-gmm model and word-symbol file.")
        
        newLat = self._lat + other._lat 

        return KaldiLattice(newLat,self._model,self._word2id)

def compute_mfcc(wavFile,rate=16000,frameWidth=25,frameShift=10,melBins=23,featDim=13,windowType='povey',useUtt='MAIN',useSuffix=None,config=None,asFile=False):
    '''
    Useage:  obj = compute_mfcc("test.wav") or compute_mfcc("test.scp")

    Compute mfcc feature. Return KaldiArk object or file path if <asFile> is True.
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

    if asFile:

        if asFile == True:
            asFile = 'mfcc.ark'

        if not asFile.endswith('.ark'):
            asFile += '.ark'
        
        if wavFile.endswith('wav') or "wav" in useSuffix:
            cmd = 'echo {} {} | {} scp:- ark:{}'.format(useUtt,wavFile,kaldiTool,asFile)
        elif wavFile.endswith('scp') or "scp" in useSuffix:
            cmd = '{} scp:{} ark:{}'.format(kaldiTool,wavFile,asFile)
        else:
            raise UnsupportedDataType('Unknown file suffix. You can declare it by using <useSuffix>="wav" or "scp".')

        p = subprocess.Popen(cmd,shell=True,stderr=subprocess.PIPE)
        (out,err) = p.communicate()
        if not os.path.isfile(asFile):
            err = err.decode()
            print(err)
            raise KaldiProcessError('Compute mfcc defeated.')
        else:
            return asFile
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

def compute_fbank(wavFile,rate=16000,frameWidth=25,frameShift=10,melBins=23,windowType='povey',useUtt='MAIN',useSuffix=None,config=None,asFile=False):
    '''
    Useage:  obj = compute_fbank("test.wav") or compute_mfcc("test.scp")

    Compute fbank feature. Return KaldiArk object or file path if <asFile> is True.
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

    if asFile:

        if asFile == True:
            asFile = 'fbank.ark'

        if not asFile.endswith('.ark'):
            asFile += '.ark'
        
        if wavFile.endswith('wav') or "wav" in useSuffix:
            cmd = 'echo {} {} | {} scp:- ark:{}'.format(useUtt,wavFile,kaldiTool,asFile)
        elif wavFile.endswith('scp') or "scp" in useSuffix:
            cmd = '{} scp:{} ark:{}'.format(kaldiTool,wavFile,asFile)
        else:
            raise UnsupportedDataType('Unknown file suffix. You can declare it by using <useSuffix>="wav" or "scp".')

        p = subprocess.Popen(cmd,shell=True,stderr=subprocess.PIPE)
        (out,err) = p.communicate()
        if not os.path.isfile(asFile):
            err = err.decode()
            print(err)
            raise KaldiProcessError('Compute fbank defeated.')
        else:
            return asFile
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

def compute_plp(wavFile,rate=16000,frameWidth=25,frameShift=10,melBins=23,featDim=13,windowType='povey',useUtt='MAIN',useSuffix=None,config=None,asFile=False):
    '''
    Useage:  obj = compute_plp("test.wav") or compute_mfcc("test.lst",useSuffix='scp')

    Compute plp feature. Return KaldiArk object or file path if <asFile> is True.
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

    if asFile:

        if asFile == True:
            asFile = 'plp.ark'

        if not asFile.endswith('.ark'):
            asFile += '.ark'
        
        if wavFile.endswith('wav') or "wav" in useSuffix:
            cmd = 'echo {} {} | {} scp:- ark:{}'.format(useUtt,wavFile,kaldiTool,asFile)
        elif wavFile.endswith('scp') or "scp" in useSuffix:
            cmd = '{} scp:{} ark:{}'.format(kaldiTool,wavFile,asFile)
        else:
            raise UnsupportedDataType('Unknown file suffix. You can declare it by using <useSuffix>="wav" or "scp".')

        p = subprocess.Popen(cmd,shell=True,stderr=subprocess.PIPE)
        (out,err) = p.communicate()
        if not os.path.isfile(asFile):
            err = err.decode()
            print(err)
            raise KaldiProcessError('Compute plp defeated.')
        else:
            return asFile
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

def compute_spectrogram(wavFile,rate=16000,frameWidth=25,frameShift=10,windowType='povey',useUtt='MAIN',useSuffix=None,config=None,asFile=False):
    '''
    Useage:  obj = compute_spetrogram("test.wav") or compute_mfcc("test.lst",useSuffix='scp')

    Compute spectrogram feature. Return KaldiArk object or file path if <asFile> is True.
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

    if asFile:

        if asFile == True:
            asFile = 'spectrogram.ark'

        if not asFile.endswith('.ark'):
            asFile += '.ark'
        
        if wavFile.endswith('wav') or "wav" in useSuffix:
            cmd = 'echo {} {} | {} scp:- ark:{}'.format(useUtt,wavFile,kaldiTool,asFile)
        elif wavFile.endswith('scp') or "scp" in useSuffix:
            cmd = '{} scp:{} ark:{}'.format(kaldiTool,wavFile,asFile)
        else:
            raise UnsupportedDataType('Unknown file suffix. You can declare it by using <useSuffix>="wav" or "scp".')

        p = subprocess.Popen(cmd,shell=True,stderr=subprocess.PIPE)
        (out,err) = p.communicate()
        if not os.path.isfile(asFile):
            err = err.decode()
            print(err)
            raise KaldiProcessError('Compute spectrogram defeated.')
        else:
            return asFile
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

def use_cmvn(feat,cmvnStatFile=None,utt2spkFile=None,spk2uttFile=None,asFile=False):
    '''
    Useage:  obj = use_cmvn(feat) or obj = use_cmvn(feat,cmvnStatFile,utt2spkFile) or obj = use_cmvn(feat,utt2spkFile,spk2uttFile)

    Apply CMVN to feature. Return KaldiArk object or file path if <asFile> is true. 
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

        if asFile != False:

            if asFile == True:
                asFile = '/cmvn.ark'

            if not asFile.endswith('.ark'):
                asFile += '.ark'

            if utt2spkFile != None:
                cmd2 = 'apply-cmvn --utt2spk=ark:{} {} ark:- ark:{}'.format(utt2spkFile,cmvnStatFileOption,asFile)
            else:
                cmd2 = 'apply-cmvn {} ark:- ark:{}'.format(cmvnStatFileOption,asFile)

            p2 = subprocess.Popen(cmd2,shell=True,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
            (out2,err2) = p2.communicate(input=feat)

            if not os.path.isfile(asFile):
                err2 = err2.decode()
                print(err2)
                raise KaldiProcessError('Use cmvn defeated.')
            else:
                return asFile
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

def compute_cmvn_stats(feat,asFile,spk2uttFile=None):
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

    if asFile.endswith('.scp'):
        cmd += ' ark,scp:{},{}'.format(asFile[0:-4]+'.ark',asFile)
    else:
        if not asFile.endswith('.ark'):
            asFile = asFile + '.ark'
        cmd += ' ark:{}'.format(asFile)

    p = subprocess.Popen(cmd,shell=True,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    (_,err) = p.communicate(input=feat)

    if not os.path.isfile(asFile):
        err = err.decode()
        print(err)
        raise KaldiProcessError('Compute cmvn stats defeated.')
    else:
        return asFile    

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

def add_delta(feat,order=2,asFile=False):
    '''
    Useage:  newObj = add_delta(feat)

    Add n-orders delta to feature. Return KaldiArk object or file path if <asFile> is True.

    ''' 
    if isinstance(feat,KaldiArk):
        pass
    elif isinstance(feat,KaldiDict):
        feat = feat.ark
    else:
        raise UnsupportedDataType("Expected KaldiArk KaldiDict but got {}.".format(type(feat)))

    if asFile:

        if asFile == True:
            asFile = 'add_deltas_{}.ark'.format(order)

        if not asFile.endswith('.ark'):
            asFile += '.ark'
            
        cmd1 = 'add-deltas --delta-order={} ark:- ark:{}'.format(order,asFile)

        p1 = subprocess.Popen(cmd1,shell=True,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
        (out,err) = p1.communicate(input=feat)

        if not os.path.isfile(asFile):
            err = err.decode()
            print(err)
            raise KaldiProcessError('Add delta defeated.')
        else:
            return asFile 
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

def get_ali(aliFile,HmmGmm,returnPhoneme=False):
    '''
    Useage:  obj = get_ali('ali.1.gz','graph/final.mdl') or obj = get_ali('ali.1.gz','graph/final.mdl',True)

    Get alignment from ali file. Return a KaldiDict object. If <returnPhoneme> is True, return phone id, or return pdf id.

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

    if isinstance(HmmGmm,str):
        if not os.path.isfile(HmmGmm):
            raise PathError("Miss Model file:{}".format(HmmGmm))
    else:
        raise UnsupportedDataType("Expected Hmm-Gmm model path but got {}.".format(type(aliFile)))

    if returnPhoneme:
        cmd = 'gunzip -c {} | ali-to-phones --per-frame=true {} ark:- ark,t:-'.format(aliFile,HmmGmm)
    else:
        cmd = 'gunzip -c {} | ali-to-pdf {} ark:- ark,t:-'.format(aliFile,HmmGmm)

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
        
def decompress(data):
    '''
    Useage:  obj = decompress(feat)

    Feat are expected KaldiArk object whose data type is "CM", that is kaldi compressed ark data. Return a KaldiArk object.

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

def load(feaFile,useSuffix=None):
    '''
    Useage:  obj = load('feat.npy') or obj = load('feat.ark') or obj = load('feat.scp') or obj = load('feat.lst', useSuffix='scp')

    Load kaldi ark feat file, kaldi scp feat file, KaldiArk file, or KaldiDict file. Return KaldiArk or KaldiDict object.

    '''      
    if isinstance(feaFile,str):
        if not os.path.isfile(feaFile):
            raise PathError("No such file:{}.".format(feaFile))
    else:
        raise UnsupportedDataType('Expected feature file.')

    if useSuffix == 'npy' or feaFile.endswith('.npy'):

        temp = np.load(feaFile)
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
        if useSuffix == 'ark' or feaFile.endswith('.ark'):
            cmd = 'copy-feats ark:{} ark:-'.format(feaFile)
        elif useSuffix == 'scp' or feaFile.endswith('.scp'):
            cmd = 'copy-feats scp:{} ark:-'.format(feaFile)
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

def decode_lattice(AmP,HmmGmm,Hclg,Lexicon,minActive=200,maxActive=7000,maxMem=50000000,beam=13,latBeam=8,Acwt=1,config=None,maxThreads=1,asFile=False):
    '''
    Useage:  kaldiLatticeObj = decode_lattice(amp,'graph/final.mdl','graph/HCLG')

    Decode by generating lattice from acoustic probability. Return KaldiLattice object or file path if <asFile> is True.
    We provide some usual options, but if you want use more, set < config > = your-configure. Note that if you do this, these usual configures we provided will be ignored.
    You can use pythonkaldi.check_config('decode-lattice') function to get configure information you could set.
    Also run shell command "latgen-faster-mapped" to look their meaning.

    '''  
    if isinstance(AmP,KaldiArk):
        pass
    elif isinstance(AmP,KaldiDict):
        AmP = AmP.ark
    else:
        raise UnsupportedDataType("Expected KaldiArk KaldiDict file")
    
    for i in [HmmGmm,Hclg,Lexicon]:
        if not os.path.isfile(i):
            raise PathError("No such file:{}".format(i))

    if maxThreads > 1:
        kaldiTool = "latgen-faster-mapped-parallel --num-threads={}".format(maxThreads)
    else:
        kaldiTool = "latgen-faster-mapped" 

    if config != None:
        
        if check_config(name='decode_lattice',config=config):
            config['--word-symbol-table'] = Lexicon
            for key in config.keys():
                kaldiTool = ' {}={}'.format(key,config[key])

    else:
        kaldiTool += ' --allow-partial=true'
        kaldiTool += ' --min-active={}'.format(minActive)
        kaldiTool += ' --max-active={}'.format(maxActive)
        kaldiTool += ' --max_mem={}'.format(maxMem)
        kaldiTool += ' --beam={}'.format(beam)
        kaldiTool += ' --lattice-beam={}'.format(latBeam)
        kaldiTool += ' --acoustic-scale={}'.format(Acwt)
        kaldiTool += ' --word-symbol-table={}'.format(Lexicon)

    if asFile:

        if asFile == True:
            asFile = 'lattice.gz'

        if not asFile.endswith('.gz'):
            asFile += '.gz'

        cmd1 = '{} {} {} ark:- ark:| gzip -c > {}'.format(kaldiTool,HmmGmm,Hclg,asFile)
        p = subprocess.Popen(cmd1,shell=True,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
        (out1,err1) = p.communicate(input=AmP)
        if not os.path.isfile(asFile):
            err1 = err1.decode()
            print(err1)
            raise KaldiProcessError('Generate lattice defeat.')
        else:
            return asFile
    else:
        cmd2 = '{} {} {} ark:- ark:-'.format(kaldiTool,HmmGmm,Hclg)
        p = subprocess.Popen(cmd2,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        (out2,err2) = p.communicate(input=AmP)
        if out2 == b'':
            err2 = err2.decode()
            print(err2)
            raise KaldiProcessError('Generate lattice defeat.')
        else:
            return KaldiLattice(out2,HmmGmm,Lexicon)

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

def compute_wer(ref,hyp,mode='present',ignore=None,p=True):
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
        score['WER']=s1[0]
        score['allWords']=s1[2]
        score['ins']=s1[3]
        score['del']=s1[4]
        score['sub']=s1[5]
        score['SER']=s2[0]
        score['wrongSentences']=s2[1]        
        score['allSentences']=s2[2]
        score['missedSentences']=s3[1]

        return score

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
    
##### Speak Client #####
# この部分は　まだ 使えない　です。

class SpeakClient(object):

    def __init__(self,mode='local',proto='TCP',targetHost=None,targetPort=9509):

        raise WrongOperation('I am sorry that Speek Client Section is unable to use now.')

        assert (mode == 'local' or mode == 'remote'),'Expected local or remote but got {}'.format(mode)

        #Define dufault audio data size, sample rate ,channels and chunk size
        self.setWaveFormat()

        self.client = None  #To save local net client
        self.p = pyaudio.PyAudio()  #To record voice data from microphone
        self.dataQueue = queue.Queue()  #Default queue to save the raw data from microphone or wave file stream
        self.resultQueue = queue.Queue() #Default queue to save the recognized result
        self._counter = 0  #To count total numbers of raw datas

        self.threadsManager = {}  #Save threads in order to manage them
        self.endFlag = False #Flag when jobs over in a right way in order to wait all threads over rightly
        self.errFlag = False #Flag when jobs over by error in order to stop other threads and server immediately 
        self.isSafeState = False  #Flag to ensure the safety of stream,threads and network communication        
        
        #When remote mode, activate local client.
        if mode == 'remote':
            if targetHost == None:
                raise WrongOperation('Expected target IP address')
            else:
                self.connectTo(proto,targetHost,targetPort)

    @property
    def counter(self):
        return self._counter

    def __enter__(self,*args):
        #when run by "with" grammar, it is safe mode
        self.isSafeState = True
        return self
    
    def __exit__(self,errType, errValue, errTrace):
        #when main thread stop by KeyboardInterrupt, we defined it over rightly,especially when user want to stop record thread
        if errType ==  KeyboardInterrupt:
            self.endFlag = True
            self.errFlag = False
        elif errType != None:
            self.endFlag = False
            self.errFlag = True
        #Wait all threads over
        self.wait()
        #if remote mode, close client(,but note that "endFlag" or "errFlag" will be sent to server before this command)
        if self.client != None:
            self.client.close()
        #Close pyaudio
        self.p.terminate()

    def exit(self):
        #elf.__exit__()
        pass
    
    def pause(self):
        #Tell all threads and server to stop rightly
        self.endFlag = True
        self.errFlag = False
        #Wait ending of all threads
        self.wait()
        #Recover stop-flag to prepare next start
        self.endFlag = False

    def over(self):
        #Tell all threads and server to stop rightly
        self.endFlag = True
        self.errFlag = False
        #Wait ending of all threads
        self.wait()
        #if remote mode, close client(,but note that "endFlag" will be sent before this command)
        #Clear client to prepare to next start
        if self.client != None:
            self.client.close()
            self.client = None
        #Clear counter
        self._counter = 1
        #Clear all of queues
        self.dataQueue.queue.clear()
        self.resultQueue.queue.clear()
        #Recover stop-flag
        self.endFlag = False

    def wait(self):
        for name in self.threadsManager.keys():
            if self.threadsManager[name] != None and self.threadsManager[name].is_alive():
                print(name)
                self.threadsManager[name].join()

    def connectTo(self,proto='TCP',targetHost=None,targetPort=9509):

        if not self.isSafeState:
            raise WrongOperation('Please run under safe state by using <with> grammar')
            #print('Running without safe mode. Please use exit() function finally to ensure tasks safely')

        if self.client != None:
            raise WrongOperation('Another client is working right now')

        assert (proto == 'TCP' or proto == 'UDP'),'Expected proto TCP or UDP but got {}'.format(proto)

        self.proto = proto
        self.targetHost = targetHost
        self.targetPort = targetPort

        #Set network communication max waitting time 10 seconds
        #socket.setdefaulttimeout(10)

        if proto == 'TCP':
            self.client = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
            try:
                self.client.connect((targetHost,targetPort))
                print('Test local-server conmunication...')
                identifiedMessage = self.client.recv(32) #In this step, it is no necessary to try timeout error because server would response certainly 
                if identifiedMessage == b'hello world':
                    print('Local-Server communicate rightly')
                else:
                    self.client.close()
                    self.client = None
                    raise WrongOperation('Net communicate anomaly')
            except ConnectionRefusedError:
                raise WrongOperation('Expected activated target server IP and Port')
        else:
            self.client = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
            print('Test local-server communication...')
            self.client.sendto(b'hello world',(targetHost,targetPort))
            try:
                i = 0
                while i < 5:  #Max validation times 
                    identifiedMessage,addr = self.client.recvfrom(32)
                    if identifiedMessage == b'hello world' and addr == (targetHost,targetPort):
                        break
                    i += 1
            except TimeoutError as e: #Try timeout error in order to check if Server has been activated
                self.client.close()
                self.client = None
                print('Please check if server has been activated')
                raise e
            else:
                if i == 5:
                    self.client.close()
                    self.client = None
                    raise WrongOperation('Net communicate anomaly')
                else:
                    print('Local-Server communicate rightly')

    def setWaveFormat(self,Format='int16',Channels=1,Rate=16000,ChunkFrames=1024):

        assert Format=='int8' or Format=='int16' or Format=="int32","Expected format int16 or int32 but got{}".format(Format)
        assert Channels==1 or Channels==2,"Expected channels 1 or 2 but got {}".format(Channels)

        self.formats = Format
        self.channels = Channels
        self.rate = Rate
        self.chunkFrames = ChunkFrames
        if Format == 'int8':
            self.chunkSize = 1*Channels*ChunkFrames
        elif Format == 'int16':
            self.chunkSize = 2*Channels*ChunkFrames
        else:
            self.chunkSize = 4*Channels*ChunkFrames

    def record(self,seconds=None,dataQueue=None):
        #Input raw data from microphone

        if not self.isSafeState:
            raise WrongOperation('Please run under safe state by using <with> grammar')
            #print('Running without safe mode. Please use exit() function finally to ensure tasks safely')

        if 'record' in self.threadsManager.keys() and self.threadsManager['record'].is_alive():
            raise WrongOperation('Another record thread is running now')

        if 'read' in self.threadsManager.keys() and self.threadsManager['read'].is_alive():
            raise WrongOperation('Can not run record and read thread at the same time.')

        if dataQueue == None:
            dataQueue = self.dataQueue

        if seconds != None:
            assert isinstance(seconds,(int,float)),'Expected seconds int or float but got {}'.format(type(seconds))
            recordTimes = math.ceil(self.rate*seconds/self.chunkFrames)
        else:
            recordTimes = None

        def recordData(recordTimes,dataQueue):
            try:
                #Wave data will be sent as first message
                firstMessage = "{},{},{},{}".format(self.formats,self.channels,self.rate,self.chunkFrames)
                dataQueue.put(firstMessage.encode())
                #Then send wav data
                if self.formats == 'int8':
                    Format = pyaudio.paInt8
                elif self.formats == 'int16':
                    Format = pyaudio.paInt16
                else:
                    Format = pyaudio.paInt32
                stream = self.p.open(format=Format,channels=self.channels,rate=self.rate,input=True,output=False)
                if recordTimes != None:
                    for i in range(recordTimes):
                        data = stream.read(self.chunkFrames)
                        dataQueue.put(data)
                        self._counter += 1
                        if self.endFlag == True: #Stopped rightly
                            break
                        elif self.errFlag == True: #Tnterupted by error
                            raise Exception('Close record thread')
                else:
                    while True:
                        data = stream.read(self.chunkFrames)
                        dataQueue.put(data)
                        self._counter += 1
                        if self.endFlag == True: #Stopped rightly
                            break
                        elif self.errFlag == True: #Interupted by error
                            raise Exception('Close Record thread')
            except Exception as e:
                #Both any other thread had error, and record thread had error, program will arrive here 
                dataQueue.put('errFlag')  #In order to tell later member the error-state
                self.errFlag = True #If record thread had error, tell others
                raise e #Show error information
            else:
                dataQueue.put('endFlag') #In order to tell later member, record is over 
            finally:
                stream.stop_stream()
                stream.close()
                self.threadsManager['record'] = None  #Clear

        self.threadsManager['record'] = threading.Thread(target=recordData,args=(recordTimes,dataQueue,))
        self.threadsManager['record'].start()

    def read(self,wavFile,dataQueue=None):
        #Input raw data frm wav file

        if not self.isSafeState:
            raise WrongOperation('Please run under safe state by using <with> grammar')
            #print('Running without safe mode. Please use exit() function finally to ensure tasks safely')
                
        assert isinstance(wavFile,str),'{} do not like a wav file'.format(wavFile)

        if not os.path.isfile(wavFile):
            raise Exception('No such a file:{}'.format(wavFile))
        
        if 'read' in self.threadsManager.keys() and self.threadsManager['read'].is_alive():
            raise WrongOperation('Another read thread is running now')

        if 'record' in self.threadsManager.keys() and self.threadsManager['record'].is_alive():
            raise WrongOperation('Can not run record and read thread at the same time.')

        if dataQueue == None:
            dataQueue = self.dataQueue

        def readData(wavFile,dataQueue):
            try:
                wf = wave.open(wavFile,'rb')
                wfRate = wf.getframerate()
                wfChannels = wf.getnchannels()
                if wf.getsampwidth() == 1:
                    wfFormat = 'int8'
                elif wf.getsampwidth() == 2:
                    wfFormat = 'int16'
                elif wf.getsampwidth() == 4:
                    wfFormat = 'int32'
                else:
                    raise WrongOperation("Now we do not accept format which is either int8, int16 or int32")
                self.setWaveFormat(wfFormat,wfChannels,wfRate)
                
                secPerRead = self.chunkFrames / self.rate
                
                firstMessage = "{},{},{},{}".format(self.formats,self.channels,self.rate,self.chunkFrames)
                #pad it
                firstMessage = firstMessage + " "*(32-len(firstMessage))
                dataQueue.put(firstMessage.encode())
                data = wf.readframes(self.chunkFrames)
                while len(data) == self.chunkSize:
                    self._counter += 1
                    if self.endFlag == True:
                        break
                    if self.errFlag == True:
                        raise Exception('Close Read thread')
                    dataQueue.put(data)
                    #time.sleep(secPerRead)

                    data = wf.readframes(self.chunkFrames)
                #When last chunk data size is smaller than chunkSize, pad it
                if data != b'':
                    lastChunkData = data + b'0'*(self.chunkSize-len(data))
                    dataQueue.put(lastChunkData)
                    self._counter += 1
            except Exception as e:
                dataQueue.put('errFlag')
                self.errFlag = True
                raise e
            else:
                dataQueue.put('endFlag')
            finally:

                self.threadsManager['read'] = None

        self.threadsManager['read'] = threading.Thread(target=readData,args=(wavFile,dataQueue,))
        self.threadsManager['read'].start()        
        
    def send(self,dataQueue=None):

        if not self.isSafeState:
            raise WrongOperation('Please run under safe state by using with grammar')
            #print('Running without safe mode. Please use exit() function finally to ensure tasks safely')

        if 'send' in self.threadsManager.keys() and self.threadsManager['send'].is_alive():
            raise WrongOperation('Another send thread is running now')

        if self.client == None:
            raise WrongOperation('No activated remote client')

        if dataQueue == None:
            dataQueue = self.dataQueue

        def sendData(dataQueue):
            try:
                #Note that wave formats will be sent as first message in order to set them at server 
                while True:
                    if self.errFlag == True:
                        raise Exception('Close send thread')
                    elif dataQueue.empty():
                        time.sleep(0.01)
                    else:
                        message = dataQueue.get()
                        if 'endFlag' == message:
                            break
                        elif 'errFlag' == message: #This code has the same ability with self.errFlag to interupt send thread
                            raise Exception('Close send thread')
                        elif self.proto == 'TCP':
                            self.client.send(message)
                        else:
                            self.client.sendto(message,(self.targetHost,self.targetPort))
            except Exception as e:
                self.errFlag = True
                #Tell server the error-state, server can interupt its threads by this sign
                if self.proto == 'TCP':
                    self.client.send(('errFlag'+" "*(self.chunkSize-7)).encode())
                else:
                    self.client.sendto(('errFlag'+" "*(self.chunkSize-7)).encode(),(self.targetHost,self.targetPort))
                raise e
            else:
                #Tell the server to stop rightly
                if self.proto == 'TCP':
                    self.client.send(('endFlag'+" "*(self.chunkSize-7)).encode())
                else:
                    self.client.sendto(('endFlag'+" "*(self.chunkSize-7)).encode(),(self.targetHost,self.targetPort))
            finally:
                self.threadsManager['send'] = None
            
        self.threadsManager['send'] = threading.Thread(target=sendData,args=(dataQueue,))
        self.threadsManager['send'].start()       

    def receive(self,resultQueue=None):

        if not self.isSafeState:
            raise WrongOperation('Please run under safe state by using with grammar')
            #print('Running without safe mode. Please use exit() function finally to ensure tasks safely')

        if 'receive' in self.threadsManager.keys() and self.threadsManager['receive'].is_alive():
            raise WrongOperation('Another receive thread is running now')

        if self.client == None:
            raise WrongOperation('No activated local client')

        if resultQueue == None:
            resultQueue = self.resultQueue

        def recvData(resultQueue):
            try:
                while True:
                    #Recvive thread is last member so it's no necessary to put error sign in the last of result queue
                    if self.errFlag == True:
                        print('Close receive thread')
                        break
                    try:
                        if self.proto == 'TCP':
                            message = self.client.recv(self.chunkSize)
                        else:
                            i = 0
                            while i < 5:
                                message,addr = self.client.recvfrom(self.chunkSize)
                                if addr == (self.targetHost,self.targetPort):
                                    break
                            if i == 5:
                                raise WrongOperation('Net communicate anomaly')
                    except TimeoutError:
                        raise WrongOperation("Net commucate anomaly")
                    else:
                        message = message.decode().strip()
                        #When server have error, 'errFlag' will be received
                        if 'endFlag' in message:
                            break
                        elif 'errFlag' in message:
                            raise Exception('Close receive thread')
                        elif message == "":
                            continue
                        else:
                            resultQueue.put(message)
            except Exception as e:
                print('Err Message:',message)
                self.errFlag = True
                raise e
            finally:
                resultQueue.put('endFlag')
                self.threadsManager['receive'] = None
            
        self.threadsManager['receive'] = threading.Thread(target=recvData,args=(resultQueue,))
        self.threadsManager['receive'].start()

class RemoteServer(object):

    def __init__(self,proto='TCP',bindHost=None,bindPort=9509):

        raise WrongOperation('I am sorry that Speek Client Section is unable to use now.')

        assert (proto == 'TCP' or proto == 'UDP'),'Expected proto TCP or UDP but got {}'.format(proto)
        assert bindHost != None, 'Expected bind host IP address' 

        self.client = None
        self.dataQueue = queue.Queue()
        self.resultQueue = queue.Queue()
        self.threadsManager = {}
        self.errFlag = False
        self.isSafeState = False

        socket.setdefaulttimeout(10)

        if proto == 'TCP':
            self.server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        else:
            self.server = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.proto = proto
        self.server.bind((bindHost,bindPort))

    def _setWaveFormat(self,Format='int16',Channels=1,Rate=16000,ChunkFrames=1024):
        #It is no necessary to set wav format manually,
        #because it will be set automatically when receive the first message
        
        assert Format=='int8' or Format=='int16' or Format=="int32","Expected format int16 or int32 but got{}".format(Format)
        assert Channels==1 or Channels==2,"Expected channels 1 or 2 but got {}".format(Channels)

        self.chunkFrames = ChunkFrames
        self.rate = Rate
        if Format=='int8':
            self.sampleWidth = 1
        elif Format=='int16':
            self.sampleWidth = 2
        else:
            self.sampleWidth = 4
        self.channels = Channels
        self.chunkSize = self.sampleWidth*Channels*ChunkFrames
    
    def __enter__(self,*args):
        self.isSafeState = True
        return self
    
    def __exit__(self,errType,errValue,errTrace):
        #if mian thread stopped by error, tell other jobs to stop 
        if errType != None:
            self.errFlag = True
        else:
            self.errFlag = False
        self.wait()
        if self.client != None:
            time.sleep(0.1)
            self.client.close()
    
    def connectFrom(self,targetHost=None):

        if not self.isSafeState:
            raise WrongOperation('Please run under safe state by using <with> grammar')
            #print('Running without safe mode. Please use exit() function finally to ensure tasks safely')

        if self.client != None:
            raise WrongOperation('Another client is running right now')

        print('Awaiting connect...')
        try:
            i = 0
            while i < 5:
                if self.proto == 'TCP':
                    self.server.listen(1)
                    self.client,addr = self.server.accept()
                    if addr[0] == targetHost:
                        print('Connected from {}'.format(targetHost))
                        self.client.send(b'hello world')
                        self.targetAddr = addr
                        break
                    else:
                        self.client.close()
                else:
                    identifiedMessage,addr = self.server.recvfrom(32)
                    if identifiedMessage == b'hello world' and addr[0] == targetHost:
                        print('Connected from {}'.format(targetHost))
                        self.client = self.server
                        self.client.sendto(b'hello world',addr)
                        self.targetAddr = addr
                        break
                i += 1
        except socket.timeout:
            raise WrongOperation('No any connect application')
        else:
            if i >= 5:
                self.client = None
                raise WrongOperation("Cannot connect from {}".format(targetHost))

    def receive(self,dataQueue=None):

        if not self.isSafeState:
            raise WrongOperation('Please run under safe state by using <with> grammar')
            #print('Running without safe mode. Please use exit() function finally to ensure tasks safely')

        if 'receive' in self.threadsManager.keys() and self.threadsManager['receive'].is_alive():
            raise WrongOperation('Another receive thread is running now')

        if self.client == None:
            raise WrongOperation('No activated client')

        if dataQueue == None:
            dataQueue = self.dataQueue
        
        def recvData(dataQueue):
            try:
                #Receive first message to set wav format
                try: 
                    if self.proto == 'TCP':
                        identifiedMessage = self.client.recv(32)
                    else:
                        i = 0
                        while i < 5:
                            identifiedMessage,addr = self.client.recvfrom(32)
                            if addr == self.targetAddr:
                                break
                        if i == 5:
                            raise WrongOperation('Net communicate anomaly')
                    idMe = identifiedMessage.decode().strip().split(',')
                    #Set wave format and chunkSize
                    self._setWaveFormat(idMe[0],int(idMe[1]),int(idMe[2]),int(idMe[3]))
                except socket.timeout:
                    raise Exception('Net communicate anomaly')                
                else:
                    while True:
                        if self.errFlag == True:
                            raise Exception('Close receive thread')
                        try:
                            if self.proto == 'TCP':
                                message = self.client.recv(self.chunkSize)
                            else:
                                while True:
                                    if self.errFlag == True:
                                        raise Exception('Close receive thread')
                                    message,addr = self.client.recvfrom(self.chunkSize)
                                    if addr == self.targetAddr:
                                        break 
                        except socket.timeout:
                            raise Exception('Communication with client anomaly')
                        else:
                            if b'endFlag' in message:
                                break
                            elif b'errFlag' in message:
                                raise Exception('Close receive thread')
                            else:
                                dataQueue.put(message)
            except Exception as e:
                dataQueue.put('errFlag')
                self.errFlag = True
                raise e
            else:
                dataQueue.put('endFlag')
            finally:
                self.threadsManager['receive'] = None

        self.threadsManager['receive'] = threading.Thread(target=recvData,args=(dataQueue,))
        self.threadsManager['receive'].start()
        
    def recognize(self,recoFunc,recoFuncArgs=None,dataQueue=None,resultQueue=None,sodFunc=None,sodFuncArgs=None,secPerReco=0.2):
        
        if not self.isSafeState:
            raise WrongOperation('Please run under safe state by using <with> grammar')
            #print('Running without safe mode. Please use exit() function finally to ensure tasks safely')

        if 'recognize' in self.threadsManager.keys() and self.threadsManager['recognize'].is_alive():
            raise WrongOperation('Other recognize thread is running now')

        if dataQueue == None:
            dataQueue = self.dataQueue

        if resultQueue == None:
            resultQueue = self.resultQueue

        #Embeded voice end detection function
        class DefaultSodFunc(object):
            def __init__(self,maxAlikeTimes=5):
                self.lastResult = None
                self.lastAlikeScore = 0
                self.maxAlikeTimes = maxAlikeTimes
            def __call__(self,result,*args):
                result = result.strip()
                if result == self.lastResult:
                    self.lastAlikeScore += 1
                    if self.lastAlikeScore >= self.maxAlikeTimes:
                        self.lastResult = None
                        self.lastAlikeScore = 0
                        return True
                else:
                    self.lastResult = result
                    self.lastAlikeScore = 0
                return False

        if sodFunc == None:
            sodFunc = DefaultSodFunc(int(0.6/secPerReco))
        
        def dataReco(dataQueue,recoFunc,recoFuncArgs,resultQueue,sodFunc,sodFuncArgs):
        
            dataPerReco = []
            count = 0
            timesPerReco = None
            try:
                while True:
                    if self.errFlag == True:
                        raise Exception('Close recognize thread')
                    elif dataQueue.empty():
                        time.sleep(0.01)
                    else:
                        if timesPerReco == None:
                            timesPerReco = math.ceil(self.rate * secPerReco/self.chunkFrames)
                        chunkData = dataQueue.get()
                        if 'errFlag' == chunkData:
                            raise Exception('close recognize thread')
                        elif 'endFlag' == chunkData:
                            if len(dataPerReco) == 0:
                                break
                            else:
                                #Last batch data
                                count = timesPerReco + 1
                        elif 'errFlag' == chunkData:
                            raise Exception('Close recognize thread')
                        else:
                            dataPerReco.append(chunkData)
                            count += 1
                        if count >= timesPerReco:
                            start = time.time()
                            with tempfile.NamedTemporaryFile('w+b',suffix='.wav') as waveFile:
                                wf = wave.open(waveFile.name, 'wb')
                                wf.setsampwidth(self.sampleWidth)
                                wf.setnchannels(self.channels)
                                wf.setframerate(self.rate)
                                wf.writeframes(b''.join(dataPerReco))
                                wf.close()
                                recoResult = recoFunc(waveFile.name,recoFuncArgs)
                                if count > timesPerReco:
                                    sectionOverFlag = True
                                else:
                                    sectionOverFlag = sodFunc(recoResult,sodFuncArgs)
                                final = time.time()
                                assert isinstance(sectionOverFlag,bool),"Expected sod function return True or False,but got {}".format(type(sectionOverFlag))
                                print("DECODE-TIME:%.5f"%(final-start)," SECTION-END:{}".format(sectionOverFlag)," RESULT:{}".format(recoResult))
                                resultQueue.put((sectionOverFlag,recoResult))
                                if count > timesPerReco:
                                    break
                                elif sectionOverFlag:
                                    dataPerReco = []
                                count = 0

            except Exception as e:
                resultQueue.put('errFlag')
                self.errFlag = True
                raise e
            else:
                resultQueue.put('endFlag')
            finally:
                self.threadsManager['recognize'] = None

        self.threadsManager['recognize'] = threading.Thread(target=dataReco,args=(dataQueue,recoFunc,recoFuncArgs,resultQueue,sodFunc,sodFuncArgs,))
        self.threadsManager['recognize'].start()

    def recognizeParallel(self,recoFunc,recoFuncArgs=None,dataQueue=None,resultQueue=None,secPerReco=0.3,maxThreads=3,silSymbol='<sil>'):
        #In parallel mode, we don't support use other sodfuctions

        if not self.isSafeState:
            raise WrongOperation('Please run under safe state by using <with> grammar')
            #print('Running without safe mode. Please use exit() function finally to ensure tasks safely')

        if 'recognize' in self.threadsManager.keys() and self.threadsManager['recognize'].is_alive():
            raise WrongOperation('Another recognize thread is running now')

        if dataQueue == None:
            dataQueue = self.dataQueue

        if resultQueue == None:
            resultQueue = self.resultQueue

        self.recoThreadsManager = {}
        for threadId in range(maxThreads):
            self.recoThreadsManager[threadId] = None

        class DefaultSodFunc(object):
            def __init__(self,maxAlikeTimes=3,startId=0,silSymbol=None):
                if maxAlikeTimes < 2:
                    maxAlikeTimes = 2
                elif maxAlikeTimes > 5:
                    maxAlikeTimes = 5
                self.lastResult = None
                self.alikeScore = 0
                self.maxAlikeTimes = maxAlikeTimes
                self.expectId = startId
                self.lock = threading.Lock()
                self.silSymbol = silSymbol
                
            def __call__(self,result):
                #If it is right order data, score it
                #if dataId == self.expectId:
                self.expectId += 1
                result = result.strip()
                if self.silSymbol != None:
                    result == self.silSymbol
                    return True
                elif result == self.lastResult:
                    self.alikeScore += 1
                    if self.alikeScore >= self.maxAlikeTimes:
                        self.lastResult = None
                        self.alikeScore = 0
                        return True
                else:
                    self.lastResult = result
                    self.alikeScore = 0
                return False
                #else:
                #    #Else, tell thread to wait until another thread brought the right order data 
                #    return 'wait'
            
            def askExpectId(self):
                return self.expectId

            def acquireLock(self):
                self.lock.acquire()

            def releaseLock(self):
                self.lock.release()

        #Recognize thread function
        def dataReco(threadId,dataId,data,recoFunc,recoFuncArgs,sodFunc):
            try:
                with tempfile.NamedTemporaryFile('w+b',suffix='.wav') as waveFile:
                    wf = wave.open(waveFile.name, 'wb')
                    wf.setsampwidth(self.sampleWidth)
                    wf.setnchannels(self.channels)
                    wf.setframerate(self.rate)
                    wf.writeframes(b''.join(data))
                    wf.close()
                    recoResult = recoFunc(threadId,waveFile.name,*recoFuncArgs)
                    """
                    while True:
                        expectId = sodFunc.askExpectId()
                        if dataId == expectId:
                            sodFunc.acquireLock()
                            sectionOverFlag = sodFunc(recoResult)
                            sodFunc.releaseLock()
                            break
                        else:
                            time.sleep(0.01)
                    """
                    sectionOverFlag = False
                    self.recoThreadsManager[threadId] = (dataId,sectionOverFlag,recoResult)

            except Exception as e:
                self.recoThreadsManager[threadId] = 'errFlag'
                raise e

        def dataBuild(dataQueue,recoFunc,recoFuncArgs,resultQueue,secPerReco,silSymbol):
        
            batchData = []
            recoData = []

            timesPerReco = None 
            count = 0

            threadIds = list(self.recoThreadsManager.keys())

            batchDataBackup = {}
            resultBackup = {}            

            trash = []

            sodFunc = DefaultSodFunc(int(1/secPerReco),0,silSymbol)

            finalBatchData = False

            dataBatchId = 0
            expectBatchId = 0

            try:
                while True:
                    if self.errFlag == True:
                        raise Exception('Close recognize thread')
                    elif dataQueue.empty():
                        time.sleep(0.01)
                    else:
                        if timesPerReco == None:
                            timesPerReco = math.ceil(self.rate * secPerReco/self.chunkFrames)
                        chunkData = dataQueue.get()
                        if 'errFlag' == chunkData:
                            raise Exception('Close recognize thread')
                        elif 'endFlag' == chunkData:
                            finalBatchData = True
                        else:
                            batchData.append(chunkData)
                            count += 1

                        if count >= timesPerReco or finalBatchData == True:
                            
                            if len(batchData) > 0:

                                print("Prepared data {} done. Search a thread avaliable to rocognize it".format(dataBatchId))

                                recoData.extend(batchData)
                                batchDataBackup[dataBatchId] = batchData
                                batchData = []
                                
                                while True:

                                    threadId = 0
                                    while threadId < len(threadIds):
                                        
                                        if self.recoThreadsManager[threadId] == None:
                                            print('Found thread {} is avaliable. Use it.'.format(threadId))
                                            self.recoThreadsManager[threadId] = threading.Thread(target=dataReco,args=(threadId,dataBatchId,recoData,recoFunc,recoFuncArgs,sodFunc,))
                                            self.recoThreadsManager[threadId].start()
                                            count = 0
                                            dataBatchId += 1
                                            break

                                        elif isinstance(self.recoThreadsManager[threadId],threading.Thread):
                                            threadId += 1
                                            continue

                                        elif isinstance(self.recoThreadsManager[threadId],tuple):
                                            dataId = self.recoThreadsManager[threadId][0]
                                            sectionOverFlag = self.recoThreadsManager[threadId][1]
                                            recoResult = self.recoThreadsManager[threadId][2]
                                            print('Take over thread {}. Analyse the result(Data id {}).'.format(threadId,dataId))
                                            print('Recognize:',recoResult)
                                            
                                            if expectBatchId == dataId:
                                                print('Data id is expected. Save the result.')
                                                resultQueue.put((sectionOverFlag,recoResult))
                                                batchDataBackup.pop(dataId)
                                                expectBatchId += 1

                                                if sectionOverFlag == True:
                                                    print('Speak-Endding: True. Clear all caches and abandon other running recognize-threads')
                                                    #First: put all threads and results into trash (for safety)
                                                    for i in threadIds:
                                                        trash.append(self.recoThreadsManager[i])
                                                        self.recoThreadsManager[i] = None
                                                    #Second: Reset sodFunction in order to avoid other running trash threads would change the parameters of old sodFunc 
                                                    sodFunc = DefaultSodFunc(int(1/secPerReco),expectBatchId,silSymbol)
                                                    #Third:Clear all data in recoData
                                                    recoData = []
                                                    #Fourth: If there are data need to be recover, recover them and then clear batchDataBackup
                                                    for recoverDataId in sorted(batchDataBackup.keys()):
                                                        if recoverDataId > dataId:
                                                            recoData.extend(batchDataBackup[recoverDataId])
                                                    count = len(recoData)
                                                    batchDataBackup = {}
                                                    #Fifth: Adjust dataBatchId coresponding to current id
                                                    print('Recover data id to from {} to {}'.format(dataBatchId,dataId+1))
                                                    dataBatchId = dataId + 1
                                                    #Sixth: Clear resultBackup
                                                    resultBackup = {}
                                                    
                                                else:
                                                    print('Speak-Endding: False. Use thread {} to process data {}'.format(threadId,dataBatchId))
                                                    self.recoThreadsManager[threadId] = threading.Thread(target=dataReco,args=(threadId,dataBatchId,recoData,recoFunc,recoFuncArgs,sodFunc,))
                                                    self.recoThreadsManager[threadId].start()
                                                    count = 0
                                                    dataBatchId += 1
                                                
                                                break

                                            elif expectBatchId in resultBackup.keys():
                                                print('Data id wrong but found expected one in result-backup. Save it.')
                                                resultQueue.put(resultBackup[expectBatchId])
                                                batchDataBackup.pop(expectBatchId)
                                                expectBatchId += 1

                                                if resultBackup[expectBatchId-1][0] == True:
                                                    print('Speak-Endding: True. Clear all caches and abandon other running recognize-threads')
                                                    #First: put all threads and results into trash (for safety)
                                                    for i in threadIds:
                                                        trash.append(self.recoThreadsManager[i])
                                                        self.recoThreadsManager[i] = None
                                                    #Second: Reset sodFunction in order to avoid other running trash threads would change the parameters of old sodFunc 
                                                    sodFunc = DefaultSodFunc(int(1/self.secPerReco),expectBatchId,silSymbol)
                                                    #Third:Clear all data in recoData 
                                                    recoData = []
                                                    #Fourth: If there are data need to be recover, recover them and then clear batchDataBackup
                                                    for recoverDataId in sorted(batchDataBackup.keys()):
                                                        if recoverDataId >= expectBatchId:
                                                            recoData.extend(batchDataBackup[recoverDataId])
                                                            count += 1
                                                    batchDataBackup = {}
                                                    #Fifth: Adjust dataBatchId coresponding to current id
                                                    dataBatchId = expectBatchId
                                                    #Sixth: Clear resultBackup
                                                    resultBackup = {}

                                                else:
                                                    print('Speak-Endding: False. Save the result to result-backup then use thread {} to process data {}'.format(threadId,dataBatchId))
                                                    #Delete the backup
                                                    resultBackup.pop(expectBatchId-1)
                                                    #Put the backup into resultBackup
                                                    resultBackup[dataId] = (sectionOverFlag,recoResult)
                                                    # Let this threadId to deal with current recoData
                                                    self.recoThreadsManager[threadId] = threading.Thread(target=dataReco,args=(threadId,dataBatchId,recoData,recoFunc,recoFuncArgs,sodFunc,))
                                                    self.recoThreadsManager[threadId].start()
                                                    #Clear
                                                    count = 0
                                                    #get the id of next batch data
                                                    dataBatchId += 1

                                                break

                                            else:
                                                print('Data id wrong and also not in result-backup. Save result to result-backup temporarily')
                                                print('Then use thread {} to process data {}'.format(threadId,dataBatchId))
                                                resultBackup[dataId]=(sectionOverFlag,recoResult)
                                                self.recoThreadsManager[threadId] = threading.Thread(target=dataReco,args=(threadId,dataBatchId,recoData,recoFunc,recoFuncArgs,sodFunc,))
                                                self.recoThreadsManager[threadId].start()
                                                count = 0
                                                dataBatchId += 1

                                                break
                                        
                                        else:
                                            raise Exception('Recognizing Thread {} Error'.format(threadId))

                                    if  threadId < len(threadIds):
                                        break
                                    else:
                                        #Indicate all threads are running now, so wait util any theard over
                                        time.sleep(0.01)
                                        continue

                            # If now arrive the end of data
                            if finalBatchData == True:

                                print('Process the last batch of tasks. Go through all threads one by one')

                                lastMessage = None
                                #Check all threadIds one by one
                                for threadId in self.recoThreadsManager.keys():

                                    if self.recoThreadsManager[threadId] == None:
                                        print('Thread {}: Over'.format(threadId))
                                        continue

                                    #If reco thread in running, wait it untill over
                                    elif isinstance(self.recoThreadsManager[threadId],threading.Thread):
                                        print('Thread {}: Running. Wait it.'.format(threadId))
                                        self.recoThreadsManager[threadId].join()
                                    
                                    print('Analyse the result of thread {}'.format(threadId))
                                    dataId = self.recoThreadsManager[threadId][0]
                                    sectionOverFlag = self.recoThreadsManager[threadId][1]
                                    recoResult = self.recoThreadsManager[threadId][2]

                                    lastMessage = recoResult

                                    if dataId == expectBatchId:
                                
                                        print('Data id is expected. Save the result')
                                        #Save it to result queue and remove the backup in batchDataBackup
                                        resultQueue.put((sectionOverFlag,recoResult))
                                        batchDataBackup.pop(dataId)
                                        expectBatchId += 1

                                        if sectionOverFlag == True:
                                            print('Speak-Endding: True. Clear all caches and abandon other running recognize-threads')
                                            #First: put all threads and results into trash (for safety)
                                            for i in threadIds:
                                                trash.append(self.recoThreadsManager[i])
                                                self.recoThreadsManager[i] = None
                                            #Second: Reset sodFunction in order to avoid other running trash threads would change the parameters of old sodFunc 
                                            sodFunc = lambda x,y:True
                                            #Third:Clear all data in recoData 
                                            recoData = []
                                            #Fourth: If there are data need to be recover, recover them and then clear batchDataBackup
                                            for recoverDataId in sorted(batchDataBackup.keys()):
                                                if recoverDataId > dataId:
                                                    recoData.extend(batchDataBackup[recoverDataId])
                                            #Fifth: Adjust dataBatchId coresponding to current id
                                            dataBatchId = dataId + 1
                                            #Appoint threadId 0 to deal with recover Data
                                            resultBackup = {}
                                            if len(recoData) > 0:
                                                print('Found data left. Process it now')
                                                self.recoThreadsManager[0] = threading.Thread(target=dataReco,args=(0,dataBatchId,recoData,recoFunc,recoFuncArgs,sodFunc,))
                                                self.recoThreadsManager[0].start()
                                                self.recoThreadsManager[0].join()
                                                lastMessage = self.recoThreadsManager[0][2]
                                                break
                                        else:
                                            print('Speak-Endding: False. Go through the next thread')
                                            continue

                                    elif expectBatchId in resultBackup.keys():
                                        print('Data id wrong but found expected one in result-backup. Save it.')
                                        resultQueue.put(resultBackup[expectBatchId])
                                        batchDataBackup.pop(expectBatchId)
                                        expectBatchId += 1

                                        if resultBackup[expectBatchId-1][0] == True:
                                            print('Speak-Endding: True. Clear all caches and abandon other running recognize-threads')
                                            #First: put all threads and results into trash (for safety)
                                            for i in threadIds:
                                                trash.append(self.recoThreadsManager[i])
                                                self.recoThreadsManager[i] = None
                                            #Second: Reset sodFunction in order to avoid other running trash threads would change the parameters of old sodFunc 
                                            sodFunc = lambda x,y:True
                                            #Third:Clear all data in recoData 
                                            recoData = []
                                            #Fourth: If there are data need to be recover, recover them and then clear batchDataBackup
                                            for recoverDataId in sorted(batchDataBackup.keys()):
                                                if recoverDataId >= expectBatchId:
                                                    recoData.extend(batchDataBackup[recoverDataId])
                                            #Fifth: Adjust dataBatchId coresponding to current id
                                            dataBatchId = expectBatchId
                                            resultBackup = {}
                                            #Appoint threadId 0 to deal with recover Data
                                            if len(recoData) > 0:
                                                print('Found data left. Process it now')
                                                self.recoThreadsManager[0] = threading.Thread(target=dataReco,args=(0,dataBatchId,recoData,recoFunc,recoFuncArgs,sodFunc,))
                                                self.recoThreadsManager[0].start()
                                                self.recoThreadsManager[0].join()
                                                lastMessage = self.recoThreadsManager[0][2]
                                                break
                                        else:
                                            print('Speak-Endding: False. Save result and go through the next thread.')
                                            resultBackup.pop(expectBatchId-1)
                                            resultBackup[dataId] = (sectionOverFlag,recoResult)
                                            self.recoThreadsManager[threadId] = None
                                            continue
                                    else:
                                        print('Data id wrong and also not in result-backup. Save result to result-backup temporarily')
                                        resultBackup[dataId]=(sectionOverFlag,recoResult)
                                        self.recoThreadsManager[threadId] = None
                                        continue

                                print('Check result-backup finally')
                                for i in sorted(resultBackup.keys()):
                                    print('Save data {}'.format(i))
                                    resultQueue.put(resultBackup[i])
                                    lastMessage = resultBackup[i][1]

                                resultQueue.put((True,lastMessage))

                                print('Recognize-tasks over rightly')

                                break #Here, All tasks over

            except Exception as e:
                resultQueue.put('errFlag')
                self.errFlag = True
                raise e
            else:
                resultQueue.put('endFlag')
            finally:
                del trash
                for threadId in self.recoThreadsManager.keys():
                    if isinstance(self.recoThreadsManager[threadId],threading.Thread):
                        self.recoThreadsManager[threadId].join()
                    else:
                        self.recoThreadsManager[threadId] = None
                self.threadsManager['recognize'] = None

        self.threadsManager['recognize'] = threading.Thread(target=dataBuild,args=(dataQueue,recoFunc,recoFuncArgs,resultQueue,secPerReco,silSymbol,))
        self.threadsManager['recognize'].start()
  
    def send(self,resultQueue=None):

        if not self.isSafeState:
            raise WrongOperation('Please run under safe state by using <with> grammar')
            #print('Running without safe mode. Please use exit() function finally to ensure tasks safely')

        if 'send' in self.threadsManager.keys() and self.threadsManager['send'].is_alive():
            raise WrongOperation('Another send thread is running now')

        if self.client == None:
            raise WrongOperation('No activated server client')

        if resultQueue == None:
            resultQueue = self.resultQueue

        def sendData(resultQueue):
            try:
                while True:
                    if self.errFlag == True:
                        raise Exception('Close send thread')
                    if resultQueue.empty():
                        time.sleep(0.01)
                    else:
                        message = resultQueue.get()
                        print('Result Length:',message[1])
                        if 'endFlag' == message:
                            break
                        elif 'errFlag' == message:
                            raise Exception('Close send thread')
                        else:
                            #message has a format: (result,sectionOverFlag)
                            data = ''
                            #we will send data with a format: sectionOverFlag(Y/N) + ' ' + resultData + placeHolderSymbol(' ')
                            rSize = self.chunkSize - 2  # -len("Y ")
                            #result is likely longer than rSize, so cut it
                            #Get the N front rSize part, give them all 'Y' sign   
                            i = 0
                            while i < len(data)/rSize:
                                data += 'Y '
                                data += message[1][i*rSize:(i+1)*rSize]
                                i += 1
                            #Deal with the rest part 
                            if message[0]:
                                data += 'Y '
                            else:
                                data += 'N '
                            data += message[1][i*rSize:]
                            #Pad it with " "
                            data += ' '*((i+1)*self.chunkSize - len(data))
                            if self.proto == 'TCP':
                                self.client.send(data.encode())
                            else:
                                self.client.sendto(data.encode(),self.targetAddr)
            except Exception as e:
                self.errFlag = True
                if self.proto == 'TCP':
                    self.client.send(('errFlag'+" "*(self.chunkSize-7)).encode())
                else:
                    self.client.sendto(('errFlag'+" "*(self.chunkSize-7)).encode(),self.targetAddr)
                raise e
            else:
                if self.proto == 'TCP':
                    self.client.send(('endFlag'+" "*(self.chunkSize-7)).encode())
                else:
                    self.client.sendto(('endFlag'+" "*(self.chunkSize-7)).encode(),self.targetAddr)
            finally:
                self.threadsManager['send'] = None
            
        self.threadsManager['send'] = threading.Thread(target=sendData,args=(resultQueue,))
        self.threadsManager['send'].start()              

    def wait(self):
        for name in self.threadsManager.keys():
            if self.threadsManager[name] != None and self.threadsManager[name].is_alive():
                self.threadsManager[name].join()

##### Chainer Model #####

import chainer
import chainer.functions as F
import chainer.links as L

class LayerNorm(chainer.Chain):

    def __init__(self, input_dim, eps=1e-6):
        super(LayerNorm,self).__init__()
        with self.init_scope():
            self.gamma = chainer.Parameter(np.ones(input_dim,dtype=np.float32))
            self.beta = chainer.Parameter(np.zeros(input_dim,dtype=np.float32))
            self.eps = eps

    def forward(self, x):
        mean = chainer.Variable(x.array.mean(keepdims=True),requires_grad=False)
        std = chainer.Variable(x.array.std(keepdims=True),requires_grad=False)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class Acfunction(chainer.Chain):
    def __init__(self,acfun_type):
        super(Acfunction,self).__init__()
        with self.init_scope():
            if acfun_type == None:
                self.acfun = lambda x : x
            else:
                acfun_type = acfun_type.lower()
                if acfun_type=="relu":
                    self.acfun = F.relu
                elif acfun_type=="tanh":
                    self.acfun = F.tanh      
                elif acfun_type=="sigmoid":
                    self.acfun = F.sigmoid      
                elif acfun_type=="leaky_relu":
                    self.acfun = F.leaky_relu      
                elif acfun_type=="elu":
                    self.acfun = F.elu      
                elif acfun_type=="log_softmax":
                    self.acfun = F.log_softmax
                elif acfun_type=="softmax":
                    self.acfun = F.softmax
                elif acfun_type == 'none':
                    self.acfun = lambda x : x
                else:
                    raise WrongOperation('No such activation function:{}'.format(acfun_type))

    def __call__(self,x):
        return self.acfun(x)

class Dropout(chainer.Chain):
    def __init__(self,ratio):
        super(Dropout,self).__init__()
        with self.init_scope():
            self.ratio = ratio
    def __call__(self,x):
        return F.dropout(x,self.ratio)

class MLP(chainer.ChainList):
    '''
    Useage: model = MLP() or model = MLP(config=config)

    Get a chainer MLP model. If <config> is None, use default configure. Or you can initialize it by setting <config>. 
    Try to use pythonkaldi.check_config('MLP') function to get configure information you could set.

    ''' 
    def __init__(self,config=None):
        super(MLP, self).__init__()
        with self.init_scope():
            
            if config == None:
                config = check_config('MLP')
            else:
                check_config('MLP',config)

            if config['layernorm_in']:
                self.add_link(LayerNorm(config['inputdim']))
            if config['batchnorm_in']:
                self.add_link(L.BatchNormalization(config['inputdim'],decay=0.95))
            
            self.layers=len(config['node'])

            for i in range(self.layers):

                if config['layernorm'][i] or config['batchnorm'][i]:
                    self.add_link(L.Linear(None, config['node'][i], nobias=True, initialW=chainer.initializers.HeNormal()))

                    if config['layernorm'][i]:
                        self.add_link(LayerNorm(config['node'][i]))

                    if config['batchnorm'][i]:
                        self.add_link(L.BatchNormalization(config['node'][i],decay=0.95))
                else:

                    self.add_link(L.Linear(None, config['node'][i], nobias=False, initialW=chainer.initializers.HeNormal(), initial_bias=chainer.initializers.Zero()))

                if config['acfunction'][i] != None and config['acfunction'][i].lower() != 'none': 
                    self.add_link(Acfunction(config['acfunction'][i]))

                if config['dropout'][i] != None and float(config['dropout'][i]) != 0.: 
                    self.add_link(Dropout(config['dropout'][i]))

            self.config = config

    def forward(self, x):
        
        for link in self.children():
            x = link(x)
        
        return x

    def __str__(self):
        print('MLP')
        return self.config

class LSTM(chainer.ChainList):
    '''
    Useage: model = LSTM() or model = LSTM(config=config)

    Get a chainer LSTM model. If <config> is None, use default configure. Or you can initialize it by setting <config>. 
    Try to use pythonkaldi.check_config('LSTM') function to get configure information you could set.

    '''   
    def __init__(self,config=None):
        super(LSTM, self).__init__()
        with self.init_scope():
            
            if config == None:
                config = check_config('LSTM')
            else:
                check_config('LSTM',config)

            # input layer normalization
            if config['layernorm_in']:
                self.add_link(LayerNorm(config['inputdim']))
            
            # input batch normalization    
            if config['batchnorm_in']:
                self.add_link(L.BatchNormalization(config['inputdim'],decay=0.95))

            self.add_link(L.NStepBiLSTM(config['layers'],config['inputdim'],config['outputdim'],config['dropout']))

            if config['acfunction_out'] != None and config['acfunction_out'].lower() != 'none': 
                self.add_link(Acfunction(config['acfunction_out']))
            if config['dropout_out'] != None and float(config['dropout_out']) != 0.:     
                self.add_link(Dropout(config['dropout_out']))

            self.config = config

    def __call__(self,x):

        for link in self.children():
            x = link(x)
        
        return x

    def __str__(self):
        print("LSTM")
        return self.config

class DataIterator(chainer.iterators.SerialIterator):
    '''
    Useage: obj = DataIterator(data,64) or obj = DataIterator('train.scp',64,chunks='auto',processFunc=function)

    This is a imporved data interator. You can not only use it as ordinary chainer.iterators.SerialIterator, but also try its distinctive ability. 
    If you give it a large scp file of train data, it will split it into n smaller chunks and load them into momery alternately with parallel thread. 
    It will shuffle the original scp file and split again while new epoch.

    '''
    def __init__(self,dataOrScpFiles,batchSize,chunks='auto',processFunc=None,shuffle=True,labelOrAliFiles=None,hmmGmm=None,validDataRatio=0.1):

        self.datasetList = []
        self.fileProcessFunc = processFunc
        self._shuffle = shuffle
        self.labels = None
        self.batch_size = batchSize
        self.epoch_size = 0
        self.countEpochSizeFlag = True
        self.validFiles = []
        self.validDataRatio = validDataRatio
        self.next_datasetId = 0

        if isinstance(dataOrScpFiles,str):
            p = subprocess.Popen('ls {}'.format(dataOrScpFiles),shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            (out,err) = p.communicate()
            if out == b'':
                raise PathError("No such file:{}".format(dataOrScpFiles))
            else:
                out = out.decode().strip().split('\n')

                if isinstance(chunks,int):
                    assert chunks>1,"Expected chunks is int and >1"
                elif chunks != 'auto':
                    raise WrongOperation('Unuseful chunks parameter')
                assert processFunc != None, 'Expected from KaldiArk feat to numpy array transform method'

                self.chunks = chunks
                self.allFiles = []

                for scpFile in out:
                    with open(scpFile,'r') as fr:
                        self.allFiles.extend(fr.read().strip().split('\n'))

                K = int(len(self.allFiles)*(1-self.validDataRatio))
                self.validFiles = self.allFiles[K:]
                self.allFiles = self.allFiles[0:K]
                
                if chunks == 'auto':
                    
                    #Compute the chunks automatically
                    sampleChunk = random.sample(self.allFiles,10)
                    with tempfile.NamedTemporaryFile('w',suffix='.scp') as sampleFile:
                        sampleFile.write('\n'.join(sampleChunk))
                        sampleFile.seek(0)
                        sampleChunkData = load(sampleFile.name)
                    meanLength = int(np.mean(sampleChunkData.lens[1]))
                    autoChunkSize = math.ceil(30000/meanLength)  # Use 30000 frames as threshold 
                    self.chunks = len(self.allFiles)//autoChunkSize
                    if self.chunks == 0: 
                        self.chunks = 1

                chunkSize = math.ceil(len(self.allFiles)/self.chunks)

                L = self.chunks-(chunkSize*self.chunks-len(self.allFiles))-1
                start = 0
                for i in range(self.chunks):
                    if i > L:
                        end = start + chunkSize - 1
                    else:
                        end = start + chunkSize
                    chunkFiles = self.allFiles[start:end]
                    start = end
                    if len(chunkFiles) > 0:
                        self.datasetList.append(chunkFiles)

                if labelOrAliFiles != None:
                    if isinstance(labelOrAliFiles,KaldiDict):
                        self.labels = labelOrAliFiles
                    elif isinstance(labelOrAliFiles,KaldiArk):
                        self.labels = labelOrAliFiles.array
                    elif isinstance(labelOrAliFiles,str):
                        if hmmGmm == None:
                            tryPath = labelOrAliFiles[:labelOrAliFiles.rfind('/')]+'/final.mdl'
                            if os.path.isfile(tryPath):
                                hmmGmm = tryPath
                            else:
                                raise WrongOperation('Expected hmm-gmm model file')
                        self.labels = get_ali(labelOrAliFiles,hmmGmm)
                    else:
                        raise UnsupportedDataType('Expected Label is KaldiDict or KaldiArk or AliFile')
                
                self.load_dataset(datasetId=0)
                self.currentDataset = self.nextDataset
                self.nextDataset = None
        else:
            K = int(len(dataOrScpFiles)*self.validDataRatio)
            dataOrScpFiles = [X for X in dataOrScpFiles]
            self.validData = dataOrScpFiles[-K:]
            self.currentDataset = dataOrScpFiles[0:-K]

        self.epoch_size = len(self.currentDataset)

        self.epoch = 0
        self.current_position = 0
        self.current_epoch_position = 0
        self.is_new_epoch = False
        self.is_new_chunk = False
        self.loadDatasetThread = None

        if len(self.datasetList) > 1:
            self.next_datasetId = 1
            self.loadDatasetThread = threading.Thread(target=self.load_dataset,args=(self.next_datasetId,))
            self.loadDatasetThread.start()

    def load_dataset(self,datasetId,shuffle=False):

        if shuffle == True:
            self.datasetList = []
            random.shuffle(self.allFiles)
            chunkSize = math.ceil(len(self.allFiles)/self.chunks)
            L = self.chunks -(chunkSize * self.chunks - len(self.allFiles))-1
            start = 0
            for i in range(self.chunks):
                if i > L:
                    end = start + chunkSize - 1
                else:
                    end = start + chunkSize
                chunkFiles = self.allFiles[start:end]
                start = end
                if len(chunkFiles) > 0:
                    self.datasetList.append(chunkFiles)

        with tempfile.NamedTemporaryFile('w',suffix='.scp') as scpFile:
            scpFile.write('\n'.join(self.datasetList[datasetId]))
            scpFile.seek(0)
            chunkData = load(scpFile.name)  

        if self.labels != None:
            uttsList = chunkData.utts
            chunkLabel = self.labels.subset(uttList=uttsList)
            self.nextDataset = self.fileProcessFunc(chunkData,chunkLabel)
        else:
            self.nextDataset = self.fileProcessFunc(chunkData)

        if self.batch_size > len(self.nextDataset):
            print("Warning: Batch Size < {} > is extremely large for this dataset, we hope you can use a more suitable value.".format(self.batch_size))
        
        self.nextDataset = [X for X in self.nextDataset]

    def next(self):
        i = self.current_position
        i_end = i + self.batch_size
        N = len(self.currentDataset)

        batch = self.currentDataset[i:i_end]

        if len(self.datasetList) <= 1:
            if i_end >= N:
                rest = i_end - N
                if self._shuffle:
                    np.random.shuffle(self.currentDataset)
                batch.extend(self.currentDataset[:rest])
                self.current_position = rest
                self.current_epoch_position = self.current_position
                self.epoch += 1
                self.is_new_epoch = True
            else:
                self.is_new_epoch = False
                self.current_position = i_end
        else:
            if i_end >= N:
                rest = i_end - N
                while self.loadDatasetThread.is_alive():
                    time.sleep(0.1)
                if self._shuffle:
                    np.random.shuffle(self.nextDataset)
                batch.extend(self.nextDataset[:rest])
                self.current_position = rest
                self.currentDataset = self.nextDataset
                self.is_new_chunk = True
                
                if self.countEpochSizeFlag:
                    self.epoch_size += len(self.currentDataset)

                self.next_datasetId = (self.next_datasetId + 1)%len(self.datasetList)

                if self.next_datasetId == 1:
                    self.epoch += 1
                    self.is_new_epoch = True

                if self.next_datasetId == 0:
                    self.countEpochSizeFlag = False
                    self.loadDatasetThread = threading.Thread(target=self.load_dataset,args=(self.next_datasetId,True,))
                else:
                    self.loadDatasetThread = threading.Thread(target=self.load_dataset,args=(self.next_datasetId,False,))
                self.loadDatasetThread.start()

            else:
                self.is_new_chunk = False
                self.is_new_epoch = False
                self.current_position = i_end

        self.current_epoch_position = (self.current_epoch_position + self.batch_size)%self.epoch_size

        return batch                            

    @property
    def epoch_detail(self):
        return self.epoch + self.current_epoch_position/self.epoch_size
    
    @property
    def currentChunkId(self):
        if self.next_datasetId == 0:
            return self.chunks
        else:
            return self.next_datasetId

    def getValiData(self,batchSize=None,chunks='auto',shuffle=False):
        if batchSize == None:
            batchSize = self.batch_size
        if isinstance(chunks,int):
            assert chunks>1,"Expected chunks is int and >1"
        elif chunks != 'auto':
            raise WrongOperation('Unuseful chunks parameter')

        if self.validDataRatio == 0:
            raise WrongOperation('No reserved validation data')            
        elif len(self.datasetList) > 0:
            with tempfile.NamedTemporaryFile('w',suffix='.scp') as validScpFile:
                validScpFile.write('\n'.join(self.validFiles))
                validScpFile.seek(0)                
                validIter = DataIterator(validScpFile.name,batchSize,chunks,self.fileProcessFunc,shuffle,self.labels,None,0)
        else:
            validIter = DataIterator(self.validData,batchSize,chunks,self.fileProcessFunc,shuffle,self.labels,None,0)
        return validIter

    @property
    def epochOver(self):
        if self.is_new_epoch:
            self.is_new_epoch = False
            return True
        else:
            return False

    @property
    def chunkOver(self):
        if self.is_new_chunk:
            self.is_new_chunk = False
            return True
        else:
            return False        

class Supporter(object):
    '''
    Useage:  supporter = Supporter(outDir='Result')

    Supporter is a class to be similar to chainer report. But we designed some useful functions such as save model by maximum accuracy and adjust learning rate.

    '''      
    def __init__(self,outDir='Result'):

        self.currentFiled = {}

        self.globalFiled = []

        if outDir.endswith('/'):
            outDir = outDir[:-1]
        self.outDir = outDir

        self.count = 0

        if not os.path.isdir(self.outDir):
            os.mkdir(self.outDir)

        self.logFile = self.outDir+'/log'
        self.log = ['[',']']

        with open(self.logFile,'w'):
            pass
        
        self.lastSavedModel = {}
        self.savingThreshold = None

        self._allKeys = []
        
    def send_report(self,x,*args):
        '''
        Useage:  supporter = obj.send_report({"epoch":epoch,"train_loss":loss,"train_acc":acc})

        Send information and thses info will be retained untill you do the statistics by using obj.collect_report().

        '''           
        keys = list(x)

        allKeys = list(self.currentFiled)
    
        for i in keys: 
            value = x[i]
            try:
                value=float(value.data)
            except:
                pass
            i = i.lower()
            if not i in allKeys:
                self.currentFiled[i] = []
            self.currentFiled[i].append(value)

    def collect_report(self,keys=None,plot=True):
        '''
        Useage:  supporter = obj.collect_report(plot=True)

        Do the statistics of received information. The result will be saved in outDir/log file. If < keys > is not None, only collect the data in keys. 
        If < plot > is True, print the statistics result to standard output.
        
        '''   
        if keys == None:
            keys = list(self.currentFiled)
    
        self.globalFiled.append({})

        allKeys = list(self.currentFiled)
        self._allKeys.extend(allKeys)
        self._allKeys = list(set(self._allKeys))

        message = ''
        for i in keys:
            if i in allKeys:
                mn = float(np.mean(self.currentFiled[i]))
                if type(self.currentFiled[i][0]) == int:
                    mn = int(mn)
                    message += (i + ':%d    '%(mn))
                else:
                    message += (i + ':%.5f    '%(mn))
                self.globalFiled[-1][i] = mn
            else:
                message += (i + ':-----    ')

        # Print to log file
        if self.log[-2] != '[':
            self.log[-2] += ','
        self.log[-1] = '    {'
        allKeys = list(self.globalFiled[-1].keys())
        for i in allKeys[:-1]:
            self.log.append('        "{}": {},'.format(i,self.globalFiled[-1][i]))
        self.log.append('        "{}": {}'.format(allKeys[-1],self.globalFiled[-1][allKeys[-1]]))
        self.log.append('    }')
        self.log.append(']')
        with open(self.logFile,'w') as fw:
            fw.write('\n'.join(self.log))
        
        # Print to screen
        if plot:
            print(message)
        # Clear
        self.currentFiled = {}

    def save_model(self,models,iterSymbol=None,byKey=None,maxValue=True,saveFunc=None):
        '''
        Useage:  obj.save_model(plot=True)

        Save model when you use this function. Your can give <iterSymbol> and it will be add to the end of file name.
        If you use < byKey > and set < maxValue >, model will be saved only while meeting the condition.
        We use chainer as default framework, but if you don't, give the < saveFunc > specially please. 
        
        ''' 
        assert isinstance(models,dict), "Expected <models> is dict whose key is model-name and value is model-object."

        if self.currentFiled != {}:
            self.collect_report(plot=False)

        if iterSymbol == None:
            try:
                suffix = '_'+str(self.globalFiled[-1]['epoch'])+'_' 
            except:
                suffix = "_"
        else:
            suffix = '_'+str(iterSymbol)+'_'

        if byKey == None:
            for name in models.keys():
                copymodel = models[name].copy()
                copymodel.to_cpu()
                fileName = self.outDir+'/'+name+suffix[:-1]+'.model'
                if saveFunc == None:
                    chainer.serializers.save_npz(fileName, copymodel)
                else:
                    saveFunc(fileName,copymodel)
                self.lastSavedModel[name] = fileName
        else:
            byKey = byKey.lower()
            try:
                value = self.globalFiled[-1][byKey]
            except:
                raise WrongOperation('Keywords {} has not be reported'.format(byKey))

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
                    copymodel = models[name].copy()
                    copymodel.to_cpu()
                    if isinstance(value,float):
                        value = ('%.5f'%(value)).replace('.','')
                    else:
                        value = str(value)
                    fileName = self.outDir+'/'+ name + suffix + value + '.model'
                    if saveFunc == None:
                        chainer.serializers.save_npz(fileName, copymodel)
                    else:
                        saveFunc(fileName,copymodel)
                    if self.lastSavedModel != {}:
                        os.remove(self.lastSavedModel[name])
                    self.lastSavedModel[name] = fileName

    @property
    def finalModel(self):
        '''
        Useage:  model = obj.finalModel

        Get the final saved model. Return a dict whose key is model name and value is model path. 
        
        ''' 
        return self.lastSavedModel

    def adjust_lr(self,oldLR,key,condition,threshold,newLR=None):
        '''
        Useage:  newLR = obj.adjust_lr(0.08,'train_loss','<',0.1,0.04)

        This is a simple function. You can use it to change learning rate.
        
        ''' 
        assert condition in ['>','>=','<=','<','==','!='], '<condiction> is not a correct conditional operator.'
        assert isinstance(threshold,(int,float)), '<threshold> should be float or int value.'

        if self.currentFiled != {}:
            self.collect_report(plot=False)
        
        if key not in self.globalFiled[-1].keys():
            raise WrongOperation('Keyword {} has not been reported.'.format(key))

        value = str(self.globalFiled[-1][key])

        condition = eval(value+condition+str(threshold))

        if condition:
            if newLR == None:
                return 0.5 * oldLR
            else:
                return newLR
        else:
            return oldLR

    def dump_item(self,key=None):
        '''
        Useage:  out = obj.dump_item(key=['epoch','train_loss','test_loss'])

        Return a dict. Get the global data in order to plot graph.
        
        ''' 
        if isinstance(key,str):
            key = [key,]

        elif key == None:
            key = self._allKeys
            
        items = {}
        for i in key:
            items[i] = []

        for filed in self.globalFiled:
            temp = {}
            for i in key:
                if i in filed.keys():
                    temp[i] = filed[i]
            if len(temp) == len(key):
                for i in key:
                    items[i].append(temp[i])

        return items

