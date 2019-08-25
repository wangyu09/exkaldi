# PythonKaldi
This is a tool which introduce kaldi tools into python in a easy-use way.

## PythonKaldi at a Glance

1. Clone the PythonKaldi project.
```
git clone https://github.com/wangyu09/pythonkaldi
```

2. In the file < CSJsample.py >, there is a sample program that showed how to use the PythonKaldi tool. Exchange the parameter < CSJpath > for yours and also other parameters such as < epoch > if you want. Then run it.
```
python CSJsample.py
```
Especially, there are three sections in this sample program: first, train chainer neural network as acoustic model, and then use this pretrained AM to forward test data and decode them by generating lattice and further compute the WER. In the third step function < OnlineRecognize >, although we wrote it, I am sorry that it cannot be used now because of debugging.

## Concepts and Usage
Most of functions in PythonKaldi tool are performed with using "subprocess" to run shell cmd of kaldi tools. But we design a series of classes and functions to use them in a flexible way and it is more familiar for python programmer. PythonKaldi tool consist of three parts: < Basis Tools > to cover kaldi functions, < Speek client > to realize online-recognization, and < Chainer Tools > to give some useful tools to help train neural network acoustic model.

### < Basis Tools >

#### 1. Basic class: **KaldiArk**   

**KaldiArk** is a subclass of **bytes**. It is a object who holds the kaldi ark data in a binary type. **KaldiArk** and **KaldiDict** object have almost the same attributes and functions, and they can do some mixed operations such as "+" and "concat" and so on.  
Moreover, alignment can also be held by KaldiArk and KaldiDict in Pythonkaldi tool, and we defined it as int32 data type.  

< Attributes >  

`__.lens__`   
return a tuple: ( the numbers of all utterances, the frames of each utterance ).  

__.dim__    
return a int: feature dim.  

__.dtype__    
return a str: data type such as 'float32'.  

__.utts__    
return a list: all utterance names.  

__.array__    
return a KaldiDict object: transform binary ark data into numpy arrar format.  

< Methods >    

__.toDtype(dtype)__    
change data dtype and return.  

__.check_format()__    
check if inner data has a correct kaldi ark format. If had, return True.  

__.save(fileName,chunks=1)__    
save as .ark file. If chunks>1, split it averagely and save them.  

__+ operator__  
KaldiArk object can use < + > operator with another KaldiArk object or KaldiDict object.  

__.concat(others,axis=1)__    
Return KaldiArk object. If any member has a dtype of float, the result will be float, or it will be int.  
It only return the concat results whose utterance id appeared in all members at the same time.

__.splice(left,right=None)__    
Return KaldiArk object. Splice front-behind frames. if right == None, we use right = left.  

__.subset(nHead=0,chunks=1,uttList=None)__    
if nhead > 0, return KaldiArk object which only has start nHead utterances.  
if chunks > 1, return list whose members are KaldiArk.  
if uttList != None, select utterances if utterance id appeared.  

#### 2. Basic class: **KaldiDict**   

**KaldiDict** is a subclass of **dict**. It is a object who holds the kaldi ark data in numpy array type. Its key are the utterance id and the value is the numpy array data. **KaldiDict** can also do some mixed operations with **KaldiArk** such as "+" and "concat" and so on.  
Note that **KaldiDict** has some functions which **KaldiArk** dosen't have. They will be introduced as follow.

< Attributes >  

__.lens__    
the same as **KaldiArk**.lens

__.dim__    
the same as **KaldiArk**.dim

__.dtype__    
the same as **KaldiArk**.dtype

__.utts__    
the same as **KaldiArk**.utts

__.array__    
return a KaldiArk object: transform numpy array data into kaldi binary format.  

< Methods >    

__.toDtype(dtype)__    
the same as **KaldiArk**.toDtype

__.check_format()__    
the same as **KaldiArk**.check_format

__.save(fileName,chunks=1)__    
the same as **KaldiArk**.save 

__+ operator__    
the same as **KaldiArk**.add. KaldiDict object can also use < + > operator with another KaldiArk object.    

__.concat(others,axis=1)__    
the same as **KaldiArk**.concat  

__.splice(left,right=None)__    
the same as **KaldiArk**.splice  

__.subset(nHead=0,chunks=1,uttList=None)__    
the same as **KaldiArk**.subset  

__.merge(keepDim=False)__    
return a tuple. if keepDim == True the first member is list whose content are numpy arrays of all utterances, and if keepDim == False, it is a integrated numpy array of all utterances. the second member is utterance ids and their frame length information. 

__.remerge(matrix,uttLens)__    
if self has not any data, do not return, or return a new KaldiDict object. this is a inverse operation of .merge function.

__.normalize(std=True,alpha=1.0,beta=0.0,epsilon=1e-6,axis=0)__    
return a KaldiDict object. if std == True, do _alpha*(x-mean)/(std+epsilon)+belta_, or do _alpha*(x-mean)+belta_.

#### 3. Basic class: **KaldiLattice**   

**KaldiLattice** holds the lattice and its related file path: HmmGmm file and WordSymbol file. PythonKaldi.decode_lattice function will return a KaldiLattice object. Aslo, you can define a empty KaldiLattice object and load its data later.


< Attributes >  

__.value__    
return a lattice with a binary data type.

< Methods >  

__.load(latFile,HmmGmm,wordSymbol)__      
load lattice. < latFile > can be file path or binary data. < HmmGmm > and < wordSymbol > are expected as file path.

__.get_1best_words(minLmwt=1,maxLmwt=None,Acwt=1.0,outDir='.',asFile=False)__  
return dict object. key is the lmwt value. if < asFile > == True or file name, the result will be save as file and values of returned dict is these files' path, or they will be 1-best words.

__.scale(Acwt=1,inAcwt=1,Ac2Lm=0,Lmwt=1,Lm2Ac=0)__  
return a new scaled KaldiLattice object.

__.add_penalty(penalty=0)__  
return a new KaldiLattice object.

__.save(fileName)__  
save lattice as .gz file.

__+ operator__  
add the numbers of lattice. Note that it is just a simple addtional operation.

#### 4. Function: compute_mfcc(wavFile,_**other parameters_)   

compute mfcc feature. return KaldiArk object or file path.  

< Parameters >  

__wavFile__   _wav file or scp file, you can declare its type by using point useSuffix_  
__rate__   _sample rate, default = 16000_  
__frameWidth__   _stride windows width, milliseconds, default = 25_  
__frameShift__   _stride windows width, milliseconds, default = 10_  
__melBins__   _numbers of mel bins, default = 23_  
__featDim__   _dimendionality of mfcc feature, default = 13_  
__windowType__   _window function, default = 'povey'_  
__useUtt__   _when file is a wave file, you can name its utterance id, default = "MAIN"_  
__useSuffix__   _when file is a scp file but withou 'scp' suffix, you can declare its file suffix, default = None_  
__configFile__   _It is unable now and must be None_  
__asFile__   _if True or file name, save result as file and return file path, or return KaldiArk, default = False_  
__wavFile__ _wav file or scp file, you can declare its type by using <useSuffix>_  
  
#### 5. Function: compute_fbank(wavFile,_**other parameters_)   

compute fbank feature. return KaldiArk object or file path.  

< Parameters >  

__wavFile__   _wav file or scp file, you can declare its type by using point useSuffix_  
__rate__   _sample rate, default = 16000_  
__frameWidth__   _stride windows width, milliseconds, default = 25_  
__frameShift__   _stride windows width, milliseconds, default = 10_  
__melBins__   _numbers of mel bins, default = 23_  
__windowType__   _window function, default = 'povey'_  
__useUtt__   _when file is a wave file, you can name its utterance id, default = "MAIN"_  
__useSuffix__   _when file is a scp file but withou 'scp' suffix, you can declare its file suffix, default = None_  
__configFile__   _It is unable now and must be None_  
__asFile__   _if True or file name, save result as file and return file path, or return KaldiArk, default = False_  

#### 6. Function: compute_plp(wavFile,_**other parameters_)   

compute plp feature. return KaldiArk object or file path.  

< Parameters >  

__wavFile__   _wav file or scp file, you can declare its type by using point useSuffix_  
__rate__   _sample rate, default = 16000_  
__frameWidth__   _stride windows width, milliseconds, default = 25_  
__frameShift__   _stride windows width, milliseconds, default = 10_  
__melBins__   _numbers of mel bins, default = 23_  
__featDim__   _dimendionality of mfcc feature, default = 13_  
__windowType__   _window function, default = 'povey'_  
__useUtt__   _when file is a wave file, you can name its utterance id, default = "MAIN"_  
__useSuffix__   _when file is a scp file but withou 'scp' suffix, you can declare its file suffix, default = None_  
__configFile__   _It is unable now and must be None_  
__asFile__   _if True or file name, save result as file and return file path, or return KaldiArk, default = False_  

#### 7. Function: compute_spectrogram(wavFile,_**other parameters_)   

compute spectrogram feature. return KaldiArk object or file path.  

< Parameters >  

__wavFile__   _wav file or scp file, you can declare its type by using point useSuffix_  
__rate__   _sample rate, default = 16000_  
__frameWidth__   _stride windows width, milliseconds, default = 25_  
__frameShift__   _stride windows width, milliseconds, default = 10_  
__windowType__   _window function, default = 'povey'_  
__useUtt__   _when file is a wave file, you can name its utterance id, default = "MAIN"_  
__useSuffix__   _when file is a scp file but withou 'scp' suffix, you can declare its file suffix, default = None_  
__configFile__   _It is unable now and must be None_  
__asFile__   _if True or file name, save result as file and return file path, or return KaldiArk, default = False_  

#### 8. Function: use_cmvn(feat,_**other parameters_) 

apply CMVN to feature. return KaldiArk object or file path.
if all of other parameters are None, compute the CMVN state within each utterance firstly and use them.

< Parameters >  

__feat__ _KaldiArk or KaldiDict object_
__cmvnStatFile__   _if None compute it firstly, default = None_  
__spk2uttFile__   _if None compute cmvn state whin each utterance, default = None_  
__spk2uttFile__   _if None and spk2uttFile != None, raise error, default = None_  
__asFile__   _if True or file name, save result as file and return file path, or return KaldiArk, default = False_  

#### 9. Function: use_cmvn_sliding(feat,_**other parameters_) 

apply sliding CMVN to feature. return KaldiArk object or file path.

< Parameters >  

__feat__ _KaldiArk or KaldiDict object_
__windowsSize__   _sliding windows width, frames, if None, set it to cover all frames at one time, default = None_  
__std__   _if False, only apply mean, default = False_ 

#### 10. Function: add_delta(feat,_**other parameters_) 

add n orders delta to feature. return KaldiArk object or file path.

< Parameters >  

__feat__ _KaldiArk or KaldiDict object_
__order__   _the times of delta, default = 2_ 
__.asFile__   _if True or file name, save result as file and return file path, or return KaldiArk, default = False_  

#### 11. Function: get_ali(faliFile,HmmGmm,_**other parameters_) 

get alignment from alignment file. return KaldiDict object.

< Parameters >  

__faliFile__ _kaldi alignment file path_
__HmmGmm__   _HmmGmm model path_ 
__returnPhoneme__   _if True, return phoneme id, or return pdf id, default = False_

#### 12. Function: decompress(data) 

decompress kaldi compressed feature data. return KaldiArk object.

< Parameters >  

__data__ _the binary data of kaldi compressed feature_

#### 13. Function: load(filePath,_**other parameters_) 

load kaldi ark feat file, kaldi scp feat file, KaldiArk file, or KaldiDict file. return KaldiArk or KaldiDict object.

< Parameters >  

__filePath__ _file path with a suffix '.ark' or '.scp' or '.npy'_
__useSuffix__   _when file has another suffix, you can declare it, default = None_

#### 13. Function: decode_lattice(AmP,HmmGmm,Hclg,Lexicon,_**other parameters_) 

decode by generating lattice. return KaldiLattice object.

< Parameters >  

__filePath__ _file path with a suffix '.ark' or '.scp' or '.npy'_
__useSuffix__   _when file has another suffix, you can declare it, default = None_



