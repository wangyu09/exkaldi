# ExKaldi Automatic Speech Recognition Toolkit
ExKaldi toolkit is an extension package for Kaldi speech recognition toolkit. 
It is developed to build an interface between the Kaldi toolkit and deep learning frameworks, 
such as PyTorch and Chainer, with Python language and further help users customize speech recognition system easily. 
A set of functions of ExKaldi are implemented by Kaldi, 
and serval Input-Output interfaces are designed to transform their data format so that it is flexible to process speech waveform, 
extract acoustic features, perform speech decoding and even deal with lattices produced by the decoder, with Python. 
Based on this, ExKaldi further provides tools to support training a DNN-based acoustic model, 
and improve their performance by, for example, multiple tasks with different labels. 
With jointing Kaldi and deep learning frameworks, 
integrated solutions are presented in ExKaldi from feature extracting to decoding to put up a customized speech recognition system quickly. 

## Start with ExKaldi

1. Install Kaldi ASR toolkit. Download the Kaldi ASR toolkit firstly.
```
git clone https://github.com/kaldi-asr/kaldi.git kaldi --origin upstream
```
And follow these three tutorial files to install and compile it.
```
less kaldi/INSTALL
less kaldi/tools/INSTALL
less kaldi/src/INSTALL
```

2. Install ExKaldi toolkit from PyPi library.
```
pip install exkaldi
```
You can also clone the ExKaldi source code from our github project, then make it a pypi package and install it.
```
git clone https://github.com/wangyu09/exkaldi.git
cd exkaldi
python setup.py sdist bdist_wheel
cd ..
pip install exkaldi/dist/*
```

3. We prepared some example programs to show how to use ExKaldi to train a coustic model and build a ASR system.
Please download them from https://github.com/wangyu09/exkaldi/examplecode (or clone the ExKaldi source code). 
Framework, Chainer or Pytorch, is expected in you machine. Concerning Chainer framework, know more about it from https://chainer.org/ .
You can install it directly by runing:
```
pip install chainer
```

4. Taking a Chainer DNN model based on TIMIT corpus example. Before this, fmllr feature file, alignment file and decoding graph are required.
Run the follow scipts to obtain them respectively. Please prepare TIMIT corpus in advance.
```
cd kaldi/egs/timit/s5
```
Modify the TIMIT path, cmd and path cofigure. Then run script up to Karel's DNN.
```
./run.sh
local/nnet/run_dnn.sh
```
Then get alignment data. It will be label to train DNN model.
```
steps/nnet/align.sh --nj 4 data-fmllr-tri3/train data/lang exp/dnn4_pretrain-dbn_dnn exp/dnn4_pretrain-dbn_dnn_ali
steps/nnet/align.sh --nj 4 data-fmllr-tri3/dev data/lang exp/dnn4_pretrain-dbn_dnn exp/dnn4_pretrain-dbn_dnn_ali_dev
steps/nnet/align.sh --nj 4 data-fmllr-tri3/test data/lang exp/dnn4_pretrain-dbn_dnn exp/dnn4_pretrain-dbn_dnn_ali_test
```

5. After finishing all work above, run the example python program. You can adjust more configure.
```
python TIMIT_DNN_fmllr_chainer.py -TIMITROOT <your path>
```
Results will be save in output directory.


## Concepts and Usage
The core functions in ExKaldi tool are performed with using "subprocess" to run shell cmd of Kaldi tools. 
Based on this, we designed a series of classes and approaches to use them in a flexible way and it is more familiar for python programmer. 
ExKaldi toolkit of current version mainly consists of one part which implements Kaldi functions such as processing feature and lattice, 
and another part which supports training DNN-based acoustic model with deep learning framework such as Chainer and Pytorch, 
and the other part which simply allows user record their voice from microphone and recognize it with their customized ASR system. 

_-----------------------------------------------< ExKaldi API >-----------------------------------------------------_

- [Core](#kaldiark)
    - [class: KaldiArk](#kaldiark)
    - [class: KaldiDict](#kaldidict)
    - [class: KaldiLattice](#kaldilatticelatnonehmmnonewordsymbolnone)
    - [function: load](#loadfilenameother-parameters)
    - [function: save](#savedataother-parameters)
    - [function: concat](#concatdatasaxis)
    - [function: cut](#cutdatamaxFrames)
    - [function: normalize](#normalizedatastdtruealpha10beta00epsilon1e-6axis0)
    - [function: merge](#mergedatakeepdimfalsesortfalse)
    - [function: remerge](#remergematrixuttLens)
    - [function: sort](#sortdatabyframereversefalse)
    - [function: splice](#splicedataleft4rightNone)
    - [function: compute_mfcc](#compute_mfccwavfileother-parameters)
    - [function: compute_fbank](#compute_fbankwavfileother-parameters)
    - [function: compute_plp](#compute_plpwavfileother-parameters)
    - [function: compute_spectrogram](#compute_spectrogramwavfileother-parameters)
    - [function: use_cmvn](#use_cmvnfeatother-parameters)
    - [function: compute_cmvn_stats](#compute_cmvn_statsfeatoutfileother-parameters)
    - [function: use_cmvn_sliding](#use_cmvn_slidingfeatother-parameters)
    - [function: add_delta](#add_deltafeatother-parameters)
    - [function: load_ali](#load_alialifilehmmother-parameters)
    - [function: load_lat](#load_latlatfilehmmwordsymbol)
    - [function: analyze_counts](#analyze_countsalifileoutfileother-parameters)
    - [function: decompress](#decompressdata)
    - [function: decode_lattice](#decode_latticeamphmmhclgwordsymbolother-parameters)
    - [function: run_shell_cmd](#run_shell_cmdcmdother-parameters)
    - [function: get_kaldi_path](#get_kaldi_path)
    - [function: get_env](#get_env)
    - [function: set_kaldi_path](#set_kaldi_pathpath)
    - [function: check_config](#check_confignameconfignone)
    - [function: split_file](#split_filefilepathother-parameters)
    - [function: pad_sequence](#pad_sequencedataother-parameters)
    - [function: unpack_padded_sequence](#unpack_padded_sequencedatalengthsother-parameters)
    - [function: wer](#werrefhypother-parameters)
    - [function: accuracy](#accuracyrefhypother-parameters)
    - [function: edit_distance](#edit_distancerefhypother-parameters)
    - [function: log_softmax](#log_softmaxdataother-parameters_)
    - [class: DataIterator](#dataiteratorscpfilesprocessfuncbatchsizechunksautootherargsnoneshufflefalseretaindata00)
    - [class: Supporter](#supporteroutdirresult)

### KaldiArk()   

< class description >  

**KaldiArk** is a subclass of **bytes**. It maks a object who holds the Kaldi ark data in a binary type. **KaldiArk** and **KaldiDict** object have almost the same attributes and functions, and they can do some mixed operations such as "+" and "concat" and so on.   Moreover, force-alignment can also be held by KaldiArk and KaldiDict in ExKaldi tool, and we defined it as int32 data type.  

< Attributes >  

`.lens`   
return a tuple: ( the numbers of all utterances, the utterance ID and its frames of each utterance ).  

`.dim`    
return an int number: the dimensions of data.  

`.dtype`    
return a str: data type such as 'float32'. 

`.utts`    
return a list: all utterance names.  

`.array`    
return a KaldiDict object: transform binary ark data to numpy arrar format.  

< Methods >    

`.to_dtype(dtype)`    
change data dtype and return a new KaldiArk object.  

`.check_format()`    
check whether data has a correct Kaldi ark format. If had, return True. Or raise error.  

`.save(fileName,chunks=1,outScpFile=False)`   
save as .ark (and scp) file. If chunks>1, split it averagely and save them.  

`__add__` 
return a new KaldiArk object: use < + > operator to plus another KaldiArk object or KaldiDict object.  

`.splice(left,right=None)`  
return a KaldiArk object. Splice front-behind frames. if right is None, we define right = left.  

`.select(left,dims,retain=False)`  
return KaldiArk object(s): select data according to dims. < dims > should be an int or string like "1,5-20".
If retain ==  True, return both selected data and non-selected data, or return only selected data.

`.subset(nHead=0,chunks=1,uttList=None)`  
if nhead > 0, return a KaldiArk object which only has start-n utterances.  
if chunks > 1, return list whose members are KaldiArk objects.  
if uttList != None, select utterances if utterance id appeared.
only one of these three options will works by order.   

### KaldiDict() 

< class description >  

**KaldiDict** is a subclass of **dict**. It generates a object who holds the Kaldi ark data in NumPy array type. 
Its keys are the utterance IDs and the values are data. **KaldiDict** can also do some mixed operations with **KaldiArk** such as "+" and "concat" and so on.  
Note that **KaldiDict** has a part of functions which **KaldiArk** dosen't have.

< Attributes >  

`.lens`    
return a tuple: ( the numbers of all utterances, the utterance ID and its frames of each utterance ).  

`.dim`   
return an int number: the dimensions of data.  

`.dtype`    
return a str: data type such as 'float32'. 

`.utts`   
return a list: all utterance names.  

`.ark`   
return a KaldiArk object: transform numpy array data into Kaldi's binary format.  

`.target`   
If it is alignment data, return the target classes that is maximum value+1.  

< Methods >    

`.to_dtype(dtype)`    
change data dtype and return a new KaldiArk object.  

`.check_format()`    
check whether data has a correct Kaldi ark format. If had, return True. Or raise error.  

`.save(fileName,chunks=1)`  
save as .npy file. If chunks>1, split it averagely and save them.  

`__add__`  
return a new KaldiDict object: use < + > operator to plus another KaldiArk object or KaldiDict object.  

`.concat(others,axis=1)`    
return a KaldiDict object. If any member has a dtype of float, the result will be float type, or it will be int type.  
It only returns the concat results whose utterance IDs appeared in all members.
If axis is 1 and some utterance only has 1 dimension or 1 frame, it will be concat to all frames of others.

`.splice(left,right=None)`    
return a KaldiDict object. Splice front-behind frames. if right is None, we define right = left.
This result is kind of different from KaidiArk().splice() function.

`.select(left,dims,retain=False)`  
return KaldiDict object(s): select data according to dims. < dims > should be an int or string like "1,5-20".
If retain ==  True, return both selected data and non-selected data, or return only selected data.

`.subset(nHead=0,chunks=1,uttList=None)`    
if nhead > 0, return a KaldiArk object which only has start-n utterances.  
if chunks > 1, return list whose members are KaldiArk objects.  
if uttList != None, select utterances if utterance id appeared.
only one of these three options will works by order.   

`.sort(by='frame',reverse=False)`
return a KaldiDict object: sort data by utterance IDs or the length of utterances.
if reverse == True, do descending order.

`.merge(keepDim=False,sortFrame=False)`    
return a tuple. if keepDim == True, the first member is list whose content are NumPy arrays with 2-dimensions of all utterances, and if keepDim == False, 
it is a integrated NumPy array with 3-dimensions of all utterances. 
the second member is utterance IDs and their respective frame length. 
if sortFrame == True , it will sort all utterances by length with ascending order before merging.

`.remerge(matrix,uttLens)`    
If self has not any data, do not return, or return a new KaldiDict object: this is a inverse operation of .merge function.

`.normalize(std=True,alpha=1.0,beta=0.0,epsilon=1e-6,axis=0)`
Return a KaldiDict object. if std == True, do _alpha*(x-mean)/(std+epsilon)+belta_, or do _alpha*(x-mean)+belta_.

`.cut(maxFrames)`    
return a KaldiDict object: traverse all utterances, and if one is longer than 1.25*maxFrames, cut it with a threshold length of maxFrames.

`.tuple_value(others,sort=False)`    
Tuple the utterance of the same ID from different objects. Return a list whose members are tuple: (utterance IDs, the utterances of others)

### KaldiLattice(lat=None,hmm=None,wordSymbol=None) 

< class description >

**KaldiLattice** holds the lattice and its related file path: HMM file and WordSymbol file. ExKaldi.decode_lattice function will return a KaldiLattice object. 
Aslo, you can define a empty KaldiLattice object and load its data later.

< init Parameters >

`lat` _expected Kaldi's lattice binary data or lattice file path which is compressed-gz file_        
`hmm` _HMM file path_  
`wordSymbol` _word to int ID file path_

< Attributes >  

`.value`    
return binary data of lattice.

`.model`    
return HMM file path.

`.lexicon`    
return word-to-id file path.

< Methods >  

`.load(latFile,hmm,wordSymbol)`        
load lattice. < latFile > can be file path or binary data. < hmm > and < wordSymbol > are expected as file path.

`.get_1best(lmwt=1,maxLmwt=None,acwt=1.0,outFile=None,phoneSymbol=None)`   
If maxLmwt != None, return Python dict object: its keys are the lmwt value and values are the 1best words output collected in a list. Or only return a list.
If < outFile > is file name, the 1best words output will be save as file and values of returned dict will be changed for these files' path.
If < phoneSymbol > is not None, will return phones outputs of 1best words. 

`.get_nbest(n,acwt=1.0,outFile=None,outAliFile=None,requireCost=False)`   
If < outFile > is not None, output results as file and if < requireCost > == True, lm cost and ac cost will be also returned as files. In this way, return a list whose members are path of these files. If < outFile > is None, also return a list but its members are n best words and their respective ac cost and lm cost.
If < outAliFile > is not None, fore-alignment file will be reserved. 

`.scale(acwt=1,inAcwt=1,ac2lm=0,lmwt=1,lm2ac=0)`  
sacle lattice and return a new scaled KaldiLattice object.

`.add_penalty(penalty=0)`  
add words insertion penalty and return a new KaldiLattice object.

`.save(fileName,copyFile=False)`
save lattice as .gz file. If < copyFile > is True, will copy HMM file and wordSymbol file to the same directory as saved lattice file. 

`__add__`  
add another lattice. Note that it is just a simple addtional operation to intergrat several lattices as a big one.

### load(fileName,_**other parameters_) 

< function description >

Load Kaldi ark feat file, Kaldi scp feat file, KaldiArk ark file, or KaldiDict npy file. 
Return KaldiArk or KaldiDict object.

< Parameters >  

`filePath` _file path with a suffix '.ark' or '.scp' or '.npy'_
`useSuffix`  _when file has another suffix, you can declare it, default = None_

### save(data,_**other parameters_)

< function description >

It is the same as .save method of KaldiArk or KaldiDict. 
< data > is expected as KaldiArk or KaldiDict object.

< Parameters >  

`data` _KaldiArk, KaldiDict, or KaldiLattice object_
`*params`  _If data is KaldiArk, or KaldiDict object, params should be filename and chunks, If KaldiLattice, fileName and copyFile shou be given_

### concat(datas,axis)

< function description >

return a KaldiDict object. It is the same as .concat method of KaldiDict. 
< datas > is expected as KaldiArk or KaldiDict object(s).

### cut(data,maxFrames)

< function description >

return KaldiDict object. It is he same as .cut method of KaldiDict. 
< data > is expected as KaldiArk or KaldiDict object.

### normalize(data,std=True,alpha=1.0,beta=0.0,epsilon=1e-6,axis=0)

< function description >

return KaldiDict object. It is he same as .normalize method of KaldiDict. 
< data > is expected as KaldiArk or KaldiDict object.

### merge(data,keepDim=False,sort=False)

< function description >

It is the same as .merge method of KaldiDict. 
< data > is expected as KaldiArk or KaldiDict object(s).

### remerge(matrix,uttLens)

< function description >

return a kaldiDict object.It is the same as .remerge method of KaldiDict. 

### sort(data,by='frame',reverse=False)

< function description >

return a KaldiDict object. It is the same as .sort method of KaldiDict. 
< data > is expected as KaldiArk or KaldiDict object(s).

### splice(data,left=4,right=None)

< function description >

return KaldiArk or KaldiDict object. It is the same as .splice method of KaldiArk or KaldiDict. 
< datas > is expected as KaldiArk or KaldiDict object.

### compute_mfcc(wavFile,_**other parameters_)

< function >

Compute mfcc feature. Return KaldiArk object or file path if < asFile > is True. We provide some common options, 
If you want to use more options, set < config > = your-configure but note that if you do this, these usual configures we provided will be ignored. 
You can use ExKaldi.check_config('compute_mfcc') function to get configure information you could set. 
Also run shell command "compute-mfcc-feats" to check their meaning. 

< Parameters >  

`wavFile`   _WAV file or scp file, you can declare its type by using point useSuffix_  
`rate`   _sampling rate, default = 16000_  
`frameWidth`   _stride windows width, milliseconds, default = 25_  
`frameShift`   _stride windows width, milliseconds, default = 10_  
`melBins`   _numbers of mel bins, default = 23_  
`featDim`   _dimendionality of mfcc feature, default = 13_  
`windowType`   _window function, default = 'povey'_  
`useSuffix`   _when file is a scp file but without 'scp' suffix, you can declare its file suffix, or error will be raised, default = None_  
`config`   _another configure setting method_  
`asFile`   _if it is True, save result as file and return file path. Or return KaldiArk, default = False_  
  
### compute_fbank(wavFile,_**other parameters_)

< function >

Compute fbank feature. Return KaldiArk object or file path if < outFile > is True. We provide some common options, 
If you want to use more options, set < config > = your-configure but note that if you do this, these usual configures we provided will be ignored. 
You can use ExKaldi.check_config('compute_fbank') function to get configure information you could set. 
Also run shell command "compute-fbank-feats" to check their meaning. 

< Parameters >  

`wavFile`   _WAV file or scp file, you can declare its type by using point useSuffix_  
`rate`   _sampling rate, default = 16000_  
`frameWidth`   _stride windows width, milliseconds, default = 25_  
`frameShift`   _stride windows width, milliseconds, default = 10_  
`melBins`   _numbers of mel bins, default = 23_  
`windowType`   _window function, default = 'povey'_  
`useSuffix`   _when file is a scp file but withou 'scp' suffix, you can declare its file suffix, default = None_  
`config`   _another configure setting method_  
`asFile`   _if it is true, save result as file and return file path. Or return KaldiArk, default = False_  


### compute_plp(wavFile,_**other parameters_)  

< function >

Compute plp feature. Return KaldiArk object or file path if < outFile > is True. We provide some common options, 
If you want to use more options, set < config > = your-configure but note that if you do this, these usual configures we provided will be ignored. 
You can use ExKaldi.check_config('compute_plp') function to get configure information you could set. 
Also run shell command "compute-plp-feats" to check their meaning. 

< Parameters >  

`wavFile`   _WAV file or scp file, you can declare its type by using point useSuffix_  
`rate`   _sample rate, default = 16000_  
`frameWidth`   _stride windows width, milliseconds, default = 25_  
`frameShift`   _stride windows width, milliseconds, default = 10_  
`melBins`   _numbers of mel bins, default = 23_  
`featDim`   _dimendionality of mfcc feature, default = 13_  
`windowType`   _window function, default = 'povey'_  
`useSuffix`   _when file is a scp file but withou 'scp' suffix, you can declare its file suffix, default = None_  
`config`   _another configure setting method_  
`asFile`   _if it is True, save result as file and return file path, or return KaldiArk, default = False_  


### compute_spectrogram(wavFile,_**other parameters_) 

< function description>

Compute spectrogram feature. Return KaldiArk object or file path if < outFile > is True. We provide some common options, 
If you want to use more options, set < config > = your-configure but note that if you do this, these usual configures we provided will be ignored. 
You can use ExKaldi.check_config('compute_spectrogram') function to get configure information you could set. 
Also run shell command "compute-spectrogram-feats" to check their meaning. 

< Parameters >  

`wavFile`   _WAV file or scp file, you can declare its type by using point useSuffix_  
`rate`   _sample rate, default = 16000_  
`frameWidth`   _stride windows width, milliseconds, default = 25_  
`frameShift`   _stride windows width, milliseconds, default = 10_  
`windowType`   _window function, default = 'povey'_  
`useSuffix`   _when file is a scp file but withou 'scp' suffix, you can declare its file suffix, default = None_  
`config`   _another configure setting method_  
`asFile`   _if it is True, save result as file and return file path, or return KaldiArk, default = False_  


### use_cmvn(feat,_**other parameters_) 

< function description >

Apply CMVN to feature. Return KaldiArk object or file path if < outFile > is not None. If < cmvnStatFile >  are None, first compute the CMVN state. But < utt2spkFile > and < spk2uttFile > are expected at the same time if they were not None.

< Parameters >  

`feat` _KaldiArk or KaldiDict object_
`cmvnStatFile`   _if None compute it firstly, default = None_  
`spk2uttFile`   _if None compute cmvn state whin each utterance, default = None_  
`utt2spkFile`   _if None and spk2uttFile != None, raise error, default = None_  
`outFile`   _if it is a file name, save result as file and return file path, or return KaldiArk, default = False_  


### compute_cmvn_stats(feat,outFile,_**other parameters_) 

< function description >

Compute CMVN state and save it as file. Return cmvn file path.   

< Parameters >  

`feat` _KaldiArk or KaldiDict object_
`spk2uttFile`   _if None, compute cmvn state whin each utterance, default = None_  
`outFile`   _file path name_  


### use_cmvn_sliding(feat,_**other parameters_) 

< function description >

Apply sliding CMVN to feature. Return KaldiArk object. 

< Parameters >  

`feat` _KaldiArk or KaldiDict object_  
`windowsSize`   _sliding windows width, frames, if None, set it to cover all frames at one time, default = None_   
`std`   _if False, only apply mean, default = False_  


### add_delta(feat,_**other parameters_) 

< function description >

Add n-orders delta to feature. Return KaldiArk object or file path if < outFile > is not None.

< Parameters >  

`feat` _KaldiArk or KaldiDict object_ 
`order`   _the times of delta, default = 2_ 
`outFile`   _if it is a file name, save result as file and return file path, or return KaldiArk, default = False_  

### load_ali(aliFile,hmm,_**other parameters_) 

< function description >

Get alignment from ali file. Return a KaldiDict object.

< Parameters >  

`aliFile` _kaldi alignment file path_
`hmm`   _HMM file path_ 
`returnPhoneme`   _if True, return phoneme IDs, or return pdf IDs, default = False_

### load_lat(latFile,hmm,wordSymbol) 

< function description >

Get alignment from ali file. Return a KaldiDict object.
The same as KaldiLattice().load() method.

### analyze_counts(aliFile,outFile,_**other parameters_) 

< function description >

Get statistical information of pdf IDs or phoneme IDs from ali file.

< Parameters >  

`aliFile` _Kaldi alignment file path_
`outFile` _outFile path_ 
`countPhone`   _if True, count statistical value of phoneme IDs, or count pdf IDs, default = False_
`hmm` _if None, find HMM file automatically, default = None_
`dim` _if None, compute dimension automatically, default = None_

### decompress(data) 

< function description >

Decompress feature data. Feat are expected KaldiArk object whose data type is "CM", that is kaldi compressed ark data. Return a KaldiArk object. 
This function is a cover of kaldi-io-for-python tools. For more information about it, please access to https://github.com/vesis84/kaldi-io-for-python/blob/master/kaldi_io/kaldi_io.py 

< Parameters >  

`data` _the binary data of kaldi compressed feature_

### decode_lattice(amp,hmm,hclg,wordSymbol,_**other parameters_) 

< function description >

Decode by generating lattice from acoustic probability. Return a KaldiLattice object or file path if < outFile > is not None. 
We provide some usual options, but if you want use more, set < config > = your-configure. Note that if you do this, these usual configures we provided will be ignored. 
You can use ExKaldi.check_config('decode-lattice') function to get configure information you could set. 
Also run shell command "latgen-faster-mapped" to look their meaning.
  
< Parameters >  

`amp` _acoustic model log-like probability, it should be a KaldiArk object_    
`hmm`   _HMM file path_    
`hclg`   _HCLG file path_    
`wordSymbol`   _word-to-int-ID file path_    
`minActive`   _minimum active, default=200_    
`maxMem`   _maximum memory, default=50000000_    
`maxActive`   _maximum active, default=7000_    
`beam`   _beam, default=10_  
`latBeam`   _lattice beam, default=8_  
`acwt`   _acoustic model weight, default=1_  
`config`   _another configure setting method_    
`maxThreads`   _the numbers of decode thread, default=1_      
`outFile`   _if it is a file name, save result as file and return file path, or return KaldiLattice object, default = False_    

### run_shell_cmd(cmd,_**other parameters_) 

< function description >

We provided a basic way to run shell command. Return binary string (out,err).

< Parameters >  

`cmd` _shell command, string_  
`inputs`   _inputs data, string, default=None_  

### get_kaldi_path() 

< function description >

return Kaldi toolkit root path if Kaldi has been found or set up, or return None.
It is a initialized action implemented automatically when importing ExKaldi toolkit.

### get_env() 

< function description >

return the current environment which exkaldi is running at.

### set_kaldi_path(path) 

< function description >

set kaldi root path manually. If another Kaldi had already been running rightly. replace it. 

### check_config(name,config=None) 

< function description >

Get default configure if < config > is None, or check if given < config > has a right format. This function will read "conf" file which is located in "./", so if there is not, it will raise error. Also you can change the content of "conf" file with expected format.

< Parameters >  

`name` _object name you want to check. such as "compute_mfcc"_     
`config` _if none, return defalut configure, or chenk the format of configure and return True if correct_  

### split_file(filePath,_**other parameters_) 

< function description >

Split a large scp file into n smaller files. The splited files will be put at the same folder as original file and return their paths as a list.

< Parameters >  

`filePath` _scp file path_       
`chunks` _expected numbers, must be larger than 1, default=2_    

### pad_sequence(data,_**other parameters_) 

< function description >

Pad a batch sequences in order to train sequential neural network model such as RNN, LSTM.
Not that the first dimension of padded data is sequence.

< Parameters >  

`data` _a list whose members are batch of sequences_       
`shuffle` _If True, pad each sequence by randomly deciding its start position, Or start position is 0. default=False_
`pad` _padded value, default=0_   

### unpack_padded_sequence(data,lengths,_**other parameters_) 

< function description >

It is a reverse operation of ExKaldi.pad_sequence function. 

< Parameters >  

`data` _NumPy array which the first dimension is expected as sequence or batch size_       
`lengths` _It should has the same format of the lengths-output of pad_sequence function_
`batchSizeDim` _assign the dimension that batch size is. default=1_   

### wer(ref,hyp,_**other parameters_) 

< function description >

Compute WER (word error rate) score between prediction result and reference text. 
Return a Python dict object with score information like: {'WER':0,'allWords':10,'ins':0,'del':0,'sub':0,'SER':0,'wrongSentences':0,'allSentences':1,'missedSentences':0}
Both < hyp > and < ref > can be text file or list object. 

< Parameters >  

`hyp` _prediction result file or result-list which obtained from KaldiLattice.get_1best_words function_       
`ref` _reference text file or result-like-list_     
`mode` _score mode, default=present_  
`ignore` _ignore some symbol such as "sil", default=None_  

### accuracy(ref,hyp,_**other parameters_) 

< function description >

Compute one-one match score. for example predict is (1,2,3,4), and label is (1,2,2,4), the score will be 0.75.
Both < ref > and < hyp > will be flattened before scoring. 

< Parameters >  

`ref` _iterative object such as list, tuple or flattened NumPy array_       
`hyp` _iterative object like ref_     
`ignore` _ignore some symbol such as padded 0, default=None_  
`mode` _if mode == all, compute one-one matching score. if present, compute appearing score. default=all_  

### edit_distance(ref,hyp,_**other parameters_) 

< function description >

Compute edit distance score between two objects.
Both < ref > and < hyp > will be flattened before scoring.

< Parameters >  

`ref` _iterative object such as list, tuple or flattened NumPy array_       
`hyp` _iterative object like ref_  
`ignore` _ignore some symbol, default=None_  

### log_softmax(data,**other parameters_) 

< function description >

Compute the log-softmax value of a NumPy array data.

< Parameters >  

`data` _NumPy array_       
`axis` _demension, default=1_ 

### DataIterator(scpFiles,processFunc,batchSize,chunks='auto',otherArgs=None,shuffle=False,retainData=0.0)

< class description >

Data iterator used to train a neural network model. It will split the scp file into n chunks then manage and load them into momery alternately with parallel thread. 
It will shuffle the original scp file and split again while new epoch.

< init Parameters >

`scpFiles` _scp file(s)_ 
`processFunc` _function to process data from scp file to iterative data format, data ierator itself and scp file name will be introduced into defautly_    
`batchSize` _mini batch size_      
`chunks` _chunk number. if chunks=='auto', compute the chunks automatically. default="auto"_    
`otherArgs` _introduce other parameters into process function_      
`shuffle` _shuffle batch data, default=False_         
`retainData` _if > 0 , will reserve a part of data as valid data, default=0.0_    

< Attributes >

`.batchSize`    
return mini batch size value.

`.chunks`    
return the number of chunks.

`.chunk`    
return the index of current chunk.

`.epoch`    
return the index of current epoch.

`.isNewEpoch`    
If finishing iterating all data of current epoch, return True. Or return False

`.isNewChunk`    
If finishing iterating all data of current chunk, return True. Or return False

`.currentEpochPosition`    
Return the index position of current iteration corresponding to entire epoch.

`.epochProgress`    
Return the progress of current epoch.

`.chunkProgress`    
Return the progress of current chunk.

< Methods >  

`.next()`        
Return a batch of data. it is a list object.

`.get_retained_data(processFunc=None,batchSize=None,chunks='auto',otherArgs=None,shuffle=False,retainData=0.0)`        
Return a new DataIterator object if data was retained before. Or raise error.
If these parameters are None, use the same value with main iterator.

### Supporter(outDir='Result')

< class description >

Supporter is a class to help to manage training information such as the change of loss and plot them to log file and standard output. 

< init Parameters >

`outDir` _out floder, model and log file will be saved here, default="Result"_

< Attributes >

`finalArch`   
_return the last saved model path_  

< Methods >

`send_report(x)`   
Send information and these information will be retained untill count the statistics.

`collect_report(keys=None,plot=True)`   
Do the statistics of retaining information which are reported since from last statistics. The result will be saved in outDir/log file. 
If < keys > is not None, only collect the data in keys. If < plot > is True, print the statistics result to standard output.

`save_arch(saveFunc,archs,byKey=None,byMax=True)`   
Save model. < saveFunc > is expected and < arch > will be introduced into this function with a format (new name, object). 
If you use < byKey > and set < byMax >, model will be saved only while meeting the condition. 

`judge(key,condition,threshold,byDeltaRate=False)`   
Acording to the value reported before, judge whether condition is right. 
If < byDeltaRate > is True, use 1-order delta to judge. Or use value itself.

`dump(keepItems=False,fromLogFile=None)`   
Return training information of each epoch reported. If < fromLogFile > is not None, read these information from file.
If < keepitems > is True, return information by name of items.

