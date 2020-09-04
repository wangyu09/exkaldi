# exkaldi

This section includes various functions to load Exkaldi archives or other Exkadli objects.

-----------------------------
>## exkaldi.load_list_table
(target, name="listTable")

Generate a list table object from dict object or file.
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/core/load.py)

**Args:**  
_target_: dict object or the text file path.  
_name_: a string.  

**Return:**  
exkaldi ListTable object.

**Examples:** 
```python
utt2spk = exkaldi.load_list_table("./utt2spk")
``` 
-----------------------------
>## exkaldi.load_index_table
(target, name="index", useSuffix=None)

Load an index table from dict, or archive table file.

**Args:**  
_target_: dict object, ArkIndexTable object, bytes archive object or archive table file.  
_name_: a string.  
_useSuffix_: if _target_ is file path and not default suffix, appoint it "scp" or "ark".  

**Return:**
exkaldi ArkIndexTable object.

**Examples:** 
Exkaldi index table can discribe all bytes archives. typically, you can load the index table from a binary archive table file.
```python
feat = exkaldi.load_list_table("./feat.ark",name="mfcc")
cmvn = exkaldi.load_list_table("./cmvn.ark",name="cmvn")
``` 
Or you can load index table from .scp file.
```python
feat = exkaldi.load_list_table("./feat.scp",name="mfcc")
``` 
We will guess whether it is .ark or scp file through the file suffix. If it has not the default suffix, you need to declare it. Generally, in Kaldi, the alignment and fmllr transform matrix don't have the expected suffix.
```python
ali = exkaldi.load_list_table("./ali",name="ali",useSuffix="ark")
``` 
If the index table is loaded from file, it has the file path infomation so that you can read the real data into memory.
```python
ali = exkaldi.load_list_table("./ali",name="ali",useSuffix="ark")
ali = ali.fetch(arkType="ali")
``` 
And the table can be saved as the same format as kaldi .scp file.
```python
feat = exkaldi.load_list_table("mfcc.ark")
ali.save("mfcc.scp")
``` 
-----------------------------
>## exkaldi.load_feat
(target, name="feat", useSuffix=None)

Load feature data.

**Args:**  
_target_: Python dict object, bytes object, exkaldi feature object, .ark file, .scp file, npy file.  
_name_: a string.  
_useSuffix_: a string. When target is file path, use this to specify file.  

**Return:**  
A BytesFeature or NumpyFeature object.  

-----------------------------
>## exkaldi.load_cmvn
(target, name="cmvn", useSuffix=None)

Load CMVN statistics data.

**Args:**  
_target_: Python dict object, bytes object, exkaldi feature object, .ark file, .scp file, npy file or index table object.  
_name_: a string.  
_useSuffix_: a string. When target is file path, use this to specify file.  
**Return:**  
A BytesFeature or NumpyFeature object.

-----------------------------
>## exkaldi.load_prob
(target, name="prob", useSuffix=None)

Load post probability data.

**Args:**    
_target_: Python dict object, bytes object, exkaldi feature object, .ark file, .scp file, npy file.  
_name_: a string.  
_useSuffix_: a string. When target is file path, use this to specify file.  
**Return:**  
A BytesProbability or NumpyProbability object.  

-----------------------------
>## exkaldi.load_fmllr
(target, name="prob", useSuffix=None)

Load fmllr transform matrix data.

**Args:**  
_target_: Python dict object, bytes object, exkaldi feature object, .ark file, .scp file, npy file.  
_name_: a string.  
_useSuffix_: a string. When target is file path, use this to specify file.  
**Return:**  
A BytesFmllrMatrix or NumpyFmllrMatrix object. 

-----------------------------
>## exkaldi.load_ali
(target, aliType=None, name="ali", hmm=None)

Load alignment data.

**Args:**  
_target_: Python dict object, bytes object, exkaldi alignment object, kaldi alignment file or .npy file.  
_aliType_: None, or one of 'transitionID', 'phoneID', 'pdfID'. It will return different alignment object.  
_name_: a string.  
_hmm_: file path or exkaldi HMM object.  

**Return:**  
exkaldi alignment data objects.  

-----------------------------
>## exkaldi.load_transcription
(target, name="transcription")

Load transcription from file.

**Args:**   
_target_: transcription file path.  
_name_: a string.  

**Return:**
An exkaldi Transcription object.  

------------------------------------------
>## exkaldi.load_ngrams
(target, name="gram")

Load a ngrams from arpa or binary language model file.

**Args:**  
_target_: file path with suffix .arpa or .binary.

**Return:**  
a exkaldi KenNGrams object.

------------------------------------------
>## exkaldi.load_tree
(target, name="tree", lexicons=None)

Restorage a tree from file. The original data will be discarded.

**Args:**  
_target_: file name.  
_name_: a string.  

**Return:**  
A exkaldi DecisionTree object.

------------------------------------------
>## exkaldi.load_hmm
(target, hmmType="triphone", name="hmm", lexicons=None)

Restorage a HMM-GMM model from file.
The original data will be discarded.

**Args:**  
_target_: file name.  
_hmmType_: "monophone" or "triphone".  
_name_: a string.  
_lexicons_: None or exkaldi LexiconBank object.  

**Return:**  
A MonophoneHMM object if hmmType is "monophone", else return TriphoneHMM object.

------------------------------------------
>## exkaldi.load_mat
(matrixFile)

Read a matrix from file.

**Args:**  
_matrixFile_: matrix file path.  

**Return:**  
a Numpy Matrix Object.

------------------------------------------
>## exkaldi.load_lex
(target)

Load LexiconBank object from file.

**Args:**  
_target_: file name with suffix .lex.

**Return:**  
an exkaldi LexiconBank object.

------------------------------------------
>## exkaldi.load_lat
(target, name="lat")

Load lattice data from file.

**Args:**  
_target_: bytes object, file path or exkaldi lattice object.
_name_: a string.  

**Return:**  
An exkaldi lattice object.

------------------------------------------
>## exkaldi.load_args
(target)

Load arguments from file.

**Args:**  
_target_: exkaldi argument file name.

**Return:**  
An args object.