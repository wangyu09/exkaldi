# exkaldi

This section includes various classes to discribe archives.

--------------------------------------------

>>## exkaldi.ListTable
(data={},name="table") 

This is a subclass of Python dict.
You can use it to hold kaldi text format tables, such as scp-files, utt2spk and so on.  
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/core/archives.py)

>### .is_void

Check whether or not this table is void.

**Return:**  
A bool value.

>### .name

Get it's name. 

**Return:**  
A string.

>### .data

Get it's inner data (the original dict object). 

**Return:**  
A python dict object.

>### .rename
(name)

Rename it.

**Args:**  
_name_: a string.

>### .reset_data
(data=None)

Reset it's data.

**Args:**  
_name_: a string. If None, clear this list table.

>### .sort
(by="key",reverse=False)

Sort by key or by value.

**Args:**  
_by_: "key" or "value".   
_reverse_: If reverse, sort in descending order.   

**Return:**
A new ListTable object.  

>### .save
(fileName=None, chunks=1, concatFunc=None)

Save to file.

**Args:**   
_fileName_: file name, opened file handle or None.  
_chunks_: If > 1, split data averagely and save them. This does only when _fileName_ is a filename.  
_concatFunc_: Depending on tasks, you can use a special function to concatenate key and value to be a string. If None, defaultly: key + space + value.

**Return:**
If _fileName_ is None, return a string including all contents of this ListTable. Or return file name or file handle.

>### .shuffle
()

Random shuffle the list table.

**Return:**  
A new ListTable object.

>### .subset
(nHead=0, nTail=0, nRandom=0, chunks=1, keys=None)

Subset. Only one mode will do when it is not the default value. 
The priority order is: nHead > nTail > nRandom > chunks > keys.

**Args:**  
_nHead_: If it > 0, extract N head utterances.  
_nTail_: If it > 0, extract N tail utterances.  
_nRandom_: If it > 0, randomly sample N utterances.  
_chunks_: If it > 1, split data into N chunks.  
_keys_: If it is not None, pick out these records whose ID in keys.  

**Return:**  
a new ListTable object or a list of new ListTable objects.

>### .keys

Get all keys.

**Return:**  
A list.

>### .\_\_add\_\_
(other)

Integrate two ListTable objects. If key existed in both two objects, the former will be retained.

**Args:**   
_other_: another ListTable object. 

**Return:**    
A new ListTable object.  

>### .reverse
()

Exchange the position of key and value.
Key and value must be one-one matching, or Error will be raised.

**Return:**    
A new ListTable object. 

--------------------------------------

>>## exkaldi.Transcription
(data={},name="transcription") 

Inherited from `ListTable`.
This is used to hold transcription text, such as decoding n-best. 

>### .is_void

Inherited from `ListTable().is_void`.

>### .name

Inherited from `ListTable().name`.

>### .data

Inherited from `ListTable().data`.

>### .rename
(name)

Inherited from `ListTable().rename`.

>### .reset_data
(data=None)

Inherited from `ListTable().reset_data`.

>### .keys

Inherited from `ListTable().keys`.

>### .utts

The same as `.keys`.

>### .sort
(by="utt", reverse=False)

Sort transcription by utterance ID or sentence.

**Args:**  
_by_: "key"/"utt", "value"/"sentence", or "sentenceLength".
_reverse_: If reverse, sort in descending order.

**Return:**  
a new ListTable object.

>### .save
(fileName=None, chunks=1, discardUttID=False)

Save to file.

**Args:**   
_fileName_: file name, opened file handle or None.  
_chunks_: If > 1, split data averagely and save them. This does only when _fileName_ is a filename.  
_discardUttID_: If True, discard the ifno of utterance IDs.  

**Return:**  
If _fileName_ is None, return a string including all contents of this ListTable. Or return file name or file handle.

>### .shuffle
()

Inherited from `ListTable().shuffle`.

>### .subset
(nHead=0, nTail=0, nRandom=0, chunks=1, keys=None)

Inherited from `ListTable().subset`.

>### .\_\_add\_\_
(other)

Integrate two transcription objects. If utt-ID existed in both two objects, the former will be retained.

**Args:**  
_other_: another Transcription object.

**Return:**  
A new Transcription object.

>### .convert
(symbolTable, unkSymbol=None, ignore=None)

Convert transcription between two types of symbol, typically text format and int format.

**Args:**  
_symbolTable_: exkaldi ListTable object.  
_unkSymbol_: symbol of oov. If symbol is out of table, use this to replace.  
_ignore_: a string or list/tuple of strings. Skip some symbols.  

**Return:**  
A new Transcription object.

**Examples:**  
For example, if we want to convert transcription from txt format to int format.
```python
words = exakldi.load_list_table("words.txt")

newTrans=trancription.convert(symbolTable=words,unkSymbol="unk")
```

>### .sentence_length
()

Count the length of each sentence ( It will count the numbers of inner space ).
 
**Return:**    
A Metric object.

**Examples:**  
```python
print( transcription.sentence_length() )
```

>### .count_word
()

Count the number of each word. You can use this method to get the words list.

**Return:**  
a Metric object.

------------------------------

>>## exkaldi.Metric
(data={},name="metric") 

Inherited from `ListTable`.
This is used to hold the Metrics, such as AM or LM scores. 
The data format in Metric is: { utterance ID : int or float score,  }

>### .is_void

Inherited from `ListTable().is_void`.

>### .name

Inherited from `ListTable().name`.

>### .data

Inherited from `ListTable().data`.

>### .rename
(name)

Inherited from `ListTable().rename`.

>### .reset_data
(data=None)

Inherited from `ListTable().reset_data`.

>### .keys

Inherited from `ListTable().keys`.

>### .shuffle
()

Inherited from `ListTable().shuffle`.

>### .sort
(by="utt", reverse=False)

Sort by utterance ID or score.

**Args:**
_by_: "key" or "score"/"value".  
_reverse_: If reverse, sort in descending order.  

**Return:**  
A new Metric object.

>### .subset
(nHead=0, nTail=0, nRandom=0, chunks=1, keys=None)

Inherited from `ListTable().subset`.

>### .\_\_add\_\_
(other)

Integrate two Metric objects. If key existed in both two objects, the former will be retained.

**Args:**  
_other_: another Metric object.

**Return:**  
A new Metric object.

>### .sum
(weight=None)

Compute the weighted sum of all scores.

**Args:**  
_weight_: a dict object or Metric object.  

**Return:**  
A float value.

>### .mean
(weight=None,epsilon=1e-8)

Compute the weighted average of all scores.

**Args:**  
_weight_: a dict object or Metric object.   
_epsilon_: a extreme small value.  

**Return:**   
A float value.  

>### .max
()

Get the maximum score.

**Return:**   
A float or int value.  

>### .argmax
()

Get the key of maximum score.

**Return:**   
A key.  

>### .min
()

Get the minimum score.

**Return:**   
A float or int value.  

>### .argmin
()

Get the key of minimum score.

**Return:**   
A key.  

>### .map
(func)

Map a function to all scores.

**Args:**   
_func_:A callable object or function.  

**Return:**   
A new Metric object.

-----------------------------------

>>## exkaldi.ArkIndexTable
(data={},name="indexTable") 

Inherited from `ListTable`.
This is used to hold the utterance index informat of Kaldi archive table (binary format). It just like the script-table file but is more useful. Its format like this:
{ "utt0": namedtuple(frames=100, startIndex=1000, dataSize=10000, filePath="./feat.ark") }

>### .is_void

Inherited from `ListTable().is_void`.

>### .name

Inherited from `ListTable().name`.

>### .data

Inherited from `ListTable().data`.

>### .rename
(name)

Inherited from `ListTable().rename`.

>### .reset_data
(data=None)

Inherited from `ListTable().reset_data`.

>### .keys

Inherited from `ListTable().keys`.

>### .utts

the same as `.keys`.

>### .shuffle
()

Inherited from `ListTable().shuffle`.
  
**Return:**   
A list of strings.

>### .sort
(by="utt", reverse=False)

Sort this index table.

**Args:**     
_by_: "frame"/"value" or "key"/"utt" or "startIndex" or "filePath".  
_reverse_: If True, sort in descending order.  

**Return:**   
A new ArkIndexTable object.

>### .\_\_add\_\_
(other)

Integrate two index table objects. If key existed in both two objects, the former will be retained.

**Args:**  
_other_: another ArkIndexTable object.

**Return:**  
A new ArkIndexTable object.

>### .save
(fileName=None, chunks=1)

Save this index informat to text file with kaidi script-file table format.
Note that the frames info will be discarded.

**Args:**  
_fileName_: file name or file handle.  
_chunks_: an positive int value. If > 1, split it into N chunks and save them. This option only work when _fileName_ is a file name.  

**Return:**  
file name or None or a string.

>### .subset
(nHead=0, nTail=0, nRandom=0, chunks=1, keys=None)

Inherited from `ListTable().subset`.

>### .fetch
(arkType=None, uttIDs=None)

Fetch data from file acording to index information.
the information of file path in the indexTable is necessary. 

**Args:**  
_uttID_: utterance ID or a list of utterance IDs.  
_arkType_: None or "feat" or "cmvn" or "prob" or "ali" or "fmllrMat" or "mat" or "vec".   

**Return:**   
an exkaldi bytes achieve object depending on the _arkType_. 

**Examples:**  
```python
featIndexTable = exkaldi.load_index_table("mfcc.scp")  
feat = featIndexTable.fetch(arkType="feat")
```
------------------------------------------------

>>## exkaldi.BytesMatrix
(data=b"", name="mat", indexTable=None)

Hold the feature with kaldi binary format.

**Initial Args:**   
_data_: bytes or BytesMatrix or NumpyMatrix or ArkIndexTable or their subclass object. If it's BytesMatrix or ArkIndexTable object (or their subclasses), the _indexTable_ option will not work.If it's NumpyMatrix or bytes object (or their subclasses), generate index table automatically if _indexTable_ is not provided.  
_name_: a string.  
_indexTable_: python dict or ArkIndexTable object.  

>### .is_void

Check whether or not this is a void object.

**Return:**  
A bool value.

>### .name

Get it's name. 

**Return:**  
A string.

>### .data

Get it's inner data. 

**Return:**  
A python dict object.

>### .rename
(name)

Rename it.

**Args:**  
_name_: a string.

>### .reset_data
(data=None)

Reset it's data.

**Args:**  
_name_: a string. If None, clear this archive.

>### .keys

Get its keys.

**Return:**   
a list of strings.

>### .indexTable

Get the index table of its inner bytes data.

**Return:**   
an exkaldi ArkIndexTable object.

>### .dtype

Get its data type.

**Return:**   
a string, "float32" or "float64".

>### .to_dtype
(dtype)

Transform its data type.

**Args:**   
_dtype_: a string, "float", "float32" or "float64".  

**Return:**   
a new feature object.

>### .dim

Get its dimensions.

**Return:**   
a int value.

>### .check_format
()

Check whether or not it has the right kaldi format.

**Return:**   
True if done. Or raise specified error.

>### .lens

Get the number of utterances.

**Return:**   
An int value.

>### .save
(fileName, chunks=1, returnIndexTable=False)

Save bytes data to file.

**Args:**  
_fileName_: file name or file handle. If it's a file name, suffix ".ark" will be add to the name defaultly.  
_chunks_: If larger than 1, data will be saved to multiple files averagely. This would be invalid when _fileName_ is a file handle.  
_returnIndexTable_: If True, return the index table containing the information of file path.  

**Return:**  
file path, file handle or ArkIndexTable object. 

>### .to_numpy
()

Transform bytes data to numpy data.

**Return:**  
a NumpyMatrix object sorted by utterance ID.

>### .\_\_add\_\_
(other)

The plus operation between two matrix objects.

**Args:**  
_other_: a BytesFeature or NumpyFeature or ArkIndexTable object or their subclasses object.  

**Return:**  
a new BytesMatrix object.  

>### .subset
(nHead=0, nTail=0, nRandom=0, chunks=1, keys=None)

Subset data.
The priority of mode is nHead > nTail > nRandom > chunks > keys.
If you chose multiple modes, only the prior one will work.

**Args:**  
_nHead_: get N head utterances.  
_nTail_: get N tail utterances.  
_nRandom_: sample N utterances randomly.  
_chunks_: split data into N chunks averagely.  
_keys_: pick out these utterances whose ID in keys.  

**Return:**  
a new BytesMatrix object or a list of new BytesMatrix objects.

>### .\_\_call\_\_
(utt)

Pick out the specified utterance.

**Args:**  
_utt_:a string.  

**Return:**  
If existed, return a new BytesMatrix object. Or return None.

>### .sort
(by="utt", reverse=False)

Sort.

**Args:**    
_by_: "frame"/"value" or "utt"/"spk"/"key".  
_reverse_: If reverse, sort in descending order.  

**Return:**  
A new BytesFeature object.  

---------------------------------------------

>>## exkaldi.BytesFeature
(data=b"", name="feat", indexTable=None)

Inherited from `BytesMatrix`.
Hold the feature with bytes format.

**Initial Args:**   
_data_: bytes or BytesFeature or NumpyFeature or ArkIndexTable object.  
_name_: a string.  
_indexTable_: python dict or ArkIndexTable object.  

>### .is_void

The same as `BytesMatrix().is_void`.

>### .name

The same as `BytesMatrix().name`.

>### .data

The same as `BytesMatrix().data`.

>### .rename
(name)

The same as `BytesMatrix().rename`.

>### .reset_data
(data=None)

The same as `BytesMatrix().reset_data`.

>### .keys

Inherited from `BytesMatrix().keys`.

>### .utts

the same as `.keys`.

>### .indexTable

Inherited from `BytesMatrix().indexTable`.

>### .dtype

Inherited from `BytesMatrix().dtype`.

>### .to_dtype
(dtype)

Inherited from `BytesMatrix().to_dtype`.

>### .dim

Inherited from `BytesMatrix().dim`.

>### .check_format
()

Inherited from `BytesMatrix().check_format`.

>### .lens

Inherited from `BytesMatrix().lens`.

>### .save
(fileName, chunks=1, returnIndexTable=False)

Inherited from `BytesMatrix().save`.

>### .to_numpy
()

Inherited from `BytesMatrix().to_numpy`.

>### .\_\_add\_\_
(other)

The plus operation between two feature objects.

**Args:**  
_other_: a BytesFeature or NumpyFeature or ArkIndexTable object.  

**Return:**  
a new BytesFeature object.  

>### .subset
(nHead=0, nTail=0, nRandom=0, chunks=1, keys=None)

Inherited from `BytesMatrix().subset`.

>### .\_\_call\_\_
(utt)

The same as `BytesMatrix().__call__`.

>### .sort
(by="utt", reverse=False)

Inherited from `BytesMatrix().sort`.

>### .splice
(left=1, right=None)

Splice front-behind N frames to generate new feature.

**Args:**  
_left_: the left N-frames to splice.  
_right_: the right N-frames to splice. If None, right = left.  

**Return**:  
A new BytesFeature object whose dim became original-dim * (1 + left + right).  

>### .select
(dims, retain=False)

Select specified dimensions of feature.

**Args:**
_dims_: A int value or string such as "1,2,5-10".  
_retain_: If True, return the rest dimensions of feature simultaneously.  

**Return:**  
A new BytesFeature object or two BytesFeature objects.

**Examples:**  
Select the specified by an int value.
```python
newFeat = feat.select(dims=0)
```
For more complicated selections, you can specified the _dims_ with a string.
```python
newFeat = feat.select(dims="1-12,25-")
```
If you want to get the selected feature and non-selected feature, set the _retain_ True.
```python
feat1, feat2 = feat.select(dims=0, reatin=True)
```

>### .add_delta
(order=2)

Add N orders delta informat to feature.

**Args:**  
_order_: A positive int value.  

**Return:**  
A new BytesFeature object whose dimention became original-dim * (1 + order). 

>### .paste
(others)

Paste feature in feature dimension level.

**Args:**  
_others_: a feature object or list of feature objects.  

**Return:**  
a new feature object.

-----------------------------

>>## exkaldi.BytesCMVNStatistics
(data=b"", name="cmvn", indexTable=None)

Inherited from `BytesMatrix`.
Hold the CMVN statistics with bytes format.

**Initial Args:**   
_data_: bytes or BytesCMVNStatistics or NumpyCMVNStatistics or ArkIndexTable object.  
_name_: a string.  
_indexTable_: python dict or ArkIndexTable object.  

>### .is_void

The same as `BytesMatrix().is_void`.

>### .name

The same as `BytesMatrix().name`.

>### .data

The same as `BytesMatrix().data`.

>### .rename
(name)

The same as `BytesMatrix().rename`.

>### .reset_data
(data=None)

The same as `BytesMatrix().reset_data`.

>### .keys

Inherited from `BytesMatrix().keys`.

>### .spks

the same as `.keys`.

>### .indexTable

Inherited from `BytesMatrix().indexTable`.

>### .dtype

Inherited from `BytesMatrix().dtype`.

>### .to_dtype
(dtype)

Inherited from `BytesMatrix().to_dtype`.

>### .dim

Inherited from `BytesMatrix().dim`.

>### .check_format
()

Inherited from `BytesMatrix().check_format`.

>### .lens

Inherited from `BytesMatrix().lens`.

>### .save
(fileName, chunks=1, returnIndexTable=False)

Inherited from `BytesMatrix().save`.

>### .to_numpy
()

Inherited from `BytesMatrix().to_numpy`.

>### .\_\_add\_\_
(other)

The plus operation between two CMVN objects.

**Args:**  
_other_: a BytesCMVNStatistics or NumpyCMVNStatistics or ArkIndexTable object.  

**Return:**  
a new BytesCMVNStatistics object.  

>### .subset
(nHead=0, nTail=0, nRandom=0, chunks=1, keys=None)

Inherited from `BytesMatrix().subset`.

>### .\_\_call\_\_
(spk)

The same as `BytesMatrix().__call__`.

>### .sort
(by="spk", reverse=False)

Inherited from `BytesMatrix().sort`.
But _by_ noly has one mode, that is "key"/"spk".

**Args:**  
_reverse_: a bool value.

**Return:**  
a new BytesCMVNStatistics object.

-----------------------------

>>## exkaldi.BytesProbability
(data=b"", name="prob", indexTable=None)

Inherited from `BytesMatrix`.
Hold the probability with bytes format.

**Initial Args:**   
_data_: bytes or BytesFeature or NumpyFeature or ArkIndexTable object.  
_name_: a string.  
_indexTable_: python dict or ArkIndexTable object.  

>### .is_void

The same as `BytesMatrix().is_void`.

>### .name

The same as `BytesMatrix().name`.

>### .data

The same as `BytesMatrix().data`.

>### .rename
(name)

The same as `BytesMatrix().rename`.

>### .reset_data
(data=None)

The same as `BytesMatrix().reset_data`.

>### .keys

Inherited from `BytesMatrix().keys`.

>### .utts

the same as `.keys`.

>### .indexTable

Inherited from `BytesMatrix().indexTable`.

>### .dtype

Inherited from `BytesMatrix().dtype`.

>### .to_dtype
(dtype)

Inherited from `BytesMatrix().to_dtype`.

>### .dim

Inherited from `BytesMatrix().dim`.

>### .check_format
()

Inherited from `BytesMatrix().check_format`.

>### .lens

Inherited from `BytesMatrix().lens`.

>### .save
(fileName, chunks=1, returnIndexTable=False)

Inherited from `BytesMatrix().save`.

>### .to_numpy
()

Inherited from `BytesMatrix().to_numpy`.

>### .\_\_add\_\_
(other)

The plus operation between two probability objects.

**Args:**  
_other_: a BytesProbability or NumpyProbability or ArkIndexTable object.  

**Return:**  
a new BytesProbability object.  

>### .subset
(nHead=0, nTail=0, nRandom=0, chunks=1, keys=None)

Inherited from `BytesMatrix().subset`.

>### .\_\_call\_\_
(utt)

The same as `BytesMatrix().__call__`.

>### .sort
(by="utt", reverse=False)

Inherited from `BytesMatrix().sort`.

-----------------------------

>>## exkaldi.BytesFmllrMatrix
(data=b"", name="fmllrMat", indexTable=None)

Inherited from `BytesMatrix`.
Hold the fmllr transform matrix with bytes format.

**Initial Args:**   
_data_: bytes or BytesFmllrMatrix or NumpyFmllrMatrix or ArkIndexTable object.  
_name_: a string.  
_indexTable_: python dict or ArkIndexTable object.  

>### .is_void

The same as `BytesMatrix().is_void`.

>### .name

The same as `BytesMatrix().name`.

>### .data

The same as `BytesMatrix().data`.

>### .rename
(name)

The same as `BytesMatrix().rename`.

>### .reset_data
(data=None)

The same as `BytesMatrix().reset_data`.

>### .keys

Inherited from `BytesMatrix().keys`.

>### .utts

the same as `.keys`.

>### .indexTable

Inherited from `BytesMatrix().indexTable`.

>### .dtype

Inherited from `BytesMatrix().dtype`.

>### .to_dtype
(dtype)

Inherited from `BytesMatrix().to_dtype`.

>### .dim

Inherited from `BytesMatrix().dim`.

>### .check_format
()

Inherited from `BytesMatrix().check_format`.

>### .lens

Inherited from `BytesMatrix().lens`.

>### .save
(fileName, chunks=1, returnIndexTable=False)

Inherited from `BytesMatrix().save`.

>### .to_numpy
()

Inherited from `BytesMatrix().to_numpy`.

>### .\_\_add\_\_
(other)

The plus operation between two fmllr matrix objects.

**Args:**  
_other_: a BytesFmllrMatrix or NumpyFmllrMatrix or ArkIndexTable object.  

**Return:**  
a new BytesFmllrMatrix object.  

>### .subset
(nHead=0, nTail=0, nRandom=0, chunks=1, keys=None)

Inherited from `BytesMatrix().subset`.

>### .\_\_call\_\_
(utt)

The same as `BytesMatrix().__call__`.

>### .sort
(by="utt", reverse=False)

Inherited from `BytesMatrix().sort`.

-----------------------------

>>## exkaldi.BytesVector
(data=b"", name="vec", indexTable=None)

Hold the vector data with kaldi binary format.
This class has almost the same attributes and methods with `BytesMatrix`.

**Initial Args:**   
_data_: bytes or BytesVector or NumpyVector or ArkIndexTable or their subclass object. If it's BytesVector or ArkIndexTable object (or their subclasses), the _indexTable_ option will not work.If it's NumpyVector bytes object (or their subclasses), generate index table automatically if _indexTable_ is not provided.  
_name_: a string.  
_indexTable_: python dict or ArkIndexTable object.  

>### .is_void

The same as `BytesMatrix().is_void`.

>### .name

The same as `BytesMatrix().name`.

>### .data

The same as `BytesMatrix().data`.

>### .rename
(name)

The same as `BytesMatrix().rename`.

>### .reset_data
(data=None)

The same as `BytesMatrix().reset_data`.

>### .keys

The same as `BytesMatrix().keys`.

>### .utts

The same as `.keys`.

>### .indexTable

The same as `BytesMatrix().indexTable`.

>### .dtype

Get the dtype of vector. In current Exkaldi, we only use vector is int32.

**Return:**  
None or "int32".

>### .dim

Get the dimensionality of this vector. Defaultly, it should be 1.

**Return:**  
None or 1.

>### .check_format
()

The same as `BytesMatrix().check_format`.

>### .lens

The same as `BytesMatrix().lens`.

>### .save
(fileName, chunks=1, returnIndexTable=False)

The same as `BytesMatrix().save`.

>### .to_numpy
()

The same as `BytesMatrix().to_numpy`.

>### .\_\_add\_\_
(other)

The plus operation between two vector objects.

**Args:**  
_other_: a BytesVector or NumpyVector or ArkIndexTable object or their subclass objects.  

**Return:**  
a new BytesVector object.  

>### .subset
(nHead=0, nTail=0, nRandom=0, chunks=1, keys=None)

The same as `BytesMatrix().subset`.

>### .\_\_call\_\_
(utt)

The same as `BytesMatrix().__call__`.

>### .sort
(by="utt", reverse=False)

The same as `BytesMatrix().sort`.

-----------------------------

>>## exkaldi.BytesAlignmentTrans
(data=b"", name="transitionID", indexTable=None)

Inherited from `BytesVector`.
Hold the transition ID alignment with kaldi binary format.

**Initial Args:**   
_data_: bytes or BytesAlignmentTrans or NumpyAlignmentTrans or ArkIndexTable object.  
_name_: a string.  
_indexTable_: python dict or ArkIndexTable object.  

>### .is_void

Inherited from `BytesVector().is_void`.

>### .name

Inherited from `BytesVector().name`.

>### .data

Inherited from `BytesVector().data`.

>### .rename
(name)

Inherited from `BytesVector().rename`.

>### .reset_data
(data=None)

Inherited from `BytesVector().reset_data`.

>### .keys

Inherited from `BytesVector().keys`.

>### .utts

The same as `.keys`.

>### .indexTable

Inherited from `BytesVector().indexTable`.

>### .dtype

Inherited from `BytesVector().dtype`.

>### .dim

Inherited from `BytesVector().dim`.

**Return:**  
None or 1.

>### .check_format
()

Inherited from `BytesVector().check_format`.

>### .lens

Inherited from `BytesVector().lens`.

>### .save
(fileName, chunks=1, returnIndexTable=False)

Inherited from `BytesVector().save`.

>### .to_numpy
(aliType="transitionID", hmm=None)

Transform alignment to Numpy format.

**Args:**  
_aliType_: "transitionID" or "pdfID" or "phoneID".
_hmm_: file name or Exkaldi HMM object.

**Return:**  
If _aliType_ is "transitionID", return a NumpyAlignmentTrans object.
If _aliType_ is "pdfID", return a NumpyAlignmentPdf object.
If _aliType_ is "phoneID", return a NumpyAlignmentPhone object.

>### .\_\_add\_\_
(other)

The plus operation between two transition ID alignment objects.

**Args:**  
_other_: a BytesAlignmentTrans or NumpyAlignmentTrans or ArkIndexTable object.  

**Return:**  
a new BytesAlignmentTrans object.  

>### .subset
(nHead=0, nTail=0, nRandom=0, chunks=1, keys=None)

Inherited from `BytesVector().subset`.

>### .\_\_call\_\_
(utt)

Inherited from `BytesVector().__call__`.

>### .sort
(by="utt", reverse=False)

Inherited from `BytesVector().sort`.

-------------------------------

>>## exkaldi.NumpyMatrix
(data={}, name=None)

Hold the matrix data with NumPy format.
It has almost the same attributes and methods with `BytesMatrix`.

**Initial Args:**
_data_: a dict, BytesMatrix, NumpyMatrix or ArkIndexTable (or their subclasses) object.
_name_: a string.

>### .is_void

Check whether or not this is a void object.

**Return:**  
A bool value.

>### .name

Get it's name. 

**Return:**  
A string.

>### .data

Get it's inner data. 

**Return:**  
A python dict object.

>### .rename
(name)

Rename it.

**Args:**  
_name_: a string.

>### .reset_data
(data=None)

Reset it's data.

**Args:**  
_name_: a string. If None, clear this archive.

>### .keys

Get all keys with list format. 

**Return:**  
a list of all keys.

>### .items

Get a iterator of the items.

**Return:**  
a iterator.

>### .dtype

Get the data type of Numpy data.
		
**Return:**  
A string, 'float32', 'float64'.

>### .dim

Get the data dimensions.

**Return:**  
If data is void, return None, or return an int value.

>### .to_dtype
(dtype)

Transform data type.

**Args:**  
_dtype_: a string of "float", "float32" or "float64". IF "float", it will be treated as "float32".  

**Return:**  
A new NumpyMatrix object.

>### .check_format
()

Check if data has right kaldi format.

**Return:**  
If data is void, return False.
If data has right format, return True, or raise Error.

>### .to_bytes
()

Transform numpy data to bytes format.  

**Return:**  
a BytesMatrix object.  

>### .save
(fileName, chunks=1)

Save numpy data to file.

**Args:**  
_fileName_: file name. Defaultly suffix ".npy" will be add to the name.  
_chunks_: If larger than 1, data will be saved to multiple files averagely.	  

**Return:**
the path of saved files.

>### .\_\_add\_\_
(other)

The Plus operation between two objects.

**Args:**  
_other_: a BytesMatrix or NumpyMatrix or ArkIndexTable (or their subclassed) object.

**Return:**  
a new NumpyMatrix object.

>### .\_\_call\_\_
(utt)

Pick out the specified utterance.

**Args:**  
_utt_:a string.  

**Return:**  
If existed, return a new NumpyMatrix object. Or return None.

>### .lens

Get the number of utterances.

**Return:**  
a int value.

>### .subset
(nHead=0, nTail=0, nRandom=0, chunks=1, keys=None)

Subset data.
The priority of mode is nHead > nTail > nRandom > chunks > keys.
If you chose multiple modes, only the prior one will work.

**Args:**  
_nHead_: get N head utterances.  
_nTail_: get N tail utterances.  
_nRandom_: sample N utterances randomly.  
_chunks_: split data into N chunks averagely.  
_keys_: pick out these utterances whose ID in keys.  

**Return:**  
a new NumpyMatrix object or a list of new NumpyMatrix objects.

>### .sort
(by='utt', reverse=False)

Sort.

**Args:**  
_by_: "frame"/"value" or "utt"/"key"/"spk"
_reverse_: If reverse, sort in descending order.

**Return:**  
A new NumpyMatrix object.

>### .map
(func)

Map all arrays to a function and get the result.

**Args:**  
_func_: callable function object.

**Return:**  
A new NumpyMatrix object.

-------------------------------

>>## exkaldi.NumpyFeature
(data={}, name=None)

Inherited from `NumpyMatrix`.
Hold the feature data with NumPy format.

**Initial Args:**  
_data_: a dict, BytesFeature, Numpy Feature or ArkIndextable object.  
_name_: a string.  

>### .is_void

The same as `NumpyMatrix().is_void`.

>### .name

The same as `NumpyMatrix().name`.

>### .data

The same as `NumpyMatrix().data`.

>### .rename
(name)

The same as `NumpyMatrix().rename`.

>### .reset_data
(data=None)

The same as `NumpyMatrix().reset_data`.

>### .keys
()

The same as `NumpyMatrix().keys`.

>### .items
()

The same as `NumpyMatrix().items`.

>### .dtype

Inherited from  `NumpyMatrix().dtype`.

>### .dim

Inherited from  `NumpyMatrix().dim`.

>### .to_dtype
(dtype)

Inherited from  `NumpyMatrix().to_dtype`.

>### .check_format
()

Inherited from  `NumpyMatrix().check_format`.

>### .to_bytes
()

Inherited from  `NumpyMatrix().to_bytes`.

>### .save
(fileName, chunks=1)

Inherited from  `NumpyMatrix().save`.

**Return:**
the path of saved files.

>### .\_\_add\_\_
(other)

The plus operation between two feature objects.

**Args:**  
_other_: a BytesFeature or NumpyFeature or ArkIndexTable object.

**Return:**  
a new NumpyFeature object.

>### .\_\_call\_\_
(utt)

The same as `NumpyMatrix().__call__`.

>### .lens

Inherited from  `NumpyMatrix().lens`.

>### .subset
(nHead=0, nTail=0, nRandom=0, chunks=1, keys=None)

Inherited from  `NumpyMatrix().subset`.

>### .sort
(by='utt', reverse=False)

Inherited from  `NumpyMatrix().sort`.

>### .map
(func)

Inherited from  `NumpyMatrix().map`.

>### .splice
(left=4, right=None)

Splice front-behind N frames to generate new feature data.

**Args:**  
_left_: the left N-frames to splice.  
_right_: the right N-frames to splice. If None, right = left.  

**Return:**  
a new NumpyFeature object whose dim became original-dim * (1 + left + right).

>### .select
(dims, retain=False)

Select specified dimensions of feature.  

**Args:**  
_dims_: A int value or string such as "1,2,5-10".
_retain_: If True, return the rest dimensions of feature simultaneously.  

**Return:**  
A new NumpyFeature object or two NumpyFeature objects.

>### .normalize
(std=True, alpha=1.0, beta=0.0, epsilon=1e-8, axis=0)

Standerd normalize a feature at a file field.
If std is True, Do: 
_alpha * (x-mean)/(stds + epsilon) + belta_, 
or do: _alpha * (x-mean) + belta_.

**Args:**  
_std_: True of False.  
_alpha_,_beta_: a float value.  
_epsilon_: a extremely small float value.  
_axis_: the dimension to normalize.  

**Return:**  
A new NumpyFeature object.

>### .cut
(maxFrames)

Cut long utterance to multiple shorter ones. 

**Args:**  
_maxFrames_: a positive int value. Cut a utterance by the thresold value of 1.25 * maxFrames into a maxFrames part and a rest part. If the rest part is still longer than 1.25 * maxFrames, continue to cut it. 

**Return:**  
A new NumpyFeature object.  

>### .paste
(others)

Concatenate feature arrays of the same uttID from multiple objects in feature dimendion.

**Args:**  
_others_: an object or a list of objects of NumpyFeature or BytesFeature.  

**Return:**  
a new NumpyFeature objects.

----------------------------------------

>>## exkaldi.NumpyCMVNStatistics
(data={}, name="cmvn")

Inherited from `NumpyMatrix`.
Hold the CMVN statistics with NumPy format.

**Initial Args:**  
_data_: a dict, BytesCMVNStatistics, NumpyCMVNStatistics or ArkIndextable object.  
_name_: a string.  

>### .is_void

The same as `NumpyMatrix().is_void`.

>### .name

The same as `NumpyMatrix().name`.

>### .data

The same as `NumpyMatrix().data`.

>### .rename
(name)

The same as `NumpyMatrix().rename`.

>### .reset_data
(data=None)

The same as `NumpyMatrix().reset_data`.

>### .keys
()

The same as `NumpyMatrix().keys`.

>### .items
()

The same as `NumpyMatrix().items`.

>### .dtype

Inherited from  `NumpyMatrix().dtype`.

>### .dim

Inherited from  `NumpyMatrix().dim`.

>### .to_dtype
(dtype)

Inherited from  `NumpyMatrix().to_dtype`.

>### .check_format
()

Inherited from  `NumpyMatrix().check_format`.

>### .to_bytes
()

Inherited from  `NumpyMatrix().to_bytes`.

>### .save
(fileName, chunks=1)

Inherited from  `NumpyMatrix().save`.

**Return:**
the path of saved files.

>### .\_\_add\_\_
(other)

The same as `NumpyMatrix().__add__`.

>### .\_\_call\_\_
(spk)

The same as `NumpyMatrix().__call__`.


>### .subset
(nHead=0, nTail=0, nRandom=0, chunks=1, keys=None)

Inherited from  `NumpyMatrix().subset`.

>### .sort
(by='spk', reverse=False)

Inherited from  `NumpyMatrix().sort`.

>### .map
(func)

Inherited from  `NumpyMatrix().map`.

----------------------------------------

>>## exkaldi.NumpyProbability
(data={}, name="prob")

Inherited from `NumpyMatrix`.
Hold the probability with NumPy format.

**Initial Args:**  
_data_: a dict, BytesProbability, NumpyProbability or ArkIndextable object.  
_name_: a string. 

>### .is_void

The same as `NumpyMatrix().is_void`.

>### .name

The same as `NumpyMatrix().name`.

>### .data

The same as `NumpyMatrix().data`.

>### .rename
(name)

The same as `NumpyMatrix().rename`.

>### .reset_data
(data=None)

The same as `NumpyMatrix().reset_data`.

>### .keys
()

The same as `NumpyMatrix().keys`.

>### .items
()

The same as `NumpyMatrix().items`.

>### .dtype

Inherited from  `NumpyMatrix().dtype`.

>### .dim

Inherited from  `NumpyMatrix().dim`.

>### .to_dtype
(dtype)

Inherited from  `NumpyMatrix().to_dtype`.

>### .check_format
()

Inherited from  `NumpyMatrix().check_format`.

>### .to_bytes
()

Inherited from  `NumpyMatrix().to_bytes`.

>### .save
(fileName, chunks=1)

Inherited from  `NumpyMatrix().save`.

**Return:**
the path of saved files.

>### .\_\_add\_\_
(other)

The same as `NumpyMatrix().__add__`.

>### .\_\_call\_\_
(utt)

The same as `NumpyMatrix().__call__`.


>### .subset
(nHead=0, nTail=0, nRandom=0, chunks=1, keys=None)

Inherited from  `NumpyMatrix().subset`.

>### .sort
(by='utt', reverse=False)

Inherited from  `NumpyMatrix().sort`.

>### .map
(func)

Inherited from  `NumpyMatrix().map`.

----------------------------------------

>>## exkaldi.NumpyFmllrMatrix
(data={}, name="fmllrMat")

Inherited from `NumpyMatrix`.
Hold the fmllr transform matrix with NumPy format.

**Initial Args:**  
_data_: a dict, BytesFmllrMatrix, NumpyFmllrMatrix or ArkIndextable object.  
_name_: a string. 

>### .is_void

The same as `NumpyMatrix().is_void`.

>### .name

The same as `NumpyMatrix().name`.

>### .data

The same as `NumpyMatrix().data`.

>### .rename
(name)

The same as `NumpyMatrix().rename`.

>### .reset_data
(data=None)

The same as `NumpyMatrix().reset_data`.

>### .keys
()

The same as `NumpyMatrix().keys`.

>### .items
()

The same as `NumpyMatrix().items`.

>### .dtype

Inherited from  `NumpyMatrix().dtype`.

>### .dim

Inherited from  `NumpyMatrix().dim`.

>### .to_dtype
(dtype)

Inherited from  `NumpyMatrix().to_dtype`.

>### .check_format
()

Inherited from  `NumpyMatrix().check_format`.

>### .to_bytes
()

Inherited from  `NumpyMatrix().to_bytes`.

>### .save
(fileName, chunks=1)

Inherited from  `NumpyMatrix().save`.

**Return:**
the path of saved files.

>### .\_\_add\_\_
(other)

The same as `NumpyMatrix().__add__`.

>### .\_\_call\_\_
(spk)

The same as `NumpyMatrix().__call__`.


>### .subset
(nHead=0, nTail=0, nRandom=0, chunks=1, keys=None)

Inherited from  `NumpyMatrix().subset`.

>### .sort
(by='spk', reverse=False)

Inherited from  `NumpyMatrix().sort`.

>### .map
(func)

Inherited from  `NumpyMatrix().map`.

----------------------------------------

>>## exkaldi.NumpyVector
(data={}, name="vec")

Inherited from `NumpyMatrix`.
Hold the vector data with NumPy format.

**Initial Args**  
_data_: Bytesvector or ArkIndexTable object or NumpyVector or dict object (or their subclasses).
_name_: a string.

>### .is_void

The same as `NumpyMatrix().is_void`.

>### .name

The same as `NumpyMatrix().name`.

>### .data

The same as `NumpyMatrix().data`.

>### .rename
(name)

The same as `NumpyMatrix().rename`.

>### .reset_data
(data=None)

The same as `NumpyMatrix().reset_data`.

>### .keys
()

The same as `NumpyMatrix().keys`.

>### .items
()

The same as `NumpyMatrix().items`.

>### .dtype

Inherited from  `NumpyMatrix().dtype`.

>### .dim

Get the dimension of vector.

**Return:**  
None or 0.

>### .to_dtype
(dtype)

Convert the dtype of vector.

**Args:**  
_dtype_: "int", "int16", "int32" or "int64"

**Return:**  
a new NumpyVector object.

>### .check_format
()

The same as  `NumpyMatrix().check_format`.

>### .to_bytes
()

Inherited from  `NumpyMatrix().to_bytes`.
But only "int32" data can be transform to bytes format.

>### .save
(fileName, chunks=1)

Inherited from  `NumpyMatrix().save`.

**Return:**
the path of saved files.

>### .\_\_add\_\_
(other)

The same as `NumpyMatrix().__add__`.

>### .\_\_call\_\_
(utt)

The same as `NumpyMatrix().__call__`.


>### .subset
(nHead=0, nTail=0, nRandom=0, chunks=1, keys=None)

Inherited from  `NumpyMatrix().subset`.

>### .sort
(by='utt', reverse=False)

Inherited from  `NumpyMatrix().sort`.

>### .map
(func)

Inherited from  `NumpyMatrix().map`.

----------------------------------------

>>## exkaldi.NumpyAlignmentTrans
(data={}, name="transitionID")

Inherited from `NumpyVector`.
Hold the transition ID alignment with NumPy format.

**Initial Args**  
_data_: BytesAlignmentTrans or ArkIndexTable object or NumpyAlignmentTrans or dict object (or their subclasses).
_name_: a string.

>### .is_void

The same as `NumpyMatrix().is_void`.

>### .name

The same as `NumpyMatrix().name`.

>### .data

The same as `NumpyMatrix().data`.

>### .rename
(name)

The same as `NumpyMatrix().rename`.

>### .reset_data
(data=None)

The same as `NumpyMatrix().reset_data`.

>### .keys
()

The same as `NumpyMatrix().keys`.

>### .items
()

The same as `NumpyMatrix().items`.

>### .dtype

Inherited from  `NumpyMatrix().dtype`.

>### .dim

Inherited from  `NumpyVector().dtype`.

>### .to_dtype
(dtype)

Inherited from  `NumpyVector().to_dtype`.

>### .check_format
()

The same as  `NumpyMatrix().check_format`.

>### .to_bytes
()

Inherited from  `NumpyVector().to_bytes`.

>### .save
(fileName, chunks=1)

Inherited from  `NumpyMatrix().save`.

**Return:**
the path of saved files.

>### .\_\_add\_\_
(other)

The same as `NumpyMatrix().__add__`.

>### .\_\_call\_\_
(utt)

The same as `NumpyMatrix().__call__`.


>### .subset
(nHead=0, nTail=0, nRandom=0, chunks=1, keys=None)

Inherited from  `NumpyMatrix().subset`.

>### .sort
(by='utt', reverse=False)

Inherited from  `NumpyMatrix().sort`.

>### .map
(func)

Inherited from  `NumpyMatrix().map`.

----------------------------------------


>### .to_phoneID
(hmm)

Transform tansition ID alignment to phone ID format.

**Args:**  
_hmm_: exkaldi HMM object or file path.  

**Return:** 
a NumpyAlignmentPhone object.

>### .to_pdfID
(hmm)

Transform tansition ID alignment to pdf ID format.

**Args:**  
_hmm_: exkaldi HMM object or file path.  

**Return:**  
a NumpyAlignmentPhone object.  
	
>### .cut
(maxFrames)

Cut long utterance to multiple shorter ones. 

**Args:**  
_maxFrames_: a positive int value. Cut a utterance by the thresold value of 1.25 * maxFrames into a maxFrames part and a rest part.If the rest part is still longer than 1.25 * maxFrames, continue to cut it. 

**Return:**  
A new NumpyAlignmentTrans object.

----------------------------------------

>>## exkaldi.NumpyAlignment
(data={}, name="ali")

Inherited from `NumpyVector`.
Hold the alignment with NumPy format.

**Initial Args**  
_data_: NumpyAlignment or dict object (or their subclasses).
_name_: a string.

>### .is_void

The same as `NumpyMatrix().is_void`.

>### .name

The same as `NumpyMatrix().name`.

>### .data

The same as `NumpyMatrix().data`.

>### .rename
(name)

The same as `NumpyMatrix().rename`.

>### .reset_data
(data=None)

The same as `NumpyMatrix().reset_data`.

>### .keys
()

The same as `NumpyMatrix().keys`.

>### .items
()

The same as `NumpyMatrix().items`.

>### .dtype

Inherited from  `NumpyMatrix().dtype`.

>### .dim

Inherited from  `NumpyVector().dtype`.

>### .to_dtype
(dtype)

Inherited from  `NumpyVector().to_dtype`.

>### .check_format
()

The same as  `NumpyMatrix().check_format`.


>### .save
(fileName, chunks=1)

Inherited from  `NumpyMatrix().save`.

**Return:**
the path of saved files.

>### .\_\_add\_\_
(other)

The same as `NumpyMatrix().__add__`.

>### .\_\_call\_\_
(utt)

The same as `NumpyMatrix().__call__`.


>### .subset
(nHead=0, nTail=0, nRandom=0, chunks=1, keys=None)

Inherited from  `NumpyMatrix().subset`.

>### .sort
(by='utt', reverse=False)

Inherited from  `NumpyMatrix().sort`.

>### .map
(func)

Inherited from  `NumpyMatrix().map`.

>### .cut
(maxFrames)

Cut long utterance to multiple shorter ones. 

**Args:**  
_maxFrames_: a positive int value. Cut a utterance by the thresold value of 1.25 * maxFrames into a maxFrames part and a rest part.If the rest part is still longer than 1.25 * maxFrames, continue to cut it. 

**Return:**  
A new NumpyAlignment object.

----------------------------------------

>>## exkaldi.NumpyAlignmentPhone
(data={}, name="phoneID")

Inherited from `NumpyAlignment`.
Hold the phone ID alignment with NumPy format.  

**Initial Args**  
_data_: NumpyAlignment or dict object (or their subclasses).
_name_: a string.

>### .is_void

The same as `NumpyMatrix().is_void`.

>### .name

The same as `NumpyMatrix().name`.

>### .data

The same as `NumpyMatrix().data`.

>### .rename
(name)

The same as `NumpyMatrix().rename`.

>### .reset_data
(data=None)

The same as `NumpyMatrix().reset_data`.

>### .keys
()

The same as `NumpyMatrix().keys`.

>### .items
()

The same as `NumpyMatrix().items`.

>### .dtype

Inherited from  `NumpyMatrix().dtype`.

>### .dim

Inherited from  `NumpyVector().dtype`.

>### .to_dtype
(dtype)

Inherited from  `NumpyVector().to_dtype`.

>### .check_format
()

The same as  `NumpyMatrix().check_format`.


>### .save
(fileName, chunks=1)

Inherited from  `NumpyMatrix().save`.

**Return:**
the path of saved files.

>### .\_\_add\_\_
(other)

The same as `NumpyMatrix().__add__`.

>### .\_\_call\_\_
(utt)

The same as `NumpyMatrix().__call__`.


>### .subset
(nHead=0, nTail=0, nRandom=0, chunks=1, keys=None)

Inherited from  `NumpyMatrix().subset`.

>### .sort
(by='utt', reverse=False)

Inherited from  `NumpyMatrix().sort`.

>### .map
(func)

Inherited from  `NumpyMatrix().map`.

>### .cut
(maxFrames)

Inherited from  `NumpyAlignment().cut`.

----------------------------------------

>>## exkaldi.NumpyAlignmentPdf
(data={}, name="phoneID")

Inherited from `NumpyAlignment`.
Hold the pdf ID alignment with NumPy format.  
 
>### .is_void

The same as `NumpyMatrix().is_void`.

>### .name

The same as `NumpyMatrix().name`.

>### .data

The same as `NumpyMatrix().data`.

>### .rename
(name)

The same as `NumpyMatrix().rename`.

>### .reset_data
(data=None)

The same as `NumpyMatrix().reset_data`.

>### .keys
()

The same as `NumpyMatrix().keys`.

>### .items
()

The same as `NumpyMatrix().items`.

>### .dtype

Inherited from  `NumpyMatrix().dtype`.

>### .dim

Inherited from  `NumpyVector().dtype`.

>### .to_dtype
(dtype)

Inherited from  `NumpyVector().to_dtype`.

>### .check_format
()

The same as  `NumpyMatrix().check_format`.


>### .save
(fileName, chunks=1)

Inherited from  `NumpyMatrix().save`.

**Return:**
the path of saved files.

>### .\_\_add\_\_
(other)

The same as `NumpyMatrix().__add__`.

>### .\_\_call\_\_
(utt)

The same as `NumpyMatrix().__call__`.

>### .subset
(nHead=0, nTail=0, nRandom=0, chunks=1, keys=None)

Inherited from  `NumpyMatrix().subset`.

>### .sort
(by='utt', reverse=False)

Inherited from  `NumpyMatrix().sort`.

>### .map
(func)

Inherited from  `NumpyMatrix().map`.

>### .cut
(maxFrames)

Inherited from  `NumpyAlignment().cut`.
