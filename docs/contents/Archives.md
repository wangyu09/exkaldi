# exkaldi

This section includes various classes to discribe archives.

--------------------------------------------

>>## exkaldi.ListTable
(data={},name="table") 

This is a subclass of Python dict.
You can use it to hold kaldi text format tables, such as scp-files, utt2spk and so on. [view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/core/archive.py)  

You can instantiate it directly and use it as same as dict.
```python
table = exkaldi.ListTable()
table["key"] = "value"
```
Or convert a dict object to it.
```python
a = {}
a["key"] = "value"

table1 = exkaldi.ListTable(a) # the first way
table2 = exkaldi.load_list_table(a) # the second way
```

<span id="list-table-is-void"></span>

>### .is_void
Check whether or not this table is void.  

**Return:**  
A bool value.

<span id="list-table-name"></span>

>### .name

Get it's name. 

**Return:**  
A string.

<span id="list-table-data"></span>

>### .data

Get it's inner data (the original dict object). 

**Return:**  
A python dict object.

<span id="list-table-rename"></span>

>### .rename
(name)

Rename it.

**Args:**  
_name_: a string.

<span id="list-table-reset-data"></span>

>### .reset_data
(data=None)

Reset it's data.

**Args:**  
_name_: a string. If None, clear this list table.

<span id="list-table-record"></span>

>### .record
(key,value)

Add or modify a record. In ListTable, it is the same as setitem function.

**Args:**  
_key_,_value_: python objects.

```python
table = exkaldi.ListTable()

table["1"] = 1
# this is the same as the following:
table.record("1",1)
```
However, in other classes, this function may be different.

<span id="list-table-sort"></span>

>### .sort
(by="key",reverse=False)

Sort by key or by value.This is just a pseudo sort operation for the dict object but it works after python 3.6.

**Args:**  
_by_: "key" or "value".   
_reverse_: If reverse, sort in descending order.   

**Return:**
A new ListTable object.  

<span id="list-table-save"></span>

>### .save
(fileName=None, chunks=1, concatFunc=None)

Save to file.

**Args:**   
_fileName_: file name, opened file handle or None.  
_chunks_: If > 1, split data averagely and save them. This does only when _fileName_ is a filename.  
_concatFunc_: Depending on tasks, you can use a special function to concatenate key and value to be a string. If None, defaultly: key + space + value.

**Return:**
If _fileName_ is None, return a string including all contents of this ListTable. Or return file name or file handle.

<span id="list-table-shuffle"></span>

>### .shuffle
()

Random shuffle the list table.

**Return:**  
A new ListTable object.

<span id="list-table-subset"></span>

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

<span id="list-table-add"></span>

>### .\_\_add\_\_
(other)

Integrate two ListTable objects. If key existed in both two objects, the former will be retained.

**Args:**   
_other_: another ListTable object. 

**Return:**    
A new ListTable object.  

<span id="list-table-reverse"></span>

>### .reverse
()

Exchange the position of key and value.
Key and value must be one-one matching, or Error will be raised.

**Return:**    
A new ListTable object. 

--------------------------------------

>>## exkaldi.Transcription
(data={},name="transcription") 

Inherited from `ListTable`, is also a subclass of Python dict class.
This is used to hold transcription text, such as decoding n-best. [view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/core/archive.py)  

Usually you can load the file directly to get a Transcription object.
```python
trans = exkaldi.load_transcription("train/text")
```
If you want to make one by yourself.
```python
trans = exkaldi.Transcription()
trans["utt-id-1"] = "this is a example"
```

>### .is_void

Inherited from [`ListTable().is_void`](#list-table-is-void). If this is a void object, return True.

>### .name

Inherited from [`ListTable().name`](#list-table-name).
Get its name.

>### .data

Inherited from [`ListTable().data`](#list-table-data).
Get its inner data (a dict object).

>### .rename
(name)

Inherited from [`ListTable().rename`](#list-table-rename)..
Rename it.

>### .reset_data
(data=None)

Inherited from [`ListTable().reset_data`](#list-table-reset-data).
Clear it or reset change its data.

>### .record
(key,value)

Add or modify a record. In Transcription, it is the same as setitem function.

**Args:**
_key_,_value_: strings.

>### .utts

Get all utterance IDs.

**Return:**
A list of strings.

>### .sort
(by="utt", reverse=False)

Sort transcription by utterance ID or sentence or sentence length.

**Args:**  
_by_: "key"/"utt", "value"/"sentence", or "sentenceLength".
_reverse_: If reverse, sort in descending order.

**Return:**  
a new ListTable object.

```python
trans = trans.sort(by="utt")
```

>### .\_\_add\_\_
(other)

Integrate two transcription objects. If utt-ID existed in both two objects, the former will be retained.

**Args:**  
_other_: another Transcription object.

**Return:**  
A new Transcription object.

>### .shuffle
()

Inherited from [`ListTable().shuffle`](#list-table-shuffle). Random shuffle the transcription.

>### .subset
(nHead=0, nTail=0, nRandom=0, chunks=1, keys=None)

Inherited from [`ListTable().subset`](#list-table-subset). Subset this transcription.

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

newTrans = trans.convert(symbolTable=words,unkSymbol="unk")
```

>### .sentence_length
()

Count the length of each sentence ( It will count the numbers of inner space ).
 
**Return:**    
A Metric object.

```python
print( trans.sentence_length() )
```

>### .save
(fileName=None, chunks=1, discardUttID=False)

Save to file.

**Args:**   
_fileName_: file name, opened file handle or None.  
_chunks_: If > 1, split data averagely and save them. This does only when _fileName_ is a filename.  
_discardUttID_: If True, discard the ifno of utterance IDs.  

**Return:**  
If _fileName_ is None, return a string including all contents of this ListTable. Or return file name or file handle.

>### .count_word
()

Count the number of each word. You can use this method to get the words list.

**Return:**  
a Metric object.

------------------------------

>>## exkaldi.Metric
(data={},name="metric") 

Inherited from `ListTable`. It is also a subclass of Python dict class. This is used to hold the Metrics, such as AM or LM scores. The data format in Metric is: { utterance ID : int or float score,  } [view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/core/archive.py)  

>### .is_void

Inherited from [`ListTable().is_void`](#list-table-is-void). If this is a void object, return True.

>### .name

Inherited from [`ListTable().name`](#list-table-name). Get its name.

>### .data

Inherited from [`ListTable().data`](#list-table-data). Get its inner data (a python dict object).

>### .rename
(name)

Inherited from [`ListTable().rename`](#list-table-rename). Rename it.

>### .reset_data
(data=None)

Inherited from [`ListTable().reset_data`](#list-table-reset-data). Clear it or change its data.

>### .record
(key,value)

Add or modify a record. In Metric, it is the same as setitem function.

**Args:**  
_key_:a python object.  
_value_: an int or float value.

>### .sort
(by="score", reverse=False)

Sort by utterance ID or score.

**Args:**
_by_: "key" or "score"/"value".  
_reverse_: If reverse, sort in descending order.  

**Return:**  
A new Metric object.

>### .\_\_add\_\_
(other)

Integrate two Metric objects. If key existed in both two objects, the former will be retained.

**Args:**  
_other_: another Metric object.

**Return:**  
A new Metric object.

>### .shuffle
()

Inherited from [`ListTable().shuffle`](#list-table-shuffle). Randomly shuffle it.

>### .subset
(nHead=0, nTail=0, nRandom=0, chunks=1, keys=None)

Inherited from [`ListTable().subset`](#list-table-subset). Subset it.

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

>>## exkaldi.IndexTable
(data={},name="indexTable") 

Inherited from `ListTable`. It is also a subclass of Python dict class.
This is used to hold the utterance index informat of Kaldi archive table (binary format). It just like the script-table file but is more useful. Its format like this:
{ "utt0": namedtuple(frames=100, startIndex=1000, dataSize=10000, filePath="./feat.ark") } [view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/core/archive.py)  

>### .is_void

Inherited from [`ListTable().is_void`](#list-table-is-void). If this is a void object, return True.

>### .name

Inherited from [`ListTable().name`](#list-table-name). Get its name.

>### .data

Inherited from [`ListTable().data`](#list-table-data). Get its inner data (a python dict object).

>### .rename
(name)

Inherited from [`ListTable().rename`](#list-table-rename). Rename it.

>### .reset_data
(data=None)

Inherited from [`ListTable().reset_data`](#list-table-reset-data). Clear it or change its data.

>### .record
(key,frames=None,startIndex=None,dataSize=None,filePath=None)

Add or modify a record. In IndexTable, this function is different with setitem function.

**Args:**  
_key_: a string, utterance ID.  
_frames_: an int value, the frames of current utterance.
_startIndex_: an int value, the start position of current utterance.
_dataSize_: an int value, the dataSize of current utterance.
_filePath_: a string, the file path of current utterance.

>### .sort
(by="utt", reverse=False)

Sort this index table.

**Args:**     
_by_: "frame"/"value" or "key"/"utt" or "startIndex" or "filePath".  
_reverse_: If True, sort in descending order.  

**Return:**   
A new IndexTable object.

>### .\_\_add\_\_
(other)

Integrate two index table objects. If key existed in both two objects, the former will be retained.

**Args:**  
_other_: another IndexTable object.

**Return:**  
A new IndexTable object.

>### .shuffle
()

Inherited from [`ListTable().shuffle`](#list-table-shuffle). Randomly shuffle it.
  
>### .subset
(nHead=0, nTail=0, nRandom=0, chunks=1, keys=None)

Inherited from [`ListTable().subset`](#list-table-subset). Subset it.

>### .save
(fileName=None, chunks=1)

Save this index informat to text file with kaidi script-file table format.Note that the frames info will be discarded.

**Args:**  
_fileName_: file name or file handle.  
_chunks_: an positive int value. If > 1, split it into N chunks and save them. This option only work when _fileName_ is a file name.  

**Return:**  
file name or None or a string.

```python
table.save("test.scp")
```

>### .fetch
(arkType=None, uttIDs=None)

Fetch data from file acording to index information.
the information of file path in the indexTable is necessary. 

**Args:**  
_uttID_: utterance ID or a list of utterance IDs.  
_arkType_: None or "feat" or "cmvn" or "prob" or "ali" or "fmllr" or "mat" or "vec".   

**Return:**   
an exkaldi bytes achieve object depending on the _arkType_. 

**Examples:**  
```python
featIndexTable = exkaldi.load_index_table("mfcc.scp")  
feat = featIndexTable.fetch(arkType="feat")
```

>### .utts

Get all utterance IDs.

**Return:**  
a list of strings.

>### .spks

The same with `.utts`.

------------------------------------------------

>>## exkaldi.WavSegment
(data={},name="segment") 

Inherited from `ListTable`. It is also a subclass of Python dict class.
It is designed to hold wave segment info. In current version. WavSegment object cannot be loaded from file, so you have to make it by you self.[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/core/archive.py)

>### .is_void

Inherited from [`ListTable().is_void`](#list-table-is-void). If this is a void object, return True.

>### .name

Inherited from [`ListTable().name`](#list-table-name). Get its name.

>### .data

Inherited from [`ListTable().data`](#list-table-data). Get its inner data (a python dict object).

>### .rename
(name)

Inherited from [`ListTable().rename`](#list-table-rename). Rename it.

>### .reset_data
(data=None)

Inherited from [`ListTable().reset_data`](#list-table-reset-data). Clear it or change its data.

>### record
(key=None,fileID=None,startTime=None,endTime=None,filePath=None,text=None)

Add or modify a record.

**Args:**  
_key_: a string. The utterance ID. If None, we will make it by: fileID-startTime-endTime.  
_fileID_: a string. the file ID.  
_startTime_: an float value. Seconds.  
_endTime_: an float value. Seconds.  
_filePath_: wav file path. a string.  
_dataSize_: an int value. The total size of an archive record. Including the utterance ID.  
_filePath_: a string. The total size of an archive record. Including the utterance ID.  
_text_: the transcription.  

>### .sort
(by="utt", reverse=False)

Sort this index table.

**Args:**     
_by_: "key"/"utt", "startTime"/"value" or "filePath".  
_reverse_: If True, sort in descending order.  

**Return:**   
A new WavSegment object.

>### .\_\_add\_\_
(other)

Integrate two wav segment objects. If key existed in both two objects, the former will be retained.

**Args:**  
_other_: another WavSegment object.

**Return:**  
A new WavSegment object.

>### .shuffle
()

Inherited from [`ListTable().shuffle`](#list-table-shuffle). Randomly shuffle it.

>### .subset
(nHead=0, nTail=0, nRandom=0, chunks=1, keys=None)

Inherited from [`ListTable().subset`](#list-table-subset). Subset it.

>### .save
(fileName=None, chunks=1)

Save this segment to file with Kaldi segment file format.
Note that the _filePath_ and _text_ information will be discarded. If you want to save them, please detach them respectively.

**Args:**  
_fileName_: file name or file handle.  
_chunks_: an positive int value. If > 1, split it into N chunks and save them. This option only work when _fileName_ is a file name.  

**Return:**  
file name or None or a string.

>### .detach_wav
()

Detach file ID - wav file path information from segments.

**Return:**  
an exkaldi ListTable object.

>### .detach_transcription
()

Detach utterance ID - text information from segments.

**Return:**  
an exkaldi Transcription object.

>### .extract_segment
(outDir=None)

Extract segment and save them to file. If _outDir_ is None, save the segment wav file in the same as original wav file.

**Return:**  
an exkaldi ListTable object. The generated wav.scp .

>### .utts

Get all utterance IDs.

**Return:** 
a list of strings.

------------------------------------------------

>>## exkaldi.BytesMatrix
(data=b"", name="mat", indexTable=None)

Hold the feature with kaldi binary format.[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/core/archive.py)

**Initial Args:**   
_data_: bytes or BytesMatrix or NumpyMatrix or IndexTable or their subclass object. If it's BytesMatrix or IndexTable object (or their subclasses), the _indexTable_ option will not work.If it's NumpyMatrix or bytes object (or their subclasses), generate index table automatically if _indexTable_ is not provided.  
_name_: a string.  
_indexTable_: python dict or IndexTable object.  

<span id="bytes-matrix-is-void"></span>

>### .is_void

Check whether or not this is a void object.

**Return:**  
A bool value.

<span id="bytes-matrix-name"></span>

>### .name

Get it's name. 

**Return:**  
A string.

<span id="bytes-matrix-data"></span>

>### .data

Get it's inner data. 

**Return:**  
A python dict object.

<span id="bytes-matrix-rename"></span>

>### .rename
(name)

Rename it.

**Args:**  
_name_: a string.

<span id="bytes-matrix-reset-data"></span>

>### .reset_data
(data=None)

Reset it's data.

**Args:**  
_name_: a string. If None, clear this archive.

<span id="bytes-matrix-index-table"></span>

>### .indexTable

Get the index table of its inner bytes data.

**Return:**   
an exkaldi IndexTable object.

<span id="bytes-matrix-dtype"></span>

>### .dtype

Get its data type.

**Return:**   
a string, "float32" or "float64".

<span id="bytes-matrix-to-dtype"></span>

>### .to_dtype
(dtype)

Transform its data type.

**Args:**   
_dtype_: a string, "float", "float32" or "float64".  

**Return:**   
a new BytesMatrix object.

<span id="bytes-matrix-dim"></span>

>### .dim

Get its dimensions.

**Return:**   
a int value.

<span id="bytes-matrix-keys"></span>

>### .keys
()

Get its keys (utterance IDs or Speaker IDs). It is a generator.

**Return:**  
a generator.

<span id="bytes-matrix-check-format"></span>

>### .check_format
()

Check whether or not it has the right kaldi format.

**Return:**   
True if done. Or raise specified error.

<span id="bytes-matrix-lens"></span>

>### .lens

Get the number of utterances.

**Return:**   
An int value.

<span id="bytes-matrix-save"></span>

>### .save
(fileName, chunks=1, returnIndexTable=False)

Save bytes data to file.

**Args:**  
_fileName_: file name or file handle. If it's a file name, suffix ".ark" will be add to the name defaultly.  
_chunks_: If larger than 1, data will be saved to multiple files averagely. This would be invalid when _fileName_ is a file handle.  
_returnIndexTable_: If True, return the index table containing the information of file path.  

**Return:**  
file path, file handle or IndexTable object. 

<span id="bytes-matrix-to-numpy"></span>

>### .to_numpy
()

Transform bytes data to numpy data.

**Return:**  
a NumpyMatrix object sorted by utterance ID.

<span id="bytes-matrix-add"></span>

>### .\_\_add\_\_
(other)

The plus operation between two matrix objects.

**Args:**  
_other_: a BytesFeat or NumpyFeat or IndexTable object or their subclasses object.  

**Return:**  
a new BytesMatrix object.  

<span id="bytes-matrix-subset"></span>

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

<span id="bytes-matrix-call"></span>

>### .\_\_call\_\_
(utt)

Pick out the specified utterance.

**Args:**  
_utt_:a string.  

**Return:**  
If existed, return a new BytesMatrix object. Or return None.

<span id="bytes-matrix-sort"></span>

>### .sort
(by="utt", reverse=False)

Sort.

**Args:**    
_by_: "frame"/"value" or "utt"/"spk"/"key".  
_reverse_: If reverse, sort in descending order.  

**Return:**  
A new BytesMatrix object.  

---------------------------------------------

>>## exkaldi.BytesFeat
(data=b"", name="feat", indexTable=None)

Inherited from `BytesMatrix`.
Hold the feature with bytes format.[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/core/archive.py)

**Initial Args:**   
_data_: bytes or BytesFeat or NumpyFeat or IndexTable object.  
_name_: a string.  
_indexTable_: python dict or IndexTable object.  

>### .is_void

The same as [`BytesMatrix().is_void`](#bytes-matrix-is-void). If this is a void object, return True.

>### .name

The same as [`BytesMatrix().name`](#bytes-matrix-name). Get its name.

>### .data

The same as [`BytesMatrix().data`]](#bytes-matrix-data). Get its data.

>### .rename
(name)

The same as [`BytesMatrix().rename`](#bytes-matrix-rename). Rename it.

>### .reset_data
(data=None)

The same as [`BytesMatrix().reset_data`](#bytes-matrix-reset-data).Clear it or change its data.

>### .keys

Inherited from [`BytesMatrix().keys`](#bytes-matrix-keys). Get a generator of all utterance IDs.

>### .utts

Get all utterance IDs.

**Return:**  
a list of strings.

>### .indexTable

Inherited from [`BytesMatrix().indexTable`]](#bytes-matrix-index-table). Get the index table.

>### .dtype

Inherited from [`BytesMatrix().dtype`](#bytes-matrix-dtype). Get its data type.

>### .to_dtype
(dtype)

Inherited from [`BytesMatrix().to_dtype`](#bytes-matrix-to-dtype). Change its data type.

>### .dim

Inherited from [`BytesMatrix().dim`]](#bytes-matrix-dim). Get the data dimensions.

>### .check_format
()

Inherited from [`BytesMatrix().check_format`](#bytes-matrix-check-format). Check data format.

>### .lens

Inherited from [`BytesMatrix().lens`](#bytes-matrix-lens). Get the numbers of utterances.

>### .save
(fileName, chunks=1, returnIndexTable=False)

Inherited from [`BytesMatrix().save`](#bytes-matrix-save). Save feature to .ark file.

>### .to_numpy
()

Inherited from [`BytesMatrix().to_numpy`](#bytes-matrix-to-numpy). Convert feature to Numpy format.

>### .\_\_add\_\_
(other)

The plus operation between two feature objects.

**Args:**  
_other_: a BytesFeat or NumpyFeat or IndexTable object.  

**Return:**  
a new BytesFeat object.  

>### .splice
(left=1, right=None)

Splice front-behind N frames to generate new feature.

**Args:**  
_left_: the left N-frames to splice.  
_right_: the right N-frames to splice. If None, right = left.  

**Return**:  
A new BytesFeat object whose dim became original-dim * (1 + left + right).  

>### .select
(dims, retain=False)

Select specified dimensions of feature.

**Args:**
_dims_: A int value or string such as "1,2,5-10".  
_retain_: If True, return the rest dimensions of feature simultaneously.  

**Return:**  
A new BytesFeat object or two BytesFeat objects.

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

>### .subset
(nHead=0, nTail=0, nRandom=0, chunks=1, keys=None)

Inherited from [`BytesMatrix().subset`](#bytes-matrix-subset). Subset this object.

>### .\_\_call\_\_
(utt)

The same as [`BytesMatrix().__call__`](#bytes-matrix-call). Pick out a utterance.

>### .add_delta
(order=2)

Add N orders delta informat to feature. 

**Args:**  
_order_: A positive int value.  

**Return:**  
A new BytesFeat object whose dimention became original-dim * (1 + order). 

>### .paste
(others)

Paste feature in feature dimension level.

**Args:**  
_others_: a feature object or list of feature objects.  

**Return:**  
a new feature object.

>### .sort
(by="utt", reverse=False)

Inherited from [`BytesMatrix().sort`](#bytes-matrix-sort). Sort it.

-----------------------------

>>## exkaldi.BytesCMVN
(data=b"", name="cmvn", indexTable=None)

Inherited from `BytesMatrix`.
Hold the CMVN statistics with bytes format.
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/core/archive.py)

**Initial Args:**   
_data_: bytes or BytesCMVN or NumpyCMVN or IndexTable object.  
_name_: a string.  
_indexTable_: python dict or IndexTable object.  

>### .is_void

The same as [`BytesMatrix().is_void`](#bytes-matrix-is-void). If this is a void object, return True.

>### .name

The same as [`BytesMatrix().name`](#bytes-matrix-name). Get its name.

>### .data

The same as [`BytesMatrix().data`](#bytes-matrix-data). Get its data (a python bytes object.)

>### .rename
(name)

The same as [`BytesMatrix().rename`](#bytes-matrix-rename). Rename it.

>### .reset_data
(data=None)

The same as [`BytesMatrix().reset_data`](#bytes-matrix-reset-data). Clear it or change its data.

>### .spks

Get all speaker (or utterance) IDs.

**Return:**  
a list of strings.

>### .indexTable

Inherited from [`BytesMatrix().indexTable`](#bytes-matrix-index-table). Get its index table.

>### .dtype

Inherited from [`BytesMatrix().dtype`](#bytes-matrix-dtype). Get its data type.

>### .to_dtype
(dtype)

Inherited from [`BytesMatrix().to_dtype`](#bytes-matrix-to-dtype). Change its data type.

>### .dim

Inherited from [`BytesMatrix().dim`](#bytes-matrix-dim). Get its data dimensions.

>### .check_format
()

Inherited from [`BytesMatrix().check_format`](#bytes-matrix-check-format). Check its data format.

>### .lens

Inherited from [`BytesMatrix().lens`](#bytes-matrix-lens). Get the numbers of records.

>### .save
(fileName, chunks=1, returnIndexTable=False)

Inherited from [`BytesMatrix().save`](#bytes-matrix-save). Save to .ark file.

>### .to_numpy
()

Inherited from [`BytesMatrix().to_numpy`](#bytes-matrix-to-numpy). Transform to Numpy format.

>### .\_\_add\_\_
(other)

The plus operation between two CMVN objects.

**Args:**  
_other_: a BytesCMVN or NumpyCMVN or IndexTable object.  

**Return:**  
a new BytesCMVN object.  

>### .subset
(nHead=0, nTail=0, nRandom=0, chunks=1, keys=None)

Inherited from [`BytesMatrix().subset`](#bytes-matrix-subset). Subset it.

>### .\_\_call\_\_
(spk)

The same as [`BytesMatrix().__call__`](#bytes-matrix-call). Pick out one record.

>### .sort
(by="spk", reverse=False)

Inherited from [`BytesMatrix().sort`](#bytes-matrix-sort). Sort it.
But _by_ noly has one mode, that is "key"/"spk".

**Args:**  
_reverse_: a bool value.

**Return:**  
a new BytesCMVN object.

-----------------------------

>>## exkaldi.BytesProb
(data=b"", name="prob", indexTable=None)

Inherited from `BytesMatrix`.
Hold the probability with bytes format.
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/core/archive.py)

**Initial Args:**   
_data_: bytes or BytesFeat or NumpyFeat or IndexTable object.  
_name_: a string.  
_indexTable_: python dict or IndexTable object.  

>### .is_void

The same as [`BytesMatrix().is_void`](#bytes-matrix-is-void). If this is a void object, return True.

>### .name

The same as [`BytesMatrix().name`](#bytes-matrix-name). Get its name.

>### .data

The same as [`BytesMatrix().data`](#bytes-matrix-data). Get its data (a python bytes object.)

>### .rename
(name)

The same as [`BytesMatrix().rename`](#bytes-matrix-rename). Rename it.

>### .reset_data
(data=None)

The same as [`BytesMatrix().reset_data`](#bytes-matrix-reset-data). Clear it or change its data.

>### .utts

Get all utterance IDs.

**Return:**  
a list of strings.

>### .indexTable

Inherited from [`BytesMatrix().indexTable`](#bytes-matrix-index-table). Get its index table.

>### .dtype

Inherited from [`BytesMatrix().dtype`](#bytes-matrix-dtype). Get its data type.

>### .to_dtype
(dtype)

Inherited from [`BytesMatrix().to_dtype`](#bytes-matrix-to-dtype). Change its data type.

>### .dim

Inherited from [`BytesMatrix().dim`](#bytes-matrix-dim). Get its data dimensions.

>### .check_format
()

Inherited from [`BytesMatrix().check_format`](#bytes-matrix-check-format). Check its data format.

>### .lens

Inherited from [`BytesMatrix().lens`](#bytes-matrix-lens). Get the numbers of records.

>### .save
(fileName, chunks=1, returnIndexTable=False)

Inherited from [`BytesMatrix().save`](#bytes-matrix-save). Save to .ark file.

>### .to_numpy
()

Inherited from [`BytesMatrix().to_numpy`](#bytes-matrix-to-numpy). Transform to Numpy format.

>### .\_\_add\_\_
(other)

The plus operation between two probability objects.

**Args:**  
_other_: a BytesProb or NumpyProb or IndexTable object.  

**Return:**  
a new BytesProb object.  

>### .subset
(nHead=0, nTail=0, nRandom=0, chunks=1, keys=None)

Inherited from [`BytesMatrix().subset`](#bytes-matrix-subset). Subset it.

>### .\_\_call\_\_
(spk)

The same as [`BytesMatrix().__call__`](#bytes-matrix-call). Pick out one record.

>### .sort
(by="utt", reverse=False)

Inherited from [`BytesMatrix().sort`](#bytes-matrix-sort). Sort it.

-----------------------------

>>## exkaldi.BytesFmllr
(data=b"", name="fmllr", indexTable=None)

Inherited from `BytesMatrix`.
Hold the fmllr transform matrix with bytes format.
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/core/archive.py)

**Initial Args:**   
_data_: bytes or BytesFmllr or NumpyFmllr or IndexTable object.  
_name_: a string.  
_indexTable_: python dict or IndexTable object.  

>### .is_void

The same as [`BytesMatrix().is_void`](#bytes-matrix-is-void). If this is a void object, return True.

>### .name

The same as [`BytesMatrix().name`](#bytes-matrix-name). Get its name.

>### .data

The same as [`BytesMatrix().data`](#bytes-matrix-data). Get its data (a python bytes object.)

>### .rename
(name)

The same as [`BytesMatrix().rename`](#bytes-matrix-rename). Rename it.

>### .reset_data
(data=None)

The same as [`BytesMatrix().reset_data`](#bytes-matrix-reset-data). Clear it or change its data.

>### .utts

Get all utterance IDs.

**Return:**  
a list of strings.

>### .indexTable

Inherited from [`BytesMatrix().indexTable`](#bytes-matrix-index-table). Get its index table.

>### .dtype

Inherited from [`BytesMatrix().dtype`](#bytes-matrix-dtype). Get its data type.

>### .to_dtype
(dtype)

Inherited from [`BytesMatrix().to_dtype`](#bytes-matrix-to-dtype). Change its data type.

>### .dim

Inherited from [`BytesMatrix().dim`](#bytes-matrix-dim). Get its data dimensions.

>### .check_format
()

Inherited from [`BytesMatrix().check_format`](#bytes-matrix-check-format). Check its data format.

>### .lens

Inherited from [`BytesMatrix().lens`](#bytes-matrix-lens). Get the numbers of records.

>### .save
(fileName, chunks=1, returnIndexTable=False)

Inherited from [`BytesMatrix().save`](#bytes-matrix-save). Save to .ark file.

>### .to_numpy
()

Inherited from [`BytesMatrix().to_numpy`](#bytes-matrix-to-numpy). Transform to Numpy format.

>### .\_\_add\_\_
(other)

The plus operation between two fMLLR matrix objects.

**Args:**  
_other_: a BytesFmllr or NumpyFmllr or IndexTable object.  

**Return:**  
a new BytesProb object.  

>### .subset
(nHead=0, nTail=0, nRandom=0, chunks=1, keys=None)

Inherited from [`BytesMatrix().subset`](#bytes-matrix-subset). Subset it.

>### .\_\_call\_\_
(spk)

The same as [`BytesMatrix().__call__`](#bytes-matrix-call). Pick out one record.

>### .sort
(by="spk", reverse=False)

Inherited from [`BytesMatrix().sort`](#bytes-matrix-sort). Sort it.
But _by_ noly has one mode, that is "key"/"spk".

**Args:**  
_reverse_: a bool value.

**Return:**  
a new BytesCMVN object.

-----------------------------

>>## exkaldi.BytesVector
(data=b"", name="vec", indexTable=None)

Hold the vector data with kaldi binary format.
This class has almost the same attributes and methods with `BytesMatrix`.
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/core/archive.py)

**Initial Args:**   
_data_: bytes or BytesVector or NumpyVector or IndexTable or their subclass object. If it's BytesVector or IndexTable object (or their subclasses), the _indexTable_ option will not work.If it's NumpyVector bytes object (or their subclasses), generate index table automatically if _indexTable_ is not provided.  
_name_: a string.  
_indexTable_: python dict or IndexTable object.  

>### .is_void

The same as [`BytesMatrix().is_void`](#bytes-matrix-is-void). If this is a void object, return True.

>### .name

The same as [`BytesMatrix().name`](#bytes-matrix-name). Get its name.

>### .data

The same as [`BytesMatrix().data`]](#bytes-matrix-data). Get its data.

>### .rename
(name)

The same as [`BytesMatrix().rename`](#bytes-matrix-rename). Rename it.

>### .reset_data
(data=None)

The same as [`BytesMatrix().reset_data`](#bytes-matrix-reset-data).Clear it or change its data.

>### .indexTable

The same as [`BytesMatrix().indexTable`](#bytes-matrix-index-table). get its index table.

>### .dtype

Get the dtype of vector. In current ExKaldi, we only use vector is int32.

**Return:**  
None or "int32".

>### .dim

Get the dimensionality of this vector. Defaultly, it should be 1.

**Return:**  
None or 1.

>### .check_format
()

The same as [`BytesMatrix().check_format`](#bytes-matrix-check-format).Check its data format.

>### .lens

The same as [`BytesMatrix().lens`](#bytes-matrix-lens). Get the numbers of records.

>### .save
(fileName, chunks=1, returnIndexTable=False)

The same as [`BytesMatrix().save`](#bytes-matrix-save). Save to file with Kaldi format.

>### .to_numpy
()

The same as [`BytesMatrix().to_numpy`](#bytes-matrix-to-numpy). Transform to Numpy format.

>### .\_\_add\_\_
(other)

The plus operation between two vector objects.

**Args:**  
_other_: a BytesVector or NumpyVector or IndexTable object or their subclass objects.  

**Return:**  
a new BytesVector object.  

>### .subset
(nHead=0, nTail=0, nRandom=0, chunks=1, keys=None)

The same as [`BytesMatrix().subset`](#bytes-matrix-subset). Subset it.

>### .\_\_call\_\_
(utt)

The same as [`BytesMatrix().__call__`](#bytes-matrix-call). Pick out one record.

>### .sort
(by="utt", reverse=False)

The same as [`BytesMatrix().sort`](#bytes-matrix-sort). Sort it.

-----------------------------

>>## exkaldi.BytesAliTrans
(data=b"", name="transitionID", indexTable=None)

Inherited from `BytesVector`.
Hold the transition ID alignment with kaldi binary format.
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/core/archive.py)

**Initial Args:**   
_data_: bytes or BytesAliTrans or NumpyAliTrans or IndexTable object.  
_name_: a string.  
_indexTable_: python dict or IndexTable object.  

>### .is_void

The same as [`BytesMatrix().is_void`](#bytes-matrix-is-void). If this is a void object, return True.

>### .name

The same as [`BytesMatrix().name`](#bytes-matrix-name). Get its name.

>### .data

The same as [`BytesMatrix().data`]](#bytes-matrix-data). Get its data.

>### .rename
(name)

The same as [`BytesMatrix().rename`](#bytes-matrix-rename). Rename it.

>### .reset_data
(data=None)

The same as [`BytesMatrix().reset_data`](#bytes-matrix-reset-data).Clear it or change its data.

>### .indexTable

The same as [`BytesMatrix().indexTable`](#bytes-matrix-index-table). get its index table.

>### .dtype

Get the dtype of vector. In current ExKaldi, we only use vector is int32.

**Return:**  
None or "int32".

>### .dim

Get the dimensionality of this vector. Defaultly, it should be 1.

**Return:**  
None or 1.

>### .check_format
()

The same as [`BytesMatrix().check_format`](#bytes-matrix-check-format).Check its data format.

>### .lens

The same as [`BytesMatrix().lens`](#bytes-matrix-lens). Get the numbers of records.

>### .save
(fileName, chunks=1, returnIndexTable=False)

The same as [`BytesMatrix().save`](#bytes-matrix-save). Save to file with Kaldi format.

>### .\_\_add\_\_
(other)

The plus operation between two transition-ID objects.

**Args:**  
_other_: a BytesAliTrans or NumpyAliTrans or IndexTable object or their subclass objects.  

**Return:**  
a new BytesAliTrans object.  

>### .subset
(nHead=0, nTail=0, nRandom=0, chunks=1, keys=None)

The same as [`BytesMatrix().subset`](#bytes-matrix-subset). Subset it.

>### .\_\_call\_\_
(utt)

The same as [`BytesMatrix().__call__`](#bytes-matrix-call). Pick out one record.

>### .sort
(by="utt", reverse=False)

The same as [`BytesMatrix().sort`](#bytes-matrix-sort). Sort it.

>### .to_numpy
(aliType="transitionID", hmm=None)

Transform alignment to Numpy format.

**Args:**  
_aliType_: "transitionID" or "pdfID" or "phoneID".
_hmm_: file name or ExKaldi HMM object.

**Return:**  
If _aliType_ is "transitionID", return a NumpyAliTrans object.
If _aliType_ is "pdfID", return a NumpyAliPdf object.
If _aliType_ is "phoneID", return a NumpyAliPhone object.

-------------------------------

>>## exkaldi.NumpyMatrix
(data={}, name=None)

Hold the matrix data with NumPy format.
It has almost the same attributes and methods with `BytesMatrix`.
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/core/archive.py)

**Initial Args:**
_data_: a dict, BytesMatrix, NumpyMatrix or IndexTable (or their subclasses) object.
_name_: a string.

<span id="numpy-matrix-is-void"></span>

>### .is_void

Check whether or not this is a void object.

**Return:**  
A bool value.

<span id="numpy-matrix-name"></span>

>### .name

Get it's name. 

**Return:**  
A string.

<span id="numpy-matrix-data"></span>

>### .data

Get it's inner data (a python dict object). 

**Return:**  
A python dict object.

<span id="numpy-matrix-rename"></span>

>### .rename
(name)

Rename it.

**Args:**  
_name_: a string.

<span id="numpy-matrix-reset-data"></span>

>### .reset_data
(data=None)

Reset it's data.

**Args:**  
_name_: a string. If None, clear this archive.

<span id="numpy-matrix-keys"></span>

>### .keys
()

Get the keys iterator.

**Return:**  
a iterator.

<span id="numpy-matrix-items"></span>

>### .items
()

Get an iterator of the items.

**Return:**  
an items iterator.

<span id="numpy-matrix-values"></span>

>### .values
()

Get an iterator of the values.

**Return:**  
an values iterator.

<span id="numpy-matrix-array"></span>

>### .array

Get all arrays.

**Return:**  
a list of Numpy arrays.

<span id="numpy-matrix-dtype"></span>

>### .dtype

Get the data type of Numpy data.
		
**Return:**  
A string, 'float32', 'float64'.

<span id="numpy-matrix-dim"></span>

>### .dim

Get the data dimensions.

**Return:**  
If data is void, return None, or return an int value.

<span id="numpy-matrix-to-dtype"></span>

>### .to_dtype
(dtype)

Transform data type.

**Args:**  
_dtype_: a string of "float", "float32" or "float64". If "float", it will be treated as "float32".  

**Return:**  
A new NumpyMatrix object.

<span id="numpy-matrix-check-format"></span>

>### .check_format
()

Check if data has right kaldi format.

**Return:**  
If data is void, return False.
If data has right format, return True, or raise Error.

<span id="numpy-matrix-to-bytes"></span>

>### .to_bytes
()

Transform numpy data to bytes format.  

**Return:**  
a BytesMatrix object.  

<span id="numpy-matrix-save"></span>

>### .save
(fileName, chunks=1)

Save numpy data to file.

**Args:**  
_fileName_: file name. Defaultly suffix ".npy" will be add to the name.  
_chunks_: If larger than 1, data will be saved to multiple files averagely.	  

**Return:**
the path of saved files.

<span id="numpy-matrix-add"></span>

>### .\_\_add\_\_
(other)

The Plus operation between two objects.

**Args:**  
_other_: a BytesMatrix or NumpyMatrix or IndexTable (or their subclassed) object.

**Return:**  
a new NumpyMatrix object.

<span id="numpy-matrix-call"></span>

>### .\_\_call\_\_
(utt)

Pick out the specified utterance.

**Args:**  
_utt_:a string.  

**Return:**  
If existed, return a new NumpyMatrix object. Or return None.

<span id="numpy-matrix-lens"></span>

>### .lens

Get the number of utterances.

**Return:**  
a int value.

<span id="numpy-matrix-subset"></span>

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

<span id="numpy-matrix-sort"></span>

>### .sort
(by='utt', reverse=False)

Sort.

**Args:**  
_by_: "frame"/"value" or "utt"/"key"/"spk"
_reverse_: If reverse, sort in descending order.

**Return:**  
A new NumpyMatrix object.

<span id="numpy-matrix-map"></span>

>### .map
(func)

Map all arrays to a function and get the result.

**Args:**  
_func_: callable function object.

**Return:**  
A new NumpyMatrix object.

>### .__getitem__
(key)

Specify a key and get its array.

**Args:**  
_key_: a string.

**Return:**  
a Numpy array object.

```python
test = exkaldi.NumpyMatrix({"utt-1": np.array([[1.,2.],[3.,4.]]])})
print( test["utt-1"] )
```
The difference between this function and `.__call__` is that the later will return an NumpyMatrix object. 

>### .__setitem__
(key,value)

Set an item.

**Args:**  
_key_: a string, the utterance or speaker ID.
_value_: a 2-d array. It must has the same dimension with current matrix object.

```python
test = exkaldi.NumpyMatrix()
test["utt-1"] = np.array([[1.,2.],[3.,4.]]])
```

-------------------------------

>>## exkaldi.NumpyFeat
(data={}, name=None)

Inherited from `NumpyMatrix`.
Hold the feature data with NumPy format.
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/core/archive.py)

**Initial Args:**  
_data_: a dict, BytesFeat, Numpy Feat or IndexTable object.  
_name_: a string.  

>### .is_void

The same as [`NumpyMatrix().is_void`](#numpy-matrix-is-void). If this is a void object, return True.

>### .name

The same as [`NumpyMatrix().name`](#numpy-matrix-name). Get its name.

>### .data

The same as [`NumpyMatrix().data`](#numpy-matrix-data). Get its data (an dict object).

>### .rename
(name)

The same as [`NumpyMatrix().rename`](#numpy-matrix-rename). Rename it.

>### .reset_data
(data=None)

The same as [`NumpyMatrix().reset_data`](#numpy-matrix-reset-data). Clear it or change its data.

>### .keys
()

The same as [`NumpyMatrix().keys`](#numpy-matrix-keys). Get the keys iterator.

>### .items
()

The same as [`NumpyMatrix().items`](#numpy-matrix-items). Get the items iterator.

>### .values
()

The same as [`NumpyMatrix().values`](#numpy-matrix-values). Get the values iterator.

>### .dtype

Inherited from  [`NumpyMatrix().dtype`](#numpy-matrix-dtype). Get the data type.

>### .dim

Inherited from  [`NumpyMatrix().dim`](#numpy-matrix-dim). get the data dimentions.

>### .to_dtype
(dtype)

Inherited from  [`NumpyMatrix().to_dtype`](#numpy-matrix-to-dtype). Change data type.

>### .check_format
()

Inherited from  [`NumpyMatrix().check_format`]](#numpy-matrix-check-format). Check data format.

>### .to_bytes
()

Inherited from  [`NumpyMatrix().to_bytes`](#numpy-matrix-to-bytes). Transform to bytes format.

>### .save
(fileName, chunks=1)

Inherited from  [`NumpyMatrix().save`](#numpy-matrix-save). Save as .npy file.

**Return:**
the path of saved files.

>### .\_\_add\_\_
(other)

The plus operation between two feature objects.

**Args:**  
_other_: a BytesFeat or NumpyFeat or IndexTable object.

**Return:**  
a new NumpyFeat object.

>### .\_\_call\_\_
(utt)

The same as [`NumpyMatrix().__call__`](#numpy-matrix-call). Pick out a record.

>### .lens

Inherited from  [`NumpyMatrix().lens`](#numpy-matrix-lens). The numbers of records.

>### .subset
(nHead=0, nTail=0, nRandom=0, chunks=1, keys=None)

Inherited from  [`NumpyMatrix().subset`](#numpy-matrix-subset). Subset it.

>### .sort
(by='utt', reverse=False)

Inherited from  [`NumpyMatrix().sort`](#numpy-matrix-sort). Sort it.

>### .map
(func)

Inherited from  [`NumpyMatrix().map`](#numpy-matrix-map). map a function to all arrays.

>### .__getitem__
(key)

Inherited from  [`NumpyMatrix().__getitem__`](#numpy-matrix-get-item). Get the array data of a specified utterance or speaker ID.

```python
test = exkaldi.load_feat("mfcc.ark")
print( test["utt-1"] )
```
The difference between this function and `.__call__` is that the later will return an NumpyFeat object. 

>### .__setitem__
(key,value)

Inherited from  [`NumpyMatrix().__setitem__`](#numpy-matrix-set-item). Set an item.

```python
test = exkaldi.NumpyFeat()
test["utt-1"] = np.array([[1.,2.],[3.,4.]]])
```

>### .splice
(left=4, right=None)

Splice front-behind N frames to generate new feature data.

**Args:**  
_left_: the left N-frames to splice.  
_right_: the right N-frames to splice. If None, right = left.  

**Return:**  
a new NumpyFeat object whose dim became original-dim * (1 + left + right).

>### .select
(dims, retain=False)

Select specified dimensions of feature.  

**Args:**  
_dims_: A int value or string such as "1,2,5-10".
_retain_: If True, return the rest dimensions of feature simultaneously.  

**Return:**  
A new NumpyFeat object or two NumpyFeat objects.

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
A new NumpyFeat object.

>### .cut
(maxFrames)

Cut long utterance to multiple shorter ones. 

**Args:**  
_maxFrames_: a positive int value. Cut a utterance by the thresold value of 1.25 * maxFrames into a maxFrames part and a rest part. If the rest part is still longer than 1.25 * maxFrames, continue to cut it. 

**Return:**  
A new NumpyFeat object.  

>### .paste
(others)

Concatenate feature arrays of the same uttID from multiple objects in feature dimendion.

**Args:**  
_others_: an object or a list of objects of NumpyFeat or BytesFeat.  

**Return:**  
a new NumpyFeat objects.

----------------------------------------

>>## exkaldi.NumpyCMVN
(data={}, name="cmvn")

Inherited from `NumpyMatrix`.
Hold the CMVN statistics with NumPy format.
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/core/archive.py)

**Initial Args:**  
_data_: a dict, BytesCMVN, NumpyCMVN or IndexTable object.  
_name_: a string.  

>### .is_void

The same as [`NumpyMatrix().is_void`](#numpy-matrix-is-void). If this is a void object, return True.

>### .name

The same as [`NumpyMatrix().name`](#numpy-matrix-name). Get its name.

>### .data

The same as [`NumpyMatrix().data`](#numpy-matrix-data). Get its data (an dict object).

>### .rename
(name)

The same as [`NumpyMatrix().rename`](#numpy-matrix-rename). Rename it.

>### .reset_data
(data=None)

The same as [`NumpyMatrix().reset_data`](#numpy-matrix-reset-data). Clear it or change its data.

>### .keys
()

The same as [`NumpyMatrix().keys`](#numpy-matrix-keys). Get the keys iterator.

>### .items
()

The same as [`NumpyMatrix().items`](#numpy-matrix-items). Get the items iterator.

>### .values
()

The same as [`NumpyMatrix().values`](#numpy-matrix-values). Get the values iterator.

>### .dtype

Inherited from  [`NumpyMatrix().dtype`](#numpy-matrix-dtype). Get the data type.

>### .dim

Inherited from  [`NumpyMatrix().dim`](#numpy-matrix-dim). get the data dimentions.

>### .to_dtype
(dtype)

Inherited from  [`NumpyMatrix().to_dtype`](#numpy-matrix-to-dtype). Change data type.

>### .check_format
()

Inherited from  [`NumpyMatrix().check_format`]](#numpy-matrix-check-format). Check data format.

>### .to_bytes
()

Inherited from  [`NumpyMatrix().to_bytes`](#numpy-matrix-to-bytes). Transform to bytes format.

>### .save
(fileName, chunks=1)

Inherited from  [`NumpyMatrix().save`](#numpy-matrix-save). Save as .npy file.

>### .\_\_add\_\_
(other)

The plus operation between two CMVN objects.

**Args:**  
_other_: a BytesCMVN or NumpyCMVN or IndexTable object.

**Return:**  
a new NumpyCMVN object.

>### .\_\_call\_\_
(key)

The same as [`NumpyMatrix().__call__`](#numpy-matrix-call). Pick out a record.

>### .lens

Inherited from  [`NumpyMatrix().lens`](#numpy-matrix-lens). The numbers of records.

>### .subset
(nHead=0, nTail=0, nRandom=0, chunks=1, keys=None)

Inherited from  [`NumpyMatrix().subset`](#numpy-matrix-subset). Subset it.

>### .sort
(by='spk', reverse=False)

Inherited from  [`NumpyMatrix().sort`](#numpy-matrix-sort). Sort it.

>### .map
(func)

Inherited from  [`NumpyMatrix().map`](#numpy-matrix-map). map a function to all arrays.

>### .__getitem__
(key)

Inherited from  [`NumpyMatrix().__getitem__`](#numpy-matrix-get-item). Get the array data of a specified utterance or speaker ID.

>### .__setitem__
(key,value)

Inherited from  [`NumpyMatrix().__setitem__`](#numpy-matrix-set-item). Set an item.

>### .spks

Get all speaker IDs.

**Return:**  
a list of strings.

----------------------------------------

>>## exkaldi.NumpyProb
(data={}, name="prob")

Inherited from `NumpyMatrix`.
Hold the probability with NumPy format.
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/core/archive.py)

**Initial Args:**  
_data_: a dict, BytesProb, NumpyProb or IndexTable object.  
_name_: a string. 

>### .is_void

The same as [`NumpyMatrix().is_void`](#numpy-matrix-is-void). If this is a void object, return True.

>### .name

The same as [`NumpyMatrix().name`](#numpy-matrix-name). Get its name.

>### .data

The same as [`NumpyMatrix().data`](#numpy-matrix-data). Get its data (an dict object).

>### .rename
(name)

The same as [`NumpyMatrix().rename`](#numpy-matrix-rename). Rename it.

>### .reset_data
(data=None)

The same as [`NumpyMatrix().reset_data`](#numpy-matrix-reset-data). Clear it or change its data.

>### .keys
()

The same as [`NumpyMatrix().keys`](#numpy-matrix-keys). Get the keys iterator.

>### .items
()

The same as [`NumpyMatrix().items`](#numpy-matrix-items). Get the items iterator.

>### .values
()

The same as [`NumpyMatrix().values`](#numpy-matrix-values). Get the values iterator.

>### .dtype

Inherited from  [`NumpyMatrix().dtype`](#numpy-matrix-dtype). Get the data type.

>### .dim

Inherited from  [`NumpyMatrix().dim`](#numpy-matrix-dim). get the data dimentions.

>### .to_dtype
(dtype)

Inherited from  [`NumpyMatrix().to_dtype`](#numpy-matrix-to-dtype). Change data type.

>### .check_format
()

Inherited from  [`NumpyMatrix().check_format`]](#numpy-matrix-check-format). Check data format.

>### .to_bytes
()

Inherited from  [`NumpyMatrix().to_bytes`](#numpy-matrix-to-bytes). Transform to bytes format.

>### .save
(fileName, chunks=1)

Inherited from  [`NumpyMatrix().save`](#numpy-matrix-save). Save as .npy file.

>### .\_\_add\_\_
(other)

The plus operation between two probability objects.

**Args:**  
_other_: a BytesProb or NumpyProb or IndexTable object.

**Return:**  
a new NumpyProb object.

>### .\_\_call\_\_
(utt)

The same as [`NumpyMatrix().__call__`](#numpy-matrix-call). Pick out a record.

>### .lens

Inherited from  [`NumpyMatrix().lens`](#numpy-matrix-lens). The numbers of records.

>### .subset
(nHead=0, nTail=0, nRandom=0, chunks=1, keys=None)

Inherited from  [`NumpyMatrix().subset`](#numpy-matrix-subset). Subset it.

>### .sort
(by='utt', reverse=False)

Inherited from  [`NumpyMatrix().sort`](#numpy-matrix-sort). Sort it.

>### .map
(func)

Inherited from  [`NumpyMatrix().map`](#numpy-matrix-map). map a function to all arrays.

>### .__getitem__
(key)

Inherited from  [`NumpyMatrix().__getitem__`](#numpy-matrix-get-item). Get the array data of a specified utterance or speaker ID.

>### .__setitem__
(key,value)

Inherited from  [`NumpyMatrix().__setitem__`](#numpy-matrix-set-item). Set an item.

>### .utts

Get all utterance IDs.

**Return:**  
a list of strings.

----------------------------------------

>>## exkaldi.NumpyFmllr
(data={}, name="fmllr")

Inherited from `NumpyMatrix`.
Hold the fmllr transform matrix with NumPy format.
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/core/archive.py)

**Initial Args:**  
_data_: a dict, BytesFmllr, NumpyFmllr or IndexTable object.  
_name_: a string. 

>### .is_void

The same as [`NumpyMatrix().is_void`](#numpy-matrix-is-void). If this is a void object, return True.

>### .name

The same as [`NumpyMatrix().name`](#numpy-matrix-name). Get its name.

>### .data

The same as [`NumpyMatrix().data`](#numpy-matrix-data). Get its data (an dict object).

>### .rename
(name)

The same as [`NumpyMatrix().rename`](#numpy-matrix-rename). Rename it.

>### .reset_data
(data=None)

The same as [`NumpyMatrix().reset_data`](#numpy-matrix-reset-data). Clear it or change its data.

>### .keys
()

The same as [`NumpyMatrix().keys`](#numpy-matrix-keys). Get the keys iterator.

>### .items
()

The same as [`NumpyMatrix().items`](#numpy-matrix-items). Get the items iterator.

>### .values
()

The same as [`NumpyMatrix().values`](#numpy-matrix-values). Get the values iterator.

>### .dtype

Inherited from  [`NumpyMatrix().dtype`](#numpy-matrix-dtype). Get the data type.

>### .dim

Inherited from  [`NumpyMatrix().dim`](#numpy-matrix-dim). get the data dimentions.

>### .to_dtype
(dtype)

Inherited from  [`NumpyMatrix().to_dtype`](#numpy-matrix-to-dtype). Change data type.

>### .check_format
()

Inherited from  [`NumpyMatrix().check_format`]](#numpy-matrix-check-format). Check data format.

>### .to_bytes
()

Inherited from  [`NumpyMatrix().to_bytes`](#numpy-matrix-to-bytes). Transform to bytes format.

>### .save
(fileName, chunks=1)

Inherited from  [`NumpyMatrix().save`](#numpy-matrix-save). Save as .npy file.

>### .\_\_add\_\_
(other)

The plus operation between two fMLLR matrix objects.

**Args:**  
_other_: a BytesFmllr or NumpyFmllr or IndexTable object.

**Return:**  
a new NumpyFmllr object.

>### .\_\_call\_\_
(utt)

The same as [`NumpyMatrix().__call__`](#numpy-matrix-call). Pick out a record.

>### .lens

Inherited from  [`NumpyMatrix().lens`](#numpy-matrix-lens). The numbers of records.

>### .subset
(nHead=0, nTail=0, nRandom=0, chunks=1, keys=None)

Inherited from  [`NumpyMatrix().subset`](#numpy-matrix-subset). Subset it.

>### .sort
(by='utt', reverse=False)

Inherited from  [`NumpyMatrix().sort`](#numpy-matrix-sort). Sort it.

>### .map
(func)

Inherited from  [`NumpyMatrix().map`](#numpy-matrix-map). map a function to all arrays.

>### .__getitem__
(key)

Inherited from  [`NumpyMatrix().__getitem__`](#numpy-matrix-get-item). Get the array data of a specified utterance or speaker ID.

>### .__setitem__
(key,value)

Inherited from  [`NumpyMatrix().__setitem__`](#numpy-matrix-set-item). Set an item.

>### .utts

Get all speaker IDs.

**Return:**  
a list of strings.

----------------------------------------

>>## exkaldi.NumpyVector
(data={}, name="vec")

Inherited from `NumpyMatrix`.
Hold the vector data with NumPy format.
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/core/archive.py)

**Initial Args**  
_data_: Bytesvector or IndexTable object or NumpyVector or dict object (or their subclasses).
_name_: a string.

>### .is_void

The same as [`NumpyMatrix().is_void`](#numpy-matrix-is-void). If this is a void object, return True.

>### .name

The same as [`NumpyMatrix().name`](#numpy-matrix-name). Get its name.

>### .data

The same as [`NumpyMatrix().data`](#numpy-matrix-data). Get its data (an dict object).

>### .rename
(name)

The same as [`NumpyMatrix().rename`](#numpy-matrix-rename). Rename it.

>### .reset_data
(data=None)

The same as [`NumpyMatrix().reset_data`](#numpy-matrix-reset-data). Clear it or change its data.

>### .keys
()

The same as [`NumpyMatrix().keys`](#numpy-matrix-keys). Get the keys iterator.

>### .items
()

The same as [`NumpyMatrix().items`](#numpy-matrix-items). Get the items iterator.

>### .values
()

The same as [`NumpyMatrix().values`](#numpy-matrix-values). Get the values iterator.

>### .dtype

Get the vector data type.

**Return:**  
a string, "int32". (In current version, we only support int32 vector.)

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

The same as  [`NumpyMatrix().check_format`](#numpy-matrix-values). Check the vector data format.

>### .to_bytes
()

Inherited from  [`NumpyMatrix().to_bytes`](#numpy-matrix-to-bytes). Transform to bytes format.
But only "int32" data can be transform to bytes format.

>### .save
(fileName, chunks=1)

Inherited from  [`NumpyMatrix().save`](#numpy-matrix-save). Save as npy file. 

>### .\_\_add\_\_
(other)

The same as [`NumpyMatrix().__add__`](#numpy-matrix-add). Add two vector objects.

>### .\_\_call\_\_
(utt)

The same as [`NumpyMatrix().__call__`](#numpy-matrix-call). Pick out a record.

>### .subset
(nHead=0, nTail=0, nRandom=0, chunks=1, keys=None)

Inherited from  [`NumpyMatrix().subset`](#numpy-matrix-subset). Subset it.

>### .sort
(by='utt', reverse=False)

Inherited from  [`NumpyMatrix().sort`](#numpy-matrix-sort). Sort it.

>### .map
(func)

Inherited from  [`NumpyMatrix().map`](#numpy-matrix-map). map a function to all arrays.

>### .__getitem__
(key)

The same as  [`NumpyMatrix().__getitem__`](#numpy-matrix-get-item). Get the array data of a specified utterance or speaker ID.

>### .__setitem__
(key,value)

The same as [`NumpyMatrix().__setitem__`](#numpy-matrix-set-item). Set an item.
But value should be a 1-d int array.

----------------------------------------

>>## exkaldi.NumpyAliTrans
(data={}, name="transitionID")

Inherited from `NumpyVector`.
Hold the transition ID alignment with NumPy format.
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/core/archive.py)

**Initial Args**  
_data_: BytesAliTrans or IndexTable object or NumpyAliTrans or dict object (or their subclasses).
_name_: a string.

>### .is_void

The same as [`NumpyMatrix().is_void`](#numpy-matrix-is-void). If this is a void object, return True.

>### .name

The same as [`NumpyMatrix().name`](#numpy-matrix-name). Get its name.

>### .data

The same as [`NumpyMatrix().data`](#numpy-matrix-data). Get its data (an dict object).

>### .rename
(name)

The same as [`NumpyMatrix().rename`](#numpy-matrix-rename). Rename it.

>### .reset_data
(data=None)

The same as [`NumpyMatrix().reset_data`](#numpy-matrix-reset-data). Clear it or change its data.

>### .keys
()

The same as [`NumpyMatrix().keys`](#numpy-matrix-keys). Get the keys iterator.

>### .items
()

The same as [`NumpyMatrix().items`](#numpy-matrix-items). Get the items iterator.

>### .values
()

The same as [`NumpyMatrix().values`](#numpy-matrix-values). Get the values iterator.

>### .dtype

The same as [`NumpyMatrix().dtype`](#numpy-matrix-dtype). Get the data type.

>### .dim

The same as [`NumpyVector().dim`](#numpy-matrix-dim). Get the data dimensions.

>### .to_dtype
(dtype)

The same as [`NumpyVector().to_dtype`](#numpy-matrix-to-dtype). Change its data type.

>### .check_format
()

The same as  [`NumpyMatrix().check_format`](#numpy-matrix-check-format). Check the data format.

>### .to_bytes
()

The same as  [`NumpyVector().to_bytes`](#numpy-matrix-to-bytes). Transform to bytes format.

>### .save
(fileName, chunks=1)

The same as  [`NumpyMatrix().save`](#numpy-matrix-save). Save as .npy file.

>### .\_\_add\_\_
(other)

The same as [`NumpyMatrix().__add__`](#numpy-matrix-add). Add two trainsition alignment objects.

>### .\_\_call\_\_
(utt)

The same as [`NumpyMatrix().__call__`](#numpy-matrix-call). Pick out a record.

>### .subset
(nHead=0, nTail=0, nRandom=0, chunks=1, keys=None)

The same as [`NumpyMatrix().subset`](#numpy-matrix-subset). Subset it.

>### .sort
(by='utt', reverse=False)

The same as [`NumpyMatrix().sort`](#numpy-matrix-sort). Sort it.

>### .map
(func)

The same as [`NumpyMatrix().map`](#numpy-matrix-map). map a functionto all arrays.


>### .to_phoneID
(hmm)

Transform tansition ID alignment to phone ID format.

**Args:**  
_hmm_: exkaldi HMM object or file path.  

**Return:** 
a NumpyAliPhone object.

>### .to_pdfID
(hmm)

Transform tansition ID alignment to pdf ID format.

**Args:**  
_hmm_: exkaldi HMM object or file path.  

**Return:**  
a NumpyAliPhone object.  
	
>### .cut
(maxFrames)

Cut long utterance to multiple shorter ones. 

**Args:**  
_maxFrames_: a positive int value. Cut a utterance by the thresold value of 1.25 * maxFrames into a maxFrames part and a rest part.If the rest part is still longer than 1.25 * maxFrames, continue to cut it. 

**Return:**  
A new NumpyAliTrans object.

----------------------------------------

>>## exkaldi.NumpyAli
(data={}, name="ali")

Inherited from `NumpyVector`.
Hold the alignment with NumPy format.
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/core/archive.py)

**Initial Args**  
_data_: NumpyAli or dict object (or their subclasses).
_name_: a string.

>### .is_void

The same as [`NumpyMatrix().is_void`](#numpy-matrix-is-void). If this is a void object, return True.

>### .name

The same as [`NumpyMatrix().name`](#numpy-matrix-name). Get its name.

>### .data

The same as [`NumpyMatrix().data`](#numpy-matrix-data). Get its data (an dict object).

>### .rename
(name)

The same as [`NumpyMatrix().rename`](#numpy-matrix-rename). Rename it.

>### .reset_data
(data=None)

The same as [`NumpyMatrix().reset_data`](#numpy-matrix-reset-data). Clear it or change its data.

>### .keys
()

The same as [`NumpyMatrix().keys`](#numpy-matrix-keys). Get the keys iterator.

>### .items
()

The same as [`NumpyMatrix().items`](#numpy-matrix-items). Get the items iterator.

>### .values
()

The same as [`NumpyMatrix().values`](#numpy-matrix-values). Get the values iterator.

>### .dtype

The same as [`NumpyMatrix().dtype`](#numpy-matrix-dtype). Get the data type.

>### .dim

The same as [`NumpyVector().dim`](#numpy-matrix-dim). Get the data dimensions.

>### .to_dtype
(dtype)

The same as [`NumpyVector().to_dtype`](#numpy-matrix-to-dtype). Change its data type.

>### .check_format
()

The same as  [`NumpyMatrix().check_format`](#numpy-matrix-check-format). Check the data format.

>### .save
(fileName, chunks=1)

The same as  [`NumpyMatrix().save`](#numpy-matrix-save). Save as .npy file.

>### .\_\_add\_\_
(other)

The same as [`NumpyMatrix().__add__`](#numpy-matrix-add). Add two alignment objects.

>### .\_\_call\_\_
(utt)

The same as [`NumpyMatrix().__call__`](#numpy-matrix-call). Pick out a record.

>### .subset
(nHead=0, nTail=0, nRandom=0, chunks=1, keys=None)

The same as [`NumpyMatrix().subset`](#numpy-matrix-subset). Subset it.

>### .sort
(by='utt', reverse=False)

The same as [`NumpyMatrix().sort`](#numpy-matrix-sort). Sort it.

>### .map
(func)

The same as [`NumpyMatrix().map`](#numpy-matrix-map). map a functionto all arrays.

>### .cut
(maxFrames)

Cut long utterance to multiple shorter ones. 

**Args:**  
_maxFrames_: a positive int value. Cut a utterance by the thresold value of 1.25 * maxFrames into a maxFrames part and a rest part.If the rest part is still longer than 1.25 * maxFrames, continue to cut it. 

**Return:**  
A new NumpyAliTrans object.

----------------------------------------

>>## exkaldi.NumpyAliPhone
(data={}, name="phoneID")

Inherited from `NumpyAli`.
Hold the phone ID alignment with NumPy format.  
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/core/archive.py)

**Initial Args**  
_data_: NumpyAli or dict object (or their subclasses).
_name_: a string.

>### .is_void

The same as [`NumpyMatrix().is_void`](#numpy-matrix-is-void). If this is a void object, return True.

>### .name

The same as [`NumpyMatrix().name`](#numpy-matrix-name). Get its name.

>### .data

The same as [`NumpyMatrix().data`](#numpy-matrix-data). Get its data (an dict object).

>### .rename
(name)

The same as [`NumpyMatrix().rename`](#numpy-matrix-rename). Rename it.

>### .reset_data
(data=None)

The same as [`NumpyMatrix().reset_data`](#numpy-matrix-reset-data). Clear it or change its data.

>### .keys
()

The same as [`NumpyMatrix().keys`](#numpy-matrix-keys). Get the keys iterator.

>### .items
()

The same as [`NumpyMatrix().items`](#numpy-matrix-items). Get the items iterator.

>### .values
()

The same as [`NumpyMatrix().values`](#numpy-matrix-values). Get the values iterator.

>### .dtype

The same as [`NumpyMatrix().dtype`](#numpy-matrix-dtype). Get the data type.

>### .dim

The same as [`NumpyVector().dim`](#numpy-matrix-dim). Get the data dimensions.

>### .to_dtype
(dtype)

The same as [`NumpyVector().to_dtype`](#numpy-matrix-to-dtype). Change its data type.

>### .check_format
()

The same as  [`NumpyMatrix().check_format`](#numpy-matrix-check-format). Check the data format.

>### .save
(fileName, chunks=1)

The same as  [`NumpyMatrix().save`](#numpy-matrix-save). Save as .npy file.

>### .\_\_add\_\_
(other)

The same as [`NumpyMatrix().__add__`](#numpy-matrix-add). Add two alignment objects.

>### .\_\_call\_\_
(utt)

The same as [`NumpyMatrix().__call__`](#numpy-matrix-call). Pick out a record.

>### .subset
(nHead=0, nTail=0, nRandom=0, chunks=1, keys=None)

The same as [`NumpyMatrix().subset`](#numpy-matrix-subset). Subset it.

>### .sort
(by='utt', reverse=False)

The same as [`NumpyMatrix().sort`](#numpy-matrix-sort). Sort it.

>### .map
(func)

The same as [`NumpyMatrix().map`](#numpy-matrix-map). map a functionto all arrays.

>### .cut
(maxFrames)

Cut long utterance to multiple shorter ones. 

**Args:**  
_maxFrames_: a positive int value. Cut a utterance by the thresold value of 1.25 * maxFrames into a maxFrames part and a rest part.If the rest part is still longer than 1.25 * maxFrames, continue to cut it. 

**Return:**  
A new NumpyAliTrans object.

----------------------------------------

>>## exkaldi.NumpyAliPdf
(data={}, name="phoneID")

Inherited from `NumpyAli`.
Hold the pdf ID alignment with NumPy format.  
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/core/archive.py)

>### .is_void

The same as [`NumpyMatrix().is_void`](#numpy-matrix-is-void). If this is a void object, return True.

>### .name

The same as [`NumpyMatrix().name`](#numpy-matrix-name). Get its name.

>### .data

The same as [`NumpyMatrix().data`](#numpy-matrix-data). Get its data (an dict object).

>### .rename
(name)

The same as [`NumpyMatrix().rename`](#numpy-matrix-rename). Rename it.

>### .reset_data
(data=None)

The same as [`NumpyMatrix().reset_data`](#numpy-matrix-reset-data). Clear it or change its data.

>### .keys
()

The same as [`NumpyMatrix().keys`](#numpy-matrix-keys). Get the keys iterator.

>### .items
()

The same as [`NumpyMatrix().items`](#numpy-matrix-items). Get the items iterator.

>### .values
()

The same as [`NumpyMatrix().values`](#numpy-matrix-values). Get the values iterator.

>### .dtype

The same as [`NumpyMatrix().dtype`](#numpy-matrix-dtype). Get the data type.

>### .dim

The same as [`NumpyVector().dim`](#numpy-matrix-dim). Get the data dimensions.

>### .to_dtype
(dtype)

The same as [`NumpyVector().to_dtype`](#numpy-matrix-to-dtype). Change its data type.

>### .check_format
()

The same as  [`NumpyMatrix().check_format`](#numpy-matrix-check-format). Check the data format.

>### .save
(fileName, chunks=1)

The same as  [`NumpyMatrix().save`](#numpy-matrix-save). Save as .npy file.

>### .\_\_add\_\_
(other)

The same as [`NumpyMatrix().__add__`](#numpy-matrix-add). Add two alignment objects.

>### .\_\_call\_\_
(utt)

The same as [`NumpyMatrix().__call__`](#numpy-matrix-call). Pick out a record.

>### .subset
(nHead=0, nTail=0, nRandom=0, chunks=1, keys=None)

The same as [`NumpyMatrix().subset`](#numpy-matrix-subset). Subset it.

>### .sort
(by='utt', reverse=False)

The same as [`NumpyMatrix().sort`](#numpy-matrix-sort). Sort it.

>### .map
(func)

The same as [`NumpyMatrix().map`](#numpy-matrix-map). map a functionto all arrays.

>### .cut
(maxFrames)

Cut long utterance to multiple shorter ones. 

**Args:**  
_maxFrames_: a positive int value. Cut a utterance by the thresold value of 1.25 * maxFrames into a maxFrames part and a rest part.If the rest part is still longer than 1.25 * maxFrames, continue to cut it. 

**Return:**  
A new NumpyAliTrans object.
