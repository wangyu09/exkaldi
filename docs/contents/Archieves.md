# exkaldi

This section includes various classes to discribe archieves.

--------------------------------------------
>>## exkaldi.ListTable
(data={},name="table") 

This is a subclass of Python dict.
You can use it to hold kaldi text format tables, such as scp-files, utt2spk and so on. 

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
_name_: a string. If None, clear this list table.

>### .sort
(reverse=False)

Sort by key.

**Args:**
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
Random shuffle the list table.

**Return:**  
A new ListTable object.

>### .subset
(nHead=0, nTail=0, nRandom=0, chunks=1, uttIDs=None)

Subset. Only one mode will do when it is not the default value. 
The priority order is: nHead > nTail > nRandom > chunks > uttIDs.

**Args:**  
_nHead_: If it > 0, extract N head utterances.  
_nTail_: If it > 0, extract N tail utterances.  
_nRandom_: If it > 0, randomly sample N utterances.  
_chunks_: If it > 1, split data into N chunks.  
_uttIDs_: If it is not None, pick out these utterances whose ID in uttIDs.  
**Return:**  
a new ListTable object or a list of new ListTable objects.

>### .\_\_add\_\_
(other)

Integrate two ListTable objects. If key existed in both two objects, the former will be retained.

**Args:**   
_other_: another ListTable object.   
**Return:**    
A new ListTable object.  

>### .reverse

Exchange the position of key and value.
Key and value must be one-one matching, or Error will be raised.

**Return:**    
A new ListTable object. 

--------------------------------------
>>## exkaldi.Transcription
(data={},name="transcription") 

Inherited from `ListTable`.
This is used to hold transcription text, such as decoding n-best. 

>### .convert
(self, symbolTable, unkSymbol=None)

Convert transcription between two types of symbol, typically text format and int format.

**Args:**  
_symbolTable_: exkaldi ListTable object.  
_unkSymbol_: symbol of oov. If symbol is out of table, use this to replace.  
**Return:**  
A new Transcription object.

>### .sentence_length 

Count the length of each sentence ( It will count the numbers of inner space ).
 
**Return:**    
A Metric object.

>### .save
(fileName=None, chunks=1, concatFunc=None, discardUttID=False)

Save to file.

**Args:**   
_fileName_: file name, opened file handle or None.  
_chunks_: If > 1, split data averagely and save them. This does only when _fileName_ is a filename.  
_discardUttID_: If True, discard the ifno of utterance IDs.  
**Return:**  
If _fileName_ is None, return a string including all contents of this ListTable. Or return file name or file handle.

>### .count_word
		
Count the number of word.

**Return:**  
a Metric object.

------------------------------
>>## exkaldi.Metric
(data={},name="metric") 

Inherited from `ListTable`.
This is used to hold the Metrics, such as AM or LM scores. 
The data format in Metric is: { utterance ID : int or float score,  }

>### .sort
(by="utt", reverse=False)

Sort by utterance ID or score.

**Args:**
_by_: "utt" or "score".  
_reverse_: If reverse, sort in descending order.  
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

Get the maximum score.

**Return:**   
A float value.  

>### .argmax

Get the utterance ID of maximum score.

**Return:**   
A string.  

>### .min

Get the minimum score.

**Return:**   
A float value.  

>### .argmin

Get the utterance ID of minimum score.

**Return:**   
A string.  

>### .map
(func)

Map a function to all scores.

**Args:**   
_func_:A callable object.   
**Return:**   
A new Metric object.

-----------------------------------
>## exkaldi.ArkIndexTable
(data={},name="indexTable") 

Inherited from `ListTable`.
This is used to hold the utterance index informat of Kaldi archive table (binary format). It just like the script-table file but is more useful. Its format like this:
{ "utt0": namedtuple(frames=100, startIndex=1000, dataSize=10000, filePath="./feat.ark") }

>### .utts

get all utterance IDs.
  
**Return:**   
A list of strings.

>### .sort
(by="utt", reverse=False)

Sort by utterance ID or frames or startIndex.

**Args:**     
_by_: "utt", "frame" or "startIndex".  
_reverse_: If True, sort in descending order.  
**Return:**   
A new ArkIndexTable object.

>### .save
(fileName=None, chunks=1)

Save this index informat to text file with kaidi script-file table format.
Note that the frames informat will be discarded.

**Args:**  
_fileName_: file name or file handle.  
_chunks_: an positive int value. If > 1, split it into N chunks and save them. This option only work when _fileName_ if a file name.  
**Return:**  
file name or None or the contents of ListTable.

>### .fetch
(arkType=None, uttIDs=None)

Fetch data from file acording to index information.
the information of file path in the indexTable is necessary. 

**Args:**  
_uttID_: utterance ID or a list of utterance IDs.  
_arkType_: None or "feat" or "cmvn" or "prob" or "ali" or "fmllrMat" or "mat" or "vec".   
**Return:**   
an exkaldi bytes achieve object depending on the _arkType_. 

------------------------------------------------
>## exkaldi.BytesMatrix
(data=b"", name="mat", indexTable=None)

Hold the feature with kaldi binary format.

**Initial Args:**   
_data_: bytes object.  
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
_name_: a string. If None, clear this list table.

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

Change its data type.

**Args:**   
_dtype_: a string, "float", "float32" or "float64".  
**Return:**   
a new feature object.

>### .dim

Change its dimensions.

**Return:**   
a int value.

>### .utts

Get its utterance IDs.

**Return:**   
a list of strings.

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
_fileName_: file name or file handle. If it7s a file name, suffix ".ark" will be add to the name defaultly.  
_chunks_: If larger than 1, data will be saved to mutiple files averagely. This would be invalid when _fileName_ is a file handle.  
_returnIndexTable_: If True, return the index table containing the information of file path.  

**Return:**  
the path of saved files.

>### .to_numpy
()

Transform bytes data to numpy data.

**Return:**  
a NumpyFeature object sorted by utterance ID.

>### .\_\_add\_\_
(other)

The plus operation between two objects.

**Args:**  
_other_: a BytesFeature or NumpyFeature object or ArkIndexTable object.  
**Return:**  
a new BytesMatrix object.  

>### .subset
(nHead=0, nTail=0, nRandom=0, chunks=1, uttIDs=None)

Subset data.
The priority of mode is nHead > nTail > nRandom > chunks > uttIDs.
If you chose mutiple modes, only the prior one will work.

**Args:**  
_nHead_: get N head utterances.  
_nTail_: get N tail utterances.  
_nRandom_: sample N utterances randomly.  
_chunks_: split data into N chunks averagely.  
_uttIDs_: pick out these utterances whose ID in uttIDs.  
**Return:**  
a new BytesFeature object or a list of new BytesFeature objects.

>### .\_\_call\_\_
(utt)

Pick out the specified utterance.

**Args:**  
_utt_:a string.  
**Return:**  
If existed, return a new BytesFeature object. Or return None.

>### .sort
(by="utt", reverse=False)

Sort utterances by frames length or uttID

**Args:**    
_by_: "frame" or "utt".  
_reverse_: If reverse, sort in descending order.  
**Return:**  
A new BytesFeature object.  

---------------------------------------------
>>## exkaldi.BytesFeature
(data=b"", name="feat", indexTable=None)

Inherited from `BytesMatrix`.
Hold the feature with bytes format.

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

>### .add_delta
(order=2)

Add N orders delta informat to feature.

**Args:**  
_order_: A positive int value.  
**Return:**  
A new BytesFeature object whose dimendion became original-dim * (1 + order). 

>### .paste
(others)

Paste feature in feature dimension level.

**Args:**  
_others_: a feature object or list of feature objects.  
_ordered_: If False, sort all objects.    
**Return:**  
a new feature object.

-----------------------------
>>## exkaldi.BytesCMVNStatistics
(data=b"", name="cmvn", indexTable=None)

Inherited from `BytesMatrix`.
Hold the CMVN statistics with bytes format.

>### .sort
(reverse=False)

**Args:**  
_reverse_: a bool value.

**Return:**  
a new BytesCMVNStatistics object.

-----------------------------
>>## exkaldi.BytesProbability
(data=b"", name="prob", indexTable=None)

Inherited from `BytesMatrix`.
Hold the probability with bytes format.

-----------------------------
>>## exkaldi.BytesFmllrMatrix
(data=b"", name="fmllrMat", indexTable=None)

Inherited from `BytesMatrix`.
Hold the fmllr transform matrix with bytes format.

-----------------------------
>>## exkaldi.BytesVector
(data=b"", name="vec", indexTable=None)

Hold the vector data with kaldi binary format.
This class has almost the same attributes and methods with `BytesMatrix`.

-----------------------------
>>## exkaldi.BytesAlignmentTrans
(data=b"", name="transitionID", indexTable=None)

Inherited from `BytesVector`.
Hold the transition ID alignment with kaldi binary format.

>### .to_numpy
(aliType="transitionID", hmm=None)

Transform alignment to numpy format.

**Args:**  
_aliType_: If it is "transitionID", transform to transition IDs. If it is "phoneID", transform to phone IDs. If it is "pdfID", transform to pdf IDs.
_hmm_: None, or hmm file or exkaldi HMM object.

**Return:**
a NumpyAlignmentTrans or NumpyAlignmentPhone or NumpyAlignmentPdf object.

-------------------------------
>>## exkaldi.NumpyMatrix
(data={}, name=None)

Hold the matrix data with NumPy format.
It has almost the same attributes and methods with `BytesMatrix`.

**Initial Args:**
_data_: a BytesMatrix or ArkIndexTable object or NumpyMatrix or dict object (or their subclasses)
_name_: a string.

>### .map
(func)

Map all arrays to a function and get the result.

**Args:**  
_func_: callable function object.

**Return:**  
A new NumpyMatrix object.

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
_chunks_: If larger than 1, data will be saved to mutiple files averagely.	  

**Return:**
the path of saved files.

-------------------------------
>>## exkaldi.NumpyFeature
(data={}, name=None)

Inherited from `NumpyMatrix`.
Hold the feature data with NumPy format.

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
            alpha * (x-mean)/(stds + epsilon) + belta, 
or do: 
            alpha * (x-mean) + belta.

**Args:**   
_std_: True of False.  
_alpha_,_beta_: a float value.  
_epsilon_: a extremely small float value.  
_axis_: the dimension to normalize.  

**Return:**  
A new NumpyFeature object.  

>### .cut
(maxFrames)

Cut long utterance to mutiple shorter ones. 

**Args:**  
_maxFrames_: a positive int value. Cut a utterance by the thresold value of 1.25 * maxFrames into a maxFrames part and a rest part. If the rest part is still longer than 1.25 * maxFrames, continue to cut it. 

**Return:**  
A new NumpyFeature object.  

>### .paste
(others)

Concatenate feature arrays of the same uttID from mutiple objects in feature dimendion.

**Args:**  
_others_: an object or a list of objects of NumpyFeature or BytesFeature.  

**Return:**  
a new NumpyFeature objects.

----------------------------------------
>>## exkaldi.NumpyCMVNStatistics

Inherited from `NumpyMatrix`.
Hold the CMVN statistics with NumPy format.

----------------------------------------
>>## exkaldi.NumpyProbability

Inherited from `NumpyMatrix`.
Hold the probability with NumPy format.

----------------------------------------
>>## exkaldi.NumpyFmllrMatrix

Inherited from `NumpyMatrix`.
Hold the fmllr transform matrix with NumPy format.

----------------------------------------
>>## exkaldi.NumpyVector

Inherited from `NumpyMatrix`.
Hold the vector data with NumPy format.

----------------------------------------
>>## exkaldi.NumpyAlignmentTrans

Inherited from `NumpyVector`.
Hold the transition ID alignment with NumPy format.

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

Cut long utterance to mutiple shorter ones. 

**Args:**  
_maxFrames_: a positive int value. Cut a utterance by the thresold value of 1.25 * maxFrames into a maxFrames part and a rest part.If the rest part is still longer than 1.25 * maxFrames, continue to cut it. 

**Return:**  
A new NumpyAlignmentTrans object.

----------------------------------------
>>## exkaldi.NumpyAlignment

Inherited from `NumpyVector`.
Hold the alignment with NumPy format.

>### .cut
(maxFrames)

Cut long utterance to mutiple shorter ones. 

**Args:**  
_maxFrames_: a positive int value. Cut a utterance by the thresold value of 1.25 * maxFrames into a maxFrames part and a rest part.If the rest part is still longer than 1.25 * maxFrames, continue to cut it. 

**Return:**  
A new NumpyAlignment object.

----------------------------------------
>>## exkaldi.NumpyAlignmentPhone

Inherited from `NumpyAlignment`.
Hold the phone ID alignment with NumPy format.  

----------------------------------------
>>## exkaldi.NumpyAlignmentPdf

Inherited from `NumpyAlignment`.
Hold the pdf ID alignment with NumPy format.  
 
