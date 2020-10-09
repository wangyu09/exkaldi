# exkaldi, exkaldi.nn

This section includes some class and function to help training NN model with DL frameworks.  

------------------------
>## exkaldi.tuple_data
(archives, frameLevel=False)

Tuple exkadli archives in "utterance" level or "frame" level. Typically, tuple feature and alignment.
This is used to generate dataset for training NN model with DL framework.

**Args:**  
_archives_: exkaldi feature or alignment objects.  
_framelevel_: If True, tuple data in frame level. Or in utterance level.  

**Return:**
a List of tupled data.

------------------------
>## exkaldi.compute_postprob_norm
(ali, probDims)

Compute alignment counts in order to normalize acoustic model posterior probability.

**Args:**
_ali_: exkaldi NumpyAliTrans, NumpyAliPhone or NumpyAliPdf object.  
_probDims_: count size for probability.  

**Return:**
A numpy array of the normalization.

**Examples:**  
```python
hmm = exkaldi.load_hmm("./final.mdl")
ali = exkaldi.load_ali("./ali",aliType="phoneID",hmm=hmm)
probDims = hmm.info.phones

norm = exkaldi.compute_postprob_norm(ali, probDims)
print(norm)
```
------------------------------------------
>>## nn.DataIterator
(indexTable, processFunc, batchSize, chunks='auto', otherArgs=None, shuffle=False, retainData=0.0)

A data iterator for training NN with a large-scale corpus. 

**Initial Args:**  
_indexTable_: exkaldi IndexTable object including file path information.    
_processFunc_: a function to process index table to dataset.  
_batchSize_: mini batch size.  
_chunks_: an int value. how many chunks to split data.    
_otherArgs_: other arguments to send to processFunc.    
_shuffle_: If True, shuffle chunk data.  
_retainData_: a float ratio in 0.0~0.9. Reserve part of data (for evaluate.)  

>### .next
()

Get one batch data.

**Return:**  
a list of batch data.

>### .batchSize

Get one batch size.

**Return:**  
an int value.

>### .chunks

Get the number of chunks.

**Return:**  
an int value.

>### .chunk

Get current chunk ID.

**Return:**  
an int value.

>### .epoch

Get current epoch ID.

**Return:**  
an int value.

>### .isNewEpoch

Query whether or not current batch has stepped into a new epoch.

**Return:**  
True or False.

>### .isNewChunk

Query whether or not current batch has stepped into a new chunk.

**Return:**  
True or False.

>### .epochProgress

Get the current progress of one epoch.

**Return:**  
a float value.

>### .chunkProgress

Get the current progress of one chunk.

**Return:**  
a float value.

>### .get_retained_data
(processFunc=None, batchSize=None, chunks='auto', otherArgs=None, shuffle=False, retainData=0.0)

Get the retained data.

**Args:**  
_processFunc_: a function to process index table to dataset.
_batchSize_: mini batch size.  
_chunks_: an int value. how many chunks to split data.    
_otherArgs_: other arguments to send to processFunc.    
_shuffle_: If True, shuffle chunk data.  
_retainData_: a float ratio in 0.0~0.9. Reserve part of data (for evaluate.)  

**Return:**
A new DataIterator.

------------------------

>## nn.softmax
(data, axis=1)

The softmax function.

**Args:**  
_data_: a Numpy array.  
_axis_: the dimension to softmax.  

**Return:**  
A new array.

------------------------

>## nn.log_softmax
(data, axis=1)

The log-softmax function.

**Args:**  
_data_: a Numpy array.  
_axis_: the dimension to softmax.  

**Return:**  
A new array.

------------------------

>## nn.pad_sequence
(data, shuffle=False, pad=0)

Pad sequences with maximum length of one batch data. 

**Args:**  
_data_: a list of numpy arrays who have various sequence-lengths.
_shuffle_: bool value. If "True", pad each sequence with random start-index and return padded data and length information of (startIndex,endIndex) of each sequence.If it's "False", align the start index of all sequences then pad them rear. This will return length information of only endIndex.

**Return:**
A list.

------------------------

>## nn.accuracy
(ref, hyp, ignore=None, mode='all')

Score one-2-one matching score between two items.

**Args:**  
_ref_, _hyp_: iterable objects like list, tuple or NumPy array. It will be flattened before scoring.  
_ignore_: Ignoring specific symbols.  
_model_: If "all", compute one-one matching score. For example, _ref_ is (1,2,3,4), and _hyp_ is (1,2,2,4), the score will be 0.75. If "present", only the members of _hyp_ which appeared in _ref_ will be scored no matter which position it is. 

**Return:**  
a namedtuple object of score information.

------------------------

>## nn.pure_edit_distance
(ref, hyp, ignore=None)

Compute edit-distance score.

**Args:**  
_ref_, _hyp_: iterable objects like list, tuple or NumPy array. It will be flattened before scoring.  
_ignore_: Ignoring specific symbols.	 

**Return:**  
a namedtuple object including score information.	








