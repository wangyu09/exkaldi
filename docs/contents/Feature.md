# exkaldi

In `exkaldi.core.feature` module, there are some wrapped function to process acoustic feature.

-----------------------------
>## exkaldi.compute_mfcc
(target, rate=16000, frameWidth=25, frameShift=10, melBins=23, featDim=13, windowType='povey', useSuffix=None, config=None, name="mfcc", outFile=None)

Compute MFCC feature. Some usual options can be specified directly. If you want to use more, set _config_=your-configure.
You can use exkaldi.utils.check_config('compute_mfcc') function to get extra configures that you can set.
Also you can run shell command "compute-mfcc-feats" to look more information.

**Share Args:**  
Null.

**Parallel Args:**  
_target_: wave file, scp file, exkaldi ListTable object. If it is wave file, use it's file name as utterance ID.  
_rate_: sample rate.  
_frameWidth_: sample windows width.  
_frameShift_: shift windows width.  
_melbins_: the numbers of mel filter banks.  
_featDim_: the output dinmensionality of MFCC feature.  
_windowType_: sample windows type.  
_useSuffix_: If the suffix of file is not .scp, use this to specify it.  
_config_: use this to assign more extra optional configures.  
_name_: the name of feature.  
_outFile_: output file name.  

**Return:**  
exkaldi feature or index table object.

**Examples:**  
You can compute MFCC feature from script-table file or directly from a wave file. We allow normal grammar.
```python
feat1 = exkaldi.compute_mfcc(target="wav.scp") # script-file
feat2 = exkaldi.compute_mfcc(target="my.wav") # a wav file
feat3 = exkaldi.compute_mfcc(target="*_wav.scp") # script-file with normal grammar.
feat4 = exkaldi.compute_mfcc(target="*_my.wav") # a wav file with normal grammar.
```
Also, we can compute feature from ListTable object.
```python
wavs = exkaldi.load_list_table("wav.scp")
feat5 = exkaldi.compute_mfcc(target=wavs) # ListTable object.
```
If you need to config other arguments.
```python
extraCfg = {"use-energy":"true"}
feat5 = exkaldi.compute_mfcc(target="wav.scp", config=extraCfg) # ListTable object.
```
Compute MFCC feature by parallel process. You only need to split the recource into N chunks.
```python
wavs = exkaldi.load_list_table("wav.scp").subset(chunks=2)
feat6 = exkaldi.compute_mfcc(target=wavs, config=extraCfg, outFile="mfcc.ark") # ListTable object.
```
In this example, we used two mutiple processes because we gave the function two resources.

-----------------------------
>## exkaldi.compute_fbank
(target, rate=16000, frameWidth=25, frameShift=10, melBins=23, windowType='povey', useSuffix=None,config=None, name="fbank", outFile=None)

Compute fbank feature. Some usual options can be assigned directly. If you want use more, set _config_= your-configure.
You can use .check_config('compute_fbank') function to get configure information that you can set.
Also you can run shell command "compute-fbank-feats" to look their meaning.

**Share Args:**  
Null.

**Parallel Args:**  
_target_: wave file, scp file, exkaldi ListTable object. If it is wave file, use it's file name as utterance ID.  
_rate_: sample rate.  
_frameWidth_: sample windows width.  
_frameShift_: shift windows width.  
_melbins_: the numbers of mel filter banks.  
_windowType_: sample windows type.  
_useSuffix_: If the suffix of file is not .scp, use this to specify it.  
_config_: use this to assign more extra optional configures.  
_name_: the name of feature.  
_outFile_: output file name.  

**Return:**  
exkaldi feature or index table object.

-----------------------------
>## exkaldi.compute_plp
(target, rate=16000, frameWidth=25, frameShift=10,melBins=23, featDim=13, windowType='povey', useSuffix=None,config=None, name="plp", outFile=None)

Compute plp feature. Some usual options can be assigned directly. If you want use more, set _config_= your-configure.
You can use .check_config('compute_plp') function to get configure information that you can set.
Also you can run shell command "compute-plp-feats" to look their meaning.

**Share Args:**  
Null.

**Parallel Args:**  
_target_: wave file, scp file, exkaldi ListTable object. If it is wave file, use it's file name as utterance ID.  
_rate_: sample rate.  
_frameWidth_: sample windows width.  
_frameShift_: shift windows width.  
_melbins_: the numbers of mel filter banks.  
_featDim_: the output dinmensionality of PLP feature.  
_windowType_: sample windows type.  
_useSuffix_: If the suffix of file is not .scp, use this to specify it.  
_config_: use this to assign more extra optional configures.  
_name_: the name of feature.  
_outFile_: output file name.  

**Return:**  
exkaldi feature or index table object.

-----------------------------
>## exkaldi.compute_spectrogram
(target, rate=16000, frameWidth=25, frameShift=10,windowType='povey', useSuffix=None, config=None, name="spectrogram", outFile=None)

Compute power spectrogram feature. Some usual options can be assigned directly. If you want use more, set _config_= your-configure. You can use .check_config('compute_spectrogram') function to get configure information that you can set.
Also you can run shell command "compute-spectrogram-feats" to look their meaning.

**Share Args:**  
Null.

**Parallel Args:**  
_target_: wave file, scp file, exkaldi ListTable object. If it is wave file, use it's file name as utterance ID.  
_rate_: sample rate.  
_frameWidth_: sample windows width.  
_frameShift_: shift windows width.  
_windowType_: sample windows type.  
_useSuffix_: If the suffix of file is not .scp, use this to specify it.  
_config_: use this to assign more extra optional configures.  
_name_: the name of feature.  
_outFile_: output file name.  

**Return:**  
exkaldi feature or index table object.

-----------------------------
>## exkaldi.transform_feat
(feat, matrixFile, outFile=None)

Transform feat by a transform matrix. Typically, LDA, MLLt matrixes.

**Share Args:**  
Null.

**Parallel Args:**  
_feat_: exkaldi feature or index table object.  
_matrixFile_: file name.  
_outFile_: file name.  

**Return:**  
exkaldi feature or index table object.

**Examples:**  
Transform feature, typically LDA+MLLT feature.  
```python
feat = exkaldi.transform_feat(feat,"trans.mat")
```
-----------------------------
>## exkaldi.use_fmllr
(feat, fmllrMat, utt2spk, outFile=None)

Transform feat by a transform matrix. Typically, LDA, MLLt matrixes.

**Share Args:**  
Null.

**Parallel Args:**  
_feat_: exkaldi feature or index table object.  
_fmllrMat_: exkaldi fMLLR transform matrix or index table object.  
_utt2spk_: file name or ListTable object.  
_outFile_: output file name.  

**Return:**  
exkaldi feature or index table object.

**Examples:**  
Allpy fmllr feature.   
```python
# compute fmllr transform matrix from alignment, and feature
fmllrTransMat = exkaldi.hmm.estimate_fMLLR_matrix(
                                    aliOrLat=alignment, # alignment.
                                    lexicons=lexicons,  # LexiconBank object.
                                    aliHmm=model, # hmm model 
                                    feat=feature, # feature
                                    spk2utt="./spk2utt",
                                )
# transform feature
newFeat = exkaldi.use_fmllr(feature, fmllrTransMat, utt2spk="./utt2spk")
```
-----------------------------
>## exkaldi.use_cmvn
(feat, cmvn, utt2spk=None, std=False, outFile=None)

Apply CMVN statistics to feature.

**Share Args:**  
Null.

**Parrallel Args:**  
_feat_: exkaldi feature or index table object.  
_cmvn_: exkaldi CMVN statistics or index object.  
_utt2spk_: file path or ListTable object.  
_std_: If true, apply std normalization.  
_outFile_: out file name.  

**Return:** 
feature or index table object. 

-----------------------------
>## exkaldi.compute_cmvn_stats
(feat, spk2utt=None, name="cmvn", outFile=None)

Compute CMVN statistics.

**Share Args:**  
Null.

**Parrallel Args:**  
_feat_: exkaldi feature object or index table object.  
_spk2utt_: spk2utt file or exkaldi ListTable object.  
_name_: a string.  
_outFile_: file name.  

**Return:**
exkaldi CMVN statistics or index table object.

**Examples:**  
```python
# compute CMVN from feature
cmvn = exkaldi.compute_cmvn_stats(feat=feat, spk2utt="./spk2utt")
```
-----------------------------
>## exkaldi.use_cmvn_sliding
(feat, windowsSize=None, std=False):

Allpy sliding CMVN statistics.

**Share Args:**  
_feat_: exkaldi feature object.  
_windowsSize_: windows size, If None, use windows size larger than the frames of feature.  
_std_: a bool value.  

**Parallel Args:**  
Null.

**Return:**  
exkaldi feature object.

-----------------------------
>## exkaldi.add_delta
(feat, order=2, outFile=None)

Add n order delta to feature.

**Share Args:**  
Null.

**Parrallel Args:**  
_feat_: exkaldi feature objects.  
_order_: the orders.  
_outFile_: output file name.  

**Return:**  
exkaldi feature or index table object.  

-----------------------------
>## exkaldi.splice_feature
(feat, left, right=None, outFile=None):

Splice left-right N frames to generate new feature.
The dimentions will become original-dim * (1 + left + right)

**Share Args:**  
Null.

**Parrallel Args:**  
_feat_: feature or index table object.  
_left_: the left N-frames to splice.  
_right_: the right N-frames to splice. If None, right = left.  
_outFile_: output file name.  

**Return:**  
exkaldi feature object or index table object.

**Examples:**  
```python
feat = exkaldi.splice_feature(feat, left=3, right=3)
print(feat.dim)
```
If original feature is 39 dimmensions, the new feature will become 39 * (3+3+1) = 273.

-----------------------------
>## exkaldi.decompress_feat
(feat, name="decompressedFeat")

Decompress a kaldi conpressed feature whose data-type is "CM".
This function is a cover of kaldi-io-for-python tools. 
For more information about it, please access to https://github.com/vesis84/kaldi-io-for-python/blob/master/kaldi_io/kaldi_io.py 

**Args:**  
_feat_: a python bytes object.  

**Return:**  
exkaldi feature object.  

