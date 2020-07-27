# exkaldi.decode.wfst

This section includes decoding based on HCLG graph and aligning.

------------------------
>## wfst.nn_decode
(prob, hmm, HCLGFile, symbolTable, beam=10, latBeam=8, acwt=1, minActive=200, maxActive=7000, maxMem=50000000,config=None,maxThreads=1,outFile=None)

Decode by generating lattice from acoustic probability output by NN model.Some usual options can be assigned directly. If you want use more, set _config_=your-configure.
You can use .check_config('nn_decode') function to get configure information you could set. Also run shell command "latgen-faster-mapped" to look their meaning.  
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/decode/wfst.py)  

**Parallel Args:**  
_prob_: An exkaldi probability object. We expect the probability didn't pass any activation function, or it may generate wrong results.  
_hmm_: file path or exkaldi HMM object.  
_HCLGFile_: HCLG graph file.  
_symbolTable_: words.txt file path or exkaldi LexiconBank or ListTable object.  
_beam_.  
_latBeam_.  
_acwt_.  
_minActive_.  
_maxActive_.  
_maxMem_.  
_config_.  
_maxThreads_.  
_outFile_.  

**Return:**  
exkaldi Lattice object.

------------------------
>## wfst.gmm_decode
(feat, hmm, HCLGFile, symbolTable, beam=10, latBeam=8, acwt=1,minActive=200, maxActive=7000, maxMem=50000000, config=None, maxThreads=1, outFile=None)

Decode by generating lattice from acoustic probability output by NN model.Some usual options can be assigned directly. If you want use more, set _config_=your-configure.
You can use .check_config('gmm_decode') function to get configure information you could set. Also run shell command "latgen-faster-mapped" to look their meaning.  
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/decode/wfst.py)  
**Parallel Args:**  
_feat_: An exkaldi feature or index table object.
_hmm_: file path or exkaldi HMM object.  
_HCLGFile_: HCLG graph file.  
_symbolTable_: words.txt file path or exkaldi LexiconBank or ListTable object.  
_beam_.  
_latBeam_.  
_acwt_.  
_minActive_.  
_maxActive_.  
_maxMem_.  
_config_.  
_maxThreads_.  
_outFile_. 

**Return:**
exkaldi Lattice object.

------------------------
>## wfst.compile_align_graph
(hmm, tree, transcription, LFile, outFile, lexicons=None)

Compile graph for training or aligning.  
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/decode/wfst.py)  

**Share Args:**  
_hmm_: file name or exkaldi HMM object.  
_tree_: file name or exkaldi decision tree object.   
_lexicons_: exkaldi lexicon bank object.   
_LFile_: file name.    

**Parallel Args:**    
_transcription_: file path or exkaldi trancription object.  
_outFile_: output file name.  

**Return:**  
output file name.

------------------------
>## wfst.nn_align
(hmm, prob, alignGraphFile=None, tree=None, transcription=None, Lfile=None, transitionScale=1.0, acousticScale=0.1, selfloopScale=0.1, beam=10, retry_beam=40, lexicons=None, name="ali", outFile=None)

Align the neural network acoustic output probability.  
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/decode/wfst.py)  

**Share Args:**  
_hmm_: file name or exkaldi HMM object.  
_tree_: file name or exkaldi decision tree object.  
_Lfile_: file name.  
_lexicons_: exkaldi LexiconBank object.  

**Parallel Args:**  
_prob_: exkaldi probability object or index table object.   
_alignGraphFile_: file name.  
_transcription_: file name or exkaldi transcription object.  
_transitionScale_.  
_acousticScale_.  
_selfloopScale_.  
_beam_.  
_retryBeam_.  
_name_: string.  
_outFile_: file name.  

**Return:**  
exkaldi alignment object or index table object.

------------------------
>## wfst.gmm_align
(hmm, feat, alignGraphFile=None, tree=None, transcription=None, Lfile=None, transitionScale=1.0, acousticScale=0.1, selfloopScale=0.1, beam=10, retry_beam=40, boost_silence=1.0, careful=False, name="ali", lexicons=None, outFile=None)

Align the feature.
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/decode/wfst.py)  

**Share Args:**    
_hmm_: file name or exkaldi HMM object.  
_tree_: file name or exkaldi decision tree object.  
_Lfile_: file name.  
_lexicons_: exkaldi LexiconBank object.  
_boostSilence_.  
_careful_.  

**Parallel Args:**  
_feat_: exkaldi feature object or index table object.  
_alignGraphFile_: file name.  
_transcription_: file name or exkaldi transcription object.  
_transitionScale_.  
_acousticScale_.  
_selfloopScale_.  
_beam_.  
_retryBeam_.  
_name_: string.  
_outFile_: file name.  

**Return:**  
exkaldi alignment object or index table object.