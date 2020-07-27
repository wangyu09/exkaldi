# exkaldi.decode.wfst, exkaldi.decode.score

This section includes lattice processing and scoring

------------------------
>>## wfst.Lattice
(data=None, symbolTable=None, hmm=None, name="lat")

Lattice holds the decoding lattice data. 
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/decode/wfst.py)  

**Initial Args:**    
_data_: bytes data.  
_symbolTable_: exkaldi ListTable object.   
_hmm_: exkaldi HMM object.  
_name_: a string.  

>### .symbolTable

Get it's inner symbol table.

**Return:**  
a ListTable object.

>### .hmm

Get it's inner HMM model.

**Return:**  
an exkaldi HMM object.

>### .save
(fileName)

Save lattice to file with kaldi format.

**Args:**  
_fileName_: file name or file handle.

**Return:**  
file name or file handle.

>### .get_1best
(symbolTable=None, hmm=None, lmwt=1, acwt=1.0, phoneLevel=False, outFile=None)

Get 1 best result with text format.

**Share Args:** 
_symbolTable_: None or file path or ListTable object or LexiconBank object.  
_hmm_: None or file path or exkaldi HMM object.    
_phoneLevel_: If Ture, return phone results.    

**Parallel Args:**  
_lmwt_: language model weight.  
_acwt_: acoustic model weight.  
_outFile_: output file name.  

**Return:**  
exkaldi Transcription object.

>### .scale
(acwt=1, invAcwt=1, ac2lm=0, lmwt=1, lm2ac=0)

Scale lattice.

**Args:**  
_acwt_: acoustic scale.  
_invAcwt_: inverse acoustic scale.  
_ac2lm_: acoustic to lm scale.  
_lmwt_: language lm scale.  
_lm2ac_: lm scale to acoustic.  

**Return:**  
a new Lattice object.

>### .add_penalty
(penalty=0)

Add penalty to lattice.

**Args:**  
_penalty_: penalty.  

**Return:**  
a new Lattice object.

>### .get_nbest
(n, symbolTable=None, hmm=None, acwt=1, phoneLevel=False, requireAli=False, requireCost=False)

Get N best result with text format.

**Args:**  
_n_: n best results.  
_symbolTable_: file or ListTable object or LexiconBank object.  
_hmm_: file or HMM object.  
_acwt_: acoustic weight.  
_phoneLevel_: If True, return phone results.  
_requireAli_: If True, return alignment simultaneously.  
_requireCost_: If True, return acoustic model and language model cost simultaneously.  

**Return:**  
A list of exkaldi Transcription objects (and their Alignment and Metric objects).

>### .determinize
(acwt=1.0, beam=6)

Determinize the lattice.  

**Args:**  
_acwt_: acoustic scale.  
_beam_: prune beam.   

**Return:**  
a new Lattice object.

>### .am_rescore
(hmm, feat)

Replace the acoustic scores with new HMM-GMM model.

Determinize the lattice.

**Args:**  
_hmm_: exkaldi HMM object or file path.  
_feat_: exkaldi feature object or index table object.  

**Return:**  
a new Lattice object.

>### .\_\_add\_\_
(other)

Sum two lattices to one.

**Args:**  
_other_: another Lattice object.

**Return:**  
a new Lattice object.

--------------------------------
>## score.wer
(ref, hyp, ignore=None, mode='all')

Compute WER (word error rate) between _ref_ and _hyp_. 

**Args:**  
_ref_, _hyp_: exkaldi transcription object or file path.  
_ignore_: ignore specified symbol.  
_mode_: "all" or "present".  

**Return:**  
a namedtuple of score information.

--------------------------------
>## score.edit_distance
(ref, hyp, ignore=None, mode='present')

Compute edit-distance score.

**Args:**  
_ref_, _hyp_: Transcription objects.  
_ignore_: Ignoring specific symbols.  
_mode_: When both are Transcription objects, if mode is 'present', skip the missed utterances.   

**Return:**  
a namedtuple object including score information.