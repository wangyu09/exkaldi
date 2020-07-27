# exkaldi.decode.e2e

This section includes some experimental functions for End-to-End decoding.

--------------------
>## e2e.convert_field
(prob, originVocabs, targetVocabs, retainOOV=False)

Tranform the dimensions of probability to target field.

**Args:**  
_prob_: An exkaldi probability object. This probalility should be an output of Neural Network.  
_originVocabs_: list of original field vocabulary.  
_originVocabs_: list of target field vocabulary.  
_retainOOV_: If True, target words which are not in original vocabulary will be retained in minimum probability of each frame.   

**Return:**  
An new exkaldi probability object and a list of new target vocabulary.  

--------------------
>## e2e.ctc_greedy_search
(prob, vocabs, blankID=None)

The best path decoding algorithm.

**Args:**  
_prob_: An exkaldi probability object. This probalility should be an output of Neural Network with CTC loss fucntion.  
_vocabs_: a list of vocabulary.  
_blankID_: specify the ID of blank symbol. If None, use the last dimentionality of _prob_.    

**Return:**  
An exkaldi Transcription object of decoding results.  

--------------------
>## e2e.ctc_prefix_beam_search
(prob, vocabs, blankID=None, beam=5, cutoff=0.999, strick=1.0, lmFile=None, alpha=1.0, beta=0)

Prefix beam search decoding algorithm. Lm score is supported.

**Args:**  
_prob_: An exkaldi postprobability object. This probalility should be an output of Neural Network with CTC loss fucntion. We expect the probability didn't pass any activation function, or it may generate wrong results.  
_vocabs_: a list of vocabulary.  
_blankID_: specify the ID of blank symbol. If None, use the last dimentionality of _prob_.  
_beam_: the beam size.  
_cutoff_: the sum threshold to cut off dimensions whose probability is extremely small.    
_strick_: When the decoding results of two adjacent frames are the same, the probability of latter will be reduced.  
_lmFile_: If not None, add language model score to beam.  
_alpha_: the weight of LM score.  
_beta_: the length normaoliztion weight of LM score.  

**Return:**  
An exkaldi Transcription object of decoding results.  
