# exkaldi.lm

-----------------------------
>## lm.train_ngrams_srilm
(lexicons, order, text, outFile, config=None)

Train n-grams language model with SriLM tookit. You can use .check_config("train_ngrams_srilm") function to get configure information that you can set. Also you can run shell command "ngram-count" to look their meaning.

**Args:**   
_lexicons_: words.txt file path or Exkaldi LexiconBank object.  
_order_: the maxinum order of n-grams.  
_text_: text corpus file or exkaldi transcription object.  
_outFile_: ARPA out file name.  
_config_: configures, a Python dict object.  

**Return:**  
output file name.

-----------------------------
>## lm.train_ngrams_kenlm
(lexicons, order, text, outFile, config=None)

Train n-grams language model with KenLM tookit. You can use .check_config("train_ngrams_kenlm") function to get configure information that you can set. Also you can run shell command "lmplz" to look their meaning.

**Args:**   
_lexicons_: words.txt file path or Exkaldi LexiconBank object.  
_order_: the maxinum order of n-grams.  
_text_: text corpus file or exkaldi transcription object.  
_outFile_: ARPA out file name.  
_config_: configures, a Python dict object.  

**Return:**  
output file name.

-----------------------------
>## lm.arpa_to_binary
(arpaFile, outFile)

Transform ARPA language model file to KenLM binary format file.

**Args:**  
_arpaFile_: ARPA file path.  
_outFile_: output binary file path.  
**Return:**  
output file name with suffix ".binary".

-----------------------------
>>## lm.KenNGrams
(filePath, name="ngram")

This is a wrapper of kenlm.Model, and we only support n-grams model with binary format.
If you want to use the ARPA format LM directly. You can use `exkaldi.load_ngrams` function.

**Initial Args:**  
_filePath_: binary LM file path.  
_name_: a string.  

>### .path

Get the source LM file path.

**Return:**  
a string.

>### .order

Get the maxinum value of order.

**Return:**  
an int value.

>### .score_sentence
(sentence, bos=True, eos=True)

Score a sentence.

**Args:**    
_sentence_: a string with out boundary symbols.  
_bos_: If True, add `<s>` to the head.  
_eos_: If True, add `</s>` to the tail.  
**Return:**  
a float log-value.

>### .score
(transcription, bos=True, eos=True)

Score a transcription.

**Args:**    
_transcription_: file path or exkaldi Transcription object.  
_bos_: If True, add `<s>` to the head.  
_eos_: If True, add `</s>` to the tail.  
**Return:**  
an exkaldi Metric object.

>### .full_scores_sentence
(sentence, bos=True, eos=True)

Generate full scores of a sentence(prob, ngram length, oov).

**Args:**  
_sentence_: a string with out boundary symbols.  
_bos_: If True, add `<s>` to the head.  
_eos_: If True, add `</s>` to the tail.  
**Return:**  
a iterator of (prob, ngram length, oov).

>### .perplexity_sentence
(sentence)

Compute perplexity of a sentence.

**Args:**    
_sentence_: a string with out boundary symbols.  
**Return:**  
a float log-value.

>### .perplexity
(transcription)

Compute perplexity of a transcription.

**Args:**    
_transcription_: file path or exkaldi Transcription object.  
**Return:**  
an exkaldi Metric object.
