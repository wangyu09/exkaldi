# exkaldi.decode.graph

>>## graph.LexiconBank
(pronFile, silWords={"sil":"sil"}, unkSymbol={"unk":"unk"}, optionalSilPhone="sil", extraQuestions=[],positionDependent=False, shareSilPdf=False, extraDisambigPhoneNumbers=1, extraDisambigWords=[])

This class is designed to hold all lexicons (about 20 depending on tasks) which are going to be used when user want to make decoding graph.

**Initial Args:**  
_pronFile_: should be a file path. We support to generate lexicon bank from 5 kinds of lexicon which are "lexicon", "lexiconp(\_disambig)" and "lexiconp\_silprob(\_disambig)". If it is not "lexicon" and silence words or unknown symbol did not exist, error will be raised.  
_silWords_: should be a list object whose members are silence words.  
_unkSymbol_: should be a string used to map the unknown words. If these words are not already existed in _pronFile_, their proninciation will be same as themself.   
_optionalSilPhone_: should be a string. It will be used as the pronunciation of < eps>.  
_extraQuestions_: extra questions to cluster phones when train decision tree.   
_positionDependent_: If True, generate position-dependent lexicons.  
_shareSilPdf_: If True, share the gaussion funtion of silence phones.  
_extraDisambigPhoneNumbers_: extra numbers of disambiguation phone.  
_extraDisambigPhoneNumbers_: extra disambiguation words.  

>### .get_parameter
(name=None)

Get the arguments.

**Return:**  
a python list object.

>### .view

Get names of all generated lexicons.

**Return:**  
a list.

>### .\_\_call\_\_
(name, returnInt=False):

Call a specified lexicon.

**Args:**  
_name_: the lexicons name.  
_returnInt_: a bool value. If True, return lexicon with int format.  

**Return:**  
If name is "words" or "phones", return a ListTable object. Or return dict or string or list object.

>### .dump_dict
(name, fileName=None, dumpInt=False)

Dump a lexicon to file with Kaldi format.

**Args:**
_name_: lexicons name.
_fileName_: output file name or file handle. If None, return a string.  
_dumpInt_: bool value. If True, dump int format.  

**Return:**  
file name, file handle or string.

>### .dump_all_dicts
(outDir="./", requireInt=False)

Dump all lexicons with their default lexicon name.

**Args:**  
_outDir_: output directory path.  
_requireInt_: bool value. If True, dump int format as the same time.

>### .save
(fileName)

Save this LexiconBank object to binary file with suffix ".lex".

**Args:**  
_fileName_: file name.  

**Return:**  
saved file name with a suffix ".lex".

>### .reset_phones
(target)

Reset phone-int table with user's own lexicon. Expected the number of phones is more than or same as default "phones" lexicon.

**Args:**  
_target_: a file or Python dict or exkadli ListTable object. 

>### .reset_words
(target)

Reset word-int table with user's own lexicon. Expected the number of words is more than or same as default "words" lexicon.

**Args:**  
_target_: a file or Python dict or exkadli ListTable object. 

>### .add_extra_question
(question)

Add one piece of extra question to extraQuestions lexicon.

**Args:**  
_question_: a list or tuple of phones.  

>### .update_prob
(targetFile)

Update relative probability of all of lexicons including "lexiconp", "lexiconp_silprob", "lexiconp_disambig", "lexiconp_silprob_disambig", "silprob".

**Args:**  
_target_: a file path. 

------------------------------------------------
>## graph.lexicon_bank
(pronFile, silWords="sil", unkSymbol="unk", optionalSilPhone="sil", extraQuestions=[],positionDependent=False, shareSilPdf=False, extraDisambigPhoneNumbers=1, extraDisambigWords=[]):)

This function will initialize a LexiconBank oject.

**Return:**  
a LexiconBank object.

-----------------------------------------------
>## graph.make_L
(lexicons, outFile, useSilprobLexicon=False, useSilprob=0.5, useDisambigLexicon=False)

Generate L.fst(or L_disambig.fst) file.

**Args:**  
_lexicons_: An exkaldi LexiconBank object.  
_outFile_: Output fst file path such as "L.fst".  
_useSilprobLexicon_: If True, use silence probability lexicon.  
_useSilprob_: If useSilprobLexicon is False, use constant silence probability.  
_useDisambigLexicon_: If true, use lexicon with disambig symbol.  

**Return:**  
the path of generated fst file.

-----------------------------------------------
>## graph.make_G
(lexicons, arpaFile, outFile, order=3)

Transform ARPA format language model to FST format. 

**Args:**   
_lexicon_: A LexiconBank object.  
_arpaFile_: An ARPA LM file path.  
_outFile_: A fst file name.  
_order_: the maximum order to use when make G fst.  

**Return:**  
thepath of generated fst file.

-----------------------------------------------
>## graph.fst_is_stochastic
(fstFile)

Check whether or not fst is stochastic.

**Args:**  
_fstFile_: fst file path.

**Return:**  
True or False.

-----------------------------------------------
>## graph.compose_LG
(Lfile, Gfile, outFile="LG.fst")

Compose L and G to LG.

**Args:**  
_Lfile_: L.fst file path.  
_Gfile_: G.fst file path.  
_outFile_: output LG.fst file.  

**Return:**  
the file path.

-----------------------------------------------
>## graph.compose_CLG
(lexicons, tree, LGfile, outFile="CLG.fst")

Compose tree and LG to CLG file.

**Args:**  
_lexicons_: LexiconBank object.  
_tree_: file path or DecisionTree object.  
_LGfile_: LG.fst file.  
_outFile_: output CLG.fst file.  

**Return:**  
CLG file path and ilabel file path.

-----------------------------------------------
>## graph.compose_HCLG
(hmm, tree, CLGfile, iLabelFile, outFile="HCLG.fst", transScale=1.0, loopScale=0.1, removeOOVFile=None)

Compose HCLG file.

**Args:**  
_hmm_: HMM object or file path.  
_tree_: DecisionTree object or file path.  
_CLGfile_: CLG.fst file path.  
_iLabelFile_: ilabel file path.  
_outFile_: output HCLG.fst file path.  
_transScale_: transform scale.  
_loopScale_: self loop scale.  

**Return:**  
the path of HCLG file.

-----------------------------------------------
>## graph.make_graph
(lexicons, hmm, tree, tempDir, useSilprobLexicon=False, useSilprob=0.5, useDisambigLexicon=False, useLfile=None, arpaFile=None, order=3, useGfile=None, outFile="HCLG.fst", transScale=1.0, loopScale=0.1, removeOOVFile=None)

This is a high-level API to make the HCLG decode graph.

**Args:**  
_lexicons_: exkaldi lexicon bank object.    
_arpaFile_: arpa file path.   
_hmm_: file path or exkaldi HMM object.   
_tree_: file path or exkaldi DecisionTree object.   
_tempDir_: a directory to storage intermidiate files.    
_LFile_: If it's not None use this Lexicon fst directly.   

**Return:**  
the path of HCLG file.

