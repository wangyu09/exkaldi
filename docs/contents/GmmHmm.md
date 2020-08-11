# exkaldi.hmm

This sction is used to train GMM-HMM.

>>## hmm.DecisionTree
(data=b"", contextWidth=3, centralPosition=1, lexicons=None, name="tree")

Decision tree.  
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/hmm/hmm.py)

**Initial Args:**  
_data_: data.  
_contextWidth_: a int value of context width.  
_centralPosition_: a int value of central position.  
_lexicons_: an exkaldi LexiconBank object.  
_name_: a string.  

>### .lex

Get the lexicons carried by this object.  

**Return:**  
An exkaldi LexiconBank object.

>### .contextWidth

Get the context width.  

**Return:**  
An int value.

>### .centralPosition

Get the central position.  

**Return:**  
An int value.

>### .accumulate_stats
(feat, hmm, ali, outFile, lexicons=None)

Accumulate tree statistics in order to compile questions.  

**Share Args:**  
_hmm_: exkaldi HMM object or file name.  
_lexicons_: None. If no any lexicons provided in DecisionTree, this is expected.  

**Parallel Args:**  
_feat_: exkaldi feature or index table object.  
_ali_: exkaldi alignment or index table object.  
_outFile_: file name. If use parallel processes, _outFile_ is necessary.  

**Return:**  
output file paths.

>### .compile_questions
(treeStatsFile, topoFile, outFile, lexicons=None)

Compile questions.

**Share Args:**  
_treeStatsFile_: file path.  
_topoFile_: topo file path.  
_outFile_: file name.  
_lexicons_: None. If no any lexicons provided in DecisionTree, this is necessary.  

**Parallel Args:**  
Null.

**Return:**  
output file path.

>### .build
(treeStatsFile, questionsFile, topoFile, numLeaves, clusterThresh=-1, lexicons=None)

Build tree.

**Share Args:**  
_treeStatsFile_: file path.  
_questionsFile_: file path.  
_numLeaves_: target numbers of leaves.  
_topoFile_: topo file path.    
_lexicons_: None. If no any lexicons provided in DecisionTree, this is expected. In this step, we will use "roots" lexicon.

**Parallel Args:**  
Null.

**Return:**  
the path of out file.

>### .train
(feat, hmm, ali, topoFile, numLeaves, tempDir, clusterThresh=-1, lexicons=None)

This is a hign-level API to build a decision tree.

**Share Args:**  
_hmm_: file path or exkaldi HMM object.  
_topoFile_: topo file path.  
_numLeaves_: target numbers of leaves.  
_tempDir_: a temp directory to storage some intermidiate files.  
_lexicons_: None. If no any lexicons provided in DecisionTree, this is expected.  

**Parallel Args:**  
_feat_: exkaldi feature object.  
_ali_: file path or exkaldi transition-ID Alignment object.  

>### .save  
(fileName="tree")

Save tree to file.

**Args:**   
_fileName_: a string or file handle.  

**Return:**   
file name or file handle.  

>### .load
(target)

Reload a tree from file.The original data will be discarded.

**Args:**  
_target_: file name.

>### .info

Get the tree information.

**Return:**  
a named tuple.

--------------------------------
>>## hmm.BaseHMM
(data=b"",name="hmm",lexicons=None)

A base GMM-HMM class.   
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/hmm/hmm.py)

**Initial Args:**  
_data_: data.  
_lexicons_: an exkaldi LexiconBank object.  
_name_: a string.  

>### .lex

Get the lexicons carried by this object.  

**Return:**  
An exkaldi LexiconBank object.

>### .compile_train_graph
(tree, transcription, LFile, outFile, lexicons=None)

Compile training graph.

**Share Args:**   
_tree_: file name or exkaldi DecisionTree object.  
_LFile_: file path.  
_lexicons_: None. If no any lexicons provided in DecisionTree, this is expected.  

**Args:**  
_transcription_: file path or exkaldi Transcription object with int format. Note that: int fotmat, not text format.  
_outFile_: graph output file path.  

**Return:**  
output file paths.

>### .update
(statsFile, numgauss, power=0.25, minGaussianOccupancy=10)

Update the parameters of HMM model.

**Share Args:**  
_statsFile_: file name.  
_numgauss_: int value.  
_power_.  
_minGaussianOccupancy_:an int value.  

**Parallel Args:**  
Null.

>### .align
(feat, trainGraphFile, transitionScale=1.0, acousticScale=0.1, selfloopScale=0.1, beam=10, retryBeam=40, boostSilence=1.0, careful=False, name="ali", lexicons=None, outFile=None)

Align acoustic feature with Kaldi vertibi algorithm.

**Share Args:**  
_lexicons_: None. If no any lexicons provided in DecisionTree, this is expected. In this step, we will use "context_indep" lexicon.

**Parallel Args:**  
_feat_: exakldi feature or index table object.  
_trainGraphFile_: file path.  
_transitionScale_: a float value.  
_acousticScale_: a float value.  
_selfloopScale_: a float value.  
_beam_: int value.  
_retryBeam_.  
_boostSilence_.  
_careful_.  
_name_. If return an exkaldi alignment object or index table object, you can name it.  
_outFile_. output filename. If use parallel processes, output file name is necessary.  

**Return:**  
exkaldi alignment object or index table object.

>### .accumulate_stats
(feat, ali, outFile)

Accumulate GMM statistics in order to update GMM parameters.

**Share Args:**  
Null.

**Parallel Args:**  
_feat_: exkaldi feature object.  
_ali_: exkaldi transitionID alignment object or file path.  
_outFile_: file name.  

**Return:**  
output file paths.

>### .align_equally
(feat, trainGraphFile, name="equal_ali", outFile=None)

Align feature averagely.

**Share Args:**  
Null.

**Parallel Args:**  
_feat_: exkaldi feature object or index table object.  
_trainGraphFile_: file path.  
_name_: a string.  
_outFile_: output file name.  

**Return:**  
exakldi alignment or index table object.   

>### .save  
(fileName)  

Save model to file.  

**Args:**  
_fileName_: a string.  

>### .load
(target)

Reload a HMM-GMM model from file. The original data will be discarded.

**Args:**  
_target_: file path.

>### .info

Get the information of model.

**Return:**  
a named tuple.

>>## hmm.MonophoneHMM
(lexicons=None, name="mono")

This is a subclass of BaseHMM ans used to hold monophone object. Almost attributes ans methods are fimiliar with BaseHMM.  
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/hmm/hmm.py)

**Initial Args:**  
_lexicons_: a LexiconBank object.
_name_: s string.  

>### .initialize
(feat, topoFile, lexicons=None)

Initialize this Monophone HMM-GMM model. This is necessary before training this model.

**Args:**
_feat_: exkaldi feature or index table object.
_topoFile_: file path.
_lexicons_: None. If no any lexicons provided in DecisionTree, this is expected. In this step, we will use "context_indep" lexicon.

>### .tree

Get the temp tree in monophone model.

**Return:**  
a DecisionTree object.

>### .train
(feat, transcription, LFile, tempDir,numIters=40, maxIterInc=30, totgauss=1000, realignIter=None,transitionScale=1.0, acousticScale=0.1, selfloopScale=0.1,initialBeam=6, beam=10, retryBeam=40,boostSilence=1.0, careful=False, power=0.25, minGaussianOccupancy=10, lexicons=None)

This is a high-level API to train the HMM-GMM model.

**Share Args**  
_LFile_: L.fst file path.  
_tempDir_: A directory to save intermidiate files.  
_numIters_: Int value, the max iteration times.  
_maxIterInc_: Int value, increase numbers of gaussian functions untill this iteration.  
_totgauss_: Int value, the rough target numbers of gaussian functions.   
_realignIter_: None or list or tuple, the iter to realign.  
_transitionScale_: a float value.  
_acousticScale_: a float value.  
_selfloopScale_: a float value.  
_initialBeam_: an int value.  
_beam>_: an int value.  
_retryBeam_: an int value.  
_boostSilence_: a float value.  
_careful_: a bool value.  
_power_: an float value.  
_minGaussianOccupancy_: an int value.  
_lexicons_: an exkaldi LexiconBank object.  

**Parallel Args:**  
_feat_: exkaldi feature or index table object.  
_transcription_: exkaldi transcription object or file name (text format).  

**Return:**  
an index table object of final alignment.

>>## hmm.TriphoneHMM
(lexicons=None, name="tri")

This is a subclass of BaseHMM ans used to hold context-phone object. Almost attributes ans methods are fimiliar with BaseHMM.  
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/hmm/hmm.py)

**Initial Args:**  
_lexicons_: a LexiconBank object.
_name_: s string.  

>### .initialize
(tree, topoFile, feat=None, treeStatsFile=None)

Initialize a Triphone Model.

**Args:**   
_tree_: file path or exkaldi DecisionTree object.  
_topoFile_: file path.  
_numgauss_: int value.  
_feat_: exkaldi feature object.  
_treeStatsFile_: tree statistics file.  

>### .tree

Get the temp tree in monophone model.

**Return:**  
a DecisionTree object.

>### .train
(feat,transcription,LFile,tree,tempDir,initialAli=None,ldaMatFile=None,fmllrTransMat=None,spk2utt=None,utt2spk=None,numIters=40,maxIterInc=30,totgauss=1000,fmllrSilWt=0.0,realignIter=None, mlltIter=None, fmllrIter=None,transitionScale=1.0, acousticScale=0.1, selfloopScale=0.1,beam=10,retryBeam=40,boostSilence=1.0,careful=False,power=0.25,minGaussianOccupancy=10,lexicons=None)

This is a high-level API to train the HMM-GMM model.

**Share Args:**  
_LFile_: Lexicon fst file path.  
_tree_: file path or exkaldi DecisionTree object.  
_tempDir_: A directory to save intermidiate files.  
_ldaMatFile_: file path. If provided, do lda_mllt training.  
_spk2utt_: file path or exkaldi ListTable object.  
_utt2spk_: file path or exkaldi ListTable object.  
_numIters_: Int value, the max iteration times.  
_maxIterInc_: Int value, increase numbers of gaussian functions untill this iteration.  
_totgauss_: Int value, the rough target numbers of gaussian functions.   
_fmllrSilWt_: a float value.  
_realignIter_: None or list or tuple, the iter to realign.  
_mlltIter_: None or list or tuple, the iter to estimate new MLLT matrix.  
_transitionScale_: a float value.  
_acousticScale_: a float value.  
_selfloopScale_: a float value.  
_initialBeam_: an int value.  
_beam_: an int value.  
_retryBeam_: an int value.  
_boostSilence_: a float value.  
_careful_: a bool value.  
_power_: a float value.  
_minGaussianOccupancy_: an int value.   
_lexicons_. exkaldi LexiconBank object.  

**Parallel Args:**  
_feat_: exkaldi feature or index table object.  
_transcription_: exkaldi transcription object or file name (text format).  
_initialAli_: exakldi alignment or index table object.  
_fmllrTransMat_. exkaldi fmllr matrix or index table object.  

**Return:**  
an index table of final alignment.

----------------------------------
>## hmm.sum_gmm_stats
(statsFiles, outFile)

Sum GMM statistics files.

**Args:**  
_statsFiles_: a string, list or tuple of multiple file paths.  
_outFile_: output file path.   
**Return:**  
the path of accumulated file.

--------------------------------------
>## hmm.sum_tree_stats
(statsFiles, outFile)

Sum tree statistics files.

**Args:**  
_statsFiles_: a string, list or tuple of multiple file paths.  
_outFile_: output file path.  
**Return:**  
the path of accumulated file.

------------------------------------------
>## hmm.make_toponology
(lexicons, outFile, numNonsilStates=3, numSilStates=5)

Make GMM-HMM toponology file.

**Args:**  
_lexicons_: an LexiconBank object.  
_outFile_: output file path.   
_numNonsilStates_: the number of non-silence states.  
_numSilStates_: the number of silence states.  
**Return:**  
the path of generated file.

-----------------------------------------
>## hmm.convert_alignment
(ali, originHmm, targetHmm, tree, outFile=None)

Convert alignment.

**Args:**  
_ali_: file path or exkaldi transition-ID alignment object.  
_originHmm_: file path or exkaldi HMM object.  
_targetHmm_: file path or exkaldi HMM object.  
_tree_: file path or exkaldi DecisionTree object.  
_outFile_: file name.  

**Return:**
exkaldi alignment or index table object.

-----------------------------------------
>## hmm.transcription_to_int
(transcription, symbolTable, unkSymbol)

Transform text format transcrption to int format file.

**Args:**  
_transcription_: file path or Transcription object.  
_symbolTable_: word2id file path or exkaldi ListTable object.  
_unkSymbol_: a string to map OOV.  

**Return:**
exkaldi Transcription object.

------------------------------------------
>## hmm.transcription_from_int
(transcription, symbolTable)

Transform int format transcrption to text format file.

**Args:**  
_transcription_: file path or Transcription object.  
_symbolTable_: word2id file path or exkaldi ListTable object.  

**Return:**
exkaldi Transcription object.

------------------------------------------
>## hmm.accumulate_LDA_stats
(ali, lexicons, hmm, feat, outFile, silenceWeight=0.0, randPrune=4)

Acumulate LDA statistics to estimate LDA tansform matrix.

**Share Args:**   
_lexicons_: exkaldi lexicons bank object.  
_hmm_: file name or exkaldi HMM object.  
_silenceWeight_.  
_randPrune_.  

**Parallel Args:**  
_ali_: exkaldi alignment or index table object.  
_feat_: exkaldi feature or index object.  
_outFile_: output file name.  

------------------------------------------
>## hmm.accumulate_MLLT_stats
(ali, lexicons, hmm, feat, outFile, silenceWeight=0.0, randPrune=4)

Acumulate MLLT statistics to estimate LDA+MLLT tansform matrix.

**Share Args:**   
_lexicons_: exkaldi lexicons bank object.  
_hmm_: file name or exkaldi HMM object.  
_silenceWeight_.  
_randPrune_.  

**Parallel Args:**  
_ali_: exkaldi alignment or index table object.  
_feat_: exkaldi feature or index object.  
_outFile_: output file name.  

------------------------------------------
>## hmm.estimate_LDA_matrix
(statsFiles, targetDim, outFile)

Estimate the LDA transform matrix from LDA statistics.

**Args:**  
_statsFiles_: str or list ot tuple of file paths.   
_targetDim_: int value.    
_outFile_: file name.   

**Return:**  
the file path of output file.

------------------------------------------
>## hmm.estimate_MLLT_matrix
(statsFiles, outFile)

Estimate the MLLT transform matrix from MLLT statistics.

**Args:**  
_statsFiles_: str or list ot tuple of file paths.  
_outFile_: file name.  

**Return:**  
the file path of output file.

--------------------------------------
>## hmm.compose_transform_matrixs
(matA, matB, bIsAffine=False, utt2spk=None, outFile=None)

The dot operator between two matrixes.

**Args:**  
_matA_: matrix file or exkaldi fMLLR transform matrx object.  
_matB_: matrix file or exkaldi fMLLR transform matrx object.  
_bIsAffine_: a bool value.  
_utt2spk_: file name or ListTable object.
_outFile_: None or file name.  

**Return:**  
If _outFile_ is not None, return the path of output file.
Else, return Numpy Matrix object or BytesFmllrMatrix object.

--------------------------------------
>## hmm.estimate_fMLLR_matrix
(matA, matB, bIsAffine=False, utt2spk=None, outFile=None)

Estimate fMLLR transform matrix.

**Share Args:**  
_lexicons_: exkaldi LexiconBank object.  
_aliHmm_: file or exkaldi HMM object.  
_adaHmm_: file or exkaldi HMM object.    
_silenceWeight_: float value.  
_acwt_: float value.  

**Parallel Args:**  
_aliOrLat_: exkaldi Transition alignment object or Lattice object.  
_feat_: exkaldi feature object.  
_spk2utt_: file path or ListTable object.  
_name_: a string.  
_outFile_: output file name.  

**Return:**  
exkaldi fMLLR transform matrix or index table object.  

