####################### Version Information ################################
# Make HCLG graph example based on: exkaldi V0.2 and Kaldi 5.5 
# Yu Wang, University of Yamanashi 
# Jan 19, 2020
############################################################################

############################################################################
# This example code shows how to make a simple HCLG graph based on kaldi CSJ corpus and exkaldi.graph module.
# But notice that:
# In current version, we only support generate new n-grams language while training a new HMM model is still unsupported.
# So we will use two already existed file, the HMM file and tree file, which are generated before.
# In this program, we will generate a new 4-grams LM (the kaldi baseline is 3-grams model) and make a HCLG graph up to tri1 of CSJ recipe. 
############################################################################

import os
import exkaldi.graph as G

csjPath = "kaldi/egs/csj/s5/"
newGraphDir = "./graph"

csjPath = os.path.abspath(csjPath)
if not os.path.isdir(newGraphDir):
    os.makedirs(newGraphDir)

# --------------------- 1 prepare dictionaries and make L-------------
# Firstly, we will prepare a exkald.LexiconBank object.
# The graph.LexiconBank object accepts a lexicon file and generates a series of related lexicons automatically.

pronLexicon = csjPath + "/data/local/dict/lexicon.txt"
silWords = ["<sp>"]
unkSymbol = "<unk>"
optionalSilPhone="sp"
extraQuestions=[]
positionDependent = True
shareSilPdf = False
extraDisambigPhoneNumbers = 1

print('Start to generate dictionaries')
print('Source file:{}'.format(pronLexicon))

dictionary = G.LexiconBank(pronLexicon,silWords,unkSymbol,optionalSilPhone,extraQuestions,positionDependent,shareSilPdf,extraDisambigPhoneNumbers)
print('Generate Done')

# The lexiocn, [phones] and [words], have been generated defaultly in the "dictionary" object.
# But in this task, we need to uniformize them as same as the corresponding lexicons of CSJ. 
phone2intTable = csjPath + "/data/lang_nosp/phones.txt"
print('Resert phone-int table from:{}'.format(phone2intTable))
dictionary.reset_phones(phone2intTable)

word2intTable = csjPath + "/data/lang_nosp/words.txt" 
print('Resert word-int table from:{}'.format(word2intTable))
dictionary.reset_words(word2intTable)

# Now, we start to generate L fst.
# We will apply the disambiguation.
print('Start to make disambiguating L (That will take a few seconds...).')
Lfile = newGraphDir + "/L_disambig.fst"
Lfile = G.make_L(dictionary, outFile=Lfile, useSilprob=False, silProb=0.5, useDisambig=True)
print('Make L Done:{}'.format(Lfile))

# --------------------- 2 train ARPA Language Model and make G-------------
# Then, we start to train a new n-grams language model and make the G file.
# We will train a 4-gram LM which has a higher ranking than 3 of CSJ baseline.
textFile = csjPath + "/data/local/lm/train.gz"
n = 4
nGramsFile = newGraphDir + "/{}grams.gz".format(n)
print('Start to train a {}-grams LM from source file:{}.'.format(n,textFile))
nGramsFile = G.train_ngrams(dictionary, n, textFile, outFile=nGramsFile, discount="kndiscount")
print('Make Arpa-LM Done:{}'.format(nGramsFile))

# Then transform ARPA LM to G file.
print('Start to make G (That will take a few seconds...).')
Gfile = newGraphDir + "/G.fst"
Gfile = G.make_G(dictionary, nGramsFile, outFile=Gfile, n=n)
print('Make G Done:{}'.format(Gfile))


# --------------------- 3 compose to HCLG -------------
# Now, we start to compose the final HCLG graph.
# The composing operation will be performed orderly: L+G=LG -> LG+tree=CLG -> CLG+hmm=HCLG .
# As mentioned above, we will utilize the existed HMM and tree file.
hmmFile = csjPath + "/exp/tri1/final.mdl"
treeFile = csjPath + "/exp/tri1/tree"

# Compose the L and G.
print('Start to make LG (That will take a few seconds...)')
LGfile = newGraphDir + "/LG.fst"
LGfile = G.compose_LG(Lfile, Gfile, outFile=LGfile)
print('Compose LG Done:{}'.format(LGfile))

# Compose the LG and tree. The compose_CLG will return not only the absolute path of CLG file but also the ilabel file.
# The ilabel file is necessary when composing the HCLG.
print('Start to make CLG. That will take a few minutes...')
CLGfile = newGraphDir + "/CLG.fst"
CLGfile, iLabelInfoFile = G.compose_CLG(dictionary, LGfile, treeFile, outFile=CLGfile)
print('Compose CLG Done:{}'.format(CLGfile))

# Compose the CLG and hmm.
print('Start to make HCLG. That will take a few minutes...')
HCLGfile = newGraphDir + "/HCLG4.fst"
HCLGfile = G.compose_HCLG(CLGfile, hmmFile, treeFile, iLabelInfoFile, outFile=HCLGfile, transScale=1.0, loopScale=0.1)
print('Compose HCLG Done:{}'.format(HCLGfile))

# Save the lexicon, [words] to file.
print("Write word-int tale (It will be useful to decoding)")
wordsFile = newGraphDir + "/words.txt"
dictionary.dump_dict("words",wordsFile)

print('Make HCLG Graph Done!')

## --------------------- 4 score WER ----------------
# In this version, we don't support decode by HMM-GMM model, so use the Kaldi Baseline to compute the WER.
# Run the shell command: 

#>> cd kaldi/egs/csj/s5
#>> . ./path.sh
#>> . ./cmd.sh
#>> steps/decode_si.sh --nj 4 --cmd "$decode_cmd" --config conf/decode.config new_graph_dir data/eval1 exp/tri1/decode_eval1_csj_4g

# The decoding result will be output in exp/tri1/decode_eval1_csj_4g. 
# You would get the best WER information just like this:
#>> %WER 22.33 [ 5812 / 26028, 468 ins, 1497 del, 3847 sub ] exp/tri1/decode_eval1_csj_4g/wer_13_0.0
# Which is better than the result of 3-grams:
#>> %WER 22.41 [ 5834 / 26028, 494 ins, 1486 del, 3854 sub ] exp/tri1/decode_eval1_csj/wer_12_0.0