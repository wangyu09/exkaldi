# ExKaldi 1.0 
# A wrapper for Kaldi ASR
ExKaldi toolkit is an wapper toolkit for Kaldi speech recognition toolkit. 
It is developed to help users customize speech recognition system easily with Python language.
We will further introduce the characteristics by some instances.

## Version 1.0

1. Version 1.0 is brand-new comparing to exkaldi 0.x.  

2. More strict data class management.

3. Support HMM-GMM training and decoding, End-to-End training and decoding, and N-grams model applying.

4. Use KenLm as languange model backend, but srilm is still avaliable.

5. Exkaldi C++ library was designed fro some functions.

Exkaildi 1.0 is still developing and its PyPi package is unavaliable.
The source code is public and you can know the design concept of exkaldi through them.
There are some instances to show how to use Exkaldi toolkit.

## Record a voice data
```
import exkaldi as E
wavFile = E.audio.record_voice(outFile="my.wav", seconds=10)
print(wavFile)
```
Read voice data from microphone stream and save it to file "my.wav".

## Processing feature
```
# Compute feature
feat = E.compute_mfcc("wav.scp", rate=16000, name="mfcc")

# Save feature
feat.save("raw_mfcc.ark", chunks=4, outScpFile=True)

#Load feature file
feat = E.load_feat("raw_mfcc_01.scp")

# Splice left-right 3 frames
feat = feat.splice(3)

# Apply CMVN statistics
cmvn = E.compute_cmvn_stats(feat, "spk2utt")
feat = E.use_cmvn(feat, cmvn, "utt2spk")

# Transform to Numpy format
feat = feat.to_numpy()
```

## Load alignment for training NN acoustic model
```
ali = E.load_ali("ali.*.gz")
# Get phone level alignment
aliPhone = ali.to_numpy(target="phoneID", hmm="final.mdl", name="phone")
# Get pdf level alignment
aliPdf = ali.to_numpy(target="pdfID", hmm="final.mdl", name=pdf)

# Tuple feature and alignment for NN training
dataset = E.tuple_data([feat, aliPhone, aliPdf], frameLevel=True):
```

## Generate lexicons 
We design a class LexiconBank to hold and manage all lexicons.
```
# Initialize all lexicons from provided "lexicon.txt" file
lexicons = E.decode.lexicon_bank("lexicon.txt", positionDependent=True)

# Display the names of all lexicons
print(lexicons.view)

# Save specified lexicon to file
lexicons.dump_dict(name="phones", outFile="./phones.txt")
```

## Compose WFST decoding graph
```
# Make L.fst
E.decode.make_L(lexicons, outFile="L_disanbig.fst", useDisambig=True)

# Make G.fst
E.decode.make_G(lexicons, arpaFile="3_grams.arpa", outFile="G.fst", n=3)

# Compose LG.fst
E.decode.compose_LG("L_disanbig.fst", "G.fst", outFile="LG.fst")

# Compose CLG.fst
E.decode.compose_CLG(lexicons, tree, "LG.fst", outFile="CLG.fst")

# Compose CLG.fst
clg, ilabel = E.decode.compose_CLG(lexicons, tree, "LG.fst", outFile="CLG.fst")

# Compose HCLG.fst
E.decode.compose_HCLG(hmm, tree, clg, ilabel, outFile="HCLG.fst")
```

## Training a language model
```
# Use srilm backend
E.lm.train_ngrams_srilm(lexicons, order=3, "text", outFile="3_grams.arpa")

# use kenlm backend
E.lm.train_ngrams_kenlm(lexicons, order=3, "text", outFile="3_grams.arpa")

# Convert ARPA file to kenlm binary format
E.lm.arpa_to_binary("3_grams.arpa", "3_grams.bianry")

# Query a language model
model = E.lm.KenNGrams("3_grams.bianry")
model.score("I love my country")
model.perplexity("I am a boy")
```

## Training HMM-GMM model
```
# Make a toponology
E.hmm.make_toponology(lexicons, outFile="topo")

# Train a monophone HMM-GMM
monoModel = E.hmm.MonophoneHMM(lexicons, name="mono")
monoModel.initialize(feat, topoFile="topo")
monoModel.train(feat, "text", "L.fst", tempDir="./trainhmm")

# Train a Desicion-Tree
tree = E.hmm.DesicionTree(lexicons)
tree.train(feat, monoModel, alignFile="./trainhmm/ali.gz", topoFile="topo", tmpdir="./trainhmm")
tree.save("tree")

# Train a triphone HMM-GMM (up to train_delta step)
triModel = E.hmm.TriphoneHMM(lexicons, name="tri")
triModel.initialize(tree, "./trainhmm/treeacc", "topo", numgauss=1000)
newAlign = E.hmm.convert_alignment("./trainhmm/ali.gz", monoModel, triModel, tree, "./trainhmm/new_ali.gz")
triModel.train(feat, "text", "L.fst", tree, "./trainhmm")
triModel.save("final.mdl")
```

## Decoding and score
```
# Decoding DMM-HMM.
norm = E.compute_postprob_norb(ali)
prob = prob.map(lambda x: x - norm ) 
lat = E.decode.nn_decode(prob, triModel, "HCLG.fst", lexicons)

# Decoding GMM-HMM
lat = E.decode.gmm_decode(feat, triModel, "HCLG.fst", lexicons)

# Process lattice
lat = lat.add_penalty(penalty=0.5)

# Get 1-best results.
results = lat.get_1best(lexicons, triModel)

# Compute WER
score = E.decode.score.wer("text", results, mode="all")
print(score.WER)

# End-2-End decoding
prob = E.decode.convert_field(prob, originalVocabs, targetVocabs)
results = E.decode.ctc_greedy_decode(prob, vocabs)

# Compute edit distance
score = E.decode.score.edit_distance("text", results, mode="all")
print("Words Accuracy:", 1 - score.editDistance / score.words )
print("Sentence Accuracy:", 1 - score.wongSentences / score.sentences )
```