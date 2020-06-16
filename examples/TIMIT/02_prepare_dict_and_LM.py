# coding=utf-8
#
# Yu Wang (University of Yamanashi)
# Jun, 2020
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Prepare the pronunciation lexicons.
'''

import os
import glob
import subprocess

import exkaldi
from exkaldi.version import version as ExkaldiInfo

import cfg

# prepare the pronumciation lexicon
allWords = []
with open( os.path.join(cfg.outDir, "data", "train", "text"), "r", encoding="UTF-8" ) as fr:
    lines = fr.readlines()
    for line in lines:
        line = line.strip().split()
        if len(line) < 2:
            continue
        else:
            allWords.extend( line[1:] )
allWords = sorted(list(set(allWords)))
allWords = list( map(lambda x: f"{x} {x}", allWords) )

pronFile = os.path.join(cfg.outDir, "data", "dict", "pronunciation.txt")
exkaldi.utils.make_dependent_dirs(path=pronFile, pathIsFile=True)
with open(pronFile, "w") as fw:
    fw.write( "\n".join(allWords) )

# generate a lexiconBank object.
lexicons = exkaldi.decode.graph.lexicon_bank(
            pronFile, 
            silWords={ cfg.silWord:cfg.silPron}, 
            unkSymbol={ cfg.unkSymbol:cfg.unkPron }, 
            optionalSilPhone=cfg.optionalSil,
            extraQuestions=[],
			positionDependent=False, 
            shareSilPdf=False,
            extraDisambigPhoneNumbers=1, 
            extraDisambigWords=[]
        )

# add two extra questions
lexicons.add_extra_question( lexicons("silence_phones") )
lexicons.add_extra_question( lexicons("nonsilence_phones") )

# save this lexicon bank
lexFile = os.path.join(cfg.outDir, "dict", "lexicons.lex")
lexicons.save(lexFile)
print(f"Generate lexicon bank done. Saved in: {lexFile}.")

# Generate the L.fst
LFile = os.path.join(cfg.outDir, "dict", "L_disambig.fst")
exkaldi.decode.graph.make_L(lexicons, outFile=LFile, useSilprob=0.0, useDisambigLexicon=True)
print(f"Generate L.fst. Saved in: {LFile}.")

# Train the ARPA language model
lmDir = os.path.join(cfg.outDir, "lm")
exkaldi.utils.make_dependent_dirs(lmDir, pathIsFile=False)

with open( os.path.join(cfg.outDir,"data","train","text"), "r", encoding="utf-8") as fr:
    lines = fr.readlines()
newLines = []
for line in lines:
    newLines.append( line.split(maxsplit=1)[1] )
trainLmTextFile = os.path.join(lmDir,"train_lm_text")
with open( trainLmTextFile, "w") as fw:
    fw.writelines(newLines)

# We have trained 2,3,4 grams model with both srilm and kenlm and chose the best one, that is 3-grams model back kenlm.
# So we directly train this one.
arpaFile = os.path.join(lmDir, f"{cfg.lmOrder}grams.arpa")
kenlmConfig = {"--discount_fallback":True} 
exkaldi.lm.train_ngrams_kenlm(lexicons, order=cfg.lmOrder, textFile=trainLmTextFile, outFile=arpaFile, config=kenlmConfig)
print(f"Generate ARPA language model done. Saved in: {arpaFile}.")

## Then test this model by compute the perplexity
# 1, make a Kenlm model object.
binaryFile = os.path.join(lmDir, f"{cfg.lmOrder}grams.binary")
exkaldi.lm.arpa_to_binary(arpaFile, binaryFile)
model = exkaldi.lm.KenNGrams(binaryFile)
# 2, prepare test transcription
testTransFile = os.path.join(cfg.outDir, "data", "test", "text")
testTrans = exkaldi.load_trans(testTransFile, name="testTrans")
# 3, score
perScore = model.perplexity(testTrans)
meanScore = perScore.mean(weight=testTrans.sentence_lenght())
print(f"The weighted average perplexity of this model is: {meanScore}.")
