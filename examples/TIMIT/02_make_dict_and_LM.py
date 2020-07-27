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
Timit HMM-GMM training recipe.

Part 2: Make dictionaries and language model.

'''
import os
import glob
import subprocess
import gc

import exkaldi
from exkaldi import args

def main():
    
    # ------------- Parse arguments from command line ----------------------
    # 1. Add a discription of this program
    args.discribe("This program is used to make dictionaries and language model") 
    # 2. Add options
    args.add("--expDir", abbreviation="-e", dtype=str, default="exp", discription="The data and output path of current experiment.")
    args.add("--order", abbreviation="-o", dtype=int, default=3, minV=1, maxV=6, discription="The maximum order of N-grams language model.")
    # 3. Then start to parse arguments. 
    args.parse()
    # 4. Take a backup of arguments
    args.print_args() # print arguments to display
    argsLogFile = os.path.join(args.expDir, "conf", "make_dict_and_LM.args")
    args.save(argsLogFile)

    # ------- Make the word-pronumciation lexicon file ------
    textFile = os.path.join(args.expDir,"data","train","text")
    trainTrans = exkaldi.load_transcription(textFile) # trans is an exkaldi Transcription object
    wordCount = trainTrans.count_word().sort() # accumulate all words and their frequency in the transcription

    word2pron = dict( (word, word) for word in wordCount.keys() ) # word to pronunciation
    word2pron = exkaldi.ListTable(word2pron) 
    pronFile = os.path.join(args.expDir, "dict", "pronunciation.txt")
    word2pron.save( pronFile ) # save it to file

    # -------  Make lexicons ------
    # 1. Generate the LexiconBank object from word-pronumciation file. 
    # Depending on task, about 20 lexicons will be generated and managed by the LexiconBank.
    lexicons = exkaldi.decode.graph.lexicon_bank(
                pronFile, 
                silWords={"sil":"sil"},  
                unkSymbol={"sil":"sil"}, 
                optionalSilPhone="sil",
                extraQuestions=[],
                positionDependent=False, 
                shareSilPdf=False,
                extraDisambigPhoneNumbers=1,
                extraDisambigWords=[]
            )

    # 2. Add two extra questions.
    lexicons.add_extra_question(lexicons("silence_phones"))
    lexicons.add_extra_question(lexicons("nonsilence_phones"))

    # 3. Save this lexicon bank for future use.
    lexicons.save(os.path.join(args.expDir,"dict","lexicons.lex"))
    print(f"Generate lexicon bank done.")

    # -------  Make Lexicon fst ------
    # 1. Generate the Lexicon fst
    exkaldi.decode.graph.make_L(
                            lexicons, 
                            outFile=os.path.join(args.expDir,"dict","L.fst"), 
                            useSilprob=0.0, 
                            useDisambigLexicon=False
                        )
    print(f"Generate lexicon fst done.")
    # 1. Generate the disambig Lexicon fst
    exkaldi.decode.graph.make_L(
                            lexicons, 
                            outFile=os.path.join(args.expDir,"dict","L_disambig.fst"), 
                            useSilprob=0.0, 
                            useDisambigLexicon=True
                        )
    print(f"Generate disambiguation lexicon fst done.")

    # -------  Make GMM-HMM topological structure for GMM-HMM ------
    exkaldi.hmm.make_toponology(
                            lexicons,
                            outFile=os.path.join(args.expDir,"dict","topo"),
                            numNonsilStates=3,
                            numSilStates=5,
                        )
    print(f"Generate topo file done.")

    # -------  Train N-Grams language model ------
    # 1. Train a LM.
    # We have trained 2,3,4 grams model with both srilm and kenlm and chose the best one, which is 3-grams model back kenlm.
    # So we directly train this one.
    exkaldi.lm.train_ngrams_kenlm(
                            lexicons,
                            order=args.order,
                            text=trainTrans,  # If "text" received an exkaldi Transcription object, the infomation of utterance IDs will be omitted automatically.
                            outFile=os.path.join(args.expDir,"lm",f"{args.order}grams.arpa"), 
                            config={"--discount_fallback":True,"-S":"20%"},
                        )
    print(f"Generate ARPA language model done.")

    # 2. Then test this model by compute the perplexity.
    exkaldi.lm.arpa_to_binary(
                            arpaFile=os.path.join(args.expDir,"lm",f"{args.order}grams.arpa"),
                            outFile=os.path.join(args.expDir,"lm",f"{args.order}grams.binary"),
                        )
    model = exkaldi.load_ngrams( os.path.join(args.expDir,"lm",f"{args.order}grams.binary") ) # Actually, "load_ngrams" also accept arpa format file.

    # 3. Prepare test transcription
    testTrans = exkaldi.load_transcription(os.path.join(args.expDir,"data","test","text"))

    # 4. score
    perScore = model.perplexity(testTrans)
    meanScore = perScore.mean(weight=testTrans.sentence_length())
    print(f"The weighted average perplexity of this model is: {meanScore}.")
    del model
    del testTrans

    # ------- Make Grammar fst ------
    exkaldi.decode.graph.make_G(
                            lexicons, 
                            arpaFile=os.path.join(args.expDir,"lm",f"{args.order}grams.arpa"),
                            outFile=os.path.join(args.expDir,"lm",f"G.{args.order}.fst"), 
                            order=args.order
                        )
    print(f"Make Grammar fst done.")

    # ------- Compose LG fst for futher use ------
    exkaldi.decode.graph.compose_LG(
                            Lfile=os.path.join(args.expDir,"dict","L_disambig.fst"), 
                            Gfile=os.path.join(args.expDir,"lm",f"G.{args.order}.fst"),
                            outFile=os.path.join(args.expDir,"lm",f"LG.{args.order}.fst"),
                        )
    print(f"Compose LG fst done.")

if __name__ == "__main__":
    main()