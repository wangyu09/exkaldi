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
Train and test N-Grams language models.
'''

import exkaldi
from exkaldi import args
from exkaldi import declare
import os

def main():

  # 1, Parse command line options.
  args.add("--order",abbr="-o",dtype=int,default=[2,3,4,5,6],minV=1,maxV=6,discription="The language model order.")
  args.add("--dataDir",abbr="-d",dtype=str,default="./exp",discription="The resource directory.")
  args.add("--exp",abbr="-e",dtype=str,default="./exp_lms",discription="Experiment output directory.")
  args.parse()

  # 2, Prepare text and lexicon.
  declare.is_file(f"./{args.dataDir}/data/train/text", debug="There is not train text file avaliable. Please run '01_prepare_data.py' to generate it.")
  declare.is_file(f"./{args.dataDir}/data/test/text", debug="There is not train text file avaliable. Please run '01_prepare_data.py' to generate it.")
  declare.is_file(f"./{args.dataDir}/dict/lexicons.lex", debug="There is not lexicon file avaliable. Please run '02_make_dict_and_LM.py' to generate it.")

  trainTrans = exkaldi.load_transcription(f"./{args.dataDir}/data/train/text")
  lexicons = exkaldi.load_lex(f"./{args.dataDir}/dict/lexicons.lex")
  testTrans = exkaldi.load_transcription(f"./{args.dataDir}/data/test/text")

  # 3, Train LMs and compute perplexities.
  for o in args.order:

    for backend in ["sri","ken"]:

      if backend == "sri":
        exkaldi.lm.train_ngrams_srilm(
                                lexicons,
                                order=o,
                                text=trainTrans,  # If "text" received an exkaldi Transcription object, the information of utterance IDs will be omitted automatically.
                                outFile=os.path.join(args.exp,f"{backend}_{o}grams.arpa"),
                                config={"-wbdiscount":True},
                              )
                            
      else:
        exkaldi.lm.train_ngrams_kenlm(
                                lexicons,
                                order=o,
                                text=trainTrans,  # If "text" received an exkaldi Transcription object, the information of utterance IDs will be omitted automatically.
                                outFile=os.path.join(args.exp,f"{backend}_{o}grams.arpa"),
                                config={"--discount_fallback":True,"-S":"20%"},
                              )        

      exkaldi.lm.arpa_to_binary(
                              arpaFile=os.path.join(args.exp,f"{backend}_{o}grams.arpa"),
                              outFile=os.path.join(args.exp,f"{backend}_{o}grams.binary"),
                            )
    
      model = exkaldi.load_ngrams( os.path.join(args.exp,f"{backend}_{o}grams.binary") )

      perScore = model.perplexity(testTrans)
      print( f"{o} {backend} score:", perScore )

if __name__ == "__main__":
  main()