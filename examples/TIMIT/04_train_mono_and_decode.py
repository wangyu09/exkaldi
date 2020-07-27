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

Part 4: train monophone GMM-HMM model.

'''
import os
import glob
import subprocess
import gc
import time

import exkaldi
from exkaldi import args

from make_graph_and_decode import make_WFST_graph, GMM_decode_mfcc_and_score

def main():

    # ------------- Parse arguments from command line ----------------------
    # 1. Add a discription of this program
    args.discribe("This program is used to train monophone GMM-HMM model") 
    # 2. Add options
    args.add("--expDir", abbreviation="-e", dtype=str, default="exp", discription="The data and output path of current experiment.")
    args.add("--delta", abbreviation="-d", dtype=int, default=2, discription="Add n-order to feature.")
    args.add("--numIters", abbreviation="-n", dtype=int, default=40, discription="How many iterations to train.")
    args.add("--maxIterInc", abbreviation="-m", dtype=int, default=30, discription="The final iteration of increasing gaussians.")
    args.add("--totgauss", abbreviation="-t", dtype=int, default=1000, discription="Target number of gaussians.")
    args.add("--boostSilence", abbreviation="-s", dtype=float, default=1.0, discription="Boost silence.")
    args.add("--realignIter", abbreviation="-r", dtype=int, 
                              default=[1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,23,26,29,32,35,38], 
                              discription="the iteration to realign feature."
                            )
    args.add("--order", abbreviation="-o", dtype=int, default=3, discription="Which N-grams model to use.")
    args.add("--beam", abbreviation="-b", dtype=int, default=13, discription="Decode beam size.")
    args.add("--latBeam", abbreviation="-l", dtype=int, default=6, discription="Lattice beam size.")
    args.add("--acwt", abbreviation="-a", dtype=float, default=0.083333, discription="Acoustic model weight.")
    args.add("--parallel", abbreviation="-p", dtype=int, default=4, minV=1, maxV=10, discription="The number of parallel process to compute feature of train dataset.")
    # 3. Then start to parse arguments. 
    args.parse()
    # 4. Take a backup of arguments
    args.print_args() # print arguments to display
    argsLogFile = os.path.join(args.expDir, "conf", "train_mono.args")
    args.save(argsLogFile)

    # ------------- Prepare feature for training ----------------------
    # 1. Load the feature for training (We use the index table format)
    feat = exkaldi.load_index_table(os.path.join(args.expDir,"mfcc","train","mfcc_cmvn.ark"))
    print(f"Load MFCC+CMVN feature.")
    feat = exkaldi.add_delta(feat, order=args.delta, outFile=os.path.join(args.expDir,"train_mono","mfcc_cmvn_delta.ark"))
    print(f"Add {args.delta}-order deltas.")
    # 2. Load lexicon bank
    lexicons = exkaldi.load_lex(os.path.join(args.expDir,"dict","lexicons.lex"))
    print(f"Restorage lexicon bank.")
    
    # ------------- Start training ----------------------
    # 1. Initialize a monophone HMM object
    model = exkaldi.hmm.MonophoneHMM(lexicons=lexicons, name="mono")
    model.initialize(
                feat=feat,
                topoFile=os.path.join(args.expDir,"dict","topo")
            )
    print(f"Initialized a monophone HMM-GMM model: {model.info}.")

    # 2. Split data for parallel training
    transcription = exkaldi.load_transcription(os.path.join(args.expDir,"data","train","text"))
    transcription = transcription.sort()
    if args.parallel > 1:
        # split feature
        feat = feat.sort(by="utt").subset(chunks=args.parallel)
        # split transcription depending on utterance IDs of each feat
        temp = []
        for f in feat:
            temp.append( transcription.subset(uttIDs=f.utts) )
        transcription = temp

    # 3. Train
    model.train(
                feat,
                transcription, 
                LFile=os.path.join(args.expDir,"dict","L.fst"),
                tempDir=os.path.join(args.expDir,"train_mono"),
                numIters=args.numIters, 
                maxIterInc=args.maxIterInc, 
                totgauss=args.totgauss, 
                realignIter=args.realignIter,
                boostSilence=args.boostSilence,
            )
    print(model.info)
    # Save the tree
    model.tree.save(os.path.join(args.expDir,"train_mono","tree"))
    print(f"Tree has been saved.")

    # 4. Realign with boostSilence 1.25
    print("Realign the training feature (boost silence = 1.25)")
    trainGraphFiles = exkaldi.utils.list_files(os.path.join(args.expDir,"train_mono","*train_graph"))
    model.align(
                feat,
                trainGraphFile=trainGraphFiles,  # train graphs have been generated in the train step.
                boostSilence=1.25, #1.5
                outFile=os.path.join(args.expDir,"train_mono","final.ali")
            )
    del feat
    print("Save the new alignment done.")

    # ------------- Compile WFST training ----------------------
    # Make a WFST decoding graph
    make_WFST_graph(
                outDir=os.path.join(args.expDir,"train_mono","graph"),
                hmm=model,
                tree=model.tree,
            )
    # Decode test data
    GMM_decode_mfcc_and_score(
                outDir=os.path.join(args.expDir,"train_mono","decode"), 
                hmm=model,
                HCLGfile=os.path.join(args.expDir,"train_mono","graph",f"HCLG.{args.order}.fst"),
            )

if __name__ == "__main__":
    main()