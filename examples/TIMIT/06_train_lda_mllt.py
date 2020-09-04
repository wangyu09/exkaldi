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

Part 5: train a GMM-HMM model with lda+mllt feature.

'''
import os
import glob
import gc
import time

import exkaldi
from exkaldi import args
from exkaldi import declare

from make_graph_and_decode import make_WFST_graph, GMM_decode_mfcc_and_score

def main():

    # ------------- Parse arguments from command line ----------------------
    # 1. Add a discription of this program
    args.discribe("This program is used to train triphone GMM-HMM model") 
    # 2. Add options
    args.add("--expDir", abbr="-e", dtype=str, default="exp", discription="The data and output path of current experiment.")
    args.add("--splice", abbr="-c", dtype=int, default=3, discription="How many left-right frames to splice.")
    args.add("--numIters", abbr="-n", dtype=int, default=35, discription="How many iterations to train.")
    args.add("--maxIterInc", abbr="-m", dtype=int, default=25, discription="The final iteration of increasing gaussians.")
    args.add("--realignIter", abbr="-r", dtype=int, default=[10,20,30], discription="the iteration to realign feature.")
    args.add("--mlltIter", abbr="-i", dtype=int, default=[2,4,6,12], discription="the iteration to estimate mllt matrix.")
    args.add("--order", abbr="-o", dtype=int, default=6, discription="Which N-grams model to use.")
    args.add("--beam", abbr="-b", dtype=int, default=13, discription="Decode beam size.")
    args.add("--latBeam", abbr="-l", dtype=int, default=6, discription="Lattice beam size.")
    args.add("--acwt", abbr="-a", dtype=float, default=0.083333, discription="Acoustic model weight.")
    args.add("--parallel", abbr="-p", dtype=int, default=4, minV=1, maxV=10, discription="The number of parallel process to compute feature of train dataset.")
    args.add("--skipTrain", abbr="-s", dtype=bool, default=False, discription="If True, skip training. Do decoding only.")
    # 3. Then start to parse arguments. 
    args.parse()
    # 4. Take a backup of arguments
    argsLogFile = os.path.join(args.expDir, "conf", "train_lda_mllt.args")
    args.save(argsLogFile)

    if not args.skipTrain:
        # ------------- Prepare feature and previous alignment for training ----------------------
        # 1. Load the feature for training
        print(f"Load MFCC+CMVN feature.")
        feat = exkaldi.load_index_table(os.path.join(args.expDir,"mfcc","train","mfcc_cmvn.ark"))
        print(f"Splice {args.splice} frames.")
        originalFeat = exkaldi.splice_feature(feat,left=args.splice,right=args.splice,outFile=os.path.join(args.expDir,"train_delta","mfcc_cmvn_splice.ark"))
        # 2. Load previous alignment and lexicons
        ali = exkaldi.load_index_table(os.path.join(args.expDir,"train_delta","*final.ali"),useSuffix="ark")
        lexicons = exkaldi.load_lex(os.path.join(args.expDir,"dict","lexicons.lex"))
        # 3. Estimate LDA matrix and convert feature
        print("Estimate LDA matrix.")
        exkaldi.hmm.accumulate_LDA_stats(
                        ali=ali, 
                        lexicons=lexicons, 
                        hmm=os.path.join(args.expDir,"train_delta","final.mdl"), 
                        feat=originalFeat, 
                        outFile=os.path.join(args.expDir,"train_lda_mllt","ldaStats.acc"),
                        silenceWeight=0,
                        randPrune=4,
                    )
        exkaldi.hmm.estimate_LDA_matrix(
                                    statsFiles=os.path.join(args.expDir,"train_lda_mllt","*ldaStats.acc"), 
                                    targetDim=40, 
                                    outFile=os.path.join(args.expDir,"train_lda_mllt","trans.mat")
                                )
        print("Transform feature")
        ldaFeat = exkaldi.transform_feat(
                                feat=originalFeat, 
                                matFile=os.path.join(args.expDir,"train_lda_mllt","trans.mat"),
                                outFile=os.path.join(args.expDir,"train_lda_mllt","lda_feat.ark"),
                            )

        # -------------- Build the decision tree ------------------------
        print("Start build a tree")
        tree = exkaldi.hmm.DecisionTree(lexicons=lexicons, contextWidth=3, centralPosition=1)
        tree.train(
                    feat=ldaFeat, 
                    hmm=os.path.join(args.expDir,"train_delta","final.mdl"), 
                    ali=ali,
                    topoFile=os.path.join(args.expDir,"dict","topo"), 
                    numLeaves=2500,
                    tempDir=os.path.join(args.expDir,"train_lda_mllt"), 
                )
        tree.save(os.path.join(args.expDir,"train_delta","tree"))
        print(f"Build tree done.")
        del ldaFeat

        # ------------- Start training ----------------------
        # 1. Initialize a monophone HMM object
        print("Initialize a triphone HMM object")
        model = exkaldi.hmm.TriphoneHMM(lexicons=lexicons)
        model.initialize(
                    tree=tree, 
                    topoFile=os.path.join(args.expDir,"dict","topo"),
                    treeStatsFile=os.path.join(args.expDir,"train_lda_mllt","treeStats.acc"), 
                )    
        print(f"Initialized a monophone HMM-GMM model: {model.info}.")

        # 2. convert the previous alignment
        print(f"Transform the alignment")
        newAli = exkaldi.hmm.convert_alignment(
                                        ali=ali,
                                        originHmm=os.path.join("exp","train_delta","final.mdl"), 
                                        targetHmm=model, 
                                        tree=tree,
                                        outFile=os.path.join(args.expDir,"train_lda_mllt","initial.ali"),
                                    )

        # 2. Split data for parallel training
        transcription = exkaldi.load_transcription(os.path.join(args.expDir,"data","train","text"))
        transcription = transcription.sort()
        if args.parallel > 1:
            # split feature
            originalFeat = originalFeat.sort(by="utt").subset(chunks=args.parallel)
            # split transcription depending on utterance IDs of each feat
            tempTrans = []
            tempAli = []
            for f in originalFeat:
                tempTrans.append( transcription.subset(keys=f.utts) )
                tempAli.append( newAli.subset(keys=f.utts) )
            transcription = tempTrans
            newAli = tempAli

        # 3. Train
        print("Train the triphone model")
        model.train(
                    originalFeat,
                    transcription, 
                    os.path.join("exp","dict","L.fst"), 
                    tree,
                    tempDir=os.path.join(args.expDir,"train_lda_mllt"),
                    initialAli=newAli,
                    ldaMatFile=os.path.join(args.expDir,"train_lda_mllt","trans.mat"),
                    numIters=args.numIters, 
                    maxIterInc=args.maxIterInc,
                    totgauss=15000,
                    realignIter=args.realignIter,
                    mlltIter=args.mlltIter,
                )
        print(model.info)
        # Save the tree
        del originalFeat
        del newAli
    
    else:
        declare.is_file( os.path.join(args.expDir,"train_lda_mllt","final.mdl") )
        declare.is_file( os.path.join(args.expDir,"train_lda_mllt","tree") )
        model = exkaldi.load_hmm( os.path.join(args.expDir,"train_lda_mllt","final.mdl") )
        tree = exkaldi.load_tree( os.path.join(args.expDir,"train_lda_mllt","tree") )

    # ------------- Compile WFST training ----------------------
    # Make a WFST decoding graph
    make_WFST_graph(
                outDir=os.path.join(args.expDir,"train_lda_mllt","graph"),
                hmm=model,
                tree=tree,
            )
    # Decode test data
    GMM_decode_mfcc_and_score(
                outDir=os.path.join(args.expDir,"train_lda_mllt",f"decode_{args.order}grams"), 
                hmm=model,
                HCLGfile=os.path.join(args.expDir,"train_lda_mllt","graph",f"HCLG.{args.order}.fst"),
                tansformMatFile=os.path.join(args.expDir,"train_lda_mllt","trans.mat"),
            )

if __name__ == "__main__":
    main()