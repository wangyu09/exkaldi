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

Part 5: train a GMM-HMM model with fmllr feature.

'''
import os
import glob
import subprocess
import gc
import time

import exkaldi
from exkaldi import args

from make_graph_and_decode import make_WFST_graph, GMM_decode_fmllr_and_score

def main():

    # ------------- Parse arguments from command line ----------------------
    # 1. Add a discription of this program
    args.discribe("This program is used to train triphone GMM-HMM model") 
    # 2. Add options
    args.add("--expDir", abbreviation="-e", dtype=str, default="exp", discription="The data and output path of current experiment.")
    args.add("--splice", abbreviation="-c", dtype=int, default=3, discription="How many left-right frames to splice.")
    args.add("--numIters", abbreviation="-n", dtype=int, default=35, discription="How many iterations to train.")
    args.add("--maxIterInc", abbreviation="-m", dtype=int, default=25, discription="The final iteration of increasing gaussians.")
    args.add("--numleaves", abbreviation="-nl", dtype=int, default=2500, discription="Target number of gaussians.")
    args.add("--totgauss", abbreviation="-t", dtype=int, default=15000, discription="Target number of gaussians.")
    args.add("--realignIter", abbreviation="-r", dtype=int, default=[10,20,30], discription="the iteration to realign feature.")
    args.add("--fmllrIter", abbreviation="-f", dtype=int, default=[2,4,6,12], discription="the iteration to estimate fmllr matrix.")
    args.add("--order", abbreviation="-o", dtype=int, default=3, discription="Which N-grams model to use.")
    args.add("--beam", abbreviation="-b", dtype=int, default=13, discription="Decode beam size.")
    args.add("--latBeam", abbreviation="-l", dtype=int, default=6, discription="Lattice beam size.")
    args.add("--acwt", abbreviation="-a", dtype=float, default=0.083333, discription="Acoustic model weight.")
    args.add("--parallel", abbreviation="-p", dtype=int, default=4, minV=1, maxV=10, discription="The number of parallel process to compute feature of train dataset.")
    # 3. Then start to parse arguments. 
    args.parse()
    # 4. Take a backup of arguments
    args.print_args() # print arguments to display
    argsLogFile = os.path.join(args.expDir, "conf", "train_sat.args")
    args.save(argsLogFile)

    # ------------- Prepare feature and previous alignment for training ----------------------
    # 1. Load the feature for training
    print(f"Load MFCC+CMVN feature.")
    feat = exkaldi.load_index_table(os.path.join(args.expDir,"mfcc","train","mfcc_cmvn.ark"))
    print(f"Splice {args.splice} frames.")
    originalFeat = exkaldi.splice_feature(feat,left=args.splice,right=args.splice,outFile=os.path.join(args.expDir,"train_delta","mfcc_cmvn_splice.ark"))
    print(f"Transform LDA feature")
    ldaFeat = exkaldi.transform_feat(
                            feat=originalFeat, 
                            matrixFile=os.path.join(args.expDir,"train_lda_mllt","trans.mat"),
                            outFile=os.path.join(args.expDir,"train_sat","lda_feat.ark"),
                        )
    del originalFeat
    # 2. Load previous alignment and lexicons
    ali = exkaldi.load_index_table(os.path.join(args.expDir,"train_lda_mllt","*final.ali"),useSuffix="ark")
    lexicons = exkaldi.load_lex(os.path.join(args.expDir,"dict","lexicons.lex"))
    # 3. Estimate the primary fMLLR transform matrix
    print("Estiminate the primary fMLLR transform matrixs")
    fmllrTransMat = exkaldi.hmm.estimate_fMLLR_matrix(
                                        aliOrLat=ali,
                                        lexicons=lexicons, 
                                        aliHmm=os.path.join(args.expDir,"train_lda_mllt","final.mdl"), 
                                        feat=ldaFeat,
                                        spk2utt=os.path.join(args.expDir,"data","train","spk2utt"),
                                        outFile=os.path.join(args.expDir,"train_sat","trans.ark"),
                                    )
    print("Transform feature")
    fmllrFeat = exkaldi.use_fmllr(
                        ldaFeat,
                        fmllrTransMat,
                        utt2spk=os.path.join("exp","data","train","utt2spk"),
                        outFile=os.path.join(args.expDir,"train_sat","fmllr_feat.ark"),
                    )

    # -------------- Build the decision tree ------------------------
    print("Start build a tree")
    tree = exkaldi.hmm.DecisionTree(lexicons=lexicons, contextWidth=3, centralPosition=1)
    tree.train(
                feat=fmllrFeat, 
                hmm=os.path.join(args.expDir,"train_lda_mllt","final.mdl"), 
                alignment=ali,
                topoFile=os.path.join(args.expDir,"dict","topo"), 
                numLeaves=args.numleaves,
                tempDir=os.path.join(args.expDir,"train_sat"), 
            )
    tree.save(os.path.join(args.expDir,"train_sat","tree"))
    print(f"Build tree done.")
    del fmllrFeat

    # ------------- Start training ----------------------
    # 1. Initialize a monophone HMM object
    print("Initialize a triphone HMM object")
    model = exkaldi.hmm.TriphoneHMM(lexicons=lexicons)
    model.initialize(
                tree=tree, 
                topoFile=os.path.join(args.expDir,"dict","topo"),
                treeStatsFile=os.path.join(args.expDir,"train_sat","treeStats.acc"), 
            )    
    print(f"Initialized a monophone HMM-GMM model: {model.info}.")

    # 2. convert the previous alignment
    print(f"Transform the alignment")
    newAli = exkaldi.hmm.convert_alignment(
                                    alignment=ali,
                                    originHmm=os.path.join(args.expDir,"train_lda_mllt","final.mdl"), 
                                    targetHmm=model, 
                                    tree=tree,
                                    outFile=os.path.join(args.expDir,"train_sat","initial.ali"),
                                )

    # 2. Split data for parallel training
    transcription = exkaldi.load_transcription(os.path.join(args.expDir,"data","train","text"))
    transcription = transcription.sort()
    
    if args.parallel > 1:
        # split feature
        ldaFeat = ldaFeat.sort(by="utt").subset(chunks=args.parallel)
        # split transcription depending on utterance IDs of each feat
        tempTrans = []
        tempAli = []
        tempFmllrMat = []
        for f in ldaFeat:
            tempTrans.append( transcription.subset(uttIDs=f.utts) )
            tempAli.append( newAli.subset(uttIDs=f.utts) )
            spks = exkaldi.utt_to_spk(f.utts, utt2spk=os.path.join(args.expDir,"data","train","utt2spk"))
            tempFmllrMat.append( fmllrTransMat.subset(uttIDs=spks) )
        transcription = tempTrans
        newAli = tempAli
        fmllrTransMat = tempFmllrMat

    # 3. Train
    print("Train the triphone model")
    model.train(
                ldaFeat,
                transcription, 
                os.path.join(args.expDir,"dict","L.fst"), 
                tree,
                tempDir=os.path.join(args.expDir,"train_sat"),
                initialAli=newAli,
                fmllrTransMat=fmllrTransMat,
                spk2utt=os.path.join(args.expDir,"data","train","spk2utt"), 
                utt2spk=os.path.join(args.expDir,"data","train","utt2spk"),                
				numIters=args.numIters, 
                maxIterInc=args.maxIterInc,
                totgauss=args.totgauss,
                realignIter=args.realignIter,
                fmllrIter=args.fmllrIter,
                boostSilence=1.0,
                power=0.2,
                fmllrSilWt=0.0,       
            )
    print(model.info)
    del ldaFeat
    del fmllrTransMat
    del newAli

    # ------------- Compile WFST training ----------------------
    # Make a WFST decoding graph
    make_WFST_graph(
                outDir=os.path.join(args.expDir,"train_sat","graph"),
                hmm=model,
                tree=tree,
            )
    # Decode test data
    GMM_decode_fmllr_and_score(
                outDir=os.path.join(args.expDir,"train_sat","decode"), 
                hmm=model,
                HCLGfile=os.path.join(args.expDir,"train_sat","graph",f"HCLG.{args.order}.fst"),
                tansformMatFile=os.path.join(args.expDir,"train_lda_mllt","trans.mat"),
            )

if __name__ == "__main__":
    main()