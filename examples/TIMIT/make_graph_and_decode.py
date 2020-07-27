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

Compile HCLG graph, decode and score.

'''
import os
import time

import exkaldi
from exkaldi import args

def make_WFST_graph(outDir, hmm, tree):

    print("Start to make WFST graph.")
    exkaldi.utils.make_dependent_dirs(outDir, pathIsFile=False)

    lexicons = exkaldi.load_lex(os.path.join(args.expDir,"dict","lexicons.lex"))
    print(f"Load lexicon bank.")
    # iLabel file will be generated in this step.
    _, ilabelFile = exkaldi.decode.graph.compose_CLG(
                                            lexicons,
                                            tree,
                                            os.path.join(args.expDir,"lm",f"LG.{args.order}.fst"),
                                            outFile=os.path.join(outDir,f"CLG.{args.order}.fst"),
                                        )
    print(f"Generate CLG fst done.")
    exkaldi.decode.graph.compose_HCLG(
                                    hmm,
                                    tree,
                                    CLGfile=os.path.join(outDir,f"CLG.{args.order}.fst"),
                                    iLabelFile=ilabelFile,
                                    outFile=os.path.join(outDir,f"HCLG.{args.order}.fst"),
                                )
    print(f"Compose HCLG fst done.")

def GMM_decode_mfcc_and_score(outDir, hmm, HCLGfile, tansformMatFile=None):

    exkaldi.utils.make_dependent_dirs(outDir, pathIsFile=False)

    lexicons = exkaldi.load_lex(os.path.join(args.expDir,"dict","lexicons.lex"))
    print(f"Load test feature.")
    featFile = os.path.join(args.expDir,"mfcc","test","mfcc_cmvn.ark")
    feat = exkaldi.load_feat(featFile)
    if tansformMatFile is None:
        print("Feature type is delta")
        feat = feat.add_delta(order=args.delta)
        print(f"Add {args.delta}-order deltas.")
    else:
        print("Feature type is lda+mllt")
        feat = feat.splice(left=args.splice,right=args.splice)
        feat = exkaldi.transform_feat(feat, tansformMatFile)
        print("Transform feature")
    
    if args.parallel > 1:
        feat = feat.subset(chunks=args.parallel)

    print("Start to decode")
    st = time.time()
    lat = exkaldi.decode.wfst.gmm_decode(
                                    feat, hmm, 
                                    HCLGfile, 
                                    symbolTable=lexicons("words"),
                                    beam=args.beam, 
                                    latBeam=args.latBeam, 
                                    acwt=args.acwt,
                                    outFile=os.path.join(outDir,"test.lat"),
                                )
    print("Decode time cost: ",time.time()-st,"seconds")
    if isinstance(lat,list):
        lat = exkaldi.merge_archieves(lat)

    print(f"Generate lattice done.")
    # Restorage 48-39  
    phoneMapFile = os.path.join("exp","dict","phones.48_to_39.map")
    phoneMap = exkaldi.load_list_table(phoneMapFile,name="48-39")
    refText = exkaldi.load_transcription(os.path.join(args.expDir,"data","test","text"))
    refText = refText.convert(phoneMap, None)
    refText.save(os.path.join(outDir,"ref.txt") ) # Take a backup
    print("Generate reference text done.")

    print("Now score:")
    bestWER = (1000, 0, 0)
    bestResult = None
    for penalty in [0., 0.5, 1.0]:
        for LMWT in range(1, 11):
            # Add penalty
            newLat = lat.add_penalty(penalty)
            # Get 1-best result (word-level)
            result = newLat.get_1best(lexicons("words"), hmm, lmwt=LMWT, acwt=1)
            # Transform from int value format to text format
            result = exkaldi.hmm.transcription_from_int(result, lexicons("words"))
            # Transform 48-phones to 39-phones
            result = result.convert(phoneMap, None)
            # Compute WER
            score = exkaldi.decode.score.wer(ref=refText, hyp=result, mode="present")
            if score.WER < bestWER[0]:
                bestResult = result
                bestWER = (score.WER, penalty, LMWT)
            print(f"Penalty: {penalty}, LMWT: {LMWT}, WER: {score.WER}%")
    print("Score done. Save the best result.")
    bestResult.save(os.path.join(outDir, "hyp.txt") )
    with open(os.path.join(outDir,"best_WER"),"w") as fw:
        fw.write( f"WER {bestWER[0]}, penalty {bestWER[1]}, LMWT {bestWER[2]}" )

def GMM_decode_fmllr_and_score(outDir, hmm, HCLGfile, tansformMatFile=None):

    exkaldi.utils.make_dependent_dirs(outDir, pathIsFile=False)

    lexicons = exkaldi.load_lex(os.path.join(args.expDir,"dict","lexicons.lex"))
    print(f"Load test feature.")
    featFile = os.path.join(args.expDir,"mfcc","test","mfcc_cmvn.ark")
    feat = exkaldi.load_feat(featFile)
    if tansformMatFile is None:
        print("Feature type is delta. Add 2-order deltas.")
        feat = feat.add_delta(order=args.delta)
        feat = feat.save(os.path.join(outDir,"test_mfcc_cmvn_delta.ark"),returnIndexTable=True)
    else:
        print("Feature type is lda+mllt")
        feat = feat.splice(left=args.splice,right=args.splice)
        print("Transform feature")
        feat = exkaldi.transform_feat(feat, tansformMatFile)
        feat = feat.save(os.path.join(outDir,"test_mfcc_cmvn_lda.ark"),returnIndexTable=True)

    ## 1. Estimate the primary transform matrix from alignment or lattice.
    ## We estimate it from lattice, so we decode it firstly.
    print("Decode the first time with original feature.")
    preLat = exkaldi.decode.wfst.gmm_decode(
                                        feat, 
                                        hmm, 
                                        HCLGfile, 
                                        symbolTable=lexicons("words"),
                                        beam=10, 
                                        latBeam=6, 
                                        acwt=args.acwt,
                                        maxActive=2000,
                                    )
    preLat.save(os.path.join(outDir,"test_premary.lat"))

    print("Estimate the primary fMLLR transform matrix.")
    preTransMatrix = exkaldi.hmm.estimate_fMLLR_matrix(
                                        aliOrLat=preLat,
                                        lexicons=lexicons,
                                        aliHmm=hmm, 
                                        feat=feat,
                                        adaHmm=None,
                                        silenceWeight=0.01,
                                        acwt=args.acwt,
                                        spk2utt=os.path.join(args.expDir,"data","test","spk2utt"),
                                    )
    del preLat
    ## 2. Transform feature. We will use new feature to estimate the secondary transform matrix from lattice.
    print("Transform feature with primary matrix.")
    fmllrFeat = exkaldi.use_fmllr(
                        feat,
                        preTransMatrix,
                        utt2spk=os.path.join(args.expDir,"data","test","utt2spk"),
                    )
    print("Decode the second time with primary fmllr feature.")
    secLat = exkaldi.decode.wfst.gmm_decode(
                                        fmllrFeat, 
                                        hmm, 
                                        HCLGfile, 
                                        symbolTable=lexicons("words"),
                                        beam=args.beam,
                                        latBeam=args.latBeam,
                                        acwt=args.acwt,
                                        maxActive=7000,
                                        config={"--determinize-lattice":"false"},
                                    )
    print("Determinize secondary lattice.")
    thiLat = secLat.determinize(acwt=args.acwt, beam=4)
    print("Estimate the secondary fMLLR transform matrix.")
    secTransMatrix = exkaldi.hmm.estimate_fMLLR_matrix(
                                        aliOrLat=thiLat,
                                        lexicons=lexicons,
                                        aliHmm=hmm, 
                                        feat=fmllrFeat,
                                        adaHmm=None,
                                        silenceWeight=0.01,
                                        acwt=args.acwt,
                                        spk2utt=os.path.join(args.expDir,"data","test","spk2utt"),
                                    )
    del fmllrFeat
    del thiLat
    ## 3. Compose the primary matrix and secondary matrix and get the final transform matrix.
    print("Compose the primary and secondary transform matrix.")
    finalTransMatrix = exkaldi.hmm.compose_transform_matrixs(
                                                matA=preTransMatrix,
                                                matB=secTransMatrix,
                                                bIsAffine=True,
                                            )
    finalTransMatrix.save(os.path.join(outDir,"trans.ark"))
    print("Transform feature with final matrix.")
    ## 4. Transform feature with the final transform matrix and use it to decode.
    ## We directly use the lattice generated in the second step. The final lattice is obtained.
    finalFmllrFeat = exkaldi.use_fmllr(
                        feat,
                        finalTransMatrix,
                        utt2spk=os.path.join(args.expDir,"data","test","utt2spk"),
                    )
    del finalTransMatrix
    print("Rescore secondary lattice.")
    lat = secLat.am_rescore(
                        hmm=hmm,
                        feat=finalFmllrFeat,
                    )
    print("Determinize secondary lattice.")
    lat = lat.determinize(acwt=args.acwt, beam=6)
    lat.save(os.path.join(outDir,"test.lat"))
    print("Generate lattice done.")

    phoneMapFile = os.path.join(args.expDir,"dict","phones.48_to_39.map")
    phoneMap = exkaldi.load_list_table(phoneMapFile,name="48-39")
    refText = exkaldi.load_transcription(os.path.join(args.expDir,"data","test","text")).convert(phoneMap, None)
    refText.save(os.path.join(outDir,"ref.txt") )
    print("Generate reference text done.")

    print("Now score:")
    bestWER = (1000, 0, 0)
    bestResult = None
    for penalty in [0., 0.5, 1.0]:
        for LMWT in range(1, 11):
            # Add penalty
            newLat = lat.add_penalty(penalty)
            # Get 1-best result (word-level)
            result = newLat.get_1best(lexicons("words"), hmm, lmwt=LMWT, acwt=1)
            # Transform from int value format to text format
            result = exkaldi.hmm.transcription_from_int(result, lexicons("words"))
            # Transform 48-phones to 39-phones
            result = result.convert(phoneMap, None)
            # Compute WER
            score = exkaldi.decode.score.wer(ref=refText, hyp=result, mode="present")
            if score.WER < bestWER[0]:
                bestResult = result
                bestWER = (score.WER, penalty, LMWT)
            print(f"Penalty: {penalty}, LMWT: {LMWT}, WER: {score.WER}%")
    print("Score done. Save the best result.")
    bestResult.save(os.path.join(outDir, "hyp.txt") )
    with open(os.path.join(outDir,"best_WER"),"w") as fw:
        fw.write( f"WER {bestWER[0]}, penalty {bestWER[1]}, LMWT {bestWER[2]}" )