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
'''

import os
import glob
import subprocess
import gc
import exkaldi
from exkaldi.version import version as ExkaldiInfo

timitRoot = "/Corpus/TIMIT"   # TIMIT data root path
lmOrder = 3

def prepare_data():

    dataOutDir = os.path.join("exp","data")
    exkaldi.utils.make_dependent_dirs(dataOutDir,pathIsFile=False)

    # Prepare tools
    ExkaldiInfo.vertify_kaldi_existed()
    sph2pipeTool = os.path.join(ExkaldiInfo.KALDI_ROOT,"tools","sph2pipe_v2.5","sph2pipe")
    if not os.path.join(sph2pipeTool):
        raise Exception(f"Expected sph2pipe tool existed.")

    # Check TIMIT data format
    if not os.path.isdir(timitRoot):
        raise Exception(f"No such directory: {timitRoot}.")
    dirNames = os.listdir(timitRoot)
    if "TRAIN" in dirNames and "TEST" in dirNames:
        uppercaseFlag = True
        trainResourceDir = "TRAIN"
        testResourceDir = "TEST"
        testWavFile = os.path.join(timitRoot,"TRAIN","DR1","FCJF0","SA1.WAV")
        wavFileSuffix = "WAV"
        txtFileSuffix = "PHN"
    elif "train" in dirNames and "test" in dirNames:
        uppercaseFlag = False
        trainResourceDir = "train"
        testResourceDir = "test"
        testWavFile = os.path.join(timitRoot,"train","dr1","fcjf0","sa1.wav")
        wavFileSuffix = "wav"
        txtFileSuffix = "phn"
    else:
        raise Exception(f"Wrong format of train or test data directories.")
    formatCheckCmd = f"{sph2pipeTool}  -f wav {testWavFile}"
    out, err, cod = exkaldi.utils.run_shell_command(formatCheckCmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if cod == 0:
        sphFlag = True
    else:
        sphFlag = False

    # Transform phones from 60 categories to 48 catagories and generate the 48 to 39 transform dictionary
    phoneMap_60_to_48 = exkaldi.ListTable(name="69-48")
    phoneMap_48_to_39 = exkaldi.ListTable(name="48-39")
    with open(os.path.join(ExkaldiInfo.KALDI_ROOT,"egs","timit","s5","conf","phones.60-48-39.map"),"r",encoding="utf-8") as fr:
        lines = fr.readlines()
        for line in lines:
            line = line.strip().split()
            if len(line) < 3:
                continue
            phoneMap_60_to_48[line[0]] = line[1]
            phoneMap_48_to_39[line[1]] = line[2]
    phoneMap_48_to_39.save(
                        os.path.join("exp","dict","phones.48_to_39.map")
                    )

    # Design a a function to generate wav.scp, spk2utt, utt2spk, text files.
    def generate_data(wavFiles, outDir):
        wavScp = exkaldi.ListTable(name="wavScp")
        utt2spk = exkaldi.ListTable(name="utt2spk")
        spk2utt = exkaldi.ListTable(name="spk2utt")
        transcription = exkaldi.ListTable(name="trans")
        for Name in wavFiles:
            if Name[-7:].upper() in ["SA1.WAV","SA2.WAV","sa1.wav","sa2.wav"]:
                continue
            speaker = os.path.basename( os.path.dirname(Name) )
            uttID = speaker + "_" + os.path.basename(Name)[0:-4]
            wavFilePath = os.path.abspath(Name)
            # wav.scp
            if sphFlag:
                wavScp[uttID] = f"{sph2pipeTool} -f wav {wavFilePath} |"
            else:
                wavScp[uttID] = wavFilePath
            # utt2spk
            utt2spk[uttID] = speaker
            # spk2utt
            if speaker not in spk2utt.keys():
                spk2utt[speaker] = f"{uttID}"
            else:
                spk2utt[speaker] += f" {uttID}"
            # transcription
            txtFile = Name[:-3] + txtFileSuffix
            phones = []
            with open(txtFile, "r", encoding="utf-8") as fr:
                lines = fr.readlines()
            for line in lines:
                line = line.strip()
                if len(line) == 0:
                    continue
                phone = line.split()[-1]
                if phone == "q":
                    continue
                else:
                    phone = phoneMap_60_to_48[phone]
                phones.append(phone)
            transcription[uttID] = " ".join(phones)
        # Save to files
        wavScp.save( os.path.join(outDir, "wav.scp") )
        utt2spk.save( os.path.join(outDir, "utt2spk") )
        spk2utt.save( os.path.join(outDir, "spk2utt") )
        transcription.save( os.path.join(outDir, "text") )
        print(f"Generate data done: {outDir}.")

    # generate train data
    wavFiles = glob.glob(os.path.join(timitRoot,trainResourceDir,"*","*",f"*.{wavFileSuffix}"))
    generate_data(
                wavFiles = wavFiles, 
                outDir = os.path.join(dataOutDir,"train"),
            )

    # generate dev and test data.
    for Name in ["dev", "test"]:
        spkListFile = os.path.join(ExkaldiInfo.KALDI_ROOT,"egs","timit","s5","conf",f"{Name}_spk.list")
        with open(spkListFile,"r",encoding="utf-8") as fr:
            spkList = fr.readlines()
        wavFiles = []
        for spk in spkList:
            spk = spk.strip()
            if len(spk) == 0:
                continue
            if uppercaseFlag:
                spk = spk.upper()
            wavFiles.extend(glob.glob(os.path.join(timitRoot,testResourceDir,"*",spk,f"*.{wavFileSuffix}")))
        generate_data(
                    wavFiles = wavFiles, 
                    outDir = os.path.join(dataOutDir,Name),
                )

def prepare_dict_and_LM():
    '''
    Dictionary
    '''
    dictOutDir = os.path.join("exp","dict")
    exkaldi.utils.make_dependent_dirs(dictOutDir, pathIsFile=False)

    # Make the word-pronumciation lexicon and save it.
    allWords = []
    with open(os.path.join("exp","data","train","text"),"r",encoding="UTF-8") as fr:
        lines = fr.readlines()
        for line in lines:
            line = line.strip().split()
            if len(line) < 2:
                continue
            else:
                allWords.extend( line[1:] )
    allWords = sorted(list(set(allWords)))
    allWord2Pron = list( map(lambda x: f"{x} {x}", allWords) )

    pronFile = os.path.join(dictOutDir,"pronunciation.txt")
    with open(pronFile,"w") as fw:
        fw.write("\n".join(allWord2Pron))

    # Generate the LexiconBank object from word-pronumciation lexicon.
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

    # Add two extra questions
    lexicons.add_extra_question(lexicons("silence_phones"))
    lexicons.add_extra_question(lexicons("nonsilence_phones"))

    # Save this lexicon bank
    lexicons.save(os.path.join(dictOutDir,"lexicons.lex"))
    print(f"Generate lexicon bank done.")

    # Generate the lexicon fst
    exkaldi.decode.graph.make_L(
                            lexicons, 
                            outFile=os.path.join(dictOutDir,"L.fst"), 
                            useSilprob=0.0, 
                            useDisambigLexicon=False
                        )
    print(f"Generate lexicon fst done.")
    exkaldi.decode.graph.make_L(
                            lexicons, 
                            outFile=os.path.join(dictOutDir,"L_disambig.fst"), 
                            useSilprob=0.0, 
                            useDisambigLexicon=True
                        )
    print(f"Generate disambiguation lexicon fst done.")

    # Now, generate a topology file
    exkaldi.hmm.make_toponology(
                            lexicons,
                            outFile=os.path.join(dictOutDir,"topo"),
                            numNonsilStates=3,
                            numSilStates=3, #5
                        )
    print(f"Generate topo file done.")

    '''
    Language model
    '''
    lmOutDir = os.path.join("exp","lm")
    exkaldi.utils.make_dependent_dirs(lmOutDir, pathIsFile=False)

    # Make the text without utt-ID in order to train the ARPA language model
    with open(os.path.join("exp","data","train","text"),"r",encoding="utf-8") as fr:
        lines = fr.readlines()
    newLines = []
    for line in lines:
        newLines.append(line.split(maxsplit=1)[1])
    with open(os.path.join(lmOutDir,"train_lm_text"),"w") as fw:
        fw.writelines(newLines)

    # We have trained 2,3,4 grams model with both srilm and kenlm and chose the best one, which is 3-grams model back kenlm.
    # So we directly train this one.
    exkaldi.lm.train_ngrams_kenlm(
                            lexicons, 
                            order=lmOrder, 
                            textFile=os.path.join(lmOutDir,"train_lm_text"), 
                            outFile=os.path.join(lmOutDir,f"{lmOrder}grams.arpa"), 
                            config={"--discount_fallback":True,"-S":"20%"},
                        )
    print(f"Generate ARPA language model done.")

    ## Then test this model by compute the perplexity
    # 1, make a Kenlm model object.
    exkaldi.lm.arpa_to_binary(
                            arpaFile=os.path.join(lmOutDir,f"{lmOrder}grams.arpa"),
                            outFile=os.path.join(lmOutDir,f"{lmOrder}grams.binary"),
                        )
    model = exkaldi.lm.KenNGrams(os.path.join(lmOutDir,f"{lmOrder}grams.binary"))
    # 2, prepare test transcription
    testTrans = exkaldi.load_trans(
                            os.path.join("exp","data","test","text"),
                        )
    # 3, score
    perScore = model.perplexity(testTrans)
    meanScore = perScore.mean(testTrans.sentence_length())
    print(f"The weighted average perplexity of this model is: {meanScore}.")
    del model
    del testTrans

    # Make Grammar fst
    exkaldi.decode.graph.make_G(
                            lexicons, 
                            arpaFile=os.path.join(lmOutDir,f"{lmOrder}grams.arpa"),
                            outFile=os.path.join(lmOutDir,f"G.{lmOrder}.fst"), 
                            order=lmOrder
                        )
    print(f"Make Grammar fst done.")

    # Compose LG fst
    exkaldi.decode.graph.compose_LG(
                            Lfile=os.path.join(dictOutDir,"L_disambig.fst"), 
                            Gfile=os.path.join(lmOutDir,f"G.{lmOrder}.fst"),
                            outFile=os.path.join(lmOutDir,f"LG.{lmOrder}.fst"),
                        )
    print(f"Compose LG fst done.")

def compute_mfcc():

    featOutDir = os.path.join("exp","mfcc")
    exkaldi.utils.make_dependent_dirs(featOutDir,pathIsFile=False)

    for Name in ["train", "dev", "test"]:
        print(f"Compute {Name} MFCC feature.")
        exkaldi.utils.make_dependent_dirs(os.path.join(featOutDir,Name),pathIsFile=False)

        # Compute feature
        feat = exkaldi.compute_mfcc(
                                wavFile=os.path.join("exp","data",Name,"wav.scp"), 
                                config={"--use-energy":"false"},
                            )
        feat.save( os.path.join(featOutDir,Name,"raw_mfcc.ark") )
        print(f"Generate raw MFCC feature done.")
        # Compute CMVN
        cmvn = exkaldi.compute_cmvn_stats(
                                        feat=feat,
                                        spk2utt=os.path.join("exp","data",Name,"spk2utt"),
                                    )
        cmvn.save( os.path.join(featOutDir,Name,"cmvn.ark") )
        print(f"Generate CMVN statistics done.")
        # Apply CMVN
        feat = exkaldi.use_cmvn(
                            feat=feat,
                            cmvn=cmvn,
                            utt2spk=os.path.join("exp","data",Name,"utt2spk"),
                        )
        feat.save(os.path.join(featOutDir,Name,"mfcc_cmvn.ark"))
        print(f"Generate MFCC feature (applied CMVN) done.")
    print("Compute MFCC done.")

def train_mono(decode=True):

    hmmOutDir = os.path.join("exp","train_mono")
    exkaldi.utils.make_dependent_dirs(hmmOutDir,pathIsFile=False)
    '''
    # Load the feature for training
    feat = exkaldi.load_feat(os.path.join("exp","mfcc","train","mfcc_cmvn.ark"))
    print(f"Load MFCC+CMVN feature.")
    feat = feat.add_delta(order=2)
    print("Add 2-order deltas.")
    # Load lexicon bank
    lexicons = exkaldi.decode.graph.load_lex(os.path.join("exp","dict","lexicons.lex"))
    print(f"Restorage lexicon bank.")
    # Initialize a monophone HMM object
    
    model = exkaldi.hmm.MonophoneHMM(lexicons=lexicons, name="mono")
    model.initialize(
                feat=feat,
                topoFile=os.path.join("exp","dict","topo")
            ) 
    
    print(f"Initialized a monophone HMM-GMM model: {model.info}.")
    # Start training
    model.train(
                feat,
                os.path.join("exp","data","train","text"), 
                os.path.join("exp","dict","L.fst"),
                tempDir=hmmOutDir,
                num_iters=40, 
                max_iter_inc=30, 
                totgauss=1000, 
                realign_iter=[1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,23,26,29,32,35,38],
                boost_silence=1.0,
            )
    print(model.info)
    # Save the tree
    model.tree.save(os.path.join(hmmOutDir,"tree"))
    print(f"Tree has been saved.")
    
    print("Realign the training feature (boost silence = 1.25)")
    #os.remove(os.path.join(hmmOutDir,"final.ali"))
    ali = model.align(
                feat,
                os.path.join(hmmOutDir,"train_graph"), 
                boost_silence=1.25, #1.5
            )
    ali.save(os.path.join(hmmOutDir,"final.ali"))
    del ali
    del feat
    print("Save the new alignment done.")
    '''
    model = exkaldi.hmm.load_hmm(os.path.join(hmmOutDir,"final.mdl"))
    tree = exkaldi.hmm.load_tree(os.path.join(hmmOutDir,"tree"))
    if decode:
        # Make a WFST decoding graph
        make_WFST_graph(
                    outDir=os.path.join(hmmOutDir,"graph"),
                    hmm=model,
                    tree=tree, #model.tree,
                )
        # Decode test data
        GMM_decode_mfcc_and_score(
                    outDir=os.path.join(hmmOutDir,"decode"), 
                    hmm=model,
                    HCLGfile=os.path.join(hmmOutDir,"graph",f"HCLG.{lmOrder}.fst"),
                )
    
    del model

def make_WFST_graph(outDir, hmm, tree):

    print("Start to make WFST graph.")
    exkaldi.utils.make_dependent_dirs(outDir,pathIsFile=False)

    lexicons = exkaldi.decode.graph.load_lex(os.path.join("exp","dict","lexicons.lex"))
    print(f"Load lexicon bank.")
    # iLabel file will be generated in this step.
    _, ilabelFile = exkaldi.decode.graph.compose_CLG(
                                            lexicons,
                                            tree,
                                            os.path.join("exp","lm",f"LG.{lmOrder}.fst"),
                                            outFile=os.path.join(outDir,f"CLG.{lmOrder}.fst"),
                                        )
    print(f"Generate CLG fst done.")
    exkaldi.decode.graph.compose_HCLG(
                                    hmm,
                                    tree,
                                    CLGfile=os.path.join(outDir,f"CLG.{lmOrder}.fst"),
                                    iLabelFile=ilabelFile,
                                    outFile=os.path.join(outDir,f"HCLG.{lmOrder}.fst"),
                                )
    print(f"Compose HCLG fst done.")

def GMM_decode_mfcc_and_score(outDir, hmm, HCLGfile, tansformMatFile=None):

    exkaldi.utils.make_dependent_dirs(outDir, pathIsFile=False)

    lexicons = exkaldi.decode.graph.load_lex(os.path.join("exp","dict","lexicons.lex"))
    print(f"Load test feature.")
    featFile = os.path.join("exp","mfcc","test","mfcc_cmvn.ark")
    feat = exkaldi.load_feat(featFile)
    if tansformMatFile is None:
        print("Feature type is delta")
        feat = feat.add_delta(order=2)
        print("Add 2-order deltas.")
    else:
        print("Feature type is lda+mllt")
        feat = feat.splice(left=3,right=3)
        feat = exkaldi.transform_feat(feat, tansformMatFile)
        print("Transform feature")

    print("Start to decode")
    lat = exkaldi.decode.wfst.gmm_decode(
                                        feat, hmm, 
                                        HCLGfile, 
                                        wordSymbolTable=lexicons("words"),
                                        beam=13, 
                                        latBeam=6, 
                                        acwt=0.083333
                                    )
    lat.save(os.path.join(outDir,"test.lat"))
    print(f"Generate lattice done.")

    phoneMapFile = os.path.join("exp","dict","phones.48_to_39.map")
    phoneMap = exkaldi.ListTable(name="48-39").load(phoneMapFile)
    refText = exkaldi.load_trans(os.path.join("exp","data","test","text")).convert(phoneMap, None)
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

def train_delta(decode=True):

    hmmOutDir = os.path.join("exp","train_delta")
    exkaldi.utils.make_dependent_dirs(hmmOutDir,pathIsFile=False)
    '''
    # Restorage lexicon bank
    lexicons = exkaldi.decode.graph.load_lex(os.path.join("exp","dict","lexicons.lex"))
    print(f"Restorage lexicon bank from file.")
    # Load feature
    feat = exkaldi.load_feat(os.path.join("exp","mfcc","train","mfcc_cmvn.ark"))
    print(f"Load train feature.")
    feat = feat.add_delta(order=2)
    print("Add 2-order deltas.")
    # Build tree
    print("Start build a tree")
    tree = exkaldi.hmm.DecisionTree(lexicons=lexicons,contextWidth=3,centralPosition=1)
    tree.train(
                feat=feat, 
                hmm=os.path.join("exp","train_mono","final.mdl"), 
                alignment=os.path.join("exp","train_mono","final.ali"), 
                topoFile=os.path.join("exp","dict","topo"), 
                numleaves=2500,
                tempDir=hmmOutDir
            )
    tree.save(os.path.join(hmmOutDir,"tree"))
    print(f"Build tree done.")
    # Initialize context HMM-GMM model
    print("Initialize a triphone HMM object")
    model = exkaldi.hmm.TriphoneHMM(lexicons=lexicons)
    model.initialize(
                tree=tree, 
                topoFile=os.path.join("exp","dict","topo"),
                treeStatsFile=os.path.join(hmmOutDir,"treeStats.acc"), 
            )
    # Transform the alignment
    print(f"Transform the alignment")
    newAli = exkaldi.hmm.convert_alignment(
                                    alignment=os.path.join("exp","train_mono","final.ali"),
                                    originHmm=os.path.join("exp","train_mono","final.mdl"), 
                                    targetHmm=model, 
                                    tree=tree,
                                )
    # Start training
    print("Train the triphone model")
    model.train(feat, 
                os.path.join("exp","data","train","text"), 
                os.path.join("exp","dict","L.fst"), 
                tree,
                tempDir=hmmOutDir,
                initialAli=newAli,
				num_iters=35, 
                max_iter_inc=25,
                totgauss=15000,
                realign_iter=[10,20,30],
                boost_silence=1.0,
            )
    print(model.info)
    del feat
    del newAli
    del lexicons
    '''
    model = exkaldi.hmm.load_hmm(os.path.join(hmmOutDir,"final.mdl"))
    tree = exkaldi.hmm.load_tree(os.path.join(hmmOutDir,"tree"))
    if decode:
        # Make a WFST decoding graph
        make_WFST_graph(
                    outDir=os.path.join(hmmOutDir,"graph"),
                    hmm=model,
                    tree=tree,
                )
        # Decode test data
        GMM_decode_mfcc_and_score(
                    outDir=os.path.join(hmmOutDir,"decode"),
                    hmm=model,
                    HCLGfile=os.path.join(hmmOutDir,"graph",f"HCLG.{lmOrder}.fst"),
                )
    
    del model

def train_lda_mllt(decode=True):

    hmmOutDir = os.path.join("exp","train_lda_mllt")
    exkaldi.utils.make_dependent_dirs(hmmOutDir,pathIsFile=False)
    '''
    # Restorage lexicon bank
    lexicons = exkaldi.decode.graph.load_lex(os.path.join("exp","dict","lexicons.lex"))
    print(f"Load lexicon bank done.")
    # Load train feature
    originalFeat = exkaldi.load_feat(os.path.join("exp","mfcc","train","mfcc_cmvn.ark"))
    print(f"Load training feature done.")
    originalFeat = originalFeat.splice(left=3,right=3)
    print("Splice left 3 and right 3 frames.")
    # Compute primary LDA transform
    print("Estimate LDA matrix.")
    exkaldi.hmm.accumulate_LDA_stats(
                    alignment=os.path.join("exp","train_delta","final.ali"), 
                    lexicons=lexicons, 
                    hmm=os.path.join("exp","train_delta","final.mdl"), 
                    feat=originalFeat, 
                    outFile=os.path.join(hmmOutDir,"ldaStats.acc"),
                    silenceWeight=0,
                    randPrune=4,
                )
    exkaldi.hmm.estimate_LDA_matrix(
                                LDAstatsFile=os.path.join(hmmOutDir,"ldaStats.acc"), 
                                targetDim=40, 
                                outFile=os.path.join(hmmOutDir,"trans.mat")
                            )
    print("Transform feature")
    ldaFeat = exkaldi.transform_feat(
                            feat=originalFeat, 
                            matrixFile=os.path.join(hmmOutDir,"trans.mat"),
                        )
    # Build tree
    print("Start build a tree")
    tree = exkaldi.hmm.DecisionTree(lexicons=lexicons,contextWidth=3,centralPosition=1)
    tree.train(feat=ldaFeat,
               hmm=os.path.join("exp","train_delta","final.mdl"), 
               alignment=os.path.join("exp","train_delta","final.ali"), 
               topoFile=os.path.join("exp","dict","topo"),
               numleaves=2500,
               tempDir=hmmOutDir
            )
    tree.save(os.path.join(hmmOutDir,"tree"))
    print(f"Build tree done.")
    del ldaFeat
    # Initialize a context HMM-GMM model
    print("Initialize a triphone HMM object")
    model = exkaldi.hmm.TriphoneHMM(lexicons=lexicons)
    model.initialize(
                tree=tree, 
                topoFile=os.path.join("exp","dict","topo"),
                treeStatsFile=os.path.join(hmmOutDir,"treeStats.acc"), 
            )
    # Transform feature
    print(f"Transform the alignment")
    newAli = exkaldi.hmm.convert_alignment(
                                    alignment=os.path.join("exp","train_delta","final.ali"),
                                    originHmm=os.path.join("exp","train_delta","final.mdl"),
                                    targetHmm=model, 
                                    tree=tree,
                                )
    # Start to train
    print("Train the triphone model")
    model.train(
                originalFeat,
                os.path.join("exp","data","train","text"), 
                os.path.join("exp","dict","L.fst"),
                tree, 
                tempDir=hmmOutDir, 
                initialAli=newAli, 
                ldaMatFile=os.path.join(hmmOutDir,"trans.mat"),
				num_iters=35,
                max_iter_inc=25,
                totgauss=15000,
                realign_iter=[10,20,30],
                mllt_iter=[2,4,6,12],
                boost_silence=1.0,
            )
    print(model.info)
    del originalFeat
    del newAli
    '''
    model = exkaldi.hmm.load_hmm(os.path.join(hmmOutDir,"final.mdl"))
    tree = exkaldi.hmm.load_tree(os.path.join(hmmOutDir,"tree"))
    if decode:
        # Make a WFST decoding graph
        make_WFST_graph(
                    outDir=os.path.join(hmmOutDir,"graph"),
                    hmm=model,
                    tree=tree,
                )
        # Decode test data
        GMM_decode_mfcc_and_score(
                    outDir=os.path.join(hmmOutDir,"decode"),
                    hmm=model,
                    HCLGfile=os.path.join(hmmOutDir,"graph",f"HCLG.{lmOrder}.fst"),
                    tansformMatFile=os.path.join(hmmOutDir,"trans.mat")
                )
    
    del model

def train_lda_mllt_sat(decode=True):

    hmmOutDir = os.path.join("exp","train_lda_mllt_sat")
    exkaldi.utils.make_dependent_dirs(hmmOutDir,pathIsFile=False)
    '''
    # Restorage lexicon bank
    lexicons = exkaldi.decode.graph.load_lex(os.path.join("exp","dict","lexicons.lex"))
    print(f"Restorage lexicon bank done.")
    # Load training feature
    originalFeat = exkaldi.load_feat(os.path.join("exp","mfcc","train","mfcc_cmvn.ark"))
    print(f"Load train feature done.")
    originalFeat = originalFeat.splice(left=3,right=3)
    print("Splice left 3 and right 3 frames.")
    # LDA transform
    print("Transform LDA feature")
    ldaFeat = exkaldi.transform_feat(
                            feat=originalFeat, 
                            matrixFile=os.path.join("exp","train_lda_mllt","trans.mat"),
                        )
    del originalFeat
    # Compute the primary fMLLR transform matrix
    print("Estiminate the primary fMLLR transform matrixs")
    preAlignment = exkaldi.load_ali(os.path.join("exp","train_lda_mllt","final.ali"))
    fmllrTransMat = exkaldi.hmm.estimate_fMLLR_matrix(
                                        aliOrLat=preAlignment,
                                        lexicons=lexicons, 
                                        aliHmm=os.path.join("exp","train_lda_mllt","final.mdl"), 
                                        feat=ldaFeat,
                                        spk2utt=os.path.join("exp","data","train","spk2utt"),
                                    )
    print("Transform feature")
    fmllrFeat = exkaldi.use_fmllr(
                        ldaFeat,
                        fmllrTransMat,
                        utt2spkFile=os.path.join("exp","data","train","utt2spk"),
                    )
    # Build a new tree
    print("Start build a tree")
    tree = exkaldi.hmm.DecisionTree(lexicons=lexicons,contextWidth=3,centralPosition=1)
    tree.train(
                feat=fmllrFeat,
                hmm=os.path.join("exp","train_lda_mllt","final.mdl"), 
                alignment=os.path.join("exp","train_lda_mllt","final.ali"), 
                topoFile=os.path.join("exp","dict","topo"), 
                numleaves=2500,
                tempDir=hmmOutDir
            )
    tree.save(os.path.join(hmmOutDir,"tree"))
    print(f"Build tree done.")
    del fmllrFeat
    # initialize a context HMM-GMM model
    print("Initialize a triphone HMM object")
    model = exkaldi.hmm.TriphoneHMM(lexicons=lexicons)
    model.initialize(
                tree=tree, 
                topoFile=os.path.join("exp","dict","topo"),
                treeStatsFile=os.path.join(hmmOutDir,"treeStats.acc"), 
            )
    # Transform alignment
    print(f"Transform the alignment")
    newAli = exkaldi.hmm.convert_alignment(
                                    alignment=os.path.join("exp","train_lda_mllt","final.ali"),
                                    originHmm=os.path.join("exp","train_lda_mllt","final.mdl"), 
                                    targetHmm=model, 
                                    tree=tree
                                )
    # Start training
    print("Train the triphone model")
    model.train(ldaFeat, 
                os.path.join("exp","data","train","text"), 
                os.path.join("exp","dict","L.fst"),
                tree,
                tempDir=hmmOutDir,
                initialAli=newAli, 
                fmllrTransMat=fmllrTransMat,
                spk2utt=os.path.join("exp","data","train","spk2utt"), 
                utt2spk=os.path.join("exp","data","train","utt2spk"),
				num_iters=35, 
                max_iter_inc=25,
                totgauss=15000,
                realign_iter=[10,20,30],
                fmllr_iter=[2,4,6,12],
                boost_silence=1.0,
                power=0.2,
                fmllrSilWt=0.0,
            )
    print(model.info)
    del ldaFeat
    del newAli
    del fmllrTransMat
    '''
    model = exkaldi.hmm.load_hmm(os.path.join(hmmOutDir,"final.mdl"))
    tree = exkaldi.hmm.load_tree(os.path.join(hmmOutDir,"tree")) 
    if decode:
        # Make a WFST decoding graph
        make_WFST_graph(
                    outDir=os.path.join(hmmOutDir,"graph"),
                    hmm=model,
                    tree=tree,
                )
        # Decode test data
        GMM_decode_fmllr_and_score(
                    outDir=os.path.join(hmmOutDir,"decode"),
                    hmm=model,
                    HCLGfile=os.path.join(hmmOutDir,"graph",f"HCLG.{lmOrder}.fst"),
                    tansformMatFile=os.path.join("exp","train_lda_mllt","trans.mat"),
                )
    
    del model

def GMM_decode_fmllr_and_score(outDir, hmm, HCLGfile, tansformMatFile=None):

    exkaldi.utils.make_dependent_dirs(outDir, pathIsFile=False)

    lexicons = exkaldi.decode.graph.load_lex(os.path.join("exp","dict","lexicons.lex"))
    print(f"Load test feature.")
    featFile = os.path.join("exp","mfcc","test","mfcc_cmvn.ark")
    feat = exkaldi.load_feat(featFile)
    if tansformMatFile is None:
        print("Feature type is delta")
        feat = feat.add_delta(order=2)
        print("Add 2-order deltas.")
    else:
        print("Feature type is lda+mllt")
        feat = feat.splice(left=3,right=3)
        feat = exkaldi.transform_feat(feat, tansformMatFile)
        print("Transform feature")

    ## 1. Estimate the primary transform matrix from alignment or lattice.
    ## We estimate it from lattice, so we decode it firstly.
    print("Decode the first time with original feature.")
    preLat = exkaldi.decode.wfst.gmm_decode(
                                        feat, 
                                        hmm, 
                                        HCLGfile, 
                                        wordSymbolTable=lexicons("words"),
                                        beam=10, 
                                        latBeam=6, 
                                        acwt=0.083333,
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
                                        acwt=0.083333,
                                        spk2utt=os.path.join("exp","data","test","spk2utt"),
                                    )
    del preLat
    ## 2. Transform feature. We will use new feature to estimate the secondary transform matrix from lattice.
    print("Transform feature with primary matrix.")
    fmllrFeat = exkaldi.use_fmllr(
                        feat,
                        preTransMatrix,
                        utt2spkFile=os.path.join("exp","data","test","utt2spk"),
                    )
    print("Decode the second time with primary fmllr feature.")
    secLat = exkaldi.decode.wfst.gmm_decode(
                                        fmllrFeat, 
                                        hmm, 
                                        HCLGfile, 
                                        wordSymbolTable=lexicons("words"),
                                        beam=13,
                                        latBeam=6, 
                                        acwt=0.083333,
                                        maxActive=7000,
                                        config={"--determinize-lattice":"false"},
                                    )
    print("Determinize secondary lattice.")
    thiLat = secLat.determinize(acwt=0.083333, beam=4)
    print("Estimate the secondary fMLLR transform matrix.")
    secTransMatrix = exkaldi.hmm.estimate_fMLLR_matrix(
                                        aliOrLat=thiLat,
                                        lexicons=lexicons,
                                        aliHmm=hmm, 
                                        feat=fmllrFeat,
                                        adaHmm=None,
                                        silenceWeight=0.01,
                                        acwt=0.083333,
                                        spk2utt=os.path.join("exp","data","test","spk2utt"),
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
                        utt2spkFile=os.path.join("exp","data","test","utt2spk"),
                    )
    del finalTransMatrix
    print("Rescore secondary lattice.")
    lat = secLat.am_rescore(
                        hmm=hmm,
                        feat=finalFmllrFeat,
                    )
    print("Determinize secondary lattice.")
    lat = lat.determinize(acwt=0.083333, beam=6)
    lat.save(os.path.join(outDir,"test.lat"))
    print("Generate lattice done.")

    phoneMapFile = os.path.join("exp","dict","phones.48_to_39.map")
    phoneMap = exkaldi.ListTable(name="48-39").load(phoneMapFile)
    refText = exkaldi.load_trans(os.path.join("exp","data","test","text")).convert(phoneMap, None)
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

if __name__ == "__main__":

    #=========== Prepare wav.scp, text, utt2spk, spk2utt files of train, dev and test respectively.
    prepare_data()
    gc.collect()
    
    #=========== Prepare various lexicons and train the N-grams language model.
    prepare_dict_and_LM()
    gc.collect()

    #=========== Compute MFCC feature and CMVN statistics
    compute_mfcc()
    gc.collect()

    #=========== Train a mono HMM-GMM model with MFCC + CMVN + 2-order-deltas feature, then decode and compute WER of test data
    train_mono(decode=True)
    gc.collect()
    
    #=========== Train triphone HMM-GMM with MFCC + CMVN + 2-order-deltas feature, then decode and compute WER of test data
    train_delta(decode=True)
    gc.collect()

    #=========== Train triphone HMM-GMM with MFCC + CMVN + splice-3-3 + LDA + MLLT feature, decode and compute WER of test data
    train_lda_mllt(decode=True)
    gc.collect()

    #=========== Train triphone HMM-GMM with MFCC + CMVN + splice-3-3 + LDA + MLLT + fMLLR feature, decode and compute WER of test data
    train_lda_mllt_sat(decode=True)
    gc.collect()




