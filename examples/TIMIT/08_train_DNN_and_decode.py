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
Timit HMM-DNN training recipe.

Part 5: train a DNN acoustic model with fmllr feature.

'''
import os
import time,datetime
import shutil,socket

import exkaldi
from exkaldi import args
from exkaldi import declare

import tensorflow as tf
from tensorflow import keras
import numpy as np
import random

def prepare_DNN_data():

  print("Start to prepare data for DNN training")
  assert os.path.isdir(f"{args.expDir}/train_sat"), "Please run previous programs up to SAT training."

  # Lexicons and Gmm-Hmm model
  lexicons = exkaldi.load_lex( f"{args.expDir}/dict/lexicons.lex" )
  hmm = f"{args.expDir}/train_sat/final.mdl"
  tree = f"{args.expDir}/train_sat/tree"

  for Name in ["train","dev","test"]:
    
    exkaldi.utils.make_dependent_dirs(f"{args.expDir}/train_dnn/data/{Name}", pathIsFile=False)
    # Make LDA feature
    print(f"Make LDA feature for '{Name}'")
    feat = exkaldi.load_feat( f"{args.expDir}/mfcc/{Name}/mfcc_cmvn.ark" )
    feat = feat.splice(left=args.LDAsplice, right=args.LDAsplice)
    feat = exkaldi.transform_feat(feat, matFile=f"{args.expDir}/train_lda_mllt/trans.mat" )
    # Compile the aligning graph
    print(f"Compile aligning graph")
    transInt = exkaldi.hmm.transcription_to_int(
                                              transcription=f"{args.expDir}/data/{Name}/text",
                                              symbolTable=lexicons("words"),
                                              unkSymbol=lexicons("oov"),
                                            )
    graphFile = exkaldi.decode.wfst.compile_align_graph(
                                        hmm,
                                        tree, 
                                        transcription=transInt,
                                        LFile= f"{args.expDir}/dict/L.fst",
                                        outFile=f"{args.expDir}/train_dnn/data/{Name}/align_graph",
                                        lexicons=lexicons,
                                    )
    # Align first time
    print(f"Align the first time")
    ali = exkaldi.decode.wfst.gmm_align(
                                    hmm,
                                    feat, 
                                    alignGraphFile=graphFile, 
                                    lexicons=lexicons,
                                )
    # Estimate transform matrix
    print(f"Estimate fMLLR transform matrix")
    fmllrTransMat = exkaldi.hmm.estimate_fMLLR_matrix(
                                aliOrLat=ali,
                                lexicons=lexicons, 
                                aliHmm=hmm, 
                                feat=feat,
                                spk2utt=f"{args.expDir}/data/{Name}/spk2utt",
                            )
    fmllrTransMat.save( f"{args.expDir}/train_dnn/data/{Name}/trans.ark" )
    # Transform feature
    print(f"Transform feature")
    feat = exkaldi.use_fmllr(
                        feat,
                        fmllrTransMat,
                        utt2spk=f"{args.expDir}/data/{Name}/utt2spk",
                    )
    # Align second time with new feature
    print(f"Align the second time")
    ali = exkaldi.decode.wfst.gmm_align(
                                    hmm,
                                    feat,
                                    alignGraphFile=graphFile, 
                                    lexicons=lexicons,
                                )
    # Save alignment and feature
    print(f"Save final fmllr feature and alignment")
    feat.save( f"{args.expDir}/train_dnn/data/{Name}/fmllr.ark" )
    ali.save( f"{args.expDir}/train_dnn/data/{Name}/ali" ) 
    # Transform alignment
    print(f"Generate pdf ID and phone ID alignment")
    ali.to_numpy(aliType="pdfID",hmm=hmm).save( f"{args.expDir}/train_dnn/data/{Name}/pdfID.npy" )
    ali.to_numpy(aliType="phoneID",hmm=hmm).save( f"{args.expDir}/train_dnn/data/{Name}/phoneID.npy" )
    del ali
    # Compute cmvn for fmllr feature
    print(f"Compute the CMVN for fmllr feature")
    cmvn = exkaldi.compute_cmvn_stats(feat, spk2utt=f"{args.expDir}/data/{Name}/spk2utt")
    cmvn.save( f"{args.expDir}/train_dnn/data/{Name}/cmvn_of_fmllr.ark" )
    del cmvn
    del feat
    # copy spk2utt utt2spk and text file
    shutil.copyfile( f"{args.expDir}/data/{Name}/spk2utt", f"{args.expDir}/train_dnn/data/{Name}/spk2utt")
    shutil.copyfile( f"{args.expDir}/data/{Name}/utt2spk", f"{args.expDir}/train_dnn/data/{Name}/utt2spk")
    shutil.copyfile( f"{args.expDir}/data/{Name}/text", f"{args.expDir}/train_dnn/data/{Name}/text" )
    transInt.save( f"{args.expDir}/data/{Name}/text.int" )

  print("Write feature and alignment dim information")
  dims = exkaldi.ListTable()
  feat = exkaldi.load_feat( f"{args.expDir}/train_dnn/data/test/fmllr.ark" ) 
  dims["fmllr"] = feat.dim
  del feat
  hmm = exkaldi.hmm.load_hmm( f"{args.expDir}/train_sat/final.mdl" )
  dims["phones"] = hmm.info.phones + 1
  dims["pdfs"] = hmm.info.pdfs
  del hmm
  dims.save( f"{args.expDir}/train_dnn/data/dims" )

def process_feat_ali(training=True):

    if training:
      Name = "train"
    else:
      Name = "dev"

    feat = exkaldi.load_feat( f"{args.expDir}/train_dnn/data/{Name}/fmllr.ark" )

    if args.useCMVN:
        cmvn = exkaldi.load_cmvn(f"{args.expDir}/train_dnn/data/{Name}/cmvn_of_fmllr.ark")
        feat = exkaldi.use_cmvn(feat,cmvn,f"{args.expDir}/train_dnn/data/{Name}/utt2spk")
        del cmvn
    
    if args.delta > 0:
        feat = feat.add_delta(args.delta)

    if args.splice > 0:
        feat = feat.splice(args.splice)

    feat = feat.to_numpy()

    if args.normalizeFeat:
        feat = feat.normalize(std=True)
    
    pdfAli = exkaldi.load_ali( f"{args.expDir}/train_dnn/data/{Name}/pdfID.npy" )
    phoneAli = exkaldi.load_ali( f"{args.expDir}/train_dnn/data/{Name}/phoneID.npy" )
    
    feat.rename("feat")
    pdfAli.rename("pdfID")
    phoneAli.rename("phoneID")

    dataset = exkaldi.tuple_dataset([feat, pdfAli, phoneAli], frameLevel=True)
    random.shuffle(dataset)

    return dataset

def make_generator(dataset):

    dataIndex = 0
    datasetSize = len(dataset)
    while True:
        if dataIndex >= datasetSize:
            random.shuffle(dataset)
            dataIndex = 0
        one = dataset[dataIndex]
        dataIndex += 1
        yield one.feat[0], {"pdfID":one.pdfID[0], "phoneID":one.phoneID[0]}

def prepare_test_data(postProbDim):

    feat = exkaldi.load_feat( f"{args.expDir}/train_dnn/data/test/fmllr.ark" )

    if args.useCMVN:
        cmvn = exkaldi.load_cmvn( f"{args.expDir}/train_dnn/data/test/cmvn_of_fmllr.ark" )
        feat = exkaldi.use_cmvn(feat, cmvn, utt2spk=f"{args.expDir}/train_dnn/data/test/utt2spk")
        del cmvn

    if args.delta > 0:
        feat = feat.add_delta(args.delta)

    if args.splice > 0:
        feat = feat.splice(args.splice)

    feat = feat.to_numpy()
    if args.normalizeFeat:
        feat = feat.normalize(std=True)

    # Normalize acoustic model output
    if args.normalizeAMP:
        ali = exkaldi.load_ali(f"{args.expDir}/train_dnn/data/train/pdfID.npy", aliType="pdfID")
        normalizeBias = exkaldi.nn.compute_postprob_norm(ali,postProbDim)
    else:
        normalizeBias = 0
    
    # ref transcription
    trans = exkaldi.load_transcription(f"{args.expDir}/train_dnn/data/test/text")
    convertTable = exkaldi.load_list_table(f"{args.expDir}/dict/phones.48_to_39.map")
    trans = trans.convert(convertTable)

    return feat, normalizeBias, trans

def make_DNN_model(inputDim,outDimPdf,outDimPho):
    
    inputs = keras.Input((inputDim,))

    ln1 = keras.layers.Dense(1024, activation=None, use_bias=False)(inputs)
    ln1_bn = keras.layers.BatchNormalization(momentum=0.95)(ln1)
    ln1_ac = keras.layers.ReLU()(ln1_bn)
    ln1_do = keras.layers.Dropout(args.dropout)(ln1_ac)

    ln2 = keras.layers.Dense(1024, activation=None, use_bias=False)(ln1_do)
    ln2_bn = keras.layers.BatchNormalization(momentum=0.95)(ln2)
    ln2_ac = keras.layers.ReLU()(ln2_bn)
    ln2_do = keras.layers.Dropout(args.dropout)(ln2_ac)

    ln3 = keras.layers.Dense(1024, activation=None, use_bias=False)(ln2_do)
    ln3_bn = keras.layers.BatchNormalization(momentum=0.95)(ln3)
    ln3_ac = keras.layers.ReLU()(ln3_bn)
    ln3_do = keras.layers.Dropout(args.dropout)(ln3_ac)

    ln4 = keras.layers.Dense(1024, activation=None, use_bias=False)(ln3_do)
    ln4_bn = keras.layers.BatchNormalization(momentum=0.95)(ln4)
    ln4_ac = keras.layers.ReLU()(ln4_bn)
    ln4_do = keras.layers.Dropout(args.dropout)(ln4_ac)

    ln5 = keras.layers.Dense(1024, activation=None, use_bias=False)(ln4_do)
    ln5_bn = keras.layers.BatchNormalization(momentum=0.95)(ln5)
    ln5_ac = keras.layers.ReLU()(ln5_bn)
    ln5_do = keras.layers.Dropout(args.dropout)(ln5_ac)

    outputs_pdf = keras.layers.Dense(outDimPdf,activation=None,use_bias=True,kernel_initializer="he_normal", bias_initializer='zeros', name="pdfID")(ln5_do)

    outputs_pho = keras.layers.Dense(outDimPho,activation=None,use_bias=True,kernel_initializer="he_normal", bias_initializer='zeros', name="phoneID")(ln5_do)

    return keras.Model(inputs=inputs, outputs=[outputs_pdf,outputs_pho])

def decode_score_test(prob,trans,outDir):

  trans.save( os.path.join(outDir,"ref.txt") )

  hmmFile = f"{args.expDir}/train_sat/final.mdl"
  HCLGFile = f"{args.expDir}/train_sat/graph/HCLG.{args.order}.fst"
  lexicons = exkaldi.load_lex( f"{args.expDir}/dict/lexicons.lex" )
  phoneMap = exkaldi.load_list_table( f"{args.expDir}/dict/phones.48_to_39.map" )
  #print("Decoding...")
  lat = exkaldi.decode.wfst.nn_decode(
                                  prob=prob, 
                                  hmm=hmmFile, 
                                  HCLGFile=HCLGFile, 
                                  symbolTable=lexicons("words"),
                                  beam=args.beam,
                                  latBeam=args.latBeam,
                                  acwt=args.acwt,
                                  minActive=200,
                                  maxActive=7000,
                              )
  #print("Score...")
  minWER = None
  for penalty in [0.,0.5,1.0]:
    for LMWT in range(1,15,1):

      newLat = lat.add_penalty(penalty)
      result = newLat.get_1best(lexicons("phones"),hmmFile,lmwt=LMWT,acwt=1,phoneLevel=True)
      result = exkaldi.hmm.transcription_from_int(result,lexicons("phones"))
      result = result.convert(phoneMap)
      fileName = f"{outDir}/penalty_{penalty}_lmwt_{LMWT}.txt"
      result.save( fileName )
      score = exkaldi.decode.score.wer(ref=trans, hyp=result, mode="present")
      if minWER == None or score.WER < minWER[0]:
          minWER = (score.WER, fileName)
      #print(f"{penalty} {LMWT}",score)

  with open( f"{outDir}/best_PER", "w") as fw:
      fw.write( f"{minWER[0]}% {minWER[1]}" )

  return minWER[0]

class EvaluateWER(keras.callbacks.Callback):

    def __init__(self, model, feat, bias, trans, outDir):
        exkaldi.utils.make_dependent_dirs(outDir, False)
        self.model = model
        self.feat = feat
        self.bias = bias
        self.trans = trans
        self.outDir = outDir

    def on_epoch_end(self, epoch, logs={}):

        if epoch >= args.testStartEpoch:
            subOutDir = os.path.join( self.outDir, f"decode_ep{epoch+1}" )
            exkaldi.utils.make_dependent_dirs(subOutDir, False)

            prob = {}
            for utt, matrix in self.feat.items():
                predPdf, predPhone = self.model(matrix, training=False)
                prob[utt] = predPdf.numpy() + self.bias
                
            prob = exkaldi.load_prob(prob)
            prob = prob.map(lambda x:exkaldi.nn.log_softmax(x,axis=1))
            prob = prob.to_bytes()

            WER = decode_score_test(prob, self.trans, subOutDir)

            logs['test_WER'] = WER

class ModelSaver(keras.callbacks.Callback):

    def __init__(self, model, outDir):

        self.model = model
        self.outDir = outDir
    
    def on_epoch_end(self, epoch, logs={}):
        
        outDir = os.path.join(self.outDir,f"model_ep{epoch+1}.h5")
        self.model.save_weights(outDir)

def output_probability():

  # ------------- Parse arguments from command line ----------------------
  # 1. Add a discription of this program
  args.discribe("This program is used to output DNN probability for realigning") 
  # 2. Add options
  args.add("--expDir", abbr="-e", dtype=str, default="exp", discription="The data and output path of current experiment.")
  args.add("--dropout", abbr="-d", dtype=float, default=0.2, discription="Dropout.")
  args.add("--useCMVN", dtype=bool, default=False, discription="Wether apply CMVN to fmllr feature.")
  args.add("--splice", dtype=int, default=10, discription="Splice how many frames to head and tail for Fmllr feature.")
  args.add("--delta", dtype=int, default=2, discription="Wether add delta to fmllr feature.")
  args.add("--normalizeFeat", dtype=bool, default=True, discription="Wether normalize the chunk dataset.")
  args.add("--predictModel", abbr="-m", dtype=str, default="", discription="If not void, skip training. Do decoding only.")
  # 3. Then start to parse arguments. 
  args.parse()
  
  declare.is_file(args.predictModel)

  dims = exkaldi.load_list_table( f"{args.expDir}/train_dnn/data/dims" )
  featDim = int(dims["fmllr"])
  pdfDim = int(dims["pdfs"])
  phoneDim = int(dims["phones"])

  # Initialize model
  if args.delta > 0:
      featDim *= (args.delta+1)
  if args.splice > 0:
      featDim *= (2*args.splice+1)

  model = make_DNN_model(featDim,pdfDim,phoneDim) 
  model.load_weights(args.predictModel)
  print(f"Restorage model from: {args.predictModel}")

  for Name in ["train","test","dev"]:
    print(f"Processing: {Name} dataset")
    feat = exkaldi.load_feat( f"{args.expDir}/train_dnn/data/{Name}/fmllr.ark" )

    if args.useCMVN:
      print("Apply CMVN")
      cmvn = exkaldi.load_cmvn( f"{args.expDir}/train_dnn/data/{Name}/cmvn_of_fmllr.ark" )
      feat = exkaldi.use_cmvn(feat, cmvn, utt2spk=f"{args.expDir}/train_dnn/data/{Name}/utt2spk")
      del cmvn

    if args.delta > 0:
      print("Add delta to feature")
      feat = feat.add_delta(args.delta)

    if args.splice > 0:
      print("Splice feature")
      feat = feat.splice(args.splice)

    feat = feat.to_numpy()
    if args.normalizeFeat:
      print("Normalize")
      feat = feat.normalize(std=True)

    outProb = {}
    print("Forward model...")
    for utt,mat in feat.items():
      predPdf, predPhone = model(mat, training=False)
      outProb[utt] = exkaldi.nn.log_softmax(predPdf.numpy(),axis=1)
    
    #outProb = exkaldi.load_prob(outProb)
    #outProb.save(f"{args.expDir}/train_dnn/prob/{Name}.npy")
    outProb = exkaldi.load_prob(outProb).to_bytes()
    outProb.save(f"{args.expDir}/train_dnn/prob/{Name}.ark") 
    print("Save done!")

def main():

  # ------------- Parse arguments from command line ----------------------
  # 1. Add a discription of this program
  args.discribe("This program is used to train triphone DNN acoustic model with Tensorflow") 
  # 2. Add options
  args.add("--expDir", abbr="-e", dtype=str, default="exp", discription="The data and output path of current experiment.")
  args.add("--LDAsplice", dtype=int, default=3, discription="Splice how many frames to head and tail for LDA feature.")
  args.add("--randomSeed", dtype=int, default=1234, discription="Random seed.")
  args.add("--batchSize", abbr="-b", dtype=int, default=128, discription="Mini batch size.")
  args.add("--gpu", abbr="-g", dtype=str, default="all", choices=["all","0","1"], discription="Use GPU.")
  args.add("--epoch", dtype=int, default=30, discription="Epoches.")
  args.add("--testStartEpoch", dtype=int, default=5, discription="Start to evaluate test dataset.")
  args.add("--dropout", abbr="-d", dtype=float, default=0.2, discription="Dropout.")
  args.add("--useCMVN", dtype=bool, default=False, discription="Wether apply CMVN to fmllr feature.")
  args.add("--splice", dtype=int, default=10, discription="Splice how many frames to head and tail for Fmllr feature.")
  args.add("--delta", dtype=int, default=2, discription="Wether add delta to fmllr feature.")
  args.add("--normalizeFeat", dtype=bool, default=True, discription="Wether normalize the chunk dataset.")
  args.add("--normalizeAMP", dtype=bool, default=False, discription="Wether normalize the post-probability.")
  args.add("--order", abbr="-o", dtype=int, default=6, discription="Language model order.")
  args.add("--beam", dtype=int, default=13, discription="Decode beam size.")
  args.add("--latBeam", dtype=int, default=6, discription="Lattice beam size.")
  args.add("--acwt", dtype=float, default=0.083333, discription="Acoustic model weight.")
  args.add("--predictModel", abbr="-m", dtype=str, default="", discription="If not void, skip training. Do decoding only.")
  # 3. Then start to parse arguments. 
  args.parse()
  # 4. Take a backup of arguments
  argsLogFile = os.path.join(args.expDir,"conf","train_dnn.args")
  args.save(argsLogFile)

  random.seed(args.randomSeed)
  np.random.seed(args.randomSeed)
  tf.random.set_seed(args.randomSeed)

  # ------------- Prepare data for dnn training ----------------------
  if not os.path.isfile(f"./{args.expDir}/train_dnn/data/dims"):
    prepare_DNN_data()

  # ------------- Prepare data for dnn training ----------------------
  stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  outDir = f"{args.expDir}/train_dnn/out_{stamp}"
  exkaldi.utils.make_dependent_dirs(outDir, pathIsFile=False)

  #------------------------ Training and Validation -----------------------------
  dims = exkaldi.load_list_table( f"{args.expDir}/train_dnn/data/dims" )
  featDim = int(dims["fmllr"])
  pdfDim = int(dims["pdfs"])
  phoneDim = int(dims["phones"])

  # Initialize model
  if args.delta > 0:
      featDim *= (args.delta+1)
  if args.splice > 0:
      featDim *= (2*args.splice+1)

  if len(args.predictModel.strip()) == 0:
    print('Prepare Data Iterator...')
    # Prepare fMLLR feature files
    trainDataset = process_feat_ali(training=True)
    traindataLen = len(trainDataset)
    train_gen = tf.data.Dataset.from_generator(
                                        lambda: make_generator(trainDataset),
                                        (tf.float32, {"pdfID":tf.int32,"phoneID":tf.int32})
                                ).batch(args.batchSize).prefetch(3)
    steps_per_epoch = traindataLen//args.batchSize

    devDataset = process_feat_ali(training=False)
    devdataLen = len(devDataset)
    dev_gen = tf.data.Dataset.from_generator(
                                        lambda: make_generator(devDataset),
                                        (tf.float32, {"pdfID":tf.int32,"phoneID":tf.int32})
                                ).batch(args.batchSize).prefetch(3)
    validation_steps = devdataLen//args.batchSize

    print('Prepare test data')
    testFeat, testBias, testTrans = prepare_test_data(postProbDim=pdfDim)

    def train_step():
        
        model = make_DNN_model(featDim,pdfDim,phoneDim)
        model.summary()

        model.compile(
                    loss = {
                            "pdfID":keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                            "phoneID":keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        },
                    loss_weights = {"pdfID":1,"phoneID":1},
                    metrics = {
                                "pdfID":keras.metrics.SparseCategoricalAccuracy(),
                                "phoneID":keras.metrics.SparseCategoricalAccuracy(),
                            },
                    optimizer = keras.optimizers.SGD(0.08,momentum=0.0),
                )

        def lrScheduler(epoch):
            if epoch > 25:
                return 0.001
            elif epoch > 22:
                return 0.0025
            elif epoch > 19:
                return 0.005
            elif epoch > 17:
                return 0.01
            elif epoch > 15:
                return 0.02
            elif epoch > 10:
                return 0.04
            else:
                return 0.08

        model.fit(
                x = train_gen,
                steps_per_epoch=steps_per_epoch,
                epochs=args.epoch,

                validation_data=dev_gen,
                validation_steps=validation_steps,
                verbose=1,

                initial_epoch=0,
                callbacks=[
                            keras.callbacks.EarlyStopping(patience=5, verbose=1),
                            keras.callbacks.TensorBoard(log_dir=outDir),
                            keras.callbacks.LearningRateScheduler(lrScheduler),
                            EvaluateWER(model,testFeat,testBias,testTrans,outDir), 
                            ModelSaver(model,outDir),         
                        ],
                    )

    print("Using GPU: ", args.gpu)
    if args.gpu != "all":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        train_step()

    else:
        my_strategy = tf.distribute.MirroredStrategy()
        with my_strategy.scope():
            train_step()

  else:
    declare.is_file(args.predictModel)

    model = make_DNN_model(featDim,pdfDim,phoneDim)
    model.summary()

    model.load_weights(args.predictModel)

    print('Prepare test data')
    testFeat, testBias, testTrans = prepare_test_data(postProbDim=pdfDim)
    scorer = EvaluateWER(model,testFeat,testBias,testTrans,outDir)

    logs = {}
    scorer.on_epoch_end(5,logs)
  
if __name__ == "__main__":
    main()
    #output_probability()

