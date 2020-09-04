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
Timit DNN-HMM training recipe.

Part 6: train a LSTM acoustic model with fmllr feature.

'''
import exkaldi
from exkaldi import args
from exkaldi import declare

import tensorflow as tf
from tensorflow import keras
import numpy as np

import shutil
import math
import random
import os
import time,datetime

def prepare_LSTM_data():

  print("Start to prepare data for LSTM training")
  declare.is_dir(f"{args.expDir}/train_dnn/prob", debug="Please run previous programs up to DNN training.")

  # Lexicons and Gmm-Hmm model
  lexicons = exkaldi.load_lex( f"{args.expDir}/dict/lexicons.lex" )
  hmm = f"{args.expDir}/train_sat/final.mdl"
  tree = f"{args.expDir}/train_sat/tree"

  for Name in ["train", "dev", "test"]:
    exkaldi.utils.make_dependent_dirs(f"{args.expDir}/train_lstm/data/{Name}", pathIsFile=False)
    # Load feature
    print(f"Make LDA feature for '{Name}'")
    feat = exkaldi.load_feat( f"{args.expDir}/mfcc/{Name}/mfcc_cmvn.ark" )
    feat = feat.splice(left=args.LDAsplice, right=args.LDAsplice)
    feat = exkaldi.transform_feat(feat, matFile=f"{args.expDir}/train_lda_mllt/trans.mat" )
    # Load probability for aligning( File has a large size, so we use index table. )
    prob = exkaldi.load_index_table( f"{args.expDir}/train_dnn/prob/{Name}.ark" )
    # Compile a aligning graph
    print(f"Copy aligning graph from DNN resources")
    shutil.copyfile( f"{args.expDir}/train_dnn/data/{Name}/align_graph",
                    f"{args.expDir}/train_lstm/data/{Name}/align_graph"
                  )
    # Align
    print("Align")
    ali = exkaldi.decode.wfst.nn_align(
                                    hmm,
                                    prob,
                                    alignGraphFile=f"{args.expDir}/train_lstm/data/{Name}/align_graph", 
                                    lexicons=lexicons,
                                    outFile=f"{args.expDir}/train_lstm/data/{Name}/ali",
                                )
    # Estimate transform matrix
    print("Estimate transform matrix")
    fmllrTransMat = exkaldi.hmm.estimate_fMLLR_matrix(
                                aliOrLat=ali,
                                lexicons=lexicons,
                                aliHmm=hmm,
                                feat=feat,
                                spk2utt=f"{args.expDir}/data/{Name}/spk2utt",
                                outFile=f"{args.expDir}/train_lstm/data/{Name}/trans.ark",
                            )
    # Transform feature
    print("Transform matrix")
    feat = exkaldi.use_fmllr(
                        feat,
                        fmllrTransMat,
                        utt2spk=f"{args.expDir}/data/{Name}/utt2spk",
                        outFile=f"{args.expDir}/train_lstm/data/{Name}/fmllr.ark",
                    )
    # Transform alignment (Because 'ali' is a index table object, we need fetch the alignment data in order to use the 'to_numpy' method.)
    ali = ali.fetch(arkType="ali")
    ali.to_numpy(aliType="pdfID",hmm=hmm).save( f"{args.expDir}/train_lstm/data/{Name}/pdfID.npy" )
    ali.to_numpy(aliType="phoneID",hmm=hmm).save( f"{args.expDir}/train_lstm/data/{Name}/phoneID.npy" )
    del ali
    # Compute cmvn for fmllr feature
    cmvn = exkaldi.compute_cmvn_stats(
                                  feat, 
                                  spk2utt=f"{args.expDir}/data/{Name}/spk2utt",
                                  outFile=f"{args.expDir}/train_lstm/data/{Name}/cmvn_of_fmllr.ark",
                                )
    del cmvn
    del feat
    # copy spk2utt utt2spk and text file
    shutil.copyfile( f"{args.expDir}/data/{Name}/spk2utt", f"{args.expDir}/train_lstm/data/{Name}/spk2utt")
    shutil.copyfile( f"{args.expDir}/data/{Name}/utt2spk", f"{args.expDir}/train_lstm/data/{Name}/utt2spk")
    shutil.copyfile( f"{args.expDir}/data/{Name}/text", f"{args.expDir}/train_lstm/data/{Name}/text" )

  print("Write feature and alignment dim information")
  dims = exkaldi.ListTable()
  feat = exkaldi.load_feat( f"{args.expDir}/train_lstm/data/test/fmllr.ark" ) 
  dims["fmllr"] = feat.dim
  del feat
  hmm = exkaldi.hmm.load_hmm( f"{args.expDir}/train_sat/final.mdl" )
  dims["phones"] = hmm.info.phones + 1
  dims["pdfs"] = hmm.info.pdfs
  del hmm
  dims.save( f"{args.expDir}/train_lstm/data/dims" )

def make_LSTM_model(args, inputDim, outDimPdf, outDimPho):

    inputs = keras.Input((None,inputDim,))

    ln0 = keras.layers.Masking(mask_value=0.0)(inputs)

    ln1 = keras.layers.Bidirectional(
                        keras.layers.LSTM(512, activation='tanh', recurrent_activation='sigmoid', use_bias=False,
                                                kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                                                bias_initializer='zeros', return_sequences=True, dropout=0.0)
                        )(ln0)
    #ln2_bn = keras.layers.BatchNormalization(momentum=0.95)(ln2)

    ln2 = keras.layers.Bidirectional(
                        keras.layers.LSTM(512, activation='tanh', recurrent_activation='sigmoid', use_bias=False,
                                                kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                                                bias_initializer='zeros', return_sequences=True, dropout=0.0)
                        )(ln1)

    ln3 = keras.layers.Bidirectional(
                        keras.layers.LSTM(512, activation='tanh', recurrent_activation='sigmoid', use_bias=False,
                                                kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                                                bias_initializer='zeros', return_sequences=True, dropout=0.0)
                        )(ln2)

    ln4 = keras.layers.Bidirectional(
                        keras.layers.LSTM(512, activation='tanh', recurrent_activation='sigmoid', use_bias=False,
                                                kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                                                bias_initializer='zeros', return_sequences=True, dropout=0.0)
                        )(ln3)         

    ln4_bn = keras.layers.BatchNormalization(momentum=0.95)(ln4)

    outputs_pdf = keras.layers.Dense(outDimPdf, activation=None, use_bias=True, kernel_initializer="he_normal", bias_initializer='zeros', name="pdfID")(ln4_bn)

    outputs_pho = keras.layers.Dense(outDimPho, activation=None, use_bias=True, kernel_initializer="he_normal", bias_initializer='zeros', name="phoneID")(ln4_bn)

    return keras.Model(inputs=inputs, outputs=[outputs_pdf,outputs_pho])

def process_feat_ali(training=True):

  if training:
    Name = "train"
  else:
    Name = "dev"

  feat = exkaldi.load_feat( f"{args.expDir}/train_lstm/data/{Name}/fmllr.ark" )

  if args.useCMVN:
      cmvn = exkaldi.load_cmvn(f"{args.expDir}/train_lstm/data/{Name}/cmvn_of_fmllr.ark")
      feat = exkaldi.use_cmvn(feat,cmvn,f"{args.expDir}/train_lstm/data/{Name}/utt2spk")
      del cmvn
  
  if args.delta > 0:
      feat = feat.add_delta(args.delta)

  if args.splice > 0:
      feat = feat.splice(args.splice)

  feat = feat.to_numpy()

  if args.normalizeFeat:
      feat = feat.normalize(std=True)
  
  pdfAli = exkaldi.load_ali( f"{args.expDir}/train_lstm/data/{Name}/pdfID.npy" )
  phoneAli = exkaldi.load_ali( f"{args.expDir}/train_lstm/data/{Name}/phoneID.npy" )
  
  feat.rename("feat")
  pdfAli.rename("pdfID")
  phoneAli.rename("phoneID")

  return feat, pdfAli, phoneAli

def tuple_dataset(feat,pdfAli,phoneAli,cutLength=None):

  if cutLength is not None:
      newFeat = feat.cut(cutLength).sort(by="frame")
      newPdfAli = pdfAli.cut(cutLength).sort(by="frame")
      newPhoneAli = phoneAli.cut(cutLength).sort(by="frame")
  else:
      newFeat = feat.sort(by="frame")
      newPdfAli = pdfAli.sort(by="frame")
      newPhoneAli = phoneAli.sort(by="frame")

  newFeat.rename("feat")
  newPdfAli.rename("pdfID")
  newPhoneAli.rename("phoneID")

  dataset = exkaldi.tuple_dataset([newFeat, newPdfAli, newPhoneAli], frameLevel=False)

  return dataset  

def prepare_test_data(postProbDim):

  feat = exkaldi.load_feat( f"{args.expDir}/train_lstm/data/test/fmllr.ark" )

  if args.useCMVN:
    cmvn = exkaldi.load_cmvn( f"{args.expDir}/train_lstm/data/test/cmvn_of_fmllr.ark" )
    feat = exkaldi.use_cmvn(feat, cmvn, utt2spk=f"{args.expDir}/train_lstm/data/test/utt2spk")
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
    ali = exkaldi.load_ali(f"{args.expDir}/train_lstm/data/train/pdfID.npy", aliType="pdfID")
    normalizeBias = exkaldi.nn.compute_postprob_norm(ali,postProbDim)
  else:
    normalizeBias = 0
  
  # ref transcription
  trans = exkaldi.load_transcription(f"{args.expDir}/train_lstm/data/test/text")
  convertTable = exkaldi.load_list_table(f"{args.expDir}/dict/phones.48_to_39.map")
  trans = trans.convert(convertTable)

  return feat, normalizeBias, trans

class DataIterator:

  def __init__(self, batchSize, training=True):
    self.feat, self.pdfAli, self.phoneAli = process_feat_ali(training)
    self.dataset = tuple_dataset(self.feat,self.pdfAli,self.phoneAli,100)
    self.batchSize = batchSize
    self.currentEpoch = 0
    self.dataIndex = 0
    self.datasetSize = len(self.dataset)
    self.isNewEpoch = False
    self.currentPosition = 0
    self.epochSize = math.ceil(self.datasetSize/self.batchSize)

  def next(self):

    batchFeat = []
    batchPdfAli = []
    batchPhoneAli = []

    self.isNewEpoch = False

    for i in range(self.batchSize):
      one = self.dataset[self.dataIndex]
      self.dataIndex += 1

      if self.dataIndex >= self.datasetSize:

        self.currentEpoch += 1
        self.isNewEpoch = True

        if self.currentEpoch < 5:
          del self.dataset
          if self.currentEpoch == 1:
            self.dataset = tuple_dataset(self.feat,self.pdfAli,self.phoneAli,200)
          elif self.currentEpoch == 2:
            self.dataset = tuple_dataset(self.feat,self.pdfAli,self.phoneAli,400)
          elif self.currentEpoch == 3:
            self.dataset = tuple_dataset(self.feat,self.pdfAli,self.phoneAli,800)
          elif self.currentEpoch == 4:
            self.dataset = tuple_dataset(self.feat,self.pdfAli,self.phoneAli,1000)

          self.datasetSize = len(self.dataset)
          self.epochSize = math.ceil(self.datasetSize/self.batchSize)
        
        random.shuffle(self.dataset)
        self.dataIndex = 0

      batchFeat.append( one.feat )
      batchPdfAli.append( one.pdfID )
      batchPhoneAli.append( one.phoneID )

    self.currentPosition += 1

    return ( tf.convert_to_tensor(keras.preprocessing.sequence.pad_sequences(batchFeat, maxlen=None, dtype='float32', padding='post', value=0.0), tf.float32),
            tf.convert_to_tensor(keras.preprocessing.sequence.pad_sequences(batchPdfAli, maxlen=None, dtype='int32', padding='post', value=0), tf.int32),
            tf.convert_to_tensor(keras.preprocessing.sequence.pad_sequences(batchPhoneAli, maxlen=None, dtype='int32', padding='post', value=0), tf.int32),
        )

class EvaluateWER:

  def __init__(self, model, feat, bias, trans, outDir):
    exkaldi.utils.make_dependent_dirs(outDir, False)
    self.model = model
    self.feat = feat
    self.bias = bias
    self.trans = trans
    self.outDir = outDir

  def test(self, epoch):
    subOutDir = os.path.join( self.outDir, f"decode_ep{epoch+1}" )
    exkaldi.utils.make_dependent_dirs(subOutDir, False)
    tf.print("forward network", end=" ")
    prob = {}
    scale = len(self.feat.utts)//10
    for index, (utt, matrix) in enumerate(self.feat.items()):
      if index%scale == 0:
          tf.print(">", end="")
      predPdf, predPhone = self.model(matrix[None,:,:], training=False)
      prob[utt] = predPdf.numpy()[0] + self.bias
    prob = exkaldi.load_prob(prob)
    prob = prob.map(lambda x:exkaldi.nn.log_softmax(x,axis=1))
    prob = prob.to_bytes()
    WER = self.decode_score_test(prob, subOutDir)
            
    return WER

  def decode_score_test(self, prob, outDir):
    trans = self.trans
    #print("Compute WER")
    trans.save( os.path.join(outDir, "ref.txt") )
    tf.print(" decode", end=" > ")
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
    tf.print("score", end="")
    minWER = None
    for penalty in [0.,0.5,1.0]:
      for LMWT in range(1,15):
        newLat = lat.add_penalty(penalty)
        result = newLat.get_1best(lexicons("phones"),hmmFile,lmwt=LMWT,acwt=1,phoneLevel=True)
        result = exkaldi.hmm.transcription_from_int(result,lexicons("phones"))
        result = result.convert(phoneMap, None)
        fileName = os.path.join( outDir, f"penalty_{penalty}_lmwt_{LMWT}.txt" )
        result.save( fileName )
        score = exkaldi.decode.score.wer(ref=trans, hyp=result, mode="present")
        if minWER == None or score.WER < minWER[0]:
            minWER = (score.WER, fileName)
        #print(score.WER, fileName)
    with open( os.path.join(outDir,"best_PER"), "w") as fw:
      fw.write( f"{minWER[0]}% {minWER[1]}" )
    return minWER[0]

class ModelSaver:

  def __init__(self, model, outDir):
    self.model = model
    self.outDir = outDir
  
  def save(self, epoch):
    outDir = os.path.join(self.outDir,f"model_ep{epoch+1}.h5")
    self.model.save_weights(outDir)

def main():

  # ------------- Parse arguments from command line ----------------------
  # 1. Add a discription of this program
  args.discribe("This program is used to train triphone LSTM scoustic model with Tensorflow") 
  # 2. Add options
  args.add("--expDir", abbr="-e", dtype=str, default="exp", discription="The data and output path of current experiment.")
  args.add("--LDAsplice", dtype=int, default=3, discription="Splice how many frames to head and tail for LDA feature.")
  args.add("--randomSeed", dtype=int, default=1234, discription="Random seed.")
  args.add("--batchSize", abbr="-b", dtype=int, default=8, discription="Mini batch size.")
  args.add("--gpu", abbr="-g", dtype=str, default="all", choices=["all","0","1"], discription="Use GPU.")
  args.add("--epoch", dtype=int, default=30, discription="Epoches.")
  args.add("--testStartEpoch", dtype=int, default=5, discription="Start to evaluate test dataset.")
  args.add("--dropout", abbr="-d", dtype=float, default=0.2, discription="Dropout.")
  args.add("--useCMVN", dtype=bool, default=False, discription="Wether apply CMVN to fmllr feature.")
  args.add("--splice", dtype=int, default=0, discription="Splice how many frames to head and tail for Fmllr feature.")
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
  args.save( f"./{args.expDir}/conf/train_lstm.args" )

  random.seed(args.randomSeed)
  np.random.seed(args.randomSeed)
  tf.random.set_seed(args.randomSeed)

  # ------------- Prepare data for dnn training ----------------------
  if not os.path.isfile(f"./{args.expDir}/train_lstm/data/dims"):
    prepare_LSTM_data()

  # ------------- Prepare data for lstm training ----------------------
  stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  outDir = f"{args.expDir}/train_lstm/out_{stamp}"
  exkaldi.utils.make_dependent_dirs(outDir, pathIsFile=False)

  #------------------------ Training and Validation -----------------------------
  dims = exkaldi.load_list_table( f"{args.expDir}/train_lstm/data/dims" )
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
    trainIterator = DataIterator(batchSize=args.batchSize, training=True)
    devIterator = DataIterator(batchSize=args.batchSize, training=False)

    print('Prepare test data')
    testFeat, testBias, testTrans = prepare_test_data(postProbDim=pdfDim)

    metris = {
            "train_loss":keras.metrics.Mean(name="train/loss", dtype=tf.float32),
            "train_pdfID_accuracy":keras.metrics.Mean(name="train/pdfID_accuracy", dtype=tf.float32),
            "train_phoneID_accuracy":keras.metrics.Mean(name="train/phoneID_accuracy", dtype=tf.float32),
            "dev_loss":keras.metrics.Mean(name="eval/loss", dtype=tf.float32),
            "dev_pdfID_accuracy":keras.metrics.Mean(name="eval/pdfID_accuracy", dtype=tf.float32),
            "dev_phoneID_accuracy":keras.metrics.Mean(name="eval/phoneID_accuracy", dtype=tf.float32),
        }

    def train_step(model,optimizer,batch):
      feat, pdfAli, phoneAli = batch
      with tf.GradientTape() as tape:
        pdfPred, phonePred = model(feat, training=True)
        L1 = keras.losses.sparse_categorical_crossentropy(pdfAli, pdfPred, from_logits=True)
        L2 = keras.losses.sparse_categorical_crossentropy(phoneAli, phonePred, from_logits=True)
        loss = L1 + L2
      gradients = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))

      metris["train_loss"](loss)

      #pdfPred = tf.convert_to_tensor(pdfPred, np.float32)
      A1 = keras.metrics.sparse_categorical_accuracy(pdfAli, pdfPred)
      metris["train_pdfID_accuracy"](A1)

      #phonePred = tf.convert_to_tensor(phonePred, np.float32)
      A2 = keras.metrics.sparse_categorical_accuracy(phoneAli, phonePred)
      metris["train_phoneID_accuracy"](A2)
      
      return float(np.mean(L1.numpy())), float(np.mean(L2.numpy())),float(np.mean(A1.numpy())),float(np.mean(A2.numpy()))
  
    def dev_step(model,batch):
      feat, pdfAli, phoneAli = batch
      pdfPred, phonePred = model(feat, training=False)
      L1 = keras.losses.sparse_categorical_crossentropy(pdfAli, pdfPred, from_logits=True)
      L2 = keras.losses.sparse_categorical_crossentropy(phoneAli, phonePred, from_logits=True)
      loss = L1 + L2

      metris["dev_loss"](loss)

      #pdfPred = tf.convert_to_tensor(pdfPred, np.float32)
      A1 = keras.metrics.sparse_categorical_accuracy(pdfAli, pdfPred)
      metris["dev_pdfID_accuracy"](A1)

      #phonePred = tf.convert_to_tensor(phonePred, np.float32)
      A2 = keras.metrics.sparse_categorical_accuracy(phoneAli, phonePred)
      metris["dev_phoneID_accuracy"](A2)      

      return float(np.mean(L1.numpy())), float(np.mean(L2.numpy())),float(np.mean(A1.numpy())),float(np.mean(A2.numpy()))

    def main_loop():

      model = make_LSTM_model(args, featDim, pdfDim, phoneDim)
      model.summary()

      optimizer = keras.optimizers.RMSprop(learning_rate=0.0004, rho=0.95, momentum=0.0, epsilon=1e-07)

      scorer = EvaluateWER( model, testFeat, testBias, testTrans, outDir)
      modelSaver = ModelSaver(model, outDir)

      for e in range(args.epoch):
          # Training
          startTime = time.time()
          for i in range(trainIterator.epochSize):
              batch = trainIterator.next()
              pdfLoss, phoneLoss, pdfAcc, phoneAcc = train_step(model, optimizer, batch)
              tf.print(f"\rtraining: {i}/{trainIterator.epochSize} pdfID loss {pdfLoss:.3f} phoneID loss {phoneLoss:.3f} pdfID accuracy {pdfAcc:.3f} phoneID accuracy {phoneAcc:.3f}", end="\t")
          tf.print()
          # Evaluate
          for i in range(devIterator.epochSize):
              batch = devIterator.next()
              pdfLoss, phoneLoss, pdfAcc, phoneAcc = dev_step(model, batch)
              tf.print(f"\revaluate: {i}/{devIterator.epochSize} pdfID loss {pdfLoss:.3f} phoneID loss {phoneLoss:.3f} pdfID accuracy {pdfAcc:.3f} phoneID accuracy {phoneAcc:.3f}", end="\t")
          tf.print()
          # Test
          tf.print("testing:", end=" ")
          testWER = scorer.test(e)
          tf.print()

          endTime = time.time()
          message = f"epoch {e} "
          for Name in metris.keys():
              message += f"{Name} {float(metris[Name].result().numpy()):.3f} "
              metris[Name].reset_states()
          message += f"test PER {testWER:.2f} time cost {int(endTime-startTime)}s"
          tf.print(message)
          
          modelSaver.save(e)
          
    main_loop()

  else:
    declare.is_file(args.predictModel)

    model = make_LSTM_model(featDim,pdfDim,phoneDim)
    model.summary()

    model.load_weights(args.predictModel)

    print('Prepare test data')
    testFeat, testBias, testTrans = prepare_test_data(postProbDim=pdfDim)
    scorer = EvaluateWER(model,testFeat,testBias,testTrans,outDir)

    scorer.test(0)

if __name__ == "__main__":
  main()