####################### Version Information ################################
# LSTM example model based on: exkaldi V0.1.1 and chainer V5.3
# Yu Wang, University of Yamanashi 
# Nov 11, 2019
############################################################################

############################################################################
# Before you run this script, the follow files are expected to have:

# 1,Fmllr Data file: 
#                    < TIMIT Root Path >/data-fmllr-tri3/train/feats.scp. 
#                    < TIMIT Root Path >/data-fmllr-tri3/dev/feats.scp. 
#                    < TIMIT Root Path >/data-fmllr-tri3/test/feats.scp. 
# You can obtain it by runing: 
#                    <TIMIT Root Path>/local/nnet/run_dnn.sh

# 2, Alignment file: 
#                    < TIMIT Root Path >/exp/dnn4_pretrain-dbn_dnn_ali/ali.*.gz. 
#                    < TIMIT Root Path >/exp/dnn4_pretrain-dbn_dnn_ali_dev/ali.*.gz. 
# You can obtain it by runing: 
#                    <TIMIT Root Path>/steps/nnet/align.sh --nj 4 data-fmllr-tri3/train data/lang exp/dnn4_pretrain-dbn_dnn exp/dnn4_pretrain-dbn_dnn_ali
#                    <TIMIT Root Path>/steps/nnet/align.sh --nj 4 data-fmllr-tri3/dev data/lang exp/dnn4_pretrain-dbn_dnn exp/dnn4_pretrain-dbn_dnn_ali_dev

# 3,If using CMVN, please prepare CMVN state file: 
#                    < TIMIT Root Path >/data-fmllr-tri3/train/cmvn.ark
#                    < TIMIT Root Path >/data-fmllr-tri3/dev/cmvn.ark
#                    < TIMIT Root Path >/data-fmllr-tri3/test/cmvn.ark
# You can obtain it by runing: 
#                    <TIMIT Root Path>/local/nnet/run_dnn.sh
# Or compute them with exkaldi tool:
#                    exkaldi.compute_cmvn_stats
#
############################################################################

from __future__ import print_function
import exkaldi as E
import chainer
import chainer.functions as F
import chainer.links as L
import random
import numpy as np
import os, datetime, socket, time
import argparse

parser = argparse.ArgumentParser(description='LSTM Acoustic model on TIMIT corpus')
parser.add_argument('--decodeMode', type=bool, default=False, help='If false, run train recipe.')
parser.add_argument('--TIMITpath', '-t', type=str, default='kaldi/egs/timit/demo', help='Kaldi timit rescipe folder')
parser.add_argument('--randomSeed', '-r', type=int, default=1234)
parser.add_argument('--batchSize', '-b', type=int, default=8)
parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU id (Negative number stands for CPU)')
parser.add_argument('--epoch', '-e', type=int, default=10)
parser.add_argument('--layer', '-l', type=int, default=4)
parser.add_argument('--hiddenNode', '-hn', type=int, default=512)
parser.add_argument('--dropout', '-do', type=float, default=0)
parser.add_argument('--outDir','-o', type=str, default='TIMIT_LSTM_fmllr_exp')
parser.add_argument('--useCMVN', '-u', type=bool, default=True)
parser.add_argument('--splice', '-s', type=int, default=0)
parser.add_argument('--delta', '-d', type=int, default=2)
parser.add_argument('--normalizeChunk', '-nC', type=bool, default=True)
parser.add_argument('--normalizeAMP', '-nA', type=bool, default=True)
parser.add_argument('--minActive', '-minA', type=int, default=200)
parser.add_argument('--maxActive', '-maxA', type=int, default=7000)
parser.add_argument('--maxMemory', '-maxM', type=int, default=50000000)
parser.add_argument('--beam', '-beam', type=int, default=13)
parser.add_argument('--latBeam', '-latB', type=int, default=8)
parser.add_argument('--acwt', '-acwt', type=float, default=1.0)
parser.add_argument('--minLmwt', '-minL', type=int, default=1)
parser.add_argument('--maxLmwt', '-maxL', type=int, default=15)
parser.add_argument('--preModel', '-p', type=str, default='')
args = parser.parse_args()

if args.outDir.endswith('/'):
    args.outDir = args.outDir[0:-1]

if args.TIMITpath.endswith('/'):
    args.TIMITpath = args.TIMITpath[0:-1]

if not os.path.isdir(args.outDir):
    os.mkdir(args.outDir)

random.seed(args.randomSeed)
np.random.seed(args.randomSeed)
chainer.configuration.config.deterministic = True
chainer.configuration.config.cudnn_deterministic = True

if args.gpu >= 0:
    import cupy as cp
    cp.random.seed(args.randomSeed)
else:
    cp = np

class LSTM(chainer.Chain):
    def __init__(self,inputDim, outDimPdf, outDimPho):
        super(LSTM,self).__init__()
        with self.init_scope():
            
            global args

            self.nl = L.NStepBiLSTM(args.layer,inputDim,args.hiddenNode,args.dropout)

            self.bn = L.BatchNormalization(args.hiddenNode*2,0.95)

            #for param in self.params():
            #    param.array[...] = np.random.uniform(-0.1, 0.1, param.shape)

            initializerW = None #chainer.initializers.HeNormal()
            initializerBias = None #chainer.initializers.Zero()
            
            self.ln1 = L.Linear(None,outDimPdf,initialW=initializerW,initial_bias=initializerBias)
            self.ln2 = L.Linear(None,outDimPho,initialW=initializerW,initial_bias=initializerBias)

    def __call__(self,x):

        hy, cy, ys = self.nl(None,None,x)

        h = F.concat(ys,axis=0)

        h = self.bn(h)

        h1 = self.ln1(h)
        h2 = self.ln2(h)

        return h1 ,h2

def train_model():

    global args

    print("\n############## Parameters Configure ##############")
    # Show configure information and write them to file
    def configLog(message,f):
        print(message)
        f.write(message+'\n')
    f = open(args.outDir+'/configure',"w")       
    configLog('Start System Time:{}'.format(datetime.datetime.now().strftime("%Y-%m-%d %X")),f)
    configLog('Host Name:{}'.format(socket.gethostname()),f)
    configLog('Fix Random Seed:{}'.format(args.randomSeed),f)
    configLog('Mini Batch Size:{}'.format(args.batchSize),f)
    configLog('GPU ID:{}'.format(args.gpu),f)
    configLog('Train Epochs:{}'.format(args.epoch),f)
    configLog('Output Folder:{}'.format(args.outDir),f)
    configLog('LSTM layers:{}'.format(args.layer),f)
    configLog('LSTM hidden nodes:{}'.format(args.hiddenNode),f)
    configLog('LSTM dropout:{}'.format(args.dropout),f)
    configLog('Use CMVN:{}'.format(args.useCMVN),f)
    configLog('Splice N Frames:{}'.format(args.splice),f)
    configLog('Add N Deltas:{}'.format(args.delta),f)
    configLog('Normalize Chunk:{}'.format(args.normalizeChunk),f)
    configLog('Normalize AMP:{}'.format(args.normalizeAMP),f)
    configLog('Decode Minimum Active:{}'.format(args.minActive),f)
    configLog('Decode Maximum Active:{}'.format(args.maxActive),f)  
    configLog('Decode Maximum Memory:{}'.format(args.maxMemory),f)  
    configLog('Decode Beam:{}'.format(args.beam),f)   
    configLog('Decode Lattice Beam:{}'.format(args.latBeam),f) 
    configLog('Decode Acoustic Weight:{}'.format(args.acwt),f) 
    configLog('Decode minimum Language Weight:{}'.format(args.minLmwt),f) 
    configLog('Decode maximum Language Weight:{}'.format(args.maxLmwt),f) 
    f.close()

    print("\n############## Train LSTM Acoustic Model ##############")

    #------------------------ STEP 1: Prepare Train and Validation Data -----------------------------

    print('Prepare Data Iterator...')
    # Fmllr feature data
    trainScpFile = args.TIMITpath + '/data-fmllr-tri3/train/feats.scp'
    devScpFile = args.TIMITpath + '/data-fmllr-tri3/dev/feats.scp'
    # Train Alignment label file
    trainAliFile = args.TIMITpath + '/exp/dnn4_pretrain-dbn_dnn_ali/ali.*.gz'
    trainLabelPdf = E.load_ali(trainAliFile)
    trainLabelPho = E.load_ali(trainAliFile,returnPhone=True)
    for i in trainLabelPho.keys():
        trainLabelPho[i] = trainLabelPho[i] - 1
    # Dev Alignment label file
    devAliFile = args.TIMITpath + '/exp/dnn4_pretrain-dbn_dnn_ali_dev/ali.*.gz'
    devLabelPdf = E.load_ali(devAliFile)
    devLabelPho = E.load_ali(devAliFile,returnPhone=True)
    for i in devLabelPho.keys():
        devLabelPho[i] = devLabelPho[i] - 1
    # CMVN file
    trainUttSpk = args.TIMITpath + '/data-fmllr-tri3/train/utt2spk'
    trainCmvnState = args.TIMITpath + '/data-fmllr-tri3/train/cmvn.ark'
    devUttSpk = args.TIMITpath + '/data-fmllr-tri3/dev/utt2spk'
    devCmvnState = args.TIMITpath + '/data-fmllr-tri3/dev/cmvn.ark'    
    # Design a process function
    def loadChunkData(iterator, feat, otherArgs):
        # <feat> is KaldiArk object
        global args
        uttSpk, cmvnState, labelPdf, labelPho, toDo = otherArgs
        # use CMVN
        if args.useCMVN: 
            feat = E.use_cmvn(feat,cmvnState,uttSpk)
        # Add delta
        if args.delta > 0:
            feat = E.add_delta(feat,args.delta)
        # Splice front-back n frames
        if args.splice > 0: 
            feat = feat.splice(args.splice)
        # Transform to KaldiDict and sort them by frame length
        feat = feat.array
        # Normalize
        if args.normalizeChunk: 
            feat = feat.normalize()
        # Concatenate label
        datas = feat.concat([labelPdf,labelPho],axis=1)
        # cut frames
        if toDo == 'train':
            if iterator.epoch >= 4:
                datas = datas.cut(1000)
            elif iterator.epoch >= 3:
                datas = datas.cut(800)
            elif iterator.epoch >= 2:
                datas = datas.cut(400)
            elif iterator.epoch >= 1:
                datas = datas.cut(200)
            elif iterator.epoch >= 0:
                datas = datas.cut(100)
        # Transform trainable numpy data
        datas,_ = datas.merge(keepDim=True,sortFrame=True)
        return datas
    # Make data iterator
    train = E.DataIterator(trainScpFile, loadChunkData, args.batchSize, chunks=5, shuffle=False, otherArgs=(trainUttSpk,trainCmvnState,trainLabelPdf,trainLabelPho,'train'))
    print('Generate train dataset done. Chunks:{} / Batch size:{}'.format(train.chunks,train.batchSize))
    dev = E.DataIterator(devScpFile, loadChunkData, args.batchSize, chunks=1, shuffle=False, otherArgs=(devUttSpk,devCmvnState,devLabelPdf,devLabelPho,'dev'))
    print('Generate validation dataset done. Chunks:{} / Batch size:{}.'.format(dev.chunks,dev.batchSize))
    
    #--------------------------------- STEP 2: Prepare Model --------------------------

    print('Prepare Model...')
    # Initialize model
    featDim = 40
    if args.delta>0:
        featDim *= (args.delta + 1)
    if args.splice > 0:
        featDim *= ( 2 * args.splice + 1 ) 
    print()
    model = LSTM(featDim, trainLabelPdf.target, trainLabelPho.target)
    if args.gpu >=0:
        model.to_gpu(args.gpu)
    # Initialize model
    #lr = [(0,0.5),(10,0.25),(15,0.125),(17,0.07),(19,0.035),(22,0.02),(25,0.01)]
    lr = [(0,0.0004)]
    print('Learning Rate (epoch,newLR):',lr)
    optimizer = chainer.optimizers.RMSprop(lr=lr[0][1],alpha=0.95, eps=1e-8)
    #optimizer = chainer.optimizers.MomentumSGD(lr[0][1],momentum=0.0)
    lr.pop(0)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(0.0))
    # Prepare a supporter to help handling training information.
    supporter = E.Supporter(args.outDir)

    #------------------ STEP 3: Prepare Decode Test Data and Function ------------------

    print('Prepare decode test data...')
    # Fmllr file
    testFilePath = args.TIMITpath + '/data-fmllr-tri3/test/feats.scp'
    testFeat = E.load(testFilePath)
    # Use CMVN
    if args.useCMVN:
        testUttSpk = args.TIMITpath + '/data-fmllr-tri3/test/utt2spk'
        testCmvnState = args.TIMITpath + '/data-fmllr-tri3/test/cmvn.ark'
        testFeat = E.use_cmvn(testFeat,testCmvnState,testUttSpk)
    # Add delta
    if args.delta > 0:  
        testFeat = E.add_delta(testFeat,args.delta)
    # Splice frames
    if args.splice > 0:   
        testFeat = testFeat.splice(args.splice)
    # Transform to array
    testFeat = testFeat.array
    # Normalize
    if args.normalizeChunk:   
        testFeat = testFeat.normalize()
    # Normalize acoustic model output
    if args.normalizeAMP:
        # Compute pdf counts in order to normalize acoustic model posterior probability.
        countFile = args.outDir+'/pdfs_counts.txt'
        # Get statistics file
        if not os.path.isfile(countFile):
            _ = E.analyze_counts(aliFile=trainAliFile,outFile=countFile)
        with open(countFile) as f:
            line = f.readline().strip().strip("[]").strip()
        # Get AMP bias value
        counts = np.array(list(map(float,line.split())),dtype=np.float32)
        normalizeBias = np.log(counts/np.sum(counts))
    else:
        normalizeBias = 0
    # Now Design a function to compute WER score
    def wer_fun(model,testFeat,normalizeBias):
        global args
        # Use decode test data to forward network
        temp = E.KaldiDict()
        print('(testing) Forward network',end=" "*20+'\r')
        with chainer.using_config('train',False),chainer.no_backprop_mode():
            for utt in testFeat.keys():
                data = [cp.array(testFeat[utt],dtype=cp.float32)]
                out1,out2 = model(data)
                out = F.log_softmax(out1,axis=1)
                out.to_cpu()
                temp[utt] = out.array - normalizeBias
        # Tansform KaldiDict to KaldiArk format
        print('(testing) Transform to ark',end=" "*20+'\r')
        amp = temp.ark
        # Decode and obtain lattice
        hmm = args.TIMITpath + '/exp/dnn4_pretrain-dbn_dnn_ali_test/final.mdl'
        hclg = args.TIMITpath + '/exp/tri3/graph/HCLG.fst'
        lexicon = args.TIMITpath + '/exp/tri3/graph/words.txt'
        print('(testing) Generate Lattice',end=" "*20+'\r')
        lattice = E.decode_lattice(amp,hmm,hclg,lexicon,args.minActive,args.maxActive,args.maxMemory,args.beam,args.latBeam,args.acwt)
        # Change language weight from 1 to 10, get the 1best words.
        print('(testing) Get 1-best words',end=" "*20+'\r')
        outs = lattice.get_1best(lmwt=args.minLmwt,maxLmwt=args.maxLmwt,outFile=args.outDir+'/outRaw')
        # If reference file is not existed, make it.
        phonemap = args.TIMITpath + '/conf/phones.60-48-39.map'
        outFilter = args.TIMITpath + '/local/timit_norm_trans.pl -i - -m {} -from 48 -to 39'.format(phonemap)       
        if not os.path.isfile(args.outDir+'/test_filt.txt'):
            refText = args.TIMITpath + '/data/test/text'
            cmd = 'cat {} | {} > {}/test_filt.txt'.format(refText,outFilter,args.outDir)
            (_,_) = E.run_shell_cmd(cmd)
        # Score WER and find the smallest one.
        print('(testing) Score WER',end=" "*20+'\r')
        minWER = None
        for k in range(args.minLmwt,args.maxLmwt+1,1):
            cmd = 'cat {} | {} > {}/test_prediction_filt.txt'.format(outs[k],outFilter,args.outDir)
            (_,_) = E.run_shell_cmd(cmd)
            os.remove(outs[k])
            score = E.wer('{}/test_filt.txt'.format(args.outDir),"{}/test_prediction_filt.txt".format(args.outDir),mode='all')
            if minWER == None or score['WER'] < minWER:
                minWER = score['WER']
            os.remove("{}/test_prediction_filt.txt".format(args.outDir))
        return minWER

    #-------------------------- STEP 4: Train Model ---------------------------

    # While first epoch, the epoch size is computed gradually, so the prograss information will be inaccurate. 
    print('Now Start to Train')
    print('Note that: The first epoch will be doing the statistics of total data size gradually.')
    print('Note that: We will evaluate the WER of test dataset after epoch which will cost a few seconds.')
    # Preprocessing batch data which is getten from data iterator
    def convert(batch):
        data = []
        label1 = []
        label2 = []
        for x in batch:
            data.append(cp.array(x[:,0:-2],dtype=cp.float32))
            label1.append(cp.array(x[:,-2],dtype=cp.int32))
            label2.append(cp.array(x[:,-1],dtype=cp.int32))
        return data,cp.concatenate(label1,axis=0),cp.concatenate(label2,axis=0)
    # We will save model during training loop, so prepare a model-save function 
    def saveFunc(archs):
        global args
        fileName, model = archs
        copymodel = model.copy()
        if args.gpu >= 0:
            copymodel.to_cpu()
        chainer.serializers.save_npz(fileName, copymodel)
    # Start training loop
    for e in range(args.epoch):
        print()
        supporter.send_report({'epoch':e})
        i = 1
        usedTime = 0
        # Train
        while True:
            start = time.time()
            # Get data >> Forward network >> Loss back propagation >> Update
            batch = train.next()
            data,label1,label2 = convert(batch)
            with chainer.using_config('Train',True):
                h1,h2 = model(data)            
                L1 = F.softmax_cross_entropy(h1,label1)
                L2 = F.softmax_cross_entropy(h2,label2)
                loss = L1 + L2
                acc = F.accuracy(F.softmax(h1,axis=1),label1)
            model.cleargrads()
            loss.backward()
            optimizer.update()
            # Compute time cost
            ut = time.time() - start
            usedTime += ut
            print("(training) Epoch:{}/{}% Chunk:{}/{}% Iter:{} Used-time:{}s Batch-loss:{} Speed:{}iters/s".format(e,int(100*train.epochProgress),train.chunk,int(100*train.chunkProgress),i,int(usedTime),"%.4f"%(float(loss.array)),"%.2f"%(1/ut)), end=" "*5+'\r')
            i += 1
            supporter.send_report({'train_loss':float(loss.data),'train_acc':float(acc.data)})
            # If forward all data, break
            if train.isNewEpoch:
                break
        print()
        i = 1
        usedTime = 0
        # Validate
        while True:
            start = time.time()
            # Get data >> Forward network >> Score
            batch = dev.next()
            data,label1,label2 = convert(batch)
            with chainer.using_config('train',False),chainer.no_backprop_mode():
                h1,h2 = model(data)
                loss = F.softmax_cross_entropy(h1,label1)
                acc = F.accuracy(F.softmax(h1,axis=1),label1)
            # Compute time cost
            ut = time.time() - start
            usedTime += ut
            print("(Validating) Epoch:{}/{}% Chunk:{}/{}% Iter:{} Used-time:{}s Batch-loss:{} Speed:{}iters/s".format(e,int(100*dev.epochProgress),dev.chunk,int(100*dev.chunkProgress),i,int(usedTime),"%.4f"%(float(loss.array)),"%.2f"%(1/ut)), end=" "*5+'\r')
            i += 1
            supporter.send_report({'dev_loss':float(loss.data),'dev_acc':float(acc.data)})
            # If forward all data, break
            if dev.isNewEpoch:
                break
        print()
        # Compute WER score
        WERscore = wer_fun(model,testFeat,normalizeBias)
        supporter.send_report({'test_wer':WERscore,'lr':optimizer.lr})
        # Collect all information of this epoch that is reported before, and show them at display
        supporter.collect_report(plot=True)
        # Save model
        supporter.save_arch(saveFunc,arch={'LSTM':model})
        # Change learning rate 
        if len(lr) > 0 and supporter.judge('epoch','>=',lr[0][0]):
            optimizer.lr = lr[0][1]
            lr.pop(0)

    print("LSTM Acoustic Model training done.")
    print("The final model has been saved as:",supporter.finalArch)
    print('Over System Time:',datetime.datetime.now().strftime("%Y-%m-%d %X"))

def decode_test(outDimPdf=1968,outDimPho=48):

    global args

    if args.preModel == '':
        raise Exception("Expected Pretrained Model.")      
    elif not os.path.isfile(args.preModel):
        raise Exception("No such file:{}.".format(args.preModel))

    print("\n############## Parameters Configure ##############")
    # Show configure information and write them to file
    def configLog(message,f):
        print(message)
        f.write(message+'\n')
    f = open(args.outDir+'/configure',"w")       
    configLog('Start System Time:{}'.format(datetime.datetime.now().strftime("%Y-%m-%d %X")),f)
    configLog('Host Name:{}'.format(socket.gethostname()),f)
    configLog('Fix Random Seed:{}'.format(args.randomSeed),f)
    configLog('GPU ID:{}'.format(args.gpu),f)
    configLog('Pretrained Model:{}'.format(args.preModel),f)
    configLog('Output Folder:{}'.format(args.outDir),f)
    configLog('Use CMVN:{}'.format(args.useCMVN),f)
    configLog('Splice N Frames:{}'.format(args.splice),f)
    configLog('Add N Deltas:{}'.format(args.delta),f)
    configLog('Normalize Chunk:{}'.format(args.normalizeChunk),f)
    configLog('Normalize AMP:{}'.format(args.normalizeAMP),f)
    configLog('Decode Minimum Active:{}'.format(args.minActive),f)
    configLog('Decode Maximum Active:{}'.format(args.maxActive),f)
    configLog('Decode Maximum Memory:{}'.format(args.maxMemory),f)  
    configLog('Decode Beam:{}'.format(args.beam),f)   
    configLog('Decode Lattice Beam:{}'.format(args.latBeam),f) 
    configLog('Decode Acoustic Weight:{}'.format(args.acwt),f) 
    configLog('Decode minimum Language Weight:{}'.format(args.minLmwt),f) 
    configLog('Decode maximum Language Weight:{}'.format(args.maxLmwt),f) 
    f.close() 

    print("\n############## Decode Test ##############")

    #------------------ STEP 1: Load Pretrained Model ------------------

    print('Load Model...')
    # Initialize model
    featDim = 40
    if args.delta>0:
        featDim *= (args.delta + 1)
    if args.splice > 0:
        featDim *= ( 2 * args.splice + 1 ) 
    model = LSTM(featDim, outDimPdf, outDimPho)
    chainer.serializers.load_npz(args.preModel,model)
    if args.gpu >=0:
        model.to_gpu(args.gpu)

    #------------------ STEP 2: Prepare Test Data ------------------

    print('Prepare decode test data...')
    # Fmllr file
    testFilePath = args.TIMITpath + '/data-fmllr-tri3/test/feats.scp'
    testFeat = E.load(testFilePath)
    # Use CMVN
    if args.useCMVN:
        testUttSpk = args.TIMITpath + '/data-fmllr-tri3/test/utt2spk'
        testCmvnState = args.TIMITpath + '/data-fmllr-tri3/test/cmvn.ark'
        testFeat = E.use_cmvn(testFeat,testCmvnState,testUttSpk)
    # Add delta
    if args.delta > 0:  
        testFeat = E.add_delta(testFeat,args.delta)
    # Splice frames
    if args.splice > 0:   
        testFeat = testFeat.splice(args.splice)
    # Transform to array
    testFeat = testFeat.array
    # Normalize
    if args.normalizeChunk:   
        testFeat = testFeat.normalize()
    # Normalize acoustic model output
    if args.normalizeAMP:
        # Compute pdf counts in order to normalize acoustic model posterior probability.
        countFile = args.outDir+'/pdfs_counts.txt'
        # Get statistics file
        if not os.path.isfile(countFile):
            trainAliFile = args.TIMITpath + '/exp/dnn4_pretrain-dbn_dnn_ali/ali.*.gz'
            _ = E.analyze_counts(aliFile=trainAliFile,outFile=countFile)
        with open(countFile) as f:
            line = f.readline().strip().strip("[]").strip()
        # Get AMP bias value
        counts = np.array(list(map(float,line.split())),dtype=np.float32)
        normalizeBias = np.log(counts/np.sum(counts))
    else:
        normalizeBias = 0

    #------------------ STEP 3: Decode  ------------------

    temp = E.KaldiDict()
    with chainer.using_config('train',False),chainer.no_backprop_mode():
        for i,utt in enumerate(testFeat.keys()):
            data = [cp.array(testFeat[utt],dtype=cp.float32)]
            out1,out2 = model(data)
            out = F.log_softmax(out1,axis=1)
            out.to_cpu()
            temp[utt] = out.array - normalizeBias
            print('Compute Test WER: Forward network {}/{}'.format(i,testFeat.lens[0]),end=" "*20+'\r')
    # Tansform KaldiDict to KaldiArk format
    print('Compute Test WER: Transform to ark',end=" "*20+'\r')
    amp = temp.ark
    # Decode and obtain lattice
    hmm = args.TIMITpath + '/exp/dnn4_pretrain-dbn_dnn_ali_test/final.mdl'
    hclg = args.TIMITpath + '/exp/tri3/graph/HCLG.fst'
    lexicon = args.TIMITpath + '/exp/tri3/graph/words.txt'
    print('Compute Test WER: Generate Lattice',end=" "*20+'\r')
    lattice = E.decode_lattice(amp,hmm,hclg,lexicon,args.minActive,args.maxActive,args.maxMemory,args.beam,args.latBeam,args.acwt)
    # Change language weight from 1 to 10, get the 1best words.
    print('Compute Test WER: Get 1Best',end=" "*20+'\r')
    outs = lattice.get_1best(lmwt=args.minLmwt,maxLmwt=args.maxLmwt,outFile=args.outDir+'/outRaw.txt')

    #------------------ STEP 4: Score  ------------------

    # If reference file is not existed, make it.
    phonemap = args.TIMITpath + '/conf/phones.60-48-39.map'
    outFilter = args.TIMITpath + '/local/timit_norm_trans.pl -i - -m {} -from 48 -to 39'.format(phonemap)       
    if not os.path.isfile(args.outDir+'/test_filt.txt'):
        refText = args.TIMITpath + '/data/test/text'
        cmd = 'cat {} | {} > {}/test_filt.txt'.format(refText,outFilter,args.outDir)
        (_,_) = E.run_shell_cmd(cmd)
    # Score WER and find the smallest one.
    print('Compute Test WER: compute WER',end=" "*20+'\r')
    minWER = (None,None)
    for k in range(1,11,1):
        cmd = 'cat {} | {} > {}/tanslation_{}.txt'.format(outs[k],outFilter,args.outDir,k)
        (_,_) = E.run_shell_cmd(cmd)
        os.remove(outs[k])
        score = E.wer('{}/test_filt.txt'.format(args.outDir),"{}/tanslation_{}.txt".format(args.outDir,k),mode='all')
        if minWER[0] == None or score['WER'] < minWER[0]:
            minWER = (score['WER'],k)
    
    print("Best WER:{}％ at {}/tanslation_{}.txt".format(minWER[0],args.outDir,k))

if __name__ == "__main__":
    
    if args.decodeMode:
        decode_test()
    else:
        train_model()






