############# Version Information #############
# PythonKaldi V1.6
# WangYu, University of Yamanashi 
# Sep 13, 2019
###############################################

from __future__ import print_function
import pythonkaldi as PK

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

import random
import numpy as np, cupy as cp
import os, datetime
import argparse

## ------ Parameter Configure -----
parser = argparse.ArgumentParser(description='DNN Acoustic model on TIMIT corpus')

parser.add_argument('--TIMITpath', '-t', type=str, default='/misc/Work18/wangyu/kaldi/egs/timit/demo', help='Kaldi timit rescipe folder')
parser.add_argument('--randomSeed', '-r', type=int, default=1234, help='Random seed')
parser.add_argument('--batchSize', '-b', type=int, default=128)
parser.add_argument('--gpu', '-g', type=int, default=0, help='Gpu id (We defaultly use gpu)')
parser.add_argument('--epoch', '-e', type=int, default=27)
parser.add_argument('--outDir','-o',type=str,default='TIMIT_DNN_fmllr_exp')

parser.add_argument('--useCMVN', '-u', type=bool, default=False,)
parser.add_argument('--splice', '-s', type=int, default=10)
parser.add_argument('--delta', '-d', type=int, default=2)
parser.add_argument('--normalize', '-n', type=bool, default=True)

args = parser.parse_args()

assert args.gpu >= 0, 'We will use gpu so it is not expected a negative value.'
if args.outDir.endswith('/'):
    args.outDir = args.outDir[0:-1]

print("\n############## Parameters Configure ##############")   
print('Random seed:',args.randomSeed)
print('Batch Size:',args.batchSize)
print('GPU:',args.gpu)
print('Epoch:',args.epoch)
print('Output Folder:',args.outDir)
print('Use CMVN:',args.useCMVN)
print('Splice N Frames:',args.splice)
print('Add N Deltas:',args.delta)
print('Normalize Dataset:',args.normalize)

## ------ Fix random seed -----
random.seed(args.randomSeed)
np.random.seed(args.randomSeed)
cp.random.seed(args.randomSeed)
chainer.configuration.config.deterministic = True
chainer.configuration.config.cudnn_deterministic = True

## ------ Define model/updater/evaluator -----
class MLP(chainer.Chain):
    def __init__(self,inputDim):
        super(MLP,self).__init__()
        with self.init_scope():

            self.ln1 = L.Linear(inputDim,1024,nobias=True)
            self.bn1 = L.BatchNormalization(1024,0.95)

            self.ln2 = L.Linear(1024,1024,nobias=True)
            self.bn2 = L.BatchNormalization(1024,0.95)

            self.ln3 = L.Linear(1024,1024,nobias=True)
            self.bn3 = L.BatchNormalization(1024,0.95) 

            self.ln4 = L.Linear(1024,1024,nobias=True)
            self.bn4 = L.BatchNormalization(1024,0.95) 

            self.ln5 = L.Linear(1024,1024,nobias=True)
            self.bn5 = L.BatchNormalization(1024,0.95) 

            self.ln7 = L.Linear(1024,1968) 
            self.ln8 = L.Linear(1024,49)

    def __call__(self,x):

        h = F.dropout(F.relu(self.bn1(self.ln1(x))),0.15)
        h = F.dropout(F.relu(self.bn2(self.ln2(h))),0.15)
        h = F.dropout(F.relu(self.bn3(self.ln3(h))),0.15)
        h = F.dropout(F.relu(self.bn4(self.ln4(h))),0.15)
        h = F.dropout(F.relu(self.bn5(self.ln5(h))),0.15)

        h1 = self.ln7(h)
        h2 = self.ln8(h)

        return h1,h2

class MLPUpdater(chainer.training.StandardUpdater):
    def __init__(self,*args,**kwargs):
        self.supporter = kwargs.pop('supporter')
        super(MLPUpdater,self).__init__(*args,**kwargs)
        
    def convert(self,batch):
        batch = cp.array(batch,dtype=cp.float32)
        data = batch[:,0:-2]
        label1 = cp.array(batch[:,-2],dtype=cp.int32)
        label2 = cp.array(batch[:,-1],dtype=cp.int32)
        return data,label1,label2

    def loss_fun(self,y1,y2,t1,t2):
        L1 = F.softmax_cross_entropy(y1,t1)
        L2 = F.softmax_cross_entropy(y2,t2)
        loss = L1 + L2
        acc = F.accuracy(F.softmax(y1,axis=1),t1)
        self.supporter.send_report({'epoch':self.epoch,'train_loss':loss,'train_acc':acc})
        return loss

    def update_core(self):
        optimizer = self.get_optimizer('main')
        model = optimizer.target

        batch = self.get_iterator('main').next()
        data,label1,label2 = self.convert(batch)

        with chainer.using_config('Train',True):
            h1,h2 = model(data)

        optimizer.update(self.loss_fun,h1,h2,label1,label2)

@chainer.training.make_extension()
class MLPEvaluator(chainer.Chain):
    def __init__(self,data,model,supporter,optimizer,outDir,lr,device=0):
        super(MLPEvaluator,self).__init__()
        with self.init_scope():

            self.model = model
            self.data = data
            self.gpu = device
            self.supporter = supporter
            self.optimizer = optimizer
            self.outDir = outDir
            self.lr = lr

            ## Prepare test feature data.
            global args
            filePath = args.TIMITpath + '/data-fmllr-tri3/test/feats.scp'
            feat = PK.load(filePath)
            if args.useCMVN:
                uttSpk = args.TIMITpath + '/data-fmllr-tri3/test/utt2spk'
                cmvnState = args.TIMITpath + '/data-fmllr-tri3/test/cmvn.ark'
                feat = PK.use_cmvn(feat,cmvnState,uttSpk)
            if args.delta > 0:  
                feat = PK.add_delta(feat,args.delta)
            if args.splice > 0:   
                feat = feat.splice(args.splice)
            feat = feat.array
            if args.normalize:   
                self.feat = feat.normalize()
            else:
                self.feat = feat               

    def convert(self,batch):
        batch = cp.array(batch,dtype=cp.float32)
        data = batch[:,0:-2]
        label1 = cp.array(batch[:,-2],dtype=cp.int32)
        label2 = cp.array(batch[:,-1],dtype=cp.int32)
        return data,label1,label2

    def loss_fun(self,y1,y2,t1,t2):
        L1 = F.softmax_cross_entropy(y1,t1)
        L2 = F.softmax_cross_entropy(y2,t2)
        loss = L1 + L2
        acc = F.accuracy(F.softmax(y1,axis=1),t1)
        self.supporter.send_report({'dev_loss':loss,'dev_acc':acc})
        return loss

    def wer_fun(self,model,outDir):

        temp = PK.KaldiDict()

        with chainer.using_config('train',False),chainer.no_backprop_mode():
            for utt in self.feat.keys():
                data = cp.array(self.feat[utt],dtype=cp.float32)
                out1,out2 = model(data)
                out = F.log_softmax(out1,axis=1)
                out.to_cpu()
                temp[utt] = out.array

        amp = temp.ark

        global args
        hmm = args.TIMITpath + '/exp/dnn4_pretrain-dbn_dnn_ali_test/final.mdl'
        hclg = args.TIMITpath + '/exp/tri3/graph/HCLG.fst'
        lexicon = args.TIMITpath + '/exp/tri3/graph/words.txt'

        lattice = PK.decode_lattice(amp,hmm,hclg,lexicon,Acwt=0.2)

        outs = lattice.get_1best_words(minLmwt=1,maxLmwt=10,outDir=outDir,asFile='outRaw.txt')

        phonemap = args.TIMITpath + '/conf/phones.60-48-39.map'
        outFilter = args.TIMITpath + '/local/timit_norm_trans.pl -i - -m {} -from 48 -to 39'.format(phonemap)       
        if not os.path.isfile(outDir+'/test_filt.txt'):
            refText = args.TIMITpath + '/data/test/text'
            cmd = 'cat {} | {} > {}/test_filt.txt'.format(refText,outFilter,outDir)
            (_,_) = PK.run_shell_cmd(cmd)

        minWER = None
        for k in range(1,11,1):
            cmd = 'cat {} | {} > {}/test_prediction_filt.txt'.format(outs[k],outFilter,outDir)
            (_,_) = PK.run_shell_cmd(cmd)
            os.remove(outs[k])
            score = PK.compute_wer('{}/test_filt.txt'.format(outDir),"{}/test_prediction_filt.txt".format(outDir),mode='all')
            if minWER == None or score['WER'] < minWER:
                minWER = score['WER']
        
        self.supporter.send_report({'test_WER':float(minWER)})

    def __call__(self,trainer):
        while True:
            batchdata = self.data.next()
            data,label1,label2 = self.convert(batchdata)
            with chainer.using_config('train',False),chainer.no_backprop_mode():
                h1,h2 = self.model(data)
                loss = self.loss_fun(h1,h2,label1,label2)
            if self.data.epochOver:
                break
        self.wer_fun(self.model,self.outDir)

        self.supporter.send_report({'lr':self.optimizer.lr})
        self.supporter.collect_report(plot=True)
        
        self.supporter.save_model(models={'MLP':self.model},iterSymbol=self.data.epoch-1,byKey='test_wer',maxValue=False)

        if self.supporter.judge('epoch','>=',self.lr[0][0]):
            self.optimizer.lr = self.lr[0][1]
            self.lr.pop(0)

## ------ Train model -----
def train_model():

    print("\n############## Train DNN Acoustic Model ##############")
    print('Start System Time:',datetime.datetime.now().strftime("%Y-%m-%d %X"))

    global args

    if not os.path.isdir(args.outDir):
        os.mkdir(args.outDir)

    print('Prepare Data Iterator...')
    # Feature data
    trainScpFile = args.TIMITpath + '/data-fmllr-tri3/train/feats.scp'
    devScpFile = args.TIMITpath + '/data-fmllr-tri3/dev/feats.scp'

    # Label
    trainAliFile = args.TIMITpath + '/exp/dnn4_pretrain-dbn_dnn_ali/ali.*.gz'
    trainHmm = args.TIMITpath + '/exp/dnn4_pretrain-dbn_dnn_ali/final.mdl'

    devAliFile = args.TIMITpath + '/exp/dnn4_pretrain-dbn_dnn_ali_dev/ali.*.gz'
    devHmm = args.TIMITpath + '/exp/dnn4_pretrain-dbn_dnn_ali_dev/final.mdl'

    trainLabelPdf = PK.get_ali(trainAliFile,trainHmm)
    trainLabelPho = PK.get_ali(trainAliFile,trainHmm,True) 

    devLabelPdf = PK.get_ali(devAliFile,devHmm)
    devLabelPho = PK.get_ali(devAliFile,devHmm,True) 
    
    # Process function
    def loadTrainChunkData(feat):
        # <feat> is KaldiArk
        global args
        # use CMVN
        if args.useCMVN:
            uttSpk = args.TIMITpath + '/data-fmllr-tri3/train/utt2spk'
            cmvnState = args.TIMITpath + '/data-fmllr-tri3/train/cmvn.ark'
            feat = PK.use_cmvn(feat,cmvnState,uttSpk)
        # Add delta
        if args.delta > 0:  
            feat = PK.add_delta(feat,args.delta)
        # Splice front-back n frames
        if args.splice > 0:   
            feat = feat.splice(args.splice)
        # Transform to KaldiDict
        feat = feat.array
        # Normalize
        if args.normalize:   
            feat = feat.normalize()
        # Concatenate label           
        datas = feat.concat([trainLabelPdf,trainLabelPho],axis=1)
        # Transform trainable numpy data
        datas,_ = datas.merge()
        return datas

    def loadDevChunkData(feat):
        # <feat> is KaldiArk
        global args
        # use CMVN
        if args.useCMVN:
            uttSpk = args.TIMITpath + '/data-fmllr-tri3/dev/utt2spk'
            cmvnState = args.TIMITpath + '/data-fmllr-tri3/dev/cmvn.ark'
            feat = PK.use_cmvn(feat,cmvnState,uttSpk)
        # Add delta
        if args.delta > 0:  
            feat = PK.add_delta(feat,args.delta)
        # Splice front-back n frames
        if args.splice > 0:   
            feat = feat.splice(args.splice)
        # Transform to KaldiDict
        feat = feat.array
        # Normalize
        if args.normalize:   
            feat = feat.normalize()
        # Concatenate label           
        datas = feat.concat([devLabelPdf,devLabelPho],axis=1)
        # Transform trainable numpy data
        datas,_ = datas.merge()
        return datas

    # Prepare data iterator
    train = PK.DataIterator(trainScpFile,args.batchSize,chunks=5,processFunc=loadTrainChunkData,validDataRatio=0)
    print('Generate train dataset done. Chunks:{} / Batch size:{}'.format(train.chunks,train.batch_size))
    dev = PK.DataIterator(devScpFile,args.batchSize,chunks='auto',processFunc=loadDevChunkData,validDataRatio=0)
    print('Generate validation dataset done. Chunks:{} / Batch size:{}.'.format(dev.chunks,dev.batch_size))

    print('Prepare Model...')

    featDim = 40
    if args.delta>0:
        featDim *= (args.delta + 1)
    if args.splice > 0:
        featDim *= ( 2 * args.splice + 1 ) 
    model = MLP(featDim)
    model.to_gpu(args.gpu)

    print('Prepare Chainer Trainer...')

    lr = [(0,0.08),(10,0.04),(15,0.02),(17,0.01),(19,0.005),(22,0.0025),(25,0.001)]
    optimizer = chainer.optimizers.MomentumSGD(lr[0][1],momentum=0.0)
    lr.pop(0)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(0.0))

    supporter = PK.Supporter(args.outDir)

    updater = MLPUpdater(train, optimizer, supporter=supporter,device=args.gpu)

    trainer = chainer.training.Trainer(updater, (args.epoch,'epoch'), out=args.outDir)

    trainer.extend(MLPEvaluator(dev,model,supporter,optimizer,args.outDir,lr,args.gpu),trigger=(1,'epoch'))

    trainer.extend(extensions.ProgressBar())
    # While first epoch, the epoch size is computed gradually, so the prograss information will be inaccurate. 

    print('Now Start to Train')
    print('Note that: The first epoch will be doing the statistics of total data size gradually.')
    print('           So the information is not reliable.')
    print('Note that: We will evaluate the WER of test dataset every epoch that will cost a few minutes.')
    trainer.run()

    print("DNN Acoustic Model training done.")
    print("The final model has been saved as:",supporter.finalModel["MLP"])
    print('Over System Time:',datetime.datetime.now().strftime("%Y-%m-%d %X"))

    return supporter.finalModel["MLP"]

pretrainedModel = train_model()

## ------ Decode testing -----
def decode_test(pretrainedModel):

    print("\n############## Now do the decode test ##############")

    global args

    print('Load pretrained acoustic model')
    featDim = 40
    if args.delta>0:
        featDim *= (args.delta + 1)
    if args.splice > 0:
        featDim *= ( 2 * args.splice + 1 ) 
    model = MLP(featDim)

    chainer.serializers.load_npz(pretrainedModel,model)
    print(pretrainedModel)

    print('Process mfcc feat to recognized result')
    filePath = args.TIMITpath + '/data-fmllr-tri3/test/feats.scp'
    feat = PK.load(filePath)
    if args.useCMVN:
        print('Apply CMVN')
        uttSpk = args.TIMITpath + '/data-fmllr-tri3/test/utt2spk'
        cmvnState = args.TIMITpath + '/data-fmllr-tri3/test/cmvn.ark'
        feat = PK.use_cmvn(feat,cmvnState,uttSpk)
    if args.delta > 0:  
        print('Add {} orders delta to feat'.format(args.delta))
        feat = PK.add_delta(feat,args.delta)
    if args.splice > 0:
        print('Splice front-back {} frames'.format(args.splice))   
        feat = feat.splice(args.splice)
    print('Transform to KaldiDict data')      
    feat = feat.array
    if args.normalize:
        print('Normalize with Mean and STD')   
        feat = feat.normalize()

    temp = PK.KaldiDict()
    with chainer.using_config('train',False),chainer.no_backprop_mode():
        for j,utt in enumerate(feat.keys(),start=1):
            print("Forward nework: {}/{}".format(j,len(feat.keys())),end='\r')
            data = np.array(feat[utt],dtype=np.float32)
            out1,out2 = model(data)
            out = F.log_softmax(out1,axis=1)
            temp[utt] = out.array
    print()

    print('Transform model output to KaldiArk data')
    amp = temp.ark

    hmm = args.TIMITpath + '/exp/dnn4_pretrain-dbn_dnn_ali_test/final.mdl'
    hclg = args.TIMITpath + '/exp/tri3/graph/HCLG.fst'
    lexicon = args.TIMITpath + '/exp/tri3/graph/words.txt'

    print('Gennerate lattice')
    lattice = PK.decode_lattice(amp,hmm,hclg,lexicon,Acwt=0.2)

    print('Get 1-bests words from lattice')
    outs = lattice.get_1best_words(minLmwt=1,maxLmwt=10,outDir=args.outDir,asFile='1best_LMWT')

    print('Score by different language model scales') 
    phonemap = args.TIMITpath + '/conf/phones.60-48-39.map'
    outFilter = args.TIMITpath + '/local/timit_norm_trans.pl -i - -m {} -from 48 -to 39'.format(phonemap)
    if not os.path.isfile(args.outDir+'/test_filt.txt'):
        refText = args.TIMITpath + '/data/test/text'
        cmd = 'cat {} | {} > {}/test_filt.txt'.format(refText,outFilter,args.outDir)
        (_,_) = PK.run_shell_cmd(cmd)

    for k in range(1,11,1):
        cmd = 'cat {} | {} > {}/test_prediction_filt.txt'.format(outs[k],outFilter,args.outDir)
        (_,_) = PK.run_shell_cmd(cmd)
        score = PK.compute_wer('{}/test_filt.txt'.format(args.outDir),"{}/test_prediction_filt.txt".format(args.outDir),mode='all')
        print('LMWT:%2d WER:%.2f'%(k,score['WER']))

#decode_test(pretrainedModel)




