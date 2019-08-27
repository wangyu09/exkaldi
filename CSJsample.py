############# Version Information #############
# PythonKaldi V1.6
# WangYu, University of Yamanashi 
# August, 24
###############################################

import pythonkaldi as PK
import chainer
import chainer.functions as F
import chainer.links as L
import random
import numpy as np
import cupy as cp
import os
from chainer.training import extensions
import time
import threading
import copy
import queue
import sys

CSJpath = '/misc/Work18/wangyu/kaldi/egs/csj/demo1'

class MLPUpdater(chainer.training.StandardUpdater):
    def __init__(self,*args,**kwargs):
        self.supporter = kwargs.pop('supporter')
        super(MLPUpdater,self).__init__(*args,**kwargs)
        
    def convert(self,batch):
        batch = cp.array(batch,dtype=cp.float32)
        data = batch[:,0:-1]
        label = cp.array(batch[:,-1],dtype=cp.int32)
        return data,label

    def loss_fun(self,y,t):
        loss = F.softmax_cross_entropy(y,t)
        acc = F.accuracy(F.softmax(y,axis=1),t)
        self.supporter.send_report({'epoch':self.epoch,'train_loss':loss,'train_acc':acc})
        return loss

    def update_core(self):
        optimizer = self.get_optimizer('main')
        model = optimizer.target
        batch = self.get_iterator('main').next()
        data,label = self.convert(batch)

        with chainer.using_config('Train',True):
            logits = model(data)
        optimizer.update(self.loss_fun,logits,label)

@chainer.training.make_extension()
class MLPEvaluator(chainer.Chain):
    def __init__(self,data,model,supporter,device=0):
        super(MLPEvaluator,self).__init__()
        with self.init_scope():
            self.model = model
            self.data = data
            self.gpu = device
            self.supporter = supporter

    def convert(self,batch):
        batch = cp.array(batch,dtype=cp.float32)
        data = batch[:,0:-1]
        label = cp.array(batch[:,-1],dtype=cp.int32)
        return data,label

    def loss_fun(self,y,t):
        loss = F.softmax_cross_entropy(y,t)
        acc = F.accuracy(F.softmax(y,axis=1),t)
        self.supporter.send_report({'test_loss':loss,'test_acc':acc})
        return loss

    def __call__(self,trainer):
        while True:
            batchdata = self.data.next()
            data,label = self.convert(batchdata)
            with chainer.using_config('train',False),chainer.no_backprop_mode():
                logits = self.model(data)
                loss = self.loss_fun(logits,label)
            if self.data.epochOver:
                break
        self.supporter.collect_report(plot=True)
        self.supporter.save_model(models={'MLP':self.model},iterSymbol=self.data.epoch-1,byKey='test_acc',maxValue=True)

def train_model():

    print("\n############## Now train acoustic model ##############")
    random.seed(1234)
    np.random.seed(1234)
    
    batchSize = 128
    epoch = 2
    lr = 0.0002
    gpu = 0
    outDir = 'Result'

    print('\nStep 1: Prepare Data Iterator...')

    def loadChunkData(feat,label):
        # <feat> is KaldiArk and <label> is KaldiDict
        global CSJpath
        uttSpk = CSJpath + '/data/train/utt2spk'
        cmvnState = CSJpath + '/data/train/cmvn.scp'
        # Apply CMVN
        feat = PK.use_cmvn(feat,cmvnState,uttSpk)  
        # Add 2 orders delta  
        feat = PK.add_delta(feat)
        # Concat front-behind 5 frames    
        feat = feat.splice(5)
        # Transform it to KaldiDict
        feat = feat.array   
        # Normalization within a chunk data
        feat = feat.normalize()
        # Make pair of data and label   
        datas = feat.concat(label,axis=1)
        # Get trainable data numpy array format 
        datas,_ = datas.merge()
        return datas

    global CSJpath
    scpFile = CSJpath + '/data/train/test.scp'
    aliFile = CSJpath + '/exp/tri4/ali.*.gz'
    # Get train data iterator 
    train = PK.DataIterator(scpFile,batchSize,chunks='auto',processFunc=loadChunkData,labelOrAliFiles=aliFile,validDataRatio=0.05)
    # Get vali data iterator 
    vali = train.getValiData()

    print('\nStep 2: Prepare Model...')

    modelConfig = {'inputdim':429,
                    'node':[1095,1095,1095,1095,1095,1095,9288],
                    'acfunction':['relu','relu','relu','relu','relu','relu','log_softmax'],
                    'batchnorm':[True,True,True,True,True,True,False],
                    'layernorm':[False,False,False,False,False,False,False],
                    'dropout':[0.15,0.15,0.15,0.15,0.15,0.15,0.0],
                    }

    model = PK.MLP(modelConfig)
    model.to_gpu(gpu)

    print('\nStep 3: Prepare Trainer...')

    optimizer = chainer.optimizers.Adam(lr)
    optimizer.setup(model)

    supporter = PK.Supporter(outDir)

    updater = MLPUpdater(train, optimizer, supporter=supporter,device=gpu)

    trainer = chainer.training.Trainer(updater, (epoch,'epoch'), out=outDir)

    trainer.extend(MLPEvaluator(vali, model, supporter=supporter,device=gpu),trigger=(1,'epoch'))

    trainer.extend(extensions.ProgressBar())

    print('\nStep 4: Start Training')
    trainer.run()

    print("\nAcoustic Model training done.")
    return modelConfig,supporter.finalModel["MLP"]

modelConfig,pretrainedModel = train_model()

def recognize_test(modelConfig,pretrainedModel):

    print("\n############## Now do the recognize test ##############")

    random.seed(1234)
    np.random.seed(1234)

    outDir = 'Result'

    print('\nStep 1: Load pretrained acoustic model')
    model = PK.MLP(modelConfig)
    chainer.serializers.load_npz(pretrainedModel,model)

    print('\nStep 2: Process mfcc feat to recognized result')

    global CSJpath
    filePath = CSJpath + '/data/eval1/feats.scp'
    # Because if load all of eval1 data at one time to CPU memory, It maybe result in memorr error.
    # So we split it firstly.
    fileList = PK.split_file(filePath,chunks=4)

    lattice = PK.KaldiLattice()

    for i,scpFile in enumerate(fileList,start=1):

        print('({}/{}) File:'.format(i,len(fileList)),scpFile)
        feat = PK.load(scpFile,useSuffix='scp')

        uttSpk = CSJpath + '/data/eval1/utt2spk'
        cmvnState = CSJpath + '/data/eval1/cmvn.scp'
        print("({}/{}) Apply CMVN".format(i,len(fileList)))
        feat = PK.use_cmvn(feat,cmvnState,uttSpk)  
        print("({}/{}) Add 2 orders delta".format(i,len(fileList)))  
        feat = PK.add_delta(feat)
        print("({}/{}) Concat front-behind 5 frames".format(i,len(fileList)))    
        feat = feat.splice(5)
        print("({}/{}) Transform it to KaldiDict".format(i,len(fileList)))
        feat = feat.array   
        print("({}/{}) Normalization within a chunk data".format(i,len(fileList)))
        feat = feat.normalize() 

        temp = PK.KaldiDict()
        with chainer.using_config('train',False),chainer.no_backprop_mode():
            for j,utt in enumerate(feat.keys(),start=1):
                print("({}/{}) Forward nework: {}/{}".format(i,len(fileList),j,len(feat.keys())),end='\r')
                data = np.array(feat[utt],dtype=np.float32)
                out = model(data)
                temp[utt] = out.array
        print()

        print('({}/{}) Transform model output to ark'.format(i,len(fileList)))
        amp = temp.ark

        print('({}/{}) Now start to decode'.format(i,len(fileList)))

        hmm = CSJpath + '/exp/dnn5b_pretrain-dbn_dnn/final.mdl'
        hclg = CSJpath + '/exp/tri4/graph_csj_tg/HCLG.fst'
        lexicon = CSJpath + '/exp/tri4/graph_csj_tg/words.txt'

        print('({}/{}) Gennerate lattice'.format(i,len(fileList)))
        lattice += PK.decode_lattice(amp,hmm,hclg,lexicon,Acwt=0.2,maxThreads=3)

        os.remove(scpFile)

    print('Generate all lattices done. Now get 1-bests words from final lattice.')
    outs = lattice.get_1best_words(minLmwt=1,maxLmwt=15,outDir=outDir,asFile='outRaw.txt')

    print('\nStep 3: Score by different language model scales') 
    refText = CSJpath + '/exp/tri4/decode_eval1_csj/scoring_kaldi/test_filt.txt'
    for k in range(1,15,1):
        
        cmd = 'bash outFilter.sh {} {}/test_prediction_filt.txt'.format(outs[k],outDir)
        (_,_) = PK.run_shell_cmd(cmd)
        os.remove(outs[k])

        score = PK.compute_wer("{}/test_prediction_filt.txt".format(outDir),refText)

        print("Lmwt:{} %WER:{}".format(k,score['WER']))

recognize_test(modelConfig,pretrainedModel)  

def OnlineRecognize(modelConfig,pretrainedModel):

    print('I am sorry that Speek Client Section is unable to use now.')
    exit(1)
    random.seed(1234)
    np.random.seed(1234)

    model = PK.MLP(modelConfig)
    chainer.serializers.load_npz(pretrainedModel,model)
    # If use cpu, it will result in thread lock while using recognizeParallel in chainer model 
    model.to_gpu(0)
    print('Prepare Done')

    def recoFunc(tid,waveFile,AMmodel):

        feat = PK.compute_mfcc(waveFile)
        feat = PK.use_cmvn_sliding(feat)       
        feat = PK.add_delta(feat) 
        feat = feat.splice(5)
        feat = feat.array   
        feat = feat.normalize()
        feat,uttLens = feat.merge()

        with chainer.using_config('train',False),chainer.no_backprop_mode():
            feat = cp.array(feat,dtype=cp.float32)
            out = model(feat)
            out.to_cpu()
            out = out.array

        temp = PK.KaldiDict()
        temp.remerge(out,uttLens)

        amp = temp.ark

        hmm = '/misc/Work18/wangyu/kaldi/egs/csj/demo1/exp/dnn5b_pretrain-dbn_dnn/final.mdl'
        hclg = '/misc/Work18/wangyu/kaldi/egs/csj/demo1/exp/tri4/graph_csj_tg/HCLG.fst'
        lexicon = '/misc/Work18/wangyu/kaldi/egs/csj/demo1/exp/tri4/graph_csj_tg/words.txt'
        lattice = PK.decode_lattice(amp,hmm,hclg,lexicon)
        out = lattice.get_1best_words(11)
        out = out[11][0].split(maxsplit=1)

        if len(out) < 2:
            out = '<sil>'
        else:
            out = out[1]

        return out

    with PK.RemoteServer(bindHost='192.168.1.2') as rs:
        rs.connectFrom(targetHost='192.168.1.192')
        rs.receive()
        rs.recognizeParallel(recoFunc=testFunc,recoFuncArgs=(model,),secPerReco=0.5,maxThreads=3,silSymbol='<sil>')
        rs.send()
        rs.wait()

#OnlineRecognize(modelConfig,pretrainedModel)

