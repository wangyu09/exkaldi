############# Version Information #############
# GRU sample model based on: ExKaldi V0.1.1, pytorch V1.3.0
# WangYu, University of Yamanashi 
# Oct 16, 2019
###############################################

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
import torch
import torch.nn.functional as F
import torch.nn as nn
import random
import numpy as np
import os, datetime
import argparse
import math
import socket
import time

## ------------- Parameter Configure -------------
parser = argparse.ArgumentParser(description='GRU Acoustic model on TIMIT corpus')
parser.add_argument('--decodeMode', type=bool, default=False, help='If false, run train recipe.')
parser.add_argument('--TIMITpath', '-t', type=str, default='kaldi/egs/timit/demo', help='Kaldi timit rescipe folder')
parser.add_argument('--randomSeed', '-r', type=int, default=2234)
parser.add_argument('--batchSize', '-b', type=int, default=8)
parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU id (We defaultly use gpu)')
parser.add_argument('--epoch', '-e', type=int, default=27)
parser.add_argument('--layer', '-l', type=int, default=5)
parser.add_argument('--hiddenNode', '-hn', type=int, default=550)
parser.add_argument('--dropout', '-do', type=float, default=0.2)
parser.add_argument('--outDir','-o', type=str, default='TIMIT_GRU_fmllr_exp')
parser.add_argument('--useCMVN', '-u', type=bool, default=True)
parser.add_argument('--splice', '-s', type=int, default=0)
parser.add_argument('--delta', '-d', type=int, default=0)
parser.add_argument('--normalizeChunk', '-nC', type=bool, default=True)
parser.add_argument('--normalizeAMP', '-nA', type=bool, default=True)
parser.add_argument('--minActive', '-minA', type=int, default=200)
parser.add_argument('--maxActive', '-maxA', type=int, default=7000)
parser.add_argument('--maxMemory', '-maxM', type=int, default=50000000)
parser.add_argument('--beam', '-beam', type=int, default=13)
parser.add_argument('--latBeam', '-latB', type=int, default=8)
parser.add_argument('--acwt', '-acwt', type=float, default=0.2)
parser.add_argument('--minLmwt', '-minL', type=int, default=1)
parser.add_argument('--maxLmwt', '-maxL', type=int, default=15)
parser.add_argument('--preModel', '-p', type=str, default='')
args = parser.parse_args()

if args.outDir.endswith('/'):
    args.outDir = args.outDir[0:-1]
if not os.path.isdir(args.outDir):
    os.mkdir(args.outDir)

## ------ Fix random seed -----
random.seed(args.randomSeed)
np.random.seed(args.randomSeed)
torch.manual_seed(args.randomSeed)

def act_fun(act_type):

    if act_type=="relu":
        return nn.ReLU()
            
    if act_type=="tanh":
        return nn.Tanh()
            
    if act_type=="sigmoid":
        return nn.Sigmoid()
           
    if act_type=="leaky_relu":
        return nn.LeakyReLU(0.2)
            
    if act_type=="elu":
        return nn.ELU()
                     
    if act_type=="softmax":
        return nn.LogSoftmax(dim=1)
        
    if act_type=="linear":
        return nn.LeakyReLU(1) # initializzed like this, but not used in forward!

class GRUblock(nn.Module):
    def __init__(self,inDim,outDim,bi=True,ratio=0.2,act='tanh'):
        super(GRUblock,self).__init__()
            
        #Feed-forward connections
        self.wh = nn.Linear(inDim, outDim, bias=False)
        self.wz = nn.Linear(inDim, outDim, bias=False)
        self.wr = nn.Linear(inDim, outDim, bias=False)

        #Recurrent connections
        self.uh = nn.Linear(outDim, outDim, bias=False)
        self.uz = nn.Linear(outDim, outDim, bias=False)
        self.ur = nn.Linear(outDim, outDim, bias=False)

        nn.init.orthogonal_(self.uh.weight)
        nn.init.orthogonal_(self.uz.weight)
        nn.init.orthogonal_(self.ur.weight)

        #Batch normarlize
        self.bn_wh = nn.BatchNorm1d(outDim, momentum=0.05)
        self.bn_wz = nn.BatchNorm1d(outDim, momentum=0.05)
        self.bn_wr = nn.BatchNorm1d(outDim, momentum=0.05)

        self.bi = bi
        self.inDim = inDim
        self.outDim = outDim
        self.ratio = ratio
        self.act = act_fun(act)

    def flip(self, x, dim):
        xsize = x.size()
        dim = x.dim() + dim if dim < 0 else dim
        x = x.contiguous()
        x = x.view(-1, *xsize[dim:])
        x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
        return x.view(xsize) 

    def __call__(self,x,is_training=True,device=-1):

        if self.bi:
            h_init = torch.zeros(2*x.shape[1], self.outDim)
            x=torch.cat([x,self.flip(x,0)],1)
        else:
            h_init = torch.zeros(x.shape[1],self.outDim)

        if is_training:
            drop_mask=torch.bernoulli(torch.Tensor(h_init.shape[0],h_init.shape[1]).fill_(1-self.ratio))
        else:
            drop_mask=torch.FloatTensor([1-self.ratio])

        if device >= 0:
            h_init=h_init.cuda(device)
            drop_mask=drop_mask.cuda(device)

        wh_out=self.wh(x)
        wz_out=self.wz(x)
        wr_out=self.wr(x)

        wh_out_bn=self.bn_wh(wh_out.view(wh_out.shape[0]*wh_out.shape[1],wh_out.shape[2]))
        wh_out=wh_out_bn.view(wh_out.shape[0],wh_out.shape[1],wh_out.shape[2])
    
        wz_out_bn=self.bn_wz(wz_out.view(wz_out.shape[0]*wz_out.shape[1],wz_out.shape[2]))
        wz_out=wz_out_bn.view(wz_out.shape[0],wz_out.shape[1],wz_out.shape[2])

        wr_out_bn=self.bn_wr(wr_out.view(wr_out.shape[0]*wr_out.shape[1],wr_out.shape[2]))
        wr_out=wr_out_bn.view(wr_out.shape[0],wr_out.shape[1],wr_out.shape[2])        

        # Processing time steps
        hiddens = []
        ht=h_init
        
        for k in range(x.shape[0]):

            # gru equation
            zt=torch.sigmoid(wz_out[k]+self.uz(ht))
            rt=torch.sigmoid(wr_out[k]+self.ur(ht))
            at=wh_out[k]+self.uh(rt*ht)
            hcand=self.act(at)*drop_mask
            ht=(zt*ht+(1-zt)*hcand)
                
            hiddens.append(ht)

        # Stacking hidden states
        h=torch.stack(hiddens)
        
        # Bidirectional concatenations
        if self.bi:
            h_f=h[:,0:int(x.shape[1]/2)]
            h_b=self.flip(h[:,int(x.shape[1]/2):x.shape[1]].contiguous(),0)
            h=torch.cat([h_f,h_b],2)
            
        # Setup x for the next hidden layer
        return h

class GRU(nn.Module):
    def __init__(self,inDim,outDimPdf,outDimPho):
        super(GRU,self).__init__()
        
        global args

        self.layers = args.layer
        self.grus  = nn.ModuleList([])
        self.grus.append(GRUblock(inDim=inDim,outDim=args.hiddenNode,ratio=args.dropout,act='tanh'))

        for i in range(self.layers-1):
            self.grus.append(GRUblock(inDim=args.hiddenNode*2,outDim=args.hiddenNode,ratio=args.dropout,act='tanh'))

        self.ln1 = nn.Linear(args.hiddenNode*2, outDimPdf, bias=True)
        self.ln1.weight = torch.nn.Parameter(torch.Tensor(outDimPdf,args.hiddenNode*2).uniform_(-np.sqrt(0.01/(args.hiddenNode*2+outDimPdf)),np.sqrt(0.01/(args.hiddenNode*2+outDimPdf))))
        self.ln1.bias = torch.nn.Parameter(torch.zeros(outDimPdf))

        self.ln2 = nn.Linear(args.hiddenNode*2, outDimPho, bias=True)
        self.ln2.weight = torch.nn.Parameter(torch.Tensor(outDimPho,args.hiddenNode*2).uniform_(-np.sqrt(0.01/(args.hiddenNode*2+outDimPho)),np.sqrt(0.01/(args.hiddenNode*2+outDimPho))))
        self.ln2.bias = torch.nn.Parameter(torch.zeros(outDimPho))

        self.lastAct = nn.LogSoftmax(dim=1)
        self.outDimPdf = outDimPdf
        self.outDimPho = outDimPho

    def __call__(self,x,is_training=True,device=-1):

        for i in range(self.layers):
            x = self.grus[i](x,is_training,device)

        h1 = self.ln1(x).view([-1,self.outDimPdf])
        h2 = self.ln2(x).view([-1,self.outDimPho])

        return self.lastAct(h1) ,self.lastAct(h2)

## ------ Train model -----
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
    configLog('GRU layers:{}'.format(args.layer),f)
    configLog('GRU hidden nodes:{}'.format(args.hiddenNode),f)
    configLog('GRU dropout:{}'.format(args.dropout),f)
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

    print("\n############## Train GRU Acoustic Model ##############")

    #----------------- STEP 1: Prepare Train Data -----------------
    print('Prepare data iterator...')
    # Fmllr data file
    trainScpFile = args.TIMITpath + '/data-fmllr-tri3/train/feats.scp'    
    devScpFile = args.TIMITpath + '/data-fmllr-tri3/dev/feats.scp'
    # Alignment label file
    trainAliFile = args.TIMITpath + '/exp/dnn4_pretrain-dbn_dnn_ali/ali.*.gz'
    devAliFile = args.TIMITpath + '/exp/dnn4_pretrain-dbn_dnn_ali_dev/ali.*.gz'
    # Load label
    trainLabelPdf = E.load_ali(trainAliFile)
    trainLabelPho = E.load_ali(trainAliFile,returnPhone=True)
    for i in trainLabelPho.keys():
        trainLabelPho[i] = trainLabelPho[i] - 1
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
        feat = feat.array.sort(by='frame')
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
    train = E.DataIterator(trainScpFile,loadChunkData,args.batchSize,chunks=5,shuffle=False,otherArgs=(trainUttSpk,trainCmvnState,trainLabelPdf,trainLabelPho,'train'))
    print('Generate train dataset. Chunks:{} / Batch size:{}'.format(train.chunks,train.batchSize))
    dev = E.DataIterator(devScpFile,loadChunkData,args.batchSize,chunks=1,shuffle=False,otherArgs=(devUttSpk,devCmvnState,devLabelPdf,devLabelPho,'dev'))
    print('Generate validation dataset. Chunks:{} / Batch size:{}.'.format(dev.chunks,dev.batchSize))
    print("Done.")

    print('Prepare model...')
    featDim = 40
    if args.delta > 0:
        featDim *= (args.delta + 1)
    if args.splice > 0:
        featDim *= (2*args.splice + 1)
    model = GRU(featDim, trainLabelPdf.target, trainLabelPho.target)
    lossfunc = nn.NLLLoss()
    if args.gpu >= 0:
        model = model.cuda(args.gpu)
        lossfunc = lossfunc.cuda(args.gpu)
    print('Generate model done.')  

    print('Prepare optimizer and supporter...')
    #lr = [(0,0.5),(8,0.25),(13,0.125),(15,0.07),(17,0.035),(20,0.02),(23,0.01)]
    lr = [(0,0.0004)]
    print('Learning Rate:',lr)
    optimizer = torch.optim.RMSprop(model.parameters(),lr=lr[0][1],alpha=0.95, eps=1e-8, weight_decay=0, momentum=0, centered=False)
    lr.pop(0)
    supporter = E.Supporter(args.outDir)
    print('Done.')

    print('Prepare test data...')
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
        # compute pdf counts in order to normalize acoustic model posterior probability.
        countFile = args.outDir+'/pdfs_counts.txt'
        if not os.path.isfile(countFile):
            _ = E.analyze_counts(aliFile=trainAliFile,outFile=countFile)
        with open(countFile) as f:
            line = f.readline().strip().strip("[]").strip()
        counts = np.array(list(map(float,line.split())),dtype=np.float32)
        normalizeBias = np.log(counts/np.sum(counts))
    else:
        normalizeBias = 0
    print('Done.')

    print('Prepare test data decode and score function...')
    # Design a function to compute WER of test data
    def wer_fun(model,feat,normalizeBias):
        global args
        # Tranform the formate of KaldiDict feature data in order to forward network 
        temp = E.KaldiDict()
        utts = feat.utts
        with torch.no_grad():
            for index,utt in enumerate(utts):
                data = torch.Tensor(feat[utt][:,np.newaxis,:])
                data = torch.autograd.Variable(data)
                if args.gpu >= 0:
                    data = data.cuda(args.gpu)
                out1,out2 = model(data,is_training=False,device=args.gpu)
                out = out1.cpu().detach().numpy() - normalizeBias
                temp[utt] = out
                print("(testing) Forward network {}/{}".format(index,len(utts)),end=" "*20+'\r')
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
        print('(testing) Score',end=" "*20+'\r')
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
    print('Done.')

    print('Now Start to Train')
    for e in range(args.epoch):
        print()
        i = 0
        usedTime = 0
        supporter.send_report({'epoch':e})
        # Train
        model.train()
        while True:
            start = time.time()
            # Get batch data and label
            batch = train.next()
            batch,lengths = E.pad_sequence(batch,shuffle=True,pad=0)
            batch = torch.Tensor(batch)
            data,label1,label2 = batch[:,:,0:-2],batch[:,:,-2],batch[:,:,-1]
            data = torch.autograd.Variable(data)
            label1 = torch.autograd.Variable(label1).view(-1).long()
            label2 = torch.autograd.Variable(label2).view(-1).long()
            # Send to GPU if use
            if args.gpu >= 0:
                data = data.cuda(args.gpu)
                label1 = label1.cuda(args.gpu) 
                label2 = label2.cuda(args.gpu) 
            # Clear grad
            optimizer.zero_grad()
            # Forward model
            out1,out2 = model(data,is_training=True,device=args.gpu)
            # Loss back propagation
            loss1 = lossfunc(out1,label1)
            loss2 = lossfunc(out2,label2)
            loss = loss1 + loss2
            loss.backward()
            # Update parameter
            optimizer.step()
            # Compute accuracy
            pred=torch.max(out1,dim=1)[1]
            acc = 1 - torch.mean((pred!=label1).float())
            # Record train information  
            supporter.send_report({'train_loss':float(loss),'train_acc':float(acc)})
            ut = time.time() - start
            usedTime += ut
            batchLoss = float(loss.cpu().detach().numpy())
            print("(training) Epoch:{}/{}% Chunk:{}/{}% Iter:{} Used-time:{}s Batch-loss:{} Speed:{}iters/s".format(e,int(100*train.epochProgress),train.chunk,int(100*train.chunkProgress),i,int(usedTime),"%.4f"%(batchLoss),"%.2f"%(1/ut)), end=" "*5+'\r')
            i += 1
            # If forwarded all data, break
            if train.isNewEpoch:
                break
        # Evaluation
        model.eval()
        with torch.no_grad():
            while True:
                start = time.time()
                # Get batch data and label
                batch = dev.next()
                batch,lengths = E.pad_sequence(batch,shuffle=True,pad=0)
                maxLen, bSize, _ = batch.shape
                batch = torch.Tensor(batch)
                data,label1,label2 = batch[:,:,0:-2],batch[:,:,-2],batch[:,:,-1]
                data = torch.autograd.Variable(data)
                label1 = torch.autograd.Variable(label1).view(-1).long()
                # Send to GPU if use
                if args.gpu >= 0:
                    data = data.cuda(args.gpu) 
                    label1 = label1.cuda(args.gpu) 
                # Forward model
                out1,out2 = model(data,is_training=False,device=args.gpu)
                # Compute accuracy of padded label
                pred = torch.max(out1,dim=1)[1]
                acc_pad = 1 - torch.mean((pred!=label1).float())
                # Compute accuracy of not padded label. This should be more correct. 
                label = label1.cpu().numpy().reshape([maxLen,bSize])
                pred =  pred.cpu().numpy().reshape([maxLen,bSize]) 
                label = E.unpack_padded_sequence(label,lengths)
                pred = E.unpack_padded_sequence(pred,lengths)
                acc_nopad = E.accuracy(pred,label)
                # Record evaluation information 
                supporter.send_report({'dev_acc_pad':float(acc_pad),'dev_acc_nopad':acc_nopad})
                ut = time.time() - start
                usedTime += ut
                batchLoss = float(loss.cpu().detach().numpy())
                print("(Validating) Epoch:{}/{}% Chunk:{}/{}% Iter:{} Used-time:{}s Batch-loss:{} Speed:{}iters/s".format(e,int(100*dev.epochProgress),dev.chunk,int(100*dev.chunkProgress),i,int(usedTime),"%.4f"%(batchLoss),"%.2f"%(1/ut)), end=" "*5+'\r')
                i += 1
                # If forwarded all data, break
                if dev.isNewEpoch:
                    break  
            print()
            # We compute WER score from 4th epoch
            if e >= 2:
                minWER = wer_fun(model,testFeat,normalizeBias)
                supporter.send_report({'test_wer':minWER})
        # one epoch is over so collect information
        supporter.collect_report(plot=True)
        # Save model
        def saveFunc(archs):
            fileName, model = archs
            torch.save(model.state_dict(), fileName)
        supporter.save_arch(saveFunc,arch={'GRU':model})
        # Change learning rate
        if len(lr) > 0 and supporter.judge('epoch','>=',lr[0][0]):
            for param_group in optimizer.param_groups:
                param_group['lr']=lr[0][1]
            lr.pop(0)
               
    print("GRU Acoustic Model training done.")
    print("The final model has been saved as:",supporter.finalArch["GRU"])
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
    model = GRU(featDim,outDimPdf,outDimPho)
    #chainer.serializers.load_npz(args.preModel,model)
    model.load_state_dict(torch.load(args.preModel))
    if args.gpu >=0:
        model = model.cuda(args.gpu)

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
    with torch.no_grad():
        for index,utt in enumerate(testFeat.utts):
            data = torch.Tensor(testFeat[utt][:,np.newaxis,:])
            data = torch.autograd.Variable(data)
            if args.gpu >= 0:
                data = data.cuda(args.gpu)
            out1,out2 = model(data,is_training=False,device=args.gpu)
            out = out1.cpu().detach().numpy() - normalizeBias
            temp[utt] = out
            print("Computing WER for TEST dataset: Forward network {}/{}".format(index,testFeat.lens[0]),end=" "*20+'\r')
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
    outs = lattice.get_1best(lmwt=args.minLmwt,maxLmwt=args.maxLmwt,outFile=args.outDir+'/outRaw')

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
    
    print("Best WER:{}％ at {}/tanslation_{}.txt".format(minWER[0],args.outDir,minWER[1]))

if __name__ == "__main__":
    
    if args.decodeMode:
        decode_test()
    else:
        train_model()

