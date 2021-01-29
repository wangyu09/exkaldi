# ExKaldi: A Python-based Extension Tool of Kaldi
![exkaldi_ubuntu_build](https://github.com/wangyu09/exkaldi/workflows/exkaldi_ubuntu_build/badge.svg)
================================

ExKaldi automatic speech recognition toolkit is developed to build an interface between [Kaldi ASR toolkit](https://github.com/kaldi-asr/kaldi) and Python. 
Differing from other Kaldi wrappers, ExKaldi have these features:  
1. Integrated APIs to build a ASR systems, including feature extraction, GMM-HMM acoustic model training, N-Grams language model training, decoding and scoring. 
2. ExKaldi provides tools to support train DNN acoustic model with Deep Learning frameworks, such as Tensorflow. 
3. ExKaldi supports CTC decoding.  

The goal of ExKaldi is to help developers build high-performance ASR systems with Python language easily.

## Installation

Current version: 1.3.5.
(We only tested our toolkit on Ubuntu >= 16., python3.6,python3.7,python3.8 with gh-action)

1. If you have not installed Kaldi ASR toolkit, clone the Kaldi ASR toolkit repository firstly (Kaldi version 5.5 is expected.)
```
git clone https://github.com/kaldi-asr/kaldi.git kaldi --origin upstream
```
And follow these three tutorial files to install and compile it.
```
less kaldi/INSTALL
less kaldi/tools/INSTALL
less kaldi/src/INSTALL
```

2. Clone the ExKaldi source code from our github project, then install it.
### Install with pip
```
$ pip install https://github.com/kpu/kenlm/archive/master.zip
$ pip install exkaldi
```
### Install with Source
```
$ git clone https://github.com/wangyu09/exkaldi.git
$ cd exkaldi
$ bash quick_install.sh
```

3. Check if it is installed correctly.
```
python3 -c "import exkaldi"
```

## Tutorial

In [exkaldi/tutorials](tutorials) directory, we prepared a simple tutorial to show how to use ExKaldi APIs to build a ASR system from the scratch.
The data is from librispeech _train\_100\_clean_ dataset. This tutorial includes:
1. Extract and process MFCC feature.  
2. Train and querying a N-grams language model.  
3. Train monophone GMM-HMM, build decision tree, and train triphone GMM-HMM.  
4. Train a DNN acoustic model with Tensorflow.  
5. Compile WFST decoding graph.  
6. Decode based on GMM-HMM and DNN-HMM.  
7. Process lattice and compute WER score.  

This ASR symtem built here is just a dummy model, and we have done some formal experiments in [exkaldi/examples](examples). Check the source code or [documents](https://wangyu09.github.io/exkaldi/#/) to look more information about APIs.

## Experiments

We have done some experiments to test ExKaldi toolkit, and they achieved a good performance.

#### TIMIT

1, The perplexity of various language models. All these systems are trained with TIMIT _train_ dataset and tested with TIMIT _test_ data. The score showed in the table is PPL score.  

|                           | __2-grams__  | __3-grams__ | __4-grams__ | __5-grams__ | __6-grams__ |
| :-----------------------: | :----------: | :---------: | :---------: | :---------: | :---------: |
| __Kaldi baseline irstlm__ | 14.41        | ---         | ---         | ---         | ---         |
| __ExKaldi srilm__         | 14.42        | 13.05       | 13.67       | 14.30       | 14.53       |
| __ExKaldi kenlm__         | 14.39        | 12.75       | 12.75       | 12.70       | __12.25__   |

2, The phone error rate (PER) of various GMM-HMM-based systems. All these systems are trained with TIMIT _train_ dataset and tested with TIMIT _test_ dataset. The Language model backend used in ExKaldi is KenLM. From the results, we can know than KenLm is avaliable to optimize the language model. And what's more, with ExKaldi, we cherry-picked the N-grams model by testing the perplexity and it improved the performance of ASR system.

|                           | __mono__  | __tri1__ | __tri2__ | __tri3__ |
| :-----------------------: | :-------: | :------: | :------: | :------: |
| __Kaldi baseline 2grams__ | 32.54     | 26.17    | 23.63    | 21.54    |
| __ExKaldi 2grams__        | 32.53     | 25.89    | 23.63    | 21.43    |
| __ExKaldi 6grams__        | 29.83     | 24.07    | 22.40    |__20.01__ |

3, The phone error rate (PER) of two DNN-HMM-based systems. We trained our models with Tensorflow 2.3. The version of PyTorch-Kaldi toolkit is 1.0 in our experiment. 

|                    | __DNN__   | __LSTM__ |
| :----------------: | :-------: | :------: |
| __Kaldi baseline__ | 18.67     | ---      | 
| __PyTorch-Kaldi__  | 17.99     | 17.01    |
| __ExKaldi__        | 15.13     | 15.01    |
