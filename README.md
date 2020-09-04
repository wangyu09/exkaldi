# ExKaldi: An advance kaldi wrapper for Python 
[![License](https://img.shields.io/hexpm/l/Apa)](https://github.com/wangyu09/exkaldi/blob/master/LICENSE)
![Developing](https://img.shields.io/badge/Debug-1.3.x-orange)
================================

ExKaldi automatic speech recognition toolkit is designed to build an interface between [Kaldi ASR toolkit](https://github.com/kaldi-asr/kaldi) and Python. 
Differing from other kaldi wrappers, ExKaldi have these features:  
1. Integrated APIs to build a ASR systems, including training HMM-GMM acoustic model, training HMM-DNN acoustic model, training and quering a N-grams language model, decoding and scoring.  
2. ExKaldi C++ library was designed to support, such as ctc decoding for End-to-End.   
3. Use KenLm as languange model backend.

The goal of ExKaldi is to help developers build high-performance ASR systems with Python language easily.

## Installation

(We only tested our toolkit on Ubuntu >= 16.)

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
```
git clone https://github.com/wangyu09/exkaldi.git
cd ExKaldi
bash quick_install.sh
```

3. Check if it is installed correctly.
```
python -c "import exkaldi"
```


## Tutorial

In [exkaldi/tutorials](tutorials) directory, we prepared a simple tutorial to show how to use ExKaldi APIs to build a ASR system from the scratch.
The data is from librispeech train_100_clean dataset. This tutorial includes:
1. Extract and process feature.  
2. Train and querying a N-grams language model.  
3. Train monophone GMM-HMM, build decision tree, triphone GMM-HMM.  
4. Train a DNN acoustic model with tensorflow.  
5. Compile WFST decoding graph.  
6. Decode based on GMM-HMM and DNN-HMM.  
7. Process lattice and compute score WER.  

This ASR symtem built here is just a dummy model, and we have done some formal experiments in [exkaldi/examples](examples). Check the source code to look more information about ExKaldi APIs.

## Experiments

We have done some experiments to test ExKaldi toolkit, and it achived a good performance.
(We will upload the results of experiments little by little.)

#### TIMIT

1, The perplexity of various language models. All these systems are trained with TIMIT train data and tested with TIMIT test data. The score showed in the table is weighted average of all utterances and weights are the sentence length of each utterance.  

|                           | __2-grams__  | __3-grams__ | __4-grams__ | __5-grams__ | __6-grams__ |
| :-----------------------: | :----------: | :---------: | :---------: | :---------: | :---------: |
| __Kaldi baseline irstlm__ | 14.41        | ---         | ---         | ---         | ---         |
| __ExKaldi srilm__         | 14.42        | 13.05       | 13.67       | 14.30       | 14.53       |
| __ExKaldi kenlm__         | 14.39        | 12.75       | 12.75       | 12.70       | __12.25__   |

2, The phone error rate(PER) of various GMM-HMM-based systems. All these systems are trained with TIMIT train dataset and tested with TIMIT test dataset. The Language model backend used in ExKaldi is KenLM. From the results, we can know than KenLm is avaliable to optimize the language model. And what's more, with ExKaldi, we cherry-picked the N-grams by testing the perplexity and it improved the performance of ASR system.

|                           | __mono__  | __tri1__ | __tri2__ | __tri3__ |
| :-----------------------: | :-------: | :------: | :------: | :------: |
| __Kaldi baseline 2grams__ | 32.54     | 26.17    | 23.63    | 21.54    |
| __ExKaldi 2grams__        | 32.53     | 25.89    | 23.63    | 21.43    |
| __ExKaldi 6grams__        | 29.83     | 24.07    | 22.40    |__20.01__ |

3, The phone error rate(PER) of various DNN-HMM-based systems. We trained our models with Tensorflow 2.3. The version of PyTorch-Kaldi toolkit is 1.0 in our experiment. (We are tuning the hyperparameter for more better result)

|                    | __DNN__   | __LSTM__ |
| :----------------: | :-------: | :------: |
| __Kaldi baseline__ | 18.67     | ---      | 
| __PyTorch-Kaldi__  | 17.99     | 17.01    |
| __ExKaldi__        | 15.13     | ---      |