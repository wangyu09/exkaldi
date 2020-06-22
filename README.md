# ExKaldi: An advance kaldi wrapper for Python 
[![License](https://img.shields.io/hexpm/l/Apa)](https://github.com/wangyu09/exkaldi/blob/master/LICENSE)
![Developing](https://img.shields.io/badge/Debug-v--1.2.x-red)
================================

ExKaldi automatic speech recognition toolkit is designed to build an interface between [Kaldi ASR toolkit](https://github.com/kaldi-asr/kaldi) and Python. 
Differing from other kaldi wrappers, exkaldi have these features:

1. Integrated APIs to build a ASR systems, including training HMM-GMM acoustic model, training HMM-DNN acoustic model, training and quering a N-grams language model, decoding and scoring.

2. Exkaldi C++ library was designed to support, such as ctc decoding for End-to-End. 

3. Use KenLm as languange model backend.

4. Support communication between local host and linux server (The ideal environment of Exkaldi is linux server).

The goal of ExKaldi is to help developers build high-performance ASR systems with Python language easily.

## Installation

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

2. You can install ExKaldi toolkit from PyPi library.
```
pip install exkaldi
```
But we recommand you can clone the ExKaldi source code from our github project, then install it.
```
git clone https://github.com/wangyu09/exkaldi.git
cd exkaldi
bash quick_install.sh
```

3. Check if it is installed correctly.
```
python -c "import exkaldi"
```


## Tutorial

In [exkaldi/tutorials](tutorials) directory, we prepared a simple tutorial to show how to use Exkaldi APIs to build a ASR system from the scratch.
The data is from librispeech train_100_clean dataset. This tutorial includes, extracting and processing feature, training and querying a N-grams language model, training HMM-GMM acoustic model, training a DNN acoustic model, decoding and scoring. This ASR symtem built here is just a dummy model, and we have done some normal experiments in [exkaldi/examples](examples). Check the source code to look more information about exkaldi APIs.

## Experiments

We have done some experiments to test ExKaldi toolkit, and it achived a good performance.
(We will upload the results of experiments little by little.)

#### TIMIT

1, The perplexity of various language models. All these systems are trained with TIMIT train data and tested with TIMIT test data. The score showed in the table is weighted average of all utterances and weights are the sentence length of each utterance.  

|                           | __2-grams__  | __3-grams__ | __4-grams__ |
| :-----------------------: | :----------: | :---------: | :---------: |
| __kaldi baseline irstlm__ | 14.67        | ---         | ---         |
| __exkaldi srilm__         | 14.69        | 13.44       | 14.26       |
| __exkaldi kenlm__         | 14.64        | __13.07__   | 13.20       |

2, The word error rate(WER) of various systems. All these systems are trained with TIMIT train dataset and tested with TIMIT test dataset. The Language model backend used in exkaldi is Kenlm. From the results, we can know than KenLm is avaliable to optimize the language model. And what's more, with Exkaldi, we cherry-picked the N-grams by testing the perplexity and it improved the performance of ASR system.

|                           | __mono__  | __tri1__ | __tri2__ | __tri3__ |
| :-----------------------: | :-------: | :------: | :------: | :------: |
| __kaldi baseline 2grams__ | 32.54     | 26.17    | 23.63    | 21.54    |
| __exkaldi 2grams__        | 32.53     | 25.89    | 23.63    | 21.43    |
| __exkaldi 3grams__        | 31.42     | 24.57    | 22.13    |__20.83__ |