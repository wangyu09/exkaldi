[![License]](https://github.com/wangyu09/exkaldi/blob/master/LICENSE)
ExKaldi: An advanced kaldi wrapper for Python
================================

ExKaldi automatic speech recognition toolkit is designed to build an interface between Kaldi and Python. 
Differing from other kaldi wrappers, exkaldi have these features:

1. Integrated APIs to build a ASR systems, including training HMM-GMM acoustic model, training HMM-DNN acoustic model, training and quering a N-grams language model, decoding and scoring.

2. Exkaldi C++ library was designed to support, such as ctc decoding for End-to-End. 

3. Use KenLm as languange model backend.

4. Support communication between local host and linux server (The ideal environment of Exkaldi is linux server).

The goal of exkaldi is to help developer build a high-performance ASR system with Python language easily.

## Installation
--------------------------

1. If you have not installed Kaldi ASR toolkit, clone the Kaldi ASR toolkit repository firstly.
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
--------------------------

In [exkaldi/tutorials](tutorials) directory, we prepared a simple tutorial to show how to use Exkaldi APIs to build a ASR system from the scratch.
The data is from librispeech train_100_clean dataset. This tutorial includes, extracting and processing feature, training and querying a N-grams language model, training HMM-GMM acoustic model, decoding and scoring. This ASR symtem built here is just a dummy model, and we have done some normal experiments in [exkaldi/examples](examples). Check the source code to look more information about exkaldi APIs.

## Experiments
--------------------------

We have done some experiments to test ExKaldi toolkit, and it achived a good performance.
(We will upload the results of experiments little by little.)

##### TIMIT

1, The perplexity of various language models.

![TIMITperplexity](images/TIMITperplexity.png)
