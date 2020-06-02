# ExKaldi: A kaldi wrapper for Python
ExKaldi automatic speech recognition toolkit is designed to build an interface between Kaldi and Python. 
Differing from other kaldi wrappers, exkaldi have these features:

1. Integrated APIs to build a ASR systems, including training HMM-GMM acoustic model, training HMM-DNN acoustic model, training and quering a N-grams language model, decoding and scoring.

2. Exkaldi C++ library was designed to support, such as ctc decoding for End-to-End. 

3. Use KenLm as languange model backend.

4. Support communication between local host and linux server (The ideal environment of Exkaldi is linux server).

## Installation
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

We prepared some tutorial to show how to use Exkaldi APIs in exkaldi/examples directory. 

### 1. [config exkaldi](github.com/wangyu09/exkaldi/blob/master/examples/01_config_exkaldi.ipynb)

### 2. process acoustic feature

### 3. prepare lexicon group

### 4. train and query a N-grams language model

### 5. train monophone HMM-GMM

### 6. train decision tree

### 7. train triphone HMM-GMM (train delta)

### 8. make HCLG decoding graph

### 9. decode based on HMM-GMM and HCLG

### 10. process lattice and score

### 11. train DNN acoustic model with Tensorflow 2.x

### 12. decode based on HMM-DNN and HCLG

Cehck the source code to look more information about exkaldi APIs.