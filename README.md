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

#### 1. [config exkaldi](examples/01_config_exkaldi.ipynb)
#### 2. [process acoustic feature](examples/02_feature_processing.ipynb)
#### 3. [prepare lexicon group](examples/03_prepare_lexicons.ipynb)
#### 4. [train and query a N-grams language model](examples/04_train_and_query_language_model.ipynb)
#### 5. [train monophone HMM-GMM](examples/05_train_mono_HMM-GMM.ipynb)
#### 6. [train decision tree](examples/06_train_decision_tree.ipynb)
#### 7. [train triphone HMM-GMM (train delta)](07_train_triphone_HMM-GMM_delta.ipynb)
#### 8. [make HCLG decoding graph](examples/08_make_HCLG_decode_graph.ipynb)
#### 9. [decode based on HMM-GMM and HCLG](examples/09_decode_back_HMM-GMM_and_WFST.ipynb)
#### 10. [process lattice and score](examples/10_process_lattice_and_score.ipynb)
#### 11. [train DNN acoustic model with Tensorflow 2.x](examples/11_train_DNN_acoustic_model_with_tensorflow.ipynb)
#### 12. [decode based on HMM-DNN and HCLG](examples/12_decode_back_HMM-DNN_and_WFST.ipynb)

Cehck the source code to look more information about exkaldi APIs.