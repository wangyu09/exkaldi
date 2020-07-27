# ExKaldi Documents

Exkaldi automatic speech recognition toolkit is designed to build an interface between [Kaldi ASR toolkit](https://github.com/kaldi-asr/kaldi) and Python. 

Differing from other kaldi wrappers, Exkaldi have these features:  
1. Integrated APIs to build a ASR systems, including training HMM-GMM acoustic model, training HMM-DNN acoustic model, training and quering a N-grams language model, decoding and scoring.  
2. Exkaldi C++ library was designed to support, such as ctc decoding for End-to-End.   
3. Use KenLm as languange model backend.

The goal of Exkaldi is to help developers build high-performance ASR systems with Python language easily.