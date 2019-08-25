# PythonKaldi
This is a tool which introduce kaldi tools into python in a easy-use way.

## PythonKaldi at a Glance

1. Clone the PythonKaldi project.
```
git clone https://github.com/wangyu09/pythonkaldi
```

2. In the file < CSJsample.py >, there is a sample program that showed how to use the PythonKaldi tool. Exchange the parameter < CSJpath > for yours and also other parameters such as < epoch > if you want. Then run it.
```
python CSJsample.py
```
Especially, there are three sections in this sample program: first, train chainer neural network as acoustic model, and then use this pretrained AM to forward test data and decode them by generating lattice and further compute the WER. In the third step function < OnlineRecognize >, although we wrote it, I am sorry that it cannot be used now because of debugging.

## Concepts and Usage
Most of functions in PythonKaldi tool are performed with using "subprocess" to run shell cmd of kaldi tools. But we design a series of classes and functions to use them in a flexible way and it is more familiar for python programmer. PythonKaldi tool consist of three parts: < Basis Tools > to cover kaldi functions, < Speek client > to realize online-recognization, and < Chainer Tools > to give some useful tools to help train neural network acoustic model.

### < Basis Tools >
