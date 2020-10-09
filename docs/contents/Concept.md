# The Concept of Exkaldi
We use three data structures to discribe the Kaldi numerical data archives: __Index Table__, __Bytes Object__ and __NumPy Array__.   

|                  | Discription   |  Examples    |
| :--------------: | :----------   |  :---------- | 
| __Index Table__  |  The index information of archive saved in files. | {'uttID':Index(frames=1,startIndex=0,dataSize=33,filePath=./dummy.ark)} |
| __Bytes Object__ |  The data instance saved in memory with bytes format. | b'uttID \x00BFM \x04\x01\x00\x00\x00\x04\x03\x00\x00\x00\x00\x00\x80?\x00\x00\x00@\x00\x00@@' |
| __NumPy Array__  |  The data instance saved in memory with NumPy array format. | {'uttID': array([[1., 2., 3.]], dtype=float32)} |

Basically, these three data structures have these features:  
1. All these three structures hold the same data and they can convert to one another easily.  
2. They are treated by Exkaldi functions without distinction.  
3. Achieves with Bytes format is the main currency in single process, but achieves with index table format are more used for multiple processes.  
4. Achieves with NumPy format can be used to generate iterable dataset and train NN model with deep learning frameworks, such as Tensorflow.  

In practice, they are designed to be used in a variety of specific data classes in Exkaldi. 
In the follow table, there is a glance of Exkaldi numerical data archive class group (up to version 1.3):  

|                  | Base Class   |  Subclass    |
| :--------------: | :----------  |  :---------- | 
| __Index Table__  |  IndexTable        |              |
| __Bytes Object__ |  BytesMatrix | BytesFeat,BytesCMVN,BytesProb,BytesFmllr |
| __Bytes Object__ |  BytesVector | BytesAliTrans |
| __NumPy Array__  |  NumpyMatrix | NumpyFeat,NumpyCMVN,NumpyProb,NumpyFmllr |
| __NumPy Array__  |  NumpyVector | NumpyAliTrans,NumpyAli,NumpyAliPdf,NumpyAliPhone |

Beside above, Exkaldi has complete approaches to carry and process other archives and objects.  
In the follow table, there is a glance of other main data classes in Exkaldi:  

|                   | Discription  |
| :--------------:  | :----------  |
| __LexiconBank__   |  Generate all lexicons automatically and manage them efficiently.  |
| __KenNGrams__     |  Hold an N-grams language model with KenLM format. | 
| __DataIterator__  |  Load and iterate data parallelly to train NN AM with a large-size corpus. | 
| __Metric__        |  Hold various scores, typically LM scores and AM scores. | 
| __Lattice__       |  The decoding lattice. | 
| __Transcription__ |  Hold the transcriptions. | 
| __MonophoneHMM__  |  Train and hold the monophone GMM-HMM model. | 
| __TriphoneHMM__   |  Train and hold the context-phone GMM-HMM model. | 
| __DecisionTree__  |  Train and hold the decision tree. | 

With the help of these classes, Exkaldi interacts with Kaldi command-line APIs to process required data and build ASR system via Python subprocess.
It make ExKaldi qualified to build a complete ASR system from the scratch to a state-of-the-art level.
