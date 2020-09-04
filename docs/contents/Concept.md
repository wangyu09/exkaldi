# The Concept of Exkaldi
We use three data structures to discribe the Kaldi numerical data archives: __Index Table__, __Bytes Object__ and __NumPy Array__. They all stand for the same data.  

![three approaches](images/threeApproachs.png)  
  
__Index Table__: hold the index information of archive which has been saved in files.  
__Bytes Object__: hold the data in memory with bytes format.   
__NumPy Array__: hold the data in memory with NumPy array format.

These three structures have been designed as various specified classes in Exkaldi. Basesd on these classes, Exkaldi interacts with Kaldi command-line API to process archives and build ASR system via Python subprocess.  
Basically, these three data structures have these features:  
1. They can convert to one another easily.  
2. They are treated by Exkaldi functions without distinction.  
3. Achieves with Bytes format is the main currency in single process, but achieves with index table format are more used for multiple processes.  
4. Achieves with NumPy format can be used to generate iterable dataset and train NN model with deep learning frameworks, such as Tensorflow.  

In the follow table, there is a glance of Exkaldi numerical data archive class group:  

![core classes](images/archiveClassGroup.png)  

Beside above, Exkaldi has complete approaches to carry and process other archives and objects.  
In the follow table, there is a glance of other main data classes in Exkaldi:  

![other main classes](images/otherMainClasses.png)  

With the help of these classes, Exkaldi is qualified to build a complete ASR system from the scratch to a state-of-the-art level.
