# exkaldi.info

`exkaldi.info` is an instance of `exkaldi.version.ExKaldiInfo` class. It holds the basic configuration information of Exkaldi.  
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/version.py)

--------------------------
>## info

Get an object that carries various exkaldi configuration information. 

**Return:**  
An object including various configure information.

**Examples**
```python
from exkaldi import info
print(info)
```
-----------------------------
>## info.EXKALDI  

Get the Exkaldi version number.  

**Return:**    
Itself.

**Examples**
```python
from exkaldi import info
print(info.EXKALDI)
```
-----------------------------
>## info.KALDI

Get the version number of existed Kaldi toolkit. 

**Return:**    
None, "unknown" or a named tuple.

**Examples**
```python
from exkaldi import info
print(info.KALDI)
```
-----------------------------
>## info.KALDI_ROOT

Look for the root path of Kaldi toolkit in system PATH.  

**Return:**
None or a string of path.

**Examples**
```python
from exkaldi import info
print(info.KALDI_ROOT)
```
---------------------------------
>## info.ENV

Get the system environment in which ExKaldi are running. 

**Return:**  
A dict object.

**Examples**
```python
from exkaldi import info
print(info.ENV["PATH"])
```
---------------------------------
>## info.reset_kaldi_root
(path)

Reset the root path of Kaldi toolkit and add related directories to system PATH manually.

**Args:**  
_path_: a directory path.

**Examples**
```python
from exkaldi import info
info.reset_kaldi_root("new_kaldi")
```
---------------------------------
>## info.export_path
(path)

Add a new path to Exkaldi environment PATH. 

**Args:**  
_path_: a string of path.

**Examples**
```python
from exkaldi import info
info.export_path("./test")
```
---------------------------------
>## info.timeout

Get the timeout value. 

**Return:**  
A int value.

**Examples**
```python
from exkaldi import info
print(info.timeout)
```
---------------------------------
>## info.set_timeout
(timeout)

Set the timeout value.

**Args:**  
_timeout_: a int value.

**Examples**
```python
from exkaldi import info
info.set_timeout(60)
```