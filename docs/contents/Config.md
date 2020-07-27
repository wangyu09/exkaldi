# exkaldi.info

`exkaldi.info` is an instance of `exkaldi.version.ExKaldi` class. It hold the base configure information of Exkaldi.  
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/version.py)

--------------------------
>## info

Get an object holding various exkaldi configure information. 

**Return:**  
An object including various configure information.

```python
from exkaldi import info

print(info)
```
-----------------------------
>## info.EXKALDI  

Get the exkaldi version number.  

**Return:**    
A named tuple.

```python
from exkaldi import info

print(info.EXKALDI)
```
-----------------------------
>## info.KALDI

Get the version number of existed Kaldi toolkit. 

**Return:**    
A named tuple.
```python
from exkaldi import info

print(info.KALDI)
```
-----------------------------
>## info.KALDI_ROOT

Get the root path of Kaldi toolkit.  

**Return:**    
A string of path.
```python
from exkaldi import info

print(info.KALDI_ROOT)
```
---------------------------------
>## info.ENV

Get the environment in which ExKaldi are running. 

**Return:**  
A dict object.
```python
from exkaldi import info

print(info.ENV["PATH"])
```
---------------------------------
>## info.reset_kaldi_root
(path)

Reset the root path of Kaldi toolkit. 

**Args:**  
_path_: a string of path.
```python
from exkaldi import info

info.reset_kaldi_root("new/kaldi")
```
---------------------------------
>## info.export_path
(path)

Add a new path to system environment. 

**Args:**  
_path_: a string of path.
```python
from exkaldi import info

info.export_path("./test")
```
---------------------------------
>## info.timeout

Get the timeout value. 

**Return:**  
A int value.
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
```python
from exkaldi import info

info.set_timeout(60)
```