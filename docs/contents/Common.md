# exkadli, exkaldi.utils

This section includes some common tools. They are distributed in different module.

------------------------
>## exkaldi.check_config 
(name,config=None) 

Get the default configure or check whether or not provided configure has the right format.  
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/utils/utils.py)

**Args:**  
_name_: an function name.  
_config_: a dict object.  

**Return:**  
A dict object if _config_ is None. Or return a bool value.

------------------------
>## utils.type_name
(obj)

Get the class name of the object.  
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/utils/utils.py)

**Args:**  
_obj_: a python object.

**Return:**  
a string.

------------------------
>## utils.run_shell_command
(cmd, stdin=None, stdout=None, stderr=None, inputs=None, env=None)

Run a shell command.  
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/utils/utils.py)

**Args:**  
_cmd_: a string include a shell command and its options.  
_stdin_: input stream.  
_stdout_: output stream.  
_stderr_: error stream.  
_inputs_: a string or bytes need to be sent to input stream.  
_env_: If None, use exkaldi.info.ENV defaultly.  

**Return:**  
a triples: (out, err, returnCode). Both out and err are bytes objects.

```python
import subprocess
cmd = "rm -rf ./test"
out,err,cod = utils.run_shell_command(cmd,stderr=subprocess.PIPE)
if cod != 0:
    print(err.decode())
```

------------------------
>## utils.run_shell_command_parallel
(cmds, env=None, timeout=exkaldi.info.timeout)

Run shell commands with mutiple processes.
In this mode, we don't allow the input and output streams are pipe lines.
If you mistakely appoint buffer to be input or output stream, we set time out error to avoid dead lock.
So you can change the time out value into a larger one to deal with large courpus as long as you rightly apply files as the input and output streams. 
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/utils/utils.py)

**Args:**  
_cmds_: a list of cmds.  
_env_: If None, use exkaldi.info.ENV defaultly.  
_timeout_: a int value.  

**Return:**  
a list of two-tuples: (returnCode, err). "err" is a bytes object.

```python
cmd = "sleep 5"
results = utils.run_shell_command([cmd,cmd])
for cod,err in results:
    if cod != 0:
        print(err.decode())
```

------------------------
>## utils.make_dependent_dirs
(path, pathIsFile=True)

Make the dependent directories recursively for a path if it has not existed.  
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/utils/utils.py)

**Args:**  
_path_: a file path or folder path.
_pathIsFile_: declare whether _path_ is a file path or folder path.

```python
path = "./a/b/c/d/e/f.py"
utils.make_dependent_dirs(path)
```

------------------------
>## utils.split_txt_file
(filePath, chunks=2)

Split a text file into N chunks by average numbers of lines.  
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/utils/utils.py)

**Args:**
_filePath_: text file path.  
_chunks_: an int value. chunk size. It must be larger than 1.

**Return:**  
a list of paths of chunk files.

```python
files = utils.split_txt_file("train_wav.scp", chunks=2)
```

------------------------
>## utils.compress_gz_file
(filePath, overWrite=False)

Compress a file to .gz file.  
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/utils/utils.py)

**Args:**  
_filePath_: file path.  
_overWrite_: If True, overwrite .gz file if it has existed.  
**Return:**  
the path of compressed file.

------------------------
>## utils.decompress_gz_file
(filePath, overWrite=False)

Decompress a .gz file.
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/utils/utils.py)

**Args:**  
_filePath_: .gz file path.  
**Return:**
the path of decompressed file.

------------------------
>## utils.flatten
(item)

Flatten an iterable object.
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/utils/utils.py)

**Args:**  
_item_: iterable objects, string, list, tuple or NumPy array.
**Return:**  
a list of flattened items.

```python
item = [1,[2,3],"456",(7,"8")]
results = utils.flatten(item)
```

------------------------
>## utils.list_files
(filePaths)

Such as os.listdir but we allow normal grammar or list or tuple object.  
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/utils/utils.py)

**Args:**  
_filePaths_: a string or list or tuple object.

**Return:**  
A list of file paths. 

```python
files = utils.list_files("*_wav.scp")
```

------------------------
>## utils.view_kaldi_usage
(toolName)

View the help information of specified kaldi command.  
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/utils/utils.py)

**Args:**  
_toolName_: kaldi tool name.

**Return:**  
a string.

```python
print( utils.view_kaldi_usage("copy-feats") )
```

-----------------------------
>>## utils.FileHandleManager
()

A class to create and manage opened file handles. 
A new FileHandleManager object should be instantiated bu python "with" grammar.
All handles will be closed automatically.  
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/utils/utils.py)

>### .view

Get names of all created file handles.

**Return:**  
a list.

>### .create
(mode, suffix=None, encoding=None, name=None)

Creat a temporary file and return the handle. 
All temporary file name will be added a default prefix "exkaldi_'.

**Args:**  
_mode_: the mode to open the file.  
_suffix_: add a suffix to the file name.  
_encoding_: encoding.  
_name_: a string. After named this handle exclusively, you can call its name to get it again.
            If None, we will use the file name as its default name.
        Note that this name is handle name not file name.
**Return:**  
a file handle.

```python
with utils.FileHandleManager() as fhm:
    h1 = fhm.create("w+", encoding="utf-8", name="txtHandle")
    print( h1.name ) # get it's file name 
```

>### .open
(filePath, mode, encoding=None, name=None)

Open a regular file and return the handle.

**Args:**  
_mode_: the mode to open the file.  
_suffix_: add a suffix to the file name.  
_encoding_: encoding.  
_name_: a string. After named this handle exclusively, you can call its name to get it again.
        If None, we will use the file name as its default name.
        We allow to open the same file in mutiple times as long as you name them differently.
        Note that this name is handle name not file name.
**Return:**  
a file handle.

```python
with utils.FileHandleManager() as fhm:
    hr = fhm.open("t1.txt", "r", encoding="utf-8", name="read")
    hw = fhm.open("t2.txt", "w", name="write")

    content = hr.read()
    hw.write(content)
```

>### .call
(name)

Get the file handle again by call its name.
If unexisted, return None.

**Args:**  
_name_: a string.
**Return:**  
a file handle.

```python
# Avoid to open the same file by mutiple times.
with utils.FileHandleManager() as fhm:
    files = utils.list_files("*_test.txt")
    for fileName in files:
        hr = fhm.call(fileName)
        if hr is None:
            hr = fhm.open(fileName, "r", encoding="utf-8")
        print( hr.readline() )
```

>### .close
(name=None)

Close file handle if need.

**Args:**  
_name_: if None, close all (Actually, all handles will be closed automatically.)

------------------------
>## exkaldi.match_utterances
(archieves)

Pick up utterances whose ID has existed in all provided archieves.  
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/core/common.py)

**Args:**  
_archieves_: a list of exkaldi archieve objects.

**Return:**  
a list of new archieve objects.

------------------------
>## exkaldi.utt2spk_to_spk2utt
(utt2spk, outFile=None)

Transform utt2spk file to spk2utt file.  
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/core/common.py)

**Args:**  
_utt2spk_: file name or exkaldi ListTable object.  
_outFile_: file name or None.  

**Return:**  
file name or exakldi ListTable object.

------------------------
>## exkaldi.spk2utt_to_utt2spk
(spk2utt, outFile=None)

Transform spk2utt file to utt2spk file.
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/core/common.py)

**Args:**  
_spk2utt_: file name or exkaldi ListTable object.  
_outFile_: file name or None.  

**Return:**  
file name or exakldi ListTable object.

------------------------
>## exkaldi.merge_archieves
(archieves)

Merge mutiple archieves to one.
exkaldi Lattice objects also support this operation.  
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/core/common.py)

**Args:**  
_archieves_: a list or tuple of mutiple exkaldi archieve objects which are the same class.

**Return:**  
a new archieve object.

------------------------
>## exkaldi.spk_to_utt
(spks, spk2utt)

Accept a list of speaker ids and return their corresponding utt IDs.  
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/core/common.py)

**Args:**  
_spks_: a list or tuple of speaker IDs.  
_spk2utt_: spk2utt file or ListTable object.  

**Return:**
a list of utterance IDs.

------------------------
>## exkaldi.utt_to_spk
(utts, utt2spk)

Accept a list of utterance IDs and return their corresponding speaker IDs.  
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/core/common.py)

**Args:**  
_utts_: a list or tuple of utterance IDs.  
_utt2spk_: utt2spk file or ListTable object.  

**Return:**  
a list of speaker IDs.

```python
feat = feat.subset(nRandom=10)
# acording to utternce ID, get corresponding speaker IDs
spks = exkaldi.utt_to_spk(utts=feat.utts, utt2spk="./utt2spk")
# then get corresponding CMVN
cmvn = cmvn.subset(uttIDs=spks)
# then apply CMVN
feat = exkaldi.use_cmvn(feat,cmvn,utt2spk="./utt2spk")
```