# exkadli, exkaldi.utils

This section includes some common tools. They are distributed in different module.

------------------------
>## exkaldi.check_config 
(name,config=None) 

Get the default configuration or check whether the provided configuration has the right format.  
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/utils/utils.py)

**Args:**  
_name_: an function name.  
_config_: None or a dict object.  

**Return:**  
if _config_ is None, return A dict object or None.  
else, return True or raise error.

**Examples:**
Some functions in Exkaldi can accept extra arguments such as "compute_mfcc". This function will help you to configure it.
```python
print(exkaldi.check_config("compute_mfcc"))
```
This operation will return a dict object that the keys are option name and the values are default values. You can set your own configurations like it.
```python
extraConfig={"--use-eneragy":"false"}
feat=exkaldi.compute_mfcc("wav.scp",config=extraConfig)
```

------------------------
>## utils.run_shell_command
(cmd, stdin=None, stdout=None, stderr=None, inputs=None, env=None)

Run a shell command with Python subprocess.  
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/utils/utils.py)

**Args:**  
_cmd_: a string include a shell command and its options.  
_stdin_,_stdout_,_stderr_: IO stream. If "PIPE", use subprocess.PIPE.  
_inputs_: a string or bytes need to be sent to input stream.  
_env_: If None, use exkaldi.info.ENV defaultly.  

**Return:**  
a triples: (out, err, returnCode). Both _out_ and _err_ are bytes objects.

**Examples:**
```python
import subprocess

cmd="rm -rf ./test"
out,err,cod=utils.run_shell_command(cmd,stderr="PIPE")
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
_pathIsFile_: a bool value to declare that _path_ is a file path or folder path.

**Examples:**
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
_chunks_: an int value. chunk size. It must be greater than 1.

**Return:**  
a list of paths of genrated chunk files.
each file has a a prefix such as "ck0_" which _0_ is the chunk ID.

**Examples:**
```python
files = utils.split_txt_file("train_wav.scp", chunks=2)
```
------------------------
>## utils.compress_gz_file
(filePath,overWrite=False,keepSource=False)

Compress a file to .gz file.  
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/utils/utils.py)

**Args:**  
_filePath_: file path.  
_overWrite_: If True, overwrite .gz file if it has existed.  
_keepSource_: If True, retain source file.

**Return:**  
the path of compressed file.

**Examples:**
```python
exkaldi.utils.compress_gz_file("ali",overWrite=True,keepSource=True)
```
------------------------
>## utils.decompress_gz_file
(filePath,overWrite=False,keepSource=False)

Decompress a .gz file.
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/utils/utils.py)

**Args:**  
_filePath_: .gz file path.  
_overWrite_: If True, overwrite output file if it has existed.  
_keepSource_: If True, retain source file.

**Return:**
the path of decompressed file.

**Examples:**
```python
exkaldi.utils.decompress_gz_file("ali.gz",overWrite=True,keepSource=True)
```
------------------------
>## utils.flatten
(item)

Flatten an iterable object.
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/utils/utils.py)

**Args:**  
_item_: iterable objects, string, list, tuple or NumPy array.

**Return:**  
a list of flattened items.

**Examples:**
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

**Examples:**
```python
files = utils.list_files("ali.*.gz")
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

**Examples:**
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
a list of file handle IDs.

>### .create
(mode, suffix=None, encoding=None, name=None)

Creat a temporary file and return the handle. 
All temporary file name will be added a default prefix "exkaldi_'.

**Args:**  
_mode_: the mode to open the file.  
_suffix_: add a suffix to the file name.  
_encoding_: encoding.  
_name_: a string. After named this handle exclusively, you can call its name to get it again.If None, we will use the file name as its default name.Note that this name is handle name not file name.

**Return:**  
a file handle.

**Examples:**  
This class must be instantiated with the `with` grammar. 
For example, if you want to create a temporary file.
```python
with utils.FileHandleManager() as fhm:
    h1 = fhm.create("w+", encoding="utf-8")
    print( h1 )
    print( h1.name ) #get it's file name
```
The returned _h1_ is a file handle. You can give the handle a unique name in order to call it whenever you want to use it.
```python
with utils.FileHandleManager() as fhm:
    h1 = fhm.create("w+", encoding="utf-8", name="temp")
    print( h1 )
    del h1
    
    h1 = fhm.call("temp")
    print( h1 ) #
```

>### .open
(filePath, mode, encoding=None, name=None)

Open a regular file and return the handle.

**Args:**  
_mode_: the mode to open the file.  
_suffix_: add a suffix to the file name.  
_encoding_: encoding.  
_name_: a string. After named this handle exclusively, you can call its name to get it again.If None, we will use the file name as its default name.We allow to open the same file in multiple times as long as you name them differently.Note that this name is handle name not file name.

**Return:**  
a file handle.

**Examples:**  
Basically, you can open the regular files.
```python
with utils.FileHandleManager() as fhm:
    hr = fhm.open("t1.txt", "r", encoding="utf-8", name="read")
    hw = fhm.open("t2.txt", "w", name="write")

    content = hr.read()
    hw.write(content)
```
A file can be opened in several times as long as you give them different names.
```python
with utils.FileHandleManager() as fhm:
    h1 = fhm.open("test.txt", "r", encoding="utf-8", name="1")
    h2 = fhm.open("test.txt", "r", encoding="utf-8", name="2")
```

>### .call
(name)

Get the file handle again by call its name.
If unexisted, return None.

**Args:**  
_name_: a string.

**Return:**  
a file handle.

**Examples:**  
```python
# Avoid to open the same file by multiple times.
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
(archives)

Pick up utterances whose ID has existed in all provided archives.   
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/core/common.py)

**Args:**  
_archives_: a list of exkaldi archive objects.

**Return:**  
a list of new archive objects.

**Examples:**  
For example, you need to split a larger corpus (here, take three archives : feature, alignment, and transcription for example) into n chunks. Of cause, if all of these three archives has the completely same utterance IDs. you can sort and subset them directly.
```
chunkFeat = feat.sort().subset(chunks=4)
chunkAli = ali.sort().subset(chunks=4)
chunkTrans = trans.sort().subset(chunks=4)
```
Their subsets should be one-to-one correspondence. But to be precise, we should subset them by matching the utterance IDs. There are several approaches to do it, for example:
```python
chunkFeat = feat.sort().subset(chunks=4)

chunkAli = []
chunkTrans = []
for fe in chunkFeat:
    chunkAli.append( ali.subset(uttIDs=fe.utts) )
    chunkTrans.append( trans.subset(uttIDs=fe.utts) )
```
Or if you want to get results collected by chunks.
```python
chunkFeat = feat.sort().subset(chunks=4)

chunkData = []
for fe in feat.sort().subset(chunks=4):
    chunkData.append( exkaldi.match_utterances([fe,ali,trans]) )
```
------------------------
>## exkaldi.utt2spk_to_spk2utt
(utt2spk, outFile=None)

Transform utt2spk to spk2utt.  
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/core/common.py)

**Args:**  
_utt2spk_: file name or exkaldi ListTable object.  
_outFile_: file name or None.  

**Return:**  
file name or exakldi ListTable object.

**Examples:** 
```python
spk2utt = exkaldi.utt2spk_to_spk2utt("./utt2spk")
```
------------------------
>## exkaldi.spk2utt_to_utt2spk
(spk2utt, outFile=None)

Transform spk2utt to utt2spk.
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/core/common.py)

**Args:**  
_spk2utt_: file name or exkaldi ListTable object.  
_outFile_: file name or None.  

**Return:**  
file name or exakldi ListTable object.

**Examples:** 
```python
utt2spk = exkaldi.utt2spk_to_spk2utt("./spk2utt")
```
------------------------
>## exkaldi.merge_archives
(archives)

Merge multiple archives to one.
Exkaldi Lattice objects also support this operation.  
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/core/common.py)

**Args:**  
_archives_: a list or tuple of multiple exkaldi archive objects which are the same class.

**Return:**  
a new archive object.

**Examples:** 
Typically, after you applied parallel processes. You can merge the output to be a complete archive object.
```python
lats = exkaldi.decode.wfst.gmm_decode(feat=[feat1,feat2],hmm="./final.mdl",HCLGFile="./HCLG.fst",symbolTable="./words.txt")

finalLat = exkaldi.merge_archives(lats)
```
------------------------
>## exkaldi.spk_to_utt
(spks, spk2utt)

Accept a list of speaker ids and return their corresponding utt IDs.  
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/core/common.py)

**Args:**  
_spks_: a string or list or tuple of speaker IDs.  
_spk2utt_: spk2utt file or ListTable object.  

**Return:**
a list of utterance IDs.

**Examples:**
Because different archieves use utterance ID or Speaker ID as index. If you need match these archives, use this function to convert them. 
```python
cmvn = cmvn.subset(nHead=1) # cmvn use speaker ID as index ID
utts = exkaldi.spk_to_utt( cmvn.utts, spk2utt="./spk2utt") # In current version, cmvn.utts actually is the speaker IDs. 
feat = feat.subset(uttIDs=utts)

newFeat = exakldi.use_cmvn(feat,cmvn,utt2spk="./utt2spk")
```
This function help to distribute data.

------------------------
>## exkaldi.utt_to_spk
(utts, utt2spk)

Accept a list of utterance IDs and return their corresponding speaker IDs.  
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/core/common.py)

**Args:**  
_utts_: a string or list or tuple of utterance IDs.  
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