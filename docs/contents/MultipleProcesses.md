# exkaldi.utils,exkaldi.core.common

------------------------
>## utils.run_shell_command_parallel
(cmds, env=None, timeout=exkaldi.info.timeout)

Run shell commands with multiple processes.
In this mode, we don't allow the input and output streams are pipe lines.
If you mistakely appoint buffer to be input or output stream, we set time out error to avoid dead lock.
So you can change the time out value into a larger one to deal with large courpus as long as you rightly apply files as the input and output streams. 
[view code distribution](https://github.com/wangyu09/exkaldi/blob/master/exkaldi/utils/utils.py)

**Args:**  
_cmds_: a list of cmds.  
_env_: If None, use exkaldi.info.ENV defaultly.  
_timeout_: a int value.  

**Return:**  
a list of two-tuples: (returnCode, err). _err_ is a bytes object.

**Examples:**  
```python
cmd = "sleep 5"
results = utils.run_shell_command_parallel([cmd,cmd])
for cod,err in results:
    if cod != 0:
        print(err.decode())
```
----------------------
>## common.check_multiple_resources
(*resources, outFile=None)

This function is used to verify the number of resources and modify the recources to match multiple processes.

**Args:**  
_*resources_: objects.  
_outFile_: None, file name, or a list of None objects, file names. If None, it means standard output stream.

**Return:**  
lists of resources.

----------------------
>## common.run_kaldi_commands_parallel
(resources, cmdPattern, analyzeResult=True, timeout=ExkaldiInfo.timeout, generateArchive=None, archiveNames=None)

Map resources to command pattern and run this command parallelly.

**Args:**  
_resources_: a dict whose keys are the name of resource and values are lists of resources objects. For example: {"feat": [BytesFeature01, BytesFeature02,... ], "outFile":{"newFeat01.ark","newFeat02.ark",...} }.The "outFile" resource is necessary.When there is only one process to run, "outFile" can be "-" which means the standard output stream.  
_cmdPattern_: a string needed to map the resources.For example: "copy-feat {feat} ark:{outFile}".  
_analyzeResult_: If True, analyze the result of processes. That means if there are errors in any processes, print the track info in standard output and stop program.  
_timeout_: a time out value. Dafaultly use _Exkaldi.info.timeout_.  
_generateArchive_: If the outputs are archives, you can get the Exkaldi archive objects directly by setting this argument "feat", or "ali", "cmvn","fmllrMat".  
_archiveNames_: If _generateArchive_ is not None, you can name them.

**Return:**  
a list of triples: (return code, error info, output file or buffer).


