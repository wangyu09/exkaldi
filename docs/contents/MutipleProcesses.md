# exkaldi,exkaldi.utils

-------------------------------------
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

----------------------
>## exkaldi.check_mutiple_resources
(*resources, outFile=None)

This function is used to check whether or not use mutiple process and verify the resources.

**Args:**  
_*resources_: objects.  
_outFile_: None, file name, or a list of None objects, file names. If None, it means standard output stream.

**Return:**  
lists of resources.

----------------------
>## exkaldi.run_kaldi_commands_parallel
(resources, cmdPattern, analyzeResult=True, timeout=ExkaldiInfo.timeout, generateArchieve=None, archieveNames=None)

Map resources to command pattern and run this command parallelly.

**Args:**  
_resources_: a dict whose keys are the name of resource and values are lists of resources objects. For example: {"feat": [BytesFeature01, BytesFeature02,... ], "outFile":{"newFeat01.ark","newFeat02.ark",...} }.The "outFile" resource is necessary.When there is only one process to run, "outFile" can be "-" which means the standard output stream.  
_cmdPattern_: a string needed to map the resources.For example: "copy-feat {feat} ark:{outFile}".  

**Return:**  
a list of triples: (return code, error info, output file or buffer).


