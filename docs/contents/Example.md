
# An Example on TIMIT corpus
This section will roughly explain the main steps to train TIMIT corpus. The [TIMIT example](https://github.com/wangyu09/exkaldi/tree/master/examples/TIMIT) has the complete recipe and source code.

## Prepare the data
[01_prepare_data.py](https://github.com/wangyu09/exkaldi/blob/master/examples/TIMIT/01_prepare_data.py)

First of all, import `exkaldi` module.
```python
import exkaldi
from exkaldi import args
from exkaldi import declare
```
`args` is a object to manage the command line options and parse the arguments obtained from shell command line or configuration file. It is a global object. Once you've successfully parsed a set of arguments, you can use them in all Python programs as long as you import it.

Then, we parse the arguments and save them to file for debugging.
```python
args.add("--timitRoot", dtype=str, abbr="-t", default="/Corpus/TIMIT", discription="The root path of timit dataset.")
args.add("--expDir", dtype=str, abbr="-e", default="exp", discription="The output path to save generated data.")
args.parse()
args.save( os.path.join(args.expDir, "conf", "prepare_data.args") )
```
This operation, `args.parse()`, is necessary.

Then the main program checks the TIMIT format. 
```python
formatCheckCmd = f"{sph2pipeTool} -f wav {testWavFile}"
out,err,cod = exkaldi.utils.run_shell_command(formatCheckCmd, stderr="PIPE")
if cod == 0:
    sphFlag = True
else:
    sphFlag = False
```
In ExKaldi, `exkaldi.utils.run_shell_command` is a crucial function to run Shell and Kaldi command through the subprocess. Here we used it to check the format or audio data.

Then 
