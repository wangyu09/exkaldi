# exkaldi.args

This is an instance of exkaldi ArgumentParser class.
It works at a global scope of all exkaldi programs.

-----------------------------------------------------
>## args.discribe
(message)

Add a discription of current program.

**Args:**  
_message_: a string.

**Examples:**  
```python
from exkaldi import args

args.discribe("This program is used to train monophone GMM-HMM.")
```
-----------------------------------------------------
>## args.add
(optionName,dtype,abbreviation=None,default=None,choices=None,minV=None,maxV=None,discription=None)

Add a new option.

**Args:**  
_optionName_: a string which must have a format such as "--exkaldi" (but can not be "--help").  
_dtype_: one of float, int, str or bool.  
_abbreviation_: a abbreviation of optionName which must have a format such as "-e" (but can not be "-h").  
_dtype_: the default value or a list/tuple of values.  
_choices_: a list/tuple of values.  
_minV_: set the minimum value if dtype is int or float. Enable when _choices_ is None.  
_maxV_: set the maximum value if dtype is int or float. Enable when _choices_ is None.  
_maxV_: a string to discribe this option.  

**Examples:**  
```python
args.add("--outDir",dtype=str,abbreviation="-o",default="./exp",discription="the output directory.")
```
Added a option of output directory. We allow the default value is a list or tuple. for example:
```python
args.add("--learningRate",dtype=float,abbreviation="-l",default=[0.5,0.2,0.1],discription="the learning rate of optimizer.")
```
-----------------------------------------------------
>## args.print_help_and_exit
()

Print help information to standard output and stop program.

-----------------------------------------------------
>## args.parse
()

Start to parse the arguments. 

**Examples:**  
After you have added a option, you can appoint the argument and it will be parsed.
```python
from exkaldi import args

args.add("--test",dtype=int,default=1)
args.parse()
```
Then you can appoint the argument in bash command line:
```bash
python test.py --test=2
```
In particular, We allow user to input mutiple values for one option with a specified format such as: 1|2|3|4. So if you appoint the argument like this:
```bash
python test.py --test=1|2|3|4
```
In the python program, `args.test` will become a list object: `[1,2,3,4]`.

-----------------------------------------------------
>## args.save
(fileName=None)

Save all arguments to file.

**Args:**  
_fileName_: file name. If None, return a string.  

**Return**  
file name or a string.

**Examples:**  
```python
from exkaldi import args

args.add("--test",dtype=int,default=1)
args.parse()
args.save("conf/test.args")
```

-----------------------------------------------------
>## args.load
(filePath)

Restorage arguments from file. All arguments can be parsed again with new value.

**Args:**  
_filePath_: file name.  

**Examples:**  
```python
from exkaldi import args

args.load("conf/test.args")
args.parse()
```




