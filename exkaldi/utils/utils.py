# coding=utf-8
#
# Yu Wang (University of Yamanashi)
# May, 2020
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import datetime
import importlib
import subprocess
from glob import glob
from collections import namedtuple
from collections.abc import Iterable
import numpy as np

from exkaldi.version import version as ExkaldiInfo
from exkaldi.version import WrongPath

class WrongOperation(Exception):pass
class WrongDataFormat(Exception):pass
class KaldiProcessError(Exception):pass
class UnsupportedDataType(Exception):pass

def type_name(obj):
	'''
	Get the class name of the object.
	
	Args:
		<obj>: a python object.
	'''
	return obj.__class__.__name__

def run_shell_command(cmd, stdin=None, stdout=None, stderr=None, inputs=None, env=None):
	'''
	Run a shell command.

	Args:
		<cmd>: a string or list.
		<inputs>: a string or bytes.
		<env>: If None, use exkaldi.version.ENV defaultly.

	Return:
		out, err, returnCode
	'''
	if isinstance(cmd, str):
		shell = True
	elif isinstance(cmd, list):
		shell = False
	else:
		raise WrongOperation("Expected <cmd> is string or list whose menbers are a command and its options.")
	
	if env is None:
		env = ExkaldiInfo.ENV

	if inputs is not None:
		if isinstance(inputs, str):
			inputs = inputs.encode()
		elif isinstance(inputs, bytes):
			pass
		else:
			raise UnsupportedDataType(f"Expected <inputs> is str or bytes but got {type_name(inputs)}.")

	p = subprocess.Popen(cmd, shell=shell, stdin=stdin, stdout=stdout, stderr=stderr, env=env)
	(out, err) = p.communicate(input=inputs)
	p.wait()

	return out, err, p.returncode

def make_dependent_dirs(path, pathIsFile=True):
	'''
	Make the dependent directories for a path if it has not existed.

	Args:
		<path>: a file path or folder path.
		<pathIsFile>: declare <path> if is a file path or folder path.
	'''
	assert isinstance(path, str), "<path> should be a file path."
	path = os.path.abspath(path.strip())

	if pathIsFile:
		dirPath = os.path.dirname()
	else:
		dirPath = path
	
	if not os.path.isdir(dirPath):
		try:
			os.makedirs(dirPath)
		except Exception as e:
			print(f"Failed to make directory:{dirPath}.")
			raise e

def write_log(message, mode="a", logFile=None):
	'''
	Write a piece of message down to log file.

	Args:
		<message>: a string that can be writed down to file.
		<mode>: 'a' or 'w'.
		<logFile>: If None, write to default log file.
	'''
	assert mode in ['w','a'], f"<mode> should be 'a' or 'w' but got {mode}."

	if logFile is None:
		fileName = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".log"
		logFile = os.path.join(ExkaldiInfo.LOG_DIR, fileName)

	make_dependent_dirs(logFile, pathIsFile=True)

	with open(logFile, mode, encoding="utf-8") as fw:
		fw.write(message)

def list_log_files():
	'''
	List all log files in default log folder.
	'''
	logFiles = glob( os.path.join(ExkaldiInfo.LOG_DIR, "*.log") )

	return list( sorted(logFiles) )

def show_log(logFile=None, tail=0):
	'''
	Print the tail N lines of log file to standerd output.

	Args:
		<logFile>: If None, use the lastest default log file.
		<tail>: If 0, print all.
	'''
	assert isinstance(tail, int) and tail >= 0, "Expected <tail> is non-negative int value."
	if logFile is None:
		logFiles = sorted(glob(os.path.join(ExkaldiInfo.LOG_DIR,"*.log")), reverse=True)
		if len(logFiles) > 0:
			logFile = logFiles[0]
	else:
		if not os.path.isfile(logFile):
			temp = os.path.join(ExkaldiInfo.LOG_DIR, logFile)
			if not os.path.isfile(temp):
				raise WrongPath(f"No such file:{logFile}.")
			else:
				logFile = temp
	
	if logFile is not None:
		with open(logFile, "r", encoding="utf-8") as fr:
			lines = fr.readlines()
			for line in lines[-tail:]:
				print(line.strip())
	else:
		print("No log information to display.")

def print_message(*args,**kwargs):
	'''
	Almost the same as Python print function.

	Args:
		<verbose>: If 0, print nothing, or print to standerd output.
	'''

	if "verbose" in kwargs.keys():
		verbos = kwargs.pop("verbose")
	else:
		verbos = 1

	if verbos != 0:
		print(*args,**kwargs)

def check_config(name, config=None):
	'''
	Check if the users' configure is right when call some functions.

	Args:
		<name>: function name.
		<config>: a list object whose items are configure name and their configure values. If None, return the default configure.
	
	Return:
		If <config> is None, return a list of default configure of <name>.
		Or return True or False.
		Or None if <name> is unavaliable.
	'''
	assert isinstance(name, str), "<name> should be a name-like string."

	ModuleNotFoundError
	try:
		module = importlib.import_module(f'exkaldi.config.{name}')
	except ModuleNotFoundError:
		print(f"Warning: no default configure for name '{name}'.")
		return None
	else:
		c = module.config

	if config is None:
		new = {}
		for key,value in c.items():
			new[key] = value[0]
		return new
	else:
		if not isinstance(config, dict):
			raise WrongOperation(f"<config> has a wrong format. You can use check_config('{name}') to look expected configure format.")
		for k in config.keys():
			if not k in c.keys():
				raise WrongOperation(f"No such key: <{k}> in {name}.")
			else:
				proto = c[k][1]
				if isinstance(config[k], bool):
					raise WrongOperation(f"configure <{k}> is bool value '{configure[k]}', but we expected str value like 'true' or 'false'.")
				elif not isinstance(config[k], proto):
					raise WrongDataFormat(f"configure <{k}> is expected {proto} but got {type_name(config[k])}.")
			return True

def split_txt_file(filePath, chunks=2):
	'''
	Split a text file into N chunks by average numbers of lines.

	Args:
		<filePath>: text file path.
		<chunks>: chunk size.
	Return:
		a list of name of chunk files.
	'''    
	assert isinstance(chunks, int) and chunks > 1, "Expected <chunks> is int value and larger than 1."

	if not os.path.isfile(filePath):
		raise WrongPath(f"No such file:{filePath}.")

	with open(filePath, 'r', encoding='utf-8') as fr:
		data = fr.readlines()

	lines = len(data)
	chunkLines = lines//chunks

	if chunkLines == 0:
		chunkLines = 1
		chunks = lines
		t = 0
	else:
		t = lines - chunkLines * chunks

	a = len(str(chunks))
	files = []

	filePath = os.path.abspath(filePath)
	dirIndex = filePath.rfind('/')
	if dirIndex == -1:
		dirName = ""
		fileName = filePath
	else:
		dirName = filePath[:dirIndex+1]
		fileName = filePath[dirIndex+1:]

	suffixIndex = fileName.rfind('.')
	if suffixIndex != -1:
		newFile = dirName + fileName[0:suffixIndex] + f"_%0{a}d" + fileName[suffixIndex:]
	else:
		newFile = dirName + fileName + f"_%0{a}d"

	for i in range(chunks):
		if i < t:
			chunkData = data[i*(chunkLines+1):(i+1)*(chunkLines+1)]
		else:
			chunkData = data[i*chunkLines:(i+1)*chunkLines]
		with open(newFile%(i), 'w', encoding='utf-8') as fw:
			fw.write(''.join(chunkData))
		files.append(newFile%(i))
	
	return files

def compress_gz_file(filePath):
	'''
	Compress a file to gz file.

	Args:
		<filePath>: file path.
	Return:
		the absolute path of compressed file.
	'''
	assert isinstance(filePath, str), f"<filePath> must be a string but got {type_name(filePath)}."
	filePath = filePath.strip()
	if not os.path.isfile(filePath):
		raise WrongPath(f"Noe such file:{filePath}.")

	cmd = f"gzip {filePath}".format(filePath)
	out, err, _ = run_shell_command(cmd, stderr=subprocess.PIPE)
	outFile = filePath+".gz"
	if not os.path.isfile(outFile):
		print(err.decode())
		raise Exception("Failed to compress file.")
	else:
		return os.path.abspath(outFile)

def decompress_gz_file(filePath):
	'''
	Decompress a gz file.

	Args:
		<filePath>: file path.
	Return:
		the absolute path of decompressed file.
	'''
	assert isinstance(filePath, str), f"<filePath> must be a string but got {type_name(filePath)}."
	filePath = filePath.strip()
	if not os.path.isfile(filePath):
		raise WrongPath(f"Noe such file:{filePath}.")
	elif not filePath.endswith(".gz"):
		raise WrongOperation(f"{filePath}: Unknown suffix.")

	cmd = "gzip -d {}".format(filePath)

	out, err, _ = run_shell_command(cmd, stderr=subprocess.PIPE)
	outFile = filePath[:-3]
	if not os.path.isfile(outFile):
		print(err.decode())
		raise Exception("Failed to decompress file.")
	else:
		return outFile

def flatten(item):
	'''
	Flatten a iterable object.

	Args:
		<item>: iterable objects, string, list, tuple or NumPy array.
	Return:
		a list of flattened items.
	'''
	assert isinstance(item, Iterable), "<item> is not a iterable object."

	new = []
	for i in item:
		if isinstance(i, (int, float)):
			new.append(i)
		if isinstance(i, str):
			if len(i) <= 1:
				new.append(i)
			else:
				new.extend(flatten(i))
		elif isinstance(i, (list, tuple, set)):
			new.extend(flatten(i))
		elif isinstance(i, np.ndarray):
			if i.shape == ():
				new.append(i)
			else:
				new.extend(flatten(i))
		else:
			raise UnsupportedDataType(f"Expected list, tuple, set, str or Numpy array object but got {type_name}.")

	return new