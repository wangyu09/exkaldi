# coding=utf-8
#
# Yu Wang (University of Yamanashi)
# May,2020
#
# Licensed under the Apache License,Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''This package defined some utilities.'''

import os
import datetime
import importlib
import subprocess
from glob import glob
from collections import namedtuple
from collections.abc import Iterable
import numpy as np
import tempfile
import time

from exkaldi.version import info as ExKaldiInfo
from exkaldi.error import *
from exkaldi.utils import declare

def type_name(obj):
	'''
	Get the class name of the object.
	
	Args:
		<obj>: a python object.
	
	Return:
		a string.
	'''
	return obj.__class__.__name__

def run_shell_command(cmd,stdin=None,stdout=None,stderr=None,inputs=None,env=None):
	'''
	Run a shell command with Python subprocess.

	Args:
		<cmd>: a string including a shell command and its options.
		<stdin>,<stdout>,<stderr>: IO streams. If "PIPE",use subprocess.PIPE.
		<inputs>: a string or bytes to send to input stream.
		<env>: If None,use exkaldi.version.ENV defaultly.

	Return:
		out,err,returnCode
	'''
	declare.is_valid_string("cmd",cmd)
	
	if env is None:
		env = ExKaldiInfo.ENV

	if inputs is not None:
		declare.is_classes("inputs",inputs,[str,bytes])
		if isinstance(inputs,str):
			inputs = inputs.encode()
	
	if stdin == "PIPE":
		stdin = subprocess.PIPE
	if stdout == "PIPE":
		stdout = subprocess.PIPE
	if stderr == "PIPE":
		stderr = subprocess.PIPE

	p = subprocess.Popen(cmd,shell=True,stdin=stdin,stdout=stdout,stderr=stderr,env=env)
	(out,err) = p.communicate(input=inputs)

	return out,err,p.returncode

def run_shell_command_parallel(cmds,env=None,timeout=ExKaldiInfo.timeout):
	'''
	Run shell commands with multiple processes.
	In this mode,we don't allow the input and output streams are PIPEs.
	If you mistakely appoint buffer to be input or output stream,we set time out error to avoid dead lock.
	So you can change the time out value into a larger one to deal with large courpus as long as you rightly apply files as the input and output streams. 

	Args:
		<cmds>: a list of strings. Each string should be a command and its options.
		<env>: If None,use exkaldi.version.ENV defaultly.
		<timeout>: a int value. Its the total timeout value of all processes.

	Return:
		a list of pairs: return code and error information.
	'''
	declare.is_classes("cmds",cmds,[tuple,list])
	declare.is_positive_int("timeout",timeout)
	
	if env is None:
		env = ExKaldiInfo.ENV

	processManager = {}
	for index,cmd in enumerate(cmds):
		declare.is_valid_string("cmd",cmd)
		processManager[index] = subprocess.Popen(cmd,shell=True,stderr=subprocess.PIPE,env=env)

	runningProcess = len(processManager)
	if runningProcess == 0:
		raise WrongOperation("<cmds> has not any command to run.")
	dtimeout = timeout//runningProcess
	assert dtimeout >= 1,f"<timeout> is extremely short: {timeout}."
	for ID,p in processManager.items():
		try:
			out,err = p.communicate(timeout=dtimeout)
		except subprocess.TimeoutExpired:
			p.kill()
			errMes = b"Time Out Error: Process was killed! If you are exactly running the right program,"
			errMes += b"you can set a greater timeout value by exkaldi.info.set_timeout()."
			processManager[ID] = (-9,errMes)
		else:
			processManager[ID] = (p.returncode,err)

	return list(processManager.values())

def make_dependent_dirs(path,pathIsFile=True):
	'''
	Make the dependent directories for a path if it has not existed.

	Args:
		<path>: a file path or folder path.
		<pathIsFile>: a bool value to declare that <path> is a file path or folder path.
	'''
	declare.is_valid_string("path",path)
	declare.is_bool("pathIsFile",pathIsFile)

	path = os.path.abspath(path.strip())

	if pathIsFile:
		if os.path.isdir(path):
			raise WrongPath(f"<path> is specified as file but it has existed as directory: {path}. You can remove it then try again.")
		else:
			dirPath = os.path.dirname(path)
	else:
		if os.path.isfile(path):
			raise WrongPath(f"<path> is specified as directory but it has existed as file: {path}. You can remove it then try again.")
		else:
			dirPath = path
	
	if not os.path.isdir(dirPath):
		try:
			os.makedirs(dirPath)
		except Exception as e:
			print(f"Failed to make directory: {dirPath}.")
			raise e

def check_config(name,config=None):
	'''
	Check the users'configures or get the default configures of some functions.

	Args:
		<name>: function name.
		<config>: a list object whose keys are configure name and values are their configure values. If None,return the default configure.
	
	Return:
		if <config> is None:
			Return none,or a dict object of example configure of <name>.
			If the value is a tuple,it standards for multiple types of value you can set.
		else:
			Return True or raise error.
	'''
	declare.is_valid_string("name",name)

	try:
		module = importlib.import_module(f'exkaldi.config.{name}')
	except ModuleNotFoundError:
		print(f"Warning: no default configure for name '{name}'.")
		return None
	else:
		c = module.config

	if config is None:
		config = {}
		for key,value in c.items():
			value = tuple(value[i] for i in range(0,len(value),2))
			value = value if len(value) > 1 else value[0]
			config[key] = value
		return config

	else:
		if not isinstance(config,dict):
			raise WrongOperation(f"<config> has a wrong format. You can use check_config('{name}') to get expected configure format.")
		for k in config.keys():
			if not k in c.keys():
				raise WrongOperation(f"No such configure name: <{k}> in {name}.")
			else:
				protos = tuple( c[k][i] for i in range(1,len(c[k]),2) )
				if not isinstance(config[k],protos):
					if isinstance(config[k],bool):
						raise WrongDataFormat(f"Configure <{k}> is bool value: {config[k]},but we expected str value like 'true' or 'false'.")
					else:
						raise WrongDataFormat(f"Configure <{k}> should be in {protos} but got {type_name(config[k])}.")
			
			return True

def split_txt_file(filePath,chunks=2):
	'''
	Split a text file into N chunks by average number of lines.

	Args:
		<filePath>: text file path.
		<chunks>: an int avlue. How many chunks to split.

	Return:
		a list of paths of genrated chunk files.
		each file has a a prefix such as "ck0_" which _0_ is the chunk ID.
	'''
	declare.is_file("filePath",filePath)
	declare.greater_equal("chunks",chunks,"minimum chunk size",2)

	with open(filePath,'r',encoding='utf-8') as fr:
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

	filePath = os.path.abspath(filePath)
	dirName = os.path.dirname(filePath)
	fileName = os.path.basename(filePath)

	fileNamePattern = os.path.join(dirName,f"ck%0{a}d_"+fileName)
	newFiles = []
	start = 0
	for i in range(chunks):
		if i < t:
			end = start + chunkLines + 1
		else:
			end = start + chunkLines
		chunkData = data[start:end]
		newFileName = fileNamePattern%(i)
		with open(newFileName,'w',encoding='utf-8') as fw:
			fw.write( ''.join(chunkData) )

		newFiles.append(newFileName)
		start = end
	
	return newFiles

def compress_gz_file(filePath,overWrite=False,keepSource=False):
	'''
	Compress a file to gz file.

	Args:
		<filePath>: file path.
		<overWrite>: If True,overwrite gz file when it has existed.
		<keepSource>: If True,retain source file.
	
	Return:
		the path of compressed file.
	'''
	declare.is_file("filePath",filePath)
	declare.is_bool("overWrite",overWrite)
	declare.is_bool("keepSource",keepSource)

	filePath = os.path.abspath(filePath)
	if filePath.endswith(".gz"):
		raise WrongOperation(f"Cannot compress a .gz file:{filePath}.")
	else:
		outFile = filePath + ".gz"

	if os.path.isfile(outFile):
		if overWrite is True:
			os.remove(outFile)
		else:
			raise WrongOperation(f"File has existed:{outFile}. If overwrite it,set option <overWrite>=True.")

	if keepSource:
		cmd = f"gzip -k {filePath}"
	else:
		cmd = f"gzip {filePath}"

	out,err,cod = run_shell_command(cmd,stderr=subprocess.PIPE)
	
	if cod != 0:
		print(err.decode())
		raise ShellProcessError("Failed to compress file.")
	else:
		return outFile

def decompress_gz_file(filePath,overWrite=False,keepSource=False):
	'''
	Decompress a gz file.

	Args:
		<filePath>: file path.
		<overWrite>: If True,overwrite gz file when it has existed.
		<keepSource>: If True,retain source file.

	Return:
		file path of decompressed file.
	'''
	declare.is_file("filePath",filePath)
	declare.is_bool("overWrite",overWrite)
	declare.is_bool("keepSource",keepSource)

	filePath = os.path.abspath(filePath)
	if not filePath.endswith(".gz"):
		raise WrongOperation(f"{filePath}: Unknown suffix.")

	outFile = filePath[:-3]
	if os.path.isfile(outFile):
		if overWrite is True:
			os.remove(outFile)
		else:
			raise WrongOperation(f"File has existed:{outFile}. If overwrite it,set option <overWrite>=True.")

	if keepSource:
		cmd = f"gzip -d -k {filePath}"
	else:
		cmd = f"gzip -d {filePath}"

	out,err,cod = run_shell_command(cmd,stderr=subprocess.PIPE)

	if cod != 0:
		print(err.decode())
		raise ShellProcessError("Failed to decompress file.")
	else:
		return outFile

def flatten(item):
	'''
	Flatten an iterable object.

	Args:
		<item>: iterable objects,string,list,tuple or NumPy array.

	Return:
		a list of flattened items.
	'''
	if not isinstance(item,Iterable):
		return [item,]

	new = []
	for i in item:
		dtype = type_name(i)
		# python int or float value or Numpy float or int value.
		if dtype.startswith("int") or dtype.startswith("float"):
			new.append(i)
		# python str value.
		elif dtype.startswith("str"):
			if len(i) <= 1:
				new.append(i)
			else:
				new.extend(flatten(i))
		# python list,tuple,set object.
		elif dtype in ["list","tuple","set"]:
			new.extend(flatten(i))
		# Numpy ndarray object.
		elif dtype == "ndarray":
			if i.shape == ():
				new.append(i)
			else:
				new.extend(flatten(i))
		# Others objects is unsupported.
		else:
			raise UnsupportedType(f"Expected list,tuple,set,str or Numpy array object but got {type_name(i)}.")

	return new

def list_files(filePaths):
	'''
	List files by a normal grammar string.

	Args:
		<filePaths>: a string or list or tuple object.
	
	Return:
		A list of file paths.
	'''
	declare.is_classes("filePaths",filePaths,[str,list,tuple])

	def list_one_record(target):
		declare.is_valid_string("filePaths",target)
		cmd = f"ls {target}"
		out,err,cod = run_shell_command(cmd,stdout=subprocess.PIPE)
		if len(out) == 0:
			return []
		else:
			out = out.decode().strip().split("\n")
			newOut = [ o for o in out if os.path.isfile(o) ]
			return newOut

	if isinstance(filePaths,str):
		outFiles = list_one_record(filePaths)
	else:
		outFiles = []
		for m in filePaths:
			outFiles.extend( list_one_record(m) )
	
	if len(outFiles) == 0:
		raise WrongPath(f"No any files have been found through the provided file paths: {filePaths}.")
	
	return outFiles

def view_kaldi_usage(toolName):
	'''
	View the help information of specified kaldi command.

	Args:
		<toolName>: kaldi tool name.
	'''
	declare.is_valid_string("toolName",toolName)
	cmd = toolName.strip().split()
	assert len(cmd) == 1,f"<toolName> must only include one command name but got: {toolName}."
	cmd = cmd[0]
	cmd += " --help"

	out,err,cod = run_shell_command(cmd,stderr=subprocess.PIPE)
	
	if cod != 0:
		print(err.decode())
		raise ShellProcessError(f"Failed to get kaldi tool info: {toolName}.")
	else:
		print(err.decode())

class FileHandleManager:
	'''
	A class to create and manage opened file handles.
	A new FileHandleManager object should be instantiated bu python "with" grammar.
	'''
	def __init__(self):
		self.__inventory = {}
		self.__safeFlag = False

	@property
	def view(self):
		'''
		Return all handle names.
		'''
		return list(self.__inventory.keys())

	def create(self,mode,suffix=None,encoding=None,name=None):
		'''
		Creat a temporary file and return the handle.

		Args:
			<name>: a string. After named this handle exclusively,you can call its name to get it again.
					If None,we will use the file name as its default name.
		
		Return:
			a file handle.
		'''
		self.verify_safety()

		if suffix is not None:
			declare.is_valid_string("suffix",suffix)
	
		if name is not None:
			declare.is_valid_string("name",name)
			assert name not in self.__inventory.keys(),f"<name> has been existed. We hope it be exclusive: {name}."
		
		handle = tempfile.NamedTemporaryFile(mode,prefix="exkaldi_",suffix=suffix,encoding=encoding)

		if name is None:
			self.__inventory[handle.name] = handle
		else:
			self.__inventory[name] = handle

		return handle

	def open(self,filePath,mode,encoding=None,name=None):
		'''
		Open a regular file and return the handle.

		Args:
			<name>: a string. After named this handle exclusively,you can call its name to get it again.
					If None,we will use the file name as its default name.
					We allow to open the same file in multiple times as long as you name them differently.
		
		Return:
			a file handle.
		'''
		self.verify_safety()
		
		if name is not None:
			declare.is_valid_string("name",name)
			assert name not in self.__inventory.keys(),f"<name> has been existed. We hope it be exclusive: {name}."
		else:
			if filePath in self.__inventory.keys():
				raise WrongOperation(f"File has been opened already: {filePath}. If you still want to open it to get another handle,please give it an exclusive name.")
			name = filePath

		handle = open(filePath,mode,encoding=encoding)

		self.__inventory[name] = handle

		return handle

	def call(self,name):
		'''
		Get the file handle again by call its name.
		If unexisted,return None.
		'''
		declare.is_valid_string("name",name)
		try:
			return self.__inventory[name]
		except KeyError:
			return None

	def close(self,name=None):
		'''
		Close file handle.
		'''
		if name is None:
			for t in self.__inventory.values():
				try:
					t.close()
				except Exception:
					pass
		else:
			declare.is_valid_string("name",name)
			if name in self.__inventory.keys():
				try:
					self.__inventory[name].close()
				except Exception:
					pass

	def __enter__(self):
		self.__safeFlag = True
		return self
	
	def __exit__(self,type,value,trace):
		self.close()
	
	def verify_safety(self):
		if self.__safeFlag is False:
			raise WrongOperation("Please run the file handle manager under the 'with' grammar.")
