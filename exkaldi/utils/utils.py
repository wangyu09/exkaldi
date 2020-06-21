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
from exkaldi.version import WrongPath, WrongOperation, WrongDataFormat, KaldiProcessError, ShellProcessError, UnsupportedType

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
		raise UnsupportedType("<cmd> should be a string,  or a list of commands and its' options.")
	
	if env is None:
		env = ExkaldiInfo.ENV

	if inputs is not None:
		if isinstance(inputs, str):
			inputs = inputs.encode()
		elif isinstance(inputs, bytes):
			pass
		else:
			raise UnsupportedType(f"Expected <inputs> is string or bytes object but got {type_name(inputs)}.")

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
	assert isinstance(path, str), "<path> should be a string."
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

def print_message(*args, **kwargs):
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
		print(*args, **kwargs)

def check_config(name, config=None):
	'''
	Check if the users' configure is right when call some functions.

	Args:
		<name>: function name.
		<config>: a list object whose items are configure name and their configure values. If None, return the default configure.
	
	Return:
		If <config> is None, return a dict object of example configure of <name>.
			If the value is a tuple, it standards for mutiple types of value you can set.
		Or return True or False.
		Or None if <name> is unavaliable.
	'''
	assert isinstance(name, str), "<name> should be a name-like string."

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
		if not isinstance(config, dict):
			raise WrongOperation(f"<config> has a wrong format. You can use check_config('{name}') to look expected configure format.")
		for k in config.keys():
			if not k in c.keys():
				raise WrongOperation(f"No such configure name: <{k}> in {name}.")
			else:
				protos = tuple( c[k][i] for i in range(1,len(c[k]),2) )
				if not isinstance(config[k], protos):
					if isinstance(config[k],bool):
						raise WrongDataFormat(f"Configure <{k}> is bool value: {config[k]}, but we expected str value like 'true' or 'false'.")
					else:
						raise WrongDataFormat(f"Configure <{k}> should be in {protos} but got {type_name(config[k])}.")
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

def utt2spk_to_spk2utt(utt2spkFile, outFile):
	'''
	Transform utt2spk file to spk2utt file.

	Args:
		<utt2spkFile>: file name.
		<outFile>: file name.
	'''
	assert isinstance(utt2spkFile, str), f"<utt2spkFile> should be a string but got: {type_name(utt2spkFile)}."
	assert isinstance(outFile, str), f"<outFile> should be a string but got: {type_name(outFile)}."

	if not os.path.isfile(utt2spkFile):
		raise WrongPath(f"No such file: {utt2spkFile}.")
	
	spk2utt = {}
	with open(utt2spkFile, "r", encoding="utf-8") as fr:
		lines = fr.readlines()
	for index,line in enumerate(lines,start=1):
		line = line.strip().split()
		if len(line) == 0:
			continue
		else:
			if len(line) != 2:
				raise WrongDataFormat(f"Mismatching between utt and spk: {line}, line {index}.")
			utt, spk = line[0], line[1]
			if spk not in spk2utt.keys():
				spk2utt[spk] = f"{utt}"
			else:
				spk2utt[spk] += f" {utt}"

	with open(outFile, "w") as fw:
		fw.write( "\n".join(map(lambda spk,utts:spk+" "+utts, spk2utt.items())) )
	
	return os.path.abspath(outFile)

def spk2utt_to_utt2spk(spk2uttFile, outFile):
	'''
	Transform spk2utt file to utt2spk file.

	Args:
		<spk2uttFile>: file name.
		<outFile>: file name.
	'''
	assert isinstance(spk2uttFile, str), f"<spk2uttFile> should be a string but got: {type_name(spk2uttFile)}."
	assert isinstance(outFile, str), f"<outFile> should be a string but got: {type_name(outFile)}."

	if not os.path.isfile(spk2uttFile):
		raise WrongPath(f"No such file: {spk2uttFile}.")
	
	utt2spk = {}
	with open(spk2uttFile, "r", encoding="utf-8") as fr:
		lines = fr.readlines()
	for index,line in enumerate(lines,start=1):
		line = line.strip().split()
		if len(line) == 0:
			continue
		else:
			if len(line) < 2:
				raise WrongDataFormat(f"Mismatching between utt and spk: {line}.")
			spk = line[0]
			for utt in line[1:]:
				if utt in utt2spk.keys():
					raise WrongDataFormat(f"utt:{utt} is repeated in line {index}.")
				utt2spk[utt] = spk

	with open(outFile, "w") as fw:
		fw.write( "\n".join(map(lambda utt,spk:utt+" "+spk, utt2spk.items())) )
	
	return os.path.abspath(outFile)

def compress_gz_file(filePath, overWrite=False):
	'''
	Compress a file to gz file.

	Args:
		<filePath>: file path.
		<overWrite>: If True, overwrite gz file when it is existed.
	Return:
		the absolute path of compressed file.
	'''
	assert isinstance(filePath, str), f"<filePath> must be a string but got {type_name(filePath)}."
	filePath = filePath.strip()
	if not os.path.isfile(filePath):
		raise WrongPath(f"No such file:{filePath}.")

	outFile = filePath + ".gz"
	if overWrite is True and os.path.isfile(outFile):
		os.remove(outFile) 

	cmd = f"gzip {filePath}"
	out, err, cod = run_shell_command(cmd, stderr=subprocess.PIPE)
	
	if cod != 0:
		print(err.decode())
		raise ShellProcessError("Failed to compress file.")
	else:
		return os.path.abspath(outFile)

def decompress_gz_file(filePath, overWrite=False):
	'''
	Decompress a gz file.

	Args:
		<filePath>: file path.
	Return:
		the absolute path of decompressed file.
	'''
	assert isinstance(filePath, str), f"<filePath> must be a string but got {type_name(filePath)}."
	filePath = filePath.rstrip()
	if not os.path.isfile(filePath):
		raise WrongPath(f"No such file:{filePath}.")
	elif not filePath.endswith(".gz"):
		raise WrongOperation(f"{filePath}: Unknown suffix.")

	outFile = filePath[:-3]
	if overWrite is True and os.path.isfile(outFile):
		os.remove(outFile) 

	cmd = f"gzip -d {filePath}"
	out, err, cod = run_shell_command(cmd, stderr=subprocess.PIPE)

	if cod != 0:
		print(err.decode())
		raise ShellProcessError("Failed to decompress file.")
	else:
		return os.path.abspath(outFile)

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
		# python list, tuple, set object.
		elif dtype in ["list", "tuple", "set"]:
			new.extend(flatten(i))
		# Numpy ndarray object.
		elif dtype == "ndarray":
			if i.shape == ():
				new.append(i)
			else:
				new.extend(flatten(i))
		# Others objects is unsupported.
		else:
			raise UnsupportedType(f"Expected list, tuple, set, str or Numpy array object but got {type_name(i)}.")

	return new

def list_files(fileName):
	'''
	List file paths.

	Args:
		<fileName>: a string.
	
	Return:
		A list of file paths.
	'''
	assert isinstance(fileName, str), f"<fileName> should be string but got: {fileName}."

	cmd = f"ls {fileName}"
	out, err, cod = run_shell_command(cmd, stdout=subprocess.PIPE)

	if len(out) == 0:
		raise WrongPath(f"No such file: {fileName}.")
	else:
		out = out.decode().strip()
		return out.split("\n")
