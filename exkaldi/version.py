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

from collections import namedtuple
import subprocess
import os
import copy
from glob import glob

'''Some Exception Classes'''

class WrongPath(Exception): pass
class WrongOperation(Exception):pass
class WrongDataFormat(Exception):pass
class ShellProcessError(Exception):pass
class KaldiProcessError(Exception):pass
class KenlmProcessError(Exception):pass
class UnsupportedType(Exception):pass
class UnsupportedKaldiVersion(Exception): pass

'''Version Control'''

_MAJOR_VERSION = '1'
_MINOR_VERSION = '0'

_EXPECTED_KALDI_VERSION = "5.5"

class ExKaldi( namedtuple("ExKaldi",["version","major","minor"]) ):

	def initialize(self):
		'''
		Initialize kaldi root information.
		'''
		self.__KALDI_ROOT = None
		self.__ENV = None
		self.__LOG_DIR = None
		
		# Update the root path of Kaldi
		_ = self.KALDI_ROOT
		# If Kaldi existed, check it's version
		if not self.__KALDI_ROOT is None:
			_ = self.KALDI
		
		return self
	
	@property
	def EXKALDI(self):
		'''
		Get the exkaldi version infomation.

		Return:
		    A named tuple.
		'''
		return self

	@property
	def KALDI(self):
		'''
		The kaldi version information. It will search the .version file in kaldi root path.

		Return:
			None, "unknown", or a string of version Number.
		'''
		if self.__KALDI_ROOT is None:
			print("Warning: Kaldi toolkit was not found.")
			return None
		else:
			filePath = os.path.join(self.__KALDI_ROOT, "src", ".version")
			if not os.path.isfile(filePath):
				print("Warning: Version information file was not found in Kaldi root directory.")
				return "unknown"
			else:
				with open(filePath, "r", encoding="utf-8") as fr:
					v = fr.readline().strip()
					major, minor = v.split(".")[0:2]
				if v != "5.5":
					raise UnsupportedKaldiVersion(f"Current Exkaldi supports Kaldi version=={_EXPECTED_KALDI_VERSION} but got {v}.")
				else:
					return namedtuple("Kaldi", ["version", "major", "minor"])(v, major, minor)

	@property
	def KALDI_ROOT(self):
		'''
		The kaldi root path. We allow Kaldi does not exist but we expect it will be assigned later. 

		Return:
			None if kaldi is not existed, or a path.
		'''
		if self.__KALDI_ROOT is None:
			cmd = "which copy-feats"
			p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ.copy())
			out, err = p.communicate()
			if out == b'':
				print("Warning: Kaldi root directory was not found in system PATH. You can assign it:")
				print("exkaldi.version.assign_kaldi_root( yourPath )")
				print("If not, ERROR will occur when implementing parts of core functions.")
			else:
				self.__KALDI_ROOT = out.decode().strip()[0:-23]
				self.assign_kaldi_root(self.__KALDI_ROOT) # In order to reset the environment

		return self.__KALDI_ROOT
	
	@property
	def ENV(self):
		'''
		System environment information.

		Return:
			a dict object.
		'''
		if self.__ENV is None:
			self.__ENV = os.environ.copy()

		# ENV is a dict object, so deepcopy it.
		return copy.deepcopy(self.__ENV)

	@property
	def LOG_DIR(self):
		'''
		Log dir path.

		Return:
			a path.
		Raise:
			If it has not been assigned, raise error. 
		'''		
		if self.__LOG_DIR is None:
			raise WrongPath("Log dir was not found. Please assign it firstly.")
		else:
			return self.__LOG_DIR
	
	def vertify_kaldi_existed(self):
		'''
		Vertify if kaldi toolkit is existed.

		Raise:
			If unexisted, raise error. 
		'''	
		if self.KALDI_ROOT is None:
			raise WrongPath("Kaldi was not found.")
		else:
			return True

	def assign_kaldi_root(self, path):
		'''
		Set the root directory of Kaldi toolkit and add it to system PATH manually.
		And it will also make a default log folder.

		Args:
			<path>: a directory.
		'''
		assert isinstance(path, str), "<path> should be a directory name-like string."
		
		if not os.path.isdir(path):
			raise WrongPath(f"No such directory: {path}.")
		else:
			path = os.path.abspath(path.strip())

		cmd = os.path.join(f"ls {path}", "src", "featbin", "copy-feats")
		p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ.copy())
		out, err = p.communicate()
		if out == b'':
			raise WrongPath(f"{path} is not kaldi path avaliable.")
		else:
			self.__KALDI_ROOT = path

		oldENV = self.ENV['PATH'] #deepcopied dict object
		systemPATH = []

		# Abandon old kaldi path of environment
		for i in oldENV.split(':'):
			if i.endswith( os.path.join("", "tools", "openfst") ):
				continue
			elif i.endswith( os.path.join("", "tools", "openfst", "bin") ):
				continue			
			elif i.endswith( os.path.join("", "src", "featbin") ):
				continue
			elif i.endswith( os.path.join("", "src", "GAMbian") ):
				continue
			elif i.endswith( os.path.join("", "src", "nnetbin") ):
				continue
			elif i.endswith( os.path.join("", "src", "bin") ):
				continue
			elif i.endswith( os.path.join("", "src", "lmbin") ):
				continue
			elif i.endswith( os.path.join("", "src", "fstbin") ):
				continue
			elif i.endswith( os.path.join("", "src", "latbin") ):
				continue
			elif i.endswith( os.path.join("", "src", "gmmbin") ):
				continue
			else:
				systemPATH.append(i)

		# Add new kaldi path of environment
		systemPATH.append( os.path.join(path, "src", "bin") )
		systemPATH.append( os.path.join(path, "tools", "openfst") )
		systemPATH.append( os.path.join(path, "tools", "openfst", "bin") )
		systemPATH.append( os.path.join(path, "src", "featbin") )
		systemPATH.append( os.path.join(path, "src", "GAMbian") )
		systemPATH.append( os.path.join(path, "src", "nnetbin") )
		systemPATH.append( os.path.join(path, "src", "lmbin") )
		systemPATH.append( os.path.join(path, "src", "fstbin") )
		systemPATH.append( os.path.join(path, "src", "latbin") )
		systemPATH.append( os.path.join(path, "src", "gmmbin") )
		
		# Assign new environment
		systemPATH = ":".join(systemPATH)
		self.__ENV['PATH'] = systemPATH

		# Make a default log dir
		self.assign_log_dir( path = os.path.join(self.__KALDI_ROOT, ".exkaldilog") )
	
	def assign_log_dir(self, path):
		'''
		Assign a log directory in order to place some log or temporary files.
		Every time when exkaldi is imported, check if this folder is existed. If existed, cleanup the log files.

		Args:
			<path>: a directory path.
		'''
		assert isinstance(path, str), f"<path> should be  string."
		
		dirPath = os.path.abspath(path.strip())

		if not os.path.isdir(dirPath):
			try:
				os.makedirs(dirPath)
			except Exception as e:
				print(f"Cannot make log directory:{dirPath}.")
				raise e
		
		self.__LOG_DIR = dirPath

		logFiles = glob(os.path.join(dirPath, "*.log" ))
		if len(logFiles) > 500:
			logFiles = sorted(logFiles)
			for i in logFiles[0:-200]: # Only keep the last 200 files.
				try:
					os.remove( i )
				except:
					continue

	def prepare_srilm(self):
		'''
		Prepare srilm toolkit and add it to system PATH.
		'''
		if self.KALDI_ROOT is None:
			raise WrongPath("Kaldi toolkit was not found.")
		else:
			SRILMROOT = os.path.join(self.KALDI_ROOT, "tools", "srilm")
			if not os.path.isdir(SRILMROOT):
				raise WrongPath("SRILM language model tool was not found. Please install it with KALDI_ROOT/tools/.install_srilm.sh .")

			systemPATH = []
			oldENV = self.ENV['PATH']
			# Abandon old srilm path of environment
			for i in oldENV.split(':'):
				if i.endswith('srilm'):
					continue
				elif i.endswith( os.path.join('srilm','bin') ):
					continue
				elif i.endswith( os.path.join('srilm','bin','i686-m64') ):
					continue		
				else:
					systemPATH.append(i)

			# Add new srilm path to environment
			systemPATH.append( SRILMROOT )
			systemPATH.append( os.path.join(SRILMROOT,'bin') )
			systemPATH.append( os.path.join(SRILMROOT,'bin','i686-m64') )

			systemPATH = ":".join(systemPATH)
			self.__ENV['PATH'] = systemPATH

version = ExKaldi(
            '.'.join([_MAJOR_VERSION,_MINOR_VERSION]),
            _MAJOR_VERSION,
            _MINOR_VERSION,
        ).initialize()