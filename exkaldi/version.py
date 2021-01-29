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

import subprocess
import os
import copy
from glob import glob
from collections import namedtuple

import sys
DIR=os.path.dirname(os.path.dirname(__file__))
sys.path.append(DIR)
from exkaldi.error import *

'''Version Control'''

_MAJOR_VERSION = '1'
_MINOR_VERSION = '3'
_PATCH_VERSION = '5'
_UPLOAD_VERSION = '1'

_EXPECTED_KALDI_VERSION = "5.5"

_TIMEOUT = 500

class ExKaldiInfo( namedtuple("ExKaldiInfo",["version","major","minor","patch","upload"]) ):
	'''
	Generate a object that carries various ExKaldi configurations.
	'''
	def initialize(self):
		'''
		Initialize.
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
		Get the ExKaldi version information.

		Return:
		  A namedtuple object.
		'''
		return self

	@property
	def KALDI(self):
		'''
		Get Kaldi version number. It will consult the ".version" file in Kaldi root path.

		Return:
			if Kaldi has not been found:
				return None.
			elif ".version" has not been found:
				return "unknown".
			else:
				return a named tuple of version number.
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
				if v != _EXPECTED_KALDI_VERSION:
					raise UnsupportedKaldiVersion(f"Current ExKaldi only supports Kaldi version=={_EXPECTED_KALDI_VERSION} but got {v}.")
				else:
					return namedtuple("Kaldi", ["version", "major", "minor"])(v, major, minor)

	@property
	def KALDI_ROOT(self):
		'''
		Get The kaldi root path. 
		We allow Kaldi does not exist now but we expect it will be appointed later. 

		Return:
			None if Kaldi has not been found in system PATH, or a string.
		'''
		if self.__KALDI_ROOT is None:
			cmd = "which copy-feats"
			p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ.copy())
			out, err = p.communicate()
			if out == b'':
				print("Warning: Kaldi root directory was not found in system PATH. You can appoint it:")
				print("exkaldi.info.reset_kaldi_root( yourPath )")
				print("If not, ERROR will occur when implementing some core functions.")
			else:
				self.__KALDI_ROOT = out.decode().strip()[0:-23]
				self.reset_kaldi_root(self.__KALDI_ROOT)

		return self.__KALDI_ROOT
	
	@property
	def ENV(self):
		'''
		Get the system environment in which ExKaldi are running.

		Return:
			a dict object.
		'''
		if self.__ENV is None:
			self.__ENV = os.environ.copy()

		# ENV is a dict object, so deepcopy it.
		return copy.deepcopy(self.__ENV)

	def reset_kaldi_root(self, path):
		'''
		Reset the root path of Kaldi toolkit and add related directories to system PATH manually.

		Args:
			<path>: a directory path.
		'''
		assert isinstance(path, str), "<path> should be a directory name-like string."
		path = path.strip()
		if not os.path.isdir(path):
			raise WrongPath(f"No such directory: {path}.")
		else:
			path = os.path.abspath(path)

		# verify this path roughly
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

		# collect new paths
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
		
		# reset the environment
		systemPATH = ":".join(systemPATH)
		self.__ENV['PATH'] = systemPATH

	def export_path(self, path):
		'''
		Add a path to ExKaldi environment PATH.
		
		Args:
			<path>: a path.
		'''
		if not os.path.exists(path):
			raise WrongPath(f"No such path: {path}.")

		systemPATH = self.ENV["PATH"]
		path = os.path.abspath(path)
		for p in systemPATH.split(":"):
			if p == path:
				return

		systemPATH += f":{path}"
		self.__ENV['PATH'] = systemPATH

	def prepare_srilm(self):
		'''
		Prepare SriLM toolkit and add it to ExKaldi system PATH.
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

	@property
	def timeout(self):
		return _TIMEOUT

	def set_timeout(self,timeout):
		'''
		Reset the global timeout value.

		Args:
			<timeout>: a positive int value.
		'''
		assert isinstance(timeout, int) and timeout > 0, f"<timeout> must be a positive int value but got: {timeout}."
		global _TIMEOUT
		_TIMEOUT = timeout

# initialize version infomation
info = ExKaldiInfo(
            '.'.join([_MAJOR_VERSION,_MINOR_VERSION,_PATCH_VERSION,_UPLOAD_VERSION]),
            _MAJOR_VERSION,
            _MINOR_VERSION,
            _PATCH_VERSION,
            _UPLOAD_VERSION,
        ).initialize()

# clear the temporary files possibly being left.
garbageFiles = glob(os.path.join(" ","tmp","exkaldi_*").lstrip())
for t in garbageFiles:
	os.remove(t)
