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

import sys
from exkaldi.utils import declare
from exkaldi.utils.utils import make_dependent_dirs, flatten, list_files
from exkaldi.error import WrongOperation, WrongDataFormat
from collections import namedtuple

class ArgumentParser:
	'''
	Parse and manage arguments.
	'''
	def __init__(self):
		self.reset()

	def reset(self):
		'''
		CLear data and reset.
		'''
		self.__arguments = {}
		self.__abb2Name = {}
		self.__name2Abb = {}
		self.__argv = None 
		self.__discription = "Arguments for ExKaldi program"

	def get_options(self):
		'''
		Check all added options.

		Return:
			a python dict object.
		'''
		return self.__arguments.copy()

	def __capture(self):
		'''
		Capture the arguments.
		'''
		if self.__argv is None:
			self.__argv = sys.argv.copy()

	def discribe(self, message):
		'''
		Add a discription of current program.

		Args:
			<message>: string.
		'''
		self.__capture()
		declare.is_valid_string("discription message", message)
		self.__discription = message

	@property
	def spec(self):
		'''
		Define the pattern to record one option.
		'''
		spec = namedtuple("Argument", ["dtype", "default", "choices", "minV", "maxV", "discription", "value"])
		spec.__new__.__defaults__ = (None,)
		return spec

	def __detect_special_char(self,item):
		item = flatten(item)
		assert "|" not in item, f"'|' is special char which is not be allowed to use in option definition."
		#assert "," not in item, f"',' is special char which is not be allowed to use in option definition."

	def add(self,name,dtype,abbr=None,default=None,choices=None,minV=None,maxV=None,discription=None):
		'''
		Add a new option.

		Args:
			_name_: a string which must have a format such as "--exkaldi" (but "--help" is inavaliable exceptionally.).  
			_dtype_: float, int, str or bool.  
			_abbr_: None or a abbreviation of name which must have a format such as "-e" (but "-h" is inavaliable exceptionally.).  
			_dtype_: the default value or a list/tuple of values.  
			_choices_: a list/tuple of values.  
			_minV_: set the minimum value if dtype is int or float. Enable when _choices_ is None.  
			_maxV_: set the maximum value if dtype is int or float. Enable when _choices_ is None.  
			_maxV_: a string to discribe this option.
		'''
		self.__capture()

		# check option name
		declare.is_valid_string("name",name)
		name = name.strip()
		self.__detect_special_char(name)
		assert name[0:2] == "--" and name[2:3] != "-", f"Option name must start with '--' but got: {name}."
		assert name != "--help", "Option name is inavaliable: --help."
		if name in self.__arguments.keys():
			raise WrongOperation(f"Option name has existed: {name}.")
		
		# check dtype
		declare.is_instances("option dtype", dtype, (float,int,bool,str))

		# check abbreviation
		if abbr is not None:
			declare.is_valid_string("abbr",abbr)
			abbr = abbr.strip()
			self.__detect_special_char(abbr)
			assert abbr[0:1] == "-" and abbr[1:2] != "-", f"Abbreviation must start with '-' but got: {abbr}."
			assert abbr != "-h", "Abbreviation is inavaliable: -h."
			if abbr in self.__abb2Name.keys():
				raise WrongOperation(f"Abbreviation has existed: {abbr}.")

		# check default value
		if default is not None:
			if isinstance(default,(list,tuple)):
				declare.members_are_classes(f"Default value of {name}", default, dtype)
			else:
				declare.is_classes(f"Default value of {name}", default, dtype)
			if dtype == str:
				self.__detect_special_char(default)

		# check choices
		if choices is not None:
			declare.is_classes(f"Choices of {name}", choices, (list,tuple))
			declare.members_are_classes(f"Choices of {name}", choices, dtype)
			if dtype == str:
				self.__detect_special_char(choices)
			if default is not None:
				if isinstance(default,(list,tuple)):
					declare.members_are_instances(f"Default value of {name}", default, choices)
				else:
					declare.is_instances(f"Default value of {name}", default, choices)
		
		# check boundary values
		if minV is not None or maxV is not None:
			assert dtype in [float,int], f"Only float and int option can set the boundary but {name} is {dtype.__name__}."
			assert choices is None, f"Cannot set choices and boundary concurrently: {name}."
			if minV is not None:
				declare.is_classes(f"Minimum value of {name}", minV, dtype)
				if default is not None:
					if isinstance(default, (list,tuple)):
						for v in default:
							declare.greater_equal(f"Default value of {name}", v, "minimum expected value", minV)
					else:
						declare.greater_equal(f"Default of {name}", default, "minimum expected value", minV)
			if maxV is not None:
				declare.is_classes(f"Maximum value of {name}", maxV, dtype)
				if default is not None:
					if isinstance(default,(list,tuple)):
						for v in default:					
							declare.less_equal(f"Default value of {name}", v, "maximum expected value", maxV)
					else:
						declare.less_equal(f"Default value of {name}", default, "maximum expected value", maxV)
			if minV is not None and maxV is not None:
				declare.less_equal(f"Minimum value of {name}", minV, f"maximum value", maxV)

		# check discription
		if discription is not None:
			declare.is_valid_string(f"Discription of {name}", discription)
			self.__detect_special_char(discription)

		self.__arguments[name] = self.spec(dtype,default,choices,minV,maxV,discription)
		self.__name2Abb[name] = abbr
		if abbr is not None:
			self.__abb2Name[abbr] = name
	
	def print_help_and_exit(self):
		'''
		Print help information of all options and stop program.
		'''
		sys.stderr.write(self.save())
		sys.stderr.flush()
		sys.exit(1)

	def print_args(self):
		'''
		Print the arguments (command line) on standard output stream.
		'''
		self.__capture()
		print(" ".join(self.__argv))

	def parse(self):
		'''
		Start to parse arguments.
		'''
		self.__capture()
		
		# extract arguments
		temp = self.__argv.copy()
		temp.reverse()
		newArgv = []
		for a in temp:
			if a.endswith(".py"):
				break
			a = a.split("=")
			a.reverse()
			newArgv.extend( a )

		# match these arguments
		result = dict( (key, proto.default) for key, proto in self.__arguments.items() )
		for i, op in enumerate(newArgv):

			if op[0:1] == "-" and op[1:2] != "-":
				if op == "-h":
					self.print_help_and_exit()
				if op not in self.__abb2Name.keys():
					raise WrongOperation(f"Option has not been defined: {op}.")
				else:
					op = self.__abb2Name[op]

			if op.startswith("--"):
				if op == "--help":
					self.print_help_and_exit()
				if op not in self.__arguments.keys():
					raise WrongOperation(f"Option has not been defined: {op}.")
				if i%2 == 0:
					raise WrongOperation(f"Missed value for option: {op}.")

				# option value might has a format such as: 1|2
				vs = newArgv[i-1].split("|")
				proto = self.__arguments[op]

				if proto.dtype in [float,int]:
					try:
						for i,v in enumerate(vs):
							vs[i] = proto.dtype(v)
					except ValueError:
						raise WrongOperation(f"Option <{op}> need a {proto.dtype.__name__} value but got: {v}.")

				elif proto.dtype == bool:
					for i,v in enumerate(vs):
						if v.lower() == "true":
							v = True
						elif v.lower() == "false":
							v = False
						else:
							raise WrongOperation(f"Option <{op}> need a bool value but got: {v}.")
						vs[i] = v
				
				# vs become a list
				if proto.choices is not None:
					declare.members_are_instances(f"Option value of {op}", vs, proto.choices)
				else:
					if proto.minV is not None:
						for v in vs:
							declare.greater_equal(f"Option value of {op}", v, "minimum expected value", proto.minV)
					if proto.maxV is not None:
						for v in vs:
							declare.less_equal(f"Option value of {op}", v, "maximum expected value", proto.maxV)

				result[op] = vs if len(vs) > 1 else vs[0]

		# set attributes
		for name, value in result.items():
			if value is None:
				raise WrongOperation(f"Missed value for option: {name}.")
			else:
				self.__arguments[name] = self.__arguments[name]._replace(value=value)
				self.__setattr__(name[2:], value)
	
	def __setattr__(self, name, value):

		if '_ArgumentParser__arguments' in self.__dict__.keys():
			if "--"+name in self.__arguments.keys() and name in self.__dict__.keys():
					# verify value
					proto = self.__arguments[name]
					if isinstance(value, (list,tuple)):
						temp = value
					else:
						temp = [value,]
					for v in temp:
						assert isinstance(v, proto.dtype), f"<{name}> need {proto.dtype.__name__} type value but got: {value}."
						if proto.choices is not None:
							assert v in proto.choices, f"<{name}> should be one of {proto.choices} but got: {value}."
						if proto.minV is not None:
							declare.greater_equal(f"option value of {name}", v, "minimum expected value", proto.minV)
						if proto.maxV is not None:
							declare.less_equal(f"option value of {name}", v, "maximum expected value", proto.maxV)
					# modify the backup
					self.__arguments[name] = proto._replace(value=value)

		super().__setattr__(name, value)

	def save(self, fileName=None):
		'''
		Save arguments to file with specified format.

		Args:
			_fileName_: Nonr, a resonable file name.
		
		Return:
			if fileName is None:
				return a string of all contents
			else:
				the saved file name
		'''
		if fileName is not None:
			declare.is_valid_file_name("file name", fileName)
			make_dependent_dirs(fileName, pathIsFile=True)

		contents = []
		contents.append(self.__discription)
		for name, info in self.__arguments.items():
			# option name
			m = "\n"
			m += f"name={name}\n"
			# option value
			if isinstance(info.value,(list,tuple)):
				value="|".join(map(str,info.value))
			else:
				value = info.value
			m += f"value={value}\n"
			# abbreviation and dtype
			m += f"abbr={self.__name2Abb[name]}\n"
			m += f"dtype={info.dtype.__name__}\n"
			# default
			if isinstance(info.default,(list,tuple)):
				default="|".join(map(str,info.default))
			else:
				default = info.default
			m += f"default={default}\n"
			# choices
			if isinstance(info.choices,(list,tuple)):
				choices = "|".join(map(str,info.choices))
			else:
				choices = info.choices
			m += f"choices={choices}\n"
			# boundary and discription
			m += f"minV={info.minV}\n"
			m += f"maxV={info.maxV}\n"
			m += f"discription={info.discription}"
			contents.append(m)
		
		contents = "\n".join(contents) + "\n"

		if fileName is not None:
			with open(fileName, "w", encoding="utf-8") as fw:
				fw.write(contents)
			return fileName
		else:
			return contents

	def load(self, filePath):
		'''
		Load auguments from file.

		Args:
			_filePath_: args file path.
		'''
		declare.is_file("file path", filePath)
		self.reset()

		with open(filePath, "r", encoding="utf-8") as fr:
			lines = fr.read()
		lines = lines.strip()
		if len(lines) == 0:
			raise WrongOperation(f"This is a void file: {filePath}.")
		
		blocks = lines.split("\n\n")
		
		def __parse(name, value, dtype):
			if dtype in [float,int]:
				try:
					value = dtype(value)
				except ValueError:
					raise WrongOperation(f"Option <{name}> need a {dtype.__name__} value but choices got: {value}.")
			elif dtype == bool:
				if value.lower() == "true":
					value = True
				elif c.lower() == "false":
					value = False
				else:
					raise WrongOperation(f"Option <{name}> need a bool value but choices got: {value}.")

			return value  

		self.__discription = blocks[0].strip()
		for blockNo, block in enumerate(blocks[1:], start=1):
			block = block.strip()
			if len(block) == 0:
				continue
			block = block.split("\n")
			# 1. match options
			values = {"name":None,"abbr":None,"dtype":None,"default":None,"choices":None,"minV":None,"maxV":None,"discription":None,"value":None}
			for m in block:
				m = m.strip()
				assert "=" in m, f"Augument should has format: key = value, but got: {m}."
				assert len(m.split("=")) == 2, f"Augument should has format: key = value, but got: {m}."
				m = m.split("=")
				name = m[0].strip()
				value = m[1].strip()
				declare.is_instances("Option key", name, list(values.keys()))
				values[name] = value

			for key, value in values.items():
				assert value is not None, f"Missed {key} information in line: {lineNo}."
			# 2. parse
			name = values["name"]
			# parse the dtype firstly
			declare.is_instances("dtype", values["dtype"], ["float","int","bool","str"])
			values["dtype"] = eval(values["dtype"])
			dtype = values["dtype"]	
			# then parse the choices
			choices = values["choices"]
			if choices in ["none", "None"]:
				choices = None
			else:
				choices = choices.split("|")
				for i, c in enumerate(choices):
					choices[i] = __parse(name, c, dtype)
			values["choices"] = choices
			# then parse the boundary value
			boundary = {"minV":None, "maxV":None}
			for i in boundary.keys():
				V = values[i]
				if V not in ["none", "None"]:
					assert dtype in [float,int], f"Only float and int option can set the boundary but {name} is {dtype.__name__}:"
					assert choices is None, f"{name} cannot set choices and boundary concurrently."
					
					toIntFlag = True
					toFloatFlag = True
					try:
						float(V)
					except ValueError:
						toFloatFlag= False
					try:
						int(V)
					except ValueError:
						toIntFlag= False
					
					if toIntFlag is False and toFloatFlag is False:
						raise WrongDataFormat(f"Boundary values of {name} should be a int or float value but got: {V}.")
					elif toIntFlag is False and toFloatFlag is True: # minV is predicted be a float value
						if dtype != float:
							raise WrongDataFormat(f"{name}'s dtype is int but try to set boundary value with a float value: {V}.")
						else:
							V = float(V)
					elif toIntFlag is True and toFloatFlag is True: # minV is predicted be a float or an int value
						V = dtype(V)
					else:
						raise WrongDataFormat(f"Failed to set {name}'s boundary value: {V}.")
				
					boundary[i] = V
			values["minV"] = boundary["minV"]
			values["maxV"] = boundary["maxV"]
			# then parse the default and value
			if values["default"].lower() == "none":
				values["default"] = None
			else:
				default = values["default"].split("|")
				for i, v in enumerate(default):
					default[i] = __parse(name, v, dtype)
				values["default"] = default if len(default) > 1 else default[0]
			
			# the judgement of "default" will be done by .parse() function, so here we only verify "value"
			if values["value"].lower() == "none":
				values["value"] = None
			else:
				value = values["value"].split("|")
				for i, v in enumerate(value):
					v = __parse(name, v, dtype)
					if values["choices"] is not None:
						declare.is_instances("Option value", v, values["choices"])
					else:
						if values["minV"] is not None:
							declare.greater_equal("Option value", v, None, values["minV"])
						if values["maxV"] is not None:
							declare.less_equal("Option value", v, None, values["maxV"])
					value[i] = v
				if len(value) == 1:
					value = value[0]
				values["value"] = value
			
			# check abbreviation
			if values["abbr"] in ["none", "None"]:
				values["abbr"] = None

			# add this options
			self.add(name=values["name"], 
							 dtype=values["dtype"], 
							 abbr=values["abbr"], 
							 default=values["default"], 
					 		 choices=values["choices"], 
							 minV=values["minV"], 
							 maxV=values["maxV"], 
							 discription=values["discription"]
							)
			
			# finally, modify the "value"
			self.__arguments[values["name"]] = self.__arguments[values["name"]]._replace(value=values["value"])
			if values["value"] is not None:
				self.__setattr__(values["name"], values["value"])

args = ArgumentParser()

def load_args(target):
	'''
	Load arguments from file.

	Args:
		<target>:file path.
	
	Return:
		an ArgumentParser object.
	'''
	fileName = list_files(target)
	assert len(fileName) == 1, "Cannot load arguments from multiple files."

	global args

	args.reset()
	args.load(fileName[0])

	return args