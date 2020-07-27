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
from exkaldi.utils.utils import make_dependent_dirs, flatten
from exkaldi.version import WrongOperation, WrongDataFormat
from collections import namedtuple

class ArgumentParser:

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
		self.__discription = "Arguments for Exkaldi program"

	def __capture(self):
		'''
		Capture the arguments.
		'''
		if self.__argv is None:
			self.__argv = sys.argv

	def discribe(self, message):
		'''
		Add a discription.

		Args:
			<message>: string.
		'''
		self.__capture()
		declare.is_valid_string("discription", message)
		self.__discription = message

	@property
	def spec(self):
		spec = namedtuple("Argument", ["dtype", "default", "choices", "minV", "maxV", "discription", "value"])
		spec.__new__.__defaults__ = (None,)
		return spec

	def __detect_special_char(self,item):
		item = flatten(item)
		assert "|" not in item, f"'|' is special char which is not be allowed to use in option definition."
		assert "," not in item, f"',' is special char which is not be allowed to use in option definition."

	def add(self,optionName,dtype,abbreviation=None,default=None,choices=None,minV=None,maxV=None,discription=None):
		'''
		Add a new option.
		'''
		self.__capture()

		# check the option name
		declare.is_valid_string("optionName", optionName)
		optionName = optionName.strip()
		self.__detect_special_char(default)
		assert optionName[0:2] == "--" and optionName[2:3] != "-", f"<optionName> must start with '--' but got: {optionName}."
		assert optionName != "--help", "<optionName> is inavaliable: --help."
		if optionName in self.__arguments.keys():
			raise WrongOperation(f"<optionName> is conflict: {optionName}.")
		
		# check dtype
		declare.is_instances("dtype", dtype, (float,int,bool,str))

		# check abbreviation
		if abbreviation is not None:
			declare.is_valid_string("abbreviation",abbreviation)
			abbreviation = abbreviation.strip()
			self.__detect_special_char(abbreviation)
			assert abbreviation[0:1] == "-" and abbreviation[1:2] != "-", f"<abbreviation> must start with '-' but got: {abbreviation}."
			assert abbreviation != "-h", "<abbreviation> is inavaliable: -h."
			if abbreviation in self.__abb2Name.keys():
				raise WrongOperation(f"<abbreviation> is conflict: {abbreviation}.")

		# check default
		if default is not None:
			if isinstance(default,(list,tuple)):
				declare.members_are_classes(f"default of {optionName}", default, dtype)
			else:
				declare.is_classes(f"default of {optionName}", default, dtype)
			if dtype == str:
				self.__detect_special_char(default)

		# check choices
		if choices is not None:
			declare.is_classes(f"choices of {optionName}", choices, (list,tuple))
			declare.members_are_classes(f"choices of {optionName}", choices, dtype)
			if dtype == str:
				self.__detect_special_char(choices)
			if default is not None:
				if isinstance(default,(list,tuple)):
					declare.members_are_instances(f"default of {optionName}", default, choices)
				else:
					declare.is_instances(f"default of {optionName}", default, choices)
		
		# check boundary values
		if minV is not None or  maxV is not None:
			assert dtype in [float,int], f"Only float and int option can set the boundary but {optionName} should be {dtype.__name__}"
			assert choices is None, f"Cannot set choices and boundary concurrently of {optionName}."
			if minV is not None:
				declare.is_classes(f"Minimum value of {optionName}", minV, dtype)
				if default is not None:
					if isinstance(default,(list,tuple)):
						for v in default:
							declare.larger(f"default of {optionName}", v, "minimum expected value", minV)
					else:
						declare.larger(f"default of {optionName}", default, "minimum expected value", minV)
			if maxV is not None:
				declare.is_classes(f"Maximum value of {optionName}", maxV, dtype)
				if default is not None:
					if isinstance(default,(list,tuple)):
						for v in default:					
							declare.smaller(f"default of {optionName}", v, "maximum expected value", maxV)
					else:
						declare.smaller(f"default of {optionName}", default, "maximum expected value", maxV)
			if minV is not None and maxV is not None:
				declare.smaller(f"Minimum value of {optionName}", minV, f"Maximum value", maxV)

		# check discription
		if discription is not None:
			declare.is_valid_string(f"discription of {optionName}", discription)
			self.__detect_special_char(discription)

		self.__arguments[optionName] = self.spec(dtype,default,choices,minV,maxV,discription)
		self.__name2Abb[optionName] = abbreviation
		if abbreviation is not None:
			self.__abb2Name[abbreviation] = optionName
	
	def print_help_and_exit(self):
		'''
		Print help information of all options.
		'''
		sys.stderr.write(self.save())
		sys.stderr.flush()
		sys.exit(1)

	def print_args(self):
		'''
		Print the arguments (command line) on stamdard output stream.
		'''
		self.__capture()
		print( " ".join (self.__argv ))

	def parse(self):
		'''
		Parse arguments.
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

				# option might has a format such as: 1|2
				vs = temp[i-1].split("|")
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
				
				if proto.choices is not None:
					declare.members_are_instances(f"option value of {op}", vs, proto.choices)
				else:
					if proto.minV is not None:
						for v in vs:
							declare.larger(f"option value of {op}", v, "minimum expected value", proto.minV)
					if proto.maxV is not None:
						for v in vs:
							declare.smaller(f"option value of {op}", v, "maximum expected value", proto.maxV)

				result[op] = vs if len(vs) > 1 else vs[0]

		# set attributes
		for name, value in result.items():
			if value is None:
				raise WrongOperation(f"Missed value for option: {op}.")
			else:
				self.__arguments[name] = self.__arguments[name]._replace(value=value)
				self.__setattr__(name[2:], value)
	
	def __setattr__(self, name, value):

		if '_ArgumentParser__arguments' in self.__dict__.keys():
			if "--"+name in self.__arguments.keys() and name in self.__dict__.keys():
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
							declare.larger(f"option value of {op}", v, "minimum expected value", proto.minV)
						if proto.maxV is not None:
							declare.smaller(f"option value of {op}", v, "maximum expected value", proto.maxV)
					self.__arguments[name] = proto._replace(value=value)
		super().__setattr__(name, value)

	def save(self, fileName=None):
		'''
		Save arguments to file with specified format.
		'''
		if fileName is not None:
			declare.is_valid_file_name("fileName", fileName)
			make_dependent_dirs(fileName, True)

		contents = []
		contents.append(self.__discription)
		for name, info in self.__arguments.items():
			#contents.append("\n")
			m = "\n"
			m += f"optionName={name}\n"

			if isinstance(info.value,(list,tuple)):
				value="|".join(map(str,info.value))
			else:
				value = info.value
			m += f"value={value}\n"

			m += f"abbreviation={self.__name2Abb[name]}\n"
			m += f"dtype={info.dtype.__name__}\n"

			if isinstance(info.default,(list,tuple)):
				default="|".join(map(str,info.default))
			else:
				default = info.default
			m += f"default={default}\n"

			if isinstance(info.choices,(list,tuple)):
				choices = "|".join(map(str,info.choices))
			else:
				choices = info.choices
			m += f"choices={choices}\n"

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
		'''
		declare.is_file("filePath", filePath)
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
			values = {"optionName":None,"abbreviation":None,"dtype":None,"default":None,"choices":None,"minV":None,"maxV":None,"discription":None,"value":None}
			for m in block:
				m = m.strip()
				assert "=" in m, f"Augument should has format: key = value, but got: {m}."
				assert len(m.split("=")) == 2, f"Augument should has format: key = value, but got: {m}."
				m = m.split("=")
				name = m[0].strip()
				value = m[1].strip()
				declare.is_instances("option key", name, list(values.keys()))
				
				values[name] = value

			for key, value in values.items():
				assert value is not None, f"Missed {key} information in line: {lineNo}."
			# 2. parse
			optionName = values["optionName"]
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
					choices[i] = __parse(optionName, c, dtype)
			values["choices"] = choices
			# then parse the boundary value
			boundary = {"minV":None, "maxV":None}
			for i in boundary.keys():
				V = values[i]
				if V not in ["none", "None"]:
					assert dtype in [float,int], f"Only float and int option can set the boundary but {optionName} is {dtype.__name__}"
					assert choices is None, f"{optionName} cannot set choices and boundary concurrently."
					
					toIntFlag = True
					toFloatFlag = True
					try:
						V = float(V)
					except ValueError:
						toFloatFlag= False
					try:
						V = int(V)
					except ValueError:
						toIntFlag= False
					
					if toIntFlag is False and toFloatFlag is False:
						raise WrongDataFormat(f"Boundary values of {optionName} should be a int or float value but got: {V}.")
					elif toIntFlag is False and toFloatFlag is True: # minV is predicted be a float value
						if dtype != float:
							raise WrongDataFormat(f"{optionName}'s dtype is int but try to set boundary value with a float value: {V}.")
						V = float(V)
					elif toIntFlag is True and toFloatFlag is True: # minV is predicted be a float or an int value
						V = dtype(V)
					else:
						raise WrongDataFormat(f"Failed to set {optionName}'s boundary value: {V}.")
				
					boundary[i] = V
			values["minV"] = boundary["minV"]
			values["maxV"] = boundary["maxV"]
			# then parse the default and value
			if values["default"] in ["none", "None"]:
				values["default"] = None
			else:
				default = values["default"].split("|")
				for i, v in enumerate(default):
					default[i] = __parse(optionName, v, dtype)
				values["default"] = default if len(default) > 1 else default[0]
			
			if values["value"] in ["none", "None"]:
				values["value"] = None
			else:
				value = values["value"].split("|")
				for i, v in enumerate(value):
					v = __parse(optionName, v, dtype)
					if values["choices"] is not None:
						declare.is_instances("option value", v, values["choices"])
					else:
						if values["minV"] is not None:
							declare.larger("option value", v, "minimum expected value", values["minV"])
						if values["maxV"] is not None:
							declare.smaller("option value", v, "maximum expected value", values["maxV"])
					value[i] = v
				values["value"] = value
			
			if values["abbreviation"] in ["none", "None"]:
				values["abbreviation"] = None

			self.add(optionName=values["optionName"], dtype=values["dtype"], abbreviation=values["abbreviation"], default=values["default"], 
					 choices=values["choices"], minV=values["minV"], maxV=values["maxV"], discription=values["discription"])
			
			self.__arguments[values["optionName"]] = self.__arguments[values["optionName"]]._replace(value=values["value"])
			if values["value"] is not None:
				self.__setattr__(values["optionName"], values["value"])

args = ArgumentParser()
