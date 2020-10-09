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

'''This package includes some function to check format of arguments.'''

import os
from exkaldi.version import info as ExKaldiInfo
from collections import Iterable
import inspect

class DeclareError(Exception):pass

def declare_wrapper(func):
	assert inspect.getfullargspec(func).defaults is None,f"Declare function {func.__name__} cannot has default value!"
	def inner(*args,**kwargs):
		if "debug" in kwargs.keys():
			errMes = kwargs.pop("debug")
		else:
			errMes = None
		# Allows missing "name" parameter.
		argsNames = inspect.getfullargspec(func).args
		needNums = len(argsNames)
		if "name" in argsNames:
			if len(args) + len(kwargs) == needNums - 1:
				args = ("target",) + args
		# Allows customizing error message.
		if errMes is not None:
			try:
				return func(*args,**kwargs)
			except AssertionError as e:
				raise DeclareError(str(errMes))
		else:
			return func(*args,**kwargs)

	return inner

def __type_name(obj):
	return obj.__class__.__name__

@declare_wrapper
def kaldi_existed():
	'''
	Verify whether or not Kaldi toolkit has existed in system environment.
	'''
	assert ExKaldiInfo.KALDI_ROOT is not None,"Kaldi toolkit was not found in system PATH."

# data type
@declare_wrapper
def is_valid_string(name,string):
	'''
	Verify whether or not this is a reasonable string.
	
	Args:
		<name>: a string,the name of this object.
		<string>: python object.
	'''
	assert isinstance(string,str),f"{name} is not a valid string: {__type_name(string)}."
	assert len(string.strip())>0,f"{name} is a void string: {string}."

@declare_wrapper
def members_are_valid_strings(name,strings):
	'''
	Verify whether or not these are reasonable strings.
	'''
	assert isinstance(strings,Iterable),f"{name} is not iterable: {__type_name(strings)}."

	for string in strings:
		is_valid_string(name,string)

@declare_wrapper
def is_file(name,filePath):
	'''
	Verify whether or not this is a existed file.
	'''
	is_valid_string(f"File path: {name}",filePath)
	assert not os.path.isdir(filePath),f"{name} is not a file but a directory: {filePath}."
	assert os.path.isfile(filePath),f"No such file: {filePath}."

@declare_wrapper
def is_dir(name,dirPath):
	'''
	Verify whether or not this is a existed folder.
	'''
	is_valid_string(f"Directory path: {name}",dirPath)
	assert not os.path.isfile(dirPath),f"{name} is not drectory but a file: {dirPath}."
	assert os.path.isdir(dirPath),f"No such directory: {dirPath}."

@declare_wrapper
def members_are_files(name,filePaths):
	'''
	Verify whether or not these are existed files.
	'''
	assert isinstance(filePaths,Iterable),f"{name} is not iterable: {__type_name(filePaths)}."

	for filePath in filePaths:
		is_file(name,filePath)

@declare_wrapper
def is_classes(name,obj,targetClasses):
	'''
	Verify whether or not this object is an instance of these classes (do not include subclasses).

	Args:
		<targetClass>: class,class name,list or tuple of classes,list or tuple of names of classes.
	'''
	className = []
	if isinstance(targetClasses,(list,tuple)):
		for c in targetClasses:
			if isinstance(c,str):
				className.append( c )
			else:
				className.append( c.__name__ )
	else:
		if isinstance(targetClasses,str):
			className.append( targetClasses )
		else:
			className.append( targetClasses.__name__ )
	classNameCon = ",".join(className)
	
	assert __type_name(obj) in className,f"{name} is not an instance included in [{classNameCon}] classes: {__type_name(obj)}"

@declare_wrapper
def members_are_classes(name,objs,targetClasses):
	'''
	Verify whether or not these objects are instances of these classes (do not include subclasses).
	'''
	assert isinstance(objs,Iterable),f"{name} is not iterable: {__type_name(objs)}."

	className = []
	if isinstance(targetClasses,(list,tuple)):
		for c in targetClasses:
			if isinstance(c,str):
				className.append( c )
			else:
				className.append( c.__name__ )
	else:
		if isinstance(targetClasses,str):
			className.append( targetClasses )
		else:
			className.append( targetClasses.__name__ )
	classNameCon = ",".join(className)
	
	for obj in objs:
		assert __type_name(obj) in className,f"{name} are not instances included [{classNameCon}] classes: {__type_name(obj)}"

@declare_wrapper
def belong_classes(name,obj,targetClasses):
	'''
	Verify whether or not this object is an instance of these classes and their subclasses.
	'''   
	if not isinstance(targetClasses,(list,tuple)):
		targetClasses = (targetClasses,)
	className = [ c.__name__ for c in targetClasses]
	className = ",".join(className)

	targetClasses = tuple(targetClasses)
		
	assert isinstance(obj,targetClasses),f"{name} is not an instance included in [{className}] clases or their subclasses but: {__type_name(obj)}"

@declare_wrapper
def members_belong_classes(name,objs,targetClasses):
	'''
	Verify whether or not these objects are instances of these classes and their subclasses.
	'''
	assert isinstance(objs,Iterable),f"{name} is not a iterable object: {__type_name(objs)}."

	if not isinstance(targetClasses,(list,tuple)):
		targetClasses = (targetClasses,)
	className = [ c.__name__ for c in targetClasses]
	className = ",".join(className)

	targetClass = tuple(targetClasses)
	
	for obj in objs:
		assert isinstance(obj,targetClasses),f"{name} are not instances included in [{className}] clases or their subclasses but: {__type_name(obj)}"

@declare_wrapper
def is_instances(name,obj,targetInstances):
	'''
	Verify whether or not this object is one of target instances.
	'''
	if not isinstance(targetInstances,(list,tuple)):
		targetInstances = [targetInstances,]

	assert obj in targetInstances,f"{name} does not exist in {targetInstances}: {obj}"

@declare_wrapper
def members_are_instances(name,objs,targetInstances):
	'''
	Verify whether or not each of these objects is one of target instances.
	'''
	assert isinstance(objs,Iterable),f"{name} is not a iterable object: {__type_name(objs)}."
	
	for obj in objs:
		is_instances(name,obj,targetInstances)

@declare_wrapper
def not_void(name,obj):
	'''
	Verify whether or not this is a void object.
	'''
	assert obj is not None, f"{name} is None."
	if __type_name(obj) in ["list","tuple","dict","str"]:
		assert len(obj) > 0,f"{name} has nothing provided."
	else:
		assert "is_void" in dir(obj),f"Cannot decide whether or not {__type_name(obj)} object is void with this function."
		assert not obj.is_void,f"{name} is void. Can not operate it."

@declare_wrapper
def is_valid_file_name_or_handle(name,obj):
	'''
	Verify whether or not this is a resonable file name or file handle.
	'''
	assert __type_name(obj) in ["str","TextIOWrapper","_TemporaryFileWrapper"],f"{name} should be a file name or an opened file handle but got: {__type_name(obj)}."

	if isinstance(obj,str):
		obj = obj.strip()
		assert len(obj) > 0,f"{name} is a void string: {obj}."
		assert not os.path.isdir(obj),f"{name} has been existed as a directory: {obj}."
	else:
		obj.seek(0)
		assert len(obj.read()) == 0,f"When {name} is an opened file handle,this file should be void."
		obj.seek(0)

@declare_wrapper
def is_valid_file_name(name,fileName):
	'''
	Verify whether or not this is a file name avaliable.
	'''
	is_valid_string(name,fileName)
	fileName = fileName.strip()
	assert not os.path.isdir(fileName),f"{name} has been existed as a directory: {fileName}."

@declare_wrapper
def is_valid_dir_name(name,dirName):
	'''
	Verify whether or not this is a directory name avaliable.
	'''
	is_valid_string(name,dirName)
	dirName = dirName.strip()
	assert not os.path.isfile(dirName),f"{name} has been existed as a file: {dirName}."

# value
@declare_wrapper
def in_boundary(name,value,minV,maxV):
	'''
	Verify whether or not this value is whithin the boundary.

	If both or <minV> and <maxV> are not float value,the <value> will be declared as int type.
	'''    
	assert isinstance(minV,(int,float)),f"Boundary min value is invalid: {minV}."
	assert isinstance(maxV,(int,float)),f"Boundary max value is invalid: {maxV}."

	if isinstance(minV,int) and isinstance(maxV,int):
		assert isinstance(value,int),f"{name} should be an int value but got: {value}."
	else:
		assert isinstance(value,(int,float)),f"{name} should be an int or float value but got: {value}."

	assert minV <= value <= maxV,f"{name} should be an value within {minV}~{maxV} but got: {value}."

@declare_wrapper
def equal(nameA,valueA,nameB,valueB):
	'''
	Verify whether or not these two value are equal.
	'''
	if nameB is None:
		assert valueA == valueB,f"{nameA} should be {valueA} but got: {valueB}."
	else:
		assert valueA == valueB,f"{nameA},{valueA} does not match the {nameB},{valueB}."

@declare_wrapper
def greater(nameA,valueA,nameB,valueB):
	'''
	Verify whether or not value A is greater than value B.
	'''
	if nameB is None:
		assert valueA > valueB,f"{nameA} should be greater than {valueB} but got: {valueA}."
	else:	
		assert valueA > valueB,f"{nameA} should be greater than {nameB} but got: {valueA} <= {valueB}."

@declare_wrapper
def greater_equal(nameA,valueA,nameB,valueB):
	'''
	Verify whether or not value A is not less than value B.
	'''
	if nameB is None:
		assert valueA >= valueB,f"{nameA} should be not less than {valueB} but got: {valueA}."
	else:	
		assert valueA >= valueB,f"{nameA} should be not less than {nameB} but got: {valueA} < {valueB}."

@declare_wrapper
def less(nameA,valueA,nameB,valueB):
	'''
	Verify whether or not value A is less than value B.
	'''  
	if nameB is None:
		assert valueA < valueB,f"{nameA} should be less than {valueB} but got: {valueA}."
	else:	
		assert valueA < valueB,f"{nameA} should be less than {nameB} but got: {valueA} >= {valueB}."

@declare_wrapper
def less_equal(nameA,valueA,nameB,valueB):
	'''
	Verify whether or not value A is not greater than value B.
	'''
	if nameB is None:
		assert valueA <= valueB,f"{nameA} should be not greater than {valueB} but got: {valueA}."
	else:	
		assert valueA <= valueB,f"{nameA} should be not greater than {nameB} but got: {valueA} > {valueB}."

@declare_wrapper
def is_positive(name,value):
	'''
	Verify whether or not value is a positive int or float value.
	'''  
	assert isinstance(value,(int,float)) and value > 0,f"{name} should be a positive int or float value but got: {value}."

@declare_wrapper
def is_positive_int(name,value):
	'''
	Verify whether or not value is a positive int value.
	'''  
	assert isinstance(value,int) and value > 0,f"{name} should be a positive int value but got: {value}."

@declare_wrapper
def is_positive_float(name,value):
	'''
	Verify whether or not value is a positive float value.
	'''  
	assert isinstance(value,float) and value > 0,f"{name} should be a positive float value but got: {value}."

@declare_wrapper
def is_non_negative(name,value):
	'''
	Verify whether or not value is a non-negative float or int value.
	'''  
	assert isinstance(value,(int,float)) and value >= 0,f"{name} should be a non-negative int or float value but got: {value}."

@declare_wrapper
def is_non_negative_int(name,value):
	'''
	Verify whether or not value is a non-negative int value.
	'''  
	assert isinstance(value,int) and value >= 0,f"{name} should be a non-negative int value but got: {value}."

@declare_wrapper
def is_non_negative_float(name,value):
	'''
	Verify whether or not value is a non-negative float value.
	'''  
	assert isinstance(value,float) and value >= 0,f"{name} should be a non-negative float value but got: {value}."

@declare_wrapper
def is_bool(name,value):
	'''
	Verify whether or not value is a bool value.
	'''  
	assert isinstance(value,bool),f"{name} should be a bool value but got: {value}."

@declare_wrapper
def is_callable(name,obj):
	'''
	Verify whether or not obj is callable.
	'''  
	assert callable(obj),f"{name} is not callable."

# special for Exkaldi
@declare_wrapper
def is_index_table(name,indexTable):
	'''
	Verify whether or not this is an Exkaldi IndexTable object.
	'''
	assert __type_name(indexTable) == "IndexTable",f"{name} should be exkaldi index table object but got: {__type_name(indexTable)}."

@declare_wrapper
def is_matrix(name,mat):
	'''
	Verify whether or not this is a reasonable Exkaldi matrix archive object that is IndexTable or NumpyMatrix or BytesMatrix object.
	'''
	targetClasses = ["IndexTable","NumpyMatrix","BytesMatrix"]

	is_classes(f"Exkaldi matrix data: {name}",mat,targetClasses)

@declare_wrapper
def is_vector(name,vec):
	'''
	Verify whether or not this is a reasonable Exkaldi vector archive object that is IndexTable or NumpyVector or BytesVector object.
	'''
	targetClasses = ["IndexTable","NumpyVector","BytesVector"]

	is_classes(f"Exkaldi vector data: {name}",vec,targetClasses)

@declare_wrapper
def is_feature(name,feat):
	'''
	Verify whether or not this is a reasonable Exkaldi feature archive object that is IndexTable or NumpyFeat or BytesFeat object.
	'''
	targetClasses = ["IndexTable","NumpyFeat","BytesFeat"]

	is_classes(f"Exkaldi feature data: {name}",feat,targetClasses)

@declare_wrapper
def is_probability(name,prob):
	'''
	Verify whether or not this is a reasonable Exkaldi probability archive object that is IndexTable or NumpyProb or BytesProb object.
	'''
	targetClasses = ["IndexTable","BytesProb","NumpyProb"]

	is_classes(f"Exkaldi probability data: {name}",prob,targetClasses)

@declare_wrapper
def is_cmvn(name,cmvn):
	'''
	Verify whether or not this is a reasonable Exkaldi CMVN archive object that is IndexTable or NumpyCMVN or BytesCMVN object.
	'''
	targetClasses = ["IndexTable","BytesCMVN","NumpyCMVN"]

	is_classes(f"Exkaldi CMVN data: {name}",cmvn,targetClasses)

@declare_wrapper
def is_fmllr_matrix(name,fmllrMat):
	'''
	Verify whether or not this is a reasonable Exkaldi CMVN archive object that is IndexTable or NumpyFmllr or BytesFmllr object.
	'''
	targetClasses = ["IndexTable","BytesFmllr","NumpyFmllr"]

	is_classes(f"Exkaldi fmllr transform matrix: {name}",fmllrMat,targetClasses)

@declare_wrapper
def is_alignment(name,ali):
	'''
	Verify whether or not this is a reasonable Exkaldi transition alignment archive object that is IndexTable or NumpyAliTrans or BytesAliTrans object.
	'''
	targetClasses = ["IndexTable","BytesAliTrans","NumpyAliTrans"]

	is_classes(f"Exkaldi transition alignment matrix: {name}",ali,targetClasses)

@declare_wrapper	
def is_potential_transcription(name,transcription):
	'''
	Verify whether or not this is a reasonable transcription that is file path exkaldi Transcription object.
	'''	
	if isinstance(transcription,str):
		is_file(name,transcription)
	else:
		assert __type_name(transcription) == "Transcription",f"{name} is a file name or exkaldi Transcription object: {__type_name(transcription)}."

@declare_wrapper
def is_transcription(name,transcription):
	'''
	Verify whether or not this is an exkaldi Transcription object.
	'''	
	assert __type_name(transcription) == "Transcription",f"{name} is not an exkaldi transcription object: {__type_name(transcription)}."

@declare_wrapper
def is_potential_list_table(name,listTable):
	'''
	Verify whether or not this is a reasonable transcription that is file path or exkaldi ListTable object.
	'''		
	if isinstance(listTable,str):
		is_file(name,listTable)
	else:
		assert __type_name(listTable) == "ListTable",f"{name} is not a file name or exkaldi ListTable object: {__type_name(listTable)}."

@declare_wrapper
def is_list_table(name,listTable):
	'''
	Verify whether or not this is an exkaldi ListTable object.
	'''	
	assert __type_name(obj) == "ListTable",f"{name} should be an exkaldi ListTable object but got: {__type_name(obj)}."

@declare_wrapper
def is_potential_hmm(name,hmm):
	'''
	Verify whether or not this is a reasonable GMM-HMM model that is a file path or exkaldi HMM object.
	'''	
	if isinstance(hmm,str):
		is_file(name,hmm)
	else:
		assert __type_name(hmm) in ["BaseHMM","MonophoneHMM","TriphoneHMM"],f"{name} should be file name or exkaldi HMM object but got: {__type_name(hmm)}."

@declare_wrapper
def is_hmm(name,hmm):
	'''
	Verify whether or not this is an exkaldi HMM object.
	'''	
	targetClasses = ["BaseHMM","MonophoneHMM","TriphoneHMM"]
	
	is_classes(f"Exkaldi HMM object: {name}",hmm,targetClasses)

@declare_wrapper
def is_potential_tree(name,tree):
	'''
	Verify whether or not this is a reasonable decision tree that is a file path or exkaldi DecisionTree object.
	'''	
	if isinstance(tree,str):
		is_file(name,tree)
	else:
		assert __type_name(tree) == "DecisionTree",f"{name} should be a file name or exkaldi DecisionTree object but got: {__type_name(tree)}."

@declare_wrapper
def is_tree(name,tree):
	'''
	Verify whether or not this is an exkaldi DecisionTree object.
	'''	
	assert __type_name(tree) == "DecisionTree",f"{name} is not an exkaldi DecisionTree object: {__type_name(tree)}."

@declare_wrapper
def is_potential_lattice(name,lat):
	'''
	Verify whether or not this is a reasonable lattice that is a file path or exkaldi Lattice object.
	'''	
	if isinstance(lat,str):
		is_file(name,lat)
	else:
		assert __type_name(lat) == "Lattice",f"{name} should be file name or exkaldi Lattice object but got: {__type_name(lat)}."

@declare_wrapper
def is_lattice(name,lat):
	'''
	Verify whether or not this is an exkaldi Lattice object.
	'''	
	assert __type_name(lat) == "Lattice",f"{name} should be an exkaldi Lattice object but got: {__type_name(lat)}."

@declare_wrapper
def is_lexicon_bank(name,lex):
	'''
	Verify whether or not this is an exkaldi LexiconBank object.
	'''	
	assert __type_name(lex) == "LexiconBank",f"{name} should be exkaldi LexiconBank object but got: {__type_name(lex)}."

