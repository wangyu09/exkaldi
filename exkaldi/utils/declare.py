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
from exkaldi.version import info as ExkaldiInfo
from collections import Iterable

def __type_name(obj):
	return obj.__class__.__name__

def kaldi_existed():
    '''
    Verify whether or not kaldi toolkit has existed in system environment.
    '''
    assert ExkaldiInfo.KALDI_ROOT is not None, "Kaldi toolkit was not found in system PATH."

def is_valid_string(objName, obj):
    '''
    Verify whether or not this is a reasonable string.
    
    Args:
        <objName>: a string, the name of this object.
        <obj>: python object.
        <errMessage>: None or a string.
    '''
    assert isinstance(obj, str) and len(obj.strip()) > 0, f"<{objName}> should be a valid string but got: {__type_name(obj)}."

def members_are_valid_strings(objName, obj):
    '''
    Verify whether or not these are reasonable strings.
    '''
    assert isinstance(obj,Iterable), f"<{objName}> is not iterable: {__type_name(obj)}."

    for m in obj:
        is_valid_string(objName, m)

def is_file(objName, obj):
    '''
    Verify whether or not this is a existed file.
    '''
    assert isinstance(obj, str), f"<{objName}> should be a file name but got: {__type_name(obj)}."
    assert len(obj) > 0, f"<{objName}> got a void string."
    assert not os.path.isdir(obj), f"<{objName}> should be a file name but directory: {obj}."
    assert os.path.isfile(obj), f"No such file: {obj}."

def members_are_files(objName, obj):
    '''
    Verify whether or not these are existed files.
    '''
    assert isinstance(obj,Iterable), f"<{objName}> is not iterable: {__type_name(obj)}."

    for m in obj:
        is_file(objName, m)

def is_classes(objName, obj, targetClass):
    '''
    Verify whether or not this object is an instance of these classes (do not include subclasses).

    Args:
        <targetClass>: class, class name, list or tuple of classes, list or tuple of names of classes.
    '''
    className = []
    if isinstance(targetClass,(list,tuple)):
        for c in targetClass:
            if isinstance(c, str):
                className.append(c)
            else:
                className.append( c.__name__ )
    else:
        if isinstance(targetClass, str):
            className.append(targetClass)
        else:
            className.append( targetClass.__name__ )
    
    assert __type_name(obj) in className, f"<{objName}> should be an object of {className} classes but got: {__type_name(obj)}"

def members_are_classes(objName, obj, targetClass):
    '''
    Verify whether or not these objects are instances of these classes (do not include subclasses).
    '''
    assert isinstance(obj, Iterable), f"<{objName}> is not iterable: {__type_name(obj)}."

    className = []
    if isinstance(targetClass,(list,tuple)):
        for c in targetClass:
            if isinstance(c, str):
                className.append(c)
            else:
                className.append( c.__name__ )
    else:
        if isinstance(targetClass, str):
            className.append(targetClass)
        else:
            className.append( targetClass.__name__ )
    
    for m in obj:
        assert __type_name(m) in className, f"All members of <{objName}> should be objects of {className} classes but got: {__type_name(m)}"

def belong_classes(objName, obj, targetClass):
    '''
    Verify whether or not this object is an instance of these classes and their subclasses.

    Args:
        <targetClass>: class, list or tuple of classes.
    '''   
    className = []
    if isinstance(targetClass, (list,tuple)):
        for c in targetClass:
            className.append( c.__name__ )
    else:
        className.append( targetClass.__name__ )
        targetClass = [targetClass, ]
    targetClass = tuple(targetClass)
        
    assert isinstance(obj, targetClass), f"<{objName}> should be an object of {className} clases or their subclasses but got: {__type_name(obj)}"

def members_belong_classes(objName, obj, targetClass):
    '''
    Verify whether or not these objects are instances of these classes and their subclasses.
    '''
    assert isinstance(obj,Iterable), f"<{objName}> is not a iterable object: {__type_name(obj)}."

    className = []
    if isinstance(targetClass,(list,tuple)):
        for c in targetClass:
            className.append( c.__name__ )
    else:
        className.append( targetClass.__name__ )
        targetClass = [targetClass, ]
    targetClass = tuple(targetClass)
    
    for m in obj:
        assert isinstance(m, targetClass), f"All members of <{objName}> should be objects of {className} classes or their subclasses but got: {__type_name(m)}"

def is_instances(objName, obj, targetInstance):

    if not isinstance(targetInstance,(list,tuple)):
        targetInstance = [targetInstance,]

    assert obj in targetInstance, f"<{objName}> should be in {targetInstance} but got: {obj}"

def members_are_instances(objName, obj, targetInstance):

    assert isinstance(obj,Iterable), f"<{objName}> is not a iterable object: {__type_name(obj)}."
    
    for m in obj:
        is_instances(objName, m, targetInstance)   

def not_void(objName, obj):

    if __type_name(obj) in ["list","tuple","dict"]:
        assert len(obj) > 0, f"<{objName}> has nothing provided."
    else:
        assert "is_void" in dir(obj), f"Cannot decide whether or not <{__type_name(obj)}> object is void with this function."
        assert not obj.is_void, f"<{objName}> is void. Can not operate it."

def is_valid_file_name_or_handle(objName, obj):

    assert __type_name(obj) in ["str","TextIOWrapper","_TemporaryFileWrapper"], f"<{objName}> should be a file name or an opened file handle but got: {__type_name(obj)}."

    if isinstance(obj, str):
        obj = obj.strip()
        assert len(obj) > 0, f"<{objName}> should be a valid string but got: {__type_name(obj)}."
        assert not os.path.isdir(obj), f"<{objName}> has been existed as a directory: {__type_name(obj)}."
    else:
        obj.seek(0)
        assert len(obj.read()) == 0, f"When <{objName}> is an opened file handle, this file should be void."
        obj.seek(0)

def is_valid_file_name(objName, obj):

    is_valid_string(objName, obj)
    obj = obj.strip()
    assert not os.path.isdir(obj), f"<{objName}> has been existed as a directory: {__type_name(obj)}."

def is_valid_dir_name(objName, obj):

    is_valid_string(objName, obj)
    obj = obj.strip()
    assert not os.path.isfile(obj), f"<{objName}> has been existed as a file: {__type_name(obj)}."

# value

def in_boundary(objName, obj, minV=None, maxV=None):

    assert minV is None or isinstance(minV,(int,float)), f"Boundary min value is invalid: {minV}."
    assert maxV is None or isinstance(maxV,(int,float)), f"Boundary max value is invalid: {maxV}."
    assert not ( maxV is None and minV is None ), f"At least one boundary value is necessary but got both None."

    if minV is None:
        if isinstance(maxV, int):
            assert isinstance(obj, int), f"<{objName}> should be an int value but got: {obj}."
        else:
            assert isinstance(obj, (int,float)), f"<{objName}> should be an int or float value but got: {obj}."
        assert obj <= maxV, f"<{objName}> should be an value no larger than {maxV} but got: {obj}."
    elif maxV is None:
        if isinstance(minV, int):
            assert isinstance(obj, int), f"<{objName}> should be an int value but got: {obj}."
        else:
            assert isinstance(obj, (int,float)), f"<{objName}> should be an int or float value but got: {obj}."        
        assert obj >= minV, f"<{objName}> should be an value no smaller than {minV} but got: {obj}."
    else:
        if isinstance(minV,int) and isinstance(maxV,int):
            assert isinstance(obj, int), f"<{objName}> should be an int value but got: {obj}."
        else:
            assert isinstance(obj, (int,float)), f"<{objName}> should be an int or float value but got: {obj}."
        assert minV <= obj <= maxV, f"<{objName}> should be an value within {minV}~{maxV} but got: {obj}."

def equal(objName1, obj1, objName2, obj2):
     
    assert obj1 == obj2, f"<{objName1}>, {obj1} does not match the <{objName2}>, {obj2}."

def larger(objName1, obj1, objName2, obj2):

    assert obj1 >= obj2, f"<{objName1}> should no smaller than <{objName2}> but got: {obj1} < {obj2}."

def smaller(objName1, obj1, objName2, obj2):

    assert obj1 <= obj2, f"<{objName1}> should no larger than <{objName2}> but got: {obj1} > {obj2}."

# special

def is_index_table(objName, obj):

    assert __type_name(obj) == "ArkIndexTable", f"<{objName}> should be exkaldi index table object but got: {__type_name(obj)}."

def is_matrix(objName, obj):

    assert __type_name(obj) in ["ArkIndexTable","NumpyMatrix","BytesMatrix"], f"<{objName}> should be exkaldi matrix table object but got: {__type_name(obj)}."

def is_vector(objName, obj):

    assert __type_name(obj) in ["ArkIndexTable","NumpyVector","BytesVector"], f"<{objName}> should be exkaldi vector table object but got: {__type_name(obj)}."

def is_feature(objName, obj):

    assert __type_name(obj) in ["ArkIndexTable","BytesFeature","NumpyFeature"], f"<{objName}> should be exkaldi feature or index table object but got: {__type_name(obj)}."

def is_probability(objName, obj):
    
    assert __type_name(obj) in ["ArkIndexTable","BytesProbability","NumpyProbability"], f"<{objName}> should be exkaldi probability or index table object but got: {__type_name(obj)}."

def is_cmvn(objName, obj):
    
    assert __type_name(obj) in ["ArkIndexTable","BytesCMVNStatistics","NumpyCMVNStatistics"], f"<{objName}> should be exkaldi CMVN statistics or index table object but got: {__type_name(obj)}."

def is_fmllr_matrix(objName, obj):
    
    assert __type_name(obj) in ["ArkIndexTable","BytesFmllrMatrix","NumpyFmllrMatrix"], f"<{objName}> should be exkaldi fmllr matrix or index table object but got: {__type_name(obj)}."

def is_alignment(objName, obj):
    
    assert __type_name(obj) in ["ArkIndexTable","BytesAlignmentTrans","NumpyAlignmentTrans"], f"<{objName}> should be exkaldi transition alignment or index table object but got: {__type_name(obj)}."

def is_potential_transcription(objName, obj):
    
    if isinstance(obj, str):
        is_file(objName, obj)
    else:
        assert __type_name(obj) == "Transcription", f"<{objName}> should be file name or exkaldi transcription object but got: {__type_name(obj)}."

def is_transcription(objName, obj):

    assert __type_name(obj) == "Transcription", f"<{objName}> should be exkaldi transcription object but got: {__type_name(obj)}."

def is_potential_list_table(objName, obj):
    
    if isinstance(obj, str):
        is_file(objName, obj)
    else:
        assert __type_name(obj) == "ListTable", f"<{objName}> should be file name or exkaldi transcription object but got: {__type_name(obj)}."

def is_list_table(objName, obj):

    assert __type_name(obj) == "ListTable", f"<{objName}> should be exkaldi list table object but got: {__type_name(obj)}."

def is_potential_hmm(objName, obj):

    if isinstance(obj, str):
        is_file(objName, obj)
    else:
        assert __type_name(obj) in ["BaseHMM","MonophoneHMM","TriphoneHMM"], f"<{objName}> should be file name or exkaldi HMM object but got: {__type_name(obj)}."

def is_hmm(objName, obj):

    assert __type_name(obj) in ["BaseHMM","MonophoneHMM","TriphoneHMM"], f"<{objName}> should be exkaldi HMM object but got: {__type_name(obj)}."

def is_potential_tree(objName, obj):

    if isinstance(obj, str):
        is_file(objName, obj)
    else:
        assert __type_name(obj) == "DecisionTree", f"<{objName}> should be file name or exkaldi decision tree object but got: {__type_name(obj)}."

def is_tree(objName, obj):

    assert __type_name(obj) == "DecisionTree", f"<{objName}> should be exkaldi decision tree object but got: {__type_name(obj)}."

def is_potential_lattice(objName, obj):

    if isinstance(obj, str):
        is_file(objName, obj)
    else:
        assert __type_name(obj) == "Lattice", f"<{objName}> should be file name or exkaldi Lattice object but got: {__type_name(obj)}."

def is_lattice(objName, obj):

    assert __type_name(obj) == "Lattice", f"<{objName}> should be an exkaldi Lattice object but got: {__type_name(obj)}."

def is_lexicon_bank(objName, obj):

    assert __type_name(obj) == "LexiconBank", f"<{objName}> should be exkaldi lexicon bank object but got: {__type_name(obj)}."

def is_positive(objName, obj):

    assert isinstance(obj,(int,float)) and obj > 0, f"<{objName}> should be a positive int or float value but got: {obj}."

def is_positive_int(objName, obj):

    assert isinstance(obj,int) and obj > 0, f"<{objName}> should be a potive int value but got: {obj}."

def is_positive_float(objName, obj):

    assert isinstance(obj,float) and obj > 0, f"<{objName}> should be a potive float value but got: {obj}."

def is_non_negative(objName, obj):

    assert isinstance(obj,(int,float)) and obj >= 0, f"<{objName}> should be a non-negasitive int or float value but got: {obj}."

def is_non_negative_int(objName, obj):

    assert isinstance(obj,int) and obj >= 0, f"<{objName}> should be a non-negasitive int value but got: {obj}."
 
def is_non_negative_float(objName, obj):

    assert isinstance(obj,float) and obj >= 0, f"<{objName}> should be a non-negasitive float value but got: {obj}."

def is_bool(objName, obj):

    assert isinstance(obj,bool), f"<{objName}> should be a bool value but got: {obj}."

def is_callable(objName, obj):

    assert callable(obj), f"<{objName}> is not callable."
