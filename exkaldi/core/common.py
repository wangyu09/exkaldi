# coding=utf-8
#
# Yu Wang (University of Yamanashi)
# Mar,2020
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

import os
import numpy as np
from collections import namedtuple

from exkaldi.version import info as ExKaldiInfo
from exkaldi.error import *
from exkaldi.utils.utils import run_shell_command,run_shell_command_parallel,type_name,list_files,make_dependent_dirs
from exkaldi.utils.utils import FileHandleManager
from exkaldi.utils import declare
from exkaldi.core.archive import BytesArchive,BytesMatrix,BytesVector,BytesFeat,BytesCMVN,BytesFmllr,BytesAliTrans
from exkaldi.core.archive import NumpyMatrix,NumpyVector
from exkaldi.core.archive import ListTable
from exkaldi.core.load import load_index_table,load_list_table

def tuple_dataset(archives,frameLevel=False):
	'''
	Tuple feature or alignment archives in "utterance" level or "frame" level.

	Args:
		<archives>: exkaldi feature or alignment objects.
		<framelevel>: If True,tuple data in frame level. Or in utterance level.

	Return:
		List of tupled data.
	'''
	declare.is_classes("archives",archives,(tuple,list))
	assert len(archives) > 1,"<archives> should has multiple items."
	declare.is_bool("frameLevel",frameLevel)
	
	archives = match_utterances(archives)

	fields = {}
	for index,ark in enumerate(archives):
		if frameLevel is True:
			declare.belong_classes("achieves",ark,(BytesMatrix,BytesVector,NumpyMatrix,NumpyVector))
		else:
			declare.belong_classes("achieves",ark,(BytesMatrix,BytesVector,NumpyMatrix,NumpyVector,ListTable))
		
		if isinstance(ark,(BytesMatrix,BytesVector)):
			ark = ark.to_numpy()

		if ark.name not in fields.keys():
			fields[ark.name] = []
		fields[ark.name].append(ark)

	fieldNames = list(fields.keys())

	try:
		if frameLevel:
			templet = namedtuple(typename="TupledData",field_names=["key","frameID",]+fieldNames)
		else:
			templet = namedtuple(typename="TupledData",field_names=["key",]+fieldNames)
	except ValueError as e:
		e.args = ('While tuple data,use "name" of archives as identity ID so they are expected Python valid identifiers.'+
							'You can use ".rename()" method to rename it and try this function again.'+"\n"+
							e.args[0],)
		raise e

	def align_tuple_data_to_frame(key,record,templet):

		if isinstance(record[0],list):
			frameSize = len(record[0][0])
		else:
			frameSize = len(record[0])

		for re in record[1:]:
			if isinstance(re,list):
				for sr in re:
					if len(sr) != frameSize:
						raise WrongOperation(f"Cannot tuple data with different frame length to frame level: {frameSize}!={len(sr)}.")
			else:
				if len(re) != frameSize:
					raise WrongOperation(f"Cannot tuple data with different frame length to frame level: {frameSize}!={len(re)}.")				
		
		result = []
		for frameIndex in range(frameSize):
			new = []
			for re in record:
				if isinstance(re,list):
					filedR = []
					for sr in re:
						filedR.append( sr[frameIndex] )
					new.append(filedR)
				else:
					new.append( re[frameIndex:frameIndex+1] )
					
			result.append(templet( key,frameIndex,*new  ))

		return result

	result = []
	for key in archives[0].keys():
		oneRecord = []
		for field in fieldNames:
			fieldData = []
			for ark in fields[field]:
				fieldData.append( ark.data[key] )
			if len(fieldData) == 1:
				fieldData = fieldData[0]
			oneRecord.append( fieldData )

		if frameLevel:
			result.extend( align_tuple_data_to_frame(key,oneRecord,templet) )
		else:
			result.append( templet(key,*oneRecord))
	
	return result

def match_utterances(archives):
	'''
	Pick up utterances whose ID has existed in all provided archives.

	Args:
		<archives>: a list of exkaldi archive objects.
	
	Return:
		a list of new exkaldi archive objects.
	'''
	declare.is_classes("archives",archives,[list,tuple])

	shareKeys = None
	for ark in archives:

		declare.belong_classes("archives",ark,[ListTable,BytesMatrix,BytesVector,NumpyMatrix,NumpyVector] )
		keys = set(ark.keys())

		if shareKeys is None:
			shareKeys = keys
		else:
			shareKeys &= keys

	shareKeys = list(shareKeys)
	if len(shareKeys) == 0:
		raise WrongOperation("Utterance IDs completely missed. We don't think it is reasonable. Please check these archives.")

	results = []
	for ark in archives:
		if len(ark.keys()) == len(shareKeys):
			results.append( ark )
		else:
			oname = ark.name
			ark = ark.subset(keys=shareKeys)
			ark.rename(oname)
			results.append( ark )

	
	return results

def check_multiple_resources(*resources,outFile=None):
	'''
	This function is used to check whether or not use multiple process and verify the resources.

	args:
		<resources>: objects.
		<outFile>: None,file name,or a list of None objects,file names.
				If None,it means standard output stream.
	
	Return:
		lists of resources.
	'''
	# check the number of parallels
	multipleFlag = [ len(re) if isinstance(re,(list,tuple)) else 1 for re in resources ]
	multipleFlag = list(set(multipleFlag))

	if len(multipleFlag) == 0:
		raise WrongOperation(f"No any resource has been received.")
	elif len(multipleFlag) > 2:
		raise WrongOperation(f"The number of resources has various sizes:{multipleFlag}. We hope they have the same amount if their size are not 1.")
	multipleFlag = max(multipleFlag)

	# check and modify the amount of each resource
	resources = list(resources)
	for index,target in enumerate(resources):
		if isinstance(target,(list,tuple)):
			if len(target) == 1:
				resources[index] = [ target[0] for i in range(multipleFlag) ]
			else:
				exType = None
				for t in target:
					if exType is None:
						exType = type_name(t)
					elif type_name(t) != exType:
						raise WrongDataFormat(f"Elements of one group should be the same data class,but got: {exType} != {type_name(t)}.")
		else:
			resources[index] = [ target for i in range(multipleFlag) ]

	# check output file format
	if multipleFlag > 1:
		assert outFile is not None,"When apply parallel processes,output file name is necessary."
		outFiles = []
		declare.is_classes("outFile",outFile,[str,list,tuple])
		if isinstance(outFile,str):
			declare.is_valid_file_name("outFile",outFile)
			outFile = os.path.abspath(outFile)
			dirName = os.path.dirname(outFile)
			fileName = os.path.basename(outFile)
			namePattern = f"nj%0{len(str(multipleFlag))}d_{fileName}"
			outFiles = [ os.path.join(dirName,namePattern%i) for i in range(multipleFlag) ]
		else:
			declare.equal("the number of output files",len(outFile),"the number of parallel processes",multipleFlag)
			outFiles = []
			for f in outFile:
				declare.is_valid_file_name("outFile",f)
				outFiles.append(f)
		
		resources.append(outFiles)

	else:
		if outFile is None:
			outFile = "-"
		else:
			declare.is_valid_file_name("outFile",outFile)

		resources.append([outFile,])

	return resources

def run_kaldi_commands_parallel(resources,cmdPattern,analyzeResult=True,timeout=ExKaldiInfo.timeout,generateArchive=None,archiveNames=None):
	'''
	Map resources to command pattern and run this command parallelly.

	Args:
		<resources>: a dict whose keys are the name of resource and values are lists of resources objects.
					For example: {"feat": [BytesFeat01,BytesFeat02,... ],"outFile":{"newFeat01.ark","newFeat02.ark",...} }.
					The "outFile" resource is necessary.
					When there is only one process to run,"outFile" can be "-" which means the standard output stream.

		<cmdPattern>: a string needed to map the resources.
					For example: "copy-feat {feat} ark:{outFile}".
	
	Return:
		a list of triples: (return code,error info,output file or buffer)
	'''
	declare.kaldi_existed()
	declare.is_classes("resources",resources,dict)
	declare.is_classes("cmdPattern",cmdPattern,str)
	assert "outFile" in resources.keys(),"<outFile> key and value is necessary in recources."

	declare.members_are_classes("the values of resources",resources.values(),[list,tuple])
	if generateArchive is not None:
		analyzeResult = True #forcely analyze the result

	# check the format of cmomand pattern
	nameIndexs = [ i for i,c in enumerate(cmdPattern) if c == "{" or c == "}" ]
	assert len(nameIndexs)%2 == 0,f"The numbers of braces do not match in command pattern: '{cmdPattern}'. "
	auxiliaryInfo = {}
	for i in range(0,len(nameIndexs),2):
		name = cmdPattern[nameIndexs[i]+1:nameIndexs[i+1]]
		if name not in resources:
			raise WrongDataFormat(f"Resource is necessary but has not been provided: {name}.")
		prefix = "" if nameIndexs[i] == 0 else cmdPattern[nameIndexs[i]-1]
		if name in auxiliaryInfo.keys():
			auxiliaryInfo[name][0] += 1
			if not prefix in auxiliaryInfo[name][1]:
				auxiliaryInfo[name][1] += prefix
		else:
			auxiliaryInfo[name] = [1,prefix]

	assert "outFile" in auxiliaryInfo.keys(),"Key: <outFile> is necessary in command pattern."
	_outFileCountInfo = auxiliaryInfo.pop("outFile")
	assert _outFileCountInfo[0] == 1,f"Only allow <outFile> appear one time in command pattern but: {_outFileCountInfo[0]}."
	outFiles = resources.pop("outFile")

	for outFile in outFiles:
		if outFile != "-":
			make_dependent_dirs(outFile,pathIsFile=True)
	parallel = len(outFiles)

	if generateArchive is not None:
		declare.is_instances("generateArchive",generateArchive,["feat","cmvn","ali","fmllr"])
		if archiveNames is None:
			archiveNames = [ generateArchive for i in range(parallel)]
		elif isinstance(archiveNames,str):
			archiveNames = [ archiveNames for i in range(parallel)]
		elif isinstance(archiveNames,(list,tuple)):
			declare.equal("the number of achieve names",len(archiveNames),"parallel",parallel)
		else:
			raise UnsupportedType(f"<archiveNames> should be string or list or tuple but got: {type_name(archiveNames)}.")

	# regulate resources and run
	with FileHandleManager() as fhm:

		newResources = {}
		if parallel == 1:
			# Detect whether there is PIPE in command pattern.
			testPlaceholder = dict( (key,value[0]) if isinstance(value[0],str) else (key,"placeholder") for key,value in resources.items() )
			testPlaceholder["outFile"] = "placeholder"
			testCmd = cmdPattern.format(**testPlaceholder)
			if "|" in testCmd:
				inputsBuffer = False
			else:
				inputsBuffer = True
			del testPlaceholder
			# regularate resources
			for key,countPrefix in auxiliaryInfo.items():
				count,prefix = countPrefix
				target = resources[key][0]

				# If target is a list-table,we can not automatically decide whether it is scp-format or ark-format.
				# So you should appoint it in the command parttern.
				if type_name(target) in ["ListTable","Transcription"]:
					if prefix not in [":","="]:
						errMes = f"There might miss prefix such as 'ark:' or 'scp:' or '--option=' in command pattern before resource: {key}."
						errMes += "Check the command line please. If you still think there dose not need the prefix,"
						errMes += "save this ListTable or Transcription into file and instead it will this file name."
						errMes += "In that case,we will skip checking the prefix."
						raise WrongOperation(errMes)

					target = target.sort()
					if (inputsBuffer is True) and count == 1:
						inputsBuffer = target.save()
						newResources[key] = "-"
					else:
						targetTemp = fhm.create("w+",encoding="utf-8")
						target.save(targetTemp)
						newResources[key] = f"{targetTemp.name}"

				# If target is an index-table,we automatically recognize it as scp-file,so you do not need appoint it.
				elif type_name(target) == "IndexTable":
					if prefix != " ":
						errMes = f"Do not need prefix such as 'ark:' or 'scp:' in command pattern before: {key}."
						errMes += f"Because we will decide the prefix depending on its data type."
						raise WrongOperation(errMes)
						
					target = target.sort()
					if (inputsBuffer is True) and count == 1:
						inputsBuffer = target.save()
						newResources[key] = "scp:-"
					else:
						targetTemp = fhm.create("w+",suffix=".scp",encoding="utf-8")
						target.save(targetTemp)
						newResources[key] = f"scp:{targetTemp.name}"
				
				elif isinstance(target,(str,int,float)):
					# file or other value parameter
					newResources[key] = f"{target}"
			
				elif isinstance(target,(BytesMatrix,BytesVector)):
					if prefix != " ":
						errMes = f"Do not need prefix such as 'ark:' or 'scp:' in command pattern before: {key}."
						errMes += f"Because we will decide the prefix depending on its data type."						
						raise WrongOperation(errMes)

					target = target.sort()
					if (inputsBuffer is True) and count == 1:
						inputsBuffer = target.data
						newResources[key] = "ark:-"
					else:					
						targetTemp = fhm.create("wb+",suffix=".ark")
						target.save(targetTemp)
						newResources[key] = f"ark:{targetTemp.name}"		

				elif isinstance(target,(NumpyMatrix,NumpyVector)):
					if prefix != " ":
						errMes = f"Do not need prefix such as 'ark:' or 'scp:' in command pattern before: {key}."
						errMes += f"Because we will decide the prefix depending on its data type."		
						raise WrongOperation(errMes)

					target = target.sort()
					if (inputsBuffer is True) and count == 1:
						inputsBuffer = target.to_bytes().data
						newResources[key] = "ark:-"
					else:
						target = target.to_bytes()
						targetTemp = fhm.create("wb+",suffix=".ark")
						target.save(targetTemp)
						newResources[key] = f"ark:{targetTemp.name}"	

				elif isinstance(target,BytesArchive):
					if (inputsBuffer is True) and count == 1:
						inputsBuffer = target.data
						newResources[key] = "-"
					else:
						targetTemp = fhm.create("wb+")
						target.save(targetTemp)
						newResources[key] = f"{targetTemp.name}"

				else:
					raise UnsupportedType(f"<target> should be IndexTable,ListTable,file name,int or float value,or exkaldi achieve object but got: {type_name(target)}.")
			
			# Then,process output stream
			outFile = outFiles[0]
			newResources["outFile"] = outFile
			inputsBuffer = None if isinstance(inputsBuffer,bool) else inputsBuffer
			# Then rum command
			finalCmd = cmdPattern.format(**newResources)
			out,err,cod = run_shell_command(finalCmd,stdin="PIPE",stdout="PIPE",stderr="PIPE",inputs=inputsBuffer)
			
			if analyzeResult:
				if cod != 0:
					finalCmd = ",".join([cmd.strip().split(maxsplit=1)[0] for cmd in finalCmd.split("|")])
					raise KaldiProcessError(f"Failed to run Kaldi command: {finalCmd}.",err.decode())
			
			if outFile == "-":
				if generateArchive is not None:
					if generateArchive == "feat":
						out = BytesFeat(data=out,name=archiveNames[0])
					elif generateArchive == "ali":
						out = BytesAliTrans(data=out,name=archiveNames[0])
					elif generateArchive == "cmvn":
						out = BytesCMVN(data=out,name=archiveNames[0])
					else:
						out = BytesFmllr(data=out,name=archiveNames[0])
					return out
				else:
					return (cod,err,out)
			else:
				if generateArchive is not None:
					return load_index_table(outFile,name=archiveNames[0],useSuffix="ark")
				else:
					return (cod,err,outFile)

		else:
			# In this case,all input IO stream must be files.
			for key,countPrefix in auxiliaryInfo.items():
				count,prefix = countPrefix
				values = resources[key]
				newValues = []
				for target in values:

					# If target is scp resource
					if type_name(target) in ["ListTable","Transcription"]:
						if prefix not in [":","="]:
							errMes = f"There might miss prefix such as 'ark:' or 'scp:' or '--option=' in command pattern before resource: {key}."
							errMes += "Check the command line please. If you still think there dose not need the prefix,"
							errMes += "save this ListTable or Transcription into file and instead it will this file name."
							errMes += "In that case,we will skip checking the prefix."
							raise WrongOperation(errMes)		

						target = target.sort()
						targetTemp = fhm.create("w+",encoding="utf-8")
						target.save(targetTemp)
						newValues.append(f"{targetTemp.name}")						

					elif type_name(target) == "IndexTable":
						if prefix != " ":
							errMes = f"Do not need prefix such as 'ark:' or 'scp:' in command pattern before: {key}."
							errMes += f"Because we will decide the prefix depending on its data type."
							raise WrongOperation(errMes)		

						target = target.sort()
						targetTemp = fhm.create("w+",suffix=".scp",encoding="utf-8")
						target.save(targetTemp)
						newValues.append(f"scp:{targetTemp.name}")
				
					elif isinstance(target,(str,float,int)):
						# file name or other value parameters
						newValues.append(f"{target}")
				
					elif isinstance(target,(BytesMatrix,BytesVector)):
						if prefix != " ":
							errMes = f"Do not need prefix such as 'ark:' or 'scp:' in command pattern before: {key}."
							errMes += f"Because we will decide the prefix depending on its data type."						
							raise WrongOperation(errMes)	

						target = target.sort()
						targetTemp = fhm.create("wb+",suffix=".ark")
						target.save(targetTemp)
						newValues.append(f"ark:{targetTemp.name}")			

					elif isinstance(target,(NumpyMatrix,NumpyVector)):
						if prefix != " ":
							errMes = f"Do not need prefix such as 'ark:' or 'scp:' in command pattern before: {key}."
							errMes += f"Because we will decide the prefix depending on its data type."						
							raise WrongOperation(errMes)

						target = target.sort().to_bytes()
						targetTemp = fhm.create("wb+",suffix=".ark")
						target.save(targetTemp)
						newValues.append(f"ark:{targetTemp.name}")

					elif isinstance(target,BytesArchive):
						targetTemp = fhm.create("wb+")
						target.save(targetTemp)	
						newValues.append(f"{targetTemp.name}")

					else:
						raise UnsupportedType(f"<target> should be IndexTable,ListTable,Transcription,file,int or float values or exkaldi achieve object but got: {type_name(target)}.")
				
				newResources[key] = newValues
			
			newResources["outFile"] = outFiles
			# assign these resources to each process and generate multiple commands
			parallelResources = []
			for i in range(parallel):
				parallelResources.append({})
				for key,items in newResources.items():
					parallelResources[-1][key] = items[i]
			cmds = [ cmdPattern.format(**re) for re in parallelResources ]
			# run
			flags = run_shell_command_parallel(cmds,timeout=timeout)

			finalResult = []
			done = True
			for index,info in enumerate(flags):
				cod,err = info
				if analyzeResult and cod != 0:
					print(f"{index}/{len(flags)} error tracking")
					print(err.decode())
					done = False	
				finalResult.append( (cod,err,outFiles[index]) )

			if analyzeResult and (not done):
				finalCmd = ",".join([cmd.strip().split(maxsplit=1)[0] for cmd in cmds[0].split("|")])
				raise KaldiProcessError(f"Failed to run Kaldi command: {finalCmd}. Look the error messages above.")
			else:
				if generateArchive is not None:
					for i,fileName in enumerate(outFiles):
						finalResult[i] = load_index_table(fileName,name=archiveNames[i],useSuffix="ark")

			return finalResult

def utt2spk_to_spk2utt(utt2spk,outFile=None):
	'''
	Transform utt2spk to spk2utt.

	Args:
		<utt2spk>: file name or exkaldi ListTable object.
		<outFile>: file name or None.
	
	Return:
		file name or exakldi ListTable object.
	'''
	declare.is_potential_list_table("utt2spk",utt2spk)
	if outFile is not None:
		declare.is_valid_file_name(outFile)
	
	if isinstance(utt2spk,str):
		utt2spk = load_list_table(utt2spk)

	spk2utt = ListTable(name="spk2utt")
	for utt,spk in utt2spk.items():
		declare.is_valid_string("utterance ID",utt)
		declare.is_valid_string("speaker ID",spk)
		assert utt.count(" ") == 0,f"<utterance ID> is not a continuous string but spaces existed: {utt}."
		assert spk.count(" ") == 0,f"<speaker ID> is not a continuous string but spaces existed: {spk}."
		
		try:
			spk2utt[spk] += f" {utt}"
		except KeyError:
			spk2utt[spk] = utt

	if outFile is None:
		return spk2utt
	else:
		spk2utt.save(outFile)
		return outFile

def spk2utt_to_utt2spk(spk2utt,outFile=None):
	'''
	Transform spk2utt file to utt2spk file.

	Args:
		<spk2utt>: file name or exkaldi ListTable object.
		<outFile>: file name or None.

	Return:
		file name or exakldi ListTable object.
	'''
	declare.is_potential_list_table("spk2utt",spk2utt)
	if outFile is not None:
		declare.is_valid_file_name(outFile)
	
	if isinstance(spk2utt,str):
		spk2utt = load_list_table(spk2utt)

	utt2spk = ListTable(name="utt2spk")
	for spk,utts in spk2utt.items():
		declare.is_valid_string("utterance IDs",utts)
		declare.is_valid_string("speaker ID",spk)
		assert spk.count(" ") == 0,f"<speaker ID> is not a continuous string but spaces existed: {spk}."

		for utt in utts.split():
			try:
				utt2spk[utt]
			except KeyError:
				utt2spk[utt] = spk
			else:
				raise WrongDataFormat(f"utterance ID:{utt} has existed toward multiple speakers.")

	if outFile is None:
		return utt2spk
	else:
		utt2spk.save(outFile)
		return outFile

def merge_archives(archives):
	'''
	Merge multiple archives to one.
	Particularly,exkaldi Lattice objects also support this operation.
	Do the plus operation between all archives.

	Args:
		<archives>: a list or tuple of multiple exkaldi archive objects which are the same class.
	
	Return:
		a new archive object.
	'''
	declare.is_classes("archives",archives,(list,tuple))
	declare.not_void("archives",archives)
	
	if type_name(archives[0]) != "Lattice":
		declare.belong_classes("archives",archives[0],[BytesMatrix,BytesVector,ListTable,NumpyMatrix,NumpyVector])

	result = archives[0]
	typeName = type_name(archives[0])
	names = [archives[0].name]

	for ark in archives[1:]:
		assert type_name(ark) == typeName,f"All archives needed to be merged must be the same class but got: {typeName}!={type_name(ark)}."
		result += ark
		names.append(ark.name)
	
	names = ",".join(names)
	result.rename(f"merge({names})")
	return result

def spk_to_utt(spks,spk2utt):
	'''
	Accept a list of speaker IDs and return their corresponding utterance IDs.

	Args:
		<spks>: a string or list or tuple of speaker IDs.
		<spk2utt>: spk2utt file or ListTable object.
	
	Return:
		a list of utterance IDs.
	'''
	declare.is_classes("speaker IDs",spks,(str,tuple,list))

	if not isinstance(spks,str):
		declare.members_are_valid_strings("speaker IDs",spks)
	else:
		spks = [spks,]
		
	declare.is_potential_list_table("spk2utt",spk2utt)
	if isinstance(spk2utt,str):
		spk2utt = load_list_table(spk2utt)
	
	utts = []
	for spk in spks:
		try:
			utt = spk2utt[spk]
		except KeyError:
			raise WrongOperation(f"Miss speaker ID {spk} in spk2utt map.")
		else:
			declare.is_valid_string("The value of spk2utt",utt)
			utts.extend(utt.strip().split())
	
	return sorted(list(set(utts)))

def utt_to_spk(utts,utt2spk):
	'''
	Accept a list of utterance IDs and return their corresponding speaker IDs.

	Args:
		<utts>: a string or list or tuple of utterance IDs.
		<utt2spk>: utt2spk file or ListTable object.
	
	Return:
		a list of speaker IDs.
	'''
	declare.is_classes("utterance IDs",utts,(str,tuple,list))
	if not isinstance(utts,str):
		declare.members_are_valid_strings("utterance IDs",utts)
	else:
		utts = [utts,]	

	declare.is_potential_list_table("utt2spk",utt2spk)
	if isinstance(utt2spk,str):
		utt2spk = load_list_table(utt2spk)
	
	spks = []
	for utt in utts:
		try:
			spk = utt2spk[utt]
		except KeyError:
			raise WrongOperation(f"Miss utterance ID {utt} in utt2spk map.")
		else:
			declare.is_valid_string("The value of utt2spk",utt)
			spktemp = spk.strip().split(maxsplit=1)
			assert len(spktemp) == 1,f"speaker ID in utt2spk has unexpected space: {spk}."
			spks.append(spktemp[0])
	
	return sorted(list(set(spks)))
