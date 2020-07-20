# coding=utf-8
#
# Yu Wang (University of Yamanashi)
# Mar, 2020
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
import numpy as np
import subprocess
import struct
from io import BytesIO
from collections import namedtuple

from exkaldi.version import info as ExkaldiInfo
from exkaldi.version import UnsupportedType, WrongOperation, KaldiProcessError, WrongDataFormat
from exkaldi.utils.utils import run_shell_command, run_shell_command_parallel, type_name, list_files, make_dependent_dirs
from exkaldi.utils.utils import FileHandleManager
from exkaldi.utils import declare
from exkaldi.core.archieve import BytesFeature, BytesCMVNStatistics, BytesFmllrMatrix, BytesAlignmentTrans, ListTable, BytesArchieve, BytesMatrix, BytesVector, NumpyMatrix, NumpyVector
from exkaldi.core.load import load_index_table, load_list_table

def tuple_data(archieves, frameLevel=False):
	'''
	Tuple feature or alignment archieves in "utterance" level or "frame" level.

	Args:
		<archieves>: exkaldi feature or alignment objects.
		<framelevel>: If True, tuple data in frame level. Or in utterance level.
	Return:
		List of tupled data.
	'''
	declare.is_classes("archieves", archieves, (tuple,list))
	assert len(archieves) > 1, "<archieves> should has mutiple items."
	declare.is_bool("frameLevel", frameLevel)
	
	archieves = match_utterances(archieves)

	fields = {}
	for index, data in enumerate(archieves):
		if frameLevel is True:
			declare.belong_classes("achieve", data, (BytesMatrix, BytesVector, NumpyMatrix, NumpyVector))
		else:
			declare.belong_classes("achieve", data, (BytesMatrix, BytesVector, NumpyMatrix, NumpyVector, ListTable))
		
		if isinstance(data, (BytesMatrix, BytesVector)):
			archieves[index] = data.to_numpy()

		if data.name not in fields.keys():
			fields[data.name] = []
		fields[data.name].append(data)

	fieldNames = list(fields.keys())

	try:
		if frameLevel:
			templet = namedtuple(typename="TupledData", field_names=["uttID","frameID",]+fieldNames)
		else:
			templet = namedtuple(typename="TupledData", field_names=["uttID",] + fieldNames )
	except ValueError as e:
		print('While tuple data, use "name" of archieves as identity ID so they are expected Python valid identifiers.')
		print('You can use ".rename()" method to rename it and try this function again.')
		raise e

	def align_tuple_data_to_frame(utt, record, templet):

		if isinstance(record[0],list):
			frameSize = len(record[0][0])
		else:
			frameSize = len(record[0])

		for r in record[1:]:
			if isinstance(r, list):
				for sr in r:
					if len(sr) != frameSize:
						raise WrongOperation(f"Cannot tuple data with different frame length to frame level: {frameSize}!={len(sr)}.")
			else:
				if len(r) != frameSize:
					raise WrongOperation(f"Cannot tuple data with different frame length to frame level: {frameSize}!={len(r)}.")				
		
		result = []
		for frameIndex in range(frameSize):
			new = []
			for r in record:
				if isinstance(r, list):
					filedR = []
					for sr in r:
						filedR.append( sr[frameIndex] )
					new.append(filedR)
				else:
					new.append( r[frameIndex:frameIndex+1] )
					
			result.append(templet( utt, frameIndex, *new  ))

		return result

	if isinstance(archieves[0], ListTable):
		uttIDs = list(archieves[0].keys())
	else:
		uttIDs = archieves[0].utts

	result = []
	for utt in uttIDs:
		oneRecord = []

		for field in fieldNames:
			fieldData = []
			for ob in fields[field]:
				fieldData.append( ob.data[utt] )
			if len(fieldData) == 1:
				fieldData = fieldData[0]
			oneRecord.append( fieldData )

		if frameLevel:
			result.extend( align_tuple_data_to_frame(utt, oneRecord, templet) )
		else:
			result.append( templet(utt,*oneRecord))
	
	return result

def compute_postprob_norm(ali, posrProbDim):
	'''
	Compute alignment counts in order to normalize acoustic model posterior probability.
	For more help information, look at the Kaldi <analyze-counts> command.

	Args:
		<ali>: exkaldi NumpyAlignmentPhone or NumpyAlignmentPdf object.
		<posrProbDim>: the dimensionality of posterior probability.
	Return:
		A numpy array of the normalization.
	''' 
	declare.kaldi_existed()
	declare.is_classes("ali", ali, ["NumpyAlignmentPhone", "NumpyAlignmentPdf"])
	declare.is_positive_int("posrProbDim", posrProbDim)

	cmd = f"analyze-counts --print-args=False --verbose=0 --binary=false --counts-dim={posrProbDim} ark:- -"
	out, err, cod = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=ali.data)
	if (isinstance(cod,int) and cod != 0) or out == b"":
		print(err.decode())
		raise KaldiProcessError('Analyze counts defailed.')
	else:
		out = out.decode().strip().strip("[]").strip().split()
		counts = np.array(out, dtype=np.int32)
		countBias = np.log(counts/np.sum(counts))
		return countBias

def match_utterances(archieves):
	'''
	Pick up utterances whose ID has existed in all provided archieves:

	Args:
		<archieves>, a list of exkaldi archieve objects.
	
	Return:
		a list of new exkaldi archieve objects.
	'''
	declare.is_classes("archieves", archieves, [list,tuple])

	shareUttIDs = None
	for t in archieves:

		declare.belong_classes("archieves", t, [ListTable, BytesMatrix, BytesVector, NumpyMatrix, NumpyVector] )

		if isinstance(t, ListTable):
			uttIDs = set(t.keys())
		else:
			uttIDs = set(t.utts)

		if shareUttIDs is None:
			shareUttIDs = uttIDs
		else:
			shareUttIDs &= uttIDs

	shareUttIDs = list(shareUttIDs)
	if len(shareUttIDs) == 0:
		raise WrongOperation("Utterance are completely missed. We think it is not reasonable. Please check these archieves.")

	results = []
	for t in archieves:
		if len(t.utts) == len(shareUttIDs):
			results.append( t )
		else:
			results.append( t.subset(uttIDs=shareUttIDs) )
	
	return results

def check_mutiple_resources(*resources, outFile=None):
	'''
	This function is used to check whether or not use mutiple process and verify the resources.
	args:
		<resources>: objects.
		<outFile>: None, file name, or a list of None objects, file names.
				If None, it means standard output stream.
	
	Return:
		lists of resources.
	'''
	# check: first pass
	mutipleFlag = [ len(re) if isinstance(re, (list,tuple)) else 1 for re in resources ]
	mutipleFlag = sorted(list(set(mutipleFlag)))

	if len(mutipleFlag) == 0:
		raise WrongOperation(f"No any resource has been received.")
	elif len(mutipleFlag) > 2:
		raise WrongOperation(f"The numbers of resources do not match: {mutipleFlag} .")
	mutipleFlag = max(mutipleFlag)

	# check and modify: second pass
	resources = list(resources)
	for index,target in enumerate(resources):

		if isinstance(target, (list,tuple)):
			if len(target) == 1:
				resources[index] = [ target[0] for i in range(mutipleFlag) ]
			else:
				exType = None
				for t in target:
					if exType is None:
						exType = type_name(t)
					elif type_name(t) != exType:
						raise WrongDataFormat(f"Elements of one group should be the same data class, but got: {exType} != {type_name(t)}.")
		else:
			resources[index] = [ target for i in range(mutipleFlag) ]

	# check output file: third pass
	if mutipleFlag > 1:
		assert outFile is not None, "When apply parallel processes, out file name is necessary."
		outFiles = []
		declare.is_classes("outFile", outFile, [str, list, tuple])
		if isinstance(outFile,str):
			declare.is_valid_file_name("outFile", outFile)
			outFile = os.path.abspath(outFile)
			dirName = os.path.dirname(outFile)
			fileName = os.path.basename(outFile)
			outFiles = [ os.path.join( dirName, f"{i}_"+fileName ) for i in range(mutipleFlag) ]
		else:
			declare.equal("the number of output files", len(outFile), "the number of parallel processes", mutipleFlag)
			outFiles = []
			for f in outFile:
				declare.is_valid_file_name("outFile", f)
				outFiles.append(f)
		
		resources.append(outFiles)

	else:
		if outFile is None:
			outFile = "-"
		else:
			declare.is_valid_file_name("outFile", outFile)

		resources.append([outFile,])

	return resources

def run_kaldi_commands_parallel(resources, cmdPattern, analyzeResult=True, timeout=ExkaldiInfo.timeout, generateArchieve=None, archieveNames=None):
	'''
	Map resources to command pattern and run this command parallelly.

	Args:
		<resources>: a dict whose keys are the name of resource and values are lists of resources objects.
					For example: {"feat": [BytesFeature01, BytesFeature02,... ], "outFile":{"newFeat01.ark","newFeat02.ark",...} }.
					The "outFile" resource is necessary.
					When there is only one process to run, "outFile" can be "-" which means the standard output stream.

		<cmdPattern>: a string needed to map the resources.
					For example: "copy-feat {feat} ark:{outFile}".
	
	Return:
		a list of triples: (return code, error info, output file or buffer)
	'''
	declare.kaldi_existed()
	declare.is_classes("resources", resources, dict)
	declare.is_classes("cmdPattern", cmdPattern, str)
	assert "outFile" in resources.keys(), "<outFile> key and value is necessary in recources."

	# check the format of cmomand pattern
	nameIndexs = [ i for i,c in enumerate(cmdPattern) if c == "{" or c == "}" ]
	assert len(nameIndexs)%2 == 0, f"The numbers of braces do not match in command pattern: '{cmdPattern}'. "
	auxiliaryInfo = {}
	for i in range(0, len(nameIndexs), 2):
		name = cmdPattern[nameIndexs[i]+1:nameIndexs[i+1]]
		if name not in resources:
			raise WrongDataFormat(f"Resource is necessary but has not been provided: {name}.")
		prefix = "" if i == 0 else cmdPattern[nameIndexs[i]-1]
		if name in auxiliaryInfo.keys():
			auxiliaryInfo[name][0] += 1
			if not prefix in auxiliaryInfo[name][1]:
				auxiliaryInfo[name][1] += prefix
		else:
			auxiliaryInfo[name] = [1, prefix]

	assert "outFile" in auxiliaryInfo.keys(), "Key: <outFile> is necessary in command pattern."
	_outFileCountInfo = auxiliaryInfo.pop("outFile")
	assert _outFileCountInfo[0] == 1, f"Only allow <outFile> appear one time in command pattern but: {_outFileCountInfo[0]}."
	outFiles = resources.pop("outFile")

	for outFile in outFiles:
		if outFile != "-":
			make_dependent_dirs(outFile, pathIsFile=True)
	parallel = len(outFiles)

	if generateArchieve is not None:
		declare.is_instances("generateArchieve", generateArchieve, ["feat","cmvn","ali","fmllrMat"])
		if archieveNames is None:
			archieveNames = [ generateArchieve for i in range(parallel)]
		elif isinstance(archieveNames, str):
			archieveNames = [ archieveNames for i in range(parallel)]
		elif isinstance(archieveNames, (list,tuple)):
			declare.equal("the number of achieve names", len(archieveNames), "parallel", parallel)
		else:
			raise UnsupportedType(f"<archieveNames> should be string or list or tuple but got: {type_name(archieveNames)}.")

	# regulate resources and run
	with FileHandleManager() as fhm:

		newResources = {}
		if parallel == 1:
			# Detect whether or not there is not PIPE.
			testPlaceholder = dict( (key,value[0]) if isinstance(value[0], str) else (key,"placeholder") for key,value in resources.items() )
			testPlaceholder["outFile"] = "placeholder"
			testCmd = cmdPattern.format(**testPlaceholder)
			if "|" in testCmd:
				inputsBuffer = False
			else:
				inputsBuffer = True
			del testPlaceholder
			# regularate resources
			for key, countPrefix in auxiliaryInfo.items():
				count, prefix = countPrefix
				target = resources[key][0]

				# If target is a list-table, we can not automatically decide whether it is scp-format or ark-format.
				# So you should appoint it in the command parttern.
				if type_name(target) in ["ListTable","Transcription"]:
					if prefix != ":":
						raise WrongDataFormat(f"Miss prefix such as 'ark:' or 'scp:' in command pattern before resource: {key}.")
					target = target.sort()
					if (inputsBuffer is True) and count == 1:
						inputsBuffer = target.save()
						newResources[key] = f"-"
					else:
						targetTemp = fhm.create("w+", encoding="utf-8")
						target.save(targetTemp)
						newResources[key] = f"{targetTemp.name}"

				# If target is an index-table, we automatically recognize it as scp-file, so you do not need appoint it.
				elif type_name(target) == "ArkIndexTable":
					if prefix != " ":
						raise WrongDataFormat(f"Do not need prefix such as 'ark:' or 'scp:' in command pattern before resource: {key}.")
					target = target.sort()
					if (inputsBuffer is True) and count == 1:
						inputsBuffer = target.save()
						newResources[key] = f"scp:-"
					else:
						targetTemp = fhm.create("w+", suffix=".scp", encoding="utf-8")
						target.save(targetTemp)
						newResources[key] = f"scp:{targetTemp.name}"
				
				elif isinstance(target, (str,int,float)):
					# file or other value parameter
					newResources[key] = f"{target}"
			
				elif isinstance(target, (BytesMatrix, BytesVector)):
					if prefix != " ":
						raise WrongDataFormat(f"Do not need prefix such as 'ark:' or 'scp:' in command pattern before resource: {key}.")
					target = target.sort()
					if (inputsBuffer is True) and count == 1:
						inputsBuffer = target.data
						newResources[key] = f"ark:-"
					else:					
						targetTemp = fhm.create("wb+", suffix=".ark")
						target.save(targetTemp)
						newResources[key] = f"ark:{targetTemp.name}"		

				elif isinstance(target, (NumpyMatrix, NumpyVector)):
					if prefix != " ":
						raise WrongDataFormat(f"Do not need prefix such as 'ark:' or 'scp:' in command pattern before resource: {key}.")
					target = target.sort()
					if (inputsBuffer is True) and count == 1:
						inputsBuffer = target.to_bytes().data
						newResources[key] = f"ark:-"
					else:
						target = target.to_bytes()
						targetTemp = fhm.create("wb+", suffix=".ark")
						target.save(targetTemp)
						newResources[key] = f"ark:{targetTemp.name}"	

				elif isinstance(target, BytesArchieve):
					if (inputsBuffer is True) and count == 1:
						inputsBuffer = target.data
						newResources[key] = f"-"
					else:
						targetTemp = fhm.create("wb+")
						target.save(targetTemp)
						newResources[key] = f"{targetTemp.name}"

				else:
					raise UnsupportedType(f"<target> should be ArkIndexTable, ListTable, file name, or exkaldi achieve object but got: {type_name(target)}.")
			
			# Now, process output stream
			outFile = outFiles[0]
			newResources["outFile"] = outFile
			inputsBuffer = None if isinstance(inputsBuffer,bool) else inputsBuffer
			# Then rum command
			finalCmd = cmdPattern.format(**newResources)

			out, err, cod = run_shell_command(finalCmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=inputsBuffer)
			
			if analyzeResult:
				if cod != 0:
					print(err.decode())
					finalCmd = finalCmd.split("|")[-1].strip().split(maxsplit=1)[0]
					raise KaldiProcessError(f"Failed to run kaldi command: {finalCmd}.")
			
			if outFile == "-":
				if generateArchieve is not None:
					if generateArchieve == "feat":
						out = BytesFeature(data=out, name=archieveNames[0])
					elif generateArchieve == "ali":
						out = BytesAlignmentTrans(data=out, name=archieveNames[0])
					elif generateArchieve == "cmvn":
						out = BytesCMVNStatistics(data=out, name=archieveNames[0])
					else:
						out = BytesFmllrMatrix(data=out, name=archieveNames[0])
					return out
				else:
					return (cod,err,out)
			else:
				if generateArchieve is not None:
					return load_index_table(outFile, name=archieveNames[0], useSuffix="ark")
				else:
					return (cod,err,outFile)

		else:
			# In this case, all input IO stream must be files.
			for key, countPrefix in auxiliaryInfo.items():
				count, prefix = countPrefix
				values = resources[key]
				newvalues = []
				for target in values:

					# If target is scp resource
					if type_name(target) in ["ListTable","Transcription"]:
						if prefix != ":":
							raise WrongDataFormat(f"Miss prefix such as 'ark:' or 'scp:' in command pattern before resource: {key}.")						
						target = target.sort()
						targetTemp = fhm.create("w+", encoding="utf-8")
						target.save(targetTemp)
						newvalues.append(f"{targetTemp.name}")						

					elif type_name(target) == "ArkIndexTable":
						if prefix != " ":
							raise WrongDataFormat(f"Do not need any prefixs such as 'ark:' or 'scp:' in command pattern before resource: {key}.")						
						target = target.sort()
						targetTemp = fhm.create("w+", suffix=".scp", encoding="utf-8")
						target.save(targetTemp)
						newvalues.append(f"scp:{targetTemp.name}")
				
					elif isinstance(target, (str,float,int)):
						# file name or other value parameters
						newvalues.append(f"{target}")
				
					elif isinstance(target, (BytesMatrix, BytesVector)):
						if prefix != " ":
							raise WrongDataFormat(f"Do not need any prefixs such as 'ark:' or 'scp:' in command pattern before resource: {key}.")			
						target = target.sort()
						targetTemp = fhm.create("wb+", suffix=".ark")
						target.save(targetTemp)
						newvalues.append(f"ark:{targetTemp.name}")			

					elif isinstance(target, (NumpyMatrix, NumpyVector)):
						if prefix != " ":
							raise WrongDataFormat(f"Do not need any prefixs such as 'ark:' or 'scp:' in command pattern before resource: {key}.")
						target = target.sort().to_bytes()
						targetTemp = fhm.create("wb+", suffix=".ark")
						target.save(targetTemp)
						newvalues.append(f"ark:{targetTemp.name}")

					elif isinstance(target, BytesArchieve):
						targetTemp = fhm.create("wb+")
						target.save(targetTemp)	
						newvalues.append(f"{targetTemp.name}")

					else:
						raise UnsupportedType(f"<target> should be ArkIndexTable, ListTable, Transcription, file, int or float values or exkaldi achieve object but got: {type_name(target)}.")
				
				newResources[key] = newvalues
			
			newResources["outFile"] = outFiles
			# assign these resources to each process and generate mutiple commands
			parallelResources = []
			for i in range(parallel):
				parallelResources.append({})
				for key, items in newResources.items():
					parallelResources[-1][key] = items[i]
			cmds = [ cmdPattern.format(**re) for re in parallelResources ]
			# run
			flags = run_shell_command_parallel(cmds, timeout=timeout)

			finalResult = []
			done = True
			for index,info in enumerate(flags):
				cod, err = info
				if analyzeResult and cod != 0:
					print(f"{index}/{len(flags)} error tracking")
					print(err.decode())
					done = False	
				finalResult.append( (cod, err, outFiles[index]) )

			if analyzeResult and (not done):
				finalCmd = cmds[0].split("|")[-1].strip().split(maxsplit=1)[0]
				raise KaldiProcessError(f"Failed to run kaldi command: {finalCmd}. Look the error messages above.")
			else:
				if generateArchieve is not None:
					for i, fileName in enumerate(outFiles):
						finalResult[i] = load_index_table(fileName, name=archieveNames[i], useSuffix="ark")

			return finalResult

def utt2spk_to_spk2utt(utt2spk, outFile=None):
	'''
	Transform utt2spk file to spk2utt file.

	Args:
		<utt2spk>: file name or exkaldi ListTable object.
		<outFile>: file name or None.
	
	Return:
		file name or exakldi ListTable object.
	'''
	declare.is_potential_list_table("utt2spk", utt2spk)
	if outFile is not None:
		declare.is_valid_file_name(outFile)
	
	if isinstance(utt2spk,str):
		utt2spk = load_list_table(utt2spk)

	spk2utt = ListTable(name="spk2utt")
	for utt, spk in utt2spk.items():
		declare.is_valid_string("utterance ID", utt)
		declare.is_valid_string("speaker ID", spk)
		assert utt.count(" ") == 0, f"<utterance ID> is not a continuous string but spaces existed: {utt}."
		assert spk.count(" ") == 0, f"<spkeaker ID> is not a continuous string but spaces existed: {utt}."
		if spk not in spk2utt.keys():
			spk2utt[spk] = utt
		else:
			spk2utt[spk] += f" {utt}"

	if outFile is None:
		return spk2utt
	else:
		spk2utt.save(outFile)
		return outFile

def spk2utt_to_utt2spk(spk2utt, outFile=None):
	'''
	Transform spk2utt file to utt2spk file.

	Args:
		<spk2utt>: file name or exkaldi ListTable object.
		<outFile>: file name or None.
	'''
	declare.is_potential_list_table("spk2utt", spk2utt)
	if outFile is not None:
		declare.is_valid_file_name(outFile)
	
	if isinstance(spk2utt,str):
		spk2utt = load_list_table(spk2utt)

	utt2spk = ListTable(name="utt2spk")
	for spk, utts in spk2utt.items():
		declare.is_valid_string("utterance IDs", utts)
		declare.is_valid_string("speaker ID", spk)
		assert spk.count(" ") == 0, f"<spkeaker ID> is not a continuous string but spaces existed: {utt}."

		for utt in utts.split():
			if utt in utt2spk.keys():
				raise WrongDataFormat(f"utterance ID:{utt} has existed towards to mutiple speakers..")
			utt2spk[utt] = spk

	if outFile is None:
		return utt2spk
	else:
		utt2spk.save(outFile)
		return outFile