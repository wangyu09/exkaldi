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

import numpy as np
import subprocess
import struct
from io import BytesIO
from collections import namedtuple

from exkaldi.version import version as ExkaldiInfo
from exkaldi.version import UnsupportedType, WrongOperation, KaldiProcessError, WrongDataFormat
from exkaldi.utils.utils import run_shell_command, type_name
from exkaldi.core.achivements import BytesFeature

def tuple_data(achivements, frameLevel=False):
	'''
	Tuple feature or alignment achivements in "utterance" level or "frame" level.

	Args:
		<achivements>: exkaldi feature or alignment objects.
		<framelevel>: If True, tuple data in frame level. Or in utterance level.
	Return:
		List of tupled data.
	'''
	assert isinstance(achivements, (tuple,list)) and len(achivements)>1, "<achivements> should has mutiple items."
	
	fields = {}
	for index,data in enumerate(achivements):
		if type_name(data) in ["BytesFeature", "BytesCMVNStatistics", "BytesPostProbability", "BytesAlignmentTrans"]:
			achivements[index] = data.to_bytes()
		elif type_name(data) in ["NumpyFeature", "NumpyCMVNStatistics", "NumpyPostProbability", 
								 "NumpyAlignment", "NumpyAlignmentTrans", "NumpyAlignmentPhone", "NumpyAlignmentPdf"]:
			pass
		else:
			raise UnsupportedType(f"Cannot tuple {type_name(data)} object.")

		if data.name not in fields.keys():
			fields[data.name] = []

		fields[data.name].append(data)

	fieldNames = list(fields.keys())

	try:
		if frameLevel:
			templet = namedtuple(typename="TupledData", field_names=["uttID","frameID",]+fieldNames)
		else:
			templet = namedtuple(typename="TupledData", field_names=["uttID",]+fieldNames)
	except ValueError as e:
		print('While tuple data, use "name" of achivements as identified symbols so they are expected Python valid identifiers.')
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

	uttIDs = achivements[0].utts
	result = []
	for utt in uttIDs:
		oneRecord = []

		missingFlag1 = False
		for field in fieldNames:
			fieldData = []
			missingFlag2 = False
			for ob in fields[field]:
				try:
					fieldData.append( ob.data[utt] )
				except KeyError:
					missingFlag2 = True
					break
				else:
					continue
			if missingFlag2:
				missingFlag1 = True
				break
			else:
				if len(fieldData) == 1:
					fieldData = fieldData[0]
				oneRecord.append( fieldData )
		if missingFlag1:
			continue
		else:
			if frameLevel:
				result.extend(align_tuple_data_to_frame(utt, oneRecord, templet))
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
	ExkaldiInfo.vertify_kaldi_existed()
	
	if type_name(ali) in ["NumpyAlignmentPhone", "NumpyAlignmentPdf"]:
		pass
	else:
		raise UnsupportedType(f'Expected exkaldi AlignmentPhone or  but got a {type_name(ali)}.')   
	

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
