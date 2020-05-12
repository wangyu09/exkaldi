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

from exkaldi.utils.utils import run_shell_command, type_name
from exkaldi.version import version as ExkaldiInfo
from exkaldi.utils.utils import UnsupportedDataType, WrongOperation, KaldiProcessError, WrongDataFormat
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
	
	result = []
	uttIDs = achivements.utts

	fields = {}
	for index,data in enumerate(achivements):
		if type_name(data) in ["BytesFeature", "BytesCMVNStatistics", "BytesPostProbability", "BytesAlignmentTrans"]:
			achivements[index] = data.to_bytes()
		elif type_name(data) in ["NumpyFeature", "NumpyCMVNStatistics", "NumpyPostProbability", 
								 "NumpyAlignment", "NumpyAlignmentTrans", "NumpyAlignmentPhone", "NumpyAlignmentPdf"]:
			pass
		else:
			raise UnsupportedDataType(f"Cannot tuple {type_name(data)} object.")

		if data.name not in fields.keys():
			fields[data.name] = []

		fields[data.name].append(data)

	fieldNames = list(fields.keys())
	if frameLevel:
		TupledData = namedtuple(typename="TupledData", field_names=["uttID","frameIndex",]+fieldNames)
	else:
		TupledData = namedtuple(typename="TupledData", field_names=["uttID",]+fieldNames)

	def align_tuple_data_to_frame(utt, record, templet):

		frameSize = len(record[0])
		for r in record[1:]:
			if len(r) != frameSize:
				raise WrongOperation(f"Cannot tuple data with different frame length to frame level: {frameSize}!={len(r)}.")
		
		result = []
		for frameIndex in range(frameSize):
			new = []
			for r in record:
				new.append( r[frameIndex] )
			result.append(templet( utt, frameIndex, *new  ))

		return result

	for utt in uttIDs:
		oneRecord = []
		missingFlag1 = False
		for field in fieldNames:
			fieldData = []
			missingFlag2 = False
			for ob in fields[field]:
				try:
					fieldData.append(ob(utt))
				except WrongOperation:
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
				result.extend(align_tuple_data_to_frame(utt, oneRecord, TupledData))
			else:
				result.append( TupledData(utt,*oneRecord))
	
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
		raise UnsupportedDataType(f'Expected exkaldi AlignmentPhone or  but got a {type_name(ali)}.')   
	

	cmd = f"analyze-counts --print-args=False --verbose=0 --binary=false --counts-dim={posrProbDim} ark:- -"
	out, err, _ = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=ali.data)
	if len(out) == 0:
		print(err.decode())
		raise KaldiProcessError('Analyze counts defailed.')
	else:
		out = out.decode().strip().strip("[]").strip().split()
		counts = np.array(out, dtype=np.int32)
		countBias = np.log(counts/np.sum(counts))
		return countBias

def decompress_feat(feat):
	'''
	Decompress a kaldi conpressed feature whose data-type is "CM"
	
	Args:
		<feat>: an exkaldi feature object.
	Return:
		An new exkaldi feature object.

	This function is a cover of kaldi-io-for-python tools. 
	For more information about it, please access to https://github.com/vesis84/kaldi-io-for-python/blob/master/kaldi_io/kaldi_io.py 
	'''
	assert isinstance(feat, BytesFeature), "Expected <feat> is a exkaldi bytes feature object."

	def _read_compressed_mat(fd):

		# Format of header 'struct',
		global_header = np.dtype([('minvalue','float32'),('range','float32'),('num_rows','int32'),('num_cols','int32')]) # member '.format' is not written,
		per_col_header = np.dtype([('percentile_0','uint16'),('percentile_25','uint16'),('percentile_75','uint16'),('percentile_100','uint16')])

		# Read global header,
		globmin, globrange, rows, cols = np.frombuffer(fd.read(16), dtype=global_header, count=1)[0]
		cols = int(cols)
		rows = int(rows)

		# The data is structed as [Colheader, ... , Colheader, Data, Data , .... ]
		#                         {           cols           }{     size         }
		col_headers = np.frombuffer(fd.read(cols*8), dtype=per_col_header, count=cols)
		col_headers = np.array([np.array([x for x in y]) * globrange * 1.52590218966964e-05 + globmin for y in col_headers], dtype=np.float32)
		data = np.reshape(np.frombuffer(fd.read(cols*rows), dtype='uint8', count=cols*rows), newshape=(cols,rows)) # stored as col-major,

		mat = np.zeros((cols,rows), dtype='float32')
		p0 = col_headers[:, 0].reshape(-1, 1)
		p25 = col_headers[:, 1].reshape(-1, 1)
		p75 = col_headers[:, 2].reshape(-1, 1)
		p100 = col_headers[:, 3].reshape(-1, 1)
		mask_0_64 = (data <= 64)
		mask_193_255 = (data > 192)
		mask_65_192 = (~(mask_0_64 | mask_193_255))

		mat += (p0  + (p25 - p0) / 64. * data) * mask_0_64.astype(np.float32)
		mat += (p25 + (p75 - p25) / 128. * (data - 64)) * mask_65_192.astype(np.float32)
		mat += (p75 + (p100 - p75) / 63. * (data - 192)) * mask_193_255.astype(np.float32)

		return mat.T,rows,cols        
	
	with BytesIO(feat.data) as sp:
		newData = []

		while True:
			data = b''
			utt = ''
			while True:
				char = sp.read(1)
				data += char
				char = char.decode()
				if (char == '') or (char == ' '):break
				utt += char
			utt = utt.strip()
			if utt == '':break
			binarySymbol = sp.read(2)
			data += binarySymbol
			binarySymbol = binarySymbol.decode()
			if binarySymbol == '\0B':
				dataType = sp.read(3).decode()
				if dataType == 'CM ':
					data += 'FM '.encode()
					matrix,rows,cols = _read_compressed_mat(sp)
					data += '\04'.encode()
					data += struct.pack(np.dtype('uint32').char, rows)
					data += '\04'.encode()
					data += struct.pack(np.dtype('uint32').char, cols)
					data += matrix.tobytes()
					newData.append(data)
				else:
					raise UnsupportedDataType("This is not a compressed binary data.")
			else:
				raise WrongDataFormat('Miss right binary symbol.')

	return BytesFeature(b''.join(newData), name=feat.name)
