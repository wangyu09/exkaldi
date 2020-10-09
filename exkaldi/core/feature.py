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

"""feature processing functions"""
import math
import os
import numpy as np
from io import BytesIO
import struct

from exkaldi.version import info as ExKaldiInfo
from exkaldi.error import *
from exkaldi.utils.utils import type_name,make_dependent_dirs,list_files,check_config
from exkaldi.utils.utils import FileHandleManager
from exkaldi.utils import declare
from exkaldi.core.archive import BytesFeat,BytesCMVN,ListTable,IndexTable
from exkaldi.core.load import load_list_table,load_index_table
from exkaldi.core.common import check_multiple_resources,run_kaldi_commands_parallel

def __compute_feature(target,kaldiTool,useSuffix=None,name="feat",outFile=None):
	'''
	The base funtion to compute feature.
	'''
	declare.kaldi_existed()

	if useSuffix != None:
		declare.is_valid_string("useSuffix",useSuffix)
		useSuffix = useSuffix.strip().lower()[-3:]
		declare.is_instances("useSuffix",useSuffix,["scp","wav"])
	else:
		useSuffix = ""	

	targets,kaldiTools,useSuffixs,names,outFiles = check_multiple_resources(target,kaldiTool,useSuffix,name,outFile=outFile)
	# pretreatment
	fromSegment = False
	with FileHandleManager() as fhm:

		segments = []
		for index,kaldiTool,target,useSuffix,name in zip(range(len(outFiles)),kaldiTools,targets,useSuffixs,names):
			
			declare.is_classes("target",target,["str","list","ListTable","WavSegment"])
			declare.is_valid_string("name",name)

			if isinstance(target,(str,list)):		
		
				allFiles = list_files(target)
				target = ListTable()

				for filePath in allFiles:
					filePath = filePath.strip()
					if filePath[-4:].lower() == ".wav":
						fileName = os.path.basename(filePath)
						uttID = fileName[0:-4].replace(".","")
						target[uttID] = filePath
					
					elif filePath[-4:].lower() == '.scp':
						target += load_list_table(filePath)
					
					elif "wav" == useSuffix:
						fileName = os.path.basename(filePath)
						uttID = fileName.replace(".","")
						target[uttID] = filePath

					elif "scp" == useSuffix:
						target += load_list_table(filePath)

					else:
						raise UnsupportedType('Unknown file suffix. You can declare whether <useSuffix> is "wav" or "scp".')
				
				if len(target) == 0:
					raise WrongDataFormat("There did not include any data to compute data in target.")

				targets[index] = target
			
			elif type_name(target) == "WavSegment":

				segTemp = fhm.create("w+",suffix=".seg",encode="utf-8")
				target.save(segTemp)
				segments.append(segTemp.name)

				targets[index] = target.detach_wav()
				fromSegment = True

	if fromSegment:
		# define the command pattern
		cmdPattern = "extract-segments scp:{wavFile} {segment} ark:- | {kaldiTool} ark:- ark:{outFile}"
		# define resources
		resources = {"wavFile":targets,"segment":segments,"kaldiTool":kaldiTools,"outFile":outFiles}
	else:
		# define the command pattern
		cmdPattern = "{kaldiTool} scp:{wavFile} ark:{outFile}"
		# define resources
		resources = {"wavFile":targets,"kaldiTool":kaldiTools,"outFile":outFiles}

	# Run
	return run_kaldi_commands_parallel(resources,cmdPattern,analyzeResult=True,generateArchive="feat",archiveNames=names)

def compute_mfcc(target,rate=16000,frameWidth=25,frameShift=10,
				melBins=23,featDim=13,windowType='povey',useSuffix=None,
				config=None,name="mfcc",outFile=None):
	'''
	Compute MFCC feature.

	Share Args:
		Null
	
	Parallel Args:
		<target>: wave file(s), scp file(s), a list of file paths, exkaldi ListTable object or WavSegment object. If it is wave file,we will use it's file name as utterance ID.
						If it is file name, regular grammar is available.
		<rate>: sample rate.
		<frameWidth>: frame windows width (ms).
		<frameShift>: shift windows width (ms).
		<melbins>: the numbers of mel filter banks.
		<featDim>: the output dinmensionality of MFCC feature.
		<windowType>: sample windows type.
		<useSuffix>: If the suffix of file is not .scp and .wav,use this to specify it.
		<config>: extra optional configurations.
		<name>: the name of output feature object.
		<outFile>: output file name.

		Some usual options can be specified directly. If you want to use more,set <config> = your-configure.
		You can use exkaldi.check_config('compute_mfcc') function to get the refereance of extra configurations.
		Also you can run shell command "compute-mfcc-feats" to look their useage.

	Return:
		exkaldi feature or index table object.
	'''
	# check the basis configure parameters to build base commands
	stdParameters = check_multiple_resources(rate,frameWidth,frameShift,melBins,featDim,windowType,config)

	baseCmds = []
	for rate,frameWidth,frameShift,melBins,featDim,windowType,config,_ in zip(*stdParameters):
		# declare
		declare.is_positive_int("rate",rate)
		declare.is_positive_int("frameWidth",frameWidth)
		declare.is_positive_int("frameShift",frameShift)
		declare.is_positive_int("melBins",melBins)
		declare.is_positive_int("featDim",featDim)
		declare.greater_equal("frameWidth",frameWidth,"frameShift",frameShift)
		declare.is_instances("windowType",windowType,["hamming","hanning","povey","rectangular","blackmann"])
		# build kaldi command
		kaldiTool = 'compute-mfcc-feats --allow-downsample --allow-upsample '
		kaldiTool += f'--sample-frequency={rate} '
		kaldiTool += f'--frame-length={frameWidth} '
		kaldiTool += f'--frame-shift={frameShift} '
		kaldiTool += f'--num-mel-bins={melBins} '
		kaldiTool += f'--num-ceps={featDim} '
		kaldiTool += f'--window-type={windowType} '
		# check config
		if config is not None:
			if check_config(name='compute_mfcc',config=config):
				for key,value in config.items():
					if isinstance(value,bool):
						if value is True:
							kaldiTool += f"{key} "
					else:
						kaldiTool += f"{key}={value} "
		baseCmds.append(kaldiTool)
	# run the common function
	return __compute_feature(target,baseCmds,useSuffix,name,outFile)

def compute_fbank(target,rate=16000,frameWidth=25,frameShift=10,
					melBins=23,windowType='povey',useSuffix=None,
					config=None,name="fbank",outFile=None):
	'''
	Compute fbank feature.
	
	Share Args:
		Null 

	Parallel Args:
		<target>: wave file,scp file,exkaldi ListTable object or WavSegment object. If it is wave file,we will use it's file name as utterance ID.
		<rate>: sample rate.
		<frameWidth>: windows width (ms).
		<frameShift>: shift windows width (ms).
		<melbins>: the numbers of mel filter banks.
		<windowType>: windows type.
		<useSuffix>: If the suffix of file is not .scp or .wav,use this to specify it.
		<config>:  extra optional configurations.
		<name>: the name of output feature.
		<outFile>: output file name.
		
		Some usual options can be assigned directly. If you want use more,set <config> = your-configure.
		You can use exkaldi.check_config('compute_fbank') function to get the reference of extra configurations.
		Also you can run shell command "compute-fbank-feats" to look their usage.

	Return:
		exkaldi feature or index table object.
	'''
	# check the basis configure parameters to build base commands
	stdParameters = check_multiple_resources(rate,frameWidth,frameShift,melBins,windowType,config)

	baseCmds = []
	for rate,frameWidth,frameShift,melBins,windowType,config,_ in zip(*stdParameters):
		declare.is_positive_int("rate",rate)
		declare.is_positive_int("frameWidth",frameWidth)
		declare.is_positive_int("frameShift",frameShift)
		declare.is_positive_int("melBins",melBins)
		declare.greater_equal("frameWidth",frameWidth,"frameShift",frameShift)
		declare.is_instances("windowType",windowType,["hamming","hanning","povey","rectangular","blackmann"])

		kaldiTool = 'compute-fbank-feats --allow-downsample --allow-upsample '
		kaldiTool += f'--sample-frequency={rate} '
		kaldiTool += f'--frame-length={frameWidth} '
		kaldiTool += f'--frame-shift={frameShift} '
		kaldiTool += f'--num-mel-bins={melBins} '
		kaldiTool += f'--window-type={windowType} '

		if config is not None:
			if check_config(name='compute_fbank',config=config):
				for key,value in config.items():
					if isinstance(value,bool):
						if value is True:
							kaldiTool += f"{key} "
					else:
						kaldiTool += f"{key}={value} "
		
		baseCmds.append(kaldiTool)
	
	# run the common function
	return __compute_feature(target,baseCmds,useSuffix,name,outFile)

def compute_plp(target,rate=16000,frameWidth=25,frameShift=10,
				melBins=23,featDim=13,windowType='povey',useSuffix=None,
				config=None,name="plp",outFile=None):
	'''
	Compute PLP feature.

	Share Args:
		Null
	
	Parallel Args:
		<target>: wave file,scp file,exkaldi ListTable object or WavSegment object. If it is wave file,we will use it's file name as utterance ID.
		<rate>: sample rate.
		<frameWidth>: frame windows width (ms).
		<frameShift>: shift windows width (ms).
		<melbins>: the numbers of mel filter banks.
		<featDim>: the output dinmensionality of PLP feature.
		<windowType>: sample windows type.
		<useSuffix>: If the suffix of file is not .scp and .wav,use this to specify it.
		<config>: extra optional configurations.
		<name>: the name of output feature object.
		<outFile>: output file name.

		Some usual options can be specified directly. If you want to use more,set <config> = your-configure.
		You can use exkaldi.check_config('compute_plp') function to get the refereance of extra configurations.
		Also you can run shell command "compute-plp-feats" to look their useage.

	Return:
		exkaldi feature or index table object.
	'''
	# check the basis configure parameters to build base commands
	stdParameters = check_multiple_resources(rate,frameWidth,frameShift,melBins,featDim,windowType,config)
	baseCmds = []
	for rate,frameWidth,frameShift,melBins,featDim,windowType,config,_ in zip(*stdParameters):
		declare.is_positive_int("rate",rate)
		declare.is_positive_int("frameWidth",frameWidth)
		declare.is_positive_int("frameShift",frameShift)
		declare.is_positive_int("melBins",melBins)
		declare.is_positive_int("featDim",featDim)
		declare.greater_equal("frameWidth",frameWidth,"frameShift",frameShift)
		declare.is_instances("windowType",windowType,["hamming","hanning","povey","rectangular","blackmann"])

		kaldiTool = 'compute-plp-feats --allow-downsample --allow-upsample '
		kaldiTool += f'--sample-frequency={rate} '
		kaldiTool += f'--frame-length={frameWidth} '
		kaldiTool += f'--frame-shift={frameShift} '
		kaldiTool += f'--num-mel-bins={melBins} '
		kaldiTool += f'--num-ceps={featDim} '
		kaldiTool += f'--window-type={windowType} '

		if config is not None:
			if check_config(name='compute_plp',config=config):
				for key,value in config.items():
					if isinstance(value,bool):
						if value is True:
							kaldiTool += f"{key} "
					else:
						kaldiTool += f"{key}={value} "
		
		baseCmds.append(kaldiTool)
	
	# run the common function
	return __compute_feature(target,baseCmds,useSuffix,name,outFile)
	
def compute_spectrogram(target,rate=16000,frameWidth=25,frameShift=10,
						windowType='povey',useSuffix=None,config=None,name="spectrogram",outFile=None):
	'''
	Compute power spectrogram feature.

	Share Args:
		Null

	Parallel Args:
		<target>: wave file,scp file,exkaldi ListTable object or WavSegment object. If it is wave file,we will use it's file name as utterance ID.
		<rate>: sample rate.
		<frameWidth>: windows width (ms).
		<frameShift>: shift windows width (ms).
		<windowType>: windows type.
		<useSuffix>: If the suffix of file is not .scp and .wav,use this to specify it.
		<config>: extra optional configurations.
		<name>: the name of output feature object.
		<outFile>: output file name.

		Some usual options can be assigned directly. If you want use more,set <config> = your-configure.
		You can use .check_config('compute_spectrogram') function to get the refereance of extra configurations.
		Also you can run shell command "compute-spectrogram-feats" to look their usage.

	Return:
		exkaldi feature or index table object.
	'''
	# check the basis configure parameters to build base commands
	stdParameters = check_multiple_resources(rate,frameWidth,frameShift,windowType,config)
	baseCmds = []
	for rate,frameWidth,frameShift,windowType,config,_ in zip(*stdParameters):
		# check
		declare.is_positive_int("rate",rate)
		declare.is_positive_int("frameWidth",frameWidth)
		declare.is_positive_int("frameShift",frameShift)
		declare.greater_equal("frameWidth",frameWidth,"frameShift",frameShift)
		declare.is_instances("windowType",windowType,["hamming","hanning","povey","rectangular","blackmann"])

		kaldiTool = 'compute-spectrogram-feats --allow-downsample --allow-upsample '
		kaldiTool += f'--sample-frequency={rate} '
		kaldiTool += f'--frame-length={frameWidth} '
		kaldiTool += f'--frame-shift={frameShift} '
		kaldiTool += f'--window-type={windowType} '

		if config is not None:
			if check_config(name='compute_spectrogram',config=config):
				for key,value in config.items():
					if isinstance(value,bool):
						if value is True:
							kaldiTool += f"{key} "
					else:
						kaldiTool += f"{key}={value} "
		
		baseCmds.append(kaldiTool)
	
	# run the common function
	return __compute_feature(target,baseCmds,useSuffix,name,outFile)

def transform_feat(feat,matFile,outFile=None):
	'''
	Transform feat by a transform matrix. Typically,LDA,MLLT matrices.
	Note that is you want to transform FMLLR,use exkaldi.use_fmllr() function.  

	Share Args:
		Null

	Parallel Args:
		<feat>: exkaldi feature or index table object.
		<matFile>: file name.
		<outFile>: output file name.
	
	Return:
		exkaldi feature or index table object.
	'''
	feats,matFiles,outFiles = check_multiple_resources(feat,matFile,outFile=outFile)

	names = []
	for feat,matFile in zip(feats,matFiles):
		declare.is_feature("feat",feat)
		declare.is_file("matFile",matFile)
		names.append( f"tansform({feat.name})" )

	cmdPattern = 'transform-feats {matFile} {feat} ark:{outFile}'
	resources = {"feat":feats,"matFile":matFiles,"outFile":outFiles}

	return run_kaldi_commands_parallel(resources,cmdPattern,analyzeResult=True,generateArchive="feat",archiveNames=names)

def use_fmllr(feat,fmllrMat,utt2spk,outFile=None):
	'''
	Transfrom to fmllr feature.

	Share Args:
		Null

	Parallel Args:
		<feat>: exkaldi feature or index table object.
		<fmllrMat>: exkaldi fMLLR transform matrix or index table object.
		<utt2spk>: file name or ListTable object.
		<outFile>: output file name.
	
	Return:
		exkaldi feature or index table object.
	'''
	feats,fmllrMats,utt2spks,outFiles = check_multiple_resources(feat,fmllrMat,utt2spk,outFile=outFile)

	names = []
	for index,feat,fmllrMat,utt2spk in zip(range(len(outFiles)),feats,fmllrMats,utt2spks):
		# verify data
		declare.is_feature("feat",feat)
		declare.is_fmllr_matrix("fmllrMat",fmllrMat)
		# verify utt2spk
		declare.is_potential_list_table("utt2spk",utt2spk)
		names.append(f"fmllr({feat.name},{fmllrMat.name})")
	
	cmdPattern = 'transform-feats --utt2spk=ark:{utt2spk} {transMat} {feat} ark:{outFile}'
	resources = {"feat":feats,"transMat":fmllrMats,"utt2spk":utt2spks,"outFile":outFiles}

	return run_kaldi_commands_parallel(resources,cmdPattern,analyzeResult=True,generateArchive="feat",archiveNames=names)

def use_cmvn(feat,cmvn,utt2spk=None,std=False,outFile=None):
	'''
	Apply CMVN statistics to feature.

	Share Args:
		Null

	Parrallel Args:
		<feat>: exkaldi feature or index table object.
		<cmvn>: exkaldi CMVN statistics or index object.
		<utt2spk>: file path or ListTable object.
		<std>: If true,apply std normalization.
		<outFile>: out file name.

	Return:
		feature or index table object.
	'''
	feats,cmvns,utt2spks,stds,outFiles = check_multiple_resources(feat,cmvn,utt2spk,std,outFile=outFile)

	names = []
	for i,feat,cmvn,utt2spk,std in zip(range(len(outFiles)),feats,cmvns,utt2spks,stds):
		# verify feature and cmvn
		declare.is_feature("feat",feat)
		declare.is_cmvn("cmvn",cmvn)
		# verify utt2spk
		if utt2spk is not None:
			declare.is_potential_list_table("utt2spk",utt2spk)
		# std
		declare.is_bool("std",std)
		#stds[i] = "true" if std else "false"
		names.append( f"cmvn({feat.name},{cmvn.name})" ) 

	if utt2spks[0] is None:
		cmdPattern = 'apply-cmvn --norm-vars={std} {cmvn} {feat} ark:{outFile}'
		resources = {"feat":feats,"cmvn":cmvns,"std":stds,"outFile":outFiles}
	else:
		cmdPattern = 'apply-cmvn --norm-vars={std} --utt2spk=ark:{utt2spk} {cmvn} {feat} ark:{outFile}'
		resources = {"feat":feats,"cmvn":cmvns,"utt2spk":utt2spks,"std":stds,"outFile":outFiles}	
	
	return run_kaldi_commands_parallel(resources,cmdPattern,analyzeResult=True,generateArchive="feat",archiveNames=names)

def compute_cmvn_stats(feat,spk2utt=None,name="cmvn",outFile=None):
	'''
	Compute CMVN statistics.

	Share Args:
		Null

	Parrallel Args:
		<feat>: exkaldi feature object or index table object.
		<spk2utt>: spk2utt file or exkaldi ListTable object.
		<name>: name of output CMVN object.
		<outFile>: output file name.

	Return:
		exkaldi CMVN statistics or index table object.
	''' 
	feats,spk2utts,names,outFiles = check_multiple_resources(feat,spk2utt,name,outFile=outFile)

	for feat,spk2utt in zip(feats,spk2utts):
		# verify feature
		declare.is_feature("feat",feat)
		# verify spk2utt
		if spk2utt is not None:
			declare.is_potential_list_table("spk2utt",spk2utt)
	
	if spk2utts[0] is None:
		cmdPattern = 'compute-cmvn-stats {feat} ark:{outFile}'
		resources  = {"feat":feats,"outFile":outFiles}
	else:
		cmdPattern = 'compute-cmvn-stats --spk2utt=ark:{spk2utt} {feat} ark:{outFile}'
		resources  = {"feat":feats,"spk2utt":spk2utts,"outFile":outFiles}

	return run_kaldi_commands_parallel(resources,cmdPattern,analyzeResult=True,generateArchive="cmvn",archiveNames=names)

def use_cmvn_sliding(feat,windowSize=None,std=False):
	'''
	Allpy sliding CMVN statistics.

	Args:
		<feat>: exkaldi feature object.
		<windowSize>: windows size,If None,use windows size greater_equal than the frames of feature.
		<std>: a bool value.
	
	Return:
		exkaldi feature object.
	'''
	declare.is_classes("feat",feat, ["BytesFeat","NumpyFeat"])
	declare.is_bool("std",std)

	if windowSize is None:
		featLen = feat.lens[1]
		maxLen = max([length for utt,length in featLen])
		windowSize = math.ceil(maxLen/100)*100
	else:
		declare.is_positive_int("windowSize",windowSize)

	if std:
		std='true'
	else:
		std='false'

	cmd = f'apply-cmvn-sliding --cmn-window={windowSize} --min-cmn-window=100 --norm-vars={std} ark:- ark:-'
	out,err,cod = run_shell_command(cmd,stdin="PIPE",stderr="PIPE",stdout="PIPE",inputs=feat.data)
	if cod != 0:
		raise KaldiProcessError("Failed to compute sliding cmvn.",err.decode())
	
	newName = f"cmvn({feat.name},{windowSize})"
	return BytesFeat(out,name=newName,indexTable=None)

def add_delta(feat,order=2,outFile=None):
	'''
	Add n order delta to feature.
	
	Share Args:
		Null

	Parrallel Args:
		<feat>: exkaldi feature objects.
		<order>: the orders.
		<outFile>: output file name.

	Return:
		exkaldi feature or index table object.
	'''
	feats,orders,outFiles = check_multiple_resources(feat,order,outFile=outFile)
	names = []
	for feat,order in zip(feats,orders):
		# check feature
		declare.is_feature("feat",feat)
		# check order
		declare.is_positive_int("order",order)
		names.append(f"add_delta({feat.name},{order})")

	# prepare command pattern and resources
	cmdPattern = "add-deltas --delta-order={order} {feat} ark:{outFile}"
	resources = {"feat":feats,"order":orders,"outFile":outFiles}
	# run 
	return run_kaldi_commands_parallel(resources,cmdPattern,analyzeResult=True,generateArchive="feat",archiveNames=names)

def splice_feature(feat,left,right=None,outFile=None):
	'''
	Splice left-right N frames to generate new feature.
	The dimentions will become original-dim * (1 + left + right)

	Share Args:
		Null

	Parrallel Args:
		<feat>: feature or index table object.
		<left>: the left N-frames to splice.
		<right>: the right N-frames to splice. If None,right = left.
		<outFile>; output file name.

	Return:
		exkaldi feature object or index table object.
	'''
	feats,lefts,rights,outFiles = check_multiple_resources(feat,left,right,outFile=outFile)
	
	names = []
	for index,feat,left,right in zip(range(len(outFiles)),feats,lefts,rights):
		# check feature
		declare.is_feature("feat",feat)
		# check left
		declare.is_non_negative_int("left",left)
		# check right
		if right is None:
			assert left != 0,f"At least one of <left> or <right> is valid but got:{left},{right}."
			rights[index] = left
		else:
			declare.is_non_negative_int("right",right)
			assert left != 0,"Al least one of <left> or <right> is not 0."
			assert not (left == 0 and right == 0),f"At least one of <left> or <right> is valid but got:{left},{right}."

		names.append( f"splice({feat.name},{left},{right})" )

	# prepare command pattern and resources
	cmdPattern = "splice-feats --left-context={left} --right-context={right} {feat} ark:{outFile}"
	resources = {"feat":feats,"left":lefts,"right":rights,"outFile":outFiles}
	# run 
	return run_kaldi_commands_parallel(resources,cmdPattern,analyzeResult=True,generateArchive="feat",archiveNames=names)

def decompress_feat(feat,name="decompressedFeat"):
	'''
	Decompress a kaldi conpressed feature whose data-type is "CM"
	
	Args:
		<feat>: a bytes object.
		
	Return:
		An new exkaldi feature object.

	This function is a cover of kaldi-io-for-python tools. 
	For more information about it,please access to https://github.com/vesis84/kaldi-io-for-python/blob/master/kaldi_io/kaldi_io.py 
	'''
	declare.is_classes("feat",feat,bytes)

	def _read_compressed_mat(fd):

		# Format of header 'struct',
		global_header = np.dtype([('minvalue','float32'),('range','float32'),('num_rows','int32'),('num_cols','int32')]) # member '.format' is not written,
		per_col_header = np.dtype([('percentile_0','uint16'),('percentile_25','uint16'),('percentile_75','uint16'),('percentile_100','uint16')])

		# Read global header,
		globmin,globrange,rows,cols = np.frombuffer(fd.read(16),dtype=global_header,count=1)[0]
		cols = int(cols)
		rows = int(rows)

		# The data is structed as [Colheader,... ,Colheader,Data,Data ,.... ]
		#                         {           cols           }{     size         }
		col_headers = np.frombuffer(fd.read(cols*8),dtype=per_col_header,count=cols)
		col_headers = np.array([np.array([x for x in y]) * globrange * 1.52590218966964e-05 + globmin for y in col_headers],dtype=np.float32)
		data = np.reshape(np.frombuffer(fd.read(cols*rows),dtype='uint8',count=cols*rows),newshape=(cols,rows)) # stored as col-major,

		mat = np.zeros((cols,rows),dtype='float32')
		p0 = col_headers[:,0].reshape(-1,1)
		p25 = col_headers[:,1].reshape(-1,1)
		p75 = col_headers[:,2].reshape(-1,1)
		p100 = col_headers[:,3].reshape(-1,1)
		mask_0_64 = (data <= 64)
		mask_193_255 = (data > 192)
		mask_65_192 = (~(mask_0_64 | mask_193_255))

		mat += (p0  + (p25 - p0) / 64. * data) * mask_0_64.astype(np.float32)
		mat += (p25 + (p75 - p25) / 128. * (data - 64)) * mask_65_192.astype(np.float32)
		mat += (p75 + (p100 - p75) / 63. * (data - 192)) * mask_193_255.astype(np.float32)

		return mat.T,rows,cols        
	
	with BytesIO(feat) as sp:
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
					data += struct.pack(np.dtype('uint32').char,rows)
					data += '\04'.encode()
					data += struct.pack(np.dtype('uint32').char,cols)
					data += matrix.tobytes()
					newData.append(data)
				else:
					raise UnsupportedType("This is not a compressed binary data.")
			else:
				raise WrongDataFormat('Miss right binary symbol.')

	return BytesFeat(b''.join(newData),name=name)