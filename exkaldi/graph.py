# -*- coding: UTF-8 -*-
################# Version Information ################
# ExKaldi.graph, version 0.2 for Kaldi 5.5
# Yu Wang (University of Yamanashi)
# Jan, 06, 2020
#
# Exkaldi.graph module is designed to make n-grams language model and HCLG decoding graph.
# More information in https://github.com/wangyu09/exkaldi
######################################################

from exkaldi.core import KaldiProcessError,UnsupportedDataType,PathError,WrongOperation,WrongDataFormat
from exkaldi.core import check_config,get_kaldi_path,get_env,make_dirs_for_outFile,print_message

import os
import subprocess
import tempfile, shutil

KALDIROOT = get_kaldi_path()
ENV = get_env()

def prepare_srilm_and_fst():
	'''
	This founction will be run automatically when you import exkaldi.
	It check the path of srilm toolkit and add it to exkaldi running enviroment.
	'''

	global KALDIROOT, ENV

	if KALDIROOT is None:
		raise PathError("Kaldi toolkit not found.")
	else:
		SRILMROOT = KALDIROOT+"/tools/srilm"
		if not os.path.isdir(SRILMROOT):
			raise PathError("We will use SRILM language model tool. Please install it with KALDIROOT/tools/.install_srilm.sh .")

		FSTROOT = KALDIROOT+ "/tools/openfst"

		systemPATH = []
		for i in ENV['PATH'].split(':'):
			if i.endswith('/srilm'):
				continue
			elif i.endswith('/srilm/bin'):
				continue
			elif i.endswith('/srilm/bin/i686-m64'):
				continue
			elif i.endswith('/openfst/bin'):
				continue				
			else:
				systemPATH.append(i)

		systemPATH.append(SRILMROOT)
		systemPATH.append(SRILMROOT+'/bin')
		systemPATH.append(SRILMROOT+'/bin/i686-m64')
		systemPATH.append(FSTROOT+'/bin')

		ENV['PATH'] = ":".join(systemPATH)
	
prepare_srilm_and_fst()

class LexiconBank(object):
	'''
	Usage: lexicons = LexiconBank("lexicon.txt", ["spn, sp"], "unk", "sp", positionDependent=True)

	This class is designed to hold all lexicons which are going to be used when user want to make decoding graph.
	
	<pronLexicon> should be a file path. We support to generate lexicon bank from 5 kinds of lexicon which are "lexicon", "lexiconp(_disambig)" and "lexiconp_silprob(_disambig)".
	If it is not "lexicon" and silence words or unknown symbol did not exist, error will be raised.
	<silWords> should be a list object whose members are silence words and <unkSymbol> should be a string used to map the unknown words. 
	If these words are not already existed in <pronLexicon>, their proninciation will be same as themself.
	<optionalSilPhone> should be a string. It will be used as the pronunciation of "<eps>".
	'''

	#------------------------------------- initialization Methods ------------------------------
	# Bug, when use lexiconp_silprob(_disambig) to initialize, need to generate lexiconp_disambig too.

	def __init__(self, pronLexicon, silWords=["<sil>"], unkSymbol="unk", optionalSilPhone="<sil>", extraQuestions=[],
					positionDependent=False, shareSilPdf=False, extraDisambigPhoneNumbers=1, extraDisambigWords=[]):

		assert isinstance(pronLexicon,str), "Expected <pronLexicon> is name like string."
		assert isinstance(silWords,list) and len(silWords) > 0, "Expected at least one silence word in <silWords> but got nothing."
		assert isinstance(unkSymbol,str) and len(unkSymbol) > 0, "Unknown symbol is necessary."
		assert isinstance(optionalSilPhone,str) and len(optionalSilPhone) > 0, "Expected one optional silence phone in <optionalSilPhone>."
		assert isinstance(extraQuestions,list), "Expected <extraQuestions> is list object."
		assert isinstance(positionDependent,bool), "Expected <positionDependent> is True or False."
		assert isinstance(shareSilPdf,bool), "Expected <shareSilPdf> is True or False."
		assert isinstance(extraDisambigPhoneNumbers,int) and extraDisambigPhoneNumbers > 0, "Expected <extraDisambigPhoneNumbers> is positive int value."
		assert isinstance(extraDisambigWords,list), "Expected <extraDisambigWords> is list object."

		if not os.path.isfile(pronLexicon):
			raise PathError("No such file:{}.".format(pronLexicon))
		
		self.__parameters = {"silWords":silWords,
							 "unkSymbol":unkSymbol,
							 "optionalSilPhone":optionalSilPhone,
							 "extraQuestions":extraQuestions,
							 "positionDependent":positionDependent,
							 "shareSilPdf":shareSilPdf,
							 "extraDisambigPhoneNumbers":extraDisambigPhoneNumbers,
							 "extraDisambigWords":extraDisambigWords,
							 "ndisambig":0,   # This value will be updated later
							}

		## Validate the extra disambig words
		self.__validate_extraDisambigWords()

		self.__dictionaries = {}

		## Satrt initializing 
		self.__initialize_dictionaries(pronLexicon)

	def __validate_extraDisambigWords(self):
		'''
		This method is used to check whether extra disambiguation words provided have a right format.
		'''
		
		if len(self.__parameters["extraDisambigWords"]) > 0:

			global KALDIROOT, ENV

			extraDisambigWords = tempfile.NamedTemporaryFile("w+",encoding='utf-8')
			try:
				extraDisambigWords.write("\n".join(self.__parameters["extraDisambigWords"]))
				extraDisambigWords.seek(0)
				cmd = KALDIROOT + '/egs/wsj/s5/utils/lang/validate_disambig_sym_file.pl --allow-numeric "false" {}'.format(extraDisambigWords.name)
				p = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=ENV)
				out, err = p.communicate()
				if p.returncode == 1:
					print(out.decode())
					raise WrongDataFormat("Validate extra disambig words defeat.")
			finally:
				extraDisambigWords.close()

	def __initialize_dictionaries(self,fileName):
		'''
		This method is used to generate all lexicons step by step
		'''

		## Check file format. We support file with 5 types of formats: [lexicon], [lexiconp(_disambig)], [lexiconp_silprob(_disambig)].
		dictType, dataList = self.__check_lexicon_type(fileName)

		## Depending on the file format gained above, initialize key lexicon: [lexiconp].
		if dictType == "lexicon":
			self.__creat_lexiconp_from_lexicon(dataList)
		elif dictType == "silprob":
			raise WrongOperation('Cannot generate lexicon bank from silprob file.')
		else:
			self.__creat_lexiconp_from_lexiconp(dataList, dictType)

		## When arrived here,
		## if <dictType> is "lexicon" or "lexiconp(_disambig)", three lexicons, [lexiconp], [lexiconp_disambig] and [diasmbig], have been generated.
		## Or if <dictType> is "lexiconp_silprob(_disambig)", four lexicons, [lexiconp], [lexiconp_silprob], [lexiconp_silprob_disambig] and [disambig], have been generated.

		silWords = self.__parameters["silWords"]
		unkSymbol = self.__parameters["unkSymbol"]
		optionalSilPhone = self.__parameters["optionalSilPhone"]
		extraQuestions = self.__parameters["extraQuestions"]
		extraDisambigWords = self.__parameters["extraDisambigWords"]

		## Make lexicon: [silence_phones]
		temp = []
		if self.__parameters["positionDependent"]:
			for symbol in silWords + [unkSymbol]:
				phone = self.__dictionaries["lexiconp"][(symbol,0)][1].split("_")[0]
				temp.append(phone)
		self.__dictionaries["silence_phones"] = list( set(temp) )

		## Make lexicon: [optional_silence]
		self.__dictionaries["optional_silence"] = optionalSilPhone

		## Make lexicon: [nonsilence_phones]
		temp = []
		for word, pron in self.__dictionaries["lexiconp"].items():
			temp.extend( map(lambda x:x.split("_")[0],pron[1:]) )
		temp = sorted(list(set(temp)))
		self.__dictionaries["nonsilence_phones"] = []
		for phone in temp:
			phone = phone.split("_")[0]
			if not ( phone in self.__dictionaries["silence_phones"] or phone == optionalSilPhone ):
				self.__dictionaries["nonsilence_phones"].append(phone)

		## Make lexicons: [phone_map], [silence_phone_map], [nonsilence_phone_map]
		self.__dictionaries["phone_map"] = {}
		self.__dictionaries["silence_phone_map"] = {}
		self.__dictionaries["nonsilence_phone_map"] = {}
		if self.__parameters["positionDependent"]:
			for phone in self.__dictionaries["silence_phones"]:
				self.__dictionaries["phone_map"][phone] = ( phone, phone+"_S", phone+"_B", phone+"_E", phone+"_I" )
				self.__dictionaries["silence_phone_map"][phone] = ( phone, phone+"_S", phone+"_B", phone+"_E", phone+"_I" )
			for phone in self.__dictionaries["nonsilence_phones"]:
				self.__dictionaries["phone_map"][phone] = ( phone+"_S", phone+"_B", phone+"_E", phone+"_I")
				self.__dictionaries["nonsilence_phone_map"][phone] = ( phone+"_S", phone+"_B", phone+"_E", phone+"_I")
		else:
			for phone in self.__dictionaries["silence_phones"]:
				self.__dictionaries["phone_map"][phone] = ( phone, )
				self.__dictionaries["silence_phone_map"][phone] = ( phone, )
			for phone in self.__dictionaries["nonsilence_phones"]:
				self.__dictionaries["phone_map"][phone] = ( phone, )
				self.__dictionaries["nonsilence_phone_map"][phone] = ( phone, )

		## Make lexicon: [extraQuestions]
		if len(extraQuestions) == 0:
			self.__dictionaries["extra_questions"] = []
		else:
			for phone in extraQuestions:
				if not phone in self.__dictionaries["silence_phones"] + self.__dictionaries["nonsilence_phones"]:
					raise WrongDataFormat('Phoneme "{}" in extra questions is not existed in "phones".'.format(phone))
			self.__dictionaries["extra_questions"] = [ tuple(extraQuestions) ]
		
		if self.__parameters["positionDependent"]:
			for suffix in ["_B","_E","_I","_S"]:
				line = []
				for phone in self.__dictionaries["nonsilence_phones"]:
					line.append( phone + suffix )
				self.__dictionaries["extra_questions"].append(tuple(line))
			for suffix in ["","_B","_E","_I","_S"]:
				line = []
				for phone in self.__dictionaries["silence_phones"]:
					line.append( phone + suffix )
				self.__dictionaries["extra_questions"].append(tuple(line))	

		## Make lexicons: [silence], [nonsilence]
		self.__dictionaries["silence"] = []
		self.__dictionaries["nonsilence"] = []
		for phones in self.__dictionaries["silence_phone_map"].values():
			self.__dictionaries["silence"].extend(phones)
		for phones in self.__dictionaries["nonsilence_phone_map"].values():
			self.__dictionaries["nonsilence"].extend(phones)
		
		## Make lexicon: [word_boundary]
		if self.__parameters["positionDependent"]:
			self.__dictionaries["word_boundary"] = {}
			for phone in self.__dictionaries["silence"] + self.__dictionaries["nonsilence"]:
				if phone.endswith("_S"):
					self.__dictionaries["word_boundary"][phone] = "singleton"
				elif phone.endswith("_B"):
					self.__dictionaries["word_boundary"][phone] = "begin"
				elif phone.endswith("_I"):
					self.__dictionaries["word_boundary"][phone] = "internal"
				elif phone.endswith("_E"):
					self.__dictionaries["word_boundary"][phone] = "end"
				else:
					self.__dictionaries["word_boundary"][phone] = "nonword"

		## Make lexicons: [wdisambig], [wdisambig_phones], [wdisambig_words]
		self.__dictionaries["wdisambig"] = ["#0"]
		if len(extraDisambigWords) > 0:
			self.__dictionaries["wdisambig"].extend(extraDisambigWords)
		self.__dictionaries["wdisambig_phones"] = self.__dictionaries["wdisambig"]
		self.__dictionaries["wdisambig_words"] = self.__dictionaries["wdisambig"]

		## Make lexicon: [align_lexicon]
		self.__dictionaries["align_lexicon"] = {}
		self.__dictionaries["align_lexicon"][("<eps>",0)] = ("<eps>", optionalSilPhone,)
		for word, pron in self.__dictionaries["lexiconp"].items():
			self.__dictionaries["align_lexicon"][word] = (word[0],) + pron[1:]

		## Make lexicon: [oov]
		self.__dictionaries["oov"] = unkSymbol

		## Make lexicon: [sets]
		self.__dictionaries["sets"] = []
		for phone in self.__dictionaries["silence_phones"] + self.__dictionaries["nonsilence_phones"]:
			self.__dictionaries["sets"].append(self.__dictionaries["phone_map"][phone])

		## Make lexincon: [roots]
		self.__dictionaries["roots"] = {}
		temp1 = []
		temp2 = []
		if self.__parameters["shareSilPdf"]:
			for phone in self.__dictionaries["silence_phones"]:
				temp1.extend(self.__dictionaries["phone_map"][phone])
			for phone in self.__dictionaries["nonsilence_phones"]:
				temp2.append(self.__dictionaries["phone_map"][phone])
		else:
			for phone in self.__dictionaries["silence_phones"] + self.__dictionaries["nonsilence_phones"]:
				temp2.append(self.__dictionaries["phone_map"][phone])
		
		self.__dictionaries["roots"]["not-shared not-split"] = tuple(temp1)
		self.__dictionaries["roots"]["shared split"] = tuple(temp2)
		
		## Make lexincon: [phones]
		self.__make_phone_int_table()

		## Make lexicon: [words]
		self.__make_word_int_table()

	def __check_lexicon_type(self,lexiconFile):
		'''
		When given a lexicon file name, this method will discrimate its type.
		If it does not belong to "lexicon", "lexiconp(_disambig)", "lexiconp_silprob(_disambig)" and "silprob", raise error.
		'''

		with open(lexiconFile,"r",encoding="utf-8") as fr:
			lines =  fr.readlines()
		
		dataList = []
		## Check if it is "silprob"
		if len(lines) >=4:
			estimateSilprob = True
			def check_if_float(s):
				try:
					float(s)
				except ValueError:
					return True
				else:
					return False

			for line,prefix in zip(lines[0:4],["<s>","</s>_s","</s>_n","overall"]):
				line = line.strip().split()
				if len(line) != 2 or line[0] != prefix or check_if_float(line[1]):
					estimateSilprob = False
					break
				else:
					dataList.append( tuple(line) )
			if estimateSilprob:
				return "silprob", dataList

		## Check if it is "lexicon" or "lexiconp" or "lexiconp_silprob" 
		dictType = None
		for line in lines:
			line = line.strip().split()
			if len(line) == 0:
				continue
			if len(line) == 1:
				raise WrongDataFormat("Missing integrated word-(probability)-pronunciation information:{}.".format(line[0]))
			if dictType == None:
				try:
					float(line[1])
				except ValueError:
					dictType = "lexicon"
				else:
					try:
						for i in [2,3,4]:
							float(line[i])
					except ValueError:
						if i == 2:
							dictType = "lexiconp"
						else:
							raise WrongDataFormat('Expected "lexicon", "lexiconp(_disambig)", "lexiconp_silprob(_disambig)", "silprob" file but got a unknown format.')
					except IndexError:
						if i == 2:
							raise WrongDataFormat("Missing integrated word-(probability)-pronunciation information:{}.".format(" ".join(line)))
						else:
							raise WrongDataFormat('Expected "lexicon", "lexiconp(_disambig)", "lexiconp_silprob(_disambig)", "silprob" file but got a unknown format.')
					else:
						try:
							float(line[5])
						except IndexError:
							raise WrongDataFormat("Missing integrated word-(probability)-pronunciation information:{}.".format(" ".join(line)))					
						except ValueError:
							dictType = "lexiconp_silprob"
						else:
							raise WrongDataFormat('Expected "lexicon", "lexiconp(_disambig)", "lexiconp_silprob(_disambig)", "silprob" file but got a unknown format.')

			dataList.append((line[0],tuple(line[1:])))

		if len(dataList) == 0:
			raise WrongOperation("Nothing found in provided lexicon file.")
		
		## Check if it is a disambiguated lexicon
		if dictType != "lexicon":
			cmd = 'grep "#1" -m 1 < {}'.format(lexiconFile)
			p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
			out, err = p.communicate()
			if len(out) > 0:
				dictType += "_disambig"

		dataList = sorted(dataList, key=lambda x:x[0])

		return dictType, dataList

	def __creat_lexiconp_from_lexicon(self,dataList):
		'''
		This method accepts "lexicon" format data and depending on it, generate three lexicons: [lexiconp], [lexiconp_disambig] and [disambig]
		'''
		
		silWords = self.__parameters["silWords"]
		unkSymbol = self.__parameters["unkSymbol"]

		self.__dictionaries["lexiconp"] = {}

		## Add silence words and their pronunciation
		## We will give every record a unique ID (but all of silence and unk words use 0) in order to protect disambiguating words
		for symbol in silWords:
			self.__dictionaries["lexiconp"][(symbol,0)] = ("1.0",symbol,)		
		## Add unknown symbol and its pronunciation
		self.__dictionaries["lexiconp"][(unkSymbol,0)] = ("1.0",unkSymbol,)
		## Add others and its pronunciation
		disambigFlg = 1
		for word, pron in dataList:
			if word in silWords:
				if len(pron) > 1:
					raise WrongDataFormat('Silence word "{}" already existed in <pronLexicon>. Expected only one phone but got {}.'.format(word,len(pron)))
				self.__dictionaries["lexiconp"][(word,0)] = ("1.0",) + pron
			elif word == unkSymbol:
				if len(pron) > 1:
					raise WrongDataFormat('Unk symbol "{}" already existed in <pronLexicon>. Expected only one phone but got {}.'.format(word,len(pron)))
				self.__dictionaries["lexiconp"][(word,0)] = ("1.0",) + pron
			elif word == "<eps>":
				## If "<eps>" existed, remove it by force
				continue
			else:
				self.__dictionaries["lexiconp"][(word,disambigFlg)] = ("1.0",) + pron
				disambigFlg += 1
		## Transform "lexiconp" to a position-dependent one
		if self.__parameters["positionDependent"]:
			self.__apply_position_dependent_to_lexiconp(dictType="lexiconp")
		## Apply disambig phones to lexiconp
		self.__add_disambig_to_lexiconp(dictType="lexiconp")

		## When arrived here, "lexiconp", "lexiconp_disambig" and "disambig" have been generated

	def __creat_lexiconp_from_lexiconp(self,dataList,dictType="lexiconp"):
		'''
		If this method accepted "lexiconp(_disambig)" format data, generate three lexicons: [lexiconp], [lexiconp_disambig] and [disambig]
		If this method accepted "lexiconp_silprob(_disambig)" format data, generate four lexicons: [lexiconp], [lexiconp_silprob], [lexiconp_silprob_disambig] and [disambig]
		'''
		## <dataList> has a format: [( word, ( probability, *pronunciation ) ), ...]
		## <dictType> should be one of "lexiconp","lexiconp_disambig","lexiconp_silprob" and "lexicon_silprob_disambig" 

		silWords = self.__parameters["silWords"]
		unkSymbol = self.__parameters["unkSymbol"]

		## Check whether the data provided is position-dependent data
		testPron = dataList[0][1][-1]
		if "#" in testPron:
			testPron = dataList[0][1][-2]
		extimatePositionDependent = False
		if len(testPron) > 2 and (testPron[-2:] in ["_S","_B","_I","_E"]):
			extimatePositionDependent = True
		if extimatePositionDependent and ( not self.__parameters["positionDependent"]):
			raise WrongOperation(" Position-dependent is unavaliable but appeared in provided lexicon file.")

		## Transform data to Python dict object as well as giving it the unique ID (but all of silence words and unk word use 0)
		## Add check whether silence words and unk word are existed already. If not, raise error.
		temp = {}
		disambigID = 1
		for word, pron in dataList:
			if word in silWords:
				if ("silprob" in dictType and len(pron) > 5) or ( (not "silprob" in dictType) and len(pron) > 2):
					raise WrongDataFormat('Silence word "{}" existed. Expected only one phone but got {}.'.format(word,len(pron)))
				temp[ (word, 0) ] = pron
			elif word == unkSymbol:
				if ("silprob" in dictType and len(pron) > 5) or ( (not "silprob" in dictType) and len(pron) > 2):
					raise WrongDataFormat('Unknown word "{}" existed. Expected only one phone but got {}.'.format(word,len(pron)))
				temp[ (word, 0) ] = pron
			else:
				temp[ (word, disambigID) ] = pron
				disambigID += 1
		for symbol in silWords:
			if not (symbol,0) in temp.keys():
				raise WrongDataFormat('Sience word "{}" not appeared in lexiconp file.'.format(symbol))
		if not (unkSymbol,0) in temp.keys():
			raise WrongDataFormat('Unknown word "{}" not appeared in lexiconp file.'.format(unkSymbol))
		
		## If <dictType> is "lexiconp", generate [lexiconp], [lexiconp_disambig] and [disambig]
		if dictType == "lexiconp":

			self.__dictionaries["lexiconp"] = temp
		
			if self.__parameters["positionDependent"] and (not extimatePositionDependent):
				self.__apply_position_dependent_to_lexiconp(dictType="lexiconp")
		
			self.__add_disambig_to_lexiconp(dictType="lexiconp")
		
		## If <dictType> is "lexiconp_disambig", generate [lexiconp], [lexiconp_disambig] and [disambig]
		elif dictType == "lexiconp_disambig":

			self.__dictionaries["lexiconp_disambig"] = temp
			
			if self.__parameters["positionDependent"] and (not extimatePositionDependent):
				self.__apply_position_dependent_to_lexiconp(dictType="lexiconp_disambig")
			
			self.__remove_disambig_from_lexiconp_disambig(dictType="lexiconp_disambig")

		## If <dictType> is "lexiconp_silprob", generate [lexiconp], [lexiconp_silprob], [lexiconp_silprob_disambig] and [disambig]
		elif dictType=="lexiconp_silprob":

			self.__dictionaries["lexiconp_silprob"] = temp

			if self.__parameters["positionDependent"] and (not extimatePositionDependent):
				self.__apply_position_dependent_to_lexiconp(dictType="lexiconp_silprob")
			
			self.__add_disambig_to_lexiconp(dictType="lexiconp_silprob")

			self.__dictionaries["lexiconp"] = {}
			self.__dictionaries["lexiconp_disambig"] = {}
			for word, pron in self.__dictionaries["lexiconp_silprob"].items():
				self.__dictionaries["lexiconp"][word] = (pron[0],) + pron[4:]
				self.__dictionaries["lexiconp_disambig"][word] = (pron[0],) + self.__dictionaries["lexiconp_silprob_disambig"][word][4:]

		## If <dictType> is "lexiconp_silprob_disambig", generate [lexiconp], [lexiconp_silprob], [lexiconp_silprob_disambig] and [disambig]	
		elif dictType=="lexiconp_silprob_disambig":

			self.__dictionaries["lexiconp_silprob_disambig"] = temp

			if self.__parameters["positionDependent"] and (not extimatePositionDependent):
				self.__apply_position_dependent_to_lexiconp(dictType=="lexiconp_silprob_disambig")
			
			self.__remove_disambig_from_lexiconp_disambig(dictType="lexiconp_silprob_disambig")

			self.__dictionaries["lexiconp"] = {}
			self.__dictionaries["lexiconp_disambig"] = {}
			for word, pron in self.__dictionaries["lexiconp_silprob"].items():
				self.__dictionaries["lexiconp"][word] = (pron[0],) + pron[4:]
				self.__dictionaries["lexiconp_disambig"][word] = (pron[0],) + self.__dictionaries["lexiconp_silprob_disambig"][word][4:]

		else:
			raise WrongOperation('Expected lexiconp type is "lexiconp", "lexiconp_disambig", "lexiconp_silprob" or "lexiconp_silprob_disambig".')
	
	def __apply_position_dependent_to_lexiconp(self,dictType="lexiconp"):
		'''
		This method is used to transform position-independent lexicon to a postion-dependent one.
		Position-independent lexicon can be "lexiconp","lexiconp_disambig","lexiconp_silprob", or "lexiconp_silprob_disambig"
		'''

		if dictType=="lexiconp":
			for word, pron in self.__dictionaries[dictType].items():
				pron = list(pron)
				if len(pron) == 2:
					pron[1] += "_S"
				else:
					pron[1] += "_B"
					pron[-1] += "_E"
					for i in range(2,len(pron)-1):
						pron[i] += "_I"
				self.__dictionaries[dictType][word] = tuple(pron)

		elif dictType=="lexiconp_silprob":
			for word, pron in self.__dictionaries[dictType].items():
				pron = list(pron)
				if len(pron) == 5:
					pron[-1] += "_S"
				else:
					pron[4] += "_B"
					pron[-1] += "_E"
					for i in range(5,len(pron)-1):
						pron[i] += "_I"
				self.__dictionaries[dictType][word] = tuple(pron)

		elif dictType=="lexiconp_disambig":
			for word, pron in self.__dictionaries[dictType].items():
				pron = list(pron)
				if "#" in pron[-1]:
					disambigSymbol = [pron[-1]]
					pron = pron[:-1]
				else:
					disambigSymbol = []
				if len(pron) == 2:
					pron[1] += "_S"
				else:
					pron[1] += "_B"
					pron[-1] += "_E"
					for i in range(2,len(pron)-1):
						pron[i] += "_I"
				self.__dictionaries[dictType][word] = tuple(pron + disambigSymbol)

		elif dictType=="lexiconp_silprob_disambig":
			for word, pron in self.__dictionaries[dictType].items():
				pron = list(pron)
				if "#" in pron[-1]:
					disambigSymbol = [pron[-1]]
					pron = pron[:-1]
				else:
					disambigSymbol = []
				if len(pron) == 5:
					pron[-1] += "_S"
				else:
					pron[4] += "_B"
					pron[-1] += "_E"
					for i in range(5,len(pron)-1):
						pron[i] += "_I"
				self.__dictionaries[dictType][word] = tuple(pron + disambigSymbol)
		else:
			raise WrongOperation('Expected lexiconp type is "lexiconp", "lexiconp_disambig","lexiconp_silprob", or "lexiconp_silprob_disambig".')

	def __add_disambig_to_lexiconp(self,dictType="lexiconp"):
		'''
		This method is used to add phone-level disambiguation to [lexiconp] or [lexiconp_silprob]
		Lexicon, [disambig], will be gained and parameter, "ndisambig", will be updated selmeanwhile
		'''

		## <dictType> is one of "lexiconp" and "lexiconp_silprob" 

		global KALDIROOT, ENV

		if dictType == "lexiconp":
			lexiconpName = "lexiconp"
			disambigLexiconpName = "lexiconp_disambig"
			cmdOption = ""
		elif dictType == "lexiconp_silprob":
			lexiconpName = "lexiconp_silprob"
			disambigLexiconpName = "lexiconp_silprob_disambig"
			cmdOption = "--sil-probs "	
		else:
			raise WrongOperation('Expected lexiconp type is "lexiconp" or "lexiconp_silprob".')		

		self.__dictionaries["disambig"] = []
		lexiconp = tempfile.NamedTemporaryFile("w+",encoding='utf-8')
		lexiconpDisambig = tempfile.NamedTemporaryFile("w+",encoding='utf-8')

		try:
			disambigFlags = []
			for word, pron in self.__dictionaries[lexiconpName].items():
				pron = " ".join(pron)
				lexiconp.write("{} {}\n".format(word[0],pron))
				disambigFlags.append(word[1])
			
			lexiconp.seek(0)
			
			cmd = KALDIROOT + "/egs/wsj/s5/utils/add_lex_disambig.pl --pron-probs {}{} {}".format(cmdOption,lexiconp.name,lexiconpDisambig.name)
			p = subprocess.Popen(cmd,shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=ENV)
			out, err = p.communicate()
			if out == b"":
				print(err.decode())
				raise KaldiProcessError("Add disambig phones to lexiconp defeated.")
			else:
				self.__parameters["ndisambig"] = int(out.decode().strip()) + self.__parameters["extraDisambigPhoneNumbers"]

				for i in range( self.__parameters["ndisambig"] + 1 ):
					self.__dictionaries["disambig"].append("#%d"%i)
				self.__dictionaries["disambig"].extend( self.__parameters["extraDisambigWords"] )
				
				lexiconpDisambig.seek(0)
				lines = lexiconpDisambig.readlines()

				self.__dictionaries[disambigLexiconpName] = {}
				for line, disambigFlg in zip(lines,disambigFlags):
					line = line.strip().split()
					self.__dictionaries[disambigLexiconpName][(line[0],disambigFlg)] = tuple(line[1:])
		finally:
			lexiconp.close()
			lexiconpDisambig.close()

	def __remove_disambig_from_lexiconp_disambig(self,dictType="lexiconp_disambig"):
		'''
		This method is used to remove phone-level disambiguation and generate a new lexicon, [lexiconp] if <dictType> is "lexiconp_disambig", 
		or [lexiconp_silprob] if <dictType> is "lexiconp_silprob_disambig".
		Lexicon, [disambig], will be gained and parameter, "ndisambig", will be updated selmeanwhile.
		'''

		## <dictType> should be one of "lexiconp_disambig" and "lexiconp_silprob_disambig"

		tempDisambig = []
		if dictType == "lexiconp_disambig":
			newName = "lexiconp"
		elif dictType == "lexiconp_silprob_disambig":
			newName = "lexiconp_silprob"
		else:
			raise WrongOperation('Expected lexiconp type is "lexiconp_disambig" or "lexiconp_silprob_disambig".')			

		self.__dictionaries[newName] = {}
		for word, pron in self.__dictionaries[dictType].items():
			if "#" in pron[-1]:
				tempDisambig.append( int(pron[-1][1:]) )
				pron = pron[:-1]
			self.__dictionaries[newName][word] = pron

		tempDisambig = sorted(list(set(tempDisambig)))[-1]
		self.__parameters["ndisambig"] = tempDisambig + self.__parameters["extraDisambigPhoneNumbers"]
		for i in range( self.__parameters["ndisambig"] + 1 ):
			self.__dictionaries["disambig"].append("#%d"%i)
		self.__dictionaries["disambig"].extend( self.__parameters["extraDisambigWords"] )

	def __make_phone_int_table(self):
		'''
		This method is used to generated a initialized phone-numberID lexicon: [phones].
		'''

		self.__dictionaries["phones"] = {}

		allPhones = []
		for lexName in ["silence","nonsilence","disambig"]:
			allPhones.extend( self.__dictionaries[lexName] )

		count = 0
		if not "<eps>" in allPhones:
			self.__dictionaries["phones"]["<eps>"] = 0
			count += 1
		for phone in allPhones:
			self.__dictionaries["phones"][phone] = count
			count += 1

	def __make_word_int_table(self):
		'''
		This method is used to generated a initialized word-numberID lexicon: [words].
		'''

		self.__dictionaries["words"] = {}
		allWords = [ x for x, _ in self.__dictionaries["lexiconp"].keys()]
		allWords = sorted(list(set(allWords)))
		count = 0
		if not "<eps>" in allWords:
			self.__dictionaries["words"]["<eps>"] = 0
			count += 1
		for word in allWords:
			self.__dictionaries["words"][word] = count
			count += 1
		for word in self.__dictionaries["wdisambig"]:
			self.__dictionaries["words"][word] = count
			count += 1
		if not "<s>" in allWords:
			self.__dictionaries["words"]["<s>"] = count
			count += 1
		if not "</s>" in allWords:
			self.__dictionaries["words"]["</s>"] = count
			count += 1	

	#------------------------------------- Basic functions ------------------------------

	def get_parameter(self,name=None):
		'''
		Usage: ndisambig = lexobj.get_parameter("ndisambig")

		Return the value of parameters saved in LexiconBank object.
		If <name> is None, return all.
		'''

		if name is None:
			return self.__parameters
		
		elif name in self.__parameters.keys():
			return self.__parameters[name]
		
		else:
			raise WrongOperation("No such parameter:{}.".format(name))

	@property
	def view(self):
		'''
		Usage: ndisambig = lexobj.view

		Return the lexicon names of all generated lexicons.
		'''
		
		return list(self.__dictionaries.keys())

	def __call__(self, name, returnInt=False):
		'''
		Usage: ndisambig = lexobj("lexiconp_disambig")

		Return the lexicon. If <returnInt> is True, replace phones or words with ID number (but with a type of Python str).
		Some lexicons have not corresponding Int-ID table. So if you require them, a warning message will be printed and None will be returned.
		'''
		assert isinstance(name,str) and len(name) > 0, "<name> should be a name-like string."

		name = name.strip()

		if not name in self.__dictionaries.keys():
			raise WrongOperation('No such lexicon: "{}".'.format(name))

		if returnInt is False:
			return self.__dictionaries[name]

		else:
			if name in ["lexiconp", "lexiconp_disambig"]:
				temp = {}
				for word, pron in self.__dictionaries[name].items():
					word = (str(self.__dictionaries["words"][word[0]]), word[1])
					new = [pron[0]]
					for phone in pron[1:]:
						new.append(str(self.__dictionaries["phones"][phone]))
					temp[word] = tuple(new)
				return temp

			elif name in ["lexiconp_silprob", "lexiconp_silprob_disambig"]:
				temp = {}
				for word, pron in self.__dictionaries[name].items():
					word = (str(self.__dictionaries["words"][word[0]]), word[1])
					new = []
					for phone in pron[4:]:
						new.append(str(self.__dictionaries["phones"][phone]))
					temp[word] = pron[0:4] + tuple(new)
				return temp

			elif name in ["phones", "words", "phone_map", "silence_phone_map", "nonsilence_phone_map", "nonsilence_phones", "silence_phones","silprob"]:
				print('Warning: "{}" is unsupported to generate corresponding int table.'.format(name))
				return None

			elif name in ["align_lexicon"]:
				temp = {}
				for word, wordPron in self.__dictionaries[name].items():
					word = (str(self.__dictionaries["words"][word[0]]), word[1])
					new = [word[0],]
					for phone in wordPron[1:]:
						new.append( str(self.__dictionaries["phones"][phone]) )
					temp[word] = tuple(new)
				return temp

			elif name in ["disambig", "silence", "nonsilence", "wdisambig_phones"]:
				temp = []
				for phone in self.__dictionaries[name]:
					temp.append( str(self.__dictionaries["phones"][phone]) )
				return temp

			elif name in ["extra_questions", "sets"]:
				temp = []
				for phones in self.__dictionaries[name]:
					new = []
					for phone in phones:
						new.append( str(self.__dictionaries["phones"][phone]) )
					temp.append(tuple(new))
				return temp

			elif name in ["wdisambig","wdisambig_words"]:
				temp = []
				for word in self.__dictionaries[name]:
					temp.append( str(self.__dictionaries["words"][word]) )
				return temp

			elif name in ["word_boundary"]:
				temp = {}
				for phone, flg in self.__dictionaries[name].items():
					phone = str(self.__dictionaries["phones"][phone])
					temp[phone] = flg
				return temp

			elif name in ["oov"]:
				return str(self.__dictionaries["words"][self.__dictionaries[name]])
			
			elif name in ["optional_silence"]:
				return str(self.__dictionaries["phones"][self.__dictionaries[name]])

			elif name in ["roots"]:
				temp1 = []
				temp2 = []
				for phone in self.__dictionaries[name]["not-shared not-split"]:
					temp1.append( str(self.__dictionaries["phones"][phone]) )
				for sharedPhones in self.__dictionaries[name]["shared split"]:
					new = []
					for phone in sharedPhones:
						new.append( str(self.__dictionaries["phones"][phone]) )
					temp2.append(tuple(new))

				return {"not-shared not-split": tuple(temp1), "shared split": tuple(temp2) }
			
			else:
				raise WrongOperation('Transform lexicon "{}" to int-number format defeated.'.format(name))

	def dump_dict(self, name, outFile, dumpInt=False):
		'''
		Usage: lexobj.dump_dict(name="lexiconp_disambig", outFile="lexiconp_disambig.txt")

		Save the lexicon to file with Kaldi format.
		If <dumpInt> is True, replace phones or words with int ID.
		Some lexicons have not corresponding int table. So if you require them, a warning message will be printed and nothing will be saved.
		In addition, <outFile> can received a tempfile._TemporaryFileWrapper object if you just want to use the file temporarily.  
		'''

		if isinstance(outFile,str):
			if dumpInt is False:
				if not outFile.endswith(".txt"):
					outFile += ".txt"
			else:
				if not outFile.endswith(".int"):
					outFile += ".int"

			make_dirs_for_outFile(outFile)

		elif not isinstance(outFile,tempfile._TemporaryFileWrapper):
			raise WrongOperation('<outFile> shoule be a name-like string.')		

		def write_file(fileName, message):
			if isinstance(fileName,tempfile._TemporaryFileWrapper):
				fileName.write(message)
				fileName.seek(0)
			else:
				with open(fileName, "w", encoding='utf-8') as fw:
					fw.write(message)
		
		## Different lexicon has different data format, So judge them before save

		## Type1: dict, { str: tuple }
		if name in ["lexiconp", "lexiconp_disambig", "lexiconp_silprob", "lexiconp_silprob_disambig", 
					"phone_map", "silence_phone_map", "nonsilence_phone_map", "align_lexicon"]:
			contents = []
			temp = self.__call__(name,dumpInt)
			if not temp is None:
				for key, value in temp.items():
					value = " ".join(value)
					if name in ["lexiconp", "lexiconp_disambig", "lexiconp_silprob",
								"lexiconp_silprob_disambig", "align_lexicon"]:
						key = key[0]
					contents.append("{} {}".format(key, value))
				write_file(outFile, "\n".join(contents))

		## Type2: tuple, ()
		elif name in ["nonsilence_phones", "silence_phones", "disambig", "silence", "nonsilence", 
					  "wdisambig", "wdisambig_phones", "wdisambig_words"]:
			contents = self.__call__(name,dumpInt)
			if not contents is None:
				write_file(outFile, "\n".join(contents))

		## Type3: tuple, (tuple,)
		elif name in ["sets","extra_questions"]:
			contents = []
			for value in self.__call__(name,dumpInt):
				contents.append(" ".join(value))
			write_file(outFile, "\n".join(contents))
		
		## Type4: dict, { str:int or str }
		elif name in ["phones", "words", "word_boundary", "silprob"]:
			contents = []
			temp = self.__call__(name,dumpInt)
			if not temp is None:
				for key, value in temp.items():
					contents.append("{} {}".format(key, value))
				write_file(outFile, "\n".join(contents))

		## Type5: str						
		elif name in ["oov", "optional_silence"]:
			contents = self.__call__(name,dumpInt)
			write_file(outFile, contents)

		## Type6: special format for roots
		elif name == "roots":
			contents = []
			temp = self.__call__(name,dumpInt)
			if len(temp["not-shared not-split"]) > 0: 
				contents.append("not-shared not-split {}".format(" ".join(temp["not-shared not-split"])))
			for phones in temp["shared split"]:
				phones = " ".join(phones)
				contents.append("shared split {}".format(phones))
			write_file(outFile, "\n".join(contents))
	
		else:
			raise WrongOperation("Unsupported lexicon: {} to dump.".format(name))

	def dump_all_dicts(self,outDir="./",requireInt=False):
		'''
		Usage: lexobj.dump_all_dicts("lang",True)

		Save all lexicons (and their corresponding int table ) to folder with their default lexicon name.
		'''

		assert isinstance(outDir,str) and len(outDir) > 0, "<outDir> should be a name-like string."

		if not os.path.isdir(outDir):
			os.makedirs(outDir)

		outDir = os.path.abspath(outDir)

		for name in self.__dictionaries.keys():

			outFile = outDir + "/" + name
			self.dump_dict(name, outFile+".txt", False)

			if requireInt:
				if not name in ["phones", "words", "phone_map", "silence_phone_map", "nonsilence_phone_map", 
								"nonsilence_phones", "silence_phones", "silprob"]:

					self.dump_dict(name, outFile+".int", True)

	#------------------------------------- Advance functions ------------------------------

	def reset_phones(self,target):
		'''
		Usage: lexobj.reset_phones("new_phones.txt")

		Reset phone-int table with user's own lexicon. Expected the kind of phones is more than or same as default [phones] lexicon.
		<target> should be a file or Python dict object. 
		'''

		temp = {}
		allPhones = []

		if isinstance(target,str):
			if os.path.isfile(target):
				with open(target,"r",encoding='utf-8') as fr:
					lines = fr.readlines()
				for line in lines:
					line = line.strip()
					if len(line) == 0:
						continue
					phone, number = line.split()[0:2]
					try:
						temp[phone] = int(number)
					except ValueError:
						raise WrongDataFormat("Incorrect phone-ID information:{} {}".format(phone,number))
					allPhones.append(phone)
				allPhones = list( set(allPhones) )
			else:
				raise PathError("No such file: {}.".format(target))

		elif isinstance(target,dict):
			allPhones = list(target.keys())
			temp = target

		else:
			raise WrongDataFormat("Expected phone-number file or dict object.")
		
		dependentFlg = False
		for phone in allPhones:
			if len(phone) > 2 and phone[-2:] in ["_S","_B","_E","_I"]:
				dependentFlg = True
				if self.__parameters["positionDependent"] is False:
					raise WrongOperation("Position dependent phones not requested, but appear in the provided phone table.")
				break
		
		if dependentFlg is False:
			if self.__parameters["positionDependent"] is True:
				raise WrongOperation("Position dependent phones requested, but not appear in the provided <phoneNumTable>.")
		
		for phone in self.__dictionaries["silence"] + self.__dictionaries["nonsilence"]:
			if not phone in allPhones:
				raise WrongOperation("Phone appears in the lexicon but not in the provided <phoneNumTable>:{}.".format(phone))

		items = sorted(temp.items(),key=lambda x:x[1])

		if items[-1][1] != len(items)-1:
			raise WrongOperation("We expected compact ID sequences whose maximum ID is one smaller than numbers of phones, but got {} and {}.".format(items[-1][1],len(items)-1))

		temp = dict((phone,number) for phone,number in items)
		count = items[-1][1] + 1
		if not "<eps>" in allPhones:
			temp["<eps>"] = count
			count += 1
		for disambigPhone in self.__dictionaries["disambig"]:
			if not disambigPhone in allPhones:
				temp[disambigPhone] = count
				count += 1
	
		self.__dictionaries["phones"] = temp
			
	def reset_words(self,target):
		'''
		Usage: lexobj.reset_words("new_words.txt")

		Reset word-int table with user's own lexicon. Expected the kind of words is more than or same as default [words] lexicon.
		<target> should be a file or Python dict object. 
		'''

		temp = {}
		allWords = []

		if isinstance(target,str):
			if os.path.isfile(target):
				with open(target,"r",encoding='utf-8') as fr:
					lines = fr.readlines()
				for line in lines:
					line = line.strip()
					if len(line) == 0:
						continue
					word, number = line.split()[0:2]
					try:
						temp[word] = int(number)
					except ValueError:
						raise WrongDataFormat("Incorrect word-ID information:{} {}".format(word,number))					
					allWords.append(word)
				allWords = list( set(allWords) )

			else:
				raise PathError("No such file: {}.".format(target))

		elif isinstance(target,dict):
			allWords = list(target.keys())
			temp = target

		else:
			raise WrongDataFormat("Expected word-number file or dict object.")

		items = sorted(temp.items(),key=lambda x:x[1])
		if items[-1][1] != len(items)-1:
			raise WrongOperation("We expected compact ID sequences whose maximum ID is one smaller than numbers of words, but got {} and {}.".format(items[-1][1],len(items)-1))
		count = items[-1][1] + 1

		if not "<eps>" in allWords:
			temp["<eps>"] = count
			count += 1	
			
		for word in self.__dictionaries["wdisambig"]:
			if not word in allWords:
				temp[word] = count
				count += 1

		if not "<s>" in allWords:
			temp["<s>"] = count
			count += 1

		if not "</s>" in allWords:
			temp["</s>"] = count
			count += 1
		
		self.__dictionaries["words"] = temp

	def update_prob(self,targetFile):
		'''
		Usage: lexobj.update_prob("lexiconp.txt")

		Update relative probability of all of lexicons including [lexiconp], [lexiconp_silprob], [lexiconp_disambig], [lexiconp_silprob_disambig], [silprob].
		And <targetFile> can be any one of them but must be a file. 
		'''
		
		if isinstance(targetFile,str):
			targetFile = targetFile.strip()

			if not os.path.isfile(targetFile):
				raise PathError("No such file: {}.".format(targetFile))

			dictType, dataList = self.__check_lexicon_type(targetFile)

		else:
			raise WrongOperation("Only support updating from target file.")

		## If it is "lexiconp", update [lexiconp(_disambig)]. If [lexiconp_silprob(disambig)] are also existed, update them too.
		if dictType == "lexiconp":

			temp = {}
			for word, pron in dataList:
				temp[ (word, * pron[1:]) ] = pron[0]

			newLex = {}
			for word, pronLex in self.__dictionaries["lexiconp"].items():
				# "word": ( word, disambigID ); "pronLex": ( "1,0", *pronunciation )
				index = (word[0], *pronLex[1:])
				if index in temp.keys():
					newP = temp[ index ]
					newLex[word] = ( newP, ) + pronLex[1:]
				else:
					raise WrongOperation('Missing probability information of "{}"'.format(" ".join(index)))
			self.__dictionaries["lexiconp"] = newLex

			for name in ["lexiconp_disambig","lexiconp_silprob","lexiconp_silprob_disambig"]:
				if name in self.view:
					new = {}
					for word, pron in self.__dictionaries[name].items():
						newP = self.__dictionaries["lexiconp"][word][0]
						new[word] = ( newP, ) + pron[1:]
					self.__dictionaries[name] = new
		
		## If it is "lexiconp_disambig", update [lexiconp(_disambig)]. If [lexiconp_silprob(disambig)] are also existed, update them too.
		elif dictType == "lexiconp_disambig":

			temp = {}
			for word, pron in dataList:
				temp[ (word, * pron[1:]) ] = pron[0]

			# If [lexiconp_disambig] existed
			if "lexiconp_disambig" in self.view:

				newLexDis = {}
				newLex = {}
				for word, pronLexDis in self.__dictionaries["lexiconp_disambig"].items():
					# "word": ( word, disambigID ); "pron": ( "1,0", *pronunciationWithDisambig )
					index = (word[0], *pronLexDis[1:])
					pronLex = self.__dictionaries["lexiconp"][word]
					if index in temp.keys():
						newP = temp[ index ]
						newLexDis[word] = ( newP, ) + pronLexDis[1:]
						newLex[word] =  ( newP, ) + pronLex[1:]
					else:
						raise WrongOperation('Missing probability information of "{}"'.format(" ".join(index)))
				self.__dictionaries["lexiconp_disambig"] = newLexDis
				self.__dictionaries["lexiconp"] = newLex

				for name in ["lexiconp_silprob","lexiconp_silprob_disambig"]:
					if name in self.view:
						new = {}
						for word, pron in self.__dictionaries[name].items():
							newP = self.__dictionaries["lexiconp"][word][0]
							new[word] = ( newP, ) + pron[1:]
						self.__dictionaries[name] = new

			else:

				newLexSilDis = {}
				newLexSil = {}
				newLexDis = {}
				newLex = {}

				for word, pronLexSilDis in self.__dictionaries["lexiconp_silprob_disambig"].items():
					# "word": ( word, disambigID ); "pron": ( "1,0","p1","p2","p3",*pronunciationWithDisambig )
					index = (word[0], *pronLexSilDis[4:])
					pronLexSil = self.__dictionaries["lexiconp_silprob"][word]
					if index in temp.keys():
						newP = temp[ index ]
						newLexSilDis[word] = ( newP, ) + pronLexSilDis[1:]
						newLexSil[word] = ( newP, ) + pronLexSil[1:]
						newLexDis[word] = ( newP, ) + pronLexSilDis[4:]
						newLex[word] = ( newP, ) + pronLexSil[4:]
					else:
						raise WrongOperation('Missing probability information of "{}"'.format(" ".join(index)))

				self.__dictionaries["lexiconp_silprob_disambig"] = newLexSilDis
				self.__dictionaries["lexiconp_silprob"] = newLexSil
				self.__dictionaries["lexiconp_disambig"] = newLexDis
				self.__dictionaries["lexiconp"] = newLex

		## If it is "lexiconp_silprob", update [lexiconp_silprob(_disambig)] and [lexiconp]. If [lexiconp_disambig] are also existed, update it too.
		elif dictType == "lexiconp_silprob":

			temp = {}
			for word, pron in dataList:
				temp[ (word, * pron[4:]) ] = pron[0:4]

			# If [lexiconp_silprob] existed
			if "lexiconp_silprob" in self.view:

				newLex = {}
				newLexSil = {}
				newLexSilDis = {}

				for word, pronLexSil in self.__dictionaries["lexiconp_silprob"].items():
					# "word": ( word, disambigID ); "pronLexSil": ( "1,0","p1","p2","p3",*pronunciationWithDisambig )
					index = (word[0], *pronLexSil[4:])
					pronLexSilDis = self.__dictionaries["lexiconp_silprob_disambig"][word]
					if index in temp.keys():
						newP = temp[ index ]
						newLex[word] = (newP[0],) + pronLexSil[4:]
						newLexSil[word] = newP + pronLexSil[4:]
						newLexSilDis[word] = newP + pronLexSilDis[4:]				
					else:
						raise WrongOperation('Missing probability information of "{}"'.format(" ".join(index)))

				self.__dictionaries["lexiconp"] = newLex
				self.__dictionaries["lexiconp_silprob"] = newLexSil
				self.__dictionaries["lexiconp_silprob_disambig"] = newLexSilDis

				if "lexiconp_disambig" in self.view:
					new = {}
					for word, pron in self.__dictionaries["lexiconp_disambig"].items():
						newP = self.__dictionaries["lexiconp"][word][0]
						new[word] = ( newP, ) + pron[1:]
					self.__dictionaries["lexiconp_disambig"] = new

			else:

				newLex = {}
				newLexSil = {}
				newLexDis = {}
				newLexSilDis = {}

				for word, pronLex in self.__dictionaries["lexiconp"].items():
					index = (word[0], *pronLex[1:])
					pronLexDis = self.__dictionaries["lexiconp_disambig"][word]
					if index in temp.keys():
						newP = temp[ index ]
						newLex[word] = (newP[0],) + pronLex[1:] 
						newLexDis[word] = (newP[0],) + pronLexDis[1:] 
						newLexSil[word] = newP + pronLex[1:]
						newLexSilDis[word] = newP + pronLexDis[1:]
					else:
						raise WrongOperation('Missing probability information of "{}"'.format(" ".join(index)))	

				self.__dictionaries["lexiconp"] = newLex
				self.__dictionaries["lexiconp_disambig"] = newLexDis
				self.__dictionaries["lexiconp_silprob"] = newLexSil
				self.__dictionaries["lexiconp_silprob_disambig"] = newLexSilDis

		## If it is "lexiconp_silprob_disambig", update [lexiconp_silprob(_disambig)] and [lexiconp]. If [lexiconp_disambig] are also existed, update it too.
		elif dictType == "lexiconp_silprob_disambig":

			temp = {}
			for word, pron in dataList:
				temp[ (word, * pron[4:]) ] = pron[0:4]

			# if it is existed
			if "lexiconp_silprob_disambig" in self.view:

				newLex = {}
				newLexSil = {}
				newLexSilDis = {}				

				for word, pronLexSilDis in self.__dictionaries["lexiconp_silprob_disambig"].items():
					index = (word[0], *pronLexSilDis[4:])
					pronLexSil = self.__dictionaries["lexiconp_silprob"][word]
					if index in temp.keys():
						newP = temp[ index ]
						newLex[word] = ( newP[0],) + pronLexSil[4:]
						newLexSil[word] = newP + pronLexSil[4:]
						newLexSilDis[word] = newP + pronLexSilDis[4:]
					else:
						raise WrongOperation('Missing probability information of "{}"'.format(" ".join(index)))

				if "lexiconp_disambig" in self.view:
					new = {}
					for word, pron in self.__dictionaries["lexiconp_disambig"].items():
						newP = self.__dictionaries["lexiconp"][word][0]
						new[word] = ( newP, ) + pron[1:]
					self.__dictionaries["lexiconp_disambig"] = new		

			else:

				newLex = {}
				newLexDis = {}
				newLexSil = {}
				newLexSilDis = {}

				for word, pronLexDis in self.__dictionaries["lexiconp_disambig"].items():
					index = (word[0], *pronLexDis[1:])
					pronLex = self.__dictionaries["lexiconp"][word]
					if index in temp.keys():
						newP = temp[ index ]
						newLex[word] = (newP[0],) + pronLex[1:]
						newLexDis[word] = (newP[0],) + pronLexDis[1:]
						newLexSil[word] = newP + pronLex[1:]
						newLexSilDis[word] = newP + pronLexDis[1:]
					else:
						raise WrongOperation('Missing probability information of "{}"'.format(" ".join(index)))	

				self.__dictionaries["lexiconp"] = newLex
				self.__dictionaries["lexiconp_disambig"] = newLexDis
				self.__dictionaries["lexiconp_silprob"] = newLexSil
				self.__dictionaries["lexiconp_silprob_disambig"] = newLexSilDis

		## If it is "silprob", update [silprob].
		elif dictType == "silprob":
			
			temp = {}
			for symbol,prob in dataList:
				temp[symbol] = prob
			
			self.__dictionaries["silprob"] = temp
		
		else:
			raise UnsupportedDataType("<targetFile> is an unknown lexicon format.")

def make_L(dictionary, outFile, useSilprob=False, silProb=0.5, useDisambig=False):
	'''
	Usage: make_L( dictionary, "lang/L.fst" )

	Generate L.fst(or L_disambig.fst) file. Return the abs-path of generated file.
	<dictionary> should be a exkaldi.graph.Dictionary object.
	If <makeDisambig>, generated "L_disambig.fst".
	'''

	assert isinstance(dictionary,LexiconBank), "Expected <dictionary> is exkaldi LexiconBank object."
	assert isinstance(outFile,str), "Expected <outFile> is a name-like string."
	assert isinstance(silProb,float) and silProb >= 0 and silProb <= 1, "Expected <silProb> is a probility-like float value."
	assert isinstance(useSilprob,bool), "Expected <useSilprob> is True or False."
	assert isinstance(useDisambig,bool), "Expected <useDisambig> is True or False."

	global KALDIROOT, ENV

	if useSilprob:
		for name in ["lexiconp_silprob","silprob"]:
			if not name in dictionary.view:
				raise WrongOperation('When making silprob, expected "{}" is existed in dictionary.'.format(name)) 

	if not outFile.endswith(".fst"):
		outFile += ".fst"
	make_dirs_for_outFile(outFile)

	silPhone = dictionary("optional_silence")
	ndisambig = dictionary.get_parameter("ndisambig")
	lexicon = tempfile.NamedTemporaryFile("w+",encoding='utf-8',suffix=".txt")
	silprob = tempfile.NamedTemporaryFile("w+",encoding='utf-8',suffix=".txt")
	phones = tempfile.NamedTemporaryFile("w+",encoding='utf-8',suffix=".txt")
	words = tempfile.NamedTemporaryFile("w+",encoding='utf-8',suffix=".txt")
	wdisambig_phones = tempfile.NamedTemporaryFile("w+",encoding='utf-8',suffix=".int")
	wdisambig_words = tempfile.NamedTemporaryFile("w+",encoding='utf-8',suffix=".int")

	try:
		dictionary.dump_dict("phones", phones)
		dictionary.dump_dict("words", words)

		if useDisambig:
			dictionary.dump_dict("wdisambig_phones", wdisambig_phones, True)
			dictionary.dump_dict("wdisambig_words", wdisambig_words, True)
			if useSilprob:
				dictionary.dump_dict("silprob", silprob)
				dictionary.dump_dict("lexiconp_silprob_disambig", lexicon)
				cmd = KALDIROOT + "/egs/wsj/s5/utils/lang/make_lexicon_fst_silprob.py "
				cmd += "--sil-phone={} --sil-disambig=#{} {} {} | ".format(silPhone,ndisambig,lexicon.name,silprob.name)
			else:
				dictionary.dump_dict("lexiconp_disambig", lexicon)
				cmd = KALDIROOT + "/egs/wsj/s5/utils/lang/make_lexicon_fst.py "
				cmd += "--sil-prob={} --sil-phone={} --sil-disambig=#{} {} | ".format(silProb,silPhone,ndisambig,lexicon.name)

			cmd += "fstcompile --isymbols={} --osymbols={} --keep_isymbols=false --keep_osymbols=false | ".format(phones.name, words.name)
			cmd += "fstaddselfloops {} {} | ".format(wdisambig_phones.name, wdisambig_words.name)
			cmd += "fstarcsort --sort_type=olabel > {}".format(outFile)
		else:
			if useSilprob:
				dictionary.dump_dict("silprob", silprob)
				dictionary.dump_dict("lexiconp_silprob", lexicon)
				cmd = KALDIROOT + "/egs/wsj/s5/utils/lang/make_lexicon_fst_silprob.py "
				cmd += "--sil-phone={} {} {} | ".format(silPhone,lexicon.name,silprob.name)
			else:
				dictionary.dump_dict("lexiconp", lexicon)
				cmd = KALDIROOT + "/egs/wsj/s5/utils/lang/make_lexicon_fst.py "
				cmd += "--sil-prob={} --sil-phone={} {} | ".format(silProb,silPhone,lexicon.name)

			cmd += "fstcompile --isymbols={} --osymbols={} --keep_isymbols=false --keep_osymbols=false | ".format(phones.name, words.name)
			cmd += "fstarcsort --sort_type=olabel > {}".format(outFile)								

		p = subprocess.Popen(cmd, shell=True, stderr=subprocess.PIPE, env=ENV)
		_, err = p.communicate()

		if (not os.path.isfile(outFile)) or (os.path.getsize(outFile) <= 64):
			print(err.decode())
			if os.path.isfile(outFile):
				os.remove(outFile)
			if useDisambig:
				raise KaldiProcessError("Generate L_disambig.fst defeat.")
			else:
				raise KaldiProcessError("Generate L.fst defeat.")
		else:
			return os.path.abspath(outFile)

	finally:
		lexicon.close()
		phones.close()
		words.close()
		silprob.close()
		wdisambig_phones.close()
		wdisambig_words.close()

def train_ngrams(dictionary, n, textFile, outFile, discount="kndiscount", config=None):
	'''
	Usage:  obj = ngrams(3,"text.txt","lm.3g.gz","word.txt")

	Generate ARPA n-grams language model. Return abspath of generated LM.
	<n> is the orders of n-grams model. <textFile> is the text file. <outFile> is expected .gz file name of generated LM.
	Notion that we will use srilm language model toolkit.
	Some usual options can be assigned directly. If you want use more, set <config> = your-configure, but if you do this, these usual configures we provided will be ignored.
	You can use .check_config("ngrams") function to get configure information that you can set.
	Also you can run shell command "ngram-count -help" to look their meaning.    
	'''
	assert isinstance(n,int) and n > 0 and n < 10, "Expected <n> is a positive int value and it must be smaller than 10."
	assert isinstance(textFile,str), "Expected <textFile> is name-like string."
	assert isinstance(outFile,str), "Expected <outFile> is name-like string."
	assert isinstance(dictionary,LexiconBank), "Expected <dictionary> is exkaldi LexiconBank object."

	if not os.path.isfile(textFile):
		raise PathError("No such file:{}".format(textFile))
	else:
		cmd1 = 'shuf {} -n 100 | sed "s/ /\\n<space>\\n/g" | sort | uniq -c | sort -n | tail -n 1'.format(textFile)
		if int(subprocess.check_output(cmd1,shell=True).decode().strip().split()[0]) < 50:
			raise WrongDataFormat("Text file sames that it were not splited by spaces.")
	
	wordlist = tempfile.NamedTemporaryFile("w+",encoding='utf-8',suffix=".txt")
	unkSymbol = dictionary("oov")

	try:
		lexiconp = dictionary("lexiconp")
		words = [ x[0] for x in lexiconp.keys() ]
		wordlist.write( "\n".join(words) )
		wordlist.seek(0)

		cmd2 = "ngram-count -text {} -order {}".format(textFile, n)

		if config == None:  
			config = {}
			config["-limit-vocab -vocab"] = "/misc/Work18/wangyu/kaldi/egs/csj/demo1/data/local/lm/wordlist" #wordlist.name
			config["-unk -map-unk"] = unkSymbol
			assert discount in ["wbdiscount","kndiscount"], "Expected <discount> is wbdiscount or kndiscount."
			config["-{}".format(discount)] = True
			config["-interpolate"] = True
		else:
			raise WrongOperation("<config> of train_ngrams function is unavaliable now.")
		#if check_config(name='ngrams',config=config):
		if True:
			for key in config.keys():
				if config[key] is True:
					cmd2 += " {}".format(key)
				elif not (config[key] is False):
					if key == "-unk -map-unk":
						cmd2 += ' {} "{}"'.format(key,config[key])
					else:
						cmd2 += ' {} {}'.format(key,config[key])

		if not outFile.endswith(".gz"):
			outFile += ".gz"
		make_dirs_for_outFile(outFile)

		cmd2 += " -lm {}".format(outFile)
		
		p = subprocess.Popen(cmd2, shell=True, stderr=subprocess.PIPE, env=ENV)
		_,err = p.communicate()

		if (not os.path.isfile(outFile)) or os.path.getsize(outFile) == 0:
			print(err.decode())
			if os.path.isfile(outFile):
				os.remove(outFile)
			raise KaldiProcessError('Generate ngrams language model defeated.')
		else:
			return os.path.abspath(outFile)
	
	finally:
		wordlist.close()

def make_G(dictionary, arpaLM, outFile, n=3):
	'''
	Usage:  obj = arpa2fst("lm.3g.gz","words.txt","G.fst",3)

	Transform ARPA format language model to FST format. Return abspath of generated fst LM.
	<arpaLM> should be a .gz file of ARPA LM.
	<outFile> should be a .fst file name.
	'''
	assert isinstance(arpaLM,str), "<arpaLM> should be a name-like string."
	assert isinstance(outFile,str), "<outFile> should be a name-like string."
	assert isinstance(dictionary,LexiconBank), "Expected <dictionary> is exkaldi LexiconBank object."
	assert isinstance(n,int) and n > 0 and n < 10, "<n> positive int value and must smaller than 10."

	global ENV

	if not os.path.isfile(arpaLM):
		raise PathError("No such file:{}.".format(arpaLM))
	
	if not outFile.endswith('.fst'):
		outFile += ".fst"
	make_dirs_for_outFile(outFile)
	
	words = tempfile.NamedTemporaryFile("w+",encoding='utf-8',suffix=".txt")
	vocab = tempfile.NamedTemporaryFile("w+",encoding='utf-8',suffix=".txt")

	try:
		dictionary.dump_dict(name="words",outFile=words)
		vocab.write("\n".join(list(dictionary("words").keys())))
		vocab.seek(0)

		srilmOpts = "-subset -prune-lowprobs -unk -tolower -order {}".format(n)
		cmd = 'change-lm-vocab -vocab {} -lm {} -write-lm - {} | arpa2fst --disambig-symbol=#0 --read-symbol-table={} - {}'.format(vocab.name,arpaLM,srilmOpts,words.name,outFile)
		p = subprocess.Popen(cmd,shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=ENV)	
		_,err = p.communicate()

		if (not os.path.isfile(outFile)) or os.path.getsize(outFile) < 100:
			print(err.decode())
			if os.path.isfile(outFile):
				os.remove(outFile)
			raise KaldiProcessError("Transform ARPA model to FST model defeated.")
		else:
			return os.path.abspath(outFile)
	
	finally:
		words.close()
		vocab.close()

def fst_is_stochastic(fstFile):
	'''
	Usage:  obj = check_fst_stochastic("G.fst")

	Check if fst is stochastic.
	'''

	assert isinstance(fstFile,str), "<fstFile> should be name-like string."
	
	global ENV

	if not os.path.isfile(fstFile):
		raise PathError("No such file:{}.".format(fstFile))

	cmd = "fstisstochastic {}".format(fstFile)
	p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=ENV)	
	out, err = p.communicate()

	if p.returncode == 1:
		print("FST is not stochastic: {}".format(out.decode()))
		return False
	else:
		return True

def compose_LG(Lfile, Gfile, outFile):
	'''
	Usage:  obj = make_LG("L.fst","G.fst","LG.fst")

	Compose L.fst and G.fst to LG.fst file.
	'''
	assert isinstance(outFile,str), "<outFile> should be name-like string."

	global ENV 

	for f in [Lfile,Gfile]:
		assert isinstance(f,str), "Expected name-like string but got:{}".format(f)
		if not os.path.isfile(f):
			raise PathError("No such file:{}.".format(f))
	if not outFile.endswith(".fst"):
		outFile += ".fst"
	make_dirs_for_outFile(outFile)

	cmd = 'fsttablecompose {} {} | fstdeterminizestar --use-log=true | fstminimizeencoded | fstpushspecial > {}'.format(Lfile,Gfile,outFile)
	p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE ,env=ENV)
	_, err = p.communicate()

	if (not os.path.isfile(outFile)) or os.path.getsize(outFile) == 0:
		print(err.decode())
		if os.path.isfile(outFile):
			os.remove(outFile)
		raise KaldiProcessError("compose L and G defeated.")
	else:
		return os.path.abspath(outFile)

def compose_CLG(dictionary, LGfile, treeFile, outFile):
	'''
	Usage:  obj = make_CLG(lexiconBank, "LG.fst","tree","CLG.fst)

	Compose LG.fst and tree to CLG.fst file.
	'''
	assert isinstance(LGfile,str), "<LGfile> should be a name-like string."
	assert isinstance(treeFile,str), "<treeFile> should be a name-like string."
	assert isinstance(dictionary,LexiconBank), "<dictionary> should be a LexiconBank object."

	global ENV

	for f in [LGfile,treeFile]:
		if not os.path.isfile(f):
			raise PathError("No such file:{}.".format(f))
	
	if not outFile.endswith('.fst'):
		outFile += ".fst"
	make_dirs_for_outFile(outFile)
	iLabelInfoFile = outFile[0:-4] + ".ilabels"

	outValue = []
	for i in ["context-width","central-position"]:
		cmd = 'tree-info {} | grep {} | cut -d" " -f2'.format(treeFile,i)
		p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=ENV)
		out,err = p.communicate()
		if out == b'':
			print(err.decode())
			raise WrongDataFormat("Error when getting {} from tree.".format(i))
		else:
			outValue.append(out.decode().strip())
	
	with open(iLabelInfoFile,'w'):
		pass

	disambig = tempfile.NamedTemporaryFile("w+",encoding='utf-8',suffix=".int")

	try:
		dictionary.dump_dict("disambig",disambig,True)
		cmd = 'fstcomposecontext --context-size={} --central-position={} --read-disambig-syms={} {} {} | fstarcsort --sort_type=ilabel > {}'.format(outValue[0],outValue[1],disambig.name,iLabelInfoFile,LGfile,outFile)
		p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=ENV)
		out,err = p.communicate()

		if (not os.path.isfile(outFile)) or os.path.getsize(outFile) == 0:
			print(err.decode())
			if os.path.isfile(outFile):
				os.remove(outFile)
			raise KaldiProcessError("Generate CLG.fst defeated.")
		else:
			return os.path.abspath(outFile), os.path.abspath(iLabelInfoFile)
	
	finally:
		disambig.close()

def compose_HCLG(CLGfile, hmmFile, treeFile, iLabelInfoFile, outFile, transScale=1.0, loopScale=0.1, removeOOVFile=None):	
	'''
	Usage:  obj = make_HCLG("CLG.fst",final.mdl","HCLG.fst")

	Compose CLG.fst and final.mdl to HCLG.fst.
	'''
	
	global ENV

	for f in [CLGfile,hmmFile,treeFile,iLabelInfoFile]:
		assert isinstance(f,str), "Expected a name-like string but got {}.".format(f)
		if not os.path.isfile(f):
			raise PathError("No such file:{}.".format(f))

	if not (removeOOVFile is None):
		assert isinstance(removeOOVFile,str), "Expected <removeOOVFile> is name-like string."
		if not os.path.isfile(removeOOVFile):
			raise PathError("No such file:{}.".format(removeOOVFile))
	
	if not outFile.endswith(".fst"):
		outFile += ".fst"
	make_dirs_for_outFile(outFile)

	disambigTID = tempfile.NamedTemporaryFile('wb+',suffix='.fst')
	Ha = tempfile.NamedTemporaryFile('wb+',suffix='.fst')
	HCLGa = tempfile.NamedTemporaryFile("wb+",suffix='.fst')

	try:
		cmd1 = "make-h-transducer --disambig-syms-out={} --transition-scale={} {} {} {} > {}".format(disambigTID.name,transScale,iLabelInfoFile,treeFile,hmmFile,Ha.name)

		p1 = subprocess.Popen(cmd1, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=ENV)
		out1, err1 = p1.communicate()

		if os.path.getsize(Ha.name) == 0:
			print(err1.decode())
			raise KaldiProcessError("Make HCLG.fst defeated.")

		if not (removeOOVFile is None):
			clg = "fstrmsymbols --remove-arcs=true --apply-to-output=true {} {}|".format(removeOOVFile,CLGfile)
		else:
			clg = CLGfile
			
		cmd2 = 'fsttablecompose {} \"{}\" | fstdeterminizestar --use-log=true | fstrmsymbols {} | fstrmepslocal | fstminimizeencoded > {}'.format(Ha.name,clg,disambigTID.name,HCLGa.name)

		p2 = subprocess.Popen(cmd2, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=ENV)
		out2, err2 = p2.communicate()

		if os.path.getsize(HCLGa.name) == 0:
			print(err2.decode())
			raise KaldiProcessError("Make HCLG.fst defeated.")
		
		cmd3 = 'add-self-loops --self-loop-scale={} --reorder=true {} {} | fstconvert --fst_type=const > {}'.format(loopScale, hmmFile, HCLGa.name, outFile)
		p3 = subprocess.Popen(cmd3, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=ENV)
		out3, err3 = p3.communicate()

		if (not os.path.isfile(outFile)) or os.path.getsize(outFile) == 0:
			print(err3.decode())
			if os.path.isfile(outFile):
				os.remove(outFile)
			raise KaldiProcessError("Generate HCLG.fst defeated.")

	finally:
		disambigTID.close()
		Ha.close()
		HCLGa.close()

	return os.path.abspath(outFile)	

