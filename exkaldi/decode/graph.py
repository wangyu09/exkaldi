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

"""Make HCLG graph"""
import tempfile
import os
import subprocess
import pickle

from exkaldi.version import version as ExkaldiInfo
from exkaldi.version import WrongPath, WrongOperation, WrongDataFormat, KaldiProcessError, ShellProcessError, UnsupportedType
from exkaldi.utils.utils import make_dependent_dirs, run_shell_command, type_name
from exkaldi.core.achivements import ListTable

class LexiconBank:
	'''
	This class is designed to hold all lexicons which are going to be used when user want to make decoding graph.

	Args:
		<pronLexicon>: should be a file path. We support to generate lexicon bank from 5 kinds of lexicon which are "lexicon", "lexiconp(_disambig)" and "lexiconp_silprob(_disambig)".
						If it is not "lexicon" and silence words or unknown symbol did not exist, error will be raised.
		<silWords>: should be a list object whose members are silence words.
		<unkSymbol>: should be a string used to map the unknown words. If these words are not already existed in <pronLexicon>, their proninciation will be same as themself.
		<optionalSilPhone>: should be a string. It will be used as the pronunciation of "<eps>".
		<extraQuestions>: extra questions to cluster phones when train decision tree.
		<positionDependent>: If True, generate position-dependent lexicons.
		<shareSilPdf>: If True, share the gaussion funtion of silence phones.
		<extraDisambigPhoneNumbers>: extra numbers of disambiguation phone.
		<extraDisambigPhoneNumbers>: extra disambiguation words.
	
	Return:
		A lexicon bank object who holds all lexicons.
	'''

	#------------------------------------- initialization Methods ------------------------------
	# Bug, when use lexiconp_silprob(_disambig) to initialize, need to generate lexiconp_disambig too.

	def __init__(self, pronLexicon, silWords={"<sil>":"<sil>"}, unkSymbol={"unk":"unk"}, optionalSilPhone="<sil>", extraQuestions=[],
					positionDependent=False, shareSilPdf=False, extraDisambigPhoneNumbers=1, extraDisambigWords=[]
				):

		assert isinstance(pronLexicon,str), "Expected <pronLexicon> is name like string."

		if isinstance(silWords, list):
			assert len(silWords) > 0, "Expected at least one silence word in <silWords> but got nothing."
			silWords = dict( (s,s) for s in silWords )
			self.__retain_original_sil_pron = True
		elif isinstance(silWords, dict):
			assert len(silWords) > 0, "Expected at least one silence word in <silWords> but got nothing."
			self.__retain_original_sil_pron = False
		else:
			raise UnsupportedType(f"<silWords> should be list or dict object but got {type_name(silWords)}.")

		if isinstance(unkSymbol, list):
			assert len(unkSymbol) == 1, "Please spicify one unk word."
			unkSymbol = dict( (s,s) for s in unkSymbol )
			self.__retain_original_unk_pron = True
		elif isinstance(unkSymbol, dict):
			assert len(unkSymbol) == 1, "Please spicify one unk word and its' pronunciation."
			self.__retain_original_unk_pron = False
		else:
			raise UnsupportedType(f"<unkSymbol> should be list or dict object but got {type_name(unkSymbol)}.")

		assert isinstance(optionalSilPhone,str) and len(optionalSilPhone) > 0, "Expected one optional silence phone in <optionalSilPhone>."
		assert isinstance(extraQuestions,list), "Expected <extraQuestions> is list object."
		assert isinstance(positionDependent,bool), "Expected <positionDependent> is True or False."
		assert isinstance(shareSilPdf,bool), "Expected <shareSilPdf> is True or False."
		assert isinstance(extraDisambigPhoneNumbers,int) and extraDisambigPhoneNumbers > 0, "Expected <extraDisambigPhoneNumbers> is positive int value."
		assert isinstance(extraDisambigWords,list), "Expected <extraDisambigWords> is list object."

		if not os.path.isfile(pronLexicon):
			raise WrongPath("No such file:{}.".format(pronLexicon))
		
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
			extraDisambigWords = tempfile.NamedTemporaryFile("w+", encoding='utf-8')
			try:
				extraDisambigWords.write("\n".join(self.__parameters["extraDisambigWords"]))
				extraDisambigWords.seek(0)
				cmd = os.path.join(ExkaldiInfo.KALDI_ROOT,'egs','wsj','s5','utils','lang','validate_disambig_sym_file.pl') + f' --allow-numeric "false" {extraDisambigWords.name}'
				out, err, cod = run_shell_command(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
				if cod == 1:
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
		for symbol in list(silWords.keys()) + list(unkSymbol.keys()):
			phone = self.__dictionaries["lexiconp"][(symbol,0)][1].split("_")[0]
			temp.append(phone)
		self.__dictionaries["silence_phones"] = list( set(temp) )

		## Make lexicon: [optional_silence]
		self.__dictionaries["optional_silence"] = optionalSilPhone

		## Make lexicon: [nonsilence_phones]
		temp = []
		for word, pron in self.__dictionaries["lexiconp"].items():
			temp.extend( map(lambda x:x.split("_")[0], pron[1:]) )
		temp = sorted(list(set(temp)))
		self.__dictionaries["nonsilence_phones"] = []
		for phone in temp:
			if (phone not in self.__dictionaries["silence_phones"]) and phone != optionalSilPhone :
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
		
		self.__dictionaries["context_indep"] = self.__dictionaries["silence"]
		
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
		self.__dictionaries["oov"] = list(unkSymbol.keys())[0]

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

	def __check_lexicon_type(self, lexiconFile):
		'''
		When given a lexicon file name, firstly discrimate its type.
		If it does not belong to "lexicon", "lexiconp(_disambig)", "lexiconp_silprob(_disambig)" and "silprob", raise error.
		'''

		with open(lexiconFile, "r", encoding="utf-8") as fr:
			lines =  fr.readlines()
		
		dataList = []
		## Check if it is "silprob"
		if len(lines) >= 4:
			MayBeSilprob = True
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
					MayBeSilprob = False
					break
				else:
					dataList.append( tuple(line) )
			if MayBeSilprob:
				return "silprob", dataList

		## Check if it is "lexicon" or "lexiconp" or "lexiconp_silprob" 
		dictType = None
		for line in lines:
			line = line.strip().split()
			if len(line) == 0:
				continue
			if len(line) == 1:
				raise WrongDataFormat(f"Missing integrated word-(probability)-pronunciation information: {line[0]}.")
			if dictType is None:
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
							line = " ".join(line)
							raise WrongDataFormat(f"Missing integrated word-(probability)-pronunciation information: {line}.")
						else:
							raise WrongDataFormat('Expected "lexicon", "lexiconp(_disambig)", "lexiconp_silprob(_disambig)", "silprob" file but got a unknown format.')
					else:
						try:
							float(line[5])
						except IndexError:
							line = " ".join(line)
							raise WrongDataFormat(f"Missing integrated word-(probability)-pronunciation information: {line}.")					
						except ValueError:
							dictType = "lexiconp_silprob"
						else:
							raise WrongDataFormat('Expected "lexicon", "lexiconp(_disambig)", "lexiconp_silprob(_disambig)", "silprob" file but got a unknown format.')

			dataList.append( (line[0],tuple(line[1:])) )

		if len(dataList) == 0:
			raise WrongOperation(f"Void file: {lexiconFile}.")
		
		## Check if it is a disambiguated lexicon
		if dictType != "lexicon":
			cmd = f'grep "#1" -m 1 < {lexiconFile}'
			out, err, cod = run_shell_command(cmd, stdout=subprocess.PIPE)
			if (isinstance(cod,int) and cod != 0):
				print(err.decode())
				raise ShellProcessError("Failed to vertify disambig symbol.")
			elif len(out) > 0:
				dictType += "_disambig"

		dataList = sorted(dataList, key=lambda x:x[0])

		return dictType, dataList

	def __creat_lexiconp_from_lexicon(self, dataList):
		'''
		This method accepts "lexicon" format data, then generate three lexicons: [lexiconp], [lexiconp_disambig] and [disambig]
		"lexicon" will be deprecated.
		'''
		
		silWords = self.__parameters["silWords"]
		unkSymbol = self.__parameters["unkSymbol"]

		self.__dictionaries["lexiconp"] = {}

		## Add silence words and their pronunciation
		## We will give every record a unique ID (but all of silence and unk words use 0) in order to protect disambiguating words.
		for w,p in silWords.items():
			self.__dictionaries["lexiconp"][(w,0)] = ("1.0",p,)		
		## Add unknown symbol and its pronunciation
		for w,p in unkSymbol.items():
			self.__dictionaries["lexiconp"][(w,0)] = ("1.0",p,)	
		## Add other words and their pronunciation
		disambigFlg = 1
		for word, pron in dataList:
			if word in silWords.keys():
				if self.__retain_original_sil_pron:
					print(f'Warning: silence word "{word}" already existed in provided lexicon. Use it.')
					print(f'If you want specify new pronunciation, give <silWords> a dict object.')
					if len(pron) > 1:
						raise WrongDataFormat(f'Expected only one phone but got {len(pron)}.')
					self.__dictionaries["lexiconp"][(word, 0)] = ("1.0",) + pron
				else:
					print(f'Warning: silence word "{word}" already existed in provided lexicon. Replace it with new pronunciation.')
					print(f'If you want retain orignal pronunciation, give <silWords> a list object.')
					pass

			elif word in unkSymbol.keys():
				if self.__retain_original_unk_pron:
					print(f'Warning: unk symbol "{word}" already existed in provided lexicon. Use it.')
					print(f'If you want specify new pronunciation, give <unkSymbol> a dict object.')
					if len(pron) > 1:
						raise WrongDataFormat(f'Expected only one phone but got {len(pron)}.')
					self.__dictionaries["lexiconp"][(word, 0)] = ("1.0",) + pron
				else:
					print(f'Warning: unk symbol "{word}" already existed in provided lexicon. Replace it with new pronunciation.')
					print(f'If you want retain orignal pronunciation, give <unkSymbol> a list object.')
					pass

			elif word == "<eps>":
				print('Warning: <eps> symbol has already existed in provided lexicon. Remove it.')
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
		If accepted "lexiconp(_disambig)" format data, generate three lexicons: [lexiconp], [lexiconp_disambig] and [disambig]
		If accepted "lexiconp_silprob(_disambig)" format data, generate four lexicons: [lexiconp], [lexiconp_silprob], [lexiconp_silprob_disambig] and [disambig]
		'''
		## <dataList> has a format: [( word, ( probability, *pronunciation ) ), ...]
		## <dictType> should be one of "lexiconp","lexiconp_disambig","lexiconp_silprob" and "lexicon_silprob_disambig" 

		silWords = self.__parameters["silWords"]
		unkSymbol = self.__parameters["unkSymbol"]

		## Check whether the data provided is position-dependent data
		testPron = dataList[0][1][-1]
		if "#" in testPron:
			testPron = dataList[0][1][-2]
		MayBePositionDependent = False
		if len(testPron) > 2 and (testPron[-2:] in ["_S","_B","_I","_E"]):
			MayBePositionDependent = True
		if MayBePositionDependent and ( not self.__parameters["positionDependent"]):
			raise WrongOperation("Position-dependent is unavaliable but appeared in provided lexicon file.")

		## Transform data to Python dict object as well as giving it the unique ID (but all of silence words and unk word use 0)
		## Add check whether silence words and unk word are existed already. If not, raise error.
		temp = {}
		existedSilAndUnk = []
		disambigID = 1
		for word, pron in dataList:
			if word in silWords.keys():

				if "silprob" in dictType:
					assert len(pron) > 5, f'Silence word "{word}" existed but only one phone is allowed in provided lexicon file.'
					if self.__retain_original_sil_pron:
						temp[ (word, 0) ] = pron
					else:
						temp[ (word, 0) ] = pron[0:3] + [silWords[word],]
				else:
					assert len(pron) > 2, f'Silence word "{word}" existed but only one phone is allowed in provided lexicon file.'
					if self.__retain_original_sil_pron:
						temp[ (word, 0) ] = pron
					else:
						temp[ (word, 0) ] = [pron[0],silWords[word],]
				
				existedSilAndUnk.append(word)

			elif word in unkSymbol.keys():

				if "silprob" in dictType:
					assert len(pron) > 5, f'Unk symbol "{word}" existed but only one phone is allowed in provided lexicon file.'
					if self.__retain_original_unk_pron:
						temp[ (word, 0) ] = pron
					else:
						temp[ (word, 0) ] = pron[0:3] + [unkSymbol[word],]
				else:
					assert len(pron) > 2, f'Unk symbol "{word}" existed but only one phone is allowed in provided lexicon file.'
					if self.__retain_original_unk_pron:
						temp[ (word, 0) ] = pron
					else:
						temp[ (word, 0) ] = [pron[0],unkSymbol[word],]
				
				existedSilAndUnk.append(word)

			else:
				temp[ (word, disambigID) ] = pron
				disambigID += 1

		for symbol in silWords.keys():
			if not symbol in existedSilAndUnk:
				raise WrongDataFormat(f'Sience word "{word}" not appeared in provided lexiconp file.')
		for symbol in unkSymbol.keys():
			if not symbol in existedSilAndUnk:
				raise WrongDataFormat(f'Unk symbol "{word}" not appeared in provided lexiconp file.')
		
		## If <dictType> is "lexiconp", generate: [lexiconp] -> [lexiconp_disambig]&[disambig]
		if dictType == "lexiconp":

			self.__dictionaries["lexiconp"] = temp
		
			if self.__parameters["positionDependent"] and (not MayBePositionDependent):
				self.__apply_position_dependent_to_lexiconp(dictType="lexiconp")
		
			self.__add_disambig_to_lexiconp(dictType="lexiconp")
		
		## If <dictType> is "lexiconp_disambig", generate: [lexiconp_disambig] -> [lexiconp]&[disambig]
		elif dictType == "lexiconp_disambig":

			self.__dictionaries["lexiconp_disambig"] = temp
			
			if self.__parameters["positionDependent"] and (not MayBePositionDependent):
				self.__apply_position_dependent_to_lexiconp(dictType="lexiconp_disambig")
			
			self.__remove_disambig_from_lexiconp_disambig(dictType="lexiconp_disambig")

		## If <dictType> is "lexiconp_silprob", generate: [lexiconp_silprob] -> [lexiconp_silprob_disambig]&[disambig] -> [lexiconp]&[lexiconp_disambig]
		elif dictType == "lexiconp_silprob":

			self.__dictionaries["lexiconp_silprob"] = temp

			if self.__parameters["positionDependent"] and (not extimatePositionDependent):
				self.__apply_position_dependent_to_lexiconp(dictType="lexiconp_silprob")
			
			self.__add_disambig_to_lexiconp(dictType="lexiconp_silprob")

			self.__dictionaries["lexiconp"] = {}
			self.__dictionaries["lexiconp_disambig"] = {}
			for word, pron in self.__dictionaries["lexiconp_silprob"].items():
				self.__dictionaries["lexiconp"][word] = (pron[0],) + pron[4:]
				self.__dictionaries["lexiconp_disambig"][word] = (pron[0],) + self.__dictionaries["lexiconp_silprob_disambig"][word][4:]

		## If <dictType> is "lexiconp_silprob_disambig", generate: [lexiconp_silprob_disambig] -> [lexiconp_silprob]&[disambig] -> [lexiconp_disambig]&[lexiconp_disambig]	
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
		if dictType == "lexiconp":
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

	def __add_disambig_to_lexiconp(self, dictType="lexiconp"):
		'''
		This method is used to add phone-level disambiguation to [lexiconp] or [lexiconp_silprob]
		Lexicon, [disambig], will be gained and parameter, "ndisambig", will be updated selmeanwhile
		'''

		## <dictType> is one of "lexiconp" and "lexiconp_silprob" 
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
			
			cmd = os.path.join(ExkaldiInfo.KALDI_ROOT,"egs","wsj","s5","utils","add_lex_disambig.pl") + f" --pron-probs {cmdOption}{lexiconp.name} {lexiconpDisambig.name}"
			out, err, cod = run_shell_command(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
			if (isinstance(cod,int) and cod!=0 ) or out == b"":
				print(err.decode())
				raise KaldiProcessError("Failed to add disambig phones to lexiconp.")
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
			result = self.__dictionaries[name]
			if name in ["words", "phones"]:
				result = ListTable(result, name=name)
			return result

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

			elif name in ["disambig", "silence", "nonsilence", "wdisambig_phones","context_indep"]:
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

	def dump_dict(self, name, outFile=None, dumpInt=False):
		'''
		Usage: lexobj.dump_dict(name="lexiconp_disambig", outFile="lexiconp_disambig.txt")

		Save the lexicon to file with Kaldi format.
		If <dumpInt> is True, replace phones or words with int ID.
		Some lexicons have not corresponding int table. So if you require them, a warning message will be printed and nothing will be saved.
		In addition, <outFile> can received a tempfile._TemporaryFileWrapper object if you just want to use the file temporarily.  
		'''
		
		if outFile is None:
			pass
		elif isinstance(outFile, str):
			if dumpInt is False:
				if not outFile.endswith(".txt"):
					outFile += ".txt"
			else:
				if not outFile.endswith(".int"):
					outFile += ".int"
			make_dependent_dirs(outFile, pathIsFile=True)
		elif not isinstance(outFile,tempfile._TemporaryFileWrapper):
			raise WrongOperation('<outFile> shoule be a name-like string.')		

		def write_file(fileName, message):
			if fileName is None:
				return message
			elif isinstance(fileName,tempfile._TemporaryFileWrapper):
				fileName.read()
				fileName.seek(0)
				fileName.write(message)
				fileName.seek(0)
				return None
			else:
				with open(fileName, "w", encoding='utf-8') as fw:
					fw.write(message)
				return os.path.abspath(fileName)
		
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
				result = write_file(outFile, "\n".join(contents))
				return	result
					
		## Type2: tuple, ()
		elif name in ["nonsilence_phones", "silence_phones", "disambig", "silence", "nonsilence", 
					  "wdisambig", "wdisambig_phones", "wdisambig_words", "context_indep"]:
			contents = self.__call__(name,dumpInt)
			if not contents is None:
				result = write_file(outFile, "\n".join(contents))
				return result

		## Type3: tuple, (tuple,)
		elif name in ["sets","extra_questions"]:
			contents = []
			for value in self.__call__(name,dumpInt):
				contents.append(" ".join(value))
			result = write_file(outFile, "\n".join(contents))
			return result
		
		## Type4: dict, { str:int or str }
		elif name in ["phones", "words", "word_boundary", "silprob"]:
			contents = []
			temp = self.__call__(name,dumpInt)
			if not temp is None:
				for key, value in temp.items():
					contents.append("{} {}".format(key, value))
				result = write_file(outFile, "\n".join(contents))
				return result

		## Type5: str						
		elif name in ["oov", "optional_silence"]:
			contents = self.__call__(name,dumpInt)
			result = write_file(outFile, contents)
			return result

		## Type6: special format for roots
		elif name == "roots":
			contents = []
			temp = self.__call__(name,dumpInt)
			if len(temp["not-shared not-split"]) > 0: 
				contents.append("not-shared not-split {}".format(" ".join(temp["not-shared not-split"])))
			for phones in temp["shared split"]:
				phones = " ".join(phones)
				contents.append("shared split {}".format(phones))
			result = write_file(outFile, "\n".join(contents))
			return result
	
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

	def save(self, fileName):
		'''
		Save LexiconBank object to binary file.

		Args:
			<fileName>: file name.
		'''
		assert isinstance(fileName, str), "<fileName> should be a string."
		if not fileName.rstrip().endswith(".lex"):
			fileName += ".lex"
		make_dependent_dirs(fileName, pathIsFile=True)

		with open(fileName, "wb") as fw:
			pickle.dump(self, fw)

		return os.path.abspath(fileName)

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
				raise WrongPath("No such file: {}.".format(target))

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
				raise WrongPath("No such file: {}.".format(target))

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
				raise WrongPath("No such file: {}.".format(targetFile))

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
			raise UnsupportedType("<targetFile> is an unknown lexicon format.")

def lexicon_bank(pronLexiconFile, silWords=["<sil>"], unkSymbol="unk", optionalSilPhone="<sil>", extraQuestions=[],
					positionDependent=False, shareSilPdf=False, extraDisambigPhoneNumbers=1, extraDisambigWords=[]):
		
		return LexiconBank(pronLexiconFile, silWords, unkSymbol, optionalSilPhone, extraQuestions, 
							positionDependent, shareSilPdf, extraDisambigPhoneNumbers, extraDisambigWords)

def load_lex(target):
	'''
	Load LexiconBank object from file.
	'''
	assert isinstance(target,str), "<target> must be file name."
	if not os.path.isfile(target):
		raise WrongPath(f"No such file: {target}.")
	
	with open(target, "rb") as fr:
		obj = pickle.load(fr)
	
	if not isinstance(obj, LexiconBank):
		raise WrongDataFormat("It is not a LexiconBank object avaliable.")
	else:
		return obj

def make_L(lexicons, outFile, useSilprobLexicon=False, useSilprob=0.5, useDisambigLexicon=False):
	'''
	Generate L.fst(or L_disambig.fst) file

	Args:
		<lexicons>: An exkaldi LexiconBank object.
		<outFile>: Output fst file path such as "L.fst".
		<useSilprobLexicon>: If True, use silence probability lexicon.
		<useSilprob>: If useSilprobLexicon is False, use constant silence probability.
		<useDisambigLexicon>: If true, use lexicon with disambig symbol.
	Return:
		Absolute path of generated fst file.
	'''
	assert isinstance(lexicons, LexiconBank), "Expected <lexicons> is exkaldi LexiconBank object."
	assert isinstance(outFile, str), "Expected <outFile> is a name-like string."
	assert isinstance(useSilprobLexicon, bool), "Expected <useSilprobLexicon> is True or False."
	assert isinstance(useSilprob, float) and useSilprob >= 0 and useSilprob <= 1, "Expected <useSilprob> is a probility-like float value."
	assert isinstance(useDisambigLexicon, bool), "Expected <useDisambigLexicon> is True or False."

	if useSilprobLexicon:
		for name in ["lexiconp_silprob", "silprob"]:
			if not name in lexicons.view:
				raise WrongOperation(f'When making silprob, expected "{name}" is existed in lexicon bank.')

	if not outFile.rstrip().endswith(".fst"):
		outFile += ".fst"
	make_dependent_dirs(outFile, pathIsFile=True)

	silPhone = lexicons("optional_silence")
	ndisambig = lexicons.get_parameter("ndisambig")

	lexiconTemp = tempfile.NamedTemporaryFile("w+", encoding='utf-8', suffix="_lexicon.txt")
	silprobTemp = tempfile.NamedTemporaryFile("w+", encoding='utf-8', suffix="_silprob.txt")
	phonesTemp = tempfile.NamedTemporaryFile("w+", encoding='utf-8', suffix="_phones.txt")
	wordsTemp = tempfile.NamedTemporaryFile("w+", encoding='utf-8', suffix="_words.txt")
	wdisambigPhonesTemp = tempfile.NamedTemporaryFile("w+", encoding='utf-8', suffix="_wdphones.int")
	wdisambigWordsTemp = tempfile.NamedTemporaryFile("w+", encoding='utf-8', suffix="_wdwords.int")

	try:
		lexicons.dump_dict("phones", phonesTemp)
		lexicons.dump_dict("words", wordsTemp)

		## Generate text format fst
		if useDisambigLexicon:
			# If use disambig lexiconp
			if useSilprobLexicon:
				# If specify silprob lexicon, use silprob disambig lexiconp
				lexicons.dump_dict("silprob", silprobTemp)
				lexicons.dump_dict("lexiconp_silprob_disambig", lexiconTemp)
				cmd1 = os.path.join(ExkaldiInfo.KALDI_ROOT,"egs","wsj","s5","utils","lang","make_lexicon_fst_silprob.py")
				cmd1 += f' --sil-phone=\"{silPhone}\" --sil-disambig=#{ndisambig} {lexiconTemp.name} {silprobTemp.name}'
			else:
				# If use disambig lexiconp
				lexicons.dump_dict("lexiconp_disambig", lexiconTemp)
				cmd1 = os.path.join(ExkaldiInfo.KALDI_ROOT,"egs","wsj","s5","utils","lang","make_lexicon_fst.py")
				cmd1 += f' --sil-prob={useSilprob} --sil-phone=\"{silPhone}\" --sil-disambig=#{ndisambig} {lexiconTemp.name}'

		else:
			# If use lexiconp
			if useSilprobLexicon:
				# If specify silprob lexicon, use silprob lexiconp
				lexicons.dump_dict("silprob", silprobTemp)
				lexicons.dump_dict("lexiconp_silprob", lexiconTemp)
				cmd1 = os.path.join(ExkaldiInfo.KALDI_ROOT,"egs","wsj","s5","utils","lang","make_lexicon_fst_silprob.py")
				cmd1 += f' --sil-phone=\"{silPhone}\" {lexiconTemp.name} {silprobTemp.name}'
			else:
				lexicons.dump_dict("lexiconp", lexiconTemp)
				cmd1 = os.path.join(ExkaldiInfo.KALDI_ROOT,"egs","wsj","s5","utils","lang","make_lexicon_fst.py")
				cmd1 += f' --sil-prob={useSilprob} --sil-phone=\"{silPhone}\" {lexiconTemp.name}'					

		out1, err1, cod1 = run_shell_command(cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		
		if (isinstance(cod1,int) and cod1 != 0) or out1 is None or (isinstance(out1,str) and len(out1) == 0):
			print(err1.decode())
			raise KaldiProcessError("Failed to generate text format fst.")
		
		cmd2 = f"fstcompile --isymbols={phonesTemp.name} --osymbols={wordsTemp.name} --keep_isymbols=false --keep_osymbols=false - | "
		if useDisambigLexicon:
			lexicons.dump_dict("wdisambig_phones", wdisambigPhonesTemp, True)
			lexicons.dump_dict("wdisambig_words", wdisambigWordsTemp, True)
			cmd2 += f"fstaddselfloops {wdisambigPhonesTemp.name} {wdisambigWordsTemp.name} | "
		cmd2 += f"fstarcsort --sort_type=olabel > {outFile}"			

		out2, err2, cod2 = run_shell_command(cmd2, stdin=subprocess.PIPE, stderr=subprocess.PIPE, inputs=out1)
		
		if isinstance(cod2,int) and cod2 != 0:
			print(err.decode())
			if os.path.isfile(outFile):
				os.remove(outFile)
			if useDisambig:
				raise KaldiProcessError("Failed to generate L_disambig.fst.")
			else:
				raise KaldiProcessError("Failed to generate L.fst.")
		else:
			return os.path.abspath(outFile)

	finally:
		lexiconTemp.close()
		silprobTemp.close()
		phonesTemp.close()
		wordsTemp.close()
		wdisambigPhonesTemp.close()
		wdisambigWordsTemp.close()

def make_G(lexicons, arpaFile, outFile, order=3):
	'''
	Transform ARPA format language model to FST format. 
	
	Args:
		<lexicon>: A LexiconBank object.
		<arpaFile>: An ARPA LM file path.
		<outFile>: A fst file name.
		<order>: the maximum order to use when make G fst.
	Return:
		Absolute path of generated fst file.
	'''
	assert isinstance(arpaFile, str), "<arpaFile> should be a name-like string."
	assert isinstance(outFile, str), "<outFile> should be a name-like string."
	assert isinstance(lexicons, LexiconBank), "Expected <lexicons> is exkaldi LexiconBank object."
	assert isinstance(order, int) and order > 0 and order < 10, "<order> positive int value and must smaller than 10."

	if not os.path.isfile(arpaFile):
		raise WrongPath(f"No such file:{arpaFile}.")
	if not outFile.rstrip().endswith('.fst'):
		outFile += ".fst"

	make_dependent_dirs(outFile, pathIsFile=True)
	
	wordsTemp = tempfile.NamedTemporaryFile("w+", encoding='utf-8', suffix="_words.txt")

	try:
		lexicons.dump_dict(name="words", outFile=wordsTemp)

		cmd =  f'arpa2fst --disambig-symbol=#0 --read-symbol-table={wordsTemp.name} {arpaFile} {outFile}'
		
		out, err, cod = run_shell_command(cmd, stderr=subprocess.PIPE)
		
		if cod != 0:
			print(err.decode())
			raise KaldiProcessError("Failed to transform ARPA model to FST format.")
		else:
			return os.path.abspath(outFile)
	
	finally:
		wordsTemp.close()

def fst_is_stochastic(fstFile):
	'''
	Check if fst is stochastic.

	Args:
		<fstFile>: fst file path.
	Return:
		true or False.
	'''
	assert isinstance(fstFile, str), "<fstFile> should be name-like string."
	
	if not os.path.isfile(fstFile):
		raise WrongPath(f"No such file:{fstFile}.")

	cmd = f"fstisstochastic {fstFile}"
	out, err, returnCode = run_shell_command(cmd, stdout=subprocess.PIPE)

	if returnCode == 1:
		print(f"FST is not stochastic: {out.decode()}")
		return False
	else:
		return True

def compose_LG(Lfile, Gfile, outFile="LG.fst"):
	'''
	Compose L and G to LG

	Args:
		<Lfile>: L.fst file path.
		<Gfile>: G.fst file path.
		<outFile>: output LG.fst file.
	Return:
	    An absolute file path.
	'''
	assert isinstance(outFile, str), "<outFile> should be name-like string."

	for fileName in [Lfile, Gfile]:
		assert isinstance(fileName, str), f"Expected name-like string but got:{fileName}"
		if not os.path.isfile(fileName):
			raise WrongPath(f"No such file:{fileName}.")
	if not outFile.rstrip().endswith(".fst"):
		outFile += ".fst"
	make_dependent_dirs(outFile, pathIsFile=True)

	cmd = f'fsttablecompose {Lfile} {Gfile} | fstdeterminizestar --use-log=true | fstminimizeencoded | fstpushspecial > {outFile}'
	out, err, cod = run_shell_command(cmd, stderr=subprocess.PIPE)

	if cod != 0:
		print(err.decode())
		if os.path.isfile(outFile):
			os.remove(outFile)
		raise KaldiProcessError("Failed to compose L and G file.")
	else:
		return os.path.abspath(outFile)

def compose_CLG(lexicons, tree, LGfile, outFile="CLG.fst"):
	'''
	Compose tree and LG to CLG file.

	Args:
		<lexicons>: LexiconBank object.
		<tree>: file path or DecisionTree object.
		<LGfile>: LG.fst file.
		<outFile>: output CLG.fst file.
	Return:
	    CLG file path and ilabel file path.
	'''
	assert isinstance(lexicons, LexiconBank), "<lexicon> should be a LexiconBank object."
	assert isinstance(LGfile, str), "<LGfile> should be a name-like string."
	
	if not os.path.isfile(LGfile):
		raise WrongPath(f"No such file:{LGfile}.")
	if not outFile.rstrip().endswith('.fst'):
		outFile += ".fst"
	make_dependent_dirs(outFile)
	iLabelFile = outFile[0:-4] + ".ilabels"

	if isinstance(tree, str):
		assert os.path.isfile(tree), f"No such file: {tree}."
		cmd = f"tree-info {tree}"
		out, err, cod = run_shell_command(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		if cod != 0:
			print(err.decode())
		else:
			out = out.decode().strip().split("\n")
			contextWidth = out[1].split()[-1]
			centralPosition = out[2].split()[-1]
	else:
		assert type_name(tree) == "DecisionTree", f"<tree> should be tree file or exkaldi DecisionTree object but got: {type_name(tree)}."
		contextWidth = tree.contextWidth
		centralPosition = tree.centralPosition

	disambigTemp = tempfile.NamedTemporaryFile("w+", encoding='utf-8', suffix="_disambig.int")
	try:
		lexicons.dump_dict("disambig", disambigTemp, True)
		cmd = f'fstcomposecontext --context-size={contextWidth} --central-position={centralPosition}'
		cmd += f' --read-disambig-syms={disambigTemp.name} {iLabelFile} {LGfile} |'
		cmd += f' fstarcsort --sort_type=ilabel > {outFile}'
		
		out, err, cod = run_shell_command(cmd, stderr=subprocess.PIPE)

		if cod != 0:
			print(err.decode())
			raise KaldiProcessError("Failed to generate CLG.fst file.")
		else:
			return os.path.abspath(outFile), os.path.abspath(iLabelFile)
	
	finally:
		disambigTemp.close()

def compose_HCLG(hmm, tree, CLGfile, iLabelFile, outFile="HCLG.fst", transScale=1.0, loopScale=0.1, removeOOVFile=None):	
	'''
	Compose HCLG file.

	Args:
		<hmm>: HMM object or file path.
		<tree>: DecisionTree object or file path.
		<CLGfile>: CLG.fst file path.
		<iLabelFile>: ilabel file path.
		<outFile>: output HCLG.fst file path.
		<transScale>: transform scale.
		<loopScale>: self loop scale.
	Return:
	    Absolute path of HCLG file.
	'''
	for fileName in [CLGfile, iLabelFile]:
		assert isinstance(fileName, str), f"Expected a name-like string but got {fileName}."
		if not os.path.isfile(fileName):
			raise WrongPath(f"No such file:{fileName}.")

	if not (removeOOVFile is None):
		assert isinstance(removeOOVFile, str), "Expected <removeOOVFile> is name-like string."
		if not os.path.isfile(removeOOVFile):
			raise WrongPath(f"No such file:{removeOOVFile}.")
	
	if not outFile.rstrip().endswith(".fst"):
		outFile += ".fst"
	make_dependent_dirs(outFile)

	disambigTID = tempfile.NamedTemporaryFile('wb+', suffix='_disambigTID.fst')
	Ha = tempfile.NamedTemporaryFile('wb+', suffix='_Ha.fst')
	HCLGa = tempfile.NamedTemporaryFile("wb+", suffix='_HCLGa.fst')
	treeTemp = tempfile.NamedTemporaryFile('wb+', suffix='.tree')
	modelTemp = tempfile.NamedTemporaryFile('wb+', suffix='.mdl')

	try:
		if isinstance(hmm, str):
			assert os.path.isfile(hmm), f"No such file: {hmm}."
			hmmFile = hmm
		else:
			assert type_name(hmm) in ["MonophoneHMM","TriphoneHMM"], f"<hmm> should be file or exkaldi HMM object but got: {type_name(hmm)}."
			modelTemp.write(hmm.data)
			modelTemp.seek(0)
			hmmFile = modelTemp.name

		if isinstance(tree, str):
			assert os.path.isfile(tree), f"No such file: {tree}."
			treeFile = tree
		else:
			assert type_name(tree) == "DecisionTree", f"<tree> should be file or exkaldi DecisionTree object but got: {type_name(tree)}."
			treeTemp.write(tree.data)
			treeTemp.seek(0)
			treeFile = treeTemp.name

		cmd1 = f"make-h-transducer --disambig-syms-out={disambigTID.name} --transition-scale={transScale} "
		cmd1 += f"{iLabelFile} {treeFile} {hmmFile} > {Ha.name}"

		out1, err1, cod1 = run_shell_command(cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

		if cod1 != 0:
			print(err1.decode())
			raise KaldiProcessError("Failed to make make H transducer.")
		
		disambigTID.seek(0)
		Ha.seek(0)

		if not (removeOOVFile is None):
			clg = f"fstrmsymbols --remove-arcs=true --apply-to-output=true {removeOOVFile} {CLGfile}|"
		else:
			clg = CLGfile
			
		cmd2 = f'fsttablecompose {Ha.name} \"{clg}\" | fstdeterminizestar --use-log=true | '
		cmd2 += f'fstrmsymbols {disambigTID.name} | fstrmepslocal | fstminimizeencoded > {HCLGa.name}'

		out2, err2, cod2 = run_shell_command(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

		if cod2 != 0:
			print(err2.decode())
			raise KaldiProcessError("Failed to make HCLGa.fst.")
		
		HCLGa.seek(0)

		cmd3 = f'add-self-loops --self-loop-scale={loopScale} --reorder=true {hmmFile} {HCLGa.name} | fstconvert --fst_type=const > {outFile}'
		out3, err3, cod3 = run_shell_command(cmd3, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

		if cod3 != 0:
			print(err3.decode())
			raise KaldiProcessError("Failed to generate HCLG.fst.")
		else:
			return os.path.abspath(outFile)	

	finally:
		disambigTID.close()
		Ha.close()
		HCLGa.close()
		treeTemp.close()
		modelTemp.close()

def make_graph(lexicons, arpaFile, hmm, tree, tempDir, order=3, useSilprobLexicon=False, useSilprob=0.5, 
				useDisambigLexicon=False, useLFile=None, outFile="HCLG.fst", transScale=1.0, loopScale=0.1, removeOOVFile=None):
	'''
	Make HCLG decode graph.

	Args:
		<lexicons>: exkaldi lexicon bank object.
		<arpaFile>: arpa file path.
		<hmm>: file path or exkaldi HMM object.
		<tree>: file path or exkaldi DecisionTree object.
		<tempDir>: a directory to storage intermidiate files.
		<LFile>: If it is None, make L.fst.
				else, do not make a new one and use this.
	
	Return:
		absolute path of HCLG file.
	'''
	assert isinstance(tempDir,str), "<tempDir> should be directory name."
	make_dependent_dirs(tempDir, pathIsFile=False)


	if useLFile is None:
		if useDisambig:
			useLFile = os.path.join(tempDir, "L_disambig.fst")
		else:
			useLFile = os.path.join(tempDir, "L.fst")
		useLFile = make_L(lexicons, useLFile, useSilprobLexicon, useSilprob, useDisambigLexicon)
		print(f"Make L done: {useLFile}.")
	else:
		print(f"Skip making L. Use: {useLFile}.")

	GFile = os.path.join(tempDir, "G.fst")
	GFile = make_G(lexicons, arpaFile, GFile, order=order)
	print(f"Make G done: {GFile}.")

	LGFile = os.path.join(tempDir, "LG.fst")
	LGFile = compose_LG(useLFile, GFile, LGFile)
	print(f"Compose LG done: {LGFile}.")

	CLGFile = os.path.join(tempDir, "CLG.fst")
	CLGFile, ilabelFile = compose_CLG(lexicons, tree, LGFile, CLGFile)
	print(f"Compose CLG done: {CLGFile}.")
	print(f"Ilabel info: {ilabelFile}.")

	HCLGFile = os.path.join(tempDir, "HCLG.fst")
	HCLGFile = compose_HCLG(hmm, tree, CLGFile, ilabelFile, HCLGFile, transScale, loopScale, removeOOVFile)
	print(f"Make HCLG done: {HCLGFile}.")

	return HCLGFile
	

