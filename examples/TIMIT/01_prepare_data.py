# coding=utf-8
#
# Yu Wang (University of Yamanashi)
# Jun, 2020
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

'''
Timit HMM-GMM training recipe.

Part 1: prepare TIMIT data.

'''
import os
import glob
import gc

import exkaldi
from exkaldi import args
from exkaldi import info
from exkaldi import declare

def generate_data(wavFiles, expDir, sphFlag, sph2pipeTool, txtFileSuffix, phoneMap_60_to_48):

    wavScp = exkaldi.ListTable(name="wavScp")
    utt2spk = exkaldi.ListTable(name="utt2spk")
    spk2utt = exkaldi.ListTable(name="spk2utt")
    transcription = exkaldi.Transcription(name="trans")

    for Name in wavFiles:
        if Name[-7:].upper() in ["SA1.WAV","SA2.WAV","sa1.wav","sa2.wav"]:
            continue
        speaker = os.path.basename( os.path.dirname(Name) )
        uttID = speaker + "_" + os.path.basename(Name)[0:-4]
        wavFilePath = os.path.abspath(Name)
        # 1. wav.scp
        if sphFlag:
            wavScp[uttID] = f"{sph2pipeTool} -f wav {wavFilePath} |"
        else:
            wavScp[uttID] = wavFilePath
        # 2. utt2spk
        utt2spk[uttID] = speaker
        # 3. spk2utt
        if speaker not in spk2utt.keys():
            spk2utt[speaker] = f"{uttID}"
        else:
            spk2utt[speaker] += f" {uttID}"
        # 4. transcription
        txtFile = Name[:-3] + txtFileSuffix
        phones = []
        with open(txtFile,"r",encoding="utf-8") as fr:
            lines = fr.readlines()
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            phone = line.split()[-1]
            if phone == "q": # discard phone "q"
                continue
            else:
                phone = phoneMap_60_to_48[phone]
            phones.append(phone)
        transcription[uttID] = " ".join(phones)
    # Save to files
    wavScp.save( os.path.join(expDir, "wav.scp") )
    utt2spk.save( os.path.join(expDir, "utt2spk") )
    spk2utt.save( os.path.join(expDir, "spk2utt") )
    transcription.save( os.path.join(expDir, "text") )
    print(f"Generate data done: {expDir}.")

def main():

    # ------------- Parse arguments from command line ----------------------
    # 1. Add a discription of this program
    args.discribe("This program is used to prepare TIMIT data.") 
    # 2. Add some options
    args.add("--timitRoot", dtype=str, abbr="-t", default="/Corpus/TIMIT", discription="The root path of timit dataset.")
    args.add("--expDir", dtype=str, abbr="-e", default="exp", discription="The output path to save generated data.")
    # 3. Then start to parse arguments. 
    args.parse()
    # 4. Take a backup of arguments
    args.save( os.path.join(args.expDir,"conf","prepare_data.args") )

    # ------------- Do some preparative work ----------------------
    # 2. Ensure Kaldi has existed
    declare.kaldi_existed()
    # 3. sph2pipe tool will be used if the timit data is sph format.
    sph2pipeTool = os.path.join(info.KALDI_ROOT,"tools","sph2pipe_v2.5","sph2pipe")
    declare.is_file("sph2pipe tool",sph2pipeTool)

    # ------------- Check TIMIT data format -------------
    # 1. Get the directory name
    declare.is_dir("TIMIT root directory", args.timitRoot)
    dirNames = os.listdir(args.timitRoot)
    if "TRAIN" in dirNames and "TEST" in dirNames:
        uppercaseFlag = True
        trainResourceDir = "TRAIN"
        testResourceDir = "TEST"
        testWavFile = os.path.join(args.timitRoot,"TRAIN","DR1","FCJF0","SA1.WAV") # used to test the file format
        wavFileSuffix = "WAV"
        txtFileSuffix = "PHN"
    elif "train" in dirNames and "test" in dirNames:
        uppercaseFlag = False
        trainResourceDir = "train"
        testResourceDir = "test"
        testWavFile = os.path.join(args.timitRoot,"train","dr1","fcjf0","sa1.wav") # used to test the file format
        wavFileSuffix = "wav"
        txtFileSuffix = "phn"
    else:
        raise Exception(f"Wrong format of train or test data directories.")
    # 2. check whether wave file is sph format.
    formatCheckCmd = f"{sph2pipeTool} -f wav {testWavFile}"
    out,err,cod = exkaldi.utils.run_shell_command(formatCheckCmd, stderr="PIPE")
    if cod == 0:
        sphFlag = True
    else:
        sphFlag = False
    
    # --------- Generate phone-map dictionary --------
    # 1. Generate 60-48 catagories and 48-39 catagories mapping dictionary
    phoneMap_60_to_48 = exkaldi.ListTable(name="69-48")
    phoneMap_48_to_39 = exkaldi.ListTable(name="48-39")
    mapFile = os.path.join(info.KALDI_ROOT,"egs","timit","s5","conf","phones.60-48-39.map")
    declare.is_file("60-48-39 phone map", mapFile) # Check whether or not it existed
    with open(mapFile,"r",encoding="utf-8") as fr:
        lines = fr.readlines()
        for line in lines:
            line = line.strip().split()
            if len(line) < 3: #phone "q" will be omitted temporarily.
                continue
            phoneMap_60_to_48[line[0]] = line[1]
            phoneMap_48_to_39[line[1]] = line[2]
    # 2. Save 48-39 phone map for futher use.
    phoneMap_48_to_39.save(os.path.join(args.expDir,"dict","phones.48_to_39.map"))

    # --------- Generate train dataset --------
    wavs = glob.glob(os.path.join(args.timitRoot,trainResourceDir,"*","*",f"*.{wavFileSuffix}"))
    out = os.path.join(args.expDir,"data","train")
    generate_data(wavs,out,sphFlag,sph2pipeTool,txtFileSuffix,phoneMap_60_to_48)

    # --------- Generate dev and test data --------
    for Name in ["dev", "test"]:
        spkListFile = os.path.join( info.KALDI_ROOT,"egs","timit","s5","conf",f"{Name}_spk.list" )
        declare.is_file(f"speakers list for {Name}", spkListFile) # Check whether or not it existed
        with open(spkListFile,"r",encoding="utf-8") as fr:
            spkList = fr.readlines()
        wavs = []
        for spk in spkList:
            spk = spk.strip()
            if len(spk) == 0:
                continue
            if uppercaseFlag:
                spk = spk.upper()
            wavs.extend(glob.glob(os.path.join(args.timitRoot,testResourceDir,"*",spk,f"*.{wavFileSuffix}")))
        
        out = os.path.join(args.expDir,"data",Name)
        generate_data(wavs,out,sphFlag,sph2pipeTool,txtFileSuffix,phoneMap_60_to_48)

if __name__ == "__main__":
    main()