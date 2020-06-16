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
Prepare the train, evaluate and test data.
'''

import os
import glob
import subprocess

import exkaldi
from exkaldi.version import version as ExkaldiInfo

import cfg

# prepare tools
ExkaldiInfo.vertify_kaldi_existed()
sph2pipeTool = os.path.join(ExkaldiInfo.KALDI_ROOT, "tools", "sph2pipe_v2.5", "sph2pipe")
if not os.path.join(sph2pipeTool):
    raise Exception(f"Expected sph2pipe tool existed.")

# vertify TIMIT data path simplely
if not os.path.isdir(cfg.timit):
    raise Exception(f"No such directory: {cfg.timit}.")

dirNames = os.listdir(cfg.timit)
if "TRAIN" in dirNames and "TEST" in dirNames:
    uppercaseFlag = True
    trainResourceDir = "TRAIN"
    testResourceDir = "TEST"
    testWavFile = os.path.join(cfg.timit, "TRAIN", "DR1", "FCJF0", "SA1.WAV")
    wavFileSuffix = "WAV"
    txtFileSuffix = "PHN"
elif "train" in dirNames and "test" in dirNames:
    uppercaseFlag = False
    trainResourceDir = "train"
    testResourceDir = "test"
    testWavFile = os.path.join(cfg.timit, "train", "dr1", "fcjf0", "sa1.wav")
    wavFileSuffix = "wav"
    txtFileSuffix = "phn"
else:
    raise Exception(f"No train or test data directory.")

formatCheckCmd = f"{sph2pipeTool}  -f wav {testWavFile}"
out, err, cod = exkaldi.utils.run_shell_command(formatCheckCmd, stderr=subprocess.PIPE)
if cod == 0:
    sphFlag = True
else:
    sphFlag = False

# transform phones 60 -> 48
if cfg.phonesNum != 60:
    phoneMap = {}
    with open( os.path.join(ExkaldiInfo.KALDI_ROOT, "egs", "timit", "s5", "conf", "phones.60-48-39.map"),"r" ,encoding="utf-8") as fr:
        lines = fr.readlines()
        for line in lines:
            line = line.strip().split()
            if len(line) < 3:
                continue
            if cfg.phonesNum == 48:
                phoneMap[line[0]] = line[1]
            elif cfg.phonesNum == 39:
                phoneMap[line[0]] = line[2]
            else:
                raise Exception(f"The numbers of phones must be 60, 48 or 39 but got: {cfg.phonesNum}.")

# A function to generate wav.scp, spk2utt, utt2spk, text files.
def generate_data(wavFiles, outDir):

    wavScp = exkaldi.ListTable(name="wavScp")
    utt2spk = exkaldi.ListTable(name="utt2spk")
    spk2utt = exkaldi.ListTable(name="spk2utt")
    transcription = exkaldi.ListTable(name="trans")

    for Name in wavFiles:

        if Name[-7:].upper() in ["SA1.WAV","SA2.WAV"]:
            continue

        speaker = os.path.basename(os.path.dirname(Name))
        uttID = speaker + "_" + os.path.basename(Name)[0:-4]
        wavFilePath = os.path.abspath(Name)

        # wav scp
        if sphFlag:
            wavScp[uttID] = f"{sph2pipeTool} -f wav {wavFilePath} |"
        else:
            wavScp[uttID] = wavFilePath

        # utt2spk
        utt2spk[uttID] = speaker

        # spk2utt
        if speaker not in spk2utt.keys():
            spk2utt[speaker] = ""
        spk2utt[speaker] += f" {uttID}"

        # transcription
        txtFile = Name[:-3] + txtFileSuffix
        phones = []
        with open(txtFile, "r", encoding="utf-8") as fr:
            lines = fr.readlines()
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            phone = line.split()[-1]
            if cfg.phonesNum != 60:
                if phone == "q":
                    continue
                else:
                    phone = phoneMap[phone]
            phones.append( phone )

        transcription[uttID] = " ".join(phones)
    
    exkaldi.utils.make_dependent_dirs(outDir, pathIsFile=False)

    wavScp.save( os.path.join(outDir, "wav.scp") )
    utt2spk.save( os.path.join(outDir, "utt2spk") )
    spk2utt.save( os.path.join(outDir, "spk2utt") )
    transcription.save( os.path.join(outDir, "text") )

dataOutDir = os.path.join(cfg.outDir, "data")

# generate train data
wavFiles = glob.glob( os.path.join(cfg.timit, trainResourceDir, "*", "*", f"*.{wavFileSuffix}") )
outDir = os.path.join(dataOutDir, "train")
generate_data(wavFiles, outDir)

print(f"Generate train data done: {outDir}.")

# generate dev and test data.
for Name in ["dev", "test"]:

    spkListFile = os.path.join(ExkaldiInfo.KALDI_ROOT, "egs", "timit", "s5", "conf", f"{Name}_spk.list")

    with open(spkListFile, "r", encoding="utf-8") as fr:
        spkList = fr.readlines()
    
    wavFiles = []
    for spk in spkList:
        spk = spk.strip()
        if len(spk) == 0:
            continue
        if uppercaseFlag:
            spk = spk.upper()
        wavFiles.extend( glob.glob( os.path.join(cfg.timit, testResourceDir, "*", spk, f"*.{wavFileSuffix}") ) )
    
    outDir = os.path.join(dataOutDir, Name)

    generate_data(wavFiles, outDir)

    print(f"Generate {Name} data done: {outDir}.")




