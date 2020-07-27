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

Part 3: Compute MFCC feature and CMVN statistics.

'''
import os
import glob
import subprocess
import gc
import time

import exkaldi
from exkaldi import args

def main():

    # ------------- Parse arguments from command line ----------------------
    # 1. Add a discription of this program
    args.discribe("This program is used to compute MFCC feature and CMVN statistics") 
    # 2. Add options
    args.add("--expDir", abbreviation="-e", dtype=str, default="exp", discription="The data and output path of current experiment.")
    args.add("--useEnergy", abbreviation="-u", dtype=bool, default=False, discription="Whether add energy to MFCC feature.")
    args.add("--parallel", abbreviation="-p", dtype=int, default=4, minV=1, maxV=10, discription="The number of parallel process to compute feature of train dataset.")
    # 3. Then start to parse arguments. 
    args.parse()
    # 4. Take a backup of arguments
    args.print_args() # print arguments to display
    argsLogFile = os.path.join(args.expDir, "conf", "compute_mfcc.args")
    args.save(argsLogFile)

    # ---------- Compute mfcc feature of train, dev and test dataset -----------
    if args.useEnergy:
        mfccConfig={"--use-energy":"true"}
    else:
        mfccConfig={"--use-energy":"false"}

    for Name in ["train", "dev", "test"]:
        print(f"Compute {Name} MFCC feature.")

        # 1. compute feature
        if Name == "train" and args.parallel > 1: # use mutiple processes
            wavFiles = exkaldi.utils.split_txt_file(
                                            os.path.join(args.expDir,"data","train","wav.scp"), 
                                            chunks=args.parallel,
                                        )
            feats = exkaldi.compute_mfcc(
                                    wavFiles, 
                                    config=mfccConfig,
                                    outFile=os.path.join(args.expDir,"mfcc","train","raw_mfcc.ark")
                                )
            feat = exkaldi.merge_archieves(feats)
        else:
            feat = exkaldi.compute_mfcc(
                                    os.path.join(args.expDir,"data",Name,"wav.scp"), 
                                    config=mfccConfig,
                                )
            feat.save( os.path.join(args.expDir,"mfcc",Name,"raw_mfcc.ark") )
        print(f"Generate raw MFCC feature done.")
        # Compute CMVN
        cmvn = exkaldi.compute_cmvn_stats(
                                        feat=feat,
                                        spk2utt=os.path.join(args.expDir,"data",Name,"spk2utt"),
                                    )
        cmvn.save( os.path.join(args.expDir,"mfcc",Name,"cmvn.ark") )
        print(f"Generate CMVN statistics done.")
        # Apply CMVN
        feat = exkaldi.use_cmvn(
                            feat=feat,
                            cmvn=cmvn,
                            utt2spk=os.path.join(args.expDir,"data",Name,"utt2spk"),
                        )
        feat.save(os.path.join(args.expDir,"mfcc",Name,"mfcc_cmvn.ark"))
        print(f"Generate MFCC feature (applied CMVN) done.")

    print("Compute MFCC done.")

if __name__ == "__main__":
    main()