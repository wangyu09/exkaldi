# coding=utf-8
#
# Yu Wang (University of Yamanashi)
# May, 2020
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
import os
import sys
from itertools import groupby
import subprocess

from exkaldi.error import *
from exkaldi.utils.utils import run_shell_command, type_name
from exkaldi.utils import declare
from exkaldi.core.archive import Transcription, NumpyProb
from exkaldi.nn.nn import softmax

def convert_field(prob, originVocabs, targetVocabs, retainOOV=False):
    '''
    Tranform the dimensions of probability to target field.

    Args:
        <prob>: An exkaldi probability object. This probalility should be an output of Neural Network.
        <originVocabs>: list of original field vocabulary.
        <originVocabs>: list of target field vocabulary.
        <retainOOV>: If True, target words which are not in original vocabulary will be retained in minimum probability of each frame. 
    Return:
        An new exkaldi probability object and a list of new target vocabulary.  
    '''	
    declare.is_classes("originVocabs", originVocabs, list)
    declare.is_classes("targetVocabs", targetVocabs, list)
    assert len(targetVocabs) > 0, f"Target vocabulary is void."

    declare.is_probability("prob", prob)
    if type_name(prob) == "BytesProb":
        prob = prob.to_numpy()
    elif type_name(prob) == "IndexTable":
        prob = prob.read_record("prob").to_numpy()
    
    probDim = prob.dim
    declare.equal( "the dimension of probability", probdim, "the number of words", len(originVocabs))

    origin_w2i = dict( (w,i) for i,w in enumerate(originVocabs) )
    
    retainIDs = []
    newTargetVocabs = []
    for w in targetVocabs:
        try:
            ID = origin_w2i[w]
        except KeyError:
            if retainOOV is True:
                newTargetVocabs.append(w)
                retainIDs.append(None)
            else:
                pass
        else:
            newTargetVocabs.append(w)
            retainIDs.append(ID) 

    results = {}
    for utt, pb in prob.items:
        declare.is_classes("prob", prob, np.ndarray)
        declare.is_classes("the rank of matrix shape", len(pb.shape), "expected rank", 2)
        if retainOOV is True:
            padding = np.min(pb, axis=1)
        new = np.zeros(shape=(pb.shape[0], len(retainIDs)), dtype=np.float32)
        for index, i in enumerate(retainIDs):
            if i is None:
                new[:, index] = padding
            else:
                new[:, index] = pb[:, i]
            results[utt] = new

        results[utt] = new
    
    newName = f"convert({prob.name})"
    return NumpyProb(data=results, name=newName), newTargetVocabs

def beam_search(prob, vocab, beam=5):
    '''
    Pure beam search.
    '''
    raise WrongOperation("exkaldi.decode.e2e.beam_search is unavaliable now.")
    
def ctc_greedy_search(prob, vocabs, blankID=None):
    '''
    The best path decoding algorithm.

    Args:
        <prob>: An exkaldi probability object. This probalility should be an output of Neural Network with CTC loss fucntion.
        <vocabs>: a list of vocabulary.
        <blankID>: specify the ID of blank symbol. If None, use the last dimentionality of <prob>.
    Return:
        An exkaldi Transcription object of decoding results.  
    '''
    declare.is_classes("vocabs", vocabs, list)

    declare.is_probability("prob", prob)
    if type_name(prob) == "BytesProb":
        prob = prob.to_numpy()
    elif type_name(prob) == "IndexTable":
        prob = prob.read_record("prob").to_numpy()
    
    probDim = prob.dim
    if len(vocabs) == probDim:
        if blankID is None:
            blankID = probDim - 1
        declare.is_positive_int("blankID", blackID)
        declare.in_boundary("blankID", blackID, 0, probDim-1)
    elif len(vocabs) == probDim - 1:
        if blankID == None:
            blankID = probDim - 1
        else:
            assert blankID == probDim - 1, f"The dimensibality of probability is {probDim} but only have {len(vocabs)} words. In this case, blank ID must be {probDim-1} but got {blankID}"
    else:
        raise WrongDataFormat(f"The dimensibality of probability {probDim} does not match the numbers of words {len(vocabs)}.")

    results = Transcription(name="bestPathResult")
    for utt, pb in prob.items:
        declare.is_classes("prob", prob, np.ndarray)
        declare.is_classes("the rank of matrix shape", len(pb.shape), "expected rank", 2)      
        best_path = np.argmax(pb, 1)
        best_chars_collapsed = [ vocabs[ID] for ID, _ in groupby(best_path) if ID != blankID ]
        try:
            results[utt] = " ".join(best_chars_collapsed)
        except Exception as e:
            e.args = ( "<vocab> might has non-string items.\n" + e.args[0], )
            raise e
    return results

def ctc_prefix_beam_search(prob, vocabs, blankID=None, beam=5, cutoff=0.999, strick=1.0, lmFile=None, alpha=1.0, beta=0):
    '''
    Prefix beam search decoding algorithm. Lm score is supported.

    Args:
        <prob>: An exkaldi postprobability object. This probalility should be an output of Neural Network with CTC loss fucntion.
                We expect the probability didn't pass any activation function, or it may generate wrong results.
        <vocabs>: a list of vocabulary.
        <blankID>: specify the ID of blank symbol. If None, use the last dimentionality of <prob>.
        <beam>: the beam size.
        <cutoff>: the sum threshold to cut off dimensions whose probability is extremely small.  
        <strick>: When the decoding results of two adjacent frames are the same, the probability of latter will be reduced.
        <lmFile>: If not None, add language model score to beam.
        <alpha>: the weight of LM score.
        <beta>: the length normaoliztion weight of LM score.
    Return:
        An exkaldi Transcription object of decoding results.  
    '''
    declare.is_classes("vocabs", vocabs, [tuple,list])

    declare.is_probability("prob", prob)
    if type_name(prob) == "BytesProb":
        prob = prob.to_numpy()
    elif type_name(prob) == "IndexTable":
        prob = prob.read_record("prob").to_numpy() 

    if lmFile is not None:
        declare.is_file("lmFile", lmFile)
    else:
        lmFile = "none"

    probDim = prob.dims
    if len(vocabs) == probDim:
        if blankID is None:
            blankID = probDim - 1
        declare.is_positive_int("blankID", blackID)
        declare.in_boundary("blankID", blackID, 0, probDim-1)

    elif len(vocabs) == probDim - 1:
        if blankID == None:
            blankID = probDim - 1
        else:
            assert blankID == probDim - 1, f"The dimensibality of probability is {probDim} but only have {len(vocabs)} words. In this case, blank ID must be {probDim-1} but got {blankID}"
    else:
        raise WrongDataFormat(f"The dimensibality of probability {probDim} does not match the numbers of words {len(vocabs)}.")

    for ID, word in enumerate(vocabs):
        if len(word.strip()) == 0:
            raise WrongDataFormat(f"Found a vocab {word} unavaliable.")

    num_classes = len(vocabs)
    vocabs = " ".join(vocabs)

    sources = [vocabs.encode(),]
    uttTemp = []
    for utt, pb in prob.items:
        declare.is_classes("prob", prob, np.ndarray)
        declare.is_classes("the rank of matrix shape", len(pb.shape), "expected rank", 2)
        pb = softmax(pb, axis=1)
        sources.append( f" {pb.shape[0]} ".encode() + pb.astype("float32").tobytes() )

    sources = b"".join(sources)

    cmd = os.path.join(sys.prefix,"exkaldisrc","tools","prefix_beam_search_decode")
    cmd += " --num_files {}".format(prob.lens[0])
    cmd += " --num_classes {}".format(num_classes)
    cmd += " --blank_id {}".format(blankID)
    cmd += " --lm_model {}".format(lmFile)
    cmd += " --beam_size {}".format(beam)
    cmd += " --cutoff_prob {}".format(cutoff)
    cmd += " --alpha {}".format(alpha)
    cmd += " --beta {}".format(beta)

    out, err, _ = run_shell_command(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, inputs=sources)

    if len(out) == 0:
        raise Exception("Failed to beam search decode.",err.decode())
    else:
        results = Transcription(name="beamSearchResults")
        out = out.decode().strip().split("file")
        results = []
        for index, re in enumerate(out[1:]):
            re = re.strip().split("\n")
            if len(re) <= 1:
                results.append(["",])
            else:
                results[uttTemp[index]] = " ".join(re[1].strip().split()[1:])

        return results


            
            

