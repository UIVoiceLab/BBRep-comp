# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 15:40:25 2023

@author: slchiu
"""
# https://pytorch.org/audio/main/tutorials/asr_inference_with_ctc_decoder_tutorial.html

import torch
import torchaudio
import os
import re
import numpy as np

torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)

#%% Greedy Decoder
class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])

decoder = GreedyCTCDecoder(labels=bundle.get_labels())

#%% Beam Search Decoder
# pip install flashlight-text
# pip install https://github.com/kpu/kenlm/archive/master.zip
#? from torchaudio.models.decoder import CTCDecoderLM, CTCDecoderLMState

# pretrained files
from torchaudio.models.decoder import download_pretrained_files
import kenlm
files = download_pretrained_files("librispeech-4-gram")

    
from torchaudio.models.decoder import ctc_decoder
beam_search_decoder = ctc_decoder(lexicon=files.lexicon,
                                  tokens=bundle.get_labels(),
                                  lm=files.lm,
                                  nbest=3,
                                  beam_size=1500,
                                  lm_weight=3.23,
                                  word_score=-0.26)

#%% correct keys
correct = { 's1': 'HE POINTED AT THE CENTS',
            's2': 'DAD TALKED ABOUT THE BOMB',
            's3': 'MOM POINTED AT HIS FATHER',
            's5': 'WE POINTED AT THE BIRD',
            's8': 'WE READ ABOUT THE FAMILY',
            's9': 'SHE POINTED AT HER HEAD',
            's11': 'DAD READ ABOUT THE SKY',
            's14': 'WE TALKED ABOUT THE WATER',
            's15': 'SHE LOOKED AT THE CLOCK',
            's16': 'HE POINTED AT THE ANIMALS',
            's18': 'DAD POINTED AT THE GRASS',
            's21': 'DAD TALKED ABOUT THE SHEETS',
            's22': 'MOM THINKS THAT IT IS YELLOW',
            's27': 'HE LOOKED AT HER WRIST',
            's28': 'WE READ ABOUT THE COACH',
            's31': 'THIS IS HER FAVORITE WEEK',
            's35': 'MOM LOOKED AT THE JUICE',
            's37': 'SHE TALKED ABOUT THEIR NECKS',
            's39': 'SHE TALKED ABOUT THE LEAVES',
            's40': 'WE LOOKED AT THE STORY',
            's41': 'THIS IS HER FAVORITE SPORT',
            's44': 'MOM LOOKED AT HER FEET',
            's46': 'HE READ ABOUT THE TREES',
            's48': 'MOM TALKED ABOUT THE PIE',
            's50': 'HE LOOKED AT THE SLEEVES',
            's51': 'THIS IS HER FAVORITE TIME',
            's52': 'THERE ARE MANY DAYS',
            's55': 'HE TALKED ABOUT THE DINNER',
            's56': 'MOM POINTED AT THE COFFEE',
            's58': 'SHE THINKS THAT IT IS FAST'}

#%% levenshtein distance
def word_error(correct, guess): 
    matrix = np.zeros((len(guess)+1, len(correct)+1))
    for i in range(1, len(guess)+1):
        matrix[i,0] = i 
    for j in range(1, len(correct)+1):
        matrix[0,j] = j  
    for j in range(1,len(correct)+1):
        for i in range(1,len(guess)+1):
            (isdiff := 0) if correct[j-1] == guess[i-1] else (isdiff := 1)
            matrix[i,j] = min(matrix[i-1, j] + 1,           # Deletion
                              matrix[i, j-1] + 1,           # Insertion
                              matrix[i-1,j-1] + isdiff)     # Substitution        
    return matrix[len(guess), len(correct)] / max(len(correct), len(guess))

#%% load audio

# fpath = '\\\\iowa.uiowa.edu\\shared\\ResearchData\\rdss_ekutlu\\VOICELab\\Stimuli\\accentedstim\\stim (from WildCat SPEECHBOX)\\6 - ampnormed\\'
fpath = 'Volumes/rdss_ekutlu/VOICELab/Stimuli/accentedstim/stim (from WildCat SPEECHBOX)/6 - ampnormed/'

audfiles = [x for x in os.listdir(fpath) if x.endswith(".wav")]
greedy_results = []
greedy_scores = []
beam_results = []
beam_scores = []

for i in audfiles[::10]:    
    print(audfiles.index(i))
    waveform, sr = torchaudio.load(fpath + i)
    waveform = waveform.to(device)

    if sr != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, bundle.sample_rate)

    with torch.inference_mode():
        emission, _ = model(waveform)

    greedy_result = decoder(emission[0])
    greedy_transcript = re.sub("\\|", " ", greedy_result).rstrip()
    greedy_results.append(greedy_transcript)

    # calculates the levenshtein distance
    greedy_num = re.split('[_.]', i)[3]
    greedy_score = word_error(correct[greedy_num], greedy_transcript)
    greedy_scores.append(greedy_score)
    
    beam_result = beam_search_decoder(emission)
    beam_transcript = " ".join(beam_result[0][0].words).strip().upper()
    beam_transcript = re.sub("\\|", " ", beam_transcript).rstrip()
    beam_results.append(beam_transcript)

    beam_num = re.split('[_.]', i)[3]
    beam_score = word_error(correct[beam_num], beam_transcript)
    beam_scores.append(beam_score)

#%% print results
with open("/Volumes/rdss_ekutlu/VOICELab/Experiments/BBRep-comp/TEST_results_BBRep.txt","w") as f:
    for (audfile,gr,gs,br,bs) in zip(audfiles,greedy_results,greedy_scores,beam_results,beam_scores):
        f.write("{0},{1},{2},{3},{4}\n".format(audfile,gr,gs,br,bs))

