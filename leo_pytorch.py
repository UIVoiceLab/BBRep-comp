#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# https://pytorch.org/audio/stable/pipelines.html#module-torchaudio.pipelines
"""
Created on Thu Nov  2 12:34:40 2023

@author: leomoore
"""

import torch
import torchaudio
import os

torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

filepath = "\\\\iowa.uiowa.edu\\shared\\ResearchData\\rdss_ekutlu\\VOICELab\\Stimuli\\accentedstim\\stim (from WildCat SPEECHBOX)\\split_sentences\\"
files = [x for x in os.listdir(filepath) if '.wav' in x]

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)

results = list()


for file in files:
    waveform, sample_rate = torchaudio.load(filepath + file)
    waveform = waveform.to(device)
    
    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
    
    with torch.inference_mode():
        emission, _ = model(waveform)
    
    class GreedyCTCDecoder(torch.nn.Module):
        def __init__(self, labels, blank=0):
            super().__init__()
            self.labels = labels
            self.blank = blank
            
        def forward(self, emission: torch.Tensor) -> str:
            indices = torch.argmax(emission, dim=-1)
            indices = torch.unique_consecutive(indices, dim=-1)
            indices = [i for i in indices if i != self.blank]
            return "".join([self.labels[i] for i in indices])
    
    decoder = GreedyCTCDecoder(labels=bundle.get_labels())
    results.append(decoder(emission[0]))

with open('leo_results.txt', 'w') as f:
    for (file, result) in zip(files, results):
        f.write(file + ',' + result + '\n')








































































