# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 10:30:30 2023

@author: slchiu
"""

# GOAL FOR THIS SCRIPT: LOAD ONE FILE IN, LOAD IN THE WAV2VEC AND SEE WHAT IT OUTPUTS 
#%% load packages 
import torch
import torchaudio

# also need to install audio backend before torchaudio.load() works
# FOR WINDOWS: pip install soundfile
# use this for info on backend https://pytorch.org/audio/0.7.0/backend.html
# BUT it's not a package to import, it's just something that torchaudio relies on BUT doesn't automatically download for you because it doesn't know what system you're using???

print(torch.__version__)
print(torchaudio.__version__)

import io
import os
import tarfile
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from IPython.display import Audio

torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% create functions 

def _hide_seek(obj):
    class _wrapper:
        def __init__(self, obj):
            self.obj = obj

        def read(self, n):
            return self.obj.read(n)

    return _wrapper(obj)

def plot_waveform(waveform, sample_rate):
    waveform = waveform.numpy() # turns waveform into numpy array

    num_channels, num_frames = waveform.shape # get the number of channels and frames of the waveform
    time_axis = torch.arange(0, num_frames) / sample_rate # get time axis, which is frames/SR

    figure, axes = plt.subplots(num_channels, 1) # create initial plot with number of channels 
    if num_channels == 1:
        axes = [axes] # subplot is the only plot, so set it to itself
    for c in range(num_channels): # for each channel
        axes[c].plot(time_axis, waveform[c], linewidth=1) # plot time x waveform 
        axes[c].grid(True) # unsure what this does... 
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}") # label channels by number 
    figure.suptitle("waveform") 

# this is the model??? there are different kinds of decoders, maybe need to look into this! 
class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])

#%% load wav2vec2

# feature extraction
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H

print("Sample Rate:", bundle.sample_rate)
print("Labels:", bundle.get_labels())


model = bundle.get_model().to(device)
print(model.__class__)

# speech algorithm? 
decoder = GreedyCTCDecoder(labels=bundle.get_labels())
#%% load audio
# get meta data from file 
# torchaudio.info()

# load audio file
# torchaudio.load()
fpath = '\\\\iowa.uiowa.edu\\shared\\ResearchData\\rdss_ekutlu\\VOICELab\\Stimuli\\accentedstim\\stim (from WildCat SPEECHBOX)\\6 - ampnormed\\'
#fpathnoise = '\\\\iowa.uiowa.edu\\shared\\ResearchData\\rdss_ekutlu\\VOICELab\\Stimuli\\accentedstim\\stim (from WildCat SPEECHBOX)\\10 - prepped with babble\\'

audfiles = [x for x in os.listdir(fpath) if x.endswith(".wav")]
results = []

for i in audfiles:
    waveform, sr = torchaudio.load(fpath + i)
    waveform = waveform.to(device)

    if sr != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, bundle.sample_rate)

    # extract and classify features? not sure what the classification part does exactly... 
    with torch.inference_mode():
        emission, _ = model(waveform)
    # decode and print output
    transcript = decoder(emission[0])
    results.append(transcript)

# calculate a score with leo's code! 

#%% print results
with open("results_BBRep.txt","w") as f:
    for (audfile,result) in zip(audfiles,results):
        f.write("{0},{1}\n".format(audfile,result))

#%% extract acoustic features 
#with torch.inference_mode():
#    features, _ = model.extract_features(waveform)

#%% NOTES FOR LATER: 

# Use wav2vec2_ASR_Base_100H, wav2vec2_ASR_BASE_960H, wa2vec2_ASR_LARGE_100H,wa2vec2_ASR_LARGE_960H, -- only trained on accents close to US english; first two are likely the quickest to run!
# finetune models: https://devblog.pytorchlightning.ai/fine-tuning-wav2vec-for-speech-recognition-with-lightning-flash-bf4b75cad99a
#^ this uses pytorch lightning, which uses pytorch as a base, but makes it much easier to read and use!! would be useful for us to start, then we can pop under the hood and build it out ourselves
# https://github.com/Lightning-Universe/lightning-flash/blob/master/examples/audio/speech_recognition.py
# ^ this the github for fine tuning wav2vec2

# https://pytorch.org/audio/stable/tutorials/speech_recognition_pipeline_tutorial.html#conclusion

# this model uses a greedy decoder, but there are other kinds! read here: https://medium.com/voice-tech-podcast/visualising-beam-search-and-other-decoding-algorithms-for-natural-language-generation-fbba7cba2c5b#:~:text=GREEDY%20DECODER,to%20the%20other%20decoding%20algorithms.