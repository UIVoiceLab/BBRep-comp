# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 13:26:33 2023

@author: saman
"""
import librosa
#import matplotlib.pyplot as plt
import numpy as np
import os 
import pickle

#'\\\\iowa.uiowa.edu\\shared\\ResearchData\\rdss_ekutlu\\VOICELab\\Stimuli\\accentedstim'
#get list of files 
fnames = os.listdir('\\\\iowa.uiowa.edu\\shared\\ResearchData\\rdss_ekutlu\\VOICELab\\Stimuli\\accentedstim\\stim (from WildCat SPEECHBOX)\\6 - ampnormed')
os.chdir('\\\\iowa.uiowa.edu\\shared\\ResearchData\\rdss_ekutlu\\VOICELab\\Stimuli\\accentedstim\\stim (from WildCat SPEECHBOX)\\6 - ampnormed')
#filter out files you don't want right now
fnames = [ x for x in fnames if ".wav" in x ]
# for each file 

#%%
for f in fnames:
    # load the file 
    samples, sample_rate = librosa.load(f, sr=None)
    
    #sgram = librosa.stft(samples)
    # get mel spectrogram 
    #sgram_mag, _ = librosa.magphase(sgram)
    #mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate)
    #librosa.display.specshow(mel_scale_sgram, x_axis='s',y_axis='mel')
    # store this somewhere + the tag 
    # use a numpy array, got it!
    
    mfccs = librosa.feature.mfcc(y=samples, sr=sample_rate, n_mfcc=13)
    if 'data' in globals():
        data[f] = mfccs #mel_scale_sgram
    else:    
        data = {f:mfccs} #mel_scale_sgram}
    # rinse and repeat 
    
# now save it and the fnames to a datafile 

# switch directory



with open('acc_mfccs.pkl', 'wb') as f:
    pickle.dump(data, f)


    # from tag, get all the sentences 
    
    # then, let's build a model? 

