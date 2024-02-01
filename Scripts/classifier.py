# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 12:09:45 2023

@author: saman
"""

#import relevant libraries 
import os 
import pandas as pd
import numpy as np
import re
from sklearn import svm
import matplotlib.pyplot as plt 
import pickle
# accented speech: 1) predict what sentence was heard; 2) predict what accent was heard 

#%% let's check our input; what do we got?? 

allfiles = os.listdir('\\\\iowa.uiowa.edu\\shared\\ResearchData\\rdss_ekutlu\\VOICELab\\Stimuli\\accentedstim\\stim (from WildCat SPEECHBOX)\\6 - ampnormed')
#get only audio files 
fnames = [i for i in allfiles if i.endswith('.wav')]
#get the .wav out of the strings 
fnames = pd.DataFrame([x.split('.') for x in fnames])
#split by _ to get accent, gender, talker#, and stim 
details = pd.DataFrame([x.split('_s') for x in fnames[0]])

talkers = details.drop_duplicates(subset=0, keep="first")
accents = pd.DataFrame([x.split('_M_') for x in talkers[0]])

numtalkersperaccent = accents[0].value_counts()
# we don't have many... talkers. huh

print(numtalkersperaccent)
#accent num
# KO    8
# CH    4
# SP    2
# TU    2
# IN    1
# IT    1
# JA    1
# RU    1
# TH    1
# IR    1

# we could probably do KO, CH, SP, TU x 2 for accent 
# lets unpickle the data!! and see if.... if we can figure out which spect is which X_X

with open('\\\\iowa.uiowa.edu\\shared\\ResearchData\\rdss_ekutlu\\VOICELab\\Experiments\\BBRep-comp\\acc_mfccs.pkl', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    data = pickle.load(f)

#%% 1) predict what sentence was heard 


# we have the most for korean, let's start there! 
# train on 7 talkers, test on the last talker 
koData = {key:value for (key,value) in data.items() if 'KO' in key}

# i'm curious; what is the range of sizes that the mfccs are? 
sizelist = [value.shape[1] for (key,value) in koData.items()]
#plt.hist(sizelist) #this is good to keep around for later; we'll see if this is necessary! 

#in order for classifier to work, i need to get all the files to be the same size; so get the max size and pad all the others with 0s

koData = list(koData.items())
koLabels = [x[0] for x in koData]
koData = [x[1] for x in koData]
max(sizelist)

for trial in range(len(koData)):
    if koData[trial].shape[1] < max(sizelist):
        padding = np.zeros((koData[trial].shape[0],max(sizelist)-koData[trial].shape[1]))
        koData[trial] = np.concatenate((koData[trial],padding),1)
# train on one talker, test on another <-- this one might not be possible given that we have 1 token of each sentence; if we have 7 tokens of each sentence, we're in better shape! 

# at this point, i dont know how WER works, so i'm gonna do stim number and just call it for now 

testtalker = [accents[1][x] for x in range(len(accents)) if 'KO' in accents[0][x]]
for i in testtalker:
    #get test data
    testdata = [koData[x] for x in range(len(koLabels)) if i in koLabels[x]]
    # get labels for test data
    testlabel = [koLabels[x] for x in range(len(koLabels)) if i in koLabels[x]]
    testlabel = [x.split('_s')[1] for x in testlabel]
    
    for t in range(len(testdata)):
        testdata[t] = testdata[t].reshape((1,testdata[t].shape[0]*testdata[t].shape[1]))
    
    testdata = np.vstack(testdata)
    # get train data
    traindata = [koData[x] for x in range(len(koLabels)) if i not in koLabels[x]]
    # we can start with 1) random forests, 2) gradient boosting, or 3) KNNs  
    trainlabel = [koLabels[x] for x in range(len(koLabels)) if i not in koLabels[x]]
    trainlabel = [x.split('_s')[1] for x in trainlabel]
    
    for t in range(len(traindata)):
        traindata[t] = traindata[t].reshape((1,traindata[t].shape[0]*traindata[t].shape[1]))
    
    traindata = np.vstack(traindata)
    
    # results: testsub, Cparam, accuracy
    
    allacc = [];
    # okay now set up the classifier...? 
    for Cparam in np.power(2,np.linspace(-2,18,16)):
        classifier = svm.SVC(C = Cparam)
        
        classifier.fit(traindata,trainlabel)
        
        pred = classifier.predict(testdata)
        acc = sum(testlabel == pred)/len(pred)
        
        allacc = np.append(allacc,acc)
    if 'results1' in locals(): 
        results1 = np.append(results1,[i]*len(np.power(2,np.linspace(-2,18,16))))
        results2 = np.append(results2,np.power(2,np.linspace(-2,18,16)))
        results3 = np.append(results3,allacc)
    else:
        results1 = [i]*len(np.power(2,np.linspace(-2,18,16)))
        results2 = np.power(2,np.linspace(-2,18,16))
        results3 = allacc
        
results = pd.DataFrame(list(zip(results1, results2, results3)),columns =['subject', 'C','accuracy'])
#%% 1B) I'm curious as to how random forests would fare! 




#%% 2) predict what sentence was heard w/ accented speech 
#input: english accent: train on 1 talker, test on another accent; then we do train on 5 talkers, test on another accent 

#%% 3) predict what accent was heard
# input: multiple accents; test on known talkers (diff sentences); then try generalization test on novel talkers 

### notes from zoey
# interpretability is out the window for CS people
# not high accuracy as of now
# let's try neural classifier! neural networks? zoey will send us a package! 
# try other off the shelf! 
# different kind of baseline/metrics; scikitlearn 

# can we get lexical information in there first? 
# fine tuning? take model that's been trained, then fine tune! 
# questions others might have: approximating human data from comps 
# expand out to other 



