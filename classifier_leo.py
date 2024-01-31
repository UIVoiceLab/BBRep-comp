import os 
import pandas as pd
import numpy as np
from sklearn import svm
import pickle

allfiles = os.listdir('/Volumes/rdss_ekutlu/VOICELab/Stimuli/accentedstim/stim (from WildCat SPEECHBOX)/6 - ampnormed/')
fnames = [i for i in allfiles if i.endswith('.wav')]
fnames = pd.DataFrame([x.split('.') for x in fnames])
details = pd.DataFrame([x.split('_s') for x in fnames[0]])
talkers = details.drop_duplicates(subset=0, keep="first")
accents = pd.DataFrame([x.split('_M_') for x in talkers[0]])

filepath = os.path.join('/', 'Volumes', 'rdss_ekutlu', 'VOICELab', 'Stimuli', 'accentedstim', 'stim (from WildCat SPEECHBOX)', 'split_sentences')
with open(filepath + '/' + 'new_acc_mfccs.pkl', 'rb') as f:
    data = pickle.load(f)

#%% 

koData = {key:value for (key,value) in data.items() if 'KO' in key}
enData = {key:value for (key,value) in data.items() if 'EN' in key}

longest = max([value.shape[1] for (key,value) in (koData | enData).items()])

koData, enData = list(koData.items()), list(enData.items())
koLabels, enLabels = [x[0] for x in koData], [x[0] for x in enData]
koData, enData = [x[1] for x in koData], [x[1] for x in enData]

for trial in range(len(koData)):
    if koData[trial].shape[1] < longest:
        padding = np.zeros((koData[trial].shape[0],longest-koData[trial].shape[1]))
        koData[trial] = np.concatenate((koData[trial],padding),1)

for trial in range(len(enData)):
    if enData[trial].shape[1] < longest:
        padding = np.zeros((enData[trial].shape[0],longest-enData[trial].shape[1]))
        enData[trial] = np.concatenate((enData[trial],padding),1)

#%% trained on EN, then tested on KO
traindata = enData
trainlabel = enLabels
trainlabel = [x.split('_s')[1] for x in trainlabel]
for t in range(len(traindata)):
    traindata[t] = traindata[t].reshape((1,traindata[t].shape[0]*traindata[t].shape[1]))
traindata = np.vstack(traindata)
    
testtalker = [accents[1][x] for x in range(len(accents)) if 'KO' in accents[0][x]]
for i in testtalker:
    testdata = [koData[x] for x in range(len(koLabels)) if i in koLabels[x]]
    testlabel = [koLabels[x] for x in range(len(koLabels)) if i in koLabels[x]]
    testlabel = [x.split('_s')[1] for x in testlabel]
    for t in range(len(testdata)):
        testdata[t] = testdata[t].reshape((1,testdata[t].shape[0]*testdata[t].shape[1]))
    testdata = np.vstack(testdata)
    
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



#%% model 2: trained on EN AND KO, then tested on another KO

testtalker = [accents[1][x] for x in range(len(accents)) if 'KO' in accents[0][x]]
for i in testtalker:
    
    testdata = [koData[x] for x in range(len(koLabels)) if i in koLabels[x]]
    testlabel = [koLabels[x] for x in range(len(koLabels)) if i in koLabels[x]]
    testlabel = [x.split('_s')[1] for x in testlabel]
    for t in range(len(testdata)):
        testdata[t] = testdata[t].reshape((1,testdata[t].shape[0]*testdata[t].shape[1]))
    testdata = np.vstack(testdata)
    
    traindata = enData + [koData[x] for x in range(len(koLabels)) if i not in koLabels[x]]
    trainlabel = enLabels + [koLabels[x] for x in range(len(koLabels)) if i not in koLabels[x]]
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


#%% Samantha's model
testtalker = [accents[1][x] for x in range(len(accents)) if 'KO' in accents[0][x]]
for i in testtalker:
    
    
    
    
    
    
    
    
    #get test data
    testdata = [koData[x] for x in range(len(koLabels)) if i in koLabels[x]]
    
    
    
    testlabel = [koLabels[x] for x in range(len(koLabels)) if i in koLabels[x]]
    testlabel = [x.split('_s')[1] for x in testlabel]
    for t in range(len(testdata)):
        testdata[t] = testdata[t].reshape((1,testdata[t].shape[0]*testdata[t].shape[1]))
    testdata = np.vstack(testdata)
    
    
    
    
    
    
    
    
    
    # get train data
    traindata = [koData[x] for x in range(len(koLabels)) if i not in koLabels[x]]



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
















