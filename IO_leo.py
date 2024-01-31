# model 1 trained on standard american english, then tested on korean
# model 2 trained on SAE AND korean, then tested on another korean

import librosa
import os 
import pickle

path = os.path.join('/', 'Volumes', 'rdss_ekutlu', 'VOICELab', 'Stimuli', 'accentedstim', 'stim (from WildCat SPEECHBOX)', 'split_sentences')
fnames = [x for x in os.listdir(path) if '.wav' in x]
os.chdir(path)

for f in fnames:
    samples, sample_rate = librosa.load(f, sr=None)
    
    mfccs = librosa.feature.mfcc(y=samples, sr=sample_rate, n_mfcc=13)
    
    if 'data' in globals():
        data[f] = mfccs
    else:    
        data = {f:mfccs}

with open('new_acc_mfccs.pkl', 'wb') as f:
    pickle.dump(data, f)




    # # Renaming Files en mass 
    # import os
    # path = os.path.join('iowa.uiowa.edu','shared','ResearchData', 'rdss_ekutlu', 'VOICELab', 'Stimuli', 'accentedstim', 'stim (from WildCat SPEECHBOX)', '1 - original', 'sentences', 'split_sentences', 'SC_S_EN_14_EN')
    # files = os.listdir(path)
    # for i in files:
    #     old = os.path.join(path, i)
    #     fixed = 'EN_M_14_s'
    #     new = os.path.join(path, fixed + str(files.index(i)) + '.wav')
    #     os.rename(old, new)