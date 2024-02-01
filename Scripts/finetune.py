import torch
import pandas as pd
#pip uninstall lightning-flash
#pip install --upgrade --pre lightning-flash
#conda install lightning-flash[audio]
import flash
import numpy as np
import random
from flash.audio import SpeechRecognition, SpeechRecognitionData
#from flash.core.data.utils import download_data

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

def flatten_list(nested_list):
    flat_list = []
    for sublist in nested_list:
        for item in sublist:
            flat_list.append(item)
    return flat_list


# 1. Create the DataModule

k_acc = pd.read_csv('\\\\iowa.uiowa.edu\\shared\\ResearchData\\rdss_ekutlu\\VOICELab\\Experiments\\BBRep-comp\\koreanaccents.csv')
all_acc = pd.read_csv('\\\\iowa.uiowa.edu\\shared\\ResearchData\\rdss_ekutlu\\VOICELab\\Experiments\\BBRep-comp\\allaccents.csv')
allaccents = all_acc[all_acc['Accent'] != 'KO']['Accent'].unique().tolist()
test_acc = k_acc['VoiceNum'].unique().tolist()

notraining_all = []
onlyko_all = []
multacc_all = []

for tacc in test_acc:

    testdata = k_acc[k_acc['VoiceNum']==tacc]['File'].to_list()
    testlabels = k_acc[k_acc['VoiceNum']==tacc]['Text'].to_list()

    curracc = random.sample(allaccents,5)
    alltraindata = []
    alltrainlabels = []
    for i in curracc:
        #find all numbers associated with the accent, 
        currvoices = all_acc[all_acc['Accent'] == i]['VoiceNum'].unique().tolist()
        #randomly pick the accent 
        currnum = random.sample(currvoices,1)[0]
        alltraindata.append(all_acc[(all_acc['Accent']==i) & (all_acc['VoiceNum']== currnum)]['File'].tolist())
        alltrainlabels.append(all_acc[(all_acc['Accent']==i) & (all_acc['VoiceNum']== currnum)]['Text'].tolist())
    alltraindata = flatten_list(alltraindata)
    alltrainlabels = flatten_list(alltrainlabels)

    test_acc = k_acc['VoiceNum'].unique().tolist()
    test_acc.remove(tacc)
    currko = random.sample(test_acc,5)
    onlykotraindata = k_acc[k_acc['VoiceNum'].isin(currko)]['File'].to_list()
    onlykotrainlabels = k_acc[k_acc['VoiceNum'].isin(currko)]['Text'].to_list()

    kodatamodule = SpeechRecognitionData.from_files(
        train_files=onlykotraindata,
        train_targets=onlykotrainlabels,
        predict_files=testdata,
        batch_size=4,
    )

    # 2. Build the task
    model = SpeechRecognition(backbone="facebook/wav2vec2-base-960h")

    # 3. Create the trainer and finetune the model
    trainer = flash.Trainer(max_epochs=1, gpus=torch.cuda.device_count())
    control = trainer.predict(model, datamodule=kodatamodule)
    control = flatten_list(control)

    trainer.finetune(model, datamodule=kodatamodule, strategy="freeze")

    # 4. Predict on audio files!
    kopred = trainer.predict(model, datamodule=kodatamodule)
    print(kopred)
    kopred = flatten_list(kopred)

    control_scores = []
    for i in range(len(control)):
        score = word_error(testlabels[i], control[i])
        control_scores.append(score)

    onlyko_scores = []
    for i in range(len(kopred)):
        score = word_error(testlabels[i], kopred[i])
        onlyko_scores.append(score)

    #### MULTIACCENT MODEL  

    model = SpeechRecognition(backbone="facebook/wav2vec2-base-960h")
    alldatamodule = SpeechRecognitionData.from_files(
        train_files=alltraindata,
        train_targets=alltrainlabels,
        predict_files=testdata,
        batch_size=4,
    )

    # 2. Build the task
    model = SpeechRecognition(backbone="facebook/wav2vec2-base-960h")

    # 3. Create the trainer and finetune the model
    trainer = flash.Trainer(max_epochs=1, gpus=torch.cuda.device_count())

    trainer.finetune(model, datamodule=alldatamodule, strategy="freeze")

    multpred = trainer.predict(model, datamodule=alldatamodule)
    multpred = flatten_list(multpred)

    mult_scores = []
    for i in range(len(multpred)):
        score = word_error(testlabels[i], multpred[i])
        mult_scores.append(score)


    notraining = sum(control_scores)/len(control_scores)
    oneacc = sum(onlyko_scores)/len(onlyko_scores)
    multacc = sum(mult_scores)/len(mult_scores)

    notraining_all.append(notraining)
    onlyko_all.append(oneacc)
    multacc_all.append(multacc)

    del trainer





# 5. Save the model!
#trainer.save_checkpoint("speech_recognition_model.pt")


#https://lightning-flash.readthedocs.io/en/stable/api/generated/flash.audio.speech_recognition.data.SpeechRecognitionData.html#flash.audio.speech_recognition.data.SpeechRecognitionData
#https://lightning-flash.readthedocs.io/en/stable/api/audio.html
#https://lightning-flash.readthedocs.io/en/stable/api/generated/flash.audio.speech_recognition.model.SpeechRecognition.html#flash.audio.speech_recognition.model.SpeechRecognition
#https://huggingface.co/models?pipeline_tag=automatic-speech-recognition
#https://lightning-flash.readthedocs.io/en/stable/api/flash.html
#https://lightning-flash.readthedocs.io/en/stable/api/generated/flash.core.trainer.Trainer.html#flash.core.trainer.Trainer
