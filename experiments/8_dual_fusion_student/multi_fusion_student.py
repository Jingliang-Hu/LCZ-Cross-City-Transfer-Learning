# example resource link: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import sys
import os

fid = open("../../.envPath","r")
envPath = fid.readline()
envPath = envPath[:-1]
fid.close
del fid
print("EnvPath:"+envPath)

sys.path.append(os.path.abspath(envPath+"/src/io"))

from load_Cul10_Semi import *
import torch
import numpy as np
import h5py
from torchvision import transforms, datasets


'''
PARAMETER SETTING
'''
print("parameter setting...")
paraDict = {
        ### network parameters
        "nbBatch": 256,
        # "nbEpoch": 728,
        "nbEpoch": 1,

        "learningRate": 1e-4,

        ### data loading parameters
        # "trainData": "lcz42", # training data could be the training data of LCZ42 data, or data of one of the cultural-10 city
        "trainData": "munich",
        # "testData": "cul10",  # testing data could be all the data of the cultural-10 cities, or one of them.
        "testData": "moscow",

        "datFlag":0, # data selection: sentinel-1, sentinel-2, or both
        "normalization_s2":"cms", # "ms": mean-std normalization, patch-wise
        "normalization_s1":"no", # "ms": mean-std normalization, patch-wise
        
        ### model name
        "modelName":'LeNet', # model name
        "nbStreams":3,
        }

cudaNow = torch.device('cuda:1')

nbBatch = paraDict["nbBatch"]
nbEpoch = paraDict["nbEpoch"]
learnRate = paraDict["learningRate"]
datFlag = paraDict["datFlag"]
modelName = paraDict["modelName"]
nbStreams = paraDict["nbStreams"]
normalization = [paraDict["normalization_s1"],paraDict["normalization_s2"]]


'''
initial folder saving outputs
'''
outcomeDir = initialOutputFolder(paraDict)
print("Experiments outcomes are saving in the directory: "+outcomeDir)
## record parameters
recordExpParameters(outcomeDir,paraDict)


'''
STEP ONE: data loading
'''

trainDataSet,testDataSet = lczIterDataSet(envPath,paraDict["trainData"],paraDict["testData"],datFlag,normalization,transform=transforms.Compose([ToTensor()]),shaffle=1)

'''
STEP TWO: initial a resnet model
'''
sys.path.append(os.path.abspath(envPath+"/src/model"))
import resnetModel
model = []

if modelName=='LeNet':
    for i in range(0,nbStreams):
        model.append(resnetModel.LeNet_feature_fusion(inChannel_1=trainDataSet.nbChannel()[0], inChannel_2=trainDataSet.nbChannel()[1], nbClass = trainDataSet.label.shape[1]).to(cudaNow))


'''
STEP THREE: Define a loss function and optimizer
'''
import torch.optim as optim
import torch.nn as nn
classification_loss = nn.CrossEntropyLoss()
consistency_loss = nn.MSELoss()
optimizers = []
for i in range(0,nbStreams):
    optimizers.append(optim.Adam(model[i].parameters(), lr=learnRate))


'''
STEP FOUR: Train the network
'''
import modelOperDataLoader
model, tra_loss, tra_arry, val_loss, val_arry, val_aver, tra_consis_loss = modelOperDataLoader.train_multi_fusion(model, trainDataSet, testDataSet, optimizers, cudaNow, classification_loss, consistency_loss, nbEpoch, nbBatch)

'''
STEP FIVE: Test the network
'''
# pred,_,acc = modelOperation.test(resnet,cudaNow,dat_te,lab_te,criterion = criterion,numBatch=nbBatch)
# print('Accuracy of the network on the %d test samples: %d %%' % ( dat_te.shape[0], acc))


'''
STEP SIX: Save the trained model
'''
'''
codes for saving and loading models:
	Save:
		torch.save(model.state_dict(), PATH)
	Load:
		model = TheModelClass(*args, **kwargs)
		model.load_state_dict(torch.load(PATH))
		model.eval()
'''
for i in range(0,nbStreams):
    model_name = 'model_stream_' +str(i+1)
    torch.save(model[i].state_dict(), os.path.join(outcomeDir,model_name))

# saveing training history
fid = h5py.File(os.path.join(outcomeDir,'training_history.h5'),'w')
fid.create_dataset('tra_Loss',data=tra_loss)
fid.create_dataset('tra_Arry',data=tra_arry)
fid.create_dataset('val_Loss',data=val_loss)
fid.create_dataset('val_Arry',data=val_arry)
fid.create_dataset('val_Aver',data=val_aver)
fid.create_dataset('consistentLoss',data=tra_consis_loss)

fid.close()



'''
STEP SEVEN: Predict with the model
'''

import modelOperDataLoader

target_data_loader = torch.utils.data.DataLoader(testDataSet, batch_size=nbBatch, shuffle=False)

confusion_matrix,oa,aa,ka,pa,ua = modelOperDataLoader.prediction_multi_fusion(model, cudaNow, target_data_loader, classification_loss)

# save accuracy
fid = h5py.File(os.path.join(outcomeDir,'test_accuracy.h5'),'w')
fid.create_dataset('confusion_matrix',data=confusion_matrix)
fid.create_dataset('oa',data=oa)
fid.create_dataset('aa',data=aa)
fid.create_dataset('ka',data=ka)
fid.create_dataset('pa',data=pa)
fid.create_dataset('ua',data=ua)
fid.close()
# plot confusion matrix
cm_disp_obj = modelOperDataLoader.ConfusionMatrixDisplay(confusion_matrix,np.linspace(1,confusion_matrix.shape[0],confusion_matrix.shape[0]))
cm_disp = cm_disp_obj.plot()
cm_disp.figure_.savefig(os.path.join(outcomeDir,'confusion_matrix.png'))




'''
plot outcomes
'''
pth = modelOperDataLoader.plotTrainHistory(outcomeDir,paraDict["modelName"])
pth.plotHistory()



