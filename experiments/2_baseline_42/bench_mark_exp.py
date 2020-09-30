# example resource link: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
import sys
import os
fid = open("../../.envPath","r")
envPath = fid.readline()
envPath = envPath[:-1]
fid.close
del fid
sys.path.append(os.path.abspath(envPath+"/src/io"))

from load_Cul10_Semi import *
import torch
import numpy as np
import h5py
from datetime import datetime
from torchvision import transforms, datasets



'''
PARAMETER SETTING
'''
print("parameter setting...")
paraDict = {
        ### network parameters
        "nbBatch": 256,
        "nbEpoch": 100,
        "learningRate": 1e-4,

        ### data loading parameters
        # "trainData": "lcz42", # training data could be the training data of LCZ42 data, or data of one of the cultural-10 city
        "trainData": "munich",

        # "testData": "cul10",  # testing data could be all the data of the cultural-10 cities, or one of them.
        "testData": "moscow",

        "normalization":"cms", # "cms": channel-wise mean-std normalization
        # "normalization":"pms", # "pms": patch-wise mean-std normalization
        "datFlag":2, # data selection: sentinel-1, sentinel-2, or both

        ### model name
        "modelName":'Sen2LCZ',#'LeNet', # model name
        }

cudaNow = torch.device('cuda:0')
nbBatch = paraDict["nbBatch"]
nbEpoch = paraDict["nbEpoch"]
learnRate = paraDict["learningRate"]
datFlag = paraDict["datFlag"]
modelName = paraDict["modelName"]


'''
initial folder saving outputs
'''
outcomeDir = initialOutputFolder(paraDict)
print("Experiments outcomes are saving in the directory: "+outcomeDir)
# record parameters
recordExpParameters(outcomeDir,paraDict)



'''
STEP ONE: data loading
'''
trainDataSet,testDataSet = lczIterDataSet(envPath,paraDict["trainData"],paraDict["testData"],datFlag,paraDict["normalization"])
trainDataLoader = torch.utils.data.DataLoader(trainDataSet, batch_size=nbBatch, shuffle=True)
testDataLoader = torch.utils.data.DataLoader(testDataSet, batch_size=512)


'''
STEP TWO: initial a resnet model
'''
sys.path.append(os.path.abspath(envPath+"/src/model"))
import resnetModel
# model = resnetModel.resnet18(pretrained=False, inChannel=trainDataSet.nbChannel()).to(cudaNow)
# predModel = resnetModel.resnet18(pretrained=False, inChannel=trainDataSet.nbChannel()).to(cudaNow)

# model = resnetModel.LeNet(inChannel=trainDataSet.nbChannel(), nbClass = trainDataSet.label.shape[1]).to(cudaNow)
# predModel = resnetModel.LeNet(inChannel=trainDataSet.nbChannel(), nbClass = trainDataSet.label.shape[1]).to(cudaNow)

model = resnetModel.Sen2LCZ(in_Channel=10, nb_class=17, nb_kernel=16, depth=17, bn_flag=1).to(cudaNow)
predModel = resnetModel.Sen2LCZ(in_Channel=10, nb_class=17, nb_kernel=16, depth=17, bn_flag=1).to(cudaNow)

'''
STEP THREE: Define a loss function and optimizer
'''
import torch.optim as optim
import torch.nn as nn
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learnRate)


'''
STEP FOUR: Train the network
'''

# import modelOperation
import modelOperDataLoader
print('Start training ...')
model,traLoss,traArry,valLoss,valArry,valAver = modelOperDataLoader.train(model,cudaNow,optimizer,trainDataLoader,criterion,nbEpoch,testDataLoader)

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

torch.save(model.state_dict(), os.path.join(outcomeDir,'model'))
fid = h5py.File(os.path.join(outcomeDir,'training_history.h5'),'w')
fid.create_dataset('traLoss',data=traLoss)
fid.create_dataset('traArry',data=traArry)
fid.create_dataset('valLoss',data=valLoss)
fid.create_dataset('valArry',data=valArry)
fid.create_dataset('valAver',data=valAver)
fid.close()

'''
STEP SEVEN: Predict with the trained model
'''
# outcomeDir = os.path.join('/data/Projects/TF/experiments/0_benchMark/channel_normalization_outcome','resnet18_benchMark_tr_moscow_te_munich_outcome_2019-12-17_13-36-01')
# outcomeDir = os.path.join('/data/Projects/TF/experiments/0_benchMark/patch_normalization_outcome','')
modelPath = os.path.join(outcomeDir,'model')
predModel.load_state_dict(torch.load(modelPath,map_location=cudaNow))
confusion_matrix,oa,aa,ka,pa,ua = modelOperDataLoader.confusionMatrix(predModel,cudaNow,testDataLoader,criterion)

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




