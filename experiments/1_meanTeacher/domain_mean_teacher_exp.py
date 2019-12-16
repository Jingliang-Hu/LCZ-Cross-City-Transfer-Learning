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

'''
PARAMETER SETTING
'''
print("parameter setting...")
print("parameter setting...")
paraDict = {
        ### network parameters
        "nbBatch": 100,
        "nbEpoch": 200,
        "learningRate": 1e-3,
        "maxEpochConsisLossWeight": "none",
        "confidenceModel":1,
        "confidenceThreshold":0.9,
        "alphaEMA": 0.99,
        "alphaMaxEpoch": 50,
 
        ### data loading parameters
        
        "trainData": "moscow", # training data could be the training data of LCZ42 data, or data of one of the cultural-10 city
        "testData": "munich",  # testing data could be all the data of the cultural-10 cities, or one of them.
        "normalization":"ms", # "ms": mean-std normalization, patch-wise
        "datFlag":2, # data selection: sentinel-1, sentinel-2, or both
        
        ### model name
        "modelName":'domain_mean_teacher', # model name
        }

cudaNow = torch.device('cuda:1')

nbBatch = paraDict["nbBatch"]
nbEpoch = paraDict["nbEpoch"]
learnRate = paraDict["learningRate"]
datFlag = paraDict["datFlag"]
modelName = paraDict["modelName"]
upperEpoch = paraDict["maxEpochConsisLossWeight"]
alphaMax = paraDict["alphaEMA"]
alphaMaxEpoch = paraDict["alphaMaxEpoch"]
confidentModel = paraDict["confidenceModel"]
confidentThres = paraDict["confidenceThreshold"]

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
print("data loading...")
x_train,y_train,x_test, y_test = lczLoader(envPath,paraDict["trainData"],paraDict["testData"],datFlag)
# Input image dimensions.
input_shape = x_train.shape[1:]
# Normalize data.
if paraDict["normalization"]=="ms":
    print("data normalization...")
    x_train = mean_Std_Normalization(x_train)
    x_test = mean_Std_Normalization(x_test)

# convert numpy to pytorch float tensor
x_train = torch.from_numpy(x_train).type('torch.FloatTensor')
y_train = torch.from_numpy(y_train).type('torch.LongTensor')
y_train = torch.max(y_train,1)[1]

x_test = torch.from_numpy(x_test).type('torch.FloatTensor')
y_test = torch.from_numpy(y_test).type('torch.LongTensor')
y_test = torch.max(y_test,1)[1]

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)



'''
STEP TWO: initial a resnet model
'''
sys.path.append(os.path.abspath(envPath+"/src/model"))
import resnetModel
student = resnetModel.resnet18(pretrained=False, inChannel=x_train.shape[1]).to(cudaNow)
teacher = resnetModel.resnet18(pretrained=False, inChannel=x_train.shape[1]).to(cudaNow)


'''
STEP THREE: Define a loss function and optimizer
'''
import torch.optim as optim
import torch.nn as nn
classification_loss = nn.CrossEntropyLoss()
consistency_loss = nn.MSELoss()
optimizer = optim.Adam(student.parameters(), lr=learnRate)


'''
STEP FOUR: Train the network
'''
import modelOperation

print('Start training ...')
if confidentModel:
    student,teacher,classificationLossTrainStudent,classificationAccuTrainStudent,classificationLossTestStudent,classificationAccuTestStudent,classificationLossTrainTeacher,classificationAccuTrainTeacher,classificationLossTestTeacher,classificationAccuTestTeacher,consistentLossTrain, consistentLossWeight, alpha = modelOperation.domainMeanTeacherConfidence_Train(student,teacher,cudaNow,x_train,y_train,optimizer,x_test,y_test,classification_loss,consistency_loss,nbBatch,nbEpoch,alphaMax,alphaMaxEpoch,confidentThres)

else:
    student,teacher,classificationLossTrainStudent,classificationAccuTrainStudent,classificationLossTestStudent,classificationAccuTestStudent,classificationLossTrainTeacher,classificationAccuTrainTeacher,classificationLossTestTeacher,classificationAccuTestTeacher,consistentLossTrain, consistentLossWeight, alpha = modelOperation.domainMeanTeacher_Train(student,teacher,cudaNow,x_train,y_train,optimizer,x_test,y_test,classification_loss,consistency_loss,nbBatch,nbEpoch,alphaMax)

#student,teacher,traSLoss,traSArry,valSLoss,valSArry,traTLoss,traTArry,valTLoss,valTArry = modelOperation.meanTeacher_Train(student,teacher,cudaNow,x_train,y_train,optimizer,x_test,y_test,classification_loss,consistency_loss,nbBatch,nbEpoch,alpha,upperEpoch)


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

torch.save(student.state_dict(), os.path.join(outcomeDir,'student_model'))
torch.save(teacher.state_dict(), os.path.join(outcomeDir,'teacher_model'))

# saveing training history
fid = h5py.File(os.path.join(outcomeDir,'training_history.h5'),'w')
fid.create_dataset('alpha',data=alpha)
fid.create_dataset('classificationLossTrainStudent',data=classificationLossTrainStudent)
fid.create_dataset('classificationLossTrainTeacher',data=classificationLossTrainTeacher)
fid.create_dataset('consistentLossTrain',data=consistentLossTrain)
fid.create_dataset('consistentLossWeight',data=consistentLossWeight)
fid.create_dataset('classificationAccuTrainTeacher',data=classificationAccuTrainTeacher)
fid.create_dataset('classificationAccuTrainStudent',data=classificationAccuTrainStudent)
fid.create_dataset('classificationLossTestStudent',data=classificationLossTestStudent)
fid.create_dataset('classificationLossTestTeacher',data=classificationLossTestTeacher)
fid.create_dataset('classificationAccuTestTeacher',data=classificationAccuTestTeacher)
fid.create_dataset('classificationAccuTestStudent',data=classificationAccuTestStudent)
fid.close()



