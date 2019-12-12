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

'''
PARAMETER SETTING
'''
print("parameter setting...")
cudaNow = torch.device('cuda:3')
nbBatch = 256
nbEpoch = 200
learnRate = 1e-3
momentum = 0.9
datFlag = 2

modelName = 'benchMark'
trainHistoryFile = 'TrainHistory.h5'
savePath = envPath+'/trained_models/'+modelName
histSavePath = envPath+'/trained_models/'+modelName+trainHistoryFile


'''
STEP ONE: data loading
'''
print("data loading...")
# Load the LCZ42 training data.
_,x_train,y_train,_ = load_Semi_Train(envPath,datFlag)

# Load the LCZ42 testing data.
_, x_test, y_test, _ = load_Semi_Test(envPath, datFlag)

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
print("data normalization...")
#v_max = np.max([np.max(x_train),np.max(x_test)])
#v_min = np.min([np.min(x_train),np.min(x_test)])
#x_train = (x_train-v_min) / (v_max-v_min)
#x_test = (x_test-v_min) / (v_max-v_min)
x_train = mean_Std_Normalization(x_train)
x_test = mean_Std_Normalization(x_test)


# If subtract pixel mean is enabled
#if subtract_pixel_mean:
#    x_train_mean = np.mean(x_train, axis=0)
#    x_train -= x_train_mean
#    x_test -= x_train_mean

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
resnet = resnetModel.resnet18(pretrained=False, inChannel=x_train.shape[1]).to(cudaNow)


'''
STEP THREE: Define a loss function and optimizer
'''
import torch.optim as optim
import torch.nn as nn
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(resnet.parameters(), lr=learnRate, momentum=momentum)
optimizer = optim.Adam(resnet.parameters(), lr=learnRate)


'''
STEP FOUR: Train the network
'''
import modelOperation

print('Start training ...')
resnet,traLoss,traArry,valLoss,valArry = modelOperation.train(resnet,cudaNow,optimizer,x_train,y_train,criterion,numBatch=nbBatch, numEpoch=nbEpoch, valDat=x_test, valLab=y_test, lrChg=False,earlyStop=False,numPatient=nbEpoch)

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
torch.save(resnet.state_dict(), savePath)
fid = h5py.File(histSavePath,'w')
fid.create_dataset('traLoss',data=traLoss)
fid.create_dataset('traArry',data=traArry)
fid.create_dataset('valLoss',data=valLoss)
fid.create_dataset('valArry',data=valArry)
fid.close()




