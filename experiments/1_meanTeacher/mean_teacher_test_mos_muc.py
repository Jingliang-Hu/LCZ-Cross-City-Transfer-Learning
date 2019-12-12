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
cudaNow = torch.device('cuda:0')
nbBatch = 100
nbEpoch = 100
learnRate = 1e-4
momentum = 0.9
datFlag = 2
upperEpoch = 15
alpha = 0.8

modelName = 'meanTeacher_mos_muc'
trainHistoryFile = 'TrainHistory.h5'
savePath = envPath+'/trained_models/'+modelName
savePathStudent = savePath+'_student'
savePathTeacher = savePath+'_teacher'


histSavePath = envPath+'/trained_models/'+modelName+trainHistoryFile

subtract_pixel_mean = True


'''
STEP ONE: data loading
'''
print("data loading...")
# Load the LCZ42 training data.
_, x_train_2, y_train, _ = load_Semi_Test_City(envPath,"moscow",datFlag)
x_train = x_train_2
del x_train_2

# Load the LCZ42 testing data.
_, x_test_2, y_test, _ = load_Semi_Test_City(envPath,'munich', datFlag)
x_test = x_test_2
del x_test_2


# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
print("data normalization...")
v_max = np.max([np.max(x_train),np.max(x_test)])
v_min = np.min([np.min(x_train),np.min(x_test)])
x_train = (x_train-v_min) / (v_max-v_min)
x_test = (x_test-v_min) / (v_max-v_min)

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean


# mix training and testing data for semi-supvision training, reproducable random mixing
x_train = np.concatenate((x_train,x_test),axis=0)
y_train = np.concatenate((y_train,np.zeros(y_test.shape,y_test.dtype)),axis=0)

np.random.seed(0)
numSeed = np.random.rand(x_train.shape[0])
idx = np.argsort(numSeed)
x_train = x_train[idx]
y_train = y_train[idx]


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

optimizer = optim.SGD(student.parameters(), lr=learnRate, momentum=momentum)


'''
STEP FOUR: Train the network
'''
import modelOperation

print('Start training ...')

student,teacher,traSLoss,traSArry,valSLoss,valSArry,traTLoss,traTArry,valTLoss,valTArry = modelOperation.meanTeacher_Train(student,teacher,cudaNow,x_train,y_train,optimizer,x_test,y_test,classification_loss,consistency_loss,nbBatch,nbEpoch,alpha,upperEpoch)


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
savePathStudent = savePath+'_student'
savePathTeacher = savePath+'_teacher'


torch.save(student.state_dict(), savePathStudent)
torch.save(teacher.state_dict(), savePathTeacher)

fid = h5py.File(histSavePath,'w')
fid.create_dataset('traSLoss',data=traSLoss)
fid.create_dataset('traSArry',data=traSArry)
fid.create_dataset('valSLoss',data=valSLoss)
fid.create_dataset('valSArry',data=valSArry)

fid.create_dataset('traTLoss',data=traTLoss)
fid.create_dataset('traTArry',data=traTArry)
fid.create_dataset('valTLoss',data=valTLoss)
fid.create_dataset('valTArry',data=valTArry)

fid.close()




