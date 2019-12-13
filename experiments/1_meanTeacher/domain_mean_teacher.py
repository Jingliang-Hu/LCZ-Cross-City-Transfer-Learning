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
cudaNow = torch.device('cuda:1')
nbBatch = 256
nbEpoch = 200
learnRate = 1e-3
momentum = 0.9
datFlag = 2
upperEpoch = 50
alphaMax = 0.99

modelName = 'domainMeanTeacher'
trainHistoryFile = 'TrainHistory.h5'
savePath = envPath+'/trained_models/'+modelName
savePathStudent = savePath+'_student'
savePathTeacher = savePath+'_teacher'


histSavePath = envPath+'/trained_models/'+modelName+trainHistoryFile


'''
STEP ONE: data loading
'''
print("data loading...")
# Load the LCZ42 training data.
_, x_train, y_train, _ = load_Semi_Train(envPath,datFlag)

# Load the LCZ42 testing data.
_, x_test, y_test, _ = load_Semi_Test(envPath, datFlag)

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
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
# optimizer = optim.SGD(student.parameters(), lr=learnRate, momentum=momentum)
optimizer = optim.Adam(student.parameters(), lr=learnRate)


'''
STEP FOUR: Train the network
'''
import modelOperation

print('Start training ...')
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
savePathStudent = savePath+'_student'
savePathTeacher = savePath+'_teacher'


torch.save(student.state_dict(), savePathStudent)
torch.save(teacher.state_dict(), savePathTeacher)

fid = h5py.File(histSavePath,'w')
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



