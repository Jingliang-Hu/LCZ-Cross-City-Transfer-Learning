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
        "nbEpoch": 728,
        "learningRate": 1e-4,

        ### data loading parameters
        "trainData": "lcz42", # training data could be the training data of LCZ42 data, or data of one of the cultural-10 city
        "testData": "cul10",  # testing data could be all the data of the cultural-10 cities, or one of them.
        
        "normalization":"cms", # "ms": mean-std normalization, patch-wise
        "datFlag":2, # data selection: sentinel-1, sentinel-2, or both
        
        ### model name
        "modelName":'LeNet_ensemble', # model name
        "nbStreams":5,
        }

cudaNow = torch.device('cuda:5')

nbBatch = paraDict["nbBatch"]
nbEpoch = paraDict["nbEpoch"]
learnRate = paraDict["learningRate"]
datFlag = paraDict["datFlag"]
modelName = paraDict["modelName"]
nbStreams = paraDict["nbStreams"]


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
trainDataSet,testDataSet = lczIterDataSet(envPath,paraDict["trainData"],paraDict["testData"],datFlag,paraDict["normalization"],transform=transforms.Compose([ToTensor()]))

data_loaders = []
for i in range(0,nbStreams):
    data_loaders.append(torch.utils.data.DataLoader(trainDataSet, batch_size=nbBatch, shuffle=True))
data_loaders.append(torch.utils.data.DataLoader(testDataSet, batch_size=nbBatch, shuffle=True))
# target_loader = torch.utils.data.DataLoader(testDataSet, batch_size=nbBatch, shuffle=True)


'''
STEP TWO: initial a resnet model
'''
sys.path.append(os.path.abspath(envPath+"/src/model"))
import resnetModel
# student1 = resnetModel.resnet18(pretrained=False, inChannel=trainDataSet.nbChannel()).to(cudaNow)
# student2 = resnetModel.resnet18(pretrained=False, inChannel=trainDataSet.nbChannel()).to(cudaNow)
students = []
for i in range(0,nbStreams):
    students.append(resnetModel.LeNet(inChannel=trainDataSet.nbChannel(), nbClass = trainDataSet.label.shape[1]).to(cudaNow))

# predModel_s1 = resnetModel.LeNet(inChannel=trainDataSet.nbChannel(), nbClass = trainDataSet.label.shape[1]).to(cudaNow)
# predModel_s2 = resnetModel.LeNet(inChannel=trainDataSet.nbChannel(), nbClass = trainDataSet.label.shape[1]).to(cudaNow)



'''
STEP THREE: Define a loss function and optimizer
'''
import torch.optim as optim
import torch.nn as nn
classification_loss = nn.CrossEntropyLoss()
consistency_loss = nn.MSELoss()
optimizers = []
for i in range(0,nbStreams):
    optimizers.append(optim.Adam(students[i].parameters(), lr=learnRate))


'''
STEP FOUR: Train the network
'''
import modelOperDataLoader_dev

students, cla_loss_train, cla_acc_train, cla_loss_test, cla_acc_test, con_loss_train, cla_averacc_test = modelOperDataLoader_dev.train_multi_ensemble(students, data_loaders, optimizers, cudaNow, classification_loss, consistency_loss, nbEpoch)

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
    torch.save(students[i].state_dict(), os.path.join(outcomeDir,model_name))

# saveing training history
fid = h5py.File(os.path.join(outcomeDir,'training_history.h5'),'w')
fid.create_dataset('cla_loss_train',data=cla_loss_train)
fid.create_dataset('cla_acc_train',data=cla_acc_train)
fid.create_dataset('cla_loss_test',data=cla_loss_test)
fid.create_dataset('cla_acc_test',data=cla_acc_test)
fid.create_dataset('con_loss_train',data=con_loss_train)
fid.create_dataset('cla_averacc_test',data=cla_averacc_test)
fid.close()



'''
STEP SEVEN: Predict with the student1 model
'''
confusion_matrix,oa,aa,ka,pa,ua = modelOperDataLoader_dev.multiStreamConfusionMatrix(students, cudaNow, data_loaders[nbStreams], classification_loss)

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
cm_disp_obj = modelOperDataLoader_dev.ConfusionMatrixDisplay(confusion_matrix,np.linspace(1,confusion_matrix.shape[0],confusion_matrix.shape[0]))
cm_disp = cm_disp_obj.plot()
cm_disp.figure_.savefig(os.path.join(outcomeDir,'confusion_matrix.png'))




'''
plot outcomes
'''
pth = modelOperDataLoader_dev.plotTrainHistory(outcomeDir,paraDict["modelName"])
pth.plotHistory()



