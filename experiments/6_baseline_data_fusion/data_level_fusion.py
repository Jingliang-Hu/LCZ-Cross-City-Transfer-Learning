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
        #"nbEpoch": 1,

        "learningRate": 1e-4,

        ### data loading parameters
        "trainData": "lcz42", # training data could be the training data of LCZ42 data, or data of one of the cultural-10 city
        #"trainData": "moscow",
        "testData": "cul10",  # testing data could be all the data of the cultural-10 cities, or one of them.
        #"testData": "munich",
        
        "normalization_s2":"cms", # "ms": mean-std normalization, patch-wise
        "datFlag_s2":2, # data selection: sentinel-1, sentinel-2, or both

        "normalization_s1":"no", # "ms": mean-std normalization, patch-wise
        "datFlag_s1":1, # data selection: sentinel-1, sentinel-2, or both

        
        ### model name
        "modelName":'LeNet', # model name
        "nbStreams": 1,
        }

cudaNow = torch.device('cuda:0')

nbBatch = paraDict["nbBatch"]
nbEpoch = paraDict["nbEpoch"]
learnRate = paraDict["learningRate"]
datFlag_s1 = paraDict["datFlag_s1"]
datFlag_s2 = paraDict["datFlag_s2"]
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
trainDataSet_s1,testDataSet_s1 = lczIterDataSet(envPath,paraDict["trainData"],paraDict["testData"],datFlag_s1,paraDict["normalization_s1"],transform=transforms.Compose([ToTensor()]),shaffle=1)
trainDataSet_s2,testDataSet_s2 = lczIterDataSet(envPath,paraDict["trainData"],paraDict["testData"],datFlag_s2,paraDict["normalization_s2"],transform=transforms.Compose([ToTensor()]),shaffle=1)




# shuffle in dataloader shuffles the data in each batch
data_loaders = []
data_loaders.append(torch.utils.data.DataLoader(trainDataSet_s1, batch_size=nbBatch, shuffle=False))
data_loaders.append(torch.utils.data.DataLoader(trainDataSet_s2, batch_size=nbBatch, shuffle=False))

data_loaders.append(torch.utils.data.DataLoader(testDataSet_s1, batch_size=nbBatch, shuffle=False))
data_loaders.append(torch.utils.data.DataLoader(testDataSet_s2, batch_size=nbBatch, shuffle=False))







'''
STEP TWO: initial a resnet model
'''
sys.path.append(os.path.abspath(envPath+"/src/model"))
import resnetModel

if modelName=='resnet18':
    model = resnetModel.resnet18(inChannel=trainDataSet_s2.nbChannel()+trainDataSet_s1.nbChannel(), nbClass = trainDataSet_s2.label.shape[1]).to(cudaNow)
elif modelName=='LeNet':
    model = resnetModel.LeNet(inChannel=trainDataSet_s2.nbChannel()+trainDataSet_s1.nbChannel(), nbClass = trainDataSet_s2.label.shape[1]).to(cudaNow)






'''
STEP THREE: Define a loss function and optimizer
'''
import torch.optim as optim
import torch.nn as nn
classification_loss = nn.CrossEntropyLoss()
optimizers = optim.Adam(model.parameters(), lr=learnRate)


'''
STEP FOUR: Train the network
'''
import modelOperDataLoader

model, cla_loss_train, cla_acc_train, cla_loss_test, cla_acc_test, cla_averacc_test = modelOperDataLoader.train_data_level_fusion(model, data_loaders, optimizers, cudaNow, classification_loss,  nbEpoch)

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
model_name = 'data_level_fusion'
torch.save(model.state_dict(), os.path.join(outcomeDir,model_name))

# saveing training history
fid = h5py.File(os.path.join(outcomeDir,'training_history.h5'),'w')
fid.create_dataset('cla_loss_train',data=cla_loss_train)
fid.create_dataset('cla_acc_train',data=cla_acc_train)
fid.create_dataset('cla_loss_test',data=cla_loss_test)
fid.create_dataset('cla_acc_test',data=cla_acc_test)
fid.create_dataset('cla_averacc_test',data=cla_averacc_test)
fid.close()



'''
STEP SEVEN: Predict with the student1 model
'''
import modelOperDataLoader
_,  _, oa, aa, ka, pa, ua, confusion_matrix = modelOperDataLoader.test_data_level_fusion(model, cudaNow, data_loaders, classification_loss)

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



