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
        "nbEpoch": 50,
        "learningRate": 1e-4,
        ### data loading parameters
        "trainData": "lcz42", # training data could be the training data of LCZ42 data, or data of one of the cultural-10 city
        #"trainData": "munich",

        "testData": "cul10",  # testing data could be all the data of the cultural-10 cities, or one of them.
        #"testData": "moscow",

        "normalization":"no", # "cms": channel-wise mean-std normalization
        # "normalization":"pms", # "pms": patch-wise mean-std normalization
        "datFlag":2, # data selection: sentinel-1, sentinel-2, or both

        ### model name
        #"modelName":'LeNet_conv5',
        #"dirSourceTrainedModel": 'LeNet_conv5_tr_lcz42_te_cul10_outcome_2020-10-16_02-47-40',

        #"modelName":'ResNet',
        #"dirSourceTrainedModel": 'ResNet_tr_lcz42_te_cul10_outcome_2020-10-15_09-17-57',

        "modelName":'Sen2LCZ',#'LeNet', # model name
        "Sen2LCZ_drop_out": 0.2,
        "dirSourceTrainedModel": 'Sen2LCZ_tr_lcz42_te_cul10_outcome_2020-10-16_15-54-07',

        "confidenceThreshold": 0.7,

        }
cudaNow = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(cudaNow)
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
STEP ONE: load samples from target domain
'''
trainDataSet,testDataSet = lczIterDataSet(envPath,paraDict["trainData"],paraDict["testData"],datFlag,paraDict["normalization"])
f = h5py.File(testDataSet.dataFile,'r')
samples_of_target = np.array(f['x_2'])
del f

'''
STEP TWO: load a trained model
'''
sys.path.append(os.path.abspath(envPath+"/src/model"))
import resnetModel
if modelName == 'ResNet':
    predModel = resnetModel.resnet18(pretrained=False, inChannel=testDataSet.nbChannel()).to(cudaNow)
elif  modelName == 'LeNet':
    predModel = resnetModel.LeNet(inChannel=testDataSet.nbChannel(), nbClass = testDataSet.label.shape[1]).to(cudaNow)
elif modelName == 'Sen2LCZ':
    predModel = resnetModel.Sen2LCZ(in_Channel=10, nb_class=17, nb_kernel=16, depth=17, bn_flag=1, drop_rate=paraDict["Sen2LCZ_drop_out"]).to(cudaNow)
elif modelName == 'LeNet_conv5':
    predModel = resnetModel.LeNet_conv_5(inChannel=testDataSet.nbChannel(), nbClass = testDataSet.label.shape[1]).to(cudaNow)
else:
    display('model not defined')


modelPath = '../2_baseline_42/'+paraDict["dirSourceTrainedModel"]+'/model'
predModel.load_state_dict(torch.load(modelPath,map_location=cudaNow))

'''
STEP THREE: predict labels for samples from the target domain and save the labels
'''
import modelOperDataLoader
pred = modelOperDataLoader.prediction_softmax_gpu(predModel, samples_of_target, cudaNow)
fid = h5py.File(os.path.join(outcomeDir,'pseudo_label.h5'),'w')
fid.create_dataset('pseudo',data=pred)
fid.close()


'''
STEP FOUR: 
'''
'''
Select labels from target domain
'''
def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

confidence = pred.max(axis=1)
pseudo_label = np.argmax(pred,axis=1)
pseudo_label = pseudo_label[np.where(confidence>paraDict["confidenceThreshold"])]
pseudo_label_one_hot = get_one_hot(pseudo_label,17)

trainDataSet.setData(samples_of_target[np.where(confidence>paraDict["confidenceThreshold"])])
trainDataSet.setLabel(pseudo_label_one_hot)

trainDataLoader = torch.utils.data.DataLoader(trainDataSet, batch_size=nbBatch, shuffle=True)
testDataLoader = torch.utils.data.DataLoader(testDataSet, batch_size=512)



'''
Define a loss function and optimizer
'''
import torch.optim as optim
import torch.nn as nn
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(predModel.parameters(), lr=learnRate)


'''
Fine tuning the network
'''
# import modelOperation
print('Start training ...')
model,traLoss,traArry,valLoss,valArry,valAver = modelOperDataLoader.train(predModel,cudaNow,optimizer,trainDataLoader,criterion,nbEpoch,testDataLoader)


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

torch.save(model.state_dict(), os.path.join(outcomeDir,'fine_tuning_model'))
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
confusion_matrix,oa,aa,ka,pa,ua = modelOperDataLoader.confusionMatrix(model,cudaNow,testDataLoader,criterion)

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




