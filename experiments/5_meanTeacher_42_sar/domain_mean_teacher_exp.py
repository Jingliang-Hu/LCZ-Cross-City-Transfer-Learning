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
        "nbEpoch": 728, #728
        "learningRate": 1e-4,
        "learningRate_decay":"step",
        "learningRate_stepSize":200,
        "maxEpochConsisLossWeight": 50,
        "confidenceModel":0,
        "confidenceThreshold":0.9,
        "alphaEMA": 0.99,
        "alphaMaxEpoch": 50,
 
        ### data loading parameters
        
        "trainData": "lcz42", # training data could be the training data of LCZ42 data, or data of one of the cultural-10 city
        "testData": "cul10",  # testing data could be all the data of the cultural-10 cities, or one of them.
        "normalization":"no", # "ms": mean-std normalization, patch-wise
        "datFlag":1, # data selection: sentinel-1, sentinel-2, or both
        
        ### model name
        "modelName":'LeNet_mean_teacher', # model name
        }

cudaNow = torch.device('cuda:6')

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
## record parameters
recordExpParameters(outcomeDir,paraDict)


'''
STEP ONE: data loading
'''
trainDataSet,testDataSet = lczIterDataSet(envPath,paraDict["trainData"],paraDict["testData"],datFlag,paraDict["normalization"])
# balancing the numbers of samples in source and target domains
print('The number of samples in source domain: '+str(trainDataSet.label.shape[0]))
print('The number of samples in target domain: '+str(testDataSet.label.shape[0]))

trainDataLoader = torch.utils.data.DataLoader(trainDataSet, batch_size=nbBatch, shuffle=True)
testDataLoader = torch.utils.data.DataLoader(testDataSet, batch_size=nbBatch, shuffle=True)







'''
STEP TWO: initial a resnet model
'''
sys.path.append(os.path.abspath(envPath+"/src/model"))
import resnetModel
# student = resnetModel.resnet18(pretrained=False, inChannel=trainDataSet.nbChannel()).to(cudaNow)
# teacher = resnetModel.resnet18(pretrained=False, inChannel=trainDataSet.nbChannel()).to(cudaNow)
# predModel_stu = resnetModel.resnet18(pretrained=False, inChannel=trainDataSet.nbChannel()).to(cudaNow)
# predModel_tea = resnetModel.resnet18(pretrained=False, inChannel=trainDataSet.nbChannel()).to(cudaNow)

student = resnetModel.LeNet(inChannel=trainDataSet.nbChannel(), nbClass = trainDataSet.label.shape[1]).to(cudaNow)
teacher = resnetModel.LeNet(inChannel=trainDataSet.nbChannel(), nbClass = trainDataSet.label.shape[1]).to(cudaNow)
predModel_stu = resnetModel.LeNet(inChannel=trainDataSet.nbChannel(), nbClass = trainDataSet.label.shape[1]).to(cudaNow)
predModel_tea = resnetModel.LeNet(inChannel=trainDataSet.nbChannel(), nbClass = trainDataSet.label.shape[1]).to(cudaNow)




'''
STEP THREE: Define a loss function and optimizer
'''
import torch.optim as optim
import torch.nn as nn
classification_loss = nn.CrossEntropyLoss()
consistency_loss = nn.MSELoss()
optimizer = optim.Adam(student.parameters(), lr=learnRate)

# Assuming optimizer uses lr = 0.01 for all groups
# lr = 0.01     if epoch < 30
# lr = 0.001    if 30 <= epoch < 60
# lr = 0.0001   if 60 <= epoch < 90
# ...
if paraDict["learningRate_decay"] =="step":
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=paraDict["learningRate_stepSize"], gamma=0.1)
else:
    scheduler = None


'''
STEP FOUR: Train the network
'''
import modelOperDataLoader

print('Start training ...')
if confidentModel:
    student,teacher,classificationLossTrainStudent,classificationAccuTrainStudent,classificationLossTestStudent,classificationAccuTestStudent,classificationLossTrainTeacher,classificationAccuTrainTeacher,classificationLossTestTeacher,classificationAccuTestTeacher,consistentLossTrain, consistentLossWeight, alpha, classificationAverAccuTestStudent, classificationAverAccuTestTeacher = modelOperDataLoader.domainMeanTeacherConfidence_Train(student,teacher,cudaNow,trainDataLoader,optimizer,testDataLoader,classification_loss,consistency_loss,nbEpoch,alphaMax,alphaMaxEpoch,confidentThres,scheduler)

else:
    student,teacher,classificationLossTrainStudent,classificationAccuTrainStudent,classificationLossTestStudent,classificationAccuTestStudent,classificationLossTrainTeacher,classificationAccuTrainTeacher,classificationLossTestTeacher,classificationAccuTestTeacher,consistentLossTrain, consistentLossWeight, alpha, classificationAverAccuTestStudent, classificationAverAccuTestTeacher, learning_rate_values = modelOperDataLoader.domainMeanTeacher_Train(student,teacher,cudaNow,trainDataLoader,optimizer,testDataLoader,classification_loss,consistency_loss,nbEpoch,alphaMax,scheduler)

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
fid.create_dataset('classificationAverAccuTestStudent',data=classificationAverAccuTestStudent)
fid.create_dataset('classificationAverAccuTestTeacher',data=classificationAverAccuTestTeacher)
fid.create_dataset('learning_rate_values',data=learning_rate_values)
fid.close()



'''
STEP SEVEN: Predict with the trained teacher model
'''
modelPath = os.path.join(outcomeDir,'teacher_model')
predModel_tea.load_state_dict(torch.load(modelPath,map_location=cudaNow))
confusion_matrix,oa,aa,ka,pa,ua = modelOperDataLoader.confusionMatrix(predModel_tea,cudaNow,testDataLoader,classification_loss)

# save accuracy
fid = h5py.File(os.path.join(outcomeDir,'teacher_test_accuracy.h5'),'w')
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
cm_disp.figure_.savefig(os.path.join(outcomeDir,'teacher_confusion_matrix.png'))


'''
STEP SEVEN: Predict with the trained student model
'''
modelPath = os.path.join(outcomeDir,'student_model')
predModel_stu.load_state_dict(torch.load(modelPath,map_location=cudaNow))
confusion_matrix,oa,aa,ka,pa,ua = modelOperDataLoader.confusionMatrix(predModel_stu,cudaNow,testDataLoader,classification_loss)

# save accuracy
fid = h5py.File(os.path.join(outcomeDir,'student_test_accuracy.h5'),'w')
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
cm_disp.figure_.savefig(os.path.join(outcomeDir,'student_confusion_matrix.png'))


'''
plot outcomes
'''
pth = modelOperDataLoader.plotTrainHistory(outcomeDir,paraDict["modelName"])
pth.plotHistory()


