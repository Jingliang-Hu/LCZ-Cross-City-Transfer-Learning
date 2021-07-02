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
        
        "normalization":"no", # "ms": mean-std normalization, patch-wise
        "datFlag":2, # data selection: sentinel-1, sentinel-2, or both
        
        ### model name
        #"modelName":'ResNet',
        "modelName":'LeNet_conv5',
        #"modelName":'LeNet', # model name
        #"modelName":'Sen2LCZ',#'LeNet', # model name
        #"Sen2LCZ_drop_out": 0.2,

        }

cudaNow = torch.device('cuda:2')

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
## record parameters
recordExpParameters(outcomeDir,paraDict)


'''
STEP ONE: data loading
'''
trainDataSet,testDataSet = lczIterDataSet(envPath,paraDict["trainData"],paraDict["testData"],datFlag,paraDict["normalization"],transform=transforms.Compose([ToTensor()]))

s1_source_data_loader = torch.utils.data.DataLoader(trainDataSet, batch_size=nbBatch, shuffle=True)
s2_source_data_loader = torch.utils.data.DataLoader(trainDataSet, batch_size=nbBatch, shuffle=True)
target_data_loader = torch.utils.data.DataLoader(testDataSet, batch_size=nbBatch, shuffle=True)

'''
STEP TWO: initial a resnet model
'''
sys.path.append(os.path.abspath(envPath+"/src/model"))
import resnetModel
if modelName == 'ResNet':
    student1 = resnetModel.resnet18(pretrained=False, inChannel=trainDataSet.nbChannel()).to(cudaNow)
    student2 = resnetModel.resnet18(pretrained=False, inChannel=trainDataSet.nbChannel()).to(cudaNow)
elif  modelName == 'LeNet':
    student1 = resnetModel.LeNet(inChannel=trainDataSet.nbChannel(), nbClass = trainDataSet.label.shape[1]).to(cudaNow)
    student2 = resnetModel.LeNet(inChannel=trainDataSet.nbChannel(), nbClass = trainDataSet.label.shape[1]).to(cudaNow)
elif modelName == 'Sen2LCZ':
    student1 = resnetModel.Sen2LCZ(in_Channel=10, nb_class=17, nb_kernel=16, depth=17, bn_flag=1, drop_rate=paraDict["Sen2LCZ_drop_out"]).to(cudaNow)
    student2 = resnetModel.Sen2LCZ(in_Channel=10, nb_class=17, nb_kernel=16, depth=17, bn_flag=1, drop_rate=paraDict["Sen2LCZ_drop_out"]).to(cudaNow)
elif modelName == 'LeNet_conv5':
    student1 = resnetModel.LeNet_conv_5(inChannel=trainDataSet.nbChannel(), nbClass = trainDataSet.label.shape[1]).to(cudaNow)
    student2 = resnetModel.LeNet_conv_5(inChannel=trainDataSet.nbChannel(), nbClass = trainDataSet.label.shape[1]).to(cudaNow)
else:
    display('model not defined')



'''
STEP THREE: Define a loss function and optimizer
'''
import torch.optim as optim
import torch.nn as nn
classification_loss = nn.CrossEntropyLoss()
consistency_loss = nn.MSELoss()
optimizer_stu1 = optim.Adam(student1.parameters(), lr=learnRate)
optimizer_stu2 = optim.Adam(student2.parameters(), lr=learnRate)



'''
STEP FOUR: Train the network
'''
import modelOperDataLoader
student1, student2, cla_loss_train_student1, cla_acc_train_student1, cla_loss_test_student1, cla_acc_test_student1, cla_loss_train_student2, cla_acc_train_student2, cla_loss_test_student2, cla_acc_test_student2, con_loss_train_student1, con_loss_train_student2, cla_averacc_test_student1, cla_averacc_test_student2  = modelOperDataLoader.train_unlabel_ensemble_aug(student1, student2, cudaNow, s1_source_data_loader, s2_source_data_loader, target_data_loader, classification_loss, consistency_loss, optimizer_stu1, optimizer_stu2, nbEpoch)

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

torch.save(student1.state_dict(), os.path.join(outcomeDir,'student1_model'))
torch.save(student2.state_dict(), os.path.join(outcomeDir,'student2_model'))

# saveing training history
fid = h5py.File(os.path.join(outcomeDir,'training_history.h5'),'w')
fid.create_dataset('cla_loss_train_student1',data=cla_loss_train_student1)
fid.create_dataset('cla_loss_train_student2',data=cla_loss_train_student2)
fid.create_dataset('con_loss_train_student1',data=con_loss_train_student1)
fid.create_dataset('con_loss_train_student2',data=con_loss_train_student2)
fid.create_dataset('cla_acc_train_student1',data=cla_acc_train_student1)
fid.create_dataset('cla_acc_train_student2',data=cla_acc_train_student2)
fid.create_dataset('cla_loss_test_student1',data=cla_loss_test_student1)
fid.create_dataset('cla_loss_test_student2',data=cla_loss_test_student2)
fid.create_dataset('cla_acc_test_student1',data=cla_acc_test_student1)
fid.create_dataset('cla_acc_test_student2',data=cla_acc_test_student2)
fid.create_dataset('cla_averacc_test_student1',data=cla_averacc_test_student1)
fid.create_dataset('cla_averacc_test_student2',data=cla_averacc_test_student2)
fid.close()



'''
STEP SEVEN: Predict with the student1 model
'''
confusion_matrix,oa,aa,ka,pa,ua = modelOperDataLoader.confusionMatrix(student1, cudaNow, target_data_loader, classification_loss)

# predModel_s1.load_state_dict(torch.load(modelPath,map_location=cudaNow))
# confusion_matrix,oa,aa,ka,pa,ua = modelOperDataLoader.confusionMatrix(predModel_s1, cudaNow, target_data_loader, classification_loss)



# save accuracy
fid = h5py.File(os.path.join(outcomeDir,'student1_test_accuracy.h5'),'w')
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
cm_disp.figure_.savefig(os.path.join(outcomeDir,'student1_confusion_matrix.png'))


'''
STEP SEVEN: Predict with the trained student2 model
'''
confusion_matrix,oa,aa,ka,pa,ua = modelOperDataLoader.confusionMatrix(student2, cudaNow, target_data_loader, classification_loss)

# save accuracy
fid = h5py.File(os.path.join(outcomeDir,'student2_test_accuracy.h5'),'w')
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
cm_disp.figure_.savefig(os.path.join(outcomeDir,'student2_confusion_matrix.png'))


'''
plot outcomes
'''
pth = modelOperDataLoader.plotTrainHistory(outcomeDir,paraDict["modelName"])
pth.plotHistory()



