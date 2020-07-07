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
        ###"trained_model":"",
        "trained_model":"domain_mean_teacher_tr_lcz42_te_cul10_outcome_2020-01-22_04-11-29",
        ### network parameters
        "nbBatch": 256,
        ### data loading parameters
        "trainData": "cul10_test", # training data could be the training data of LCZ42 data, or data of one of the cultural-10 city
        "testData": "cul10_test",  # testing data could be all the data of the cultural-10 cities, or one of them.
        "normalization":"cms", # "cms": channel-wise mean-std normalization
        # "normalization":"pms", # "pms": patch-wise mean-std normalization
        "datFlag":2, # data selection: sentinel-1, sentinel-2, or both
        }

cudaNow = torch.device('cuda:0')
nbBatch = paraDict["nbBatch"]
datFlag = paraDict["datFlag"]
modelName = paraDict["trained_model"]

'''
initial folder saving outputs
'''
outcomeDir =  os.path.join(modelName,"test_cul10_perc_50")
if not os.path.exists(outcomeDir):
    os.mkdir(outcomeDir)
print("Testing outcomes are saving in the directory: "+outcomeDir)

'''
STEP ONE: data loading
'''
trainDataSet,testDataSet = lczIterDataSet(envPath,paraDict["trainData"],paraDict["testData"],datFlag,paraDict["normalization"])
testDataLoader = torch.utils.data.DataLoader(testDataSet, batch_size=512)


'''
STEP TWO: Predict with the trained model
'''
sys.path.append(os.path.abspath(envPath+"/src/model"))
import resnetModel
import modelOperDataLoader

modelPath = os.path.join(modelName,'teacher_model')
predModel = resnetModel.resnet18(pretrained=False, inChannel=trainDataSet.nbChannel()).to(cudaNow)
predModel.load_state_dict(torch.load(modelPath,map_location=cudaNow))
confusion_matrix,oa,aa,ka,pa,ua = modelOperDataLoader.confusionMatrix(predModel,cudaNow,testDataLoader,criterion=0)
print('Teacher Overall accuracy:'+str(oa))
print('Teacher Average accuracy:'+str(aa))
print('Teacher kappa:'+str(ka))



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


modelPath = os.path.join(modelName,'student_model')
predModel = resnetModel.resnet18(pretrained=False, inChannel=trainDataSet.nbChannel()).to(cudaNow)
predModel.load_state_dict(torch.load(modelPath,map_location=cudaNow))
confusion_matrix,oa,aa,ka,pa,ua = modelOperDataLoader.confusionMatrix(predModel,cudaNow,testDataLoader,criterion=0)
print('Student Overall accuracy:'+str(oa))
print('Student Average accuracy:'+str(aa))
print('Student kappa:'+str(ka))



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




