# example resource link: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html



import sys
sys.path.insert(0,'../io')
import dataReader


'''
STEP ONE: data loading
'''
dr = dataReader.dataReader()
# read 's2' data
dr.changeDatOpt('s2')
#       a. load one .h5 file
dat,lab = dr.loadOneFile('/data/hu/TF/data/testCites/munich.h5')
#       b. train and test spliting
dat_tr,lab_tr,dat_te,lab_te = dr.randomSplit(dat,lab)


'''
STEP TWO: initial a resnet model
'''
import resnetModel
resnet = resnetModel.resnet18(pretrained=False, inChannel=dat_tr.shape[3])
print(resnet)


'''
STEP THREE: Define a loss function and optimizer
'''
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


'''
STEP FOUR: Train the network
'''




