from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
import sys
import math
import time
import data_loader
import ResNet as models
#from tqdm import tqdm
import numpy as np



fid = open("../../../.envPath","r")
envPath = fid.readline()
envPath = envPath[:-1]
fid.close
del fid
sys.path.append(os.path.abspath(envPath+"/src/io"))
from load_Cul10_Semi import *

# Training settings
paraDict = {
        "batchSize":256,
        "epoch":728,
        "learning_rate":0.002,
        "modelName":"amran",
        "trainData": "lcz42",
        "testData": "cul10",
        "normalization":"no",
        "datFlag":2,
        }

parser = argparse.ArgumentParser(description='PyTorch Transfer Framework')
parser.add_argument('--batch-size', type=int, default=paraDict["batchSize"])
parser.add_argument('--epochs', type=int, default=paraDict["epoch"])
parser.add_argument('--lr', type=float, default=paraDict["learning_rate"])
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=3)
parser.add_argument('--log-interval', type=int, default=5)
parser.add_argument('--l2_decay', type=float, default=5e-4)
parser.add_argument('--mu', type=float, default=0)
parser.add_argument('--root_path', type=str, default="/data/sihan.zhu/transfer learning/deep/dataset/RSTL/")
parser.add_argument('--source_dir', type=str, default="UCM")
parser.add_argument('--test_dir', type=str, default="RSSCN7")
# RSTL
# UCM WHU AID RSSCN7
args = parser.parse_args()



cudaNow = torch.device("cuda:1")


# load data
trainDataSet,testDataSet = lczIterDataSet(envPath,paraDict["trainData"],paraDict["testData"],paraDict["datFlag"],paraDict["normalization"])
target_loader = torch.utils.data.DataLoader(testDataSet,  batch_size=args.batch_size, shuffle=True)


len_target_dataset = len(target_loader.dataset)
len_target_loader = len(target_loader)


def train(epoch, model):
    LEARNING_RATE = args.lr / math.pow((1 + 10 * (epoch - 1) / args.epochs), 0.75)
    print('learning rate{: .4f}'.format(LEARNING_RATE))
    optimizer = torch.optim.SGD([
        {'params': model.sharedNet.parameters()},
        {'params': model.Inception.parameters(), 'lr': LEARNING_RATE},
    ], lr=LEARNING_RATE / 10, momentum=args.momentum, weight_decay=args.l2_decay)

    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    for i_batch, sample in enumerate(zip(source_loader,target_loader)):
        data_source = sample[0]['data'].to(cudaNow,dtype=torch.float)
        label_source = torch.argmax(sample[0]['label'],dim=1)
        label_source = label_source.to(cudaNow,dtype=torch.long)
        data_target = sample[1]['data'].to(cudaNow,dtype=torch.float)
        if data_target.shape[0]!=args.batch_size:
            continue
        optimizer.zero_grad()
        # print(data_source.shape)
        s_output, mmd_loss = model(data_source, data_target, label_source, args.mu)
        cls_loss = criterion(s_output, label_source)
        gamma = 2 / (1 + math.exp(-10 * (epoch) / args.epochs)) - 1
        loss = cls_loss + gamma * mmd_loss
        loss.backward()
        optimizer.step()
    print('Train Epoch: {}; Train Loss: {:.6f}; Classification Loss: {:.6f}; mmd Loss: {:.6f}'.format(epoch, loss.item(), cls_loss.item(), mmd_loss.item()))


def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for i_batch, sample in enumerate(target_loader):
            data = sample['data'].to(cudaNow,dtype=torch.float)
            target = torch.argmax(sample['label'],dim=1)
            target = target.to(cudaNow,dtype=torch.long)
            s_output, _ = model(data, data, target, args.mu)
            test_loss += criterion(s_output, target)# sum up batch loss
            pred = torch.max(s_output, 1)[1]  # get the index of the max log-probability
            correct += torch.sum(pred == target)
        test_loss /= len_target_dataset
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len_target_dataset,
            100. * correct / len_target_dataset))
    return correct

def predict(model):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    confusion_matrix = np.zeros((17,17))
    with torch.no_grad():
        for i_batch, sample in enumerate(target_loader):
            data = sample['data'].to(cudaNow,dtype=torch.float)
            target = torch.argmax(sample['label'],dim=1)
            target = target.to(cudaNow,dtype=torch.long)
            s_output, _ = model(data, data, target, args.mu)
            pred = torch.max(s_output, 1)[1]  # get the index of the max log-probability
            for l, p in zip(target.view(-1), pred.view(-1)):
                confusion_matrix[l.long(), p.long()] += 1

    pa = np.diagonal(confusion_matrix)/np.sum(confusion_matrix,1)
    ua = np.diagonal(confusion_matrix)/np.sum(confusion_matrix,0)
    oa = np.trace(confusion_matrix)/np.sum(confusion_matrix)
    aa = np.sum(pa[~np.isnan(pa)])/np.sum(~np.isnan(pa))
    # kappa coefficient
    po = oa
    pe = np.sum(np.sum(confusion_matrix,0)*np.sum(confusion_matrix,1))/np.square(np.sum(confusion_matrix))
    ka = (po-pe)/(1-pe)

    return confusion_matrix,oa,aa,ka,pa,ua



if __name__ == '__main__':
    # model_directories = ['amran_tr_lcz42_te_cul10_outcome_2021-05-20_22-32-50','amran_tr_lcz42_te_cul10_outcome_2021-05-23_09-16-29','amran_tr_lcz42_te_cul10_outcome_2021-05-29_05-19-35']
    model_directories = ['amran_tr_lcz42_te_cul10_outcome_2021-05-29_05-19-35']
    for model_name in model_directories:
        print(' ------------------------------------- ')
        modelPath = os.path.join(model_name,'model')
        if not os.path.exists(modelPath):
            print('Model does not exists: {}'.format(model_name))
            continue
        else:
            print('Loading model in: {}'.format(model_name))
        model = models.AMRANNet(num_classes=17).to(cudaNow)
        print('Predicting ...')
        model.load_state_dict(torch.load(modelPath,map_location=cudaNow))
        confusion_matrix,oa,aa,ka,pa,ua = predict(model)
        print('Saving ...')
        fid = h5py.File(os.path.join(model_name,'test_accuracy.h5'),'w')
        fid.create_dataset('confusion_matrix',data=confusion_matrix)
        fid.create_dataset('oa',data=oa)
        fid.create_dataset('aa',data=aa)
        fid.create_dataset('ka',data=ka)
        fid.create_dataset('pa',data=pa)
        fid.create_dataset('ua',data=ua)
        fid.close()
