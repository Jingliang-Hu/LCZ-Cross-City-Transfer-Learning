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
from tqdm import tqdm

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



cudaNow = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# load data
trainDataSet,testDataSet = lczIterDataSet(envPath,paraDict["trainData"],paraDict["testData"],paraDict["datFlag"],paraDict["normalization"])
source_loader = torch.utils.data.DataLoader(trainDataSet, batch_size=args.batch_size, shuffle=True)
target_loader = torch.utils.data.DataLoader(testDataSet,  batch_size=args.batch_size, shuffle=True)


len_source_dataset = len(source_loader.dataset)
len_target_dataset = len(target_loader.dataset)
len_source_loader = len(source_loader)
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
    for i_batch, sample in tqdm(enumerate(zip(source_loader,target_loader))):
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
        for i_batch, sample in tqdm(enumerate(target_loader)):
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


if __name__ == '__main__':
    model = models.AMRANNet(num_classes=17).to(cudaNow)
    print(args)
    outcomeDir = initialOutputFolder(paraDict)
    recordExpParameters(outcomeDir,paraDict)

    for epoch in range(1, args.epochs + 1):
        train(epoch, model)
        if epoch%10==0:
            t_correct = test(model)
    save_model_dir = os.path.join(outcomeDir, 'model')
    torch.save(model.state_dict(), save_model_dir)
