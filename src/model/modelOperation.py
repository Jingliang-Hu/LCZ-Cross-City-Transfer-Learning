import torch 
import torch.nn as nn 
import torch.optim as optim 
import sys
sys.path.insert(0,'../io')
import dataReader
import numpy as np
from tqdm import tqdm


def test(model,device,dat,lab,criterion = nn.CrossEntropyLoss(),numBatch=64):
    '''
    Input:
            - model         -- pytorch model
            - device        -- defined cpu or gpu device
            - dat        	-- data for testing
            - traLab        -- label for testing
            - criterion     -- objective function for optimization                  (default cross entropy)
            - numBatch      -- number of samples in a batch                         (default 64)
    Output:
            - pred          -- prediction of input data using model
            - testLoss      -- mean testing error of batches
            - accuracy      -- testing overall accuracy
    '''

    model.eval()
    testLoss = 0.0
    accuracy = 0.0 # overall accuracy
    pred = np.zeros_like(lab)

    with torch.no_grad():
        for t in range(np.int(np.ceil(dat.shape[0]/numBatch))):
            # batch organization
            if t == np.int(np.floor(dat.shape[0]/numBatch)):
                idx = np.arange(t*numBatch,dat.shape[0])
            else:
                idx = np.arange(t*numBatch,(t+1)*numBatch)
            inDat = dat[idx,:,:,:].cuda(device)
            inLab = lab[idx].cuda(device)
            # predicting
            output = model(inDat) 
            # prediction error
            loss = criterion(output, inLab)
            testLoss += loss.item()
            predTmp = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability 
            pred[idx] = np.squeeze(predTmp.cpu().numpy())
    testLoss = testLoss / np.ceil(dat.shape[0]/numBatch)
    lab = lab.numpy()
    accuracy = np.sum(pred==lab)/lab.shape[0]*100
    return pred, testLoss, accuracy 


def train(model,device,optimizer,traDat,traLab,criterion = nn.CrossEntropyLoss(),numBatch=64, numEpoch=200, valDat=0, valLab=0, lrChg=False,earlyStop=False,numPatient=20):
    '''
    Input:
	- model 	-- pytorch model
	- device 	-- defined cpu or gpu device
	- optimizer	-- optimizer
	- traDat 	-- data for training
	- traLab 	-- label for training, single number
	- criterion	-- objective function for optimization 			(default cross entropy)
	- numBatch 	-- number of samples in a batch				(default 64)
	- numEpoch 	-- number of epoch for training				(default 200)
	- valDat 	-- validation data explicitly given 
	- valLab 	-- validation label explicitly given
	- lrChg 	-- is learning rate changable				(default False)
	- earlyStop 	-- apply earlyStop or not 				(default False)
	- numPatient	-- patient number for early stop 			(default 20)
    Output:
	- model 	-- trained pytorch model
	- traLoss 	-- mean loss of batches
	- traArry 	-- training arrucacy for all epoch
	- valLoss 	-- validation loss for all epoch
	- valArry 	-- validation accuracy for all epoch

    '''
    model.train()
    # initialize outputs 
    traLoss = np.zeros((numEpoch))
    traArry = np.zeros((numEpoch))
    valLoss = np.zeros((numEpoch))
    valArry = np.zeros((numEpoch))


    # training
    for epoch in range(numEpoch):
        running_loss = 0.0
        correct = 0.0
        for t in tqdm(range(np.int(np.ceil(traDat.shape[0]/numBatch)))):
            # zero the parameter gradients for each minibatch
            optimizer.zero_grad()

            # batch organization
            if t == np.int(np.floor(traDat.shape[0]/numBatch)):
                idx = np.arange(t*numBatch,traDat.shape[0])
            else:
                idx = np.arange(t*numBatch,(t+1)*numBatch)
            inDat = traDat[idx,:,:,:].cuda(device)
            inLab = traLab[idx].cuda(device)

            # forward
            outputs = model(inDat)
            # loss
            loss = criterion(outputs, inLab)
            # backward
            loss.backward()
            # optimize
            optimizer.step()
            # training loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(inLab).sum().item()

        # training loss and accuracy
        traLoss[epoch] = running_loss/np.ceil(traDat.shape[0]/numBatch)
        traArry[epoch] = correct/traDat.shape[0]*100
        # validation loss and accuracy
        _, valLoss[epoch], valArry[epoch] = test(model,device,valDat,valLab,criterion,numBatch)
        # print
        print('epoch %d: training loss: %.4f; training acc: %.2f; validation loss: %.4f; validation acc: %.2f' % (epoch+1, traLoss[epoch], traArry[epoch],valLoss[epoch],valArry[epoch]))

    print(' --- training done --- ')
    return model,traLoss,traArry,valLoss,valArry





