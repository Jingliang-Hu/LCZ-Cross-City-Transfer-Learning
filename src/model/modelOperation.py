import torch 
import torch.nn as nn 
import torch.optim as optim 
import sys
sys.path.insert(0,'../io')
import dataReader
import numpy as np
from tqdm import tqdm


def weightOfConsistentLoss(currentEpoch, maxEpoch):
    '''
    This function changes the weight of consistent loss for mean teacher model
    '''
    tmp = np.clip(currentEpoch,0,maxEpoch)
    return np.exp(-5*np.square(1-np.float(tmp)/np.float(maxEpoch)))


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



def meanTeacher_Train(student,teacher,device,traDat,traLab,optimizer,valDat,valLab,classification_loss,consistency_loss,numBatch,numEpoch,upperEpoch=80):
    student.train()
    teacher.train()
    # initialize outputs
    traTLoss = np.zeros((numEpoch))
    traTArry = np.zeros((numEpoch))
    valTLoss = np.zeros((numEpoch))
    valTArry = np.zeros((numEpoch))

    traSLoss = np.zeros((numEpoch))
    traSArry = np.zeros((numEpoch))
    valSLoss = np.zeros((numEpoch))
    valSArry = np.zeros((numEpoch))


    # training
    for epoch in range(numEpoch):
        running_loss_t = 0.0
        running_loss_s = 0.0
        correct_t = 0.0
        correct_s = 0.0
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
            outputs_s = student(inDat)
            outputs_t = teacher(inDat)

            # mask for training data that have labels
            labIdx = inLab>0

            # losses
            # classification_loss = nn.CrossEntropyLoss()
            # consistency_loss = nn.MSELoss()
            classLoss = classification_loss(outputs_s[labIdx,:],inLab[labIdx])
            consisLoss = consistency_loss(outputs_s,outputs_t)

            consistLossWeight = weightOfConsistentLoss(epoch, upperEpoch)
            loss = classLoss + consistLossWeight*consisLoss

            # backward
            loss.backward()
            # optimize
            # optimizer = optim.SGD(student.parameters(), lr=learnRate, momentum=momentum)
            optimizer.step()

            # training loss and accuracy
            teacherError = classification_loss(outputs_t[labIdx,:],inLab[labIdx])

            running_loss_s += classLoss.item()
            running_loss_t += teacherError.item()

            _, predicted_s = torch.max(outputs_s.data[labIdx,:], 1)
            _, predicted_t = torch.max(outputs_t.data[labIdx,:], 1)

            correct_s += predicted_s.eq(inLab[labIdx]).sum().item()
            correct_t += predicted_t.eq(inLab[labIdx]).sum().item()

            # update teacher model
            alpha=0.99
            alpha = min(1 - 1 / (epoch + 1), alpha)
            for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
                teacher_param.data.mul_(alpha).add_(1 - alpha, student_param.data)




        # training loss and accuracy
        traSLoss[epoch] = running_loss_s/np.ceil(traDat.shape[0]/numBatch)
        traTLoss[epoch] = running_loss_t/np.ceil(traDat.shape[0]/numBatch)

        traSArry[epoch] = correct_s/traDat.shape[0]*100
        traTArry[epoch] = correct_t/traDat.shape[0]*100

        # validation loss and accuracy
        _, valSLoss[epoch], valSArry[epoch] = test(student,device,valDat,valLab,classification_loss,512)
        _, valTLoss[epoch], valTArry[epoch] = test(teacher,device,valDat,valLab,classification_loss,512)

        # print
        print('Epoch %d:' % (epoch+1))
        print('Student model: training loss: %.4f; training acc: %.2f; validation loss: %.4f; validation acc: %.2f' % (traSLoss[epoch], traSArry[epoch],valSLoss[epoch],valSArry[epoch]))
        print('Teacher model: training loss: %.4f; training acc: %.2f; validation loss: %.4f; validation acc: %.2f' % (traTLoss[epoch], traTArry[epoch],valTLoss[epoch],valTArry[epoch]))


    print(' --- training done --- ')
    return student,teacher,traSLoss,traSArry,valSLoss,valSArry,traTLoss,traTArry,valTLoss,valTArry













