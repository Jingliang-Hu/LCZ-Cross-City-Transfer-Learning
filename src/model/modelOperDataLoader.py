import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import h5py
import torch 
import torch.nn as nn 
from torch.nn import functional as F
import torch.optim as optim 
import sys
sys.path.insert(0,'../io')
import numpy as np
from tqdm import tqdm
from itertools import product
from torch.nn import functional as F
import random

class ImageAugmentation (object):
    def __init__(self, hflip, gaussian_noise_std=0.0):
        self.hflip = hflip        
        self.gaussian_noise_std = gaussian_noise_std
    def augment(self, X):
        X = X.clone()
        if self.hflip:
            X_flip = torch.flip(X,[2])
            idx_hflip = np.random.binomial(1, 0.5, size=(len(X),))
            X[idx_hflip,:,:,:] = X_flip[idx_hflip,:,:,:]
            del X_flip
        if self.gaussian_noise_std > 0.0:
            X += torch.empty(X.shape).normal_(mean=0,std=self.gaussian_noise_std)
        return X



def weightOfConsistentLossTrAcc(trAcc):
    '''
    This function calcuates the weight of the consistent loss based on the training accuracy of the previous epoch.
    '''
    return np.exp(6.93147*(trAcc-1))


def weightOfConsistentLoss(currentEpoch, maxEpoch):
    '''
    This function changes the weight of consistent loss for mean teacher model
    '''
    tmp = np.clip(currentEpoch,0,maxEpoch)
    return np.exp(-5*np.square(1-np.float(tmp)/np.float(maxEpoch)))

def calculateEMAAlpha(currentEpoch, maxEpoch,maxAlpha):
    '''
    This function changes the weight of consistent loss for mean teacher model
    '''
    tmp = np.clip(currentEpoch,0,maxEpoch)
    return np.exp(-5*np.square(1-np.float(tmp)/np.float(maxEpoch)))*maxAlpha


def confusionMatrix(model,device,testDataLoad,criterion):
    model.eval()
    nb_class = testDataLoad.dataset.label.shape[1]
    confusion_matrix = np.zeros((nb_class,nb_class))
    with torch.no_grad():
        for i_batch, sample in enumerate(testDataLoad):
            inDat = sample['data'].to(device,dtype=torch.float)
            inLab = sample['label'].to(device,dtype=torch.float)
            inLab = torch.max(inLab,1)[1]
            # predicting
            output = model(inDat)
            _, pred = torch.max(output.data, 1) # get the index of the max log-probability 
            for l, p in zip(inLab.view(-1), pred.view(-1)):
                confusion_matrix[l.long(), p.long()] += 1

    pa = np.diagonal(confusion_matrix)/np.sum(confusion_matrix,1)
    ua = np.diagonal(confusion_matrix)/np.sum(confusion_matrix,0)
    oa = np.trace(confusion_matrix)/np.sum(confusion_matrix)
    aa = np.sum(pa[~np.isnan(pa)])/np.sum(~np.isnan(pa))
    # kappa coefficient
    po = oa
    pe = np.sum(np.sum(confusion_matrix,0)*np.sum(confusion_matrix,1))/np.square(np.sum(confusion_matrix))
    ka = (po-pe)/(1-pe)
    print("Number of test samples: %d" %(len(testDataLoad.dataset)))
    print("Overall accuracy: %.4f; Average accuracy: %.4f " %(oa, aa))
    print("Producer accuracy: ")
    print(pa)
    print("User accuracy: " )
    print(ua)
    return confusion_matrix,oa,aa,ka,pa,ua


def multiStreamConfusionMatrix(model_list,device,testDataLoad,criterion):
    for i in range(len(model_list)):
        model_list[i].eval()

    nb_class = testDataLoad.dataset.label.shape[1]
    confusion_matrix = np.zeros((nb_class,nb_class))
    with torch.no_grad():
        for i_batch, sample in enumerate(testDataLoad):
            inDat = sample['data'].to(device,dtype=torch.float)
            inLab = sample['label'].to(device,dtype=torch.float)
            inLab = torch.max(inLab,1)[1]
            # predicting
            output = model_list[0](inDat)
            for j in range(1,len(model_list)):
                output += model_list[j](inDat)

            _, pred = torch.max(output.data, 1) # get the index of the max log-probability
            for l, p in zip(inLab.view(-1), pred.view(-1)):
                confusion_matrix[l.long(), p.long()] += 1

    pa = np.diagonal(confusion_matrix)/np.sum(confusion_matrix,1)
    ua = np.diagonal(confusion_matrix)/np.sum(confusion_matrix,0)
    oa = np.trace(confusion_matrix)/np.sum(confusion_matrix)
    aa = np.sum(pa[~np.isnan(pa)])/np.sum(~np.isnan(pa))
    # kappa coefficient
    po = oa
    pe = np.sum(np.sum(confusion_matrix,0)*np.sum(confusion_matrix,1))/np.square(np.sum(confusion_matrix))
    ka = (po-pe)/(1-pe)
    print("Number of test samples: %d" %(len(testDataLoad.dataset)))
    print("Overall accuracy: %.4f; Average accuracy: %.4f " %(oa, aa))
    print("Producer accuracy: ")
    print(pa)
    print("User accuracy: " )
    print(ua)
    return confusion_matrix,oa,aa,ka,pa,ua

def semiFusionConfusionMatrix(model_list,device,testDataLoad_s1,testDataLoad_s2,criterion):
    for i in range(len(model_list)):
        model_list[i].eval()

    nb_class = testDataLoad_s1.dataset.label.shape[1]
    confusion_matrix = np.zeros((nb_class,nb_class))
    label_consist_check = 0
    with torch.no_grad():
        for i_batch, sample in enumerate(zip(testDataLoad_s1,testDataLoad_s2)):
            inDat_s1 = sample[0]['data'].to(device,dtype=torch.float)
            inLab_s1 = sample[0]['label'].to(device,dtype=torch.float)
            inLab_s1 = torch.max(inLab_s1,1)[1]
            inDat_s2 = sample[1]['data'].to(device,dtype=torch.float)
            inLab_s2 = sample[1]['label'].to(device,dtype=torch.float)
            inLab_s2 = torch.max(inLab_s2,1)[1]
            label_consist_check+=torch.sum(inLab_s1-inLab_s2)

            # predicting
            output = model_list[0](inDat_s1)
            output += model_list[1](inDat_s1)
            output += model_list[2](inDat_s2)
            output += model_list[3](inDat_s2)

            _, pred = torch.max(output.data, 1) # get the index of the max log-probability
            for l, p in zip(inLab_s1.view(-1), pred.view(-1)):
                confusion_matrix[l.long(), p.long()] += 1

    print('label consistent check: %d' %(label_consist_check))

    pa = np.diagonal(confusion_matrix)/np.sum(confusion_matrix,1)
    ua = np.diagonal(confusion_matrix)/np.sum(confusion_matrix,0)
    oa = np.trace(confusion_matrix)/np.sum(confusion_matrix)
    aa = np.sum(pa[~np.isnan(pa)])/np.sum(~np.isnan(pa))
    # kappa coefficient
    po = oa
    pe = np.sum(np.sum(confusion_matrix,0)*np.sum(confusion_matrix,1))/np.square(np.sum(confusion_matrix))
    ka = (po-pe)/(1-pe)
    print("Number of test samples: %d" %(len(testDataLoad_s1.dataset)))
    print("Overall accuracy: %.4f; Average accuracy: %.4f " %(oa, aa))
    print("Producer accuracy: ")
    print(pa)
    print("User accuracy: " )
    print(ua)
    return confusion_matrix,oa,aa,ka,pa,ua

def prediction_4_mapping_gpu(model,data_patches,device):
    model.eval()
    model = model.float()
    model = model.to(device)
    data_patches = np.transpose(data_patches,(0,3,1,2))
    pred = np.zeros((data_patches.shape[0]))
    batch_num = 100
    batch_total = np.ceil(data_patches.shape[0]/batch_num)
    for i in tqdm(range(batch_total.astype(int))):
        if i == batch_total-1:
            data_batch = torch.from_numpy(data_patches[i*batch_num:,:,:,:]).float()
            output = model(data_batch.to(device))
            _, predTmp = torch.max(output.data, 1)
            pred[i*batch_num:] = predTmp.cpu().numpy()
        else:
            data_batch = torch.from_numpy(data_patches[i*batch_num:(i+1)*batch_num,:,:,:]).float()
            output = model(data_batch.to(device))
            _, predTmp = torch.max(output.data, 1)
            pred[i*batch_num:(i+1)*batch_num] = predTmp.cpu().numpy()

    return pred

def prediction_softmax_gpu(model,data_patches,device):
    model.eval()
    model = model.float()
    model = model.to(device)
    data_patches = np.transpose(data_patches,(0,3,1,2))
    pred = np.zeros((data_patches.shape[0],17))
    batch_num = 100
    batch_total = np.ceil(data_patches.shape[0]/batch_num)
    softmax_calculater = nn.Softmax(dim = 1)
    for i in tqdm(range(batch_total.astype(int))):
        if i == batch_total-1:
            data_batch = torch.from_numpy(data_patches[i*batch_num:,:,:,:]).float()
            output = model(data_batch.to(device))
            pred[i*batch_num:] = softmax_calculater(output).cpu().detach().numpy()
        else:
            data_batch = torch.from_numpy(data_patches[i*batch_num:(i+1)*batch_num,:,:,:]).float()
            output = model(data_batch.to(device))
            pred[i*batch_num:(i+1)*batch_num] = softmax_calculater(output).cpu().detach().numpy()

    return pred



def prediction_4_mapping(model,data_patches):
    model.eval()
    model = model.float()
    data_patches = np.transpose(data_patches,(0,3,1,2))
    pred = np.zeros((data_patches.shape[0]))
    batch_num = 100
    batch_total = np.ceil(data_patches.shape[0]/batch_num)

    for i in tqdm(range(batch_total.astype(int))):
        if i == batch_total-1:
            data_batch = torch.from_numpy(data_patches[i*batch_num:,:,:,:])
            output = model(data_batch.float())
            _, predTmp = torch.max(output.data, 1)
            pred[i*batch_num:] = predTmp.numpy()
        else:
            data_batch = torch.from_numpy(data_patches[i*batch_num:(i+1)*batch_num,:,:,:])
            output = model(data_batch.float())
            _, predTmp = torch.max(output.data, 1)
            pred[i*batch_num:(i+1)*batch_num] = predTmp.numpy()

    return pred
    

def test(model,device,valDataLoader,criterion):
    '''
    Input:
            - model         -- pytorch model
            - device        -- defined cpu or gpu device
            - valDataLoader -- pytorch dataloader for testing data
            - criterion     -- objective function for optimization                  (default cross entropy)
    Output:
            - pred          -- prediction of input data using model
            - testLoss      -- mean testing error of batches
            - accuracy      -- testing overall accuracy
    '''

    model.eval()
    try:
        nb_class = valDataLoader.dataset.label.shape[1]
    except:
        nb_class = valDataLoader.dataset.dataset.label.shape[1]
    confusion_matrix = np.zeros((nb_class,nb_class))
    
    nb_test_samples = len(valDataLoader.dataset)

    testLoss = 0.0
    accuracy = 0.0 # overall accuracy
    pred = np.zeros((nb_test_samples))
    batch_size = valDataLoader.batch_size
    with torch.no_grad():
        for i_batch, sample in enumerate(valDataLoader):
            inDat = sample['data'].to(device,dtype=torch.float)
            inLab = sample['label'].to(device,dtype=torch.float)
            inLab = torch.max(inLab,1)[1]
            # predicting
            output = model(inDat) 
            # prediction error
            loss = criterion(output, inLab)
            testLoss += loss.item()*inDat.size(0)

            _, predTmp = torch.max(output.data, 1) # get the index of the max log-probability 
            for l, p in zip(inLab.view(-1), predTmp.view(-1)):
                confusion_matrix[l.long(), p.long()] += 1

            if batch_size==np.squeeze(predTmp.cpu().numpy()).shape[0]:
                pred[i_batch*batch_size:(i_batch+1)*batch_size] = np.squeeze(predTmp.cpu().numpy())
            else:
                pred[i_batch*batch_size:] = np.squeeze(predTmp.cpu().numpy())

    testLoss = testLoss/nb_test_samples 
    accuracy = np.trace(confusion_matrix)/nb_test_samples*100
    pa = np.diagonal(confusion_matrix)/np.sum(confusion_matrix,1)
    averacc = np.sum(pa[~np.isnan(pa)])/np.sum(~np.isnan(pa))*100


    return pred, testLoss, accuracy, averacc


def train(model, device, optimizer, traDataLoader, criterion, numEpoch, valDataLoader):
    '''
    Input:
	- model 	-- pytorch model
	- device 	-- defined cpu or gpu device
	- optimizer	-- optimizer
	- traDataLoader	-- pytorch dataloader for training
	- criterion	-- objective function for optimization 			(default cross entropy)
	- numEpoch 	-- number of epoch for training				(default 200)
	- valDataLoader	-- pytorch dataloader for validation data  
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
    valAver = np.zeros((numEpoch))
    print(" ----------------------------------------- ")
    num_batch = len(traDataLoader) 
    nb_train_sample = len(traDataLoader.dataset)
    # training
    for epoch in range(numEpoch):
        running_loss = 0.0
        correct = 0.0
        print("Number of batches (%d in total): " % (num_batch)) 
        for i_batch, sample in tqdm(enumerate(traDataLoader)):
            inDat = sample['data'].to(device,dtype=torch.float)
            inLab = sample['label'].to(device,dtype=torch.float)
            inLab = torch.max(inLab,1)[1]
            # zero the parameter gradients for each minibatch
            optimizer.zero_grad()
            # forward
            outputs = model(inDat)
            # loss
            loss = criterion(outputs, inLab)
            # backward
            loss.backward()
            # optimize
            optimizer.step()
            # statistics
            running_loss += loss.item() * inDat.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(inLab).sum().item()


        # training loss and accuracy
        traLoss[epoch] = running_loss/nb_train_sample
        traArry[epoch] = correct/nb_train_sample*100
        # validation loss and accuracy
        _, valLoss[epoch], valArry[epoch], valAver[epoch] = test(model,device,valDataLoader,criterion)
        # print
        print('epoch %d: training loss: %.4f; training acc: %.2f; validation loss: %.4f; validation acc: %.2f; validation average acc: %.2f' % (epoch+1, traLoss[epoch], traArry[epoch],valLoss[epoch],valArry[epoch],valAver[epoch]))

    print(' --- training done --- ')
    return model,traLoss,traArry,valLoss,valArry,valAver

def train_valid_early_stop(model, model_out_dir, device, optimizer, traDataLoader, criterion, numEpoch, valDataLoader, patient=10):
    '''
    Input:
        - model         -- pytorch model
        - model_out_dir -- directory to save trained model
        - device        -- defined cpu or gpu device
        - optimizer     -- optimizer
        - traDataLoader -- pytorch dataloader for training
        - criterion     -- objective function for optimization                  (default cross entropy)
        - numEpoch      -- number of epoch for training                         (default 200)
        - valDataLoader -- pytorch dataloader for validation data
        - patient       -- patient number of epochs
    Output:
        - model         -- trained pytorch model
        - traLoss       -- mean loss of batches
        - traArry       -- training arrucacy for all epoch
        - valLoss       -- validation loss for all epoch
        - valArry       -- validation accuracy for all epoch
    '''
    model.train()
    # initialize outputs
    traLoss = np.zeros((numEpoch))
    traArry = np.zeros((numEpoch))
    valLoss = np.zeros((numEpoch))
    valArry = np.zeros((numEpoch))
    valAver = np.zeros((numEpoch))
    print(" ----------------------------------------- ")
    num_batch = len(traDataLoader)
    nb_train_sample = len(traDataLoader.dataset)
    # training
    for epoch in range(numEpoch):
        running_loss = 0.0
        correct = 0.0
        print("Number of batches (%d in total): " % (num_batch))
        for i_batch, sample in tqdm(enumerate(traDataLoader)):
            inDat = sample['data'].to(device,dtype=torch.float)
            inLab = sample['label'].to(device,dtype=torch.float)
            inLab = torch.max(inLab,1)[1]
            # zero the parameter gradients for each minibatch
            optimizer.zero_grad()
            # forward
            outputs = model(inDat)
            # loss
            loss = criterion(outputs, inLab)
            # backward
            loss.backward()
            # optimize
            optimizer.step()
            # statistics
            running_loss += loss.item() * inDat.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(inLab).sum().item()
        # training loss and accuracy
        traLoss[epoch] = running_loss/nb_train_sample
        traArry[epoch] = correct/nb_train_sample*100
        # validation loss and accuracy
        _, valLoss[epoch], valArry[epoch], valAver[epoch] = test(model,device,valDataLoader,criterion)
        print('epoch %d: training loss: %.4f; training acc: %.2f; validation loss: %.4f; validation acc: %.2f; validation average acc: %.2f' % (epoch+1, traLoss[epoch], traArry[epoch],valLoss[epoch],valArry[epoch],valAver[epoch]))
        # save the best model
        model_save_flag = save_best_model_func(model, 1, model_out_dir, epoch, valLoss)
        if epoch-patient<0:
            continue
        elif (valLoss[epoch-patient]<valLoss[epoch-patient+1:epoch+1]).all():
            print('Early stopping patient (%d) reached' % (patient))
            break
    print(' --- training done --- ')
    return traLoss,traArry,valLoss,valArry,valAver




def save_best_model_func(model, save_best_model, model_dir, epoch, val_loss):
    if epoch == 0 and save_best_model:
        torch.save(model.state_dict(), model_dir)
        print('The model after %d th epoch training has the lowest validation loss and saved' % (epoch+1))        
        return 1
    elif val_loss[epoch] < val_loss[:epoch].min() and save_best_model:
        torch.save(model.state_dict(), model_dir)
        print('The model after %d th epoch training has the lowest validation loss and saved' % (epoch+1))
        return 1
    return 0



def train_DataAug(model, device, optimizer, traDataLoader, criterion, numEpoch, valDataLoader, aug_flip, aug_noise_std):
    '''
    Input:
        - model         -- pytorch model
        - device        -- defined cpu or gpu device
        - optimizer     -- optimizer
        - traDataLoader -- pytorch dataloader for training
        - criterion     -- objective function for optimization                  (default cross entropy)
        - numEpoch      -- number of epoch for training                         (default 200)
        - valDataLoader -- pytorch dataloader for validation data
    Output:
        - model         -- trained pytorch model
        - traLoss       -- mean loss of batches
        - traArry       -- training arrucacy for all epoch
        - valLoss       -- validation loss for all epoch
        - valArry       -- validation accuracy for all epoch

    '''
    model.train()
    # initialize outputs
    traLoss = np.zeros((numEpoch))
    traArry = np.zeros((numEpoch))
    valLoss = np.zeros((numEpoch))
    valArry = np.zeros((numEpoch))
    valAver = np.zeros((numEpoch))
    print(" ----------------------------------------- ")
    num_batch = len(traDataLoader)
    nb_train_sample = len(traDataLoader.dataset)
    aug = ImageAugmentation(aug_flip, aug_noise_std)
    # training
    for epoch in range(numEpoch):
        running_loss = 0.0
        correct = 0.0
        print("Number of batches (%d in total): " % (num_batch))
        for i_batch, sample in tqdm(enumerate(traDataLoader)):
            inDat = sample['data'].to(dtype=torch.float)
            inDat = aug.augment(inDat).to(device)
            inLab = sample['label'].to(device,dtype=torch.float)
            inLab = torch.max(inLab,1)[1]
            # zero the parameter gradients for each minibatch
            optimizer.zero_grad()
            # forward
            outputs = model(inDat)
            # loss
            loss = criterion(outputs, inLab)
            # backward
            loss.backward()
            # optimize
            optimizer.step()
            # statistics
            running_loss += loss.item() * inDat.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(inLab).sum().item()


        # training loss and accuracy
        traLoss[epoch] = running_loss/nb_train_sample
        traArry[epoch] = correct/nb_train_sample*100
        # validation loss and accuracy
        _, valLoss[epoch], valArry[epoch], valAver[epoch] = test(model,device,valDataLoader,criterion)
        # print
        print('epoch %d: training loss: %.4f; training acc: %.2f; validation loss: %.4f; validation acc: %.2f; validation average acc: %.2f' % (epoch+1, traLoss[epoch], traArry[epoch],valLoss[epoch],valArry[epoch],valAver[epoch]))

    print(' --- training done --- ')
    return model,traLoss,traArry,valLoss,valArry,valAver


def meanTeacher_Train(student,teacher,device,traDat,traLab,optimizer,valDat,valLab,classification_loss,consistency_loss,numBatch,numEpoch,alpha,upperEpoch=80):
    # this function trains the mean teacher model: the training data of target and source domains are mixed together.

    student.train()
    teacher.train()
    teacherUpdateCount = 0
    # initialize outputs
    traTLoss = np.zeros((numEpoch))
    traTArry = np.zeros((numEpoch))
    valTLoss = np.zeros((numEpoch))
    valTArry = np.zeros((numEpoch))

    traSLoss = np.zeros((numEpoch))
    traSArry = np.zeros((numEpoch))
    valSLoss = np.zeros((numEpoch))
    valSArry = np.zeros((numEpoch))

    consistentLoss = np.zeros((numEpoch))


    # training
    for epoch in range(numEpoch):
        running_loss_t = 0.0
        running_loss_s = 0.0
        running_loss_c = 0.0
        correct_t = 0.0
        correct_s = 0.0

        # changing the weight of consistent loss, based on the epoch
        consistLossWeight = weightOfConsistentLoss(epoch, upperEpoch)

        # alpha value for updating teacher model, weight in exponential moving average (EMA)
        # alpha=0.99
        # alpha = min(1 - 1 / (epoch + 1), alpha)
        

        for t in tqdm(range(np.int(np.ceil(traDat.shape[0]/numBatch)))):
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

            loss = classLoss + consistLossWeight*consisLoss

            # zero the parameter gradients for each minibatch
            optimizer.zero_grad()
            # backward
            loss.backward()
            # optimize
            # optimizer = optim.SGD(student.parameters(), lr=learnRate, momentum=momentum)
            optimizer.step()

            # training loss and accuracy
            teacherError = classification_loss(outputs_t[labIdx,:],inLab[labIdx])

            running_loss_s += classLoss.item()
            running_loss_t += teacherError.item()
            running_loss_c += consisLoss.item()

            _, predicted_s = torch.max(outputs_s.data[labIdx,:], 1)
            _, predicted_t = torch.max(outputs_t.data[labIdx,:], 1)

            correct_s += predicted_s.eq(inLab[labIdx]).sum().item()
            correct_t += predicted_t.eq(inLab[labIdx]).sum().item()

        # training loss and accuracy
        traSLoss[epoch] = running_loss_s/np.ceil(traDat.shape[0]/numBatch)
        traTLoss[epoch] = running_loss_t/np.ceil(traDat.shape[0]/numBatch)
        consistentLoss[epoch] = running_loss_c/np.ceil(traDat.shape[0]/numBatch)
        traSArry[epoch] = correct_s/traDat.shape[0]*100
        traTArry[epoch] = correct_t/traDat.shape[0]*100



        # update teacher model
        if traSArry[epoch]>75:
            if teacherUpdateCount == 0:
                for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
                    teacher_param.data = student_param.data
            else:
                for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
                    teacher_param.data.mul_(alpha).add_(1 - alpha, student_param.data)


        # validation loss and accuracy
        _, valSLoss[epoch], valSArry[epoch] = test(student,device,valDat,valLab,classification_loss,512)
        _, valTLoss[epoch], valTArry[epoch] = test(teacher,device,valDat,valLab,classification_loss,512)

        # print
        print('Epoch %d:' % (epoch+1))
        print('Weight of consistent loss: %.6f' % (consistLossWeight))
        print('Alpha in EMA: %.6f' % (alpha))
        print('Student model: training loss: %.4f; training acc: %.2f; validation loss: %.4f; validation acc: %.2f' % (traSLoss[epoch], traSArry[epoch],valSLoss[epoch],valSArry[epoch]))
        print('Teacher model: training loss: %.4f; training acc: %.2f; validation loss: %.4f; validation acc: %.2f' % (traTLoss[epoch], traTArry[epoch],valTLoss[epoch],valTArry[epoch]))


    print(' --- training done --- ')
    return student,teacher,traSLoss,traSArry,valSLoss,valSArry,traTLoss,traTArry,valTLoss,valTArry


def train_multi_ensemble(students, data_loaders, optimizers, device, classification_loss, consistency_loss, numEpoch):
    nb_streams = len(students)
    nb_train_samples = len(data_loaders[0].dataset)
    nb_test_samples = len(data_loaders[nb_streams].dataset)
    if nb_train_samples<nb_test_samples:
        nb_samples = nb_train_samples
    else:
        nb_samples = nb_test_samples

    nb_batches = len(data_loaders[0])
    print('The number of samples in source domain: % d' % (nb_train_samples))
    print('The number of samples in target domain: % d' % (nb_test_samples))

    cla_loss_train = np.zeros((numEpoch, nb_streams))
    con_loss_train = np.zeros((numEpoch, nb_streams))
    cla_acc_train = np.zeros((numEpoch, nb_streams))
    cla_loss_test = np.zeros((numEpoch, nb_streams))
    cla_acc_test = np.zeros((numEpoch, nb_streams))
    cla_averacc_test = np.zeros((numEpoch, nb_streams))
    for i in range(nb_streams):
        students[i].train()

    # start training
    print(" ----------------------------------------- ")
    # training
    for epoch in range(numEpoch):
        print('+................................................+')
        print('Epoch %d:' % (epoch+1))
        running_loss_class = np.zeros((nb_streams))
        running_loss_consis = np.zeros((nb_streams))
        correct = np.zeros((nb_streams))
        # iterations in batches
        print("Number of batches (%d in total): " % (nb_batches))
        for i, data in tqdm(enumerate(zip(*data_loaders))):
            targetDat = data[nb_streams]['data'].to(device,dtype=torch.float)
            sourceLab = []
            stream_out_source = []
            stream_out_target = []
            stream_out_target_prob = []
            class_loss = []
            target_aver_prob = 0

            for j in range(nb_streams):
                sourceDat = data[j]['data'].to(device,dtype=torch.float)
                sourceLab.append(data[j]['label'].to(device,dtype=torch.float))
                sourceLab[j] = torch.max(sourceLab[j],1)[1]
                stream_out_source.append(students[j](sourceDat))
                stream_out_target.append(students[j](targetDat))
                stream_out_target_prob.append(F.softmax(stream_out_target[j],dim=1))
                class_loss.append(classification_loss(stream_out_source[j],sourceLab[j]))

            # consistent loss
            target_aver_prob = torch.zeros(stream_out_target_prob[0].shape).to(device,dtype=torch.float)
            for x in stream_out_target_prob:
                target_aver_prob += x
            target_aver_prob = target_aver_prob/len(students)

            consis_loss = []
            loss = []

            for j in range(nb_streams):
                consis_loss.append(consistency_loss(stream_out_target_prob[j],target_aver_prob))
                loss.append(class_loss[j]+consis_loss[j])

                optimizers[j].zero_grad()
                if j == nb_streams:
                    loss[j].backward()
                else:
                    loss[j].backward(retain_graph=True)
                optimizers[j].step()

                running_loss_class[j] += class_loss[j].item()*sourceDat.size(0)
                running_loss_consis[j] += consis_loss[j].item()*sourceDat.size(0)
                _, predicted_s1 = torch.max(stream_out_source[j].data, 1)
                correct[j] += predicted_s1.eq(sourceLab[j]).sum().item()

        for j in range(nb_streams):
            cla_loss_train[epoch,j] = running_loss_class[j]/nb_samples
            con_loss_train[epoch,j] = running_loss_consis[j]/nb_samples
            cla_acc_train[epoch,j] = correct[j]/nb_samples*100
            _, cla_loss_test[epoch, j], cla_acc_test[epoch, j], cla_averacc_test[epoch, j] = test(students[j],device,data_loaders[nb_streams],classification_loss) 
            # print
            print('Stream %d: train class loss: %.4f; train consis loss: %.4f; train acc: %.4f; test acc: %.4f; test averacc: %.4f' % (j, cla_loss_train[epoch,j], con_loss_train[epoch,j], cla_acc_train[epoch,j], cla_acc_test[epoch,j],cla_averacc_test[epoch,j]))



    print(' --- training done --- ')
    return students, cla_loss_train, cla_acc_train, cla_loss_test, cla_acc_test, con_loss_train, cla_averacc_test


def prediction_dual_fusion_student(model, device, data_loaders, criterion):
    model[0].eval()
    model[1].eval()

    nb_class = data_loaders[0].dataset.label.shape[1]
    confusion_matrix = np.zeros((nb_class,nb_class))
    label_consist_check = 0

    with torch.no_grad():
        for i_batch, sample in enumerate(zip(data_loaders[0], data_loaders[1])):
            inDat_s1 = sample[0]['data'].to(device,dtype=torch.float)
            inLab_s1 = sample[0]['label'].to(device,dtype=torch.float)
            inLab_s1 = torch.max(inLab_s1,1)[1]
            inDat_s2 = sample[1]['data'].to(device,dtype=torch.float)
            inLab_s2 = sample[1]['label'].to(device,dtype=torch.float)
            inLab_s2 = torch.max(inLab_s2,1)[1]
            label_consist_check+=torch.sum(inLab_s1-inLab_s2)

            # predicting
            output = model[0](inDat_s1, inDat_s2)
            output += model[1](inDat_s1, inDat_s2)

            _, pred = torch.max(output.data, 1) # get the index of the max log-probability
            for l, p in zip(inLab_s1.view(-1), pred.view(-1)):
                confusion_matrix[l.long(), p.long()] += 1

    print('label consistent check: %d' %(label_consist_check))

    pa = np.diagonal(confusion_matrix)/np.sum(confusion_matrix,1)
    ua = np.diagonal(confusion_matrix)/np.sum(confusion_matrix,0)
    oa = np.trace(confusion_matrix)/np.sum(confusion_matrix)
    aa = np.sum(pa[~np.isnan(pa)])/np.sum(~np.isnan(pa))
    # kappa coefficient
    po = oa
    pe = np.sum(np.sum(confusion_matrix,0)*np.sum(confusion_matrix,1))/np.square(np.sum(confusion_matrix))
    ka = (po-pe)/(1-pe)
    print("Number of test samples: %d" %(len(data_loaders[0].dataset)))
    print("Overall accuracy: %.4f; Average accuracy: %.4f " %(oa, aa))
    print("Producer accuracy: ")
    print(pa)
    print("User accuracy: " )
    print(ua)
    return confusion_matrix,oa,aa,ka,pa,ua


def train_multi_fusion(model, source_data, target_data, optimizers, device, classification_loss, consistency_loss,numEpoch, batch_number):
    nb_train_samples = source_data.dat_s1.shape[0]
    nb_test_samples = target_data.dat_s1.shape[0]
    print('The number of samples in source domain: % d' % (nb_train_samples))
    print('The number of samples in target domain: % d' % (nb_test_samples))
    if nb_test_samples<nb_train_samples:
        nb_samples = nb_test_samples
    else:
        nb_samples = nb_train_samples
 
    nbStreams = len(model)
    tra_loss = [np.zeros((numEpoch))]*nbStreams
    tra_arry = [np.zeros((numEpoch))]*nbStreams
    tra_consis_loss = [np.zeros((numEpoch))]*nbStreams
    val_loss = [np.zeros((numEpoch))]*nbStreams
    val_arry = [np.zeros((numEpoch))]*nbStreams
    val_aver = [np.zeros((numEpoch))]*nbStreams

    for i in range(0,nbStreams):
        model[i].train()
    # start training

    # for calculating testing accuracy at each epoch 
    target_data_loader = torch.utils.data.DataLoader(target_data, batch_size=batch_number, shuffle=True)

    # training
    for epoch in range(numEpoch):
        print(" ----------------------------------------- ")
        print('Epoch %d:' % (epoch+1))

        # initialize lists to save variables for multiple streams
        running_loss_class = [0.0]*nbStreams
        running_loss_consis = [0.0]*nbStreams
        correct = [0.0]*nbStreams

        class_loss = [None]*nbStreams
        consis_loss = [None]*nbStreams
        total_loss = [None]*nbStreams

        source_output = [None]*nbStreams
        source_lab = [None]*nbStreams
        target_output_prob = [None]*nbStreams

        # iterations in batches
        nb_batches = np.ceil(nb_samples/batch_number)
        nb_samples_used = (batch_number*nb_batches)
        print("Number of batches (%d in total): " % (nb_batches))
        for idx in tqdm(range(nb_batches.astype(np.int))):
            target_random_idx = random.sample(range(0,nb_test_samples),batch_number)
            target_dat_1 = torch.from_numpy(target_data.dat_s1[target_random_idx,:,:,:].transpose((0, 3, 1, 2))).to(device,dtype=torch.float)
            target_dat_2 = torch.from_numpy(target_data.dat_s2[target_random_idx,:,:,:].transpose((0, 3, 1, 2))).to(device,dtype=torch.float)
            target_lab = torch.from_numpy(target_data.label[target_random_idx,:]).to(device,dtype=torch.float)
            target_lab = torch.max(target_lab,1)[1]

            for i in range(nbStreams):
                source_random_idx = random.sample(range(0,nb_train_samples),batch_number)
                source_dat_1 = torch.from_numpy(source_data.dat_s1[source_random_idx,:,:,:].transpose((0, 3, 1, 2))).to(device,dtype=torch.float)
                source_dat_2 = torch.from_numpy(source_data.dat_s2[source_random_idx,:,:,:].transpose((0, 3, 1, 2))).to(device,dtype=torch.float)
                source_lab[i] = torch.from_numpy(source_data.label[source_random_idx,:]).to(device,dtype=torch.float)
                source_lab[i] = torch.max(source_lab[i],1)[1]
                # forward
                source_output[i] = model[i](source_dat_1,source_dat_2)
                # classification loss
                class_loss[i] = classification_loss(source_output[i],source_lab[i])
                # inferences of data samples of target domain
                target_output_prob[i] = F.softmax(model[i](target_dat_1,target_dat_2),dim=1)

            target_aver_prob = torch.zeros(target_output_prob[0].shape).to(device,dtype=torch.float)
            for x in target_output_prob:
                target_aver_prob += x
            target_aver_prob = target_aver_prob/nbStreams

            for i in range(nbStreams):
                consis_loss[i] = consistency_loss(target_output_prob[i],target_aver_prob)
                total_loss[i] = consis_loss[i] + class_loss[i]
                optimizers[i].zero_grad()
                if i == nbStreams-1:
                    total_loss[i].backward()
                else:
                    total_loss[i].backward(retain_graph=True)
                optimizers[i].step()

                running_loss_class[i] += class_loss[i].item()*batch_number
                running_loss_consis[i] += consis_loss[i].item()*batch_number
                _, predicted = torch.max(source_output[i].data, 1)
                correct[i] += predicted.eq(source_lab[i]).sum().item()

        for i in range(nbStreams):
            tra_loss[i][epoch] = running_loss_class[i]/nb_samples_used
            tra_arry[i][epoch] = correct[i]/nb_samples_used
            tra_consis_loss[i][epoch] = running_loss_consis[i]/nb_samples_used
            _, val_loss[i][epoch], val_arry[i][epoch], val_aver[i][epoch], _, _, _, _, = test_feature_level_fusion_unified_loader(model[i], device, target_data_loader, classification_loss)
            print('sub_model %d: train class loss: %.4f; train consis loss: %.4f; train acc: %.4f; test acc: %.4f; test averacc: %.4f' % (i, tra_loss[i][epoch], tra_consis_loss[i][epoch], tra_arry[i][epoch], val_arry[i][epoch], val_aver[i][epoch]))
            # print('sub_model {}: train class loss: {}; train consis loss: {}; train acc: {}; test acc: {}; test averacc: {}'.format(i, tra_loss[i][epoch], tra_consis_loss[i][epoch], tra_arry[i][epoch], val_arry[i][epoch], val_aver[i][epoch]))

    return model, tra_loss, tra_arry, val_loss, val_arry, val_aver, tra_consis_loss



def prediction_multi_fusion(models, device, target_data, classification_loss):
    nbStreams = len(models)
    for i in range(nbStreams):
        models[i].eval()

    nb_class = target_data.dataset.label.shape[1]
    confusion_matrix = np.zeros((nb_class,nb_class))

    with torch.no_grad():
        for i_batch, sample in enumerate(target_data):
            inDat_s1 = sample['data1'].to(device,dtype=torch.float)
            inDat_s2 = sample['data2'].to(device,dtype=torch.float)
            inLab = sample['label'].to(device,dtype=torch.float)
            inLab = torch.max(inLab,1)[1]
            # predicting
            for i in range(nbStreams):
                if i == 0:
                    output = models[i](inDat_s1, inDat_s2)
                else:
                    output += models[i](inDat_s1, inDat_s2)

            _, pred = torch.max(output.data, 1) # get the index of the max log-probability
            for l, p in zip(inLab.view(-1), pred.view(-1)):
                confusion_matrix[l.long(), p.long()] += 1

    pa = np.diagonal(confusion_matrix)/np.sum(confusion_matrix,1)
    ua = np.diagonal(confusion_matrix)/np.sum(confusion_matrix,0)
    oa = np.trace(confusion_matrix)/np.sum(confusion_matrix)
    aa = np.sum(pa[~np.isnan(pa)])/np.sum(~np.isnan(pa))
    # kappa coefficient
    po = oa
    pe = np.sum(np.sum(confusion_matrix,0)*np.sum(confusion_matrix,1))/np.square(np.sum(confusion_matrix))
    ka = (po-pe)/(1-pe)
    print("Number of test samples: %d" %(len(target_data.dataset)))
    print("Overall accuracy: %.4f; Average accuracy: %.4f " %(oa, aa))
    print("Producer accuracy: ")
    print(pa)
    print("User accuracy: " )
    print(ua)
    return confusion_matrix,oa,aa,ka,pa,ua




def train_dual_fusion_student(model, data_loaders, optimizer, device, classification_loss, consistency_loss, numEpoch):
    nb_train_samples = len(data_loaders[0].dataset)
    nb_test_samples = len(data_loaders[4].dataset)
    nb_batches = len(data_loaders[0])
    print('The number of samples in source domain: % d' % (nb_train_samples))
    print('The number of samples in target domain: % d' % (nb_test_samples))
    if nb_test_samples<nb_train_samples:
        nb_samples = nb_test_samples
    else:
        nb_samples = nb_train_samples

    tra_Loss_s1 = np.zeros((numEpoch))
    tra_Loss_s2 = np.zeros((numEpoch))
    tra_Arry_s1 = np.zeros((numEpoch))
    tra_Arry_s2 = np.zeros((numEpoch))
    tra_Loss_consis = np.zeros((numEpoch))

    val_Loss_s1 = np.zeros((numEpoch))
    val_Loss_s2 = np.zeros((numEpoch))
    val_Arry_s1 = np.zeros((numEpoch))
    val_Arry_s2 = np.zeros((numEpoch))
    val_Aver_s1 = np.zeros((numEpoch))
    val_Aver_s2 = np.zeros((numEpoch))


    model[0].train()
    model[1].train()
    # start training
    print(" ----------------------------------------- ")
    # training
    for epoch in range(numEpoch):
        print('+................................................+')
        print('Epoch %d:' % (epoch+1))
        running_loss_s1 = 0.0
        running_loss_s2 = 0.0
        running_loss_c = 0.0
        correct_s1 = 0.0
        correct_s2 = 0.0

        # iterations in batches
        print("Number of batches (%d in total): " % (nb_batches))
        for i, data in tqdm(enumerate(zip(data_loaders[0],data_loaders[1],data_loaders[2],data_loaders[3],data_loaders[4],data_loaders[5]))):
            targetDat1 = data[4]['data'].to(device,dtype=torch.float)
            targetDat2 = data[5]['data'].to(device,dtype=torch.float)

            sourceDat1 = data[0]['data'].to(device,dtype=torch.float)
            sourceLab1 = data[0]['label'].to(device,dtype=torch.float)
            sourceLab1 = torch.max(sourceLab1,1)[1]

            sourceDat2 = data[2]['data'].to(device,dtype=torch.float)
            sourceLab2 = data[2]['label'].to(device,dtype=torch.float)
            sourceLab2 = torch.max(sourceLab2,1)[1]

            sourceDat3 = data[1]['data'].to(device,dtype=torch.float)
            sourceLab3 = data[1]['label'].to(device,dtype=torch.float)
            sourceLab3 = torch.max(sourceLab3,1)[1]

            sourceDat4 = data[3]['data'].to(device,dtype=torch.float)
            sourceLab4 = data[3]['label'].to(device,dtype=torch.float)
            sourceLab4 = torch.max(sourceLab4,1)[1]

            # zero the parameter gradients for each minibatch
            # forward
            source_output_s1 = model[0](sourceDat1,sourceDat2)
            source_output_s2 = model[1](sourceDat3,sourceDat4)
            class_loss_s1 = classification_loss(source_output_s1,sourceLab1)
            class_loss_s2 = classification_loss(source_output_s2,sourceLab3)

            target_output_prob_s1 = F.softmax(model[0](targetDat1,targetDat2),dim=1)
            target_output_prob_s2 = F.softmax(model[1](targetDat1,targetDat2),dim=1)
            consis_loss = consistency_loss(target_output_prob_s1,target_output_prob_s2)

            loss_s1 = class_loss_s1 + consis_loss
            loss_s2 = class_loss_s2 + consis_loss

            optimizer[0].zero_grad()
            loss_s1.backward(retain_graph=True)
            optimizer[0].step()

            optimizer[1].zero_grad()
            loss_s2.backward()
            optimizer[1].step()

            nb_samples_one_batch = sourceDat1.size(0)
            running_loss_s1 += class_loss_s1.item() * nb_samples_one_batch
            running_loss_s2 += class_loss_s2.item() * nb_samples_one_batch
            running_loss_c += consis_loss.item() * nb_samples_one_batch

            _, predicted_s1 = torch.max(source_output_s1.data, 1)
            _, predicted_s2 = torch.max(source_output_s2.data, 1)

            correct_s1 += predicted_s1.eq(sourceLab1).sum().item()
            correct_s2 += predicted_s2.eq(sourceLab3).sum().item()

        # training loss and accuracy
        tra_Loss_s1[epoch] = running_loss_s1/nb_samples
        tra_Loss_s2[epoch] = running_loss_s2/nb_samples
        tra_Loss_consis[epoch] = running_loss_c/nb_samples
        tra_Arry_s1[epoch] = correct_s1/nb_samples
        tra_Arry_s2[epoch] = correct_s2/nb_samples

        # validation loss and accuracy
        _, val_Loss_s1[epoch], val_Arry_s1[epoch], val_Aver_s1[epoch], _, _, _, _, = test_feature_level_fusion(model[0],device,data_loaders[2:],classification_loss)
# validation loss and accuracy
        _, val_Loss_s2[epoch], val_Arry_s2[epoch], val_Aver_s2[epoch], _, _, _, _, = test_feature_level_fusion(model[1],device,data_loaders[2:],classification_loss)

        print('Student 1: train class loss: %.4f; train consis loss: %.4f; train acc: %.4f; test acc: %.4f; test averacc: %.4f' % (tra_Loss_s1[epoch], tra_Loss_consis[epoch], tra_Arry_s1[epoch], val_Arry_s1[epoch], val_Aver_s1[epoch]))
        print('Student 2: train class loss: %.4f; train consis loss: %.4f; train acc: %.4f; test acc: %.4f; test averacc: %.4f' % (tra_Loss_s2[epoch], tra_Loss_consis[epoch], tra_Arry_s2[epoch], val_Arry_s2[epoch], val_Aver_s2[epoch]))

    return model, tra_Loss_s1, tra_Loss_s2, tra_Arry_s1, tra_Arry_s2, val_Loss_s1, val_Arry_s1, val_Aver_s1, val_Loss_s2, val_Arry_s2, val_Aver_s2, tra_Loss_consis



def train_feature_level_fusion_unified_loader(model, source_loader, target_loader, optimizer, device, classification_loss, numEpoch):
    nb_train_samples = len(source_loader.dataset)
    nb_test_samples = len(target_loader.dataset)

    nb_batches = len(source_loader)
    print('The number of samples in source domain: % d' % (nb_train_samples))
    print('The number of samples in target domain: % d' % (nb_test_samples))

    traLoss = np.zeros((numEpoch))
    traArry = np.zeros((numEpoch))
    valLoss = np.zeros((numEpoch))
    valArry = np.zeros((numEpoch))
    valAver = np.zeros((numEpoch))

    model.train()

    # start training
    print(" ----------------------------------------- ")
    # training
    for epoch in range(numEpoch):
        print('+................................................+')
        print('Epoch %d:' % (epoch+1))
        running_loss = 0.0
        correct = 0.0
        # iterations in batches
        print("Number of batches (%d in total): " % (nb_batches))
        for i, data in tqdm(enumerate(source_loader)):
            sourceDat1 = data['data1'].to(device,dtype=torch.float)
            sourceDat2 = data['data2'].to(device,dtype=torch.float)

            inLab = data['label'].to(device,dtype=torch.float)
            inLab = torch.max(inLab,1)[1]

            # zero the parameter gradients for each minibatch
            optimizer.zero_grad()
            # forward
            outputs = model(sourceDat1,sourceDat2)
            # loss
            loss = classification_loss(outputs, inLab)
            # backward
            loss.backward()
            # optimize
            optimizer.step()
            # statistics
            running_loss += loss.item() * sourceDat1.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(inLab).sum().item()
        # training loss and accuracy
        traLoss[epoch] = running_loss/nb_train_samples
        traArry[epoch] = correct/nb_train_samples*100

        # validation loss and accuracy
        _, valLoss[epoch], valArry[epoch], valAver[epoch], _, _, _, _, = test_feature_level_fusion_unified_loader(model,device,target_loader,classification_loss)
        # print
        print('epoch %d: training loss: %.4f; training acc: %.2f; validation loss: %.4f; validation acc: %.2f; validation average acc: %.2f' % (epoch+1, traLoss[epoch], traArry[epoch],valLoss[epoch],valArry[epoch],valAver[epoch]))

    print(' --- training done --- ')
    return model,traLoss,traArry,valLoss,valArry,valAver







def train_feature_level_fusion(model, data_loaders, optimizer, device, classification_loss, numEpoch):
    nb_train_samples = len(data_loaders[0].dataset)
    nb_test_samples = len(data_loaders[2].dataset)

    nb_batches = len(data_loaders[0])
    print('The number of samples in source domain: % d' % (nb_train_samples))
    print('The number of samples in target domain: % d' % (nb_test_samples))

    traLoss = np.zeros((numEpoch))
    traArry = np.zeros((numEpoch))
    valLoss = np.zeros((numEpoch))
    valArry = np.zeros((numEpoch))
    valAver = np.zeros((numEpoch))

    model.train()

    # start training
    print(" ----------------------------------------- ")
    # training
    for epoch in range(numEpoch):
        print('+................................................+')
        print('Epoch %d:' % (epoch+1))
        running_loss = 0.0
        correct = 0.0
        # iterations in batches
        print("Number of batches (%d in total): " % (nb_batches))
        for i, data in tqdm(enumerate(zip(data_loaders[0],data_loaders[1]))):
            sourceDat1 = data[0]['data'].to(device,dtype=torch.float)
            sourceLab1 = data[0]['label'].to(device,dtype=torch.float)
            sourceLab1 = torch.max(sourceLab1,1)[1]

            sourceDat2 = data[1]['data'].to(device,dtype=torch.float)
            sourceLab2 = data[1]['label'].to(device,dtype=torch.float)
            sourceLab2 = torch.max(sourceLab2,1)[1]
            if all(sourceLab1==sourceLab2):
                inLab = sourceLab1
            else:
                print('S1 data and S2 data are not corresponded.')
            # zero the parameter gradients for each minibatch
            optimizer.zero_grad()
            # forward
            outputs = model(sourceDat1,sourceDat2)
            # loss
            loss = classification_loss(outputs, inLab)
            # backward
            loss.backward()
            # optimize
            optimizer.step()
            # statistics
            running_loss += loss.item() * sourceDat1.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(inLab).sum().item()
        # training loss and accuracy
        traLoss[epoch] = running_loss/nb_train_samples
        traArry[epoch] = correct/nb_train_samples*100


        # validation loss and accuracy
        _, valLoss[epoch], valArry[epoch], valAver[epoch], _, _, _, _, = test_feature_level_fusion(model,device,data_loaders,classification_loss)
        # print
        print('epoch %d: training loss: %.4f; training acc: %.2f; validation loss: %.4f; validation acc: %.2f; validation average acc: %.2f' % (epoch+1, traLoss[epoch], traArry[epoch],valLoss[epoch],valArry[epoch],valAver[epoch]))

    print(' --- training done --- ')
    return model,traLoss,traArry,valLoss,valArry,valAver


def train_data_level_fusion_unified_loader(model, source_loader, target_loader, optimizer, device, classification_loss, numEpoch):
    nb_train_samples = len(source_loader.dataset)
    nb_test_samples = len(target_loader.dataset)

    nb_batches = len(source_loader)
    print('The number of samples in source domain: % d' % (nb_train_samples))
    print('The number of samples in target domain: % d' % (nb_test_samples))

    traLoss = np.zeros((numEpoch))
    traArry = np.zeros((numEpoch))
    valLoss = np.zeros((numEpoch))
    valArry = np.zeros((numEpoch))
    valAver = np.zeros((numEpoch))

    model.train()
    # start training
    print(" ----------------------------------------- ")
    # training
    for epoch in range(numEpoch):
        print('+................................................+')
        print('Epoch %d:' % (epoch+1))
        running_loss = 0.0
        correct = 0.0
        # iterations in batches
        print("Number of batches (%d in total): " % (nb_batches))
        for i, data in tqdm(enumerate(source_loader)):
            sourceDat1 = data['data1'].to(device,dtype=torch.float)
            sourceDat2 = data['data2'].to(device,dtype=torch.float)
            inLab = data['label'].to(device,dtype=torch.float)
            inLab = torch.max(inLab,1)[1]
            inDat = torch.cat([sourceDat1,sourceDat2],dim=1)

            # zero the parameter gradients for each minibatch
            optimizer.zero_grad()
            # forward
            outputs = model(inDat)
            # loss
            loss = classification_loss(outputs, inLab)
            # backward
            loss.backward()
            # optimize
            optimizer.step()
            # statistics
            running_loss += loss.item() * inDat.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(inLab).sum().item()
        # training loss and accuracy
        traLoss[epoch] = running_loss/nb_train_samples
        traArry[epoch] = correct/nb_train_samples*100

        # validation loss and accuracy
        _, valLoss[epoch], valArry[epoch], valAver[epoch], _, _, _, _, = test_data_level_fusion_unified_loader(model,device,target_loader,classification_loss)
        # print
        print('epoch %d: training loss: %.4f; training acc: %.2f; validation loss: %.4f; validation acc: %.2f; validation average acc: %.2f' % (epoch+1, traLoss[epoch], traArry[epoch],valLoss[epoch],valArry[epoch],valAver[epoch]))

    print(' --- training done --- ')
    return model,traLoss,traArry,valLoss,valArry,valAver




def train_data_level_fusion(model, data_loaders, optimizer, device, classification_loss, numEpoch):
    nb_train_samples = len(data_loaders[0].dataset)
    nb_test_samples = len(data_loaders[2].dataset)

    nb_batches = len(data_loaders[0])
    print('The number of samples in source domain: % d' % (nb_train_samples))
    print('The number of samples in target domain: % d' % (nb_test_samples))

    traLoss = np.zeros((numEpoch))
    traArry = np.zeros((numEpoch))
    valLoss = np.zeros((numEpoch))
    valArry = np.zeros((numEpoch))
    valAver = np.zeros((numEpoch))

    model.train()

    # start training
    print(" ----------------------------------------- ")
    # training
    for epoch in range(numEpoch):
        print('+................................................+')
        print('Epoch %d:' % (epoch+1))
        running_loss = 0.0
        correct = 0.0
        # iterations in batches
        print("Number of batches (%d in total): " % (nb_batches))
        for i, data in tqdm(enumerate(zip(data_loaders[0],data_loaders[1]))):
            sourceDat1 = data[0]['data'].to(device,dtype=torch.float)
            sourceLab1 = data[0]['label'].to(device,dtype=torch.float)
            sourceLab1 = torch.max(sourceLab1,1)[1]

            sourceDat2 = data[1]['data'].to(device,dtype=torch.float)
            sourceLab2 = data[1]['label'].to(device,dtype=torch.float)
            sourceLab2 = torch.max(sourceLab2,1)[1]

            inDat = torch.cat([sourceDat1,sourceDat2],dim=1)
            if all(sourceLab1==sourceLab2):
                inLab = sourceLab1
            else:
                print('S1 data and S2 data are not corresponded.')

            # zero the parameter gradients for each minibatch
            optimizer.zero_grad()
            # forward
            outputs = model(inDat)
            # loss
            loss = classification_loss(outputs, inLab)
            # backward
            loss.backward()
            # optimize
            optimizer.step()
            # statistics
            running_loss += loss.item() * inDat.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(inLab).sum().item()
        # training loss and accuracy
        traLoss[epoch] = running_loss/nb_train_samples
        traArry[epoch] = correct/nb_train_samples*100

  
        # validation loss and accuracy
        _, valLoss[epoch], valArry[epoch], valAver[epoch], _, _, _, _, = test_data_level_fusion(model,device,data_loaders,classification_loss)
        # print
        print('epoch %d: training loss: %.4f; training acc: %.2f; validation loss: %.4f; validation acc: %.2f; validation average acc: %.2f' % (epoch+1, traLoss[epoch], traArry[epoch],valLoss[epoch],valArry[epoch],valAver[epoch]))

    print(' --- training done --- ')
    return model,traLoss,traArry,valLoss,valArry,valAver


def test_feature_level_fusion_unified_loader(model,device,data_loaders,criterion):
    model.eval()
    nb_class = data_loaders.dataset.label.shape[1]
    confusion_matrix = np.zeros((nb_class,nb_class))
    nb_test_samples = len(data_loaders.dataset)

    testLoss = 0.0
    pred = np.zeros((nb_test_samples))
    batch_size = data_loaders.batch_size
    with torch.no_grad():
        for i_batch, data in enumerate(data_loaders):
            sourceDat1 = data['data1'].to(device,dtype=torch.float)
            sourceDat2 = data['data2'].to(device,dtype=torch.float)

            inLab = data['label'].to(device,dtype=torch.float)
            inLab = torch.max(inLab,1)[1]
            # predicting
            output = model(sourceDat1,sourceDat2)
            # prediction error
            loss = criterion(output, inLab)
            testLoss += loss.item()*sourceDat1.size(0)

            _, predTmp = torch.max(output.data, 1) # get the index of the max log-probability 
            for l, p in zip(inLab.view(-1), predTmp.view(-1)):
                confusion_matrix[l.long(), p.long()] += 1

            if batch_size==np.squeeze(predTmp.cpu().numpy()).shape[0]:
                pred[i_batch*batch_size:(i_batch+1)*batch_size] = np.squeeze(predTmp.cpu().numpy())
            else:
                pred[i_batch*batch_size:] = np.squeeze(predTmp.cpu().numpy())

    testLoss = testLoss/np.sum(confusion_matrix)
    oa = np.trace(confusion_matrix).astype(np.float)/np.sum(confusion_matrix)
    pa = np.diagonal(confusion_matrix)/np.sum(confusion_matrix,1)
    aa = np.sum(pa[~np.isnan(pa)])/np.sum(~np.isnan(pa))
    ua = np.diagonal(confusion_matrix)/np.sum(confusion_matrix,0)
    # kappa coefficient
    po = oa
    pe = np.sum(np.sum(confusion_matrix,0)*np.sum(confusion_matrix,1))/np.square(np.sum(confusion_matrix))
    ka = (po-pe)/(1-pe)
    return pred, testLoss, oa, aa, ka, pa, ua, confusion_matrix







def test_feature_level_fusion(model,device,data_loaders,criterion):
    model.eval()
    nb_class = data_loaders[0].dataset.label.shape[1]
    confusion_matrix = np.zeros((nb_class,nb_class))

    nb_test_samples = len(data_loaders[2].dataset)

    testLoss = 0.0
    pred = np.zeros((nb_test_samples))
    batch_size = data_loaders[2].batch_size
    with torch.no_grad():
        for i_batch, data in enumerate(zip(data_loaders[2],data_loaders[3])):
            sourceDat1 = data[0]['data'].to(device,dtype=torch.float)
            sourceLab1 = data[0]['label'].to(device,dtype=torch.float)
            sourceLab1 = torch.max(sourceLab1,1)[1]

            sourceDat2 = data[1]['data'].to(device,dtype=torch.float)
            sourceLab2 = data[1]['label'].to(device,dtype=torch.float)
            sourceLab2 = torch.max(sourceLab2,1)[1]

            if all(sourceLab1==sourceLab2):
                inLab = sourceLab1
            else:
                print('S1 data and S2 data are not corresponded.')

            # predicting
            output = model(sourceDat1,sourceDat2)
            # prediction error
            loss = criterion(output, inLab)
            testLoss += loss.item()*sourceDat1.size(0)

            _, predTmp = torch.max(output.data, 1) # get the index of the max log-probability 
            for l, p in zip(inLab.view(-1), predTmp.view(-1)):
                confusion_matrix[l.long(), p.long()] += 1

            if batch_size==np.squeeze(predTmp.cpu().numpy()).shape[0]:
                pred[i_batch*batch_size:(i_batch+1)*batch_size] = np.squeeze(predTmp.cpu().numpy())
            else:
                pred[i_batch*batch_size:] = np.squeeze(predTmp.cpu().numpy())

    testLoss = testLoss/np.sum(confusion_matrix)
    oa = np.trace(confusion_matrix).astype(np.float)/np.sum(confusion_matrix)
    pa = np.diagonal(confusion_matrix)/np.sum(confusion_matrix,1)
    aa = np.sum(pa[~np.isnan(pa)])/np.sum(~np.isnan(pa))
    ua = np.diagonal(confusion_matrix)/np.sum(confusion_matrix,0)
    # kappa coefficient
    po = oa
    pe = np.sum(np.sum(confusion_matrix,0)*np.sum(confusion_matrix,1))/np.square(np.sum(confusion_matrix))
    ka = (po-pe)/(1-pe)
    return pred, testLoss, oa, aa, ka, pa, ua, confusion_matrix



def test_data_level_fusion_unified_loader(model,device,data_loaders,criterion):
    model.eval()
    nb_class = data_loaders.dataset.label.shape[1]
    confusion_matrix = np.zeros((nb_class,nb_class))
    nb_test_samples = len(data_loaders.dataset)
    testLoss = 0.0
    pred = np.zeros((nb_test_samples))
    batch_size = data_loaders.batch_size

    with torch.no_grad():
        for i_batch, data in enumerate(data_loaders):
            sourceDat1 = data['data1'].to(device,dtype=torch.float)
            sourceDat2 = data['data2'].to(device,dtype=torch.float)
            inDat = torch.cat([sourceDat1,sourceDat2],dim=1)

            inLab = data['label'].to(device,dtype=torch.float)
            inLab = torch.max(inLab,1)[1]

            # predicting
            output = model(inDat)
            # prediction error
            loss = criterion(output, inLab)
            testLoss += loss.item()*inDat.size(0)

            _, predTmp = torch.max(output.data, 1) # get the index of the max log-probability 
            for l, p in zip(inLab.view(-1), predTmp.view(-1)):
                confusion_matrix[l.long(), p.long()] += 1

            if batch_size==np.squeeze(predTmp.cpu().numpy()).shape[0]:
                pred[i_batch*batch_size:(i_batch+1)*batch_size] = np.squeeze(predTmp.cpu().numpy())
            else:
                pred[i_batch*batch_size:] = np.squeeze(predTmp.cpu().numpy())

    testLoss = testLoss/nb_test_samples
    oa = np.trace(confusion_matrix)/nb_test_samples
    pa = np.diagonal(confusion_matrix)/np.sum(confusion_matrix,1)
    aa = np.sum(pa[~np.isnan(pa)])/np.sum(~np.isnan(pa))
    ua = np.diagonal(confusion_matrix)/np.sum(confusion_matrix,0)
    # kappa coefficient
    po = oa
    pe = np.sum(np.sum(confusion_matrix,0)*np.sum(confusion_matrix,1))/np.square(np.sum(confusion_matrix))
    ka = (po-pe)/(1-pe)
    return pred, testLoss, oa, aa, ka, pa, ua, confusion_matrix






def test_data_level_fusion(model,device,data_loaders,criterion):
    model.eval()
    nb_class = data_loaders[0].dataset.label.shape[1]
    confusion_matrix = np.zeros((nb_class,nb_class))

    nb_test_samples = len(data_loaders[2].dataset)

    testLoss = 0.0
    pred = np.zeros((nb_test_samples))
    batch_size = data_loaders[2].batch_size
    with torch.no_grad():
        for i_batch, data in enumerate(zip(data_loaders[2],data_loaders[3])):
            sourceDat1 = data[0]['data'].to(device,dtype=torch.float)
            sourceLab1 = data[0]['label'].to(device,dtype=torch.float)
            sourceLab1 = torch.max(sourceLab1,1)[1]

            sourceDat2 = data[1]['data'].to(device,dtype=torch.float)
            sourceLab2 = data[1]['label'].to(device,dtype=torch.float)
            sourceLab2 = torch.max(sourceLab2,1)[1]

            inDat = torch.cat([sourceDat1,sourceDat2],dim=1)
            if all(sourceLab1==sourceLab2):
                inLab = sourceLab1
            else:
                print('S1 data and S2 data are not corresponded.')

            # predicting
            output = model(inDat)
            # prediction error
            loss = criterion(output, inLab)
            testLoss += loss.item()*inDat.size(0)

            _, predTmp = torch.max(output.data, 1) # get the index of the max log-probability 
            for l, p in zip(inLab.view(-1), predTmp.view(-1)):
                confusion_matrix[l.long(), p.long()] += 1

            if batch_size==np.squeeze(predTmp.cpu().numpy()).shape[0]:
                pred[i_batch*batch_size:(i_batch+1)*batch_size] = np.squeeze(predTmp.cpu().numpy())
            else:
                pred[i_batch*batch_size:] = np.squeeze(predTmp.cpu().numpy())

    testLoss = testLoss/nb_test_samples
    oa = np.trace(confusion_matrix)/nb_test_samples
    pa = np.diagonal(confusion_matrix)/np.sum(confusion_matrix,1)
    aa = np.sum(pa[~np.isnan(pa)])/np.sum(~np.isnan(pa))
    ua = np.diagonal(confusion_matrix)/np.sum(confusion_matrix,0)
    # kappa coefficient
    po = oa
    pe = np.sum(np.sum(confusion_matrix,0)*np.sum(confusion_matrix,1))/np.square(np.sum(confusion_matrix))
    ka = (po-pe)/(1-pe)
    return pred, testLoss, oa, aa, ka, pa, ua, confusion_matrix


def train_semi_fusion_opt(students, data_loaders, optimizers, device, classification_loss, consistency_loss, numEpoch):
    nb_streams = len(students)
    nb_train_samples = len(data_loaders[0].dataset)
    nb_test_samples = len(data_loaders[4].dataset)
    if nb_train_samples<nb_test_samples:
        nb_samples = nb_train_samples
    else:
        nb_samples = nb_test_samples

    nb_batches = len(data_loaders[0])
    print('The number of samples in source domain: % d' % (nb_train_samples))
    print('The number of samples in target domain: % d' % (nb_test_samples))

    cla_loss_train = np.zeros((numEpoch, nb_streams))
    con_loss_train = np.zeros((numEpoch, nb_streams))
    cla_acc_train = np.zeros((numEpoch, nb_streams))
    cla_loss_test = np.zeros((numEpoch, nb_streams))
    cla_acc_test = np.zeros((numEpoch, nb_streams))
    cla_averacc_test = np.zeros((numEpoch, nb_streams))
    for i in range(nb_streams):
        students[i].train()

    # start training
    print(" ----------------------------------------- ")
    # training
    for epoch in range(numEpoch):
        print('+................................................+')
        print('Epoch %d:' % (epoch+1))
        running_loss_class = np.zeros((nb_streams))
        running_loss_consis = np.zeros((nb_streams))
        correct = np.zeros((nb_streams))
        # iterations in batches
        print("Number of batches (%d in total): " % (nb_batches))
        for i, data in tqdm(enumerate(zip(data_loaders[0],data_loaders[1],data_loaders[2],data_loaders[3],data_loaders[4],data_loaders[5]))):
            targetDat_s1 = data[4]['data'].to(device,dtype=torch.float)
            targetDat_s2 = data[5]['data'].to(device,dtype=torch.float)

            sourceLab = []
            stream_out_source = []
            stream_out_target = []
            stream_out_target_prob = []
            class_loss = []
            target_aver_prob = 0

            for j in range(nb_streams):
                sourceDat = data[j]['data'].to(device,dtype=torch.float)
                sourceLab.append(data[j]['label'].to(device,dtype=torch.float))
                sourceLab[j] = torch.max(sourceLab[j],1)[1]
                stream_out_source.append(students[j](sourceDat))
                class_loss.append(classification_loss(stream_out_source[j],sourceLab[j]))

                if j<2:
                    stream_out_target.append(students[j](targetDat_s1))
                    stream_out_target_prob.append(F.softmax(stream_out_target[j],dim=1))
                else:
                    stream_out_target.append(students[j](targetDat_s2))
                    stream_out_target_prob.append(F.softmax(stream_out_target[j],dim=1))


            # consistent loss
            target_aver_prob = torch.zeros(stream_out_target_prob[0].shape).to(device,dtype=torch.float)
            for x in stream_out_target_prob:
                target_aver_prob += x
            target_aver_prob = (stream_out_target_prob[2]+stream_out_target_prob[3])/2

            consis_loss = []
            loss = []
            for j in range(nb_streams):
                consis_loss.append(consistency_loss(stream_out_target_prob[j],target_aver_prob))
                loss.append(class_loss[j]+consis_loss[j])

                optimizers[j].zero_grad()
                if j == nb_streams:
                    loss[j].backward()
                else:
                    loss[j].backward(retain_graph=True)
                optimizers[j].step()

                running_loss_class[j] += class_loss[j].item()*sourceDat.size(0)
                running_loss_consis[j] += consis_loss[j].item()*sourceDat.size(0)
                _, predicted_s1 = torch.max(stream_out_source[j].data, 1)
                correct[j] += predicted_s1.eq(sourceLab[j]).sum().item()

        for j in range(nb_streams):
            cla_loss_train[epoch,j] = running_loss_class[j]/nb_samples
            con_loss_train[epoch,j] = running_loss_consis[j]/nb_samples
            cla_acc_train[epoch,j] = correct[j]/nb_samples*100
            if j<2:
                _, cla_loss_test[epoch, j], cla_acc_test[epoch, j], cla_averacc_test[epoch, j] = test(students[j],device,data_loaders[4],classification_loss)
            else:
                _, cla_loss_test[epoch, j], cla_acc_test[epoch, j], cla_averacc_test[epoch, j] = test(students[j],device,data_loaders[5],classification_loss)

            # print
            print('Stream %d: train class loss: %.4f; train consis loss: %.4f; train acc: %.4f; test acc: %.4f; test averacc: %.4f' % (j, cla_loss_train[epoch,j], con_loss_train[epoch,j], cla_acc_train[epoch,j], cla_acc_test[epoch,j],cla_averacc_test[epoch,j]))



    print(' --- training done --- ')
    return students, cla_loss_train, cla_acc_train, cla_loss_test, cla_acc_test, con_loss_train, cla_averacc_test




def train_semi_fusion(students, data_loaders, optimizers, device, classification_loss, consistency_loss, numEpoch):
    nb_streams = len(students)
    nb_train_samples = len(data_loaders[0].dataset)
    nb_test_samples = len(data_loaders[4].dataset)
    if nb_train_samples<nb_test_samples:
        nb_samples = nb_train_samples
    else:
        nb_samples = nb_test_samples

    nb_batches = len(data_loaders[0])
    print('The number of samples in source domain: % d' % (nb_train_samples))
    print('The number of samples in target domain: % d' % (nb_test_samples))

    cla_loss_train = np.zeros((numEpoch, nb_streams))
    con_loss_train = np.zeros((numEpoch, nb_streams))
    cla_acc_train = np.zeros((numEpoch, nb_streams))
    cla_loss_test = np.zeros((numEpoch, nb_streams))
    cla_acc_test = np.zeros((numEpoch, nb_streams))
    cla_averacc_test = np.zeros((numEpoch, nb_streams))
    for i in range(nb_streams):
        students[i].train()

    # start training
    print(" ----------------------------------------- ")
    # training
    for epoch in range(numEpoch):
        print('+................................................+')
        print('Epoch %d:' % (epoch+1))
        running_loss_class = np.zeros((nb_streams))
        running_loss_consis = np.zeros((nb_streams))
        correct = np.zeros((nb_streams))
        # iterations in batches
        print("Number of batches (%d in total): " % (nb_batches))
        for i, data in tqdm(enumerate(zip(data_loaders[0],data_loaders[1],data_loaders[2],data_loaders[3],data_loaders[4],data_loaders[5]))):
            targetDat_s1 = data[4]['data'].to(device,dtype=torch.float)
            targetDat_s2 = data[5]['data'].to(device,dtype=torch.float)

            sourceLab = []
            stream_out_source = []
            stream_out_target = []
            stream_out_target_prob = []
            class_loss = []
            target_aver_prob = 0

            for j in range(nb_streams):
                sourceDat = data[j]['data'].to(device,dtype=torch.float)
                sourceLab.append(data[j]['label'].to(device,dtype=torch.float))
                sourceLab[j] = torch.max(sourceLab[j],1)[1]
                stream_out_source.append(students[j](sourceDat))
                class_loss.append(classification_loss(stream_out_source[j],sourceLab[j]))

                if j<2:
                    stream_out_target.append(students[j](targetDat_s1))
                    stream_out_target_prob.append(F.softmax(stream_out_target[j],dim=1))
                else:
                    stream_out_target.append(students[j](targetDat_s2))
                    stream_out_target_prob.append(F.softmax(stream_out_target[j],dim=1))


            # consistent loss
            target_aver_prob = torch.zeros(stream_out_target_prob[0].shape).to(device,dtype=torch.float)
            for x in stream_out_target_prob:
                target_aver_prob += x
            target_aver_prob = target_aver_prob/len(students)


            consis_loss = []
            loss = []

            for j in range(nb_streams):
                consis_loss.append(consistency_loss(stream_out_target_prob[j],target_aver_prob))
                loss.append(class_loss[j]+consis_loss[j])

                optimizers[j].zero_grad()
                if j == nb_streams:
                    loss[j].backward()
                else:
                    loss[j].backward(retain_graph=True)
                optimizers[j].step()

                running_loss_class[j] += class_loss[j].item()*sourceDat.size(0)
                running_loss_consis[j] += consis_loss[j].item()*sourceDat.size(0)
                _, predicted_s1 = torch.max(stream_out_source[j].data, 1)
                correct[j] += predicted_s1.eq(sourceLab[j]).sum().item()

        for j in range(nb_streams):
            cla_loss_train[epoch,j] = running_loss_class[j]/nb_samples
            con_loss_train[epoch,j] = running_loss_consis[j]/nb_samples
            cla_acc_train[epoch,j] = correct[j]/nb_samples*100
            if j<2:
                _, cla_loss_test[epoch, j], cla_acc_test[epoch, j], cla_averacc_test[epoch, j] = test(students[j],device,data_loaders[4],classification_loss)
            else:
                _, cla_loss_test[epoch, j], cla_acc_test[epoch, j], cla_averacc_test[epoch, j] = test(students[j],device,data_loaders[5],classification_loss)

            # print
            print('Stream %d: train class loss: %.4f; train consis loss: %.4f; train acc: %.4f; test acc: %.4f; test averacc: %.4f' % (j, cla_loss_train[epoch,j], con_loss_train[epoch,j], cla_acc_train[epoch,j], cla_acc_test[epoch,j],cla_averacc_test[epoch,j]))



    print(' --- training done --- ')
    return students, cla_loss_train, cla_acc_train, cla_loss_test, cla_acc_test, con_loss_train, cla_averacc_test


def data_augment(dat_samples):
    # vertical random flip
    idx = torch.randperm(dat_samples.shape[0])
    dat_samples[idx[:int(idx.shape[0]/2)],:,:,:] = torch.flip(dat_samples[idx[:int(idx.shape[0]/2)],:,:,:],[2])
    # horizontal random flip
    idx = torch.randperm(dat_samples.shape[0])
    dat_samples[idx[:int(idx.shape[0]/2)],:,:,:] = torch.flip(dat_samples[idx[:int(idx.shape[0]/2)],:,:,:],[3])
    # add normal noise
    noise = torch.from_numpy(np.random.normal(loc=0, scale=0.001, size=np.shape(dat_samples))).float()
    idx = torch.randperm(dat_samples.shape[0])
    dat_samples[idx[:int(idx.shape[0]/2)],:,:,:] = dat_samples[idx[:int(idx.shape[0]/2)],:,:,:] + noise[idx[:int(idx.shape[0]/2)],:,:,:]

    return dat_samples



def train_unlabel_ensemble_aug(student1, student2, device, s1_data_load_source, s2_data_load_source, data_load_target, classification_loss, consistency_loss, optimizer1, optimizer2, numEpoch):
    nb_train_samples = len(s1_data_load_source.dataset)
    nb_test_samples = len(data_load_target.dataset)
    if nb_train_samples<nb_test_samples:
        nb_samples = nb_train_samples
    else:
        nb_samples = nb_test_samples

    nb_batches = len(s1_data_load_source)
    print('The number of samples in source domain: % d' % (nb_train_samples))
    print('The number of samples in target domain: % d' % (nb_test_samples))
    #
    student1.train()
    student1.train()
    '''
    initialize outputs
    '''
    # learning rate
    learning_rate_values = np.zeros((numEpoch))
    # training student 1 classification loss
    cla_loss_train_student1 = np.zeros((numEpoch))
    # training student 2 classification loss
    cla_loss_train_student2 = np.zeros((numEpoch))
    # training consistent loss
    con_loss_train_student1 = np.zeros((numEpoch))
    con_loss_train_student2 = np.zeros((numEpoch))
    # training student 1 classification accuracy
    cla_acc_train_student1 = np.zeros((numEpoch))
    # training student 2 classification accuracy
    cla_acc_train_student2 = np.zeros((numEpoch))
    ###  testing
    # testing student 1 classification loss
    cla_loss_test_student1 = np.zeros((numEpoch))
    # testing teacher classification loss
    cla_loss_test_student2 = np.zeros((numEpoch))
    # testing consisitent loss
    # consistentLossTest = np.zeros((numEpoch))
    # testing student 1 classification accuracy
    cla_acc_test_student1 = np.zeros((numEpoch))
    # testing student 2 classification accuracy
    cla_acc_test_student2 = np.zeros((numEpoch))
    # testing student 1 classification average accuracy
    cla_averacc_test_student1 = np.zeros((numEpoch))
    # testing student 2 classification average accuracy
    cla_averacc_test_student2 = np.zeros((numEpoch))

    '''
    start training
    '''
    print(" ----------------------------------------- ")
    # training
    for epoch in range(numEpoch):
        print('+................................................+')
        print('Epoch %d:' % (epoch+1))

        running_loss_stu1_class = 0.0
        running_loss_stu2_class = 0.0
        running_loss_stu1_consis = 0.0
        running_loss_stu2_consis = 0.0

        correct_s1 = 0.0
        correct_s2 = 0.0

        # iterations in batches
        print("Number of batches (%d in total): " % (nb_batches))
        for i, data in tqdm(enumerate(zip(s1_data_load_source, s2_data_load_source, data_load_target))):
            sourceLab1 = data[0]['label'].to(device,dtype=torch.float)
            sourceLab1 = torch.max(sourceLab1,1)[1]
            sourceDat1 = data_augment(data[0]['data']).to(device,dtype=torch.float)

            sourceLab2 = data[1]['label'].to(device,dtype=torch.float)
            sourceLab2 = torch.max(sourceLab2,1)[1]
            sourceDat2 = data_augment(data[1]['data']).to(device,dtype=torch.float)

            targetDat1 = data_augment(data[2]['data'].float()).to(device,dtype=torch.float)
            targetDat2 = data_augment(data[2]['data'].float()).to(device,dtype=torch.float)

            # forward
            student1_out_source = student1(sourceDat1)
            # student1_out_source_prob = F.softmax(student1_out_source, dim=1)
            student1_out_target = student1(targetDat1)
            student1_out_target_prob = F.softmax(student1_out_target, dim=1)

            student2_out_source = student2(sourceDat2)
            # student2_out_source_prob = F.softmax(student2_out_source, dim=1)
            student2_out_target = student2(targetDat2)
            student2_out_target_prob = F.softmax(student2_out_target, dim=1)

            # classification loss
            stu1_class_loss = classification_loss(student1_out_source,sourceLab1)
            stu2_class_loss = classification_loss(student2_out_source,sourceLab2)
            # consistent loss
            target_aver_prob = (student2_out_target_prob+student1_out_target_prob)/2
            stu1_consis_loss = consistency_loss(student1_out_target_prob,target_aver_prob)
            stu2_consis_loss = consistency_loss(student2_out_target_prob,target_aver_prob)
            # total loss
            stu1_loss = stu1_class_loss+stu1_consis_loss
            stu2_loss = stu2_class_loss+stu2_consis_loss
            # zero the parameter gradients for each minibatch
            optimizer1.zero_grad()
            # backward
            stu1_loss.backward(retain_graph=True)
            # update weights in student
            optimizer1.step()

            # zero the parameter gradients for each minibatch
            optimizer2.zero_grad()
            # backward
            stu2_loss.backward()
            # update weights in student
            optimizer2.step()
            # training loss and accuracy
            running_loss_stu1_class += stu1_class_loss.item()*sourceDat1.size(0)
            running_loss_stu2_class += stu2_class_loss.item()*sourceDat2.size(0)

            running_loss_stu1_consis += stu1_consis_loss.item()*sourceDat1.size(0)
            running_loss_stu2_consis += stu1_consis_loss.item()*sourceDat2.size(0)

            _, predicted_s1 = torch.max(student1_out_source.data, 1)
            _, predicted_s2 = torch.max(student2_out_source.data, 1)

            correct_s1 += predicted_s1.eq(sourceLab1).sum().item()
            correct_s2 += predicted_s2.eq(sourceLab2).sum().item()

        # training loss
        cla_loss_train_student1[epoch] = running_loss_stu1_class/nb_samples
        cla_loss_train_student2[epoch] = running_loss_stu2_class/nb_samples

        con_loss_train_student1[epoch] = running_loss_stu1_consis/nb_samples
        con_loss_train_student2[epoch] = running_loss_stu2_consis/nb_samples

        # training accuracy
        cla_acc_train_student1[epoch] = correct_s1/nb_samples*100
        cla_acc_train_student2[epoch] = correct_s2/nb_samples*100

        # validation loss and accuracy
        _, cla_loss_test_student1[epoch], cla_acc_test_student1[epoch],cla_averacc_test_student1[epoch] = test(student1,device,data_load_target,classification_loss)
        _, cla_loss_test_student2[epoch], cla_acc_test_student2[epoch],cla_averacc_test_student2[epoch] = test(student2,device,data_load_target,classification_loss)

        # print
        print('Student1: train class loss: %.4f; train consis loss: %.4f; train acc: %.4f; test acc: %.4f; test averacc: %.4f' % (cla_loss_train_student1[epoch], con_loss_train_student1[epoch], cla_acc_train_student1[epoch], cla_acc_test_student1[epoch],cla_averacc_test_student1[epoch]))
        print('Student2: train class loss: %.4f; train consis loss: %.4f; train acc: %.4f; test acc: %.4f; test averacc: %.4f' % (cla_loss_train_student2[epoch], con_loss_train_student2[epoch], cla_acc_train_student2[epoch], cla_acc_test_student2[epoch],cla_averacc_test_student2[epoch]))


    print(' --- training done --- ')
    return student1, student2, cla_loss_train_student1, cla_acc_train_student1, cla_loss_test_student1, cla_acc_test_student1, cla_loss_train_student2, cla_acc_train_student2, cla_loss_test_student2, cla_acc_test_student2, con_loss_train_student1, con_loss_train_student2, cla_averacc_test_student1, cla_averacc_test_student2





def train_unlabel_ensemble(student1, student2, device, s1_data_load_source, s2_data_load_source, data_load_target, classification_loss, consistency_loss, optimizer1, optimizer2, numEpoch):
    nb_train_samples = len(s1_data_load_source.dataset)
    nb_test_samples = len(data_load_target.dataset)
    if nb_train_samples<nb_test_samples:
        nb_samples = nb_train_samples
    else:
        nb_samples = nb_test_samples

    nb_batches = len(s1_data_load_source)
    print('The number of samples in source domain: % d' % (nb_train_samples))
    print('The number of samples in target domain: % d' % (nb_test_samples))
    #
    student1.train()
    student1.train()
    '''
    initialize outputs
    '''
    # learning rate
    learning_rate_values = np.zeros((numEpoch))
    # training student 1 classification loss
    cla_loss_train_student1 = np.zeros((numEpoch))
    # training student 2 classification loss
    cla_loss_train_student2 = np.zeros((numEpoch))
    # training consistent loss
    con_loss_train_student1 = np.zeros((numEpoch))
    con_loss_train_student2 = np.zeros((numEpoch))
    # training student 1 classification accuracy
    cla_acc_train_student1 = np.zeros((numEpoch))
    # training student 2 classification accuracy
    cla_acc_train_student2 = np.zeros((numEpoch))
    ###  testing
    # testing student 1 classification loss
    cla_loss_test_student1 = np.zeros((numEpoch))
    # testing teacher classification loss
    cla_loss_test_student2 = np.zeros((numEpoch))
    # testing consisitent loss
    # consistentLossTest = np.zeros((numEpoch))
    # testing student 1 classification accuracy
    cla_acc_test_student1 = np.zeros((numEpoch))
    # testing student 2 classification accuracy
    cla_acc_test_student2 = np.zeros((numEpoch))
    # testing student 1 classification average accuracy
    cla_averacc_test_student1 = np.zeros((numEpoch))
    # testing student 2 classification average accuracy
    cla_averacc_test_student2 = np.zeros((numEpoch))

    '''
    start training
    '''
    print(" ----------------------------------------- ")
    # training
    for epoch in range(numEpoch):
        print('+................................................+')
        print('Epoch %d:' % (epoch+1))

        running_loss_stu1_class = 0.0
        running_loss_stu2_class = 0.0
        running_loss_stu1_consis = 0.0
        running_loss_stu2_consis = 0.0

        correct_s1 = 0.0
        correct_s2 = 0.0

        # iterations in batches
        print("Number of batches (%d in total): " % (nb_batches))
        for i, data in tqdm(enumerate(zip(s1_data_load_source, s2_data_load_source, data_load_target))):
            sourceDat1 = data[0]['data'].to(device,dtype=torch.float)
            sourceLab1 = data[0]['label'].to(device,dtype=torch.float)
            sourceLab1 = torch.max(sourceLab1,1)[1]

            sourceDat2 = data[1]['data'].to(device,dtype=torch.float)
            sourceLab2 = data[1]['label'].to(device,dtype=torch.float)
            sourceLab2 = torch.max(sourceLab2,1)[1]

            targetDat = data[2]['data'].to(device,dtype=torch.float)

            # forward
            student1_out_source = student1(sourceDat1)
            # student1_out_source_prob = F.softmax(student1_out_source, dim=1)
            student1_out_target = student1(targetDat)
            student1_out_target_prob = F.softmax(student1_out_target, dim=1)

            student2_out_source = student2(sourceDat2)
            # student2_out_source_prob = F.softmax(student2_out_source, dim=1)
            student2_out_target = student2(targetDat)
            student2_out_target_prob = F.softmax(student2_out_target, dim=1)

            # classification loss
            stu1_class_loss = classification_loss(student1_out_source,sourceLab1)
            stu2_class_loss = classification_loss(student2_out_source,sourceLab2)
            # consistent loss
            target_aver_prob = (student2_out_target_prob+student1_out_target_prob)/2
            stu1_consis_loss = consistency_loss(student1_out_target_prob,target_aver_prob)
            stu2_consis_loss = consistency_loss(student2_out_target_prob,target_aver_prob)
            # total loss
            stu1_loss = stu1_class_loss+stu1_consis_loss
            stu2_loss = stu2_class_loss+stu2_consis_loss
            # zero the parameter gradients for each minibatch
            optimizer1.zero_grad()
            # backward
            stu1_loss.backward(retain_graph=True)
            # update weights in student
            optimizer1.step()

            # zero the parameter gradients for each minibatch
            optimizer2.zero_grad()
            # backward
            stu2_loss.backward()
            # update weights in student
            optimizer2.step()
            # training loss and accuracy
            running_loss_stu1_class += stu1_class_loss.item()*sourceDat1.size(0)
            running_loss_stu2_class += stu2_class_loss.item()*sourceDat2.size(0)

            running_loss_stu1_consis += stu1_consis_loss.item()*sourceDat1.size(0)
            running_loss_stu2_consis += stu1_consis_loss.item()*sourceDat2.size(0)

            _, predicted_s1 = torch.max(student1_out_source.data, 1)
            _, predicted_s2 = torch.max(student2_out_source.data, 1)

            correct_s1 += predicted_s1.eq(sourceLab1).sum().item()
            correct_s2 += predicted_s2.eq(sourceLab2).sum().item()

        # training loss
        cla_loss_train_student1[epoch] = running_loss_stu1_class/nb_samples
        cla_loss_train_student2[epoch] = running_loss_stu2_class/nb_samples

        con_loss_train_student1[epoch] = running_loss_stu1_consis/nb_samples
        con_loss_train_student2[epoch] = running_loss_stu2_consis/nb_samples

        # training accuracy
        cla_acc_train_student1[epoch] = correct_s1/nb_samples*100
        cla_acc_train_student2[epoch] = correct_s2/nb_samples*100

        # validation loss and accuracy
        _, cla_loss_test_student1[epoch], cla_acc_test_student1[epoch],cla_averacc_test_student1[epoch] = test(student1,device,data_load_target,classification_loss)
        _, cla_loss_test_student2[epoch], cla_acc_test_student2[epoch],cla_averacc_test_student2[epoch] = test(student2,device,data_load_target,classification_loss)

        # print
        print('Student1: train class loss: %.4f; train consis loss: %.4f; train acc: %.4f; test acc: %.4f; test averacc: %.4f' % (cla_loss_train_student1[epoch], con_loss_train_student1[epoch], cla_acc_train_student1[epoch], cla_acc_test_student1[epoch],cla_averacc_test_student1[epoch]))
        print('Student2: train class loss: %.4f; train consis loss: %.4f; train acc: %.4f; test acc: %.4f; test averacc: %.4f' % (cla_loss_train_student2[epoch], con_loss_train_student2[epoch], cla_acc_train_student2[epoch], cla_acc_test_student2[epoch],cla_averacc_test_student2[epoch]))



    print(' --- training done --- ')
    return student1, student2, cla_loss_train_student1, cla_acc_train_student1, cla_loss_test_student1, cla_acc_test_student1, cla_loss_train_student2, cla_acc_train_student2, cla_loss_test_student2, cla_acc_test_student2, con_loss_train_student1, con_loss_train_student2, cla_averacc_test_student1, cla_averacc_test_student2






def domainMeanTeacher_Train(student, teacher, device, traDataLoad, optimizer, valDataLoad, classification_loss, consistency_loss, numEpoch, alphaMax,lr_scheduler=None, upperEpoch=50):
    # this function trains the mean teacher model, which organizes the data of source and target domains in separated batches.
    nb_train_samples = len(traDataLoad.dataset)
    nb_test_samples = len(valDataLoad.dataset)
    if nb_train_samples<nb_test_samples:
        nb_samples = nb_train_samples
    else:
        nb_samples = nb_test_samples


    nb_batches = len(traDataLoad)
    print('The number of samples in source domain: % d' % (nb_train_samples))
    print('The number of samples in target domain: % d' % (nb_test_samples))
    #
    student.train()
    teacher.train()

    '''
    initialize outputs 
    '''
    ###  training
    alpha = np.zeros((numEpoch))
    # learning rate
    learning_rate_values = np.zeros((numEpoch))
    # training student classification loss
    classificationLossTrainStudent = np.zeros((numEpoch))
    # training teacher classification loss
    classificationLossTrainTeacher = np.zeros((numEpoch))
    # training consistent loss
    consistentLossTrain = np.zeros((numEpoch))
    consistentLossWeight = np.zeros((numEpoch))
    # training teacher classification accuracy
    classificationAccuTrainTeacher = np.zeros((numEpoch))
    # training student classification accuracy
    classificationAccuTrainStudent = np.zeros((numEpoch))


    ###  testing
    # testing student classification loss
    classificationLossTestStudent = np.zeros((numEpoch))
    # testing teacher classification loss
    classificationLossTestTeacher = np.zeros((numEpoch))
    # testing consisitent loss
    # consistentLossTest = np.zeros((numEpoch))
    # testing teacher classification accuracy
    classificationAccuTestTeacher = np.zeros((numEpoch))
    # testing student classification accuracy
    classificationAccuTestStudent = np.zeros((numEpoch))

    # testing teacher classification average accuracy
    classificationAverAccuTestTeacher = np.zeros((numEpoch))
    # testing student classification average accuracy
    classificationAverAccuTestStudent = np.zeros((numEpoch))


    '''
    start training 
    '''
    print(" ----------------------------------------- ")
    # training
    for epoch in range(numEpoch):
        print('+................................................+')
        print('Epoch %d:' % (epoch+1))

        running_loss_stu_class = 0.0
        running_loss_tea_class = 0.0
        running_loss_consis = 0.0
        correct_t = 0.0
        correct_s = 0.0
        
        # changing the weight of consistent loss, based on the epoch
        consistentLossWeight[epoch] = weightOfConsistentLoss(epoch, upperEpoch)
        # changing the weight of consistent loss, based on the training accuracy of the previous epoch
        #if epoch==0:
        #    consistentLossWeight[epoch] = 0
        #else:
        #    consistentLossWeight[epoch] = weightOfConsistentLossTrAcc(classificationAccuTrainStudent[epoch-1]/100) 

        # alpha value for updating teacher model, weight in exponential moving average (EMA)
        # alpha=0.99
        # alpha[epoch] = min(1 - 1 / (epoch + 1), alphaMax)
        alpha[epoch] = calculateEMAAlpha(epoch, upperEpoch, alphaMax)

        # learning rate decay
        if lr_scheduler !=None:
            lr_scheduler.step()
            learning_rate_values[epoch] = lr_scheduler.get_lr()[0]

        # iterations in batches
        print("Number of batches (%d in total): " % (nb_batches))

        for i, data in tqdm(enumerate(zip(traDataLoad,valDataLoad))):
            sourceDat = data[0]['data'].to(device,dtype=torch.float)
            sourceLab = data[0]['label'].to(device,dtype=torch.float)
            sourceLab = torch.max(sourceLab,1)[1]
            targetDat = data[1]['data'].to(device,dtype=torch.float)

            # forward           
            student_out_source = student(sourceDat)
            student_out_target = student(targetDat)
            student_out_target_prob = F.softmax(student_out_target, dim=1)

            teacher_out_source = teacher(sourceDat)
            teacher_out_target = teacher(targetDat)
            teacher_out_target_prob = F.softmax(teacher_out_target, dim=1)

            # losses for backpropagation
            classLoss = classification_loss(student_out_source,sourceLab)
            consisLoss = consistency_loss(student_out_target_prob,teacher_out_target_prob)
            # weighted combination of losses
            loss = classLoss + consistentLossWeight[epoch]*consisLoss

            # zero the parameter gradients for each minibatch
            optimizer.zero_grad()
            # backward
            loss.backward()
            # update weights in student
            optimizer.step()
            # update weights in teacher
            for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
                teacher_param.data.mul_(alpha[epoch]).add_(1 - alpha[epoch], student_param.data)

            # losses for recording the training
            classTeaLoss = classification_loss(teacher_out_source,sourceLab)
            # training loss and accuracy
            running_loss_stu_class += classLoss.item()*sourceDat.size(0)
            running_loss_tea_class += classTeaLoss.item()*sourceDat.size(0)
            running_loss_consis += consisLoss.item()*sourceDat.size(0)
            
            _, predicted_s = torch.max(student_out_source.data, 1)
            _, predicted_t = torch.max(teacher_out_source.data, 1)
            
            correct_s += predicted_s.eq(sourceLab).sum().item()
            correct_t += predicted_t.eq(sourceLab).sum().item()
                
        # training loss
        classificationLossTrainStudent[epoch] = running_loss_stu_class/nb_samples
        classificationLossTrainTeacher[epoch] = running_loss_tea_class/nb_samples
        consistentLossTrain[epoch] = running_loss_consis/nb_samples
        # training accuracy
        classificationAccuTrainStudent[epoch] = correct_s/nb_samples*100
        classificationAccuTrainTeacher[epoch] = correct_t/nb_samples*100
        
        # validation loss and accuracy
        _, classificationLossTestStudent[epoch], classificationAccuTestStudent[epoch],classificationAverAccuTestStudent[epoch] = test(student,device,valDataLoad,classification_loss)
        _, classificationLossTestTeacher[epoch], classificationAccuTestTeacher[epoch],classificationAverAccuTestTeacher[epoch] = test(teacher,device,valDataLoad,classification_loss)
         

        # print
        #print('Total loss = classification loss + weight of consistent loss * consistent loss: %.4f = %.4f + %.4f * %.4f' % (classificationLossTrainStudent[epoch] + consistentLossWeight[epoch] * consistentLossTrain[epoch], classificationLossTrainStudent[epoch], consistentLossWeight[epoch], consistentLossTrain[epoch]))
        print("The learning rate of the %d epoch: %.6f" % (epoch+1, learning_rate_values[epoch]))
        print('Total loss (%.4f) = classification loss (%.4f) + weight of consistent loss (%.4f) * consistent loss (%.4f)' % (classificationLossTrainStudent[epoch] + consistentLossWeight[epoch] * consistentLossTrain[epoch], classificationLossTrainStudent[epoch], consistentLossWeight[epoch], consistentLossTrain[epoch]))
        print('Alpha in EMA: %.6f' % (alpha[epoch]))
        print('Student model: training loss: %.4f; training acc: %.2f; testing loss: %.4f; testing acc: %.2f; testing average acc: %.2f' % (classificationLossTrainStudent[epoch], classificationAccuTrainStudent[epoch],classificationLossTestStudent[epoch],classificationAccuTestStudent[epoch],classificationAverAccuTestStudent[epoch]))
        print('Teacher model: training loss: %.4f; training acc: %.2f; testing loss: %.4f; testing acc: %.2f; testing average acc: %.2f' % (classificationLossTrainTeacher[epoch], classificationAccuTrainTeacher[epoch],classificationLossTestTeacher[epoch],classificationAccuTestTeacher[epoch],classificationAverAccuTestTeacher[epoch]))
        
        
    print(' --- training done --- ')
    return student,teacher,classificationLossTrainStudent,classificationAccuTrainStudent,classificationLossTestStudent,classificationAccuTestStudent,classificationLossTrainTeacher,classificationAccuTrainTeacher,classificationLossTestTeacher,classificationAccuTestTeacher,consistentLossTrain, consistentLossWeight, alpha, classificationAverAccuTestStudent, classificationAverAccuTestTeacher, learning_rate_values




def domainMeanTeacherDataAug_Train(student, teacher, device, traDataLoad, optimizer, tarDataLoad, classification_loss, consistency_loss, numEpoch, alphaMax, lr_scheduler=None, upperEpoch=50, aug_flip=0, aug_noise_std=0):
    
	
    nb_train_samples = len(traDataLoad.dataset)
    nb_test_samples = len(tarDataLoad.dataset)
    nb_batches = len(traDataLoad)
    print('The number of samples in source domain: % d' % (nb_train_samples))
    print('The number of samples in target domain: % d' % (nb_test_samples))
    #
    student.train()
    teacher.train()

    '''
    initialize outputs
    '''
    ###  training
    alpha = np.zeros((numEpoch))
    # learning rate
    learning_rate_values = np.zeros((numEpoch))
    # training student classification loss
    classificationLossTrainStudent = np.zeros((numEpoch))
    # training teacher classification loss
    classificationLossTrainTeacher = np.zeros((numEpoch))
    # training consistent loss
    consistentLossTrain = np.zeros((numEpoch))
    consistentLossWeight = np.zeros((numEpoch))
    # training teacher classification accuracy
    classificationAccuTrainTeacher = np.zeros((numEpoch))
    # training student classification accuracy
    classificationAccuTrainStudent = np.zeros((numEpoch))


    ###  testing
    # testing student classification loss
    classificationLossTestStudent = np.zeros((numEpoch))
    # testing teacher classification loss
    classificationLossTestTeacher = np.zeros((numEpoch))
    # testing consisitent loss
    # consistentLossTest = np.zeros((numEpoch))
    # testing teacher classification accuracy
    classificationAccuTestTeacher = np.zeros((numEpoch))
    # testing student classification accuracy
    classificationAccuTestStudent = np.zeros((numEpoch))

    # testing teacher classification average accuracy
    classificationAverAccuTestTeacher = np.zeros((numEpoch))
    # testing student classification average accuracy
    classificationAverAccuTestStudent = np.zeros((numEpoch))

    # initialize data augmentation tool 
    aug = ImageAugmentation(aug_flip, aug_noise_std)

    '''
    start training
    '''
    print(" ----------------------------------------- ")
    # training
    for epoch in range(numEpoch):
        print('+................................................+')
        print('Epoch %d:' % (epoch+1))

        running_loss_stu_class = 0.0
        running_loss_tea_class = 0.0
        running_loss_consis = 0.0
        correct_t = 0.0
        correct_s = 0.0

        # changing the weight of consistent loss, based on the epoch
        consistentLossWeight[epoch] = weightOfConsistentLoss(epoch, upperEpoch)
   
        # alpha value for updating teacher model, weight in exponential moving average (EMA)
        # alpha=0.99
        # alpha[epoch] = min(1 - 1 / (epoch + 1), alphaMax)
        alpha[epoch] = calculateEMAAlpha(epoch, upperEpoch, alphaMax)

        # learning rate decay
        if lr_scheduler !=None:
            lr_scheduler.step()
            learning_rate_values[epoch] = lr_scheduler.get_lr()[0]

        # iterations in batches
        print("Number of batches (%d in total): " % (nb_batches))
        for i, data in tqdm(enumerate(zip(traDataLoad,tarDataLoad))):
	    # data load and data augmentation
            sourceDat = data[0]['data'].to(dtype=torch.float)
            sourceDat = aug.augment(sourceDat).to(device)
            sourceLab = data[0]['label'].to(device,dtype=torch.float)
            sourceLab = torch.max(sourceLab,1)[1]
	
            targetDat = data[1]['data'].to(dtype=torch.float)
            targetDat_stu = aug.augment(targetDat).to(device)
            targetDat_tea = aug.augment(targetDat).to(device)

            # forward
            student_out_source = student(sourceDat)
            student_out_target = student(targetDat_stu)
            student_out_target_prob = F.softmax(student_out_target, dim=1)

            teacher_out_source = teacher(sourceDat)
            teacher_out_target = teacher(targetDat_tea)
            teacher_out_target_prob = F.softmax(teacher_out_target, dim=1)
            # losses for backpropagation
            classLoss = classification_loss(student_out_source,sourceLab)
            consisLoss = consistency_loss(student_out_target_prob,teacher_out_target_prob)
            # weighted combination of losses
            loss = classLoss + consistentLossWeight[epoch]*consisLoss

            # zero the parameter gradients for each minibatch
            optimizer.zero_grad()
            # backward
            loss.backward()
            # update weights in student
            optimizer.step()
            # update weights in teacher
            for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
                teacher_param.data.mul_(alpha[epoch]).add_(1 - alpha[epoch], student_param.data)

            # losses for recording the training
            classTeaLoss = classification_loss(teacher_out_source,sourceLab)
            # training loss and accuracy
            running_loss_stu_class += classLoss.item()*sourceDat.size(0)
            running_loss_tea_class += classTeaLoss.item()*sourceDat.size(0)
            running_loss_consis += consisLoss.item()*sourceDat.size(0)

            _, predicted_s = torch.max(student_out_source.data, 1)
            _, predicted_t = torch.max(teacher_out_source.data, 1)

            correct_s += predicted_s.eq(sourceLab).sum().item()
            correct_t += predicted_t.eq(sourceLab).sum().item()

        # training loss
        classificationLossTrainStudent[epoch] = running_loss_stu_class/nb_train_samples
        classificationLossTrainTeacher[epoch] = running_loss_tea_class/nb_train_samples
        consistentLossTrain[epoch] = running_loss_consis/nb_train_samples
        # training accuracy
        classificationAccuTrainStudent[epoch] = correct_s/nb_train_samples*100
        classificationAccuTrainTeacher[epoch] = correct_t/nb_train_samples*100

        # validation loss and accuracy
        _, classificationLossTestStudent[epoch], classificationAccuTestStudent[epoch],classificationAverAccuTestStudent[epoch] = test(student,device,tarDataLoad,classification_loss)
        _, classificationLossTestTeacher[epoch], classificationAccuTestTeacher[epoch],classificationAverAccuTestTeacher[epoch] = test(teacher,device,tarDataLoad,classification_loss)


        # print
        #print('Total loss = classification loss + weight of consistent loss * consistent loss: %.4f = %.4f + %.4f * %.4f' % (classificationLossTrainStudent[epoch] + consistentLossWeight[epoch] * consistentLossTrain[epoch], classificationLossTrainStudent[epoch], consistentLossWeight[epoch], consistentLossTrain[epoch]))
        print("The learning rate of the %d epoch: %.6f" % (epoch+1, learning_rate_values[epoch]))
        print('Total loss (%.4f) = classification loss (%.4f) + weight of consistent loss (%.4f) * consistent loss (%.4f)' % (classificationLossTrainStudent[epoch] + consistentLossWeight[epoch] * consistentLossTrain[epoch], classificationLossTrainStudent[epoch], consistentLossWeight[epoch], consistentLossTrain[epoch]))
        print('Alpha in EMA: %.6f' % (alpha[epoch]))
        print('Student model: training loss: %.4f; training acc: %.2f; testing loss: %.4f; testing acc: %.2f; testing average acc: %.2f' % (classificationLossTrainStudent[epoch], classificationAccuTrainStudent[epoch],classificationLossTestStudent[epoch],classificationAccuTestStudent[epoch],classificationAverAccuTestStudent[epoch]))
        print('Teacher model: training loss: %.4f; training acc: %.2f; testing loss: %.4f; testing acc: %.2f; testing average acc: %.2f' % (classificationLossTrainTeacher[epoch], classificationAccuTrainTeacher[epoch],classificationLossTestTeacher[epoch],classificationAccuTestTeacher[epoch],classificationAverAccuTestTeacher[epoch]))


    print(' --- training done --- ')
    return student,teacher,classificationLossTrainStudent,classificationAccuTrainStudent,classificationLossTestStudent,classificationAccuTestStudent,classificationLossTrainTeacher,classificationAccuTrainTeacher,classificationLossTestTeacher,classificationAccuTestTeacher,consistentLossTrain, consistentLossWeight, alpha, classificationAverAccuTestStudent, classificationAverAccuTestTeacher, learning_rate_values









def domainMeanTeacherConfidence_Train(student, teacher, device, traDataLoad, optimizer, valDataLoad, classification_loss, consistency_loss, numEpoch, alphaMax, alphaMaxEpoch, confident_thres=0.9):
    # this function trains a mean teacher model which separates the data of the source and target domains in different batches, and introduce the confident mask for teacher model
    nb_train_samples = len(traDataLoad.dataset)
    nb_test_samples = len(valDataLoad.dataset)
    nb_batches = len(traDataLoad)

    print('The number of samples in source domain: % d' % (nb_train_samples))
    print('The number of samples in target domain: % d' % (nb_test_samples))

    #
    student.train()
    teacher.train()
    '''
    initialize outputs 
    '''
    ###  training
    alpha = np.zeros((numEpoch))
    # training student classification loss
    classificationLossTrainStudent = np.zeros((numEpoch))
    # training teacher classification loss
    classificationLossTrainTeacher = np.zeros((numEpoch))
    # training consistent loss
    consistentLossTrain = np.zeros((numEpoch))
    consistentLossWeight = np.zeros((numEpoch))
    # training teacher classification accuracy
    classificationAccuTrainTeacher = np.zeros((numEpoch))
    # training student classification accuracy
    classificationAccuTrainStudent = np.zeros((numEpoch))

    ###  testing
    # testing student classification loss
    classificationLossTestStudent = np.zeros((numEpoch))
    # testing teacher classification loss
    classificationLossTestTeacher = np.zeros((numEpoch))
    # testing consisitent loss
    # consistentLossTest = np.zeros((numEpoch))
    # testing teacher classification accuracy
    classificationAccuTestTeacher = np.zeros((numEpoch))
    # testing student classification accuracy
    classificationAccuTestStudent = np.zeros((numEpoch))

    # testing teacher classification average accuracy
    classificationAverAccuTestTeacher = np.zeros((numEpoch))
    # testing student classification average accuracy
    classificationAverAccuTestStudent = np.zeros((numEpoch))


    '''
    start training 
    '''
    print(" ----------------------------------------- ")
    # training
    for epoch in range(numEpoch):
        running_loss_stu_class = 0.0
        running_loss_tea_class = 0.0
        running_loss_consis = 0.0
        correct_t = 0.0
        correct_s = 0.0
        count_con = 0.0

        # iterations in batches
        print("Number of batches (%d in total): " % (nb_batches))
        for i, data in tqdm(enumerate(zip(traDataLoad,valDataLoad))):
            sourceDat = data[0]['data'].to(device,dtype=torch.float)
            sourceLab = data[0]['label'].to(device,dtype=torch.float)
            sourceLab = torch.max(sourceLab,1)[1]
            targetDat = data[1]['data'].to(device,dtype=torch.float)

            # forward           
            student_out_source = student(sourceDat)
            student_out_target = student(targetDat)
            teacher_out_source = teacher(sourceDat)
            teacher_out_target = teacher(targetDat)

            # confidence mask
            teacher_prob = F.softmax(teacher_out_target,dim=1)
            confident_mask = torch.max(teacher_prob,1)[0]>confident_thres
            # losses for backpropagation
            classLoss = classification_loss(student_out_source,sourceLab)
            if torch.sum(confident_mask)>=0.5*traDataLoad.batch_size:
                count_con += torch.sum(confident_mask)
                consisLoss = consistency_loss(student_out_target[confident_mask,:],teacher_out_target[confident_mask,:])
                running_loss_consis += consisLoss.item()*torch.sum(confident_mask)
            else:
                consisLoss = 0
            # weighted combination of losses
            loss = classLoss + consisLoss

            # zero the parameter gradients for each minibatch
            optimizer.zero_grad()
            # backward
            loss.backward()
            # update weights in student
            optimizer.step()
            # update weights in teacher
            for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
                teacher_param.data.mul_(alpha[epoch]).add_(1 - alpha[epoch], student_param.data)

            # losses for recording the training
            classTeaLoss = classification_loss(teacher_out_source,sourceLab)
            # training loss and accuracy
            running_loss_stu_class += classLoss.item()*sourceDat.size(0)
            running_loss_tea_class += classTeaLoss.item()*sourceDat.size(0)

            _, predicted_s = torch.max(student_out_source.data, 1)
            _, predicted_t = torch.max(teacher_out_source.data, 1)

            correct_s += predicted_s.eq(sourceLab).sum().item()
            correct_t += predicted_t.eq(sourceLab).sum().item()

        # training loss
        classificationLossTrainStudent[epoch] = running_loss_stu_class/nb_train_samples
        classificationLossTrainTeacher[epoch] = running_loss_tea_class/nb_train_samples
        consistentLossTrain[epoch] = running_loss_consis/count_con
        # training accuracy
        classificationAccuTrainStudent[epoch] = correct_s/nb_train_samples*100
        classificationAccuTrainTeacher[epoch] = correct_t/nb_train_samples*100





        # validation loss and accuracy
        _, classificationLossTestStudent[epoch], classificationAccuTestStudent[epoch], classificationAverAccuTestStudent[epoch] = test(student,device,valDataLoad,classification_loss)
        _, classificationLossTestTeacher[epoch], classificationAccuTestTeacher[epoch], classificationAverAccuTestTeacher[epoch] = test(teacher,device,valDataLoad,classification_loss)

        # print
        print('Epoch %d:' % (epoch+1))
        #print('Total loss = classification loss + weight of consistent loss * consistent loss: %.4f = %.4f + %.4f * %.4f' % (classificationLossTrainStudent[epoch] + consistentLossWeight[epoch] * consistentLossTrain[epoch], classificationLossTrainStudent[epoch], consistentLossWeight[epoch], consistentLossTrain[epoch]))
        print('Total loss (%.4f) = classification loss (%.4f) +  consistent loss (%.4f)' % (classificationLossTrainStudent[epoch] + consistentLossTrain[epoch], classificationLossTrainStudent[epoch], consistentLossTrain[epoch]))
        print('Alpha in EMA: %.6f' % (alpha[epoch]))
        print('Student model: training loss: %.4f; training acc: %.2f; testing loss: %.4f; testing acc: %.2f; testing average acc: %.2f' % (classificationLossTrainStudent[epoch], classificationAccuTrainStudent[epoch],classificationLossTestStudent[epoch],classificationAccuTestStudent[epoch],classificationAverAccuTestStudent[epoch]))
        print('Teacher model: training loss: %.4f; training acc: %.2f; testing loss: %.4f; testing acc: %.2f; testing average acc: %.2f' % (classificationLossTrainTeacher[epoch], classificationAccuTrainTeacher[epoch],classificationLossTestTeacher[epoch],classificationAccuTestTeacher[epoch],classificationAverAccuTestTeacher[epoch]))


    print(' --- training done --- ')
    return student,teacher,classificationLossTrainStudent,classificationAccuTrainStudent,classificationLossTestStudent,classificationAccuTestStudent,classificationLossTrainTeacher,classificationAccuTrainTeacher,classificationLossTestTeacher,classificationAccuTestTeacher,consistentLossTrain, consistentLossWeight, alpha, classificationAverAccuTestStudent, classificationAverAccuTestTeacher



def domainMeanTeacherDataAugConf_Train(student, teacher, device, traDataLoad, optimizer, tarDataLoad, classification_loss, consistency_loss, numEpoch, alphaMax, lr_scheduler=None, upperEpoch=50, aug_flip=0, aug_noise_std=0, confFlag=1, confThres=0.9):
    nb_train_samples = len(traDataLoad.dataset)
    nb_test_samples = len(tarDataLoad.dataset)
    nb_batches = len(traDataLoad)
    print('The number of samples in source domain: % d' % (nb_train_samples))
    print('The number of samples in target domain: % d' % (nb_test_samples))
    #
    student.train()
    teacher.train()

    '''
    initialize outputs
    '''
    ###  training
    alpha = np.zeros((numEpoch))
    # learning rate
    learning_rate_values = np.zeros((numEpoch))
    # training student classification loss
    classificationLossTrainStudent = np.zeros((numEpoch))
    # training teacher classification loss
    classificationLossTrainTeacher = np.zeros((numEpoch))
    # training consistent loss
    consistentLossTrain = np.zeros((numEpoch))
    # training teacher classification accuracy
    classificationAccuTrainTeacher = np.zeros((numEpoch))
    # training student classification accuracy
    classificationAccuTrainStudent = np.zeros((numEpoch))


    ###  testing
    # testing student classification loss
    classificationLossTestStudent = np.zeros((numEpoch))
    # testing teacher classification loss
    classificationLossTestTeacher = np.zeros((numEpoch))
    # testing consisitent loss
    # consistentLossTest = np.zeros((numEpoch))
    # testing teacher classification accuracy
    classificationAccuTestTeacher = np.zeros((numEpoch))
    # testing student classification accuracy
    classificationAccuTestStudent = np.zeros((numEpoch))

    # testing teacher classification average accuracy
    classificationAverAccuTestTeacher = np.zeros((numEpoch))
    # testing student classification average accuracy
    classificationAverAccuTestStudent = np.zeros((numEpoch))

    # initialize data augmentation tool
    aug = ImageAugmentation(aug_flip, aug_noise_std)

    '''
    start training
    '''
    print(" ----------------------------------------- ")
    # training
    for epoch in range(numEpoch):
        print('+................................................+')
        print('Epoch %d:' % (epoch+1))

        running_loss_stu_class = 0.0
        running_loss_tea_class = 0.0
        running_loss_consis = 0.0
        correct_t = 0.0
        correct_s = 0.0
        count_con = 0.0

        # alpha value for updating teacher model, weight in exponential moving average (EMA)
        # alpha=0.99
        # alpha[epoch] = min(1 - 1 / (epoch + 1), alphaMax)
        alpha[epoch] = calculateEMAAlpha(epoch, upperEpoch, alphaMax)

        # learning rate decay
        if lr_scheduler !=None:
            lr_scheduler.step()
            learning_rate_values[epoch] = lr_scheduler.get_lr()[0]

        # iterations in batches
        print("Number of batches (%d in total): " % (nb_batches))
        for i, data in tqdm(enumerate(zip(traDataLoad,tarDataLoad))):
            # data load and data augmentation
            sourceDat = data[0]['data'].to(dtype=torch.float)
            sourceDat = aug.augment(sourceDat).to(device)
            sourceLab = data[0]['label'].to(device,dtype=torch.float)
            sourceLab = torch.max(sourceLab,1)[1]

            targetDat = data[1]['data'].to(dtype=torch.float)
            targetDat_stu = aug.augment(targetDat).to(device)
            targetDat_tea = aug.augment(targetDat).to(device)

            # forward
            student_out_source = student(sourceDat)
            student_out_target = student(targetDat_stu)
            student_out_target_prob = F.softmax(student_out_target, dim=1)

            teacher_out_source = teacher(sourceDat)
            teacher_out_target = teacher(targetDat_tea)
            teacher_out_target_prob = F.softmax(teacher_out_target, dim=1)
            # confidence mask
            confident_mask = torch.max(teacher_out_target_prob,1)[0]>confThres			
            # losses for backpropagation
            classLoss = classification_loss(student_out_source,sourceLab)
            if torch.sum(confident_mask)>=0.5*traDataLoad.batch_size:
                count_con += torch.sum(confident_mask)
                consisLoss = consistency_loss(student_out_target[confident_mask,:],teacher_out_target[confident_mask,:])
                running_loss_consis += consisLoss.item()*torch.sum(confident_mask)
            else:
                consisLoss = 0
	
            # weighted combination of losses
            loss = classLoss + consisLoss

            # zero the parameter gradients for each minibatch
            optimizer.zero_grad()
            # backward
            loss.backward()
            # update weights in student
            optimizer.step()
            # update weights in teacher
            for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
                teacher_param.data.mul_(alpha[epoch]).add_(1 - alpha[epoch], student_param.data)

            # losses for recording the training
            classTeaLoss = classification_loss(teacher_out_source,sourceLab)
            # training loss and accuracy
            running_loss_stu_class += classLoss.item()*sourceDat.size(0)
            running_loss_tea_class += classTeaLoss.item()*sourceDat.size(0)
            if consisLoss!=0:
                running_loss_consis += consisLoss.item()*sourceDat.size(0)

            _, predicted_s = torch.max(student_out_source.data, 1)
            _, predicted_t = torch.max(teacher_out_source.data, 1)

            correct_s += predicted_s.eq(sourceLab).sum().item()
            correct_t += predicted_t.eq(sourceLab).sum().item()

        # training loss
        classificationLossTrainStudent[epoch] = running_loss_stu_class/nb_train_samples
        classificationLossTrainTeacher[epoch] = running_loss_tea_class/nb_train_samples
        consistentLossTrain[epoch] = running_loss_consis/nb_train_samples
        # training accuracy
        classificationAccuTrainStudent[epoch] = correct_s/nb_train_samples*100
        classificationAccuTrainTeacher[epoch] = correct_t/nb_train_samples*100

        # validation loss and accuracy
        _, classificationLossTestStudent[epoch], classificationAccuTestStudent[epoch],classificationAverAccuTestStudent[epoch] = test(student,device,tarDataLoad,classification_loss)
        _, classificationLossTestTeacher[epoch], classificationAccuTestTeacher[epoch],classificationAverAccuTestTeacher[epoch] = test(teacher,device,tarDataLoad,classification_loss)


        # print
        print("The learning rate of the %d epoch: %.6f" % (epoch+1, learning_rate_values[epoch]))
        print('Total loss (%.4f) = classification loss (%.4f) + consistent loss (%.4f)' % (classificationLossTrainStudent[epoch] + consistentLossTrain[epoch], classificationLossTrainStudent[epoch], consistentLossTrain[epoch]))
        print('Alpha in EMA: %.6f' % (alpha[epoch]))
        print('Student model: training loss: %.4f; training acc: %.2f; testing loss: %.4f; testing acc: %.2f; testing average acc: %.2f' % (classificationLossTrainStudent[epoch], classificationAccuTrainStudent[epoch],classificationLossTestStudent[epoch],classificationAccuTestStudent[epoch],classificationAverAccuTestStudent[epoch]))
        print('Teacher model: training loss: %.4f; training acc: %.2f; testing loss: %.4f; testing acc: %.2f; testing average acc: %.2f' % (classificationLossTrainTeacher[epoch], classificationAccuTrainTeacher[epoch],classificationLossTestTeacher[epoch],classificationAccuTestTeacher[epoch],classificationAverAccuTestTeacher[epoch]))


    print(' --- training done --- ')
    return student,teacher,classificationLossTrainStudent,classificationAccuTrainStudent,classificationLossTestStudent,classificationAccuTestStudent,classificationLossTrainTeacher,classificationAccuTrainTeacher,classificationLossTestTeacher,classificationAccuTestTeacher,consistentLossTrain, alpha, classificationAverAccuTestStudent, classificationAverAccuTestTeacher, learning_rate_values









class ConfusionMatrixDisplay(object):
    """Confusion Matrix visualization.
    Parameters
    ----------
    confusion_matrix : ndarray of shape (n_classes, n_classes)
        Confusion matrix.
    display_labels : ndarray of shape (n_classes,)
        Display labels for plot.
    Attributes
    ----------
    im_ : matplotlib AxesImage
        Image representing the confusion matrix.
    text_ : ndarray of shape (n_classes, n_classes), dtype=matplotlib Text, \
            or None
        Array of matplotlib axes. `None` if `include_values` is false.
    ax_ : matplotlib Axes
        Axes with confusion matrix.
    figure_ : matplotlib Figure
        Figure containing the confusion matrix.
    """
    def __init__(self, confusion_matrix, display_labels, width=7, height=6.5, colorbarFlag=0, picFormat = 'png', savedir = '.'):
        self.confusion_matrix = confusion_matrix
        self.display_labels = display_labels
        self.width = width
        self.height = height
        self.colorbarFlag = colorbarFlag
        self.picFormat = picFormat
        self.savedir = savedir
        self.savePath = os.path.join(savedir,'confusion_matrix.'+self.picFormat)


    def savefig(self,picFormat = 'png'):
        self.figure_.tight_layout(pad=0)
        if picFormat =='eps':
            self.savePath = os.path.join(self.savedir,'confusion_matrix.eps')
        self.figure_.savefig(self.savePath, dpi=300)


    def plot(self, include_values=True, cmap='viridis',
             xticks_rotation='horizontal', values_format=None, ax=None):
        """Plot visualization.
        Parameters
        ----------
        include_values : bool, default=True
            Includes values in confusion matrix.
        cmap : str or matplotlib Colormap, default='viridis'
            Colormap recognized by matplotlib.
        xticks_rotation : {'vertical', 'horizontal'} or float, \
                         default='horizontal'
            Rotation of xtick labels.
        values_format : str, default=None
            Format specification for values in confusion matrix. If `None`,
            the format specification is '.2f' for a normalized matrix, and
            'd' for a unnormalized matrix.
        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.
        Returns
        -------
        display : :class:`~sklearn.metrics.ConfusionMatrixDisplay`
        """
        import matplotlib.pyplot as plt
        from itertools import product

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        fig.set_size_inches(self.width, self.height) # self.height
        cm = self.confusion_matrix
        n_classes = cm.shape[0]
        self.im_ = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        self.text_ = None

        cmap_min, cmap_max = self.im_.cmap(0), self.im_.cmap(256)

        if include_values:
            self.text_ = np.empty_like(cm, dtype=object)
            if values_format is None:
                values_format = 'g'

            # print text with appropriate color depending on background
            thresh = (cm.max() - cm.min()) / 2.
            for i, j in product(range(n_classes), range(n_classes)):
                color = cmap_max if cm[i, j] < thresh else cmap_min
                self.text_[i, j] = ax.text(j, i,
                                           format(cm[i, j], values_format),
                                           ha="center", va="center",
                                           color=color)
        if self.colorbarFlag:
            fig.colorbar(self.im_, ax=ax)

        ax.set(xticks=np.arange(n_classes),
               yticks=np.arange(n_classes),
               xticklabels=self.display_labels,
               yticklabels=self.display_labels,
               ylabel="True label",
               xlabel="Predicted label")

        ax.set_ylim((n_classes - 0.5, -0.5))
        plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)

        self.figure_ = fig
        self.ax_ = ax
        return self



class plotTrainHistory(object):
    # this class plot the training history of a model
    def __init__(self, saveDir, model):
        self.saveDir = saveDir
        self.model = model

    def loadHistory(self):
        try:
            fid = h5py.File(os.path.join(outcomeDir,'training_history.h5'),'r')
            fid.close
        except:
            print('No training history data is found in: '+self.saveDir)
            return 1

    def plotHistory(self):
        if self.model == 'domain_mean_teacher':
            fid = h5py.File(os.path.join(self.saveDir,'training_history.h5'),'r')
            EMA_alpha = np.array(fid['alpha'])
            epoch = np.linspace(1, EMA_alpha.shape[0], EMA_alpha.shape[0])
            import matplotlib.pyplot as plt
            
            # student model acc
            fig, ax = plt.subplots()
            ax.grid()
            plt.ylim(0,100)
            ax.set(xlabel='Epoch',ylabel='Percentage (%)')
            ax.plot(epoch,np.array(fid['classificationAccuTrainStudent']),epoch,np.array(fid['classificationAccuTestTeacher']),epoch,np.array(fid['classificationAverAccuTestStudent'])*100)
            ax.legend(['student training overall accuracy','student testing overall accuracy','student testing average accuracy'],loc=4)
            fig.savefig(os.path.join(self.saveDir,"student_model_hist.png"),dpi=300)
            del fig, ax


            # teacher model acc
            fig, ax = plt.subplots()
            ax.grid()
            plt.ylim(0,100)
            ax.set(xlabel='Epoch',ylabel='Percentage (%)')
            ax.plot(epoch,np.array(fid['classificationAccuTrainTeacher']),epoch,np.array(fid['classificationAccuTestTeacher']),epoch,np.array(fid['classificationAverAccuTestTeacher'])*100)
            ax.legend(['teacher training overall accuracy','teacher testing overall accuracy','teacher testing average accuracy'],loc=4)
            fig.savefig(os.path.join(self.saveDir,"teacher_model_hist.png"),dpi=300)
            del fig, ax


            # EMA alpha and weight of consistent loss
            fig, ax = plt.subplots()
            ax.grid()
            plt.ylim(0,1)
            ax.set(xlabel='Epoch',ylabel='Unitless')
            ax.plot(epoch,np.array(fid['alpha']))
            ax.legend(['EMA Alpha'],loc=4)
            fig.savefig(os.path.join(self.saveDir,"EMA_alpha.png"),dpi=300)
            del fig, ax


            # loss
            fig, ax = plt.subplots()
            ax.grid()
            ax.set(xlabel='Epoch',ylabel='Unitless')
            totalLoss = np.array(fid['consistentLossTrain'])+np.array(fid['classificationLossTrainStudent'])
            ax.plot(epoch,np.array(fid['consistentLossTrain']),epoch,np.array(fid['classificationLossTrainStudent']),epoch, totalLoss, epoch,np.array(fid['classificationLossTrainTeacher']))
            ax.legend(['consistent loss in train','classification loss in train','total loss','teacher classification loss in train'],loc=1)
            fig.savefig(os.path.join(self.saveDir,"training_loss.png"),dpi=300)
            del fig, ax



            # test loss
            fig, ax = plt.subplots()
            ax.grid()
            ax.set(xlabel='Epoch',ylabel='Unitless')
            ax.plot(epoch,np.array(fid['classificationLossTestStudent']),epoch,np.array(fid['classificationLossTestTeacher']))
            ax.legend(['student classification loss in test','teacher classification loss in test'],loc=1)
            fig.savefig(os.path.join(self.saveDir,"test_loss.png"),dpi=300)
            del fig, ax





















