import os
os.environ['QT_QPA_PLATFORM']='offscreen'

import torch 
import torch.nn as nn 
from torch.nn import functional as F
import torch.optim as optim 
import sys
sys.path.insert(0,'../io')
import dataReader
import numpy as np
from tqdm import tqdm
from itertools import product

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
    nb_class = valDataLoader.dataset.label.shape[1]
    confusion_matrix = np.zeros((nb_class,nb_class))

    testLoss = 0.0
    accuracy = 0.0 # overall accuracy
    pred = np.zeros((len(valDataLoader.dataset)))
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

    testLoss = testLoss/len(valDataLoader.dataset) 
    accuracy = np.trace(confusion_matrix)/len(valDataLoader.dataset)*100
    pa = np.diagonal(confusion_matrix)/np.sum(confusion_matrix,1)
    averacc = np.sum(pa[~np.isnan(pa)])/np.sum(~np.isnan(pa))*100


    return pred, testLoss, accuracy, averacc


def train(model,device,optimizer,traDataLoader,criterion,numBatch, numEpoch, valDataLoader):
    '''
    Input:
	- model 	-- pytorch model
	- device 	-- defined cpu or gpu device
	- optimizer	-- optimizer
	- traDataLoader	-- pytorch dataloader for training
	- criterion	-- objective function for optimization 			(default cross entropy)
	- numBatch 	-- number of samples in a batch				(default 64)
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
    # training
    for epoch in range(numEpoch):
        running_loss = 0.0
        correct = 0.0
        print("Number of batches (%d in total): " % (len(traDataLoader))) 
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
        traLoss[epoch] = running_loss/len(traDataLoader.dataset)
        traArry[epoch] = correct/len(traDataLoader.dataset)*100
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







def domainMeanTeacher_Train(student,teacher,device,traDataLoad,optimizer,valDataLoad,classification_loss,consistency_loss,numBatch,numEpoch,alphaMax,upperEpoch=50):
    # this function trains the mean teacher model, which organizes the data of source and target domains in separated batches.

    print('The number of samples in source domain: % d' % (len(traDataLoad.dataset)))
    print('The number of samples in target domain: % d' % (len(valDataLoad.dataset)))
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
        
        # changing the weight of consistent loss, based on the epoch
        consistentLossWeight[epoch] = weightOfConsistentLoss(epoch, upperEpoch)
        
        # alpha value for updating teacher model, weight in exponential moving average (EMA)
        # alpha=0.99
        # alpha[epoch] = min(1 - 1 / (epoch + 1), alphaMax)
        alpha[epoch] = calculateEMAAlpha(epoch, upperEpoch, alphaMax)


        # iterations in batches
        print("Number of batches (%d in total): " % (len(traDataLoad)))

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
            
            # losses for backpropagation
            classLoss = classification_loss(student_out_source,sourceLab)
            consisLoss = consistency_loss(student_out_target,teacher_out_target)
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
        classificationLossTrainStudent[epoch] = running_loss_stu_class/len(traDataLoad.dataset)
        classificationLossTrainTeacher[epoch] = running_loss_tea_class/len(traDataLoad.dataset)
        consistentLossTrain[epoch] = running_loss_consis/len(traDataLoad.dataset)
        # training accuracy
        classificationAccuTrainStudent[epoch] = correct_s/len(traDataLoad.dataset)*100
        classificationAccuTrainTeacher[epoch] = correct_t/len(traDataLoad.dataset)*100
        
        # validation loss and accuracy
        _, classificationLossTestStudent[epoch], classificationAccuTestStudent[epoch],classificationAverAccuTestStudent[epoch] = test(student,device,valDataLoad,classification_loss)
        _, classificationLossTestTeacher[epoch], classificationAccuTestTeacher[epoch],classificationAverAccuTestTeacher[epoch] = test(teacher,device,valDataLoad,classification_loss)
         

        # print
        print('Epoch %d:' % (epoch+1))
        #print('Total loss = classification loss + weight of consistent loss * consistent loss: %.4f = %.4f + %.4f * %.4f' % (classificationLossTrainStudent[epoch] + consistentLossWeight[epoch] * consistentLossTrain[epoch], classificationLossTrainStudent[epoch], consistentLossWeight[epoch], consistentLossTrain[epoch]))
        print('Total loss (%.4f) = classification loss (%.4f) + weight of consistent loss (%.4f) * consistent loss (%.4f)' % (classificationLossTrainStudent[epoch] + consistentLossWeight[epoch] * consistentLossTrain[epoch], classificationLossTrainStudent[epoch], consistentLossWeight[epoch], consistentLossTrain[epoch]))
        print('Alpha in EMA: %.6f' % (alpha[epoch]))
        print('Student model: training loss: %.4f; training acc: %.2f; testing loss: %.4f; testing acc: %.2f; testing average acc: %.2f' % (classificationLossTrainStudent[epoch], classificationAccuTrainStudent[epoch],classificationLossTestStudent[epoch],classificationAccuTestStudent[epoch],classificationAverAccuTestStudent[epoch]))
        print('Teacher model: training loss: %.4f; training acc: %.2f; testing loss: %.4f; testing acc: %.2f; testing average acc: %.2f' % (classificationLossTrainTeacher[epoch], classificationAccuTrainTeacher[epoch],classificationLossTestTeacher[epoch],classificationAccuTestTeacher[epoch],classificationAverAccuTestTeacher[epoch]))
        
        
    print(' --- training done --- ')
    return student,teacher,classificationLossTrainStudent,classificationAccuTrainStudent,classificationLossTestStudent,classificationAccuTestStudent,classificationLossTrainTeacher,classificationAccuTrainTeacher,classificationLossTestTeacher,classificationAccuTestTeacher,consistentLossTrain, consistentLossWeight, alpha, classificationAverAccuTestStudent, classificationAverAccuTestTeacher











def domainMeanTeacherConfidence_Train(student,teacher,device,traDataLoad,optimizer,valDataLoad,classification_loss,consistency_loss,numBatch,numEpoch,alphaMax,alphaMaxEpoch,confident_thres=0.9):
    # this function trains a mean teacher model which separates the data of the source and target domains in different batches, and introduce the confident mask for teacher model

    print('The number of samples in source domain: % d' % (len(traDataLoad.dataset)))
    print('The number of samples in target domain: % d' % (len(valDataLoad.dataset)))

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

        # alpha value for updating teacher model, weight in exponential moving average (EMA)
        # alpha=0.99
        # alpha[epoch] = min(1 - 1 / (epoch + 1), alphaMax)
        alpha[epoch] = calculateEMAAlpha(epoch, alphaMaxEpoch, alphaMax)

        # iterations in batches
        print("Number of batches (%d in total): " % (len(traDataLoad)))
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
            if torch.sum(confident_mask)>=0.5*len(traDataLoad):
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
        classificationLossTrainStudent[epoch] = running_loss_stu_class/len(traDataLoad.dataset)
        classificationLossTrainTeacher[epoch] = running_loss_tea_class/len(traDataLoad.dataset)
        consistentLossTrain[epoch] = running_loss_consis/count_con
        # training accuracy
        classificationAccuTrainStudent[epoch] = correct_s/len(traDataLoad.dataset)*100
        classificationAccuTrainTeacher[epoch] = correct_t/len(traDataLoad.dataset)*100





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
    def __init__(self, confusion_matrix, display_labels):
        self.confusion_matrix = confusion_matrix
        self.display_labels = display_labels

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
        fig.set_size_inches(18, 16)
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



















