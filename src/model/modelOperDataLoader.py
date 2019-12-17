import torch 
import torch.nn as nn 
from torch.nn import functional as F
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

def calculateEMAAlpha(currentEpoch, maxEpoch,maxAlpha):
    '''
    This function changes the weight of consistent loss for mean teacher model
    '''
    tmp = np.clip(currentEpoch,0,maxEpoch)
    return np.exp(-5*np.square(1-np.float(tmp)/np.float(maxEpoch)))*maxAlpha


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
    testLoss = 0.0
    accuracy = 0.0 # overall accuracy
    correct_nb = 0.0
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
            correct_nb += predTmp.eq(inLab).sum().item()

            if batch_size==np.squeeze(predTmp.cpu().numpy()).shape[0]:
                pred[i_batch*batch_size:(i_batch+1)*batch_size] = np.squeeze(predTmp.cpu().numpy())
            else:
                pred[i_batch*batch_size:] = np.squeeze(predTmp.cpu().numpy())

    testLoss = testLoss/len(valDataLoader.dataset) 
    accuracy = correct_nb/len(valDataLoader.dataset)*100
    return pred, testLoss, accuracy 


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
        _, valLoss[epoch], valArry[epoch] = test(model,device,valDataLoader,criterion)
        # print
        print('epoch %d: training loss: %.4f; training acc: %.2f; validation loss: %.4f; validation acc: %.2f' % (epoch+1, traLoss[epoch], traArry[epoch],valLoss[epoch],valArry[epoch]))

    print(' --- training done --- ')
    return model,traLoss,traArry,valLoss,valArry



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
        _, classificationLossTestStudent[epoch], classificationAccuTestStudent[epoch] = test(student,device,valDataLoad,classification_loss)
        _, classificationLossTestTeacher[epoch], classificationAccuTestTeacher[epoch] = test(teacher,device,valDataLoad,classification_loss)
         
        # print
        print('Epoch %d:' % (epoch+1))
        #print('Total loss = classification loss + weight of consistent loss * consistent loss: %.4f = %.4f + %.4f * %.4f' % (classificationLossTrainStudent[epoch] + consistentLossWeight[epoch] * consistentLossTrain[epoch], classificationLossTrainStudent[epoch], consistentLossWeight[epoch], consistentLossTrain[epoch]))
        print('Total loss (%.4f) = classification loss (%.4f) + weight of consistent loss (%.4f) * consistent loss (%.4f)' % (classificationLossTrainStudent[epoch] + consistentLossWeight[epoch] * consistentLossTrain[epoch], classificationLossTrainStudent[epoch], consistentLossWeight[epoch], consistentLossTrain[epoch]))
        print('Alpha in EMA: %.6f' % (alpha[epoch]))
        print('Student model: training loss: %.4f; training acc: %.2f; testing loss: %.4f; testing acc: %.2f' % (classificationLossTrainStudent[epoch], classificationAccuTrainStudent[epoch],classificationLossTestStudent[epoch],classificationAccuTestStudent[epoch]))
        print('Teacher model: training loss: %.4f; training acc: %.2f; testing loss: %.4f; testing acc: %.2f' % (classificationLossTrainTeacher[epoch], classificationAccuTrainTeacher[epoch],classificationLossTestTeacher[epoch],classificationAccuTestTeacher[epoch]))
        
        
    print(' --- training done --- ')
    return student,teacher,classificationLossTrainStudent,classificationAccuTrainStudent,classificationLossTestStudent,classificationAccuTestStudent,classificationLossTrainTeacher,classificationAccuTrainTeacher,classificationLossTestTeacher,classificationAccuTestTeacher,consistentLossTrain, consistentLossWeight, alpha











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
        _, classificationLossTestStudent[epoch], classificationAccuTestStudent[epoch] = test(student,device,valDataLoad,classification_loss)
        _, classificationLossTestTeacher[epoch], classificationAccuTestTeacher[epoch] = test(teacher,device,valDataLoad,classification_loss)

        # print
        print('Epoch %d:' % (epoch+1))
        #print('Total loss = classification loss + weight of consistent loss * consistent loss: %.4f = %.4f + %.4f * %.4f' % (classificationLossTrainStudent[epoch] + consistentLossWeight[epoch] * consistentLossTrain[epoch], classificationLossTrainStudent[epoch], consistentLossWeight[epoch], consistentLossTrain[epoch]))
        print('Total loss (%.4f) = classification loss (%.4f) +  consistent loss (%.4f)' % (classificationLossTrainStudent[epoch] + consistentLossTrain[epoch], classificationLossTrainStudent[epoch], consistentLossTrain[epoch]))
        print('Alpha in EMA: %.6f' % (alpha[epoch]))
        print('Student model: training loss: %.4f; training acc: %.2f; testing loss: %.4f; testing acc: %.2f' % (classificationLossTrainStudent[epoch], classificationAccuTrainStudent[epoch],classificationLossTestStudent[epoch],classificationAccuTestStudent[epoch]))
        print('Teacher model: training loss: %.4f; training acc: %.2f; testing loss: %.4f; testing acc: %.2f' % (classificationLossTrainTeacher[epoch], classificationAccuTrainTeacher[epoch],classificationLossTestTeacher[epoch],classificationAccuTestTeacher[epoch]))


    print(' --- training done --- ')
    return student,teacher,classificationLossTrainStudent,classificationAccuTrainStudent,classificationLossTestStudent,classificationAccuTestStudent,classificationLossTrainTeacher,classificationAccuTrainTeacher,classificationLossTestTeacher,classificationAccuTestTeacher,consistentLossTrain, consistentLossWeight, alpha























