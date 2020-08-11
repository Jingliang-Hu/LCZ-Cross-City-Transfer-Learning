import matplotlib.pyplot as plt
import h5py
import numpy as np
import glob
import os

exp_file = '6_semi_supervised_ensemble_fusion'
exp_conf = 'LeNet'


hist_files = glob.glob(exp_conf+'*-*')

f = h5py.File(os.path.join(hist_files[0],'training_history.h5'),'r')
mTraArry = np.zeros(np.array(f['cla_acc_train']).shape)
mValArry = np.zeros(np.array(f['cla_acc_test']).shape)
mValAver = np.zeros(np.array(f['cla_averacc_test']).shape)
mTraLoss = np.zeros(np.array(f['cla_loss_train']).shape)
mValLoss = np.zeros(np.array(f['cla_loss_test']).shape)


for i in range(0,len(hist_files)):
    f = h5py.File(os.path.join(hist_files[i],'training_history.h5'),'r')
    traArry = np.array(f['cla_acc_train'])
    valArry = np.array(f['cla_acc_test'])
    valAver = np.array(f['cla_averacc_test'])
    traLoss = np.array(f['cla_loss_train'])
    valLoss = np.array(f['cla_loss_test'])
    # mean acc and loss
    mTraArry += traArry
    mValArry += valArry
    mValAver += valAver
    mTraLoss += traLoss
    mValLoss += valLoss
    accFig = plt.figure(1)
    plt.plot(traArry,'r.',valArry,'b.', valAver, 'k.')
    losFig = plt.figure(2)
    plt.plot(traLoss,'r.',valLoss,'b.')

mTraArry = mTraArry/len(hist_files)
mValArry = mValArry/len(hist_files)
mValAver = mValAver/len(hist_files)
mTraLoss = mTraLoss/len(hist_files)
mValLoss = mValLoss/len(hist_files)

accFig = plt.figure(1)
plt.plot(mTraArry,'r',mValArry,'b', mValAver, 'k')
plt.grid()
plt.ylim(20, 100)
plt.savefig(exp_file+'_acc_train_history.png')

losFig = plt.figure(2)
plt.plot(mTraLoss,'r',mValLoss,'b')
plt.grid()
plt.savefig(exp_file+'_los_train_history.png')



