import matplotlib.pyplot as plt
import h5py
import numpy as np


'''
bench mark training history
'''
benchHistFile = '/data/hu/TF/experiments/0_benchMark/resnet_benchMark_tr_lcz42_te_cul10_outcome_2019-12-13_13:40:31/training_history.h5'
fid = h5py.File(benchHistFile,'r')
traArry = np.array(fid['traArry'])
traLoss = np.array(fid['traLoss'])
valArry = np.array(fid['valArry'])
valLoss = np.array(fid['valLoss'])
fid.close

epoch = range(1,traArry.shape[0]+1)

plt.plot(epoch, traArry, 'r', epoch, valArry, 'b')
plt.legend(('training accuracy', 'test accuracy'), loc='lower right', shadow=True)
plt.xlim([1,traArry.shape[0]+1])
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Accuracy in percentage')
plt.savefig('ResNet_bench_acc.png',dpi=300)

plt.plot(epoch, traLoss, 'r', epoch, valLoss, 'b')
plt.legend(('training loss', 'test loss'), loc='lower right', shadow=True)
plt.xlim([1,traArry.shape[0]+1])
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('ResNet_bench_loss.png',dpi=300)



'''
compare training history
'''
meanTeacherHistFile = '/data/hu/TF/experiments/1_meanTeacher/domain_mean_teacher_tr_lcz42_te_cul10_outcome_2019-12-13_13:44:36/training_history.h5'
fid = h5py.File(meanTeacherHistFile,'r')


plt.plot(epoch, np.array(fid['classificationAccuTestStudent']), 'r--', epoch, np.array(fid['classificationAccuTestTeacher']), 'r', epoch, valArry, 'b')
plt.legend(('student test accuracy','teacher test accuracy', 'bench resnet test accuracy'), loc='lower right', shadow=True)
plt.xlim([1,traArry.shape[0]+1])
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Accuracy in percentage')
plt.savefig('acc.png',dpi=300)




plt.plot(epoch, np.array(fid['classificationLossTestStudent']), 'r--', epoch, np.array(fid['classificationLossTestTeacher']), 'r', epoch, valLoss, 'b')
plt.legend(('student test ','teacher test accuracy', 'bench resnet test accuracy'), loc='lower right', shadow=True)
plt.xlim([1,traArry.shape[0]+1])
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('loss.png',dpi=300)



