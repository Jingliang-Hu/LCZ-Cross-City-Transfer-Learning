import matplotlib.pyplot as plt
import h5py
import numpy as np
import os

'''
bench mark training history
'''
# benchHistFile = '/data/hu/TF/experiments/0_benchMark/resnet_benchMark_tr_lcz42_te_cul10_outcome_2019-12-13_13:40:31/training_history.h5'
benchHistFile = '/data/hu/TF/experiments/0_benchMark/resnet_benchMark_tr_moscow_te_munich_outcome_2019-12-14_14:32:43'
fid = h5py.File(os.path.join(benchHistFile,'training_history.h5'),'r')
traArry = np.array(fid['traArry'])
traLoss = np.array(fid['traLoss'])
valArry = np.array(fid['valArry'])
valLoss = np.array(fid['valLoss'])
fid.close

epoch = range(1,traArry.shape[0]+1)

plt.plot(epoch, traArry, 'r', epoch, valArry, 'b')
plt.legend(('training accuracy', 'test accuracy'), shadow=True)
plt.xlim([1,traArry.shape[0]+1])
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Accuracy in percentage')
plt.savefig(os.path.join(benchHistFile,'ResNet_bench_acc.png'),dpi=300)
plt.show()

plt.plot(epoch, traLoss, 'r', epoch, valLoss, 'b')
plt.legend(('training loss', 'test loss'),  shadow=True)
plt.xlim([1,traArry.shape[0]+1])
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(os.path.join(benchHistFile,'ResNet_bench_loss.png'),dpi=300)
plt.show()


'''
compare training history
'''
#meanTeacherHistFile = '/data/hu/TF/experiments/1_meanTeacher/domain_mean_teacher_tr_lcz42_te_cul10_outcome_2019-12-13_13:44:36/training_history.h5'
meanTeacherHistFile = '/data/hu/TF/experiments/1_meanTeacher/domain_mean_teacher_tr_moscow_te_munich_outcome_2019-12-15_00:50:00'
#'/data/hu/TF/experiments/1_meanTeacher/domain_mean_teacher_tr_moscow_te_munich_outcome_2019-12-14_23:00:44'
fid = h5py.File(os.path.join(meanTeacherHistFile,'training_history.h5'),'r')

plt.plot(epoch, valArry, 'b')
epoch1 = range(1,np.array(fid['classificationAccuTestStudent']).shape[0]+1)
plt.plot(epoch1, np.array(fid['classificationAccuTestStudent']), 'r--', epoch1, np.array(fid['classificationAccuTestTeacher']), 'r')
plt.legend(('bench resnet test accuracy','student test accuracy','teacher test accuracy'),  shadow=True)
#plt.xlim([1,traArry.shape[0]+1])
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Accuracy in percentage')
plt.savefig(os.path.join(meanTeacherHistFile,'acc.png'),dpi=300)
plt.show()



plt.plot(epoch, valLoss, 'b')
plt.plot(epoch1, np.array(fid['classificationLossTestStudent']), 'r--', epoch1, np.array(fid['classificationLossTestTeacher']), 'r')
plt.legend(('bench resnet test loss','student loss','teacher test loss'),  shadow=True)
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(os.path.join(meanTeacherHistFile,'loss.png'),dpi=300)
plt.show()



