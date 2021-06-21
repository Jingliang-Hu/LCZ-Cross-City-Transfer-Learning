import h5py
import numpy as np
import glob as glob

num = np.array([522,2516,4818,1706,1520,3842,977,6891,3927,1768,4655,815,2368,5182,407,1244,5149])
accuracy_files = glob.glob('*/test_accuracy.h5')

pa = np.zeros((17,5))
C = np.zeros((17,17))
for i in range(len(accuracy_files)):
    f = h5py.File(accuracy_files[i],'r')
    pa[:,i] = np.array(f['pa'])
    C+=np.array(f['confusion_matrix'])
    f.close()




