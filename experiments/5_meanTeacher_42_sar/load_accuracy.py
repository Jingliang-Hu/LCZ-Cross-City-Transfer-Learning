import h5py
import numpy as np
import glob
import os

files = glob.glob('LeNet_mean*')

for f in files:
    print('--------------------------')
    print(f)

    fid = h5py.File(os.path.join(f,'teacher_test_accuracy.h5'),'r')
    print('teacher model:')
    print('oa:')
    print(np.array(fid['oa']))
    print('aa:')
    print(np.array(fid['aa']))
    print('ka:')
    print(np.array(fid['ka']))


    fid = h5py.File(os.path.join(f,'student_test_accuracy.h5'),'r')
    print('student model:')
    print('oa:')
    print(np.array(fid['oa']))
    print('aa:')
    print(np.array(fid['aa']))
    print('ka:')
    print(np.array(fid['ka']))


