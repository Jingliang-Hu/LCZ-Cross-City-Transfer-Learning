import h5py
import numpy as np
import glob
import os

files = glob.glob('LeNet_en*-*')

for f in files:
    print('--------------------------')
    print(f)

    fid = h5py.File(os.path.join(f,'student1_test_accuracy.h5'),'r')
    print('student 1 model:')
    print('oa:')
    print(np.array(fid['oa']))
    print('aa:')
    print(np.array(fid['aa']))
    print('ka:')
    print(np.array(fid['ka']))


    fid = h5py.File(os.path.join(f,'student2_test_accuracy.h5'),'r')
    print('student 2 model:')
    print('oa:')
    print(np.array(fid['oa']))
    print('aa:')
    print(np.array(fid['aa']))
    print('ka:')
    print(np.array(fid['ka']))


