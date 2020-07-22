import h5py
import numpy as np
import glob
import os

files = glob.glob('LeNet_*')

for f in files:
    print('--------------------------')
    print(f)

    fid = h5py.File(os.path.join(f,'test_accuracy.h5'),'r')
    print('oa:')
    print(np.array(fid['oa']))
    print('aa:')
    print(np.array(fid['aa']))
    print('ka:')
    print(np.array(fid['ka']))




