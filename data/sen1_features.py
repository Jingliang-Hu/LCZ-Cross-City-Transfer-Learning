'''
The transfer study started with the TIANCHI DATASET. 
The sentinel-1 features in TIANCHI DATASET are in their original format. 
It was later proven that Sentinel-1 data should be applied (1) feature extraction and (2)city-wise normalization. (IN LCZ BIG PAPER SENTINEL-1 EXPERIMENT RESULTS)

In this script, it applies feature extraction to the original TIANCHI DATASET for TF experiments.
    (1) if this data could reach the performance of what has been achieved in LCZPAPER, it means feature extraction is enough and critical;
    (2) if this data could not, the city-wise normalization should be applied.

'''


import numpy as np
import h5py

f = h5py.File('./train/train.h5')
s1 = np.array(f['x_1'])
[nb,rw,cl,ch] = s1.shape

x_1 = np.zeros((nb,rw,cl,6),dtype='float32')
x_1[:,:,:,0] = 10*np.log10(np.sum(np.square(s1[:,:,:,:2])))
x_1[:,:,:,1] = 10*np.log10(np.sum(np.square(s1[:,:,:,2:4])))
x_1[:,:,:,2] = 10*np.log10(np.sum(s1[:,:,:,4]))
x_1[:,:,:,3] = 10*np.log10(np.sum(s1[:,:,:,5]))
x_1[:,:,:,4] = np.sqrt(np.add(np.square(s1[:,:,:,6]),np.square(s1[:,:,:,7])))/np.sqrt(s1[:,:,:,4]*s1[:,:,:,5])
x_1[:,:,:,5] = np.cos(np.arctan2(s1[:,:,:,7],s1[:,:,:,6]))
del s1

x_2 = np.array(f['x_2'])
y = np.array(f['y'])
coord = np.array(f['coord'])
f.close
del f

f = h5py.File('./train/train_s1.h5','w')
f.create_dataset('x_1',data=x_1)
f.create_dataset('x_2',data=x_2)
f.create_dataset('y',data=y)
f.create_dataset('coord',data=coord)
f.close
del f, x_1, x_2, y, coord



f = h5py.File('./test/cul10.h5')
s1 = np.array(f['x_1'])
[nb,rw,cl,ch] = s1.shape

x_1 = np.zeros((nb,rw,cl,6),dtype='float32')
x_1[:,:,:,0] = 10*np.log10(np.sum(np.square(s1[:,:,:,:2])))
x_1[:,:,:,1] = 10*np.log10(np.sum(np.square(s1[:,:,:,2:4])))
x_1[:,:,:,2] = 10*np.log10(np.sum(s1[:,:,:,4]))
x_1[:,:,:,3] = 10*np.log10(np.sum(s1[:,:,:,5]))
x_1[:,:,:,4] = np.sqrt(np.add(np.square(s1[:,:,:,6]),np.square(s1[:,:,:,7])))/np.sqrt(s1[:,:,:,4]*s1[:,:,:,5])
x_1[:,:,:,5] = np.cos(np.arctan2(s1[:,:,:,7],s1[:,:,:,6]))
del s1

x_2 = np.array(f['x_2'])
y = np.array(f['y'])
coord = np.array(f['coord'])
f.close
del f

f = h5py.File('./test/cul10_s1.h5','w')
f.create_dataset('x_1',data=x_1)
f.create_dataset('x_2',data=x_2)
f.create_dataset('y',data=y)
f.create_dataset('coord',data=coord)
f.close
del f, x_1, x_2, y, coord








