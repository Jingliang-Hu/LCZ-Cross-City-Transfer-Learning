import h5py
import glob
import numpy as np


count = np.zeros((10,17))
files = glob.glob('test/*.h5')
for i in range(len(files)):
    fil = files[i]
    fid = h5py.File(fil,'r')
    lab = np.array(fid['y'])
    count[i,:] = np.sum(lab,axis=0)
    print(fil.split('/')[-1].split('.')[0])
    fid.close
    del fid

print(count.astype(np.uint8))



