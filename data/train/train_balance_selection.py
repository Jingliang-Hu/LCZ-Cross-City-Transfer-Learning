import h5py
import numpy as np

# set the number of sample for each class
samplePerClass = 5000

# open the train data h5 file
f = h5py.File('train.h5','r')
y = np.array(f['y'])

# initialize the sample index array
sampleIdx = np.zeros(y.shape[0],dtype=np.uint32)

# create a random sample index in a reproduction manner
for idxCla in range(0,y.shape[1]):
    # set random seed for reproduction
    np.random.seed(10)
    nbSamples = np.sum(y[:,idxCla]==1)
    idxTmp = np.where(y[:,idxCla]==1)
    if samplePerClass < nbSamples:
        shuffle = np.argsort(np.random.rand(idxTmp[0].shape[0]))
        sampleIdx[idxTmp[0][shuffle[:samplePerClass]]] = 1
    else:
        sampleIdx[idxTmp[0]] = 1

# save the label
f_save = h5py.File('train_bal.h5','w')
y = y[sampleIdx==1,:]
f_save.create_dataset('y',data=y)
del y

# save the sentinel-1 data
tmp = np.array(f['x_1'])
x_1 = tmp[sampleIdx==1,:]
del tmp
f_save.create_dataset('x_1',data=x_1)
del x_1 

# save the sentinel-2 data
tmp = np.array(f['x_2'])
x_2 = tmp[sampleIdx==1,:]
del tmp
f_save.create_dataset('x_2',data=x_2)
del x_2

f.close()
f_save.close()


