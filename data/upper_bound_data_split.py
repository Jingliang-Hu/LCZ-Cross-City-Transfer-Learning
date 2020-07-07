import h5py
import numpy as np

trPerc = 0.5

f = h5py.File('/data/hu/TF/data/test/cul10.h5','r')
coord = np.array(f['coord'])
x_1 =  np.array(f['x_1'])
x_2 =  np.array(f['x_2'])
y =  np.array(f['y'])


for i in range(0,y.shape[1]):
    claPos = np.argwhere(y[:,i] == 1)
    np.random.seed(0)
    claIdx = np.argsort(np.random.rand(claPos.shape[0]))
    splitPoint = np.floor(claIdx.shape[0]*trPerc).astype(int)
    trIdx = claPos[claIdx[:splitPoint]]
    teIdx = claPos[claIdx[splitPoint:]]
    print(['Working on the ',str(i),'th class.'])
    if i == 0:
        idxCheck_tr = trIdx[:,0]
        idxCheck_te = teIdx[:,0]
        #
        coord_tr = coord[trIdx[:,0],:]
        x_1_tr = x_1[trIdx[:,0],:]
        x_2_tr = x_2[trIdx[:,0],:]
        y_tr = y[trIdx[:,0],:]
        #
        coord_te = coord[teIdx[:,0],:]
        x_1_te = x_1[teIdx[:,0],:]
        x_2_te = x_2[teIdx[:,0],:]
        y_te = y[teIdx[:,0],:]
    else:
        idxCheck_tr =  np.concatenate((idxCheck_tr,trIdx[:,0]),axis=0)
        idxCheck_te =  np.concatenate((idxCheck_te,teIdx[:,0]),axis=0)
        #
        coord_tr = np.concatenate((coord_tr,coord[trIdx[:,0],:]),axis=0)
        x_1_tr = np.concatenate((x_1_tr,x_1[trIdx[:,0],:]),axis=0)
        x_2_tr = np.concatenate((x_2_tr,x_2[trIdx[:,0],:]),axis=0)
        y_tr = np.concatenate((y_tr,y[trIdx[:,0],:]),axis=0)
        #
        coord_te = np.concatenate((coord_te,coord[teIdx[:,0],:]),axis=0)
        x_1_te = np.concatenate((x_1_te,x_1[teIdx[:,0],:]),axis=0)
        x_2_te = np.concatenate((x_2_te,x_2[teIdx[:,0],:]),axis=0)
        y_te = np.concatenate((y_te,y[teIdx[:,0],:]),axis=0)

hf = h5py.File('train/cul10_train.h5','w')
hf.create_dataset('x_1', data=x_1_tr)
hf.create_dataset('x_2', data=x_2_tr)
hf.create_dataset('y', data=y_tr)
hf.create_dataset('coord', data=coord_tr)
hf.close()

hf = h5py.File('test/cul10_test.h5','w')
hf.create_dataset('x_1', data=x_1_te)
hf.create_dataset('x_2', data=x_2_te)
hf.create_dataset('y', data=y_te)
hf.create_dataset('coord', data=coord_te)
hf.close()


