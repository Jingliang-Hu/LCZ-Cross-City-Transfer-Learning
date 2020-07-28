import h5py
import numpy as np
import glob
import os

files = glob.glob('LeNet_tr_lcz42_te_cul10*')

for f in files:
    print('--------------------------')
    print(f)

    fid = h5py.File(os.path.join(f,'test_accuracy.h5'),'r')
    cm = np.array(fid['confusion_matrix'])

    pa = np.diagonal(cm)/np.sum(cm,1)
    ua = np.diagonal(cm)/np.sum(cm,0)
    oa = np.trace(cm)/np.sum(cm)
    aa = np.sum(pa[~np.isnan(pa)])/np.sum(~np.isnan(pa))
    # kappa coefficient
    po = oa
    pe = np.sum(np.sum(cm,0)*np.sum(cm,1))/np.square(np.sum(cm))
    ka = (po-pe)/(1-pe)




    print('oa:')
    print(oa)
    print('aa:')
    print(aa)
    print('ka:')
    print(ka)




