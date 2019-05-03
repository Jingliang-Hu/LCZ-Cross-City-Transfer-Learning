import h5py
import glob
import numpy as np

fileDirs = glob.glob("/datastore/DATA/classification/TIANCHI_DATASET/internal_v1/cultural_10/2_cityBalance/*_test1_sub.h5")

for cv_dir in range(0,len(fileDirs)):
    # load data
    fid = h5py.File(fileDirs[cv_dir])
    print('load data from: '+fileDirs[cv_dir])
 
    coord = np.array(fid['coord_sub'])
    label = np.array(fid['label_sub'])
    sen1 = np.array(fid['sen1_sub'])
    sen2 = np.array(fid['sen2_sub'])
    sampleIdx = np.array(fid['subSampleIdx'])
    fid.close()

    fileTmp = fileDirs[cv_dir].replace('_test1_','_test2_')
    print('load data from: '+fileTmp)
    fid1 = h5py.File(fileTmp)
    coord = np.concatenate((coord,np.array(fid1['coord_sub'])),axis=0)
    label = np.concatenate((label,np.array(fid1['label_sub'])),axis=0)
    sen1 = np.concatenate((sen1,np.array(fid1['sen1_sub'])),axis=0)
    sen2 = np.concatenate((sen2,np.array(fid1['sen2_sub'])),axis=0)
    sampleIdx = np.concatenate((sampleIdx,np.array(fid1['subSampleIdx'])),axis=0)
    fid1.close()

    dirTmp = fileDirs[cv_dir].replace('TIANCHI_DATASET/internal_v1/cultural_10/2_cityBalance/','SEN1/TF/data/')
    dirTmp = dirTmp.replace('_test1_sub.h5','.h5')
    print('Concatenating data and save to: '+str(dirTmp))

    print('instances number of this city: '+str(label.shape[0]))
    print(str(np.sum(label,axis=0)))

    fh = h5py.File(dirTmp,'w')
    fh.create_dataset('coord',data=coord)
    fh.create_dataset('label',data=label)
    fh.create_dataset('sen1', data=sen1)
    fh.create_dataset('sen2', data=sen2)
    fh.create_dataset('sampleIdx',data=sampleIdx)
    fh.close()

    print('city '+ str(cv_dir+1) +' done')
    print('-------------------------------------------------------------------------')

     
