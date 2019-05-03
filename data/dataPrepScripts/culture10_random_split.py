import h5py
import glob
import numpy as np

fileDirs = glob.glob("/datastore/DATA/classification/TIANCHI_DATASET/internal_v1/cultural_10/2_cityBalance/*_test1_sub.h5")

trPerc = .5
# rdSeed = 5
# rdSeed = 10;
# rdSeed = 15;
# rdSeed = 20;
rdSeed = 25;



for cv_dir in range(0,len(fileDirs)):
    # load data
    fid = h5py.File(fileDirs[cv_dir])
    coord = np.array(fid['coord_sub'])
    label = np.array(fid['label_sub'])
    sen1 = np.array(fid['sen1_sub'])
    sen2 = np.array(fid['sen2_sub'])
    sampleIdx = np.array(fid['subSampleIdx'])
    fid.close()

    fid = h5py.File(fileDirs[cv_dir].replace('_test1_','_test2_'))
    coord = np.concatenate((coord,np.array(fid['coord_sub'])),axis=0)
    label = np.concatenate((label,np.array(fid['label_sub'])),axis=0)
    sen1 = np.concatenate((sen1,np.array(fid['sen1_sub'])),axis=0)
    sen2 = np.concatenate((sen2,np.array(fid['sen2_sub'])),axis=0)
    sampleIdx = np.concatenate((sampleIdx,np.array(fid['subSampleIdx'])),axis=0)
    fid.close()


    for cv_cla in range(0,label.shape[1]):
        idx = np.squeeze(np.array(np.where(label[:,cv_cla] == 1)))
        if idx.size == 1:
            print('Only 1 pixel of class '+str(cv_cla+1)+' in '+fileDirs[cv_dir])
            print('IGNORED')
            continue

        np.random.seed(rdSeed+cv_cla)
        order = np.argsort(np.random.randn(idx.shape[0]))
        nbTr = np.int(np.ceil(idx.shape[0]*trPerc))

        if cv_cla == 0:
            coord_tr = coord[idx[order[:nbTr]]]
            label_tr = label[idx[order[:nbTr]]]
            sen1_tr  = sen1[idx[order[:nbTr]]]
            sen2_tr  = sen2[idx[order[:nbTr]]]
            sampleIdx_tr = sampleIdx[idx[order[:nbTr]]]

            coord_te = coord[idx[order[nbTr:]]]
            label_te = label[idx[order[nbTr:]]]
            sen1_te  = sen1[idx[order[nbTr:]]]
            sen2_te  = sen2[idx[order[nbTr:]]]
            sampleIdx_te = sampleIdx[idx[order[nbTr:]]]
        else:
            coord_tr = np.concatenate((coord_tr,coord[idx[order[:nbTr]]]),axis=0)
            label_tr = np.concatenate((label_tr,label[idx[order[:nbTr]]]),axis=0)
            sen1_tr  = np.concatenate((sen1_tr,sen1[idx[order[:nbTr]]]),axis=0)
            sen2_tr  = np.concatenate((sen2_tr,sen2[idx[order[:nbTr]]]),axis=0)
            sampleIdx_tr = np.concatenate((sampleIdx_tr,sampleIdx[idx[order[:nbTr]]]),axis=0)

            coord_te = np.concatenate((coord_te,coord[idx[order[nbTr:]]]),axis=0)
            label_te = np.concatenate((label_te,label[idx[order[nbTr:]]]),axis=0)
            sen1_te  = np.concatenate((sen1_te,sen1[idx[order[nbTr:]]]),axis=0)
            sen2_te  = np.concatenate((sen2_te,sen2[idx[order[nbTr:]]]),axis=0)
            sampleIdx_te = np.concatenate((sampleIdx_te,sampleIdx[idx[order[nbTr:]]]),axis=0)
       
    dirTmp = fileDirs[cv_dir].replace('TIANCHI_DATASET/internal_v1/cultural_10/2_cityBalance/','SEN1/TF/data/')
    dirTmp = dirTmp.replace('_test1_sub.h5','_TR_SEED_'+str(rdSeed)+'.h5')

    fh = h5py.File(dirTmp,'w')
    fh.create_dataset('coord_tr',data=coord_tr)
    fh.create_dataset('label_tr',data=label_tr)
    fh.create_dataset('sen1_tr', data=sen1_tr)
    fh.create_dataset('sen2_tr', data=sen2_tr)
    fh.create_dataset('sampleIdx_tr',data=sampleIdx_tr)
    fh.close()


    dirTmp = dirTmp.replace('_TR_','_TE_')
    fh = h5py.File(dirTmp,'w')
    fh.create_dataset('coord_te',data=coord_tr)
    fh.create_dataset('label_te',data=label_tr)
    fh.create_dataset('sen1_te', data=sen1_tr)
    fh.create_dataset('sen2_te', data=sen2_tr)
    fh.create_dataset('sampleIdx_te',data=sampleIdx_tr)
    fh.close()


    print('city '+ str(cv_dir+1) +' done')


     
