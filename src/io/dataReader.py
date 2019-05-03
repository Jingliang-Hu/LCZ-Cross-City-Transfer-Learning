import os
import h5py
import glob
import numpy as np


class dataReader:
    def __init__(self,datOpt='s1',trOpt='seednull',locDir='./data',trainPerc = .5):
        self.datOpt = datOpt.lower()
        self.trOpt = trOpt.lower()
        self.dir = locDir
        self.rdSeed = 0 
        self.trPerc = trainPerc
        if self.trOpt=='seed1':
            self.rdSeed = 5
        elif self.trOpt=='seed2':
            self.rdSeed = 10
        elif self.trOpt=='seed3':
            self.rdSeed = 15
        elif self.trOpt=='seed4':
            self.rdSeed = 20
        elif self.trOpt=='seed5':
            self.rdSeed = 25
        
    def changeDatOpt(self,datOpt):
        self.datOpt = datOpt.lower()

    def changeTrOpt(self,trOpt):
        self.trOpt = trOpt.lower()

    def changeLocDir(self,locDir):
        self.dir = locDir

    def loadData(self,fid):
        # load sentinel-1 or sentinel-2 or both data set from any '.h5' file
        if self.datOpt == 's1':
            trainM = np.array(fid['sen1'])
        elif self.datOpt == 's2':
            trainM = np.array(fid['sen2'])
        elif self.datOpt == 'both':
            trainM = np.array(fid['sen2'])
            tmp = np.array(fid['sen1'])
            trainM = np.concatenate((tmp,trainM),axis=3)
            del tmp
        else:
            disp('Wrong setting of data to be use, \'datOpt\'.')
            return 1
        labelM = np.array(fid['label'])
        return trainM,labelM


    def readTrainMain(self,path2file):
        # load the LCZ32 data (training.h5)
        if os.path.isfile(path2file):
            try:
                fid = h5py.File(path2file,'r')
            except:
                print('The file indicated is not a \'.h5\' file')
                print(path2file)
                return 1
        else:
            print('The file indicated doesn\'t exist, file name: ')
            print(path2file)
            return 1
        # load sentinel-1 or sentinel-2 or both data set from 'training.h5'
        trainM,labelM = self.loadData(fid)
        return trainM,labelM

    def randomSplit(self,dat,lab):
        for cv_cla in range(0,lab.shape[1]):
            idx = np.squeeze(np.array(np.where(lab[:,cv_cla] == 1)))
            if idx.size == 1:
                print('Only 1 pixel of class '+str(cv_cla+1)+' in one of the cities')
                print('IGNORED')
                continue
            # reproducable random spliting
            np.random.seed(self.rdSeed+cv_cla)
            order = np.argsort(np.random.randn(idx.shape[0]))
            nbTr = np.int(np.ceil(idx.shape[0]*self.trPerc))
            if cv_cla == 0:
                lab_tr = lab[idx[order[:nbTr]]]
                dat_tr = dat[idx[order[:nbTr]]]

                lab_te = lab[idx[order[nbTr:]]]
                dat_te = dat[idx[order[nbTr:]]]
            else:
                lab_tr = np.concatenate((lab_tr,lab[idx[order[:nbTr]]]),axis=0)
                dat_tr = np.concatenate((dat_tr,dat[idx[order[:nbTr]]]),axis=0)

                lab_te = np.concatenate((lab_te,lab[idx[order[nbTr:]]]),axis=0)
                dat_te = np.concatenate((dat_te,dat[idx[order[nbTr:]]]),axis=0)

        return dat_tr,lab_tr,dat_te,lab_te


    def readCulturalTen4Train(self,path2Dir):
        h5Dirs = glob.glob(path2Dir+'/*.h5')
        for cv_dir in range(0,len(h5Dirs)):
            fid = h5py.File(h5Dirs[cv_dir],'r')
            # load data
            dat,lab = self.loadData(fid)
            fid.close
            datTmp_tr, labTmp_tr, datTmp_te, labTmp_te = self.randomSplit(dat,lab)
            if cv_dir == 0:
                dat_tr = datTmp_tr
                dat_te = datTmp_te
                lab_tr = labTmp_tr
                lab_te = labTmp_te
            else:
                dat_tr = np.concatenate((dat_tr,datTmp_tr),axis=0)
                dat_te = np.concatenate((dat_te,datTmp_te),axis=0)
                lab_tr = np.concatenate((lab_tr,labTmp_tr),axis=0)
                lab_te = np.concatenate((lab_te,labTmp_te),axis=0)
            del datTmp_tr, labTmp_tr, datTmp_te, labTmp_te
        return dat_tr,lab_tr,dat_te,lab_te


    def readCulturalTen4Test(self,path2Dir):
        h5Dirs = glob.glob(path2Dir+'/*.h5')
        for cv_dir in range(0,len(h5Dirs)):
            fid = h5py.File(h5Dirs[cv_dir],'r')
            # load data
            dat,lab = self.loadData(fid)
            fid.close
            if cv_dir == 0:
                dat_te = dat
                lab_te = lab
            else:
                dat_te = np.concatenate((dat_te,dat),axis=0)
                lab_te = np.concatenate((lab_te,lab),axis=0)
            del dat,lab
        return dat_te,lab_te

    def loadLCZ(self):
        # training data: 	LCZ32
        # testing data: 	All cultural 10
        if self.rdSeed==0:
            dat_te,lab_te = self.readCulturalTen4Test(self.dir+'/testCites')         
            print('CULTURAL-10 loaded') 
            dat_tr,lab_tr = self.readTrainMain(self.dir+'/training.h5')
            print('LCZ32 loaded')
        else:
            dat_tr_tmp,lab_tr_tmp,dat_te,lab_te = self.readCulturalTen4Train(self.dir+'/testCites')
            print('CULTURAL-10 loaded')
            dat_tr,lab_tr = self.readTrainMain(self.dir+'/training.h5')
            dat_tr = np.concatenate((dat_tr,dat_tr_tmp),axis=0)
            del dat_tr_tmp
            lab_tr = np.concatenate((lab_tr,lab_tr_tmp),axis=0)
            print('LCZ32 loaded')

        return dat_tr,lab_tr,dat_te,lab_te







