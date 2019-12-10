import os
import h5py
import glob
import numpy as np


class dataReader:
    '''
    Class properties:
     - datOpt    	-- options of data types ('s1':sentinel-1; 's2':sentinel-2; 'both': sentinel 1 and 2)
     - locDir 	        -- the directory of data '<directory to local repo>/data/'
     - trPerc    	-- percentage data to be randomly picked up
     - rdSeed     	-- seeds for initializing the random cross validation (for reproduction and comparison) 
			   [5,10,15,20,25,0]
     - trOpt        -- choosing the fold of cross validation [seed1,seed2,seed3,seed4,seed5,seednull];
		       [seed1,seed2,seed3,seed4,seed5,seednull] corresponds to [5,10,15,20,25,0] of 'rdSeed' 
		       if 'seednull':only read the train.h5
    '''
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
	# set value for datOpt
        self.datOpt = datOpt.lower()

    def changeTrOpt(self,trOpt):
	# set value for trOpt
        self.trOpt = trOpt.lower()
        self.rdSeed = 0
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


    def changeLocDir(self,locDir):
	# set value for locDir
        self.dir = locDir

    def changePerc(self,trainPerc):
        # set value for percentage
        self.trPerc = trainPerc


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
        # call 'loadData' func to load 'training.h5'
        trainM,labelM = self.loadData(fid)
        return trainM,labelM

    def miniBatchRandomIndex(self,datArr,nbSample):
        # randomly return an index, with which a mini-batch of given number of samples (nbSample) can be picked out
        order = np.argsort(np.random.randn(datArr.shape[0]))
        index = order[:nbSample]
        return index

    def oneHotEncoding(self,lab,numCls):
        if len(np.squeeze(lab).shape) != 1:
           print('can not one hot encoding, not scalar label')
           return 1
        if np.max(np.squeeze(lab))>numCls:
           print('The max coding number in lab is larger than the given number of classes')
           print('reset lab or maxCls')
           print('the scalar class coding has to be set sequantially from zero to the number of classes')
           return 1
        oneHotLab = np.zeros((np.squeeze(lab).shape[0],numCls))
        for cv_cls in range(0,numCls):
            oneHotLab[lab==cv_cls,cv_cls] = 1
        return oneHotLab



    def randomSplit(self,dat,lab):
        dat = np.array(dat)
        lab = np.array(lab)
        # check lab is one-hot or not, if not, change it to one-hot
        notOnehot = False
        if len(np.squeeze(lab).shape) == 1: # not in one-hot format
            notOnehot = True
            numCls = 17
            lab = self.oneHotEncoding(lab,numCls)

	# reproductive random spliting for the benchmark cross validation
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

        # if input lab is not onehot, the output should also not be onehot
        if notOnehot:
            lab_tr = np.argmax(lab_tr,axis=1)
            lab_te = np.argmax(lab_te,axis=1)
        return dat_tr,lab_tr,dat_te,lab_te


    def readCulturalTen4Train(self,path2Dir):
	# read the cultural 10 data, and randomly split them for training and testing, in a reproductive way. 
	# part of the benchmark experiment
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
	# read all the cultural 10 data only for test.
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

    def loadOneFile(self,directory):
        # read a .h5 data (for testing)
        if directory[-3:]!='.h5':
            print('The given file is not a h5 data file')
            dat = 1
            lab = 1
            return dat,lab
        fid = h5py.File(directory,'r')
        dat,lab = self.loadData(fid)
        return dat,lab


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







