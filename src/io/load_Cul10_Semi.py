import numpy as np
import glob
import os
import h5py
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
# import matplotlib.pyplot as plt



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""    
    def __call__(self, sample):
        if len(sample)==2: 
            data, label = sample['data'], sample['label']
            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            data = data.transpose((2, 0, 1))
            return {'data': torch.from_numpy(data),
                    'label': torch.from_numpy(label)}

        elif len(sample)==3:
            data1, data2, label = sample['data1'], sample['data2'], sample['label']
            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            data1 = data1.transpose((2, 0, 1))
            data2 = data2.transpose((2, 0, 1))
            return {'data1': torch.from_numpy(data1),
                    'data2': torch.from_numpy(data2),
                    'label': torch.from_numpy(label)}


class randomNoise(object):
    """
    add random noise with 0 mean and a given standard deviation
    """
    def __init__(self, std):
        self.std = std
            
    def __call__(self, sample):
        data, label = sample['data'],sample['label']
        data += np.random.normal(0, self.std, data.shape)
        return {'data': data, 'label': label}


class LCZDataset(Dataset):
    """
    pytorch iterative data loader costomized to lcz data
    """
    def __init__(self, dataFile, dataFlag, normalization, transform=transforms.Compose([ToTensor()]), shaffle=1, shuffleSeed=0):
        self.dataFile = dataFile
        self.shaffle = shaffle
        self.shuffleSeed = shuffleSeed
        self.dataFlag = dataFlag
        self.transform = transform
        self.loadData()
        self.normalization = normalization
        self.dataNormalization()


    def dataNormalization(self):
        if self.dataFlag==1 or self.dataFlag==2:
            if self.normalization=='pms':
                print("patch-wise mean standard deviation normalization")
                self.data = patch_mean_Std_Normalization(self.data)
            elif self.normalization=='cms':
                print("channel-wise mean standard deviation normalization")
                self.data = channel_mean_Std_Normalization(self.data)      
            elif self.normalization=='no': 
                print("no normalization is carried out")
            else:
                print("no setting for the normalization so no normalization is carried out")

        elif self.dataFlag==0:
            if self.normalization[0]=='pms':
                print("S1 data: patch-wise mean standard deviation normalization")
                self.dat_s1 = patch_mean_Std_Normalization(self.dat_s1)
            elif self.normalization[0]=='cms':
                print("S1 data: channel-wise mean standard deviation normalization")
                self.dat_s1 = channel_mean_Std_Normalization(self.dat_s1)
            elif self.normalization[0]=='no':
                print("S1 data: no normalization is carried out")

            if self.normalization[1]=='pms':
                print("S2 data: patch-wise mean standard deviation normalization")
                self.dat_s2 = patch_mean_Std_Normalization(self.dat_s2)
            elif self.normalization[1]=='cms':
                print("S2 data: channel-wise mean standard deviation normalization")
                self.dat_s2 = channel_mean_Std_Normalization(self.dat_s2)
            elif self.normalization[1]=='no':
                print("S2 data: no normalization is carried out")


    def __len__(self):
        return self.label.shape[0]

    def nbChannel(self):
        fid = h5py.File(self.dataFile,'r')
        if self.dataFlag == 1:
            nb_channel = fid['x_1'].shape[3]
        elif self.dataFlag ==2:
            nb_channel = fid['x_2'].shape[3]
        elif self.dataFlag ==0:
            nb_channel = [fid['x_1'].shape[3],fid['x_2'].shape[3]]
        fid.close()
        del fid
        return nb_channel

    def setData(self,data):
        self.data = data

    def setLabel(self,label):
        self.label = label

    def loadData(self):
        fid = h5py.File(self.dataFile,'r')
        self.label = np.array(fid['y'])
        if self.dataFlag == 1:
           self.data = np.array(fid['x_1'])
        elif self.dataFlag == 2:
           self.data = np.array(fid['x_2'])
        elif self.dataFlag == 0:
           self.dat_s1 = np.array(fid['x_1'])
           self.dat_s2 = np.array(fid['x_2'])
        else:
           print('dataFlag can only be set as 0, 1, or 2')
        fid.close()
        del fid

        if self.shaffle:
            if self.shuffleSeed == 0:
                np.random.seed(0)
            else:
                np.random.seed(self.shuffleSeed)
            idx = np.argsort(np.random.rand(self.label.shape[0])).astype(int)
            self.label = self.label[idx,:]
            if self.dataFlag == 0:
                self.dat_s1 = self.dat_s1[idx,:,:,:]
                self.dat_s2 = self.dat_s2[idx,:,:,:]
            else:
                self.data = self.data[idx,:,:,:]

    def setTimes(self, times):
        self.times = times

    def replicate(self):
        self.data = np.repeat(self.data,self.times,axis=0)
        self.label = np.repeat(self.label,self.times,axis=0)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.dataFlag == 1 or self.dataFlag ==2:
            data = self.data[idx,:,:,:]
            label = self.label[idx,:]                            
            sample = {'data': data, 'label': label}
        elif self.dataFlag == 0:
            data1 = self.dat_s1[idx,:,:,:]
            data2 = self.dat_s2[idx,:,:,:]
            label = self.label[idx,:]
            sample = {'data1': data1, 'data2':data2, 'label': label}

        if self.transform:                            
            sample = self.transform(sample)
        return sample


def lczIterDataSet(envPath,train,test,datFlag,normalization,transform=transforms.Compose([ToTensor()]),shaffle=0,shuffleSeed=0):
    # load training data
    if train=="lcz42":
        datDir = envPath + '/data/train/train.h5'
    elif train=="lcz42_s1":
        datDir = envPath + '/data/train/train_s1.h5'
    elif train=="lcz42_bal":
        datDir = envPath + '/data/train/train_bal.h5'
    elif train=="cul10_train":
        datDir = envPath + '/data/train/cul10_train.h5'
    elif train=="asia":
        datDir = envPath + '/data/train/train_asia.h5'
    elif train=="euro":
        datDir = envPath + '/data/train/train_euro.h5'
    elif train=="north_am":
        datDir = envPath + '/data/train/train_namr.h5'
    elif train=="south_am":
        datDir = envPath + '/data/train/train_samr.h5'
    elif train=="africa":
        datDir = envPath + '/data/train/train_afri.h5'
    else:
        datDir = envPath+'/data/test/'+train+'.h5'
    trainDataSet = LCZDataset(datDir,datFlag,normalization,transform,shaffle,shuffleSeed)

    # load testing data
    if test=="cul10":
        datDir = envPath+'/data/test/cul10.h5'
    elif test=="cul10_s1":
        datDir = envPath+'/data/test/cul10_s1.h5'
    elif test=="cul10_test":
        datDir = envPath+'/data/test/cul10_test.h5'
    elif train=="asia":
        datDir = envPath + '/data/train/train_asia.h5'
    elif train=="euro":
        datDir = envPath + '/data/train/train_euro.h5'
    elif train=="north_am":
        datDir = envPath + '/data/train/train_namr.h5'
    elif train=="south_am":
        datDir = envPath + '/data/train/train_samr.h5'
    elif train=="africa":
        datDir = envPath + '/data/train/train_afri.h5'
    else:
        datDir = envPath+'/data/test/'+test+'.h5'

    testDataSet = LCZDataset(datDir,datFlag,normalization,transform,shaffle,shuffleSeed)
        
    return trainDataSet,testDataSet    




#def show_sample(data, label):
#    tmp = data[:,:,1:4]
#    plt.imshow(tmp)
#    plt.title('Label: %d' % (np.argmax(label)))
    



def initialOutputFolder(paraDict):
    # get time stamp
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    if 'nbStreams' in paraDict:
        outcomeDir = paraDict["modelName"] +'_'+str(paraDict['nbStreams'])+'_stream_tr_'+paraDict["trainData"]+'_te_'+paraDict["testData"]+ '_outcome_' + timestamp
    else:
        outcomeDir = paraDict["modelName"] + '_tr_'+paraDict["trainData"]+'_te_'+paraDict["testData"]+ '_outcome_' + timestamp
    # mkdir output folder
    os.mkdir(outcomeDir)
    return outcomeDir

def recordExpParameters(outcomeDir,paraDict):
    # record parameters
    f = open(os.path.join(outcomeDir,'parameters.txt'),"w")
    f.write("PARAMETERS:")
    f.write("\n")
    for key,value in paraDict.items():
        try:
            f.write(key+':'+value)
        except:
            f.write(key+':'+str(value))
        f.write("\n")
    f.close()
    del f
    return 0



def patch_mean_Std_Normalization(x):
    print("mean standard deviation normalization")
    x = x - x.mean(axis=(1,2,3),keepdims=True)
    x = x/x.std(axis=(1,2,3),keepdims=True)
    return x

def channel_mean_Std_Normalization(x):
    print("mean standard deviation normalization; mean values are:")
    print(x.mean(axis=(0,1,2)))
    x = x - x.mean(axis=(0,1,2),keepdims=True)
    x = x/x.std(axis=(0,1,2),keepdims=True)
    return x





def load_Semi_Test_City(envPath, cityName, datFlag=0):
    # Input:
    # 	 - envPath 	  path to the local repository
    #    - cityName 	  the city name of the testing data
    #    - datFlag        0: sentinel-1 and sentinel-2 data
    #                     1: sentinel-1 data
    #                     2: sentinel-2 data

    # this function load the test data of a given city (a city among the cultural-10 cities). data of those cities are stored in '/data/hu/so2sat_CNN/student_model/data_semi/test'


    # get the directory to h5 data file
    datFile = envPath+'/data/test/'+cityName+'.h5'
    if not os.path.isfile(datFile):
        print('ERROR: File '+cityName+'.h5 does not exist!')
        return 0

    fid = h5py.File(datFile,'r')
    if datFlag==0:
        x_test_1 = np.array(fid['x_1']).astype('float32')
        x_test_2 = np.array(fid['x_2']).astype('float32')
        x_test_1 = np.transpose(x_test_1,(0,3,1,2))
        x_test_2 = np.transpose(x_test_2,(0,3,1,2))

    elif datFlag==1:
        x_test_1 = np.array(fid['x_1']).astype('float32')
        x_test_1 = np.transpose(x_test_1,(0,3,1,2))
        x_test_2 = 0
    elif datFlag==2:
        x_test_1 = 0
        x_test_2 = np.array(fid['x_2']).astype('float32')
        x_test_2 = np.transpose(x_test_2,(0,3,1,2))
    y_test = np.array(fid['y']).astype('uint8')
    coord = np.array(fid['coord']).astype('float32')
    fid.close()
    del fid
    # re-organize dimension for pytorch [nb_of_samples, nb_of_channels, width, height]
    # x_test_1 = np.transpose(x_test_1,(0,3,1,2))
    # x_test_2 = np.transpose(x_test_2,(0,3,1,2))


    return x_test_1, x_test_2, y_test, coord


def load_Semi_Test(envPath, datFlag=0):
    # Input:
    #    - envPath        path to the local repository
    #    - datFlag        0: sentinel-1 and sentinel-2 data
    #                     1: sentinel-1 data
    #                     2: sentinel-2 data
    # this function load the test data of all the cultural-10 cities. data of those cities are stored in '/data/hu/so2sat_CNN/student_model/data_semi/test'
    x_test_1 = []
    x_test_2 = []
    y_test = []
    coord = []

    # get the directory to h5 data file
    datDir = envPath+'/data/test'
    h5Files = glob.glob(datDir+'/*.h5')

    # load data of 10 cities
    for city in h5Files:
        fid = h5py.File(city,'r')
        if datFlag==0:
            x_test_1.append(np.array(fid['x_1']).astype('float32'))
            x_test_2.append(np.array(fid['x_2']).astype('float32'))
            y_test.append(np.array(fid['y']).astype('uint8'))
            coord.append(np.array(fid['coord']).astype('float32'))
            fid.close()
        elif datFlag==1:
            x_test_1.append(np.array(fid['x_1']).astype('float32'))
            x_test_2.append(0)
            y_test.append(np.array(fid['y']).astype('uint8'))
            coord.append(np.array(fid['coord']).astype('float32'))
            fid.close()
        elif datFlag==2:
            x_test_1.append(0)
            x_test_2.append(np.array(fid['x_2']).astype('float32'))
            y_test.append(np.array(fid['y']).astype('uint8'))
            coord.append(np.array(fid['coord']).astype('float32'))
            fid.close()

    x_test_1 = np.array(x_test_1)
    if x_test_1[0].size>1:
        x_test_1 = np.concatenate((x_test_1[:]),axis=0)
        x_test_1 = np.transpose(x_test_1,(0,3,1,2))

    x_test_2 = np.array(x_test_2)
    if x_test_2[0].size>1:
        x_test_2 = np.concatenate((x_test_2[:]),axis=0)
        x_test_2 = np.transpose(x_test_2,(0,3,1,2))

    y_test = np.array(y_test)
    y_test = np.concatenate((y_test[:]),axis=0)

    coord = np.array(coord)
    coord = np.concatenate((coord[:]),axis=0)

    # re-organize dimension for pytorch [nb_of_samples, nb_of_channels, width, height]
    # x_test_1 = np.transpose(x_test_1,(0,3,1,2))
    # x_test_2 = np.transpose(x_test_2,(0,3,1,2))

    return x_test_1, x_test_2, y_test, coord


def load_Semi_Train(envPath,datFlag=0):
    # Input:
    #    - envPath        path to the local repository
    #    - datFlag        0: sentinel-1 and sentinel-2 data
    #                     1: sentinel-1 data
    #                     2: sentinel-2 data
    # this function load the test data of all the cultural-10 cities. data of those cities are stored in '/data/hu/so2sat_CNN/student_model/data_semi/test'
    datDir = envPath + '/data/train/train.h5'
    fid = h5py.File(datDir,'r')

    if datFlag==0:
        x_train_1 = np.array(fid['sen1'])
        x_train_2 = np.array(fid['sen2'])
        x_train_1 = np.transpose(x_train_1,(0,3,1,2))
        x_train_2 = np.transpose(x_train_2,(0,3,1,2))
    elif datFlag==1:
        x_train_1 = np.array(fid['sen1'])
        x_train_1 = np.transpose(x_train_1,(0,3,1,2))
        x_train_2 = 0
    elif datFlag==2:
        x_train_1 = 0
        x_train_2 = np.array(fid['sen2'])
        x_train_2 = np.transpose(x_train_2,(0,3,1,2))


    y_train = np.array(fid['label'])
    coord = np.array(fid['coord'])


    # re-organize dimension for pytorch [nb_of_samples, nb_of_channels, width, height]
    # x_train_1 = np.transpose(x_train_1,(0,3,1,2))
    # x_train_2 = np.transpose(x_train_2,(0,3,1,2))

    
    return x_train_1,x_train_2,y_train,coord







def load_Cul10(envPath,datFlag=0):
    # Input:
    #    - datFlag        0: sentinel-1 and sentinel-2 data
    #                     1: sentinel-1 data
    #                     2: sentinel-2 data


    # get the directory to h5 data file
    datDir = envPath+'/data/data.h5'
    fid = h5py.File(datDir,'r')

    if datFlag==0:
        x_train_1 = np.array(fid['tr1'])
        x_train_2 = np.array(fid['tr2'])
        x_test_1 = np.array(fid['te1'])
        x_test_2 = np.array(fid['te2'])

    elif datFlag==1:
        x_train_1 = np.array(fid['tr1'])
        x_test_1 = np.array(fid['te1'])
        x_train_2 = 0
        x_test_2 = 0

    elif datFlag==2:
        x_train_2 = np.array(fid['tr2'])
        x_test_2 = np.array(fid['te2'])
        x_train_1 = 0
        x_test_1 = 0

    y_train = np.array(fid['trLab'])
    y_test = np.array(fid['teLab'])

    fid.close()
    del fid

    return x_train_1,x_train_2,x_test_1,x_test_2,y_train,y_test
    

def trainValidSplit(x_train,y_train,perc=0.1,randSeed=1):

    nb_samples = x_train.shape[0]
    nb_valid = np.floor(nb_samples*perc)

    # reproduceble random sampling validation data from training data
    np.random.seed(randSeed)
    idx_rand = np.random.rand(nb_samples)
    idx_valid = np.argsort(idx_rand)<=nb_valid

    x_valid = x_train[idx_valid,:,:,:]
    y_valid = y_train[idx_valid,:]

    x_train = x_train[~idx_valid,:,:,:]
    y_train = y_train[~idx_valid,:]

    return x_train,y_train,x_valid,y_valid


def lczLoader(envPath,train,test,datFlag):
    # load training data
    if train=="lcz42":
        x_train_1,x_train_2,y_train,_ = load_Semi_Train(envPath,datFlag)
    else:
        x_train_1,x_train_2,y_train,_ = load_Semi_Test_City(envPath,train,datFlag)

    # load testing data
    if test=="cul10":
        x_test_1,x_test_2,y_test,_ = load_Semi_Test(envPath, datFlag)
    else:
        x_test_1,x_test_2,y_test,_ = load_Semi_Test_City(envPath,test,datFlag)


    # choose data: s1 or s2
    if datFlag==0:
        return x_train_1,x_train_2,y_train,x_test_1,x_test_2,y_test
    elif datFlag==1:
        return x_train_1,y_train,x_test_1,y_test
    elif datFlag==2:
        return x_train_2,y_train,x_test_2,y_test
