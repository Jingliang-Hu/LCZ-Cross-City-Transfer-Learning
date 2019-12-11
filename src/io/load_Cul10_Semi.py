import numpy as np
import glob
import os
import h5py

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




