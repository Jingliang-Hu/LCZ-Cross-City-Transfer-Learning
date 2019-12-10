import numpy as np
import h5py
import glob
import os

cityNames = ['guangzhou','jakarta','moscow','mumbai','munich','nairobi','sanfrancisco','santiago','sydney','tehran']

datFiles = glob.glob('/data/hu/so2sat_CNN/student_model/data/*.h5')

saveDir = '/data/hu/TF/data_semi'

for city in cityNames:
    datFiles = glob.glob('/data/hu/so2sat_CNN/student_model/data/'+city+'*.h5')
    x_1=[]
    x_2=[]
    y=[]
    coord=[]
    for dataFile in datFiles:
        fid = h5py.File(dataFile,'r')
        x_1.append(np.array(fid['sen1_sub']))
        x_2.append(np.array(fid['sen2_sub']))
        y.append(np.array(fid['label_sub']))
        coord.append(np.array(fid['coord_sub']))
        fid.close()

    x_1 = np.array(x_1)
    x_1 = np.concatenate((x_1[:]),axis=0)
    x_2 = np.array(x_2)
    x_2 = np.concatenate((x_2[:]),axis=0)
    y = np.array(y)
    y = np.concatenate((y[:]),axis=0)
    coord = np.array(coord)
    coord= np.concatenate((coord[:]),axis=0)

    saveFileName = os.path.join(saveDir,city+'.h5')
    hf = h5py.File(saveFileName,'w')
    hf.create_dataset('x_1', data=x_1)
    hf.create_dataset('x_2', data=x_2)
    hf.create_dataset('y', data=y)
    hf.create_dataset('coord', data=coord)
    hf.close()


