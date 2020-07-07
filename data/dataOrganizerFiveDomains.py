import numpy as np
import h5py
import glob
import os

asiaCities = ['beijing.h5','changsha.h5','dhaka.h5','dongying.h5','hongkong.h5','islamabad.h5','istanbul.h5','kyoto.h5','nanjing.h5','orangitown.h5','qingdao.h5','quezon.h5','shanghai.h5','shenzhen.h5','tokyo.h5','wuhan.h5']
euroCities = ['amsterdam.h5','berlin.h5','cologne.h5','lisbon.h5','london.h5','madrid.h5','milan.h5','paris.h5','rome.h5','zurich.h5']
afriCities = ['cairo.h5','capetown.h5']
norACities = ['chicago.h5','losangeles.h5','melbourne.h5','newyork.h5','philadelphia.h5','vancouver.h5','washingtondc.h5']
souACities = ['bogota.h5','buenosaires.h5','caracas.h5','lima.h5','riodejaneiro.h5','salvador.h5','saopaulo.h5']

domains = [asiaCities,euroCities,afriCities,norACities,souACities]
directory = '/data/hu/TF/data/original_dat'

domain_name = ['train_asia.h5','train_euro.h5','train_afri.h5','train_namr.h5','train_samr.h5']
save_dir = '/data/hu/TF/data/train'


for idx in range(len(domains)):
    x_1=[]
    x_2=[]
    y=[]
    lst = domains[idx]
    for city in lst:
        file_dir = os.path.join(directory,city)
        fid = h5py.File(file_dir,'r')
        x_1.append(np.array(fid['sen1']))
        x_2.append(np.array(fid['sen2']))
        y.append(np.array(fid['label']))
        fid.close()

    x_1 = np.array(x_1)
    x_1 = np.concatenate((x_1[:]),axis=0)
    x_2 = np.array(x_2)
    x_2 = np.concatenate((x_2[:]),axis=0)
    y = np.array(y)
    y = np.concatenate((y[:]),axis=0)
    print('-------------------------------')
    print(domain_name[idx])
    print(x_1.shape)
    print(x_2.shape)
    print(y.shape)

    save_file_name = os.path.join(save_dir,domain_name[idx])
    hf = h5py.File(save_file_name,'w')
    hf.create_dataset('x_1', data=x_1)
    hf.create_dataset('x_2', data=x_2)
    hf.create_dataset('y', data=y)
    hf.close()




