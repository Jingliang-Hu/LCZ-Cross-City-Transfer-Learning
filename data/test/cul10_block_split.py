import h5py
import numpy as np


#% training percentage
perc = 0.5

cul10Cities = ['mumbai.h5', 'santiago.h5', 'sydney.h5', 'munich.h5', 'guangzhou.h5', 'tehran.h5', 'moscow.h5', 'nairobi.h5', 'sanfrancisco.h5', 'jakarta.h5']

count = 0
for n in cul10Cities:
    print(n)
    f = h5py.File(n,'r')
    y = np.array(f['y'])
    x_1 = np.array(f['x_1'])
    x_2 = np.array(f['x_2'])
    coord = np.array(f['coord'])
    
    for i in range(0,y.shape[1]):
        print('class: '+str(i+1))
        idx = np.where(y[:,i]==1)
        if idx[0].shape[0]==0:
            continue    
        split_point = np.floor(idx[0].shape[0]*perc).astype(int)
        idx_w = idx[0][:split_point].astype(int)
        idx_e = idx[0][split_point:].astype(int)
        if count == 0:
            coord_w = coord[idx_w,:]
            x_1_w = x_1[idx_w,:]
            x_2_w = x_2[idx_w,:]
            y_w = y[idx_w,:]        
            coord_e = coord[idx_e,:]
            x_1_e = x_1[idx_e,:]
            x_2_e = x_2[idx_e,:]
            y_e = y[idx_e,:]
            count = count+1    
        else:
            tmp = coord[idx_w,:]
            coord_w = np.concatenate((coord_w,tmp),axis=0)
            tmp = x_1[idx_w,:]
            x_1_w = np.concatenate((x_1_w,tmp),axis=0)
            tmp = x_2[idx_w,:]
            x_2_w = np.concatenate((x_2_w,tmp),axis=0)
            tmp = y[idx_w,:]
            y_w = np.concatenate((y_w,tmp),axis=0)
            tmp = coord[idx_e,:]
            coord_e = np.concatenate((coord_e,tmp),axis=0)
            tmp = x_1[idx_e,:]
            x_1_e = np.concatenate((x_1_e,tmp),axis=0)
            tmp = x_2[idx_e,:]
            x_2_e = np.concatenate((x_2_e,tmp),axis=0)
            tmp = y[idx_e,:]
            y_e = np.concatenate((y_e,tmp),axis=0)                
    f.close()
    del f


f = h5py.File('cul10_west.h5','w')
f.create_dataset('coord',data=coord_w)
f.create_dataset('x_1',data=x_1_w)
f.create_dataset('x_2',data=x_2_w)
f.create_dataset('y',data=y_w)
f.close()

f = h5py.File('cul10_east.h5','w')
f.create_dataset('coord',data=coord_e)
f.create_dataset('x_1',data=x_1_e)
f.create_dataset('x_2',data=x_2_e)
f.create_dataset('y',data=y_e)
f.close()

