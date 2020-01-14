import h5py
import numpy as np



cul10Cities = ['mumbai.h5', 'santiago.h5', 'sydney.h5', 'munich.h5', 'guangzhou.h5', 'tehran.h5', 'moscow.h5', 'nairobi.h5', 'sanfrancisco.h5', 'jakarta.h5']


for n in cul10Cities:
    f = h5py.File(n,'r')
    if n=="mumbai.h5":
        coord = np.array(f['coord'])
        x_1 = np.array(f['x_1'])
        x_2 = np.array(f['x_2'])
        y = np.array(f['y'])
    else:
        tmp = np.array(f['coord'])
        coord = np.concatenate((coord,tmp),axis=0)
        tmp = np.array(f['x_1'])
        x_1 = np.concatenate((x_1,tmp),axis=0)
        tmp = np.array(f['x_2'])
        x_2 = np.concatenate((x_2,tmp),axis=0)
        tmp = np.array(f['y'])
        y = np.concatenate((y,tmp),axis=0)
    f.close()
    del f


f = h5py.File('cul10.h5','w')
f.create_dataset('coord',data=coord)
f.create_dataset('x_1',data=x_1)
f.create_dataset('x_2',data=x_2)
f.create_dataset('y',data=y)

f.close()




