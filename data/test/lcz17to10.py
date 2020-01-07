import numpy as np
import h5py

h5File = 'munich'

f = h5py.File(h5File+'.h5','r')
x_1 = np.array(f['x_1'])
x_2 = np.array(f['x_2'])
coord = np.array(f['coord'])
y = np.array(f['y'])
idx = y.copy()
f.close()

# delete the 7th class
x_1 = np.delete(x_1,np.where(idx[:,6]==1),axis=0)
x_1 = x_1.astype(np.float32)
x_2 = np.delete(x_2,np.where(idx[:,6]==1),axis=0)
x_2 = x_2.astype(np.float32)
coord = np.delete(coord,np.where(idx[:,6]==1),axis=0)
coord = coord.astype(np.float32)
y = np.delete(y,np.where(idx[:,6]==1),axis=0)
idx = y.copy()

# change the classes
y = np.zeros((idx.shape[0],10))
y[idx[:,0]==1,0]=1
y[idx[:,1]==1,0]=1
y[idx[:,2]==1,0]=1

y[idx[:,3]==1,1]=1
y[idx[:,4]==1,1]=1
y[idx[:,5]==1,1]=1

y[idx[:,7]==1,2]=1
y[idx[:,9]==1,2]=1

y[idx[:,10]==1,3]=1
y[idx[:,11]==1,4]=1
y[idx[:,12]==1,5]=1
y[idx[:,13]==1,6]=1
y[idx[:,14]==1,7]=1
y[idx[:,15]==1,8]=1
y[idx[:,16]==1,9]=1

# save output h5 data
f = h5py.File(h5File+'1.h5','w')
f.create_dataset('x_1',data=x_1)
f.create_dataset('x_2',data=x_2)
f.create_dataset('coord',data=coord)
f.create_dataset('y',data=y)
f.close()





