import matplotlib.pyplot as plt
import h5py
import numpy as np

histFile = '/data/Projects/TF/trained_models/benchMarkTrainHistory.h5'

fid = h5py.File(histFile,'r')

traArry = np.array(fid['traArry'])
traLoss = np.array(fid['traLoss'])
valArry = np.array(fid['valArry'])
valLoss = np.array(fid['valLoss'])

epoch = range(1,traArry.shape[0]+1)


plt.plot(epoch, traArry, 'r', epoch, valArry, 'b')
plt.show()


plt.plot(epoch, traLoss, 'r', epoch, valLoss, 'b')
plt.show()






