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
plt.legend(('training accuracy', 'test accuracy'), loc='bottom right', shadow=True)
plt.xlim([1,traArry.shape[0]+1])
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Accuracy in percentage')
plt.title('ResNet_18_train_mos_test_muc')
plt.show()


plt.plot(epoch, traLoss, 'r', epoch, valLoss, 'b')
plt.legend(('training loss', 'test loss'), loc='bottom right', shadow=True)
plt.xlim([1,traArry.shape[0]+1])
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('ResNet_18_train_mos_test_muc')
plt.show()







