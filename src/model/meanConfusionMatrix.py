import sys
import os
fid = open("../../.envPath","r")
envPath = fid.readline()
envPath = envPath[:-1]
fid.close
del fid
sys.path.append(os.path.abspath(envPath+"/src/model"))
import modelOperDataLoader


import numpy as np
import h5py
import os
import glob

directory = '/data/hu/TF/experiments/0_benchMark_MosMuc_SameClass'
filePattern = 'resnet18_benchMark_tr_moscow*'
h5Name = 'test_accuracy.h5'

h5Files = glob.glob(os.path.join(directory,filePattern))
print(len(h5Files))

c_sum = np.zeros((17,17))
for fi in h5Files:
	f = h5py.File(os.path.join(fi,h5Name),'r')
	c = np.array(f['confusion_matrix'])
	c_sum += c
	f.close()
	
	
c_aver = c_sum.astype(np.float)/len(h5Files)



# plot confusion matrix
cm_disp_obj = modelOperDataLoader.ConfusionMatrixDisplay(c_aver,np.linspace(1,c_aver.shape[0],c_aver.shape[0]))
cm_disp = cm_disp_obj.plot()
cm_disp.savefig()

