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

directory = '/data/hu/TF/experiments/1_meanTeacher_MosMuc_SameClass'
filePattern = 'domain_mean_teacher_tr_munich*'
h5Name = 'teacher_test_accuracy.h5'

h5Files = glob.glob(os.path.join(directory,filePattern))
print(len(h5Files))

c_sum = np.zeros((17,17))
for fi in h5Files:
	f = h5py.File(os.path.join(fi,h5Name),'r')
	c = np.array(f['confusion_matrix'])
	c_sum += c
	f.close()
	
	
c_aver = np.round(c_sum.astype(np.float)/len(h5Files))



# plot confusion matrix
cm_disp_obj = modelOperDataLoader.ConfusionMatrixDisplay(c_aver,np.linspace(1,c_aver.shape[0],c_aver.shape[0], dtype=np.uint8),width = 7, height = 6.5)
cm_disp = cm_disp_obj.plot()
cm_disp.savefig()

