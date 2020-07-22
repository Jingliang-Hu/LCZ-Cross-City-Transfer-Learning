import sys
import os
fid = open("../../.envPath","r")
envPath = fid.readline()
envPath = envPath[:-1]
fid.close
del fid
sys.path.append(os.path.abspath(envPath+"/src/model"))

import glob
import h5py
import numpy as np
import modelOperDataLoader


experiments = "LeNet_ensemble_tr_lcz42_te_cul10_outcome"
output_file_name = experiments+'_mean_confusion_matrix.png'


files = glob.glob(experiments+'*/*test_accuracy.h5')
mean_confusion_matrix = np.zeros((17,17))

for fil in files:
    f = h5py.File(fil)
    mean_confusion_matrix += np.array(f['confusion_matrix'])

mean_confusion_matrix = np.round(mean_confusion_matrix/len(files))
print(np.diag(mean_confusion_matrix))
print(np.sum(np.diag(mean_confusion_matrix)))
print(np.sum(np.diag(mean_confusion_matrix))/np.sum(mean_confusion_matrix))



cm_disp_obj = modelOperDataLoader.ConfusionMatrixDisplay(mean_confusion_matrix,np.linspace(1,mean_confusion_matrix.shape[0],mean_confusion_matrix.shape[0]).astype(int))
cm_disp = cm_disp_obj.plot()
cm_disp.figure_.savefig(os.path.join('.',output_file_name))


