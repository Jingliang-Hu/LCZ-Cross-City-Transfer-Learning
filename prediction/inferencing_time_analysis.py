import sys
sys.path.insert(0,'../src/io')
sys.path.insert(0,'../src/model')

import resnetModel
import modelOperDataLoader
import torch
import numpy as np
import glob
import h5py
import timeit
import os
from scipy import stats

def modelPath(model_name, backbone_name):
    out_path = os.path.join(model_name,backbone_name+'*')
    if model_name == "baseline":
        out_path = os.path.join(out_path,'model')
    elif model_name == "self_ensemble":
        out_path = os.path.join(out_path,'teacher_model')
    elif  model_name == "dual_student":
        out_path = os.path.join(out_path,'student2_model')
    elif model_name == "our_model":
        out_path = os.path.join(out_path,'model_stream*')
    out_path = glob.glob(out_path)
    return out_path


models = ["baseline","self_ensemble","dual_student","our_model"]
backbone = ["LeNet","Sen2LCZ","ResNet"]

cuda_dev = torch.device('cuda:1')

repeat_times = 5
times = np.zeros((len(models),len(backbone),repeat_times))


# load the data
city_tmp_h5data = glob.glob('test_cities/Syndey/data_patch_tmp.h5')
data_file = city_tmp_h5data[0]
f = h5py.File(data_file)
dataPatches = np.array(f["dataPatches"])
print(dataPatches.shape)
f.close()
print(data_file.split('/')[-2])

for i in range(repeat_times):
    for j in range(len(models)):
        print('---------------------------')
        print("model: {}".format(models[j]))
        for k in range(len(backbone)):
            print("backbone: {}".format(backbone[k]))
            model_path = modelPath(models[j], backbone[k])
            start = timeit.default_timer()
            trained_model = []
            pred = []
            for m in range(len(model_path)):
                
                if backbone[k]=="LeNet":
                    trained_model.append(resnetModel.LeNet_conv_5(inChannel=10, nbClass = 17))
                elif backbone[k]=="Sen2LCZ":
                    trained_model.append(resnetModel.Sen2LCZ(in_Channel=10, nb_class=17, nb_kernel=16, depth=17, bn_flag=1, drop_rate=0.2))
                elif backbone[k]=="ResNet":
                    trained_model.append(resnetModel.resnet18(pretrained=False, inChannel=10))

                trained_model[m].load_state_dict(torch.load(model_path[m], map_location=torch.device(cuda_dev)))
                pred.append(modelOperDataLoader.prediction_4_mapping_gpu(trained_model[m], dataPatches, cuda_dev))
            if m>0:
                pred = np.stack(pred[:], axis=0)
                pred, _ = stats.mode(pred, axis=0)
            else:
                pred = pred[0]
            stop = timeit.default_timer()
            times[j,k,i] = stop - start

f = h5py.File('inferencing_time.h5','w')
f.create_dataset('time_record', data=times)
f.create_dataset('data_size', data=dataPatches.shape)
f.close()













