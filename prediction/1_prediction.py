import sys
sys.path.insert(0,'../src/io')
sys.path.insert(0,'../src/model')

import resnetModel
import modelOperDataLoader
import torch
import numpy as np
import glob
import h5py

models = ["baseline","self_ensemble","dual_student","our_model","upper_boundary"]

model_paths = ["baseline/LeNet_conv5_tr_lcz42_te_cul10_outcome_2020-10-16_01-45-04/model",
               "self_ensemble/LeNet_conv5_tr_lcz42_te_cul10_outcome_2020-10-16_17-32-45/teacher_model",
               "dual_student/LeNet_conv5_tr_lcz42_te_cul10_outcome_2020-10-13_08-27-35/student2_model",
               "our_model/LeNet_conv5_5_stream_tr_lcz42_te_cul10_outcome_2020-10-13_13-26-29/model_stream*",
               "upper_boundary/LeNet_conv5_tr_cul10_west_te_cul10_east_outcome_2020-10-13_11-13-12/model"]

city_tmp_h5data = glob.glob('test_cities/*/data_patch_tmp.h5')

cuda_dev = torch.device('cuda:2')



for i in range(len(city_tmp_h5data)):
    # load the data
    data_file = city_tmp_h5data[i]
    f = h5py.File(data_file)
    dataPatches = np.array(f["dataPatches"])
    f.close()
    print(data_file.split('/')[-2])
    #
    city_file = '/'.join(data_file.split('/')[:-1])
    for j in range(len(models)):
        model_path = model_paths[j]
        print("model: "+models[j])
        if j == 3:
            trained_model = []
            pred = []
            ensemble_models = glob.glob(model_path)
            for idx in range(len(ensemble_models)):
                trained_model.append(resnetModel.LeNet_conv_5(inChannel=10, nbClass = 17))
                trained_model[idx].load_state_dict(torch.load(ensemble_models[idx], map_location=torch.device(cuda_dev)))
                pred.append(modelOperDataLoader.prediction_4_mapping_gpu(trained_model[idx], dataPatches, cuda_dev))

            pred = np.stack(pred[:], axis=0)
            from scipy import stats
            pred, _ = stats.mode(pred, axis=0)
        else:
            trained_model = resnetModel.LeNet_conv_5(inChannel=10, nbClass = 17)
            trained_model.load_state_dict(torch.load(model_path, map_location=torch.device(cuda_dev)))
            pred = modelOperDataLoader.prediction_4_mapping_gpu(trained_model, dataPatches, cuda_dev)

        f = h5py.File(data_file,'a')
        try:
            f.create_dataset(models[j]+'_pred', data=pred)
        except:
            del f[models[j]+'_pred']
            f.create_dataset(models[j]+'_pred', data=pred)
        f.close()












