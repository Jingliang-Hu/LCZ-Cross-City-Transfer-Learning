import sys
sys.path.insert(0,'../src/io')
sys.path.insert(0,'../src/model')
sys.path.insert(0,'../experiments/9_amran/AMRAN-master')
import ResNet as arman_model
import resnetModel
import modelOperDataLoader
import torch
import numpy as np
import glob
import h5py

models = ["baseline","self_ensemble","dual_student","our_model","upper_boundary","pseudo_label","ARMAN"]

model_paths = [
               "baseline/ResNet_tr_lcz42_te_cul10_outcome_2020-10-13_16-54-23/model",
               "self_ensemble/ResNet_tr_lcz42_te_cul10_outcome_2020-10-16_17-11-06/teacher_model",
               "dual_student/ResNet_tr_lcz42_te_cul10_outcome_2020-10-16_15-19-46/student2_model",
               "our_model/ResNet_5_stream_tr_lcz42_te_cul10_outcome_2020-10-23_10-20-39/model_stream*",
               "upper_boundary/ResNet_tr_cul10_west_te_cul10_east_outcome_2020-10-09_16-33-29/model",
               "pseudo_label/ResNet_tr_lcz42_te_cul10_outcome_2021-05-26_00-25-00/fine_tuning_model",
               "arman/amran_tr_lcz42_te_cul10_outcome_2021-05-20_22-32-50/model"]

city_tmp_h5data = glob.glob('test_cities/*_1/data_patch_tmp.h5')

cuda_dev = torch.device('cuda:7')



for i in range(len(city_tmp_h5data)):
    # load the data
    data_file = city_tmp_h5data[i]
    print(data_file.split('/')[-2])

    f = h5py.File(data_file)
    dataPatches = np.array(f["dataPatches"])
    f.close()
    #
    city_file = '/'.join(data_file.split('/')[:-1])
    for j in range(len(models)):
        model_path = model_paths[j]
        print("model: "+models[j])
        f = h5py.File(data_file,'a')
        if models[j]+'_resnet_pred' in f.keys():
            print("prediction results exist")
            f.close()
            continue
        if j == 3:
            trained_model = []
            pred = []
            ensemble_models = glob.glob(model_path)
            for idx in range(len(ensemble_models)):
                trained_model.append(resnetModel.resnet18(pretrained=False, inChannel=10).to(cuda_dev))
                #trained_model.append(resnetModel.LeNet_conv_5(inChannel=10, nbClass = 17))
                trained_model[idx].load_state_dict(torch.load(ensemble_models[idx], map_location=torch.device(cuda_dev)))
                pred.append(modelOperDataLoader.prediction_4_mapping_gpu(trained_model[idx], dataPatches, cuda_dev))

            pred = np.stack(pred[:], axis=0)
            from scipy import stats
            pred, _ = stats.mode(pred, axis=0)
        elif j == 6:
            trained_model = arman_model.AMRANNet(num_classes=17).to(cuda_dev)
            trained_model.load_state_dict(torch.load(model_path, map_location=torch.device(cuda_dev)))
            pred = modelOperDataLoader.arman_prediction_4_mapping_gpu(trained_model, dataPatches, cuda_dev)
        else:
            trained_model = resnetModel.resnet18(pretrained=False, inChannel=10).to(cuda_dev)
            #trained_model = resnetModel.LeNet_conv_5(inChannel=10, nbClass = 17)
            trained_model.load_state_dict(torch.load(model_path, map_location=torch.device(cuda_dev)))
            pred = modelOperDataLoader.prediction_4_mapping_gpu(trained_model, dataPatches, cuda_dev)

        try:
            f.create_dataset(models[j]+'_resnet_pred', data=pred)
        except:
            del f[models[j]+'_pred']
            f.create_dataset(models[j]+'_resnet_pred', data=pred)
        f.close()












