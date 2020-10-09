import sys
sys.path.insert(0,'../src/io')
sys.path.insert(0,'../src/model')

import mapping_uil as map_tool
import resnetModel
import modelOperDataLoader
import torch
import numpy as np

path_2_s2_data = '../data/test_cities/Munich/204371_autumn.tif'

city = path_2_s2_data.split('/')[-2]
output_directory = '/'.join(path_2_s2_data.split('/')[:-1])
# set resolution for a LCZ grid
lcz_resolution = 100
# mapping models
models_name = ['baseline']#,'self_ensemble','dual_student','ensemble']

# mapping model saving paths
baseline_model = '../experiments/2_baseline_42/LeNet_tr_lcz42_te_cul10_outcome_2020-10-09_07-01-21/model'
self_ensemble_model = ''
dual_student_model = ''
my_model = ''


# patchsize for prediction
patchsize = 32
# gpu device
cuda_dev = torch.device('cuda:0')


# s2 feature preparation
data_feature = map_tool.get_s2_feature(path_2_s2_data)
print(data_feature.dtype)
data_feature = data_feature.astype(np.float)/1e4
print(data_feature.mean())

for i in range(len(models_name)):
    model = models_name[i]
    lcz_map_dir = output_directory + '/' + city + '_' + model + '_LCZ.tif'
    lcz_rgb_dir = output_directory + '/' + city + '_' + model + '_LCZ_rgb.tif'

    # initial LCZ grid
    map_tool.initialLCZGridsRes(path_2_s2_data, lcz_map_dir, lcz_resolution)
    # find the coordinate of each cell in LCZ grid
    coordCell = map_tool.getCoordLCZGrid(lcz_map_dir)
    # find the cells in data grid that correspond to cells in LCZ grid, via coordinates
    coordImage = map_tool.getImageCoordByXYCoord(coordCell,path_2_s2_data)
    # get data patches
    dataPatches = map_tool.getPatch(data_feature,coordImage,patchsize)

    if model == 'baseline':
        model = resnetModel.LeNet(inChannel=10, nbClass = 17)
        # model.load_state_dict(torch.load(models_path[i], map_location=torch.device('cpu')))
        model.load_state_dict(torch.load(baseline_model, map_location=torch.device(cuda_dev)))
        pred = modelOperDataLoader.prediction_4_mapping_gpu(model,dataPatches,cuda_dev)
    elif model == 'self_ensemble':
        model = resnetModel.LeNet(inChannel=10, nbClass = 17)
        model.load_state_dict(torch.load(self_ensemble_model, map_location=torch.device(cuda_dev)))
        pred = modelOperDataLoader.prediction_4_mapping_gpu(model,dataPatches,cuda_dev)
    elif model == 'dual_student':
        model = resnetModel.LeNet(inChannel=10, nbClass = 17)
        model.load_state_dict(torch.load(dual_student_model, map_location=torch.device(cuda_dev)))
        pred = modelOperDataLoader.prediction_4_mapping_gpu(model,dataPatches,cuda_dev)

    elif model == 'ensemble':
        model = resnetModel.LeNet(inChannel=10, nbClass = 17)
        model.load_state_dict(torch.load(my_model, map_location=torch.device(cuda_dev)))
        pred = modelOperDataLoader.prediction_4_mapping_gpu(model,dataPatches,cuda_dev)




    map_tool.saveLabelPrediction(pred,lcz_map_dir)
    map_tool.create_LCZ_map_in_rgb(lcz_map_dir, lcz_rgb_dir)




