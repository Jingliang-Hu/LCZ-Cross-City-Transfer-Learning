import sys
sys.path.insert(0,'../src/io')
sys.path.insert(0,'../src/model')

import mapping_uil as map_tool
import resnetModel
import modelOperDataLoader


path_2_s2_data = '/datastore/DATA/classification/SEN2/SEN2_LCZ42/LCZ52_utm_s2/LCZ42_204371_Munich/autumn/204371_autumn.tif'

output_directory = './Munich'


# set resolution for a LCZ grid
lcz_resolution = 100
# mapping models
models_name = ['baseline']#,'self_ensemble','dual_student','ensemble']
# mapping model saving paths
models_path = [ '../experiments/2_baseline_42/LeNet_tr_lcz42_te_cul10_outcome_2020-04-28_11-08-20/model']
'''
,...
		'',...
                '',...
                '',...
]
'''
# patchsize for prediction
patchsize = 32


# s2 feature preparation
data_feature = map_tool.get_s2_feature(path_2_s2_data)


for i in range(len(models_name)):
    model = models_name(i)
    lcz_map_dir = output_directory + '/' + model + '_LCZ.tif'
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
        model.load_state_dict(torch.load(modelPath[i]))
        pred = modelOperDataLoader.prediction_4_mapping(model,dataPatches)


    modelOperDataLoader.saveLabelPrediction(pred,lcz_map_dir)





