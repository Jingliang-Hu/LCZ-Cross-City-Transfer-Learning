import sys
sys.path.insert(0,'../src/io')
sys.path.insert(0,'../src/model')

import mapping_uil as map_tool
import numpy as np
import glob
import h5py

tif_data_files_pattern = 'test_cities/*/*autu*.tif'
tif_data_files = glob.glob(tif_data_files_pattern)

patchsize = 32
lcz_resolution = 100
model_names = ['baseline','self_ensemble','dual_student','our_model','upper_boundary']

for i in range(len(tif_data_files)):
    tif_file = tif_data_files[i]
    tif_file_dir = '/'.join(tif_file.split('/')[:-1])

    city = tif_file.split('/')[-2]
    print(city)
    features = map_tool.get_s2_feature(tif_file)

    for j in range(len(model_names)):
        lcz_map_dir = tif_file_dir+'/LCZ_map_'+model_names[j]+'.tif'
        # initial LCZ grid
        map_tool.initialLCZGridsRes(tif_file, lcz_map_dir, lcz_resolution)


    # find the coordinate of each cell in LCZ grid
    coordCell = map_tool.getCoordLCZGrid(lcz_map_dir)
    # find the cells in data grid that correspond to cells in LCZ grid, via coordinates
    coordImage = map_tool.getImageCoordByXYCoord(coordCell,tif_file)
    # get data patches
    dataPatches = map_tool.getPatch(features,coordImage,patchsize)

    data_tmp_dir = tif_file_dir + '/data_patch_tmp.h5'    
    f = h5py.File(data_tmp_dir, 'w')    
    f.create_dataset("dataPatches", data=dataPatches)
    f.close()













