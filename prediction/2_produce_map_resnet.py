import sys
sys.path.insert(0,'../src/io')
sys.path.insert(0,'../src/model')

import mapping_uil as map_tool
import numpy as np
import glob
import h5py
import os

tif_data_files_pattern = 'test_cities/*/*autu*.tif'
tif_data_files = glob.glob(tif_data_files_pattern)

patchsize = 32
lcz_resolution = 100
model_names = ["baseline","self_ensemble","dual_student","our_model","upper_boundary","pseudo_label","ARMAN"]

for i in range(len(tif_data_files)):
    tif_file = tif_data_files[i]
    tif_file_dir = '/'.join(tif_file.split('/')[:-1])

    city = tif_file.split('/')[-2]
    print(city)

    data_tmp_dir = tif_file_dir + '/data_patch_tmp.h5'
    if os.path.exists(data_tmp_dir):
        f = h5py.File(data_tmp_dir, 'r')
    else:
        print("Intermediate data file does not exist for the city: {}. {}".format(city,data_tmp_dir))
        continue

    for j in range(len(model_names)):
        lcz_map_dir = tif_file_dir+'/LCZ_map_resnet_'+model_names[j]+'.tif'
        lcz_rgb_dir = tif_file_dir+'/LCZ_rgb_resnet_'+model_names[j]+'.tif'

        variable_name = model_names[j]+'_resnet_pred'
        pred = np.array(f[variable_name])
        if pred.ndim == 2:
            pred = np.squeeze(pred)

        basemap = tif_file_dir+'/LCZ_map_baseline.tif '
        command = "cp "+basemap+lcz_map_dir
        os.system(command)
        map_tool.saveLabelPrediction(pred,lcz_map_dir)
        map_tool.create_LCZ_map_in_rgb(lcz_map_dir, lcz_rgb_dir)


    f.close()




















