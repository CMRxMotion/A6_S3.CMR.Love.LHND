import os
import torch
import argparse
from batchgenerators.utilities.file_and_folder_operations import *
import shutil

def main():
    """
    The main function of your running scripts. 
    """
    # default data folder
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, nargs='?', default='/input', help='input directory')
    parser.add_argument('--output', type=str, nargs='?', default='/output', help='output directory')
    args = parser.parse_args()

    tmp_input_folder = '/tmp_input'
    tmp_output_folder = '/tmp_output'
    maybe_mkdir_p(tmp_input_folder)

    for p in subfiles(args.input, join=False):
        fname = 'CM_'+p[2:4]+p[5]
        if p[7:9] == 'ES':
            fname = fname + '1_0000.nii.gz'
        if p[7:9] == 'ED':
            fname = fname + '2_0000.nii.gz'
        shutil.copy(join(args.input, p), join(tmp_input_folder, fname))

    ## Read in your trained model
    os.system("export RESULTS_FOLDER=/workdir/models")
    
    ## inference
    tmp_output_folder_2d = join(tmp_output_folder, 'raw_output_2d')
    tmp_output_folder_3d = join(tmp_output_folder, 'raw_output_3d')
    tmp_output_folder_ensemble = join(tmp_output_folder, 'ensemble')
    os.system("nnUNet_predict -i {} -o {} -tr nnUNetTrainerV2 -m 2d -p nnUNetPlansv2.1 -t Task700_CMRxMotion -z".format(tmp_input_folder, tmp_output_folder_2d))
    os.system("nnUNet_predict -i {} -o {} -tr nnUNetTrainerV2 -m 3d_fullres -p nnUNetPlansv2.1 -t Task700_CMRxMotion -z".format(tmp_input_folder, tmp_output_folder_3d))
    os.system("nnUNet_ensemble -f {} {} -o {}".format(tmp_output_folder_2d, tmp_output_folder_3d, tmp_output_folder_ensemble))
    
    for p in subfiles(tmp_output_folder_ensemble, join=False):
        if p.find('CM_')==-1 or p.find('.nii.gz')==-1:
            continue
        fname = 'P0'+p[3:5]+'-'+p[5]+'-'
        if p[6] == '1':
            fname = fname + 'ES.nii.gz'
        if p[6] == '2':
            fname = fname + 'ED.nii.gz'
        shutil.copy(join(tmp_output_folder_ensemble, p), join(args.output, fname))

if __name__ == "__main__":
	main()