import os
import torch
import argparse
import functools
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import nibabel as nib
import random

import dataloader_Huili
import models_classification_Huili

import numpy as np
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
import hdf5storage
import time
import h5py
import torchvision.transforms as TT

fliplr = TT.RandomHorizontalFlip(p = 0.5)
flipud = TT.RandomVerticalFlip(p = 0.5)
imrotate = TT.RandomRotation(180)

def transforms(orig):
    trans = fliplr(orig)
    trans = flipud(trans)
    trans = imrotate(trans[:,0])[:,None]
    # trans = imrotate(np.squeeze(trans))[None][None]
    return trans


def main():
    """
    The main function of your running scripts. 
    """
    # default data folder
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, nargs='?', default='/input', help='input directory')
    parser.add_argument('--output', type=str, nargs='?', default='/output', help='output directory')
    args = parser.parse_args()

    ## functions are not real python functions, but are examples here.

    ## Read in your trained model
    model_name = ["MICCAI_Classification_TEST7_ensemble_im512_aug_bs2_shuffle_0.pth", 
            "MICCAI_Classification_TEST7_ensemble_im512_aug_bs2_shuffle_1.pth",
            "MICCAI_Classification_TEST7_ensemble_im512_aug_bs2_shuffle_2.pth",
            "MICCAI_Classification_TEST7_ensemble_im512_aug_bs2_shuffle_3.pth",
            "MICCAI_Classification_TEST7_ensemble_im512_aug_bs2_shuffle_4.pth",]
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mrinet = []
    for ii in range(len(model_name)):
        mrinet.append(models_classification_Huili.CNNClassifier().to(device))
        mrinet[ii].load(path = "./models", filename = model_name[ii])
        # mrinet.load(path = model_path, filename = model_name)

    ## Read in input files
    input_files = sorted([args.input + '/' + f for f in os.listdir(args.input)])
   
    ## dataloader
    dataset_valid = dataloader_Huili.Loader_classification_valid(input_files, imsize = 512, t_slices = 12) #-1)
    loader_valid = DataLoader(dataset_valid,
            batch_size = 2,
            shuffle = False,
            drop_last = False,
            num_workers = 0)
    
    ## inference
    out_list = []
    m = nn.Softmax(dim=1)
    for model in mrinet:
        model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(loader_valid):
            x = data["x"].to(device)
            prob_all = []
            for model in mrinet:
                valid_out1 = model(transforms(x))
                valid_out2 = model(transforms(x))
                valid_out3 = model(transforms(x))
                valid_out = (valid_out1+valid_out2+valid_out3)/3
                prob_all.extend(m(valid_out)[None])
            prob = torch.mean(torch.stack(prob_all),dim=0)
            out_list.extend((torch.argmax(prob,dim=1,keepdim =True)+1).detach().cpu().numpy()[:,0])
    
    valid_list = [f.split('/')[-1].split('.nii.gz')[0] for f in input_files]
    
    ## output file
    df = pd.DataFrame(list(zip(valid_list, out_list)), columns=["Image", "Label"])
    df.to_csv(args.output + '/output.csv', index=False)

if __name__ == "__main__":
	main()