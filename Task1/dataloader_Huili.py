import os
import torch
import functools
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import nibabel as nib
import numpy as np
import torch
from collections import OrderedDict
import os
import hdf5storage
#import pywt
from scipy import misc


class Loader_classification(Dataset):
	def __init__(self, x_files , lbls, imsize = 144, t_slices = -1):
		super(Loader_classification, self).__init__()
		self.x_files = x_files
		self.imsize = imsize
		self.t_slices = t_slices
		self.lbls = lbls

	def crop(self, data, imsize,t_slices):
		nx = data.shape[1]
		ny = data.shape[2]
		nz = data.shape[0]
		res1 = (nx-imsize)//2
		res2 = (ny-imsize)//2
		new = np.zeros([data.shape[0], imsize, imsize])
		new[:, max(0,-res1):imsize-max(0,-res1), max(0,-res2):imsize-max(0,-res2)] = \
			data[:, max(0,res1):nx-max(0,res1), max(0,res2):ny-max(0,res2)]
		if t_slices == -1:
			return new
		else:
			if nz%2 == 1:
				nz = nz-1
			res0 = (nz-t_slices)//2
			new1 = np.zeros([t_slices, imsize, imsize])
			new1[max(0,-res0):t_slices-max(0,-res0)] = \
				new[max(0,res0):nz-max(0,res0)]
			return new1

	def __len__(self):
		return len(self.x_files)

	def __getitem__(self, idx):
		x = np.squeeze(np.array(list(nib.load(self.x_files[idx]).get_fdata())))
		y_files = self.x_files[idx][-16:-7]
		y = int(self.lbls[y_files])		
		x = np.transpose(x, (2, 0, 1))
		x = self.crop(x, self.imsize, self.t_slices)
		x = x/np.percentile(np.abs(x), 98)

		return {"x": torch.FloatTensor(x)[None], "y": y}

class Loader_classification_hardatten(Dataset):
	def __init__(self, x_files , lbls, y_files, imsize = 144, t_slices = -1):
		super().__init__()
		self.x_files = x_files
		self.y_files = y_files
		self.imsize = imsize
		self.t_slices = t_slices
		self.lbls = lbls

	def crop_slc_center(self, data, imsize, center):
		# print(center, imsize, data.shape)
		return data[center[0]-imsize//2: center[0]+imsize//2, center[1]-imsize//2: center[1]+imsize//2]

	def __len__(self):
		return len(self.x_files)

	def __getitem__(self, idx):
		x = np.squeeze(np.array(list(nib.load(self.x_files[idx]).get_fdata())))
		y = np.squeeze(np.array(list(nib.load(self.y_files[idx]).get_fdata())))	
		x = np.transpose(x, (2, 0, 1))
		y = np.transpose(y, (2, 0, 1))
		nz = x.shape[0]
		if self.t_slices != -1: 
			if nz%2 == 1:
				nz = nz-1
			res0 = (nz-self.t_slices)//2
			x = x[res0: nz-res0]
			y = y[res0: nz-res0]

		# center = []
		x1 = []
		for slc in range(y.shape[0]):
			if np.sum(y[slc]) != 0: #labels available
				center = np.array([y[slc].nonzero()[0].mean(), y[slc].nonzero()[1].mean()])
				center = np.round(center).astype(int)
				x1.append(self.crop_slc_center(x[slc], self.imsize, center))
		x1 = np.array(x1)

		lbl_file = self.x_files[idx][-16:-7]
		lbl = int(self.lbls[lbl_file])		
		x1 = x1/np.percentile(np.abs(x1), 98)

		return {"x": torch.FloatTensor(x1)[None], "y": lbl}

class Loader_classification_valid(Dataset):
	def __init__(self, x_files, imsize = 144, t_slices = -1):
		super().__init__()
		self.x_files = x_files
		self.imsize = imsize
		self.t_slices = t_slices
		# self.lbls = lbls

	def crop(self, data, imsize,t_slices):
		nx = data.shape[1]
		ny = data.shape[2]
		nz = data.shape[0]
		res1 = (nx-imsize)//2
		res2 = (ny-imsize)//2
		new = np.zeros([data.shape[0], imsize, imsize])
		new[:, max(0,-res1):imsize-max(0,-res1), max(0,-res2):imsize-max(0,-res2)] = \
			data[:, max(0,res1):nx-max(0,res1), max(0,res2):ny-max(0,res2)]
		if t_slices == -1:
			return new
		else:
			if nz%2 == 1:
				nz = nz-1
			res0 = (nz-t_slices)//2
			new1 = np.zeros([t_slices, imsize, imsize])
			new1[max(0,-res0):t_slices-max(0,-res0)] = \
				new[max(0,res0):nz-max(0,res0)]
			return new1

	def __len__(self):
		return len(self.x_files)

	def __getitem__(self, idx):
		x = np.squeeze(np.array(list(nib.load(self.x_files[idx]).get_fdata())))
		#print(x.shape)
		# y_files = self.x_files[idx][-16:-7]
		# y = int(self.lbls[y_files])		
		x = np.transpose(x, (2, 0, 1))
		x = self.crop(x, self.imsize, self.t_slices)
		x = x/np.percentile(np.abs(x), 98)

		return {"x": torch.FloatTensor(x)[None]} #, "y": y}

class Loader_segmentation(Dataset):
	def __init__(self, x_files , y_files, imsize = 144, t_slices = -1, is_test = 0):
		super(Loader_segmentation, self).__init__()
		self.x_files = x_files
		self.y_files = y_files
		self.imsize = imsize
		self.t_slices = t_slices
		self.is_test = is_test
		#self.lbls = lbls

	def crop_FOV(self, data, imsize):
		nx = data.shape[1]
		ny = data.shape[2]
		res1 = (nx-imsize)//2
		res2 = (ny-imsize)//2
		new = np.zeros([data.shape[0], imsize, imsize])
		new[:, max(0,-res1):imsize-max(0,-res1), max(0,-res2):imsize-max(0,-res2)] = \
			data[:, max(0,res1):nx-max(0,res1), max(0,res2):ny-max(0,res2)]
		return new
		
	def crop_Slice(self, data, t_slices):
		nz = data.shape[0]
		if nz%2 == 1:
		    nz = nz-1
		res0 = (nz-t_slices)//2
		return data[res0: nz-res0,:]

	def __len__(self):
		return len(self.x_files)

	def __getitem__(self, idx):

		x = np.squeeze(np.array(list(nib.load(self.x_files[idx]).get_fdata())))
		x = np.transpose(x, (2, 0, 1))
		s = x.shape
		x = self.crop_FOV(x, self.imsize)
		if self.t_slices != -1:
		    x = self.crop_Slice(x, self.t_slices)
		x = x/np.percentile(np.abs(x), 98)

		if not self.is_test:
			y = np.squeeze(np.array(list(nib.load(self.y_files[idx]).get_fdata())))
			y = np.transpose(y, (2, 0, 1))
			y = self.crop_FOV(y, self.imsize)
			if self.t_slices != -1:
				y = self.crop_Slice(y, self.t_slices)
			return {"x": torch.FloatTensor(x)[None], "y": torch.FloatTensor(y), "s": torch.FloatTensor(s)}

		return {"x": torch.FloatTensor(x)[None], "s": torch.FloatTensor(s)}

class Loader_segmentation_hardatten(Dataset):
	def __init__(self, x_files , y_files, imsize = 144, t_slices = -1):
		super().__init__()
		self.x_files = x_files
		self.y_files = y_files
		self.imsize = imsize
		self.t_slices = t_slices

	def crop_slc_center(self, data, imsize, center):
		# print(center, imsize, data.shape)
		return data[center[0]-imsize//2: center[0]+imsize//2, center[1]-imsize//2: center[1]+imsize//2]

	def __len__(self):
		return len(self.x_files)

	def __getitem__(self, idx):
		x = np.squeeze(np.array(list(nib.load(self.x_files[idx]).get_fdata())))
		y = np.squeeze(np.array(list(nib.load(self.y_files[idx]).get_fdata())))
		x = np.transpose(x, (2, 0, 1))
		y = np.transpose(y, (2, 0, 1))
		nz = x.shape[0]
		if self.t_slices != -1: 
			if nz%2 == 1:
				nz = nz-1
			res0 = (nz-self.t_slices)//2
			x = x[res0: nz-res0]
			y = y[res0: nz-res0]

		center_all = []
		x1 = []
		y1 = []
		for slc in range(y.shape[0]):
			if np.sum(y[slc]) != 0: #labels available
				center = np.array([y[slc].nonzero()[0].mean(), y[slc].nonzero()[1].mean()])
				center = np.round(center).astype(int)
				center_all.append(center)
				x1.append(self.crop_slc_center(x[slc], self.imsize, center))
				y1.append(self.crop_slc_center(y[slc], self.imsize, center))
		x1 = np.array(x1)
		y1 = np.array(y1)
	
		x1 = x1/np.percentile(np.abs(x1), 98)

		return {"x": torch.FloatTensor(x1)[None], "y": torch.FloatTensor(y1)}

class Loader_segmentation_regress(Dataset):
	def __init__(self, x_files , y_files, t_slices = -1):
		super().__init__()
		self.x_files = x_files
		self.y_files = y_files
		self.t_slices = t_slices

	def __len__(self):
		return len(self.x_files)

	def __getitem__(self, idx):
		x = np.squeeze(np.array(list(nib.load(self.x_files[idx]).get_fdata())))
		y = np.squeeze(np.array(list(nib.load(self.y_files[idx]).get_fdata())))
		x = np.transpose(x, (2, 0, 1))
		y = np.transpose(y, (2, 0, 1))
		nz = x.shape[0]
		if self.t_slices != -1: 
			if nz%2 == 1:
				nz = nz-1
			res0 = (nz-self.t_slices)//2
			x = x[res0: nz-res0]
			y = y[res0: nz-res0]
	
		x = x/np.percentile(np.abs(x), 98)

		return {"x": torch.FloatTensor(x)[None], "y": torch.FloatTensor(y)}

class Loader_regression(Dataset):
	def __init__(self, x_files , y_files): #, imsize = 144, t_slices = -1):
		super(Loader_regression, self).__init__()
		self.x_files = x_files
		self.y_files = y_files

	def __len__(self):
		return len(self.x_files)

	def __getitem__(self, idx):

		x = np.squeeze(np.array(list(nib.load(self.x_files[idx]).get_fdata())))
		y = np.squeeze(np.array(list(nib.load(self.y_files[idx]).get_fdata())))	
		x = np.transpose(x, (2, 0, 1))
		y = np.transpose(y, (2, 0, 1))

		x = x/np.percentile(np.abs(x), 98)
		center = []
		for slc in range(y.shape[0]):
			if np.sum(y[slc]) != 0: #labels available
				center.append(np.array([y[slc].nonzero()[0].mean(), y[slc].nonzero()[1].mean()]))
		# center = np.squeeze(np.array([y.nonzero()[1].mean(), y.nonzero()[2].mean()]))

		return {"x": torch.FloatTensor(x)[None], "y": torch.FloatTensor(y), "center": torch.FloatTensor(center)}






