import torch.nn as nn
import torch.nn.functional as F
import torch
# from collections import OrderedDict
from model_utils import load_model, save_model
# import torchvision.models as models


class CNN_seg(nn.Module):
	def __init__(self,features = 16,drop_out = 0.0,mode = "none"):
		super(CNN_seg, self).__init__()
		self.features = features
		self.mode = mode
		self.max_pool = nn.MaxPool3d((2,2,1),stride = (2,2,1))
		self.conv1 = ConvBlock3D(1,features,drop_out)
		self.conv2 = ConvBlock3D(features,features*2,drop_out)
		self.conv3 = ConvBlock3D(features*2,features*4,drop_out)
		self.conv4 = ConvBlock3D(features*4,features*8,drop_out)
        
		self.fc1 = nn.Linear(50*50*features*8*12, 128) #48*48*128, 128)
		self.fc2 = nn.Linear(128, 64)
		self.fc3 = nn.Linear(64, 3)

	def load(self, path, filename, mode = "single", device = None):
		load_model(self, path = path, model_name = filename, mode = mode, device = device)

	def save(self, path, filename, optimizer = None):
		save_model(self, optimizer, path, filename)

	def forward(self,x):
		x = self.conv1(x)
		x1 = self.max_pool(self.conv2(x))
		x2 = self.max_pool(self.conv3(x1))
		x3 = self.max_pool(self.conv4(x2))
		x4 = torch.flatten(x3,1)
		x5 = F.relu(self.fc1(x4))
		x6 = F.relu(self.fc2(x5))
		out = self.fc3(x6)
		return out

class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, drop_out = 0, mode = None):
        super(ConvBlock3D, self).__init__()
        self.mode = mode
        self.drop = nn.Dropout3d(drop_out)
        self.conv = nn.Conv3d(in_channels, out_channels,3, stride=1,padding = 1, bias = False)
        self.bn = nn.BatchNorm3d(out_channels, affine = False, track_running_stats = False)
    def forward(self, x):
        out = self.conv(x)
        out = self.drop(out)
        out = F.relu(self.bn(out))
        if self.mode =="res":
            return out + x
        else:
            return out
            
class CNN_seg2(nn.Module):
	def __init__(self,features = 16,drop_out = 0.0,mode = "none"):
		super(CNN_seg2, self).__init__()
		self.features = features
		self.mode = mode
		self.max_pool = nn.MaxPool3d((2,2,1),stride = (2,2,1))
		self.conv1 = ConvBlock3D(1,features,drop_out)
		self.conv2 = ConvBlock3D(features,features*2,drop_out)
		self.conv3 = ConvBlock3D(features*2,features*4,drop_out)
		self.conv4 = ConvBlock3D(features*4,features*8,drop_out)
		self.conv5 = ConvBlock3D(features*8,features*16,drop_out)
		
        
		self.fc1 = nn.Linear(25*25*features*16*8, features*16) #48*48*128, 128)
		#self.fc2 = nn.linear(50*50*features, 50*50*features)
		self.fc3 = nn.Linear(features*16, 128)
		#self.fc4 = nn.Linear(50*50, 50)
		self.fc5 = nn.Linear(128, 3)


	def load(self, path, filename, mode = "single", device = None):
		load_model(self, path = path, model_name = filename, mode = mode, device = device)

	def save(self, path, filename, optimizer = None):
		save_model(self, optimizer, path, filename)

	def forward(self,x):
		x = self.conv1(x)
		x1 = self.max_pool(self.conv2(x))
		x2 = self.max_pool(self.conv3(x1))
		x3 = self.max_pool(self.conv4(x2))
		x4 = self.max_pool(self.conv5(x3))		
		x5 = torch.flatten(x4,1)
		x6 = F.relu(self.fc1(x5))
		#x7 = F.relu(self.fc2(x6))
		x8 = F.relu(self.fc3(x6))
		#x9 = F.relu(self.fc4(x8))
		out = self.fc5(x8)
		return out


class CNNClassifier(nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, stride=1):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=stride),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(3, 2, 1),  ## stride=2
                torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(3, 2, 1)  ## stride=2
            )
            self.downsample = None
            if n_input != n_output or stride != 1:
                self.downsample = torch.nn.Sequential(torch.nn.Conv2d(n_input, n_output, 1, stride=stride),
                                                      torch.nn.BatchNorm2d(n_output),
                                                      torch.nn.MaxPool2d(3, 2, 1),
                                                      torch.nn.MaxPool2d(3, 2, 1))

        def forward(self, x):
            identity = x
            if self.downsample is not None:
                identity = self.downsample(identity)
            # print(self.net(x).shape)
            # print(identity.shape)
            return self.net(x) + identity

    def __init__(self, layers=[64,128], n_input_channels=1, kernel_size=3, stride=2):
        super().__init__()
        L = [
            # ## it's efficient to first use a large conv for high-res inputs
            # torch.nn.Conv2d(n_input_channels, 32, kernel_size=7, padding=3, stride=2),
            # torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            ## this pooling layer hurts performance a little bit but accelerates training
        ]
        c = n_input_channels  # 32
        for l in layers:
            L.append(self.Block(c, l, stride=stride))
            c = l
        self.network = nn.Sequential(*L)
        self.classifier = nn.Linear(8192, 3) #8192, 3) #3200, 3) #2048, 4) #6272, 4) #64*7*7, 4)

    def load(self, path, filename, mode = "single", device = None):
        load_model(self, path = path, model_name = filename, mode = mode, device = device)
        
    def save(self, path, filename, optimizer = None):
        save_model(self, optimizer, path, filename)

    def forward(self, x):
        out = []
        # print(x.shape)
        for slc in range(x.shape[2]):
            # print(self.network(x[:,:,slc]).shape)
            tmp = torch.flatten(self.network(x[:,:,slc]),1)
            # print(tmp.shape)
            out.append(self.classifier(tmp)) #(self.network(x[:,:,slc]).mean(dim=[2, 3]))))
        out = torch.stack(out,dim=2)
        # print(out.shape)
        out = out.mean(dim=[2])
        # print(out.shape)
        return out


class CNNClassifier2(nn.Module):
    class Block(nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.c1 = nn.Conv2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                      stride=stride, bias=False)
            self.c2 = nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
            self.c3 = nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
            self.b1 = nn.BatchNorm2d(n_output)
            self.b2 = nn.BatchNorm2d(n_output)
            self.b3 = nn.BatchNorm2d(n_output)
            self.skip = nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride)

        def forward(self, x):
            return F.relu(self.b3(self.c3(F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x)))))))) + self.skip(x))

    def __init__(self, layers=[32,64], n_output_channels=4, kernel_size=3):
        super().__init__()
        # self.input_mean = torch.Tensor([0.3521554, 0.30068502, 0.28527516])
        # self.input_std = torch.Tensor([0.18182722, 0.18656468, 0.15938024])

        L = []
        c = 1
        for l in layers:
            L.append(self.Block(c, l, kernel_size, 2))
            c = l
        self.network = nn.Sequential(*L)
        self.classifier = nn.Linear(640000, n_output_channels)

    def load(self, path, filename, mode = "single", device = None):
        load_model(self, path = path, model_name = filename, mode = mode, device = device)
        
    def save(self, path, filename, optimizer = None):
        save_model(self, optimizer, path, filename)

    def forward(self, x):
        # z = self.network((x - self.input_mean[None, :, None, None].to(x.device)) / self.input_std[None, :, None, None].to(x.device))
        # return self.classifier(z.mean(dim=[2, 3]))
        out = []
        for slc in range(x.shape[2]):
            # print(torch.flatten(self.network(x[:,:,slc]),1).shape)
            tmp = torch.flatten(self.network(x[:,:,slc]),1)
            out.append(self.classifier(tmp)) #(self.network(x[:,:,slc]).mean(dim=[2, 3]))))
        out = torch.stack(out,dim=2).mean(dim=[2])
        # print(out.shape)
        return out

