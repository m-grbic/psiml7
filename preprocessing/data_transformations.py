import numpy as np
import torch
from torch.types import Device
from torchvision import transforms
from sklearn.model_selection import train_test_split
from random import sample, randint
from PIL import Image


#imagenet pretrained normalization
NYUD_MEAN = [0.485, 0.456, 0.406]
NYUD_STD = [0.229, 0.224, 0.225]

def get_split(path='./dataset/', split_ratio=0.1):

    # Input data
    images = np.load(path + "images.npy")
    # Target data
    depths = np.load(path + "depths.npy")

    # Split data on train and test
    X_train, X_test, y_train, y_test = train_test_split(images, depths, test_size=split_ratio, random_state=42)

    # Split train data on train and validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=split_ratio, random_state=42)

    return DataSet(X_train, y_train), DataSet(X_val, y_val), DataSet(X_test, y_test)


class DataSet():

    def __init__(self, images=None, depths=None, dtype='train'):
        self.images = images
        self.depths = depths
        self.N, self.C, self.H, self.W = images.shape
        self.itr = 0
        self.batch_size = 1
        self.N_itr = 1

        # Normalization
        for i in range(3):
            self.images[:,i,:,:] = (self.images[:,i,:,:] - NYUD_MEAN[i]) / NYUD_STD[i]

    def size(self):
        return self.N

    def initBatch(self, batch_size=16):
        # Shuffle indices
        shuffle = np.random.permutation(self.N)

        # Shuffle data
        self.images = self.images[shuffle]
        self.depths = self.depths[shuffle]

        # Reset iterator
        self.itr = 0

        # Batch size
        self.batch_size = batch_size

        # Number of possible iterations
        self.N_itr = int(self.N // self.batch_size)

        return self.N_itr

    def getBatch(self):

        # Extract batch
        imgs = self.images[(self.itr * self.batch_size) : ((self.itr + 1) * self.batch_size)]
        dpts = self.depths[(self.itr * self.batch_size) : ((self.itr + 1) * self.batch_size)]

        # Increment iterator
        self.itr += 1

        # Tensor conversion
        
        tensor_images = torch.from_numpy(imgs).float()
        tensor_depths = torch.from_numpy(dpts).float()
        '''

        print(imgs.shape)

        tensor_images = self.data_transforms(Image.fromarray(imgs))
        tensor_depths = self.data_transforms(dpts)
        '''
        return tensor_images, tensor_depths


    def getSample(self, num=None):

        if num == None:
            num = randint(0, self.N-1)

        # Get sample
        sample_img = torch.from_numpy(self.images[num]).float().view(1,self.C, self.H, self.W)
        sample_dpt = torch.from_numpy(self.depths[num]).float().view(1,self.H, self.W)

        return sample_img, sample_dpt


