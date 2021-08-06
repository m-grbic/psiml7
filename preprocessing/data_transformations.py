import numpy as np
from numpy.core.fromnumeric import reshape
import torch
from torch.types import Device
from torchvision import transforms
from sklearn.model_selection import train_test_split
from random import sample, randint
from PIL import Image
from random import random
from skimage.transform import resize


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
        self.images = images/255 # 0-255 => 0-1
        self.depths = depths
        self.N, self.C, self.H, self.W = images.shape
        self.itr = 0
        self.batch_size = 1
        self.N_itr = 1

        # Rescale images
        self.rescale()

        # Normalization
        for i in range(3):
            self.images[:,i,:,:] = (self.images[:,i,:,:] - NYUD_MEAN[i]) / NYUD_STD[i]


    def size(self):
        return self.N


    def rescale(self, output_shape=(96,128)):

        # Rescaled images
        images_rescaled = np.zeros((self.N, 3, output_shape[0], output_shape[1]))
        depths_rescaled = np.zeros((self.N, 1, output_shape[0], output_shape[1]))

        # Iterate through dataset
        for i in range(self.N):
            # Rescale images
            images_rescaled[i,:,:,:] = np.swapaxes(np.swapaxes(resize(
                np.swapaxes(np.swapaxes(self.images[i,:,:,:],0,1),1,2),
                output_shape = output_shape + (3,),
                clip=True, 
                anti_aliasing=True,
                #multichannel=True,
                preserve_range=True
                ),2,1),0,1)

            # Rescale depth
            depths_rescaled[i,0,:,:] = resize(
                self.depths[i,0,:,:],
                output_shape=output_shape,
                clip=True, 
                anti_aliasing=True,
                #multichannel=False,
                preserve_range=True
                )

        # Save images
        self.images = images_rescaled
        self.depths = depths_rescaled 

        self.H = output_shape[0]
        self.W = output_shape[1]


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

        # Random flip
        imgs, dpts = RandomHorizontalFlip(imgs,dpts)
        imgs, dpts = RandomVerticalFlip(imgs,dpts)

        # Tensor conversion
        tensor_images = torch.from_numpy(imgs).float()
        tensor_depths = torch.from_numpy(dpts).float()

        return tensor_images, tensor_depths


    def getSample(self, num=None):

        if num == None:
            num = randint(0, self.N-1)

        image, depth = self.images[num], self.depths[num]

        # Get sample
        sample_img = torch.from_numpy(image).float().view(1,self.C, self.H, self.W)
        sample_dpt = torch.from_numpy(depth).float().view(1,self.H, self.W)

        return sample_img, sample_dpt



def RandomHorizontalFlip(images, depth, p=0.5):

    # Iterate through batch
    for i in range(images.shape[0]):
        if random() < p:
            images[i] = np.flip(images[i], 1)
            depth[i] = np.flip(depth[i], 1)

    return images, depth


def RandomVerticalFlip(images, depth, p=0.5):

    # Iterate through batch
    for i in range(images.shape[0]):
        if random() < p:
            images[i] = np.flip(images[i], 2)
            depth[i] = np.flip(depth[i], 2)

    return images, depth


def denormalize(image):

    # Denormalization
    for i in range(3):
        image[:,i,:,:] = (image[:,i,:,:]*NYUD_STD[i] + NYUD_MEAN[i])*255

    # Hard limit
    image[image<0] = 0
    image[image>255] = 255

    return image

