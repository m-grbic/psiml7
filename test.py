#from Dataloader import NYU_Depth_V2
from matplotlib import pyplot as plt
import numpy as np
import torch
from time import time
from loss.loss_functions import l1_loss, smooth_loss
from preprocessing.data_transformations import get_split

'''
gas = NYU_Depth_V2()

print(gas.images.shape)
print(gas.images[0].shape)
ide = np.swapaxes(gas.images[0], 0, 2).astype('uint8')
ide2 = np.swapaxes(ide, 0, 1).astype('uint8')
plt.imshow(ide2,cmap="gray")
plt.show()
'''

train_data, val_data, test_data = get_split()

imgs, dpts = train_data.getBatch()

print(imgs.shape, dpts.shape)