from random import Random
from models.vgg16bn_disp import DepthNet
from copy import deepcopy
import matplotlib.pyplot as plt
from numpy import float32
from models.vgg16bn_disp import DepthNet
from time import time
import torch
from torch.optim import Adam
from loss import loss_functions
from preprocessing.data_transformations import denormalize, get_split, RandomHorizontalFlip, RandomVerticalFlip

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Available device is', device)


def visualize_sample(model, dataset):

    img, gt_depth = dataset.getSample()

    # To Cuda
    img = img.to(device).float()
    gt_depth = gt_depth.to(device).float()

    with torch.no_grad():
        # Prediction
        disp = model(img)
        depth = 1 / disp

        # Limit values
        depth = torch.clamp(depth, min=1e-3, max=10)
        
        # Calculate loss
        loss_1 = loss_functions.l1_loss(gt_depth, depth)
        loss_3 = loss_functions.smooth_loss(depth)
        loss = 1*loss_1 + 1e-1*loss_3

    fig, axes = plt.subplots(ncols=3)
    ax = axes.ravel()
    ax[0].imshow(depth[0,0,:,:].cpu().numpy())
    ax[0].set_axis_off()
    ax[0].set_title('Prediction depth')
    ax[1].imshow(gt_depth[0,:,:].cpu().numpy())
    ax[1].set_axis_off()
    ax[1].set_title('Ground truth depth')
    ax[2].imshow(denormalize(img)[0,:,:,:].swapaxes(0,1).swapaxes(1,2).to(torch.uint8).cpu().numpy())
    ax[2].set_axis_off()
    ax[2].set_title('Original image')
    plt.tight_layout()
    plt.show()


def test(model, test_set):
    # Initialize running loss
    running_loss_photo = 0
    running_loss_smooth = 0
    running_loss = 0

    # Evaluation on test dataset
    N_test = test_set.initBatch(batch_size=1)

    # Iterate through test dataset
    for itr in range(N_test):
        # Verbose
        print('Iteration %d/%d' %(itr+1, N_test), end='\n')

        # Get images and depths
        # tgt_img, gt_depth = val_set.get_batch(batch_size=batch_size)
        tgt_img, gt_depth = test_set.getBatch()

        # Move tensors to device
        tgt_img = tgt_img.to(device).float()
        gt_depth = gt_depth.to(device).float()
        gt_depth = torch.squeeze(gt_depth[:, 0, :, :])

        with torch.no_grad():
            # Prediction
            disparities = model(tgt_img)
            depth = 1 / disparities
            
            # Calculate loss
            loss_1 = loss_functions.l1_loss(gt_depth, depth)
            loss_3 = loss_functions.smooth_loss(depth)
            loss = 1*loss_1 + 1e-1*loss_3
            
            # Update running loss
            running_loss_photo += loss_1.item() / N_test
            running_loss_smooth += loss_3.item() / N_test
            running_loss += loss.item() / N_test

            torch.cuda.empty_cache()

    # Print results on training dataset
    print('-----------------------------------------------------')
    print('################## Test results #####################')
    print('Photometric loss {}, Smooth loss {}, Overall loss {}'.format(running_loss_photo, running_loss_smooth, running_loss))
    print('-----------------------------------------------------')

    
if __name__ == '__main__':

    # Load pretrained network
    print('Loading model...')
    model = DepthNet()
    model.load_state_dict(torch.load('models/pretrained_vgg16_BN_Bane_RELU'))
    model.to(device).eval()
    print('Model loaded!')

    # Load dataset
    print('Loading data...')
    train_set, val_set, test_set = get_split()
    print('Data loaded!')
    
    """
    # Test model
    print('Testing a model...')
    test(model=model, test_set=test_set)
    print('Testing finished!')
    """
    
    # Visualization of results
    visualize_sample(model, train_set)
    visualize_sample(model, val_set)
    visualize_sample(model, test_set)