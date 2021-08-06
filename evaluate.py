from models.vgg16bn_disp import DepthNet
from copy import deepcopy
import matplotlib.pyplot as plt
from numpy import float32
from models.vgg16bn_disp import DepthNet
from time import time
import torch
from torch.optim import Adam
from loss import loss_functions
from preprocessing.data_transformations import get_split

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Available device is', device)

def visualize_sample(model, dataset):

    img, gt_depth = dataset.getSample()

    # To Cuda
    img = img.to(device).float()
    gt_depth = gt_depth.to(device).float()

    with torch.no_grad():
        # Prediction
        pred = model(img)
        depth = 1 / pred

        '''
        plt.figure(figsize=(12,12))
        plt.hist(depth.cpu().numpy().flatten())
        plt.title('Depth')
        plt.show()

        plt.figure(figsize=(12,12))
        plt.hist(pred.cpu().numpy().flatten())
        plt.title('Disparity')
        plt.show()
        '''


        #print("Najmanji - najveci")
        #print(torch.min(depth), torch.max(depth))

        depth = torch.clamp(depth, min=1e-3, max=10)

        #print("Depth")
        #print(depth)
        #print("Gt depth")
        #print(gt_depth)
        
        # Calculate loss
        loss_1 = loss_functions.l1_loss(gt_depth, depth)
        loss_3 = loss_functions.smooth_loss(depth)
        loss = 1*loss_1 + 1e-1*loss_3

        print('Photometric loss {}, Smooth loss {}, Overall loss {}'.format(loss_1.item(), loss_3.item(), loss.item()))

    fig, axes = plt.subplots(ncols=3)
    ax = axes.ravel()
    ax[0].imshow(depth[0,0,:,:].cpu().numpy())
    ax[0].set_axis_off()
    ax[0].set_title('Prediction depth')
    ax[1].imshow(gt_depth[0,:,:].cpu().numpy())
    ax[1].set_axis_off()
    ax[1].set_title('Ground truth depth')
    ax[2].imshow(img[0,:,:,:].swapaxes(0,1).swapaxes(1,2).to(torch.uint8).cpu().numpy())
    ax[2].set_axis_off()
    ax[2].set_title('Original image')
    plt.tight_layout()
    plt.show()

    
if __name__ == '__main__':

    # Load pretrained network
    model = DepthNet()
    model.load_state_dict(torch.load('models/pretrained_vgg16_BN'))
    model.to(device).eval()

    # Get dataset
    train_set, val_set, test_set = get_split()
    
    # Initialize running loss
    running_loss_photo = 0
    running_loss_smooth = 0
    running_loss = 0

    # Evaluation on test dataset
    N_test = test_set.initBatch(batch_size=1)

    """
    for itr in range(N_test):
        print('Iteration %d/%d' %(itr+1, N_test))

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
    """
    
    # Visualization of results
    visualize_sample(model, train_set)
    visualize_sample(model, val_set)
    visualize_sample(model, test_set)

