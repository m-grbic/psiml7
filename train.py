from copy import deepcopy
import matplotlib.pyplot as plt
from numpy import float32
from models.vgg16bn_disp import DepthNet
from time import time
import torch
from torch.optim import Adam
from loss import loss_functions
from preprocessing.data_transformations import get_split
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Available device is', device)

def train(batch_size, epochs):

    print('Loading data...')

    # Loading dataset
    train_set, val_set, test_set = get_split()

    print('Model setup...')

    # Creating a model
    model = DepthNet().to(device)

    # Optimizer setup
    optimizer = Adam(params = filter(lambda x: x.requires_grad, model.parameters()), lr=1e-4,
                     betas=(0.9, 0.999),
                     weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5, verbose = True)

    print('Training...')

    best_loss = None
    best_model = None

    training_loss = []
    validation_loss = []

    N_train = int(train_set.size() / batch_size)
    N_val = int(val_set.size() / batch_size)

    for epoch in range(epochs):
        torch.cuda.empty_cache()

        print('Epoch %d/%d' %(epoch+1,epochs))
        
        # Prepare model for training
        model.train()

        # Initialize running loss
        running_loss_photo = 0
        running_loss_smooth = 0
        running_loss = 0

        # Number of training iterations
        N_train = train_set.initBatch(batch_size=batch_size)

        for itr in range(N_train):
            print('Iteration %d/%d' %(itr+1,N_train))

            # Get images and depths
            # tgt_img, gt_depth = train_set.get_batch(batch_size=batch_size)
            tgt_img, gt_depth = train_set.getBatch()

            # Move tensors to device
            tgt_img = tgt_img.to(device).float()
            gt_depth = gt_depth.to(device).float()
            gt_depth = torch.squeeze(gt_depth[:,0,:,:], 1)

            # Clear gradients
            optimizer.zero_grad()

            # Prediction
            disparities = model(tgt_img)
            depth = 1 / disparities

            """
            plt.figure()
            plt.imshow(depth[0,0,:,:].detach().cpu())
            plt.show()
            """


            # Calculate loss
            loss_1 = loss_functions.l1_loss(gt_depth, depth)
            loss_3 = loss_functions.smooth_loss(depth)
            loss = 1*loss_1 + 1e-1*loss_3

            # Calculate gradients
            loss.backward()

            # Update weights
            optimizer.step()

            print('loss = {}'.format(loss.item()))

            # Update running loss
            running_loss_photo += loss_1.item() / N_train
            running_loss_smooth += loss_3.item() / N_train
            running_loss += loss.item() / N_train

            torch.cuda.empty_cache()

        # Print results on training dataset
        print('-----------------------------------------------------')
        print('################ Training results ###################')
        print('Photometric loss {}, Smooth loss {}, Overall loss {}'.format(running_loss_photo, running_loss_smooth, running_loss))
        print('-----------------------------------------------------')

        # Save training loss for current epoch
        training_loss.append(running_loss)

        # Initialize running loss
        running_loss_photo = 0
        running_loss_smooth = 0
        running_loss = 0

        # Prepare model for validation
        model.eval()

        # Number of validation iterations
        N_val = val_set.initBatch(batch_size=batch_size)

        for itr in range(N_val):
            print('Iteration %d/%d' %(itr+1, N_val))

            # Get images and depths
            # tgt_img, gt_depth = val_set.get_batch(batch_size=batch_size)
            tgt_img, gt_depth = val_set.getBatch()

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

                print('loss = {}'.format(loss.item()))
                
                # Update running loss
                running_loss_photo += loss_1.item() / N_val
                running_loss_smooth += loss_3.item() / N_val
                running_loss += loss.item() / N_val

                torch.cuda.empty_cache()

        # Print results on validation dataset
        print('-----------------------------------------------------')
        print('############### Validation results ##################')
        print('Photometric loss {}, Smooth loss {}, Overall loss {}'.format(running_loss_photo, running_loss_smooth, running_loss))
        print('-----------------------------------------------------')

        # Save validation loss for current epoch
        validation_loss.append(running_loss)

        # Saving the best model
        if (best_loss == None) or (loss.item() < best_loss):
            best_loss = loss.item()
            best_model = deepcopy(model)

        
        scheduler.step()

    return best_model, (training_loss, validation_loss)

if __name__ == '__main__':

    model, loss = train(batch_size=64, epochs=15)
    torch.save(model.state_dict(), 'models/pretrained_vgg16_BN_Bane_RELU')

    plt.figure(figsize=(12,12))
    plt.plot(loss[0])
    plt.plot(loss[1])
    plt.legend(['Training loss','Validation loss'])
    plt.title('Loss function')
    plt.grid(True)
    plt.show()
