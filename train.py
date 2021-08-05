
from copy import deepcopy
import matplotlib.pyplot as plt
from numpy import float32
from models.vgg16bn_disp import DepthNet
from time import time
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from loss import loss_functions
from preprocessing.dataloader import get_DataSets

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Available device is', device)
device = "cpu"

def train(batch_size, epochs):

    print('Loading data...')

    # Loading dataset
    train_set, val_set = get_DataSets()

    # Train & Val Loader
    train_loader = DataLoader(dataset=train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=True)

    val_loader = DataLoader(dataset=val_set,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=0,
                              pin_memory=True)

    print('Model setup...')
    
    # Creating a model
    model = DepthNet().to(device)

    # Optimizer setup
    optimizer = Adam(params = filter(lambda x: x.requires_grad, model.parameters()), lr=1e-4,
                     betas=(0.9, 0.999),
                     weight_decay=0)


    print('Training...')

    best_loss = None
    best_model = None

    training_loss = []
    validation_loss = []

    N_train = len(train_loader)
    N_val = len(val_loader)

    for epoch in range(epochs):
        print('Epoch %d/%d' %(epoch+1,epochs))
        
        # Prepare model for training
        model.train()

        # Initialize running loss
        running_loss_photo = 0
        running_loss_smooth = 0
        running_loss = 0

        for tgt_img, gt_depth in train_loader:

            # Move tensors to device
            tgt_img = tgt_img.to(device).float()
            gt_depth = gt_depth.to(device).float()
            gt_depth = torch.squeeze(gt_depth[:,0,:,:])

            # Clear gradients
            optimizer.zero_grad()

            # Prediction
            disparities = model(tgt_img)
            depth = 1 / disparities
  
            # Calculate loss
            loss_1 = loss_functions.l1_loss(gt_depth, depth)
            loss_3 = loss_functions.smooth_loss(depth)
            loss = 1*loss_1 + 1*loss_3
            
            # Calculate gradients
            loss.backward()

            # Update weights
            optimizer.step()

            print('Photometric loss {}, Smooth loss {}, Overall loss {}'.format(loss_1.item(), loss_3.item(), loss.item()))
            
            # Update running loss
            running_loss_photo += loss_1.item()/N_train
            running_loss_smooth += loss_3.item()/N_train
            running_loss += loss.item()/N_train
        
        # Print results on training dataset
        print('Training results...')
        print('Photometric loss {}, Smooth loss {}, Overall loss {}'.format(running_loss_photo, running_loss_smooth, running_loss))
        
        # Save training loss for current epoch
        training_loss.append(running_loss)

        # Prepare model for validation
        model.eval()

        for tgt_img, gt_depth in val_loader:
            
            # Move tensors to device
            tgt_img = tgt_img.to(device).float()
            gt_depth = gt_depth.to(device).float()
            gt_depth = torch.squeeze(gt_depth[:,0,:,:])


            # Prediction
            disparities = model(tgt_img)
            depth = 1 / disparities
            
            # Calculate loss
            loss_1 = loss_functions.l1_loss(gt_depth, depth)
            loss_3 = loss_functions.smooth_loss(depth)
            loss = 1*loss_1 + 1*loss_3

            print('Photometric loss {}, Smooth loss {}, Overall loss {}'.format(loss_1.item(), loss_3.item(), loss.item()))

            
            # Update running loss
            running_loss_photo += loss_1.item()/N_val
            running_loss_smooth += loss_3.item()/N_val
            running_loss += loss.item()/N_val

        # Print results on validation dataset
        print('Validation results...')
        print('Photometric loss {}, Smooth loss {}, Overall loss {}'.format(running_loss_photo, running_loss_smooth, running_loss))
        
        # Save validation loss for current epoch
        validation_loss.append(running_loss)

        # Svaing the best model
        if (best_loss == None) or (loss.item() < best_loss):
            best_loss = loss.item()
            best_model = deepcopy(model)

    return best_model, (training_loss, validation_loss)
if __name__ == '__main__':

    model, loss = train(batch_size=32, epochs=20)
    torch.save(model.state_dict(), 'models/pretrained_vgg16_BN')

    plt.figure(figisze=(12,12))
    plt.plot(loss[0])
    plt.plot(loss[1])
    plt.legend(['Training loss','Validation loss'])
    plt.title('Loss function')
    plt.show()