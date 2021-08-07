import torch.cuda
import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torchvision.models as models
from torchsummary import summary

#from .model_utils import * #use . represent relative address
#from utils.util_functions import unsqueeze_dim0_tensor

# Upsampling image
def upsample(x):
    return F.interpolate(x, scale_factor=2, mode='nearest')


def Conv2dBlock(c_in, c_out, k_size, stride, padding, activation='lrelu'):

    if activation=='lrelu':
    	return nn.Sequential(
            nn.Conv2d(c_in, c_out, k_size, stride, padding),
            nn.LeakyReLU(0.1)
        )
    elif activation=='relu':
    	return nn.Sequential(
            nn.Conv2d(c_in, c_out, k_size, stride, padding),
            nn.ReLU(inplace=True)
        )
    elif activation=='elu':
    	return nn.Sequential(
            nn.Conv2d(c_in, c_out, k_size, stride, padding),
            nn.ELU(inplace=True)
        )
    else:
        raise ValueError('Activation function {} is not supported'.format(activation))


def ConvTranspose2dBlock(c_in, c_out, k_size, stride, padding, output_padding, activation='lrelu'):

    if activation=='lrelu':
    	return nn.Sequential(
            nn.ConvTranspose2d(c_in, c_out, k_size, stride, padding, output_padding),
            nn.LeakyReLU(0.1)
        )
    elif activation=='relu':
    	return nn.Sequential(
            nn.ConvTranspose2d(c_in, c_out, k_size, stride, padding, output_padding),
            nn.ReLU(inplace=True)
        )
    elif activation=='elu':
    	return nn.Sequential(
            nn.ConvTranspose2d(c_in, c_out, k_size, stride, padding, output_padding),
            nn.ELU(inplace=True)
        )
    else:
        raise ValueError('Activation function {} is not supported'.format(activation))


def predict_disp(in_planes):
    '''
    Output layer
    '''
    return nn.Sequential(
        nn.Conv2d(in_planes, 1, kernel_size=3, padding=1),
        nn.Sigmoid()
    )


class DepthNet(nn.Module):
    
    def __init__(self, datasets ='nyu'):

        super(DepthNet, self).__init__()

        self.only_train_decoder = False

        if datasets == 'nyu':
            self.alpha = 10
            self.beta = 0.1
        else:
            raise ValueError('Dataset name {} is unknown.'.format(datasets))

        # Load encoder
        self.features = models.vgg16_bn(pretrained=True)
        # Freeze parameters
        for param in self.features.parameters():
            param.requires_grad = True

        # Create decoder
        self.upconv4 = ConvTranspose2dBlock(512, 256, 4, 2, 1, 0)
        self.iconv4 = Conv2dBlock(256 + 512, 256, 3, 1, 1)

        self.upconv3 = ConvTranspose2dBlock(256, 128, 4, 2, 1, 0)
        self.iconv3 = Conv2dBlock(128 + 256, 128, 3, 1, 1)

        self.upconv2 = ConvTranspose2dBlock(128, 64, 4, 2, 1, 0)
        self.iconv2 = Conv2dBlock(64 + 128 + 1, 64, 3, 1, 1)

        self.upconv1 = ConvTranspose2dBlock(64, 32, 4, 2, 1, 0)
        self.iconv1 = Conv2dBlock(32 + 64 + 1, 32, 3, 1, 1)

        self.upconv0 = ConvTranspose2dBlock(32, 16, 4, 2, 1, 0)
        self.iconv0 = Conv2dBlock(16 + 1, 16, 3, 1, 1)

        self.disp3 = predict_disp(128)
        self.disp2 = predict_disp(64)
        self.disp1 = predict_disp(32)
        self.disp0 = predict_disp(16)


    def forward(self, x):

        conv1 = self.features._modules['features'][0:7](x)
        conv2 = self.features._modules['features'][7:14](conv1)
        conv3 = self.features._modules['features'][14:24](conv2)
        conv4 = self.features._modules['features'][24:34](conv3)
        conv5 = self.features._modules['features'][34:44](conv4)

        if self.only_train_decoder:
            conv1 = conv1.detach()
            conv2 = conv2.detach()
            conv3 = conv3.detach()
            conv4 = conv4.detach()
            conv5 = conv5.detach()

        skip1 = conv1
        skip2 = conv2
        skip3 = conv3
        skip4 = conv4

        upconv4 = self.upconv4(conv5)
        concat4 = torch.cat((upconv4, skip4), 1)
        iconv4  = self.iconv4(concat4)

        upconv3 = self.upconv3(iconv4)
        concat3 = torch.cat((upconv3, skip3), 1)
        iconv3  = self.iconv3(concat3)
        disp3   = self.alpha * self.disp3(iconv3) + self.beta
        disp3up = upsample(disp3)

        upconv2 = self.upconv2(iconv3)
        concat2 = torch.cat((upconv2, skip2, disp3up), 1)
        iconv2  = self.iconv2(concat2)
        disp2   = self.alpha * self.disp2(iconv2) + self.beta
        disp2up = upsample(disp2)

        upconv1 = self.upconv1(iconv2)
        concat1 = torch.cat((upconv1, skip1, disp2up), 1)
        iconv1  = self.iconv1(concat1)
        disp1   = self.alpha * self.disp1(iconv1) + self.beta
        disp1up = upsample(disp1)

        upconv0 = self.upconv0(iconv1)
        concat0 = torch.cat((upconv0, disp1up), 1)
        iconv0  = self.iconv0(concat0)
        disp0   = self.alpha * self.disp0(iconv0) + self.beta

        return disp0


if __name__ == '__main__':
    model = DepthNet().eval()
    summary(model, (3, 284, 392))
    #print(model(torch.rand(1,3,284,392)))