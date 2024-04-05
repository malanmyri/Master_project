'''

Network Structure: 

Conv block 1: num filters = 96 with kernel size = 7
Max pool 1: kernel size = 7, stride 1


Conv block 2: num filters = 96 with kernel size = 7

Conv block 3: num filters = 128 with kernel size = 5
Max pool 3: kernel size = 5, stride 1

Conv block 4: num filters = 128 with kernel size = 5
Max pool 4: kernel size = 5, stride 1

Conv block 5: num filters = 128 with kernel size = 3

FC block 1: num neurons = 500

Linear output layer: num neurone = num sectors

Activation functions: Softmax


This is a classifier
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from helper import get_pad
import numpy as np


def act_fun(act_type):
 '''
 This function returns the activation function based on the input string
 '''

 if act_type == "relu":
    return nn.ReLU()

 if act_type == "tanh":
    return nn.Tanh()

 if act_type == "sigmoid":
    return nn.Sigmoid()

 if act_type == "leaky_relu":
    return nn.LeakyReLU(0.2)

 if act_type == "elu":
    return nn.ELU()

 if act_type == "softmax":
    return nn.LogSoftmax(dim=1)

 if act_type == "linear":
    return nn.LeakyReLU(1)  # initializzed like this, but not used in forward!


def flip(x, dim):

    '''
    This function is used to reverse the order of a tensor along a given dimension
    '''
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1,
                                                                 -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)




class Net(nn.Module):
    def __init__(self, 
                 num_stacked,
                 n_outputs,

                 ):
        super().__init__()

        self.num_stacked = num_stacked

        self.channels_1 = [3, 96]
        self.kernels_1 = [11]                                                                      # Defining the kernel sizes for the convolutional layers


        self.channels_2 = [96, 96, 128, 128, 128]                                                            
        self.kernels_2 = [7,5,5,3]                                                                      # Defining the kernel sizes for the convolutional layers            
        
        self.layer_1 = nn.Sequential(
            nn.Conv2d(self.channels_1[0], self.channels_1[1] , kernel_size= (self.num_stacked, self.kernels_1[0])),
            nn.MaxPool2d(kernel_size=(1, self.kernels_1[0]), stride=1),
            nn.BatchNorm2d(self.channels_1[1]),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        
        self.layer_2 = nn.ModuleList([nn.Sequential(
            nn.Conv1d(self.channels_2[i], self.channels_2[i+1] , kernel_size= self.kernels_2[i]),
            nn.MaxPool1d(kernel_size=self.kernels_2[i], stride=1),
            nn.BatchNorm1d(self.channels_2[i+1]),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)) for i in range(len(self.kernels_2))])

        self.fc_1 =  nn.Linear(121344, 500)  # This input layer has to be updated manually
        self.relu_1 = nn.LeakyReLU(0.2)
        self.batch_norm_1 = nn.BatchNorm1d(500)

        self.linear = nn.Linear(500, n_outputs)          
        
    def forward(self, x1, x2, x3):
        

        batch_size = x1.shape[0]
        num_stack = x1.shape[1]
        length = x1.shape[2]

        x1 = x1.view(batch_size, 1, num_stack, length)
        x2 = x2.view(batch_size, 1, num_stack, length)
        x3 = x3.view(batch_size, 1, num_stack, length)

        x = torch.cat((x1, x2, x3), dim=1)                                                            # Concatenating the three input tensors along the channel dimension

        x = self.layer_1(x)                                                                         # Applying the first convolutional layer to the input tensor

        x = x.view(x.shape[0], x.shape[1]*x.shape[2], x.shape[3])
        for layer in self.layer_2:
            x = layer(x)
        
        x = x.view(x.shape[0], x.shape[1]*x.shape[2])
        x = self.fc_1(x) 
        x = self.relu_1(x)
        x = self.batch_norm_1(x)
        x = self.linear(x)                                                                        # Applying the linear layer to the GCC-PHAT tensor
        x = F.softmax(x, dim=1)                                                                     # Applying the softmax activation function to the GCC-PHAT tensor
        return x
    
    def create_gcc(self, x1, x2, x3):
        
        length = x1.shape[2]
        
        n = x1.shape[-1] + x2.shape[-1]
        X1 = torch.fft.rfft(x1, n=n)
        X2 = torch.fft.rfft(x2, n=n)
        GX1X2 = X1 * torch.conj(X2)

        n = x1.shape[-1] + x3.shape[-1]
        X1 = torch.fft.rfft(x1, n=n)
        X3 = torch.fft.rfft(x3, n=n)
        GX1X3 = X1 * torch.conj(X3)

        n = x2.shape[-1] + x3.shape[-1]
        X2 = torch.fft.rfft(x2, n=n)
        X3 = torch.fft.rfft(x3, n=n)
        GX2X3 = X2 * torch.conj(X3)

        
        phi_X1X2 = 1 / (torch.abs(GX1X2) + self.epsilon)
        phi_X1X3 = 1 / (torch.abs(GX1X3) + self.epsilon)
        phi_X2X3 = 1 / (torch.abs(GX2X3) + self.epsilon)


        GX1X2 =  GX1X2 * torch.pow(phi_X1X2, 1)
        GX1X3 =  GX1X3 * torch.pow(phi_X1X3, 1)
        GX2X3 =  GX2X3 * torch.pow(phi_X2X3, 1)

        GX1X2 = GX1X2.imag
        GX1X3 = GX1X3.imag
        GX2X3 = GX2X3.imag


        for layer in self.layer_1:
            padding = get_pad(size=length, kernel_size=layer[0].kernel_size[0], stride=1, dilation=1)
            GX1X2 = F.pad(GX1X2, pad=padding, mode='constant')
            GX1X3 = F.pad(GX1X3, pad=padding, mode='constant')
            GX2X3 = F.pad(GX2X3, pad=padding, mode='constant')
            GX1X2 = layer(GX1X2)
            GX1X3 = layer(GX1X3)
            GX2X3 = layer(GX2X3)


        
        G12 = torch.fft.irfft(GX1X2, n)
        G13 = torch.fft.irfft( GX1X3, n)
        G23 = torch.fft.irfft( GX2X3, n)

        cc = torch.cat((G12, G13, G23), dim=1)
        cc = torch.cat((cc[:,:,-self.max_tau:], cc[:,:,  :self.max_tau+1]), dim=-1)

        return cc
            
