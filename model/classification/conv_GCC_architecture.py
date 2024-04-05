'''

Network Structure: 

Fourier Transform 
Cross power spectrum
Conv 1D layer 
Inverse Fourier Transform
Cross Correlation
Conv 1D layer
linear layer 
softmax

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
        num_filters = 16
        self.channels_1 = [self.num_stacked,num_filters,num_filters]                                                            
        self.kernels_1 = [11,7]                                                                      # Defining the kernel sizes for the convolutional layers

        self.channels_2 = [3*self.channels_1[-1],num_filters,num_filters]
        self.kernels_2 = [11,7]

        self.epsilon = 1e-6
        self.beta = np.array([1])                                                                
        self.activation = act_fun("leaky_relu") 
        self.max_tau = 20   
                                              
        
        
        self.layer_1 = nn.ModuleList([nn.Sequential(
            nn.Conv1d(self.channels_1[i], self.channels_1[i+1] , kernel_size= self.kernels_1[i]),
            nn.BatchNorm1d(self.channels_2[i+1]),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)) for i in range(len(self.kernels_1))])
        
        self.layer_2 = nn.ModuleList([nn.Sequential(
            nn.Conv1d(self.channels_2[i], self.channels_2[i+1], kernel_size= self.kernels_2[i]),
            nn.BatchNorm1d(self.channels_2[i+1]),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)) for i in range(len(self.kernels_2))])
        
        self.batch_norm = nn.BatchNorm1d((2 * self.max_tau + 1)*self.channels_2[-1])   
        self.leaky_relu = nn.LeakyReLU(0.2)                                                          # Initializing the leaky relu activation function 
        self.linear = nn.Linear((2 * self.max_tau + 1)*self.channels_2[-1], n_outputs)          # Initializing the linear layer
        
    def forward(self, x1, x2, x3):

        batch_size = x1.shape[0]
        num_stacked = x1.shape[1]
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

        for layer in self.layer_2:
           padding = get_pad(size=cc.shape[2], kernel_size=layer[0].kernel_size[0], stride=1, dilation=1)
           cc = F.pad(cc, pad=padding, mode='constant')
           cc = layer(cc)


        cc = cc.view(cc.shape[0], cc.shape[1]*cc.shape[2])                                            # Reshaping the GCC-PHAT tensor so that it can be used in the convolutional layers
        cc = self.batch_norm(cc)                                                                      # Applying the batch normalization layer to the GCC-PHAT tensor
        cc = self.leaky_relu(cc)                                                                      # Applying the leaky relu activation function to the GCC-PHAT tensor
        cc = self.linear(cc)                                                                          # Applying the linear layer to the GCC-PHAT tensor
        cc = F.softmax(cc, dim=1)                                                                     # Applying the softmax activation function to the GCC-PHAT tensor
        return cc
    
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
            
