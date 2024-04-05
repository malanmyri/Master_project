'''

Comments: 

The following code can be used to create unfiltered cross correlations.
cc_12_no_filter = self.gcc(x1.view(batch_size, 1, num_stacked, length), x2.view(batch_size, 1, num_stacked, length))
cc_13_no_filter = self.gcc(x1.view(batch_size, 1, num_stacked, length), x3.view(batch_size, 1, num_stacked, length))
cc_23_no_filter = self.gcc(x2.view(batch_size, 1, num_stacked, length), x3.view(batch_size, 1, num_stacked, length))
cc_no_filter = torch.cat((cc_12_no_filter, cc_13_no_filter, cc_23_no_filter), dim=2)


The following code describes the straight forward way of interleaving the arrays.
cc = torch.zeros((batch_size, self.num_filters*3,num_stacked, 2*self.max_tau+1))            
for i in range(self.num_filters):                                                           
    cc[:,i*3,:,:] = cc_12[:,i,:,:]
    cc[:,i*3+1,:,:] = cc_13[:,i,:,:]
    cc[:,i*3+2,:,:] = cc_23[:,i,:,:]

I am unsure wehter the following is the best way.
cc = cc.view(cc.shape[0], cc.shape[1]*cc.shape[2], cc.shape[3]) 

The following code stems from when I tried to use groups in the convolutional layers.
self.layer_1 = nn.Sequential(
            nn.Conv2d(num_filters*3, num_channels*num_filters, kernel_size= (5,11), groups=num_filters),
            nn.BatchNorm2d(num_channels*num_filters),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )

stride = 2
self.layer_2 = nn.Sequential(
            nn.Conv1d(num_channels*num_filters, num_filters, kernel_size=11, stride = stride, groups=num_filters),
            nn.BatchNorm1d(num_filters),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )



'''





import torch
import torch.nn as nn
import torch.nn.functional as F
from GCC import GCC
from torch_pad import get_pad
import math
from torch.autograd import Variable
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



class SincConv_fast(nn.Module):

    def __init__(self, 
                 num_filters, 
                 num_taps, 
                 sample_rate, 
                 max_hz, 
                 low_hz, 
                 min_band_hz, 
                 in_channels=1,
                 stride=1,   
                 padding=0, 
                 dilation=1, 
                 bias=False, 
                 groups=1):

        super(SincConv_fast, self).__init__()

        self.num_filters = num_filters
        self.num_taps = num_taps
        self.max_hz = max_hz

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if num_taps % 2 == 0:
            self.num_taps = self.num_taps+1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.sample_rate = sample_rate
        self.min_low_hz = low_hz
        self.min_band_hz = min_band_hz

        hz = np.linspace(self.min_low_hz/self.max_hz,1, self.num_filters + 1)          # Defining the normalized low cut-off frequencies, these will be in the range [0, 1]
        self.lower_cut_off_frequency = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1)) # Defining the lower cut-off frequencies as trainable parameters
        self.band_widths = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))         # Defining the band widths as trainable parameters

        n_lin = torch.linspace(0, (self.num_taps/2)-1,steps=int((self.num_taps/2)))    # Defining the time axis for the sinc filters
        self.window_left = 0.54-0.46*torch.cos(2*math.pi*n_lin/self.num_taps)          # Defining the left part of the sinc filters                               
        self.n_ = 2*math.pi*torch.arange(-(self.num_taps - 1) / 2.0  , 0).view(1, -1) / self.sample_rate  # Defining the left part of the time axis for the sinc filters


    def forward(self, waveforms):

        self.n_ = self.n_.to(waveforms.device)                                                                     # Moving the tensors to the same device as the input
        self.window_left = self.window_left.to(waveforms.device)                                                   # Moving the tensors to the same device as the input

        fc_low = torch.clamp(self.lower_cut_off_frequency, self.min_low_hz/self.max_hz, 1- self.min_band_hz/self.max_hz) # Defining the low cut-off frequencies
        band_widths = torch.clamp(self.band_widths, self.min_band_hz/self.max_hz, 1 - self.min_low_hz/self.max_hz)
        fc_high = fc_low + band_widths                                                                                   # Defining the high cut-off frequencies
        band = (fc_high-fc_low)[:, 0]  * self.max_hz                                                                     # Defining the band width of the filters

        f_times_t_low = torch.matmul(fc_low, self.n_)* self.max_hz                                                       # Calculating the low frequency values inside the sinc functions 
        f_times_t_high = torch.matmul(fc_high, self.n_)* self.max_hz                                               # Calculating the high frequency values inside the sinc functions
        
        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low))/(self.n_/2))*self.window_left     # Calculating the left part of the sinc filters
        band_pass_center = 2*band.view(-1, 1)                                                                      # Calculating the center part of the sinc filters
        band_pass_right = torch.flip(band_pass_left, dims=[1])                                                     # Calculating the right part of the sinc filters
        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)                          # Concatenating the left, center and right parts of the sinc filters
        
        
        band_pass = band_pass / (2*band[:, None])                                                                  # Normalizing the sinc filters
        self.filters = (band_pass).view(self.num_filters, 1, self.num_taps)                                        # Reshaping the sinc filters to the correct shape
        waveforms = waveforms.view(waveforms.shape[0], 1, waveforms.shape[1])
        return F.conv1d(waveforms, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                        bias=None, groups=1)

class SincNetGCC(nn.Module):
    def __init__(self, 
                 max_tau, 
                 num_filters, 
                 num_channels,
                 activation,
                 max_fc, 
                 min_fc,
                 min_band,
                 num_taps, 
                 fs,
                 num_stacked,
                 n_outputs,
                 ):
        super().__init__()


        self.max_tau = max_tau
        self.num_stacked = num_stacked
        self.num_filters = num_filters
        self.channels = [(num_filters)*num_stacked*3, num_channels*num_filters]                      # Defining the number of channels for the convolutional layers
        self.num_taps = num_taps                                                                     
        self.mlp_kernels = [11]                                                                      # Defining the kernel sizes for the convolutional layers
        self.final_kernel = 5                                                                        # Defining the kernel size for the final convolutional layer
        self.gcc = GCC(max_tau = max_tau)                                                            # Initializing the GCC layer
        self.activation = act_fun(activation)                                                        # Initializing the activation function
        self.backbone = SincConv_fast(num_filters, num_taps, fs, max_fc, min_fc, min_band)           # Initializing the SincNet backbone
        
        self.layer_1 = nn.Sequential(
            nn.Conv2d(num_filters*3, num_channels, kernel_size= (5,11)),
            nn.BatchNorm2d(num_channels),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )

        self.layer_2 = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, kernel_size=self.final_kernel),
            nn.BatchNorm1d(num_channels),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )

        
        self.batch_norm = nn.BatchNorm1d((2 * self.max_tau + 1)*num_channels)                                        # Initializing the batch normalization layer
        self.leaky_relu = nn.LeakyReLU(0.2)                                                          # Initializing the leaky relu activation function 
        self.linear = nn.Linear((2 * self.max_tau + 1)* num_channels, n_outputs)          # Initializing the linear layer
        self.linear = nn.Linear((2 * self.max_tau + 1)* num_channels, n_outputs)          # Initializing the linear layer
        
    def forward(self, x1, x2, x3):

        batch_size = x1.shape[0]
        num_stacked = x1.shape[1]
        length = x1.shape[2]

        padding = get_pad(size=length, kernel_size=self.num_taps, stride=1, dilation=1)              # Calculating the padding for the input signals
        x1 = F.pad(x1, pad=padding, mode='constant')                                                 # Applying the padding to the input signals
        x2 = F.pad(x2, pad=padding, mode='constant')                                                 # Applying the padding to the input signals
        x3 = F.pad(x3, pad=padding, mode='constant')                                                 # Applying the padding to the input signals



        y1 = torch.zeros((batch_size, self.num_filters,num_stacked, length))
        y2 = torch.zeros((batch_size, self.num_filters,num_stacked, length))
        y3 = torch.zeros((batch_size, self.num_filters,num_stacked, length))

        for i in range(num_stacked):
            y1[:, :, i, :] = self.backbone(x1[:, i, :])                                              # Applying the SincNet backbone to the input signals
            y2[:, :, i, :] = self.backbone(x2[:, i, :])                                              # Applying the SincNet backbone to the input signals
            y3[:, :, i, :] = self.backbone(x3[:, i, :])                                              # Applying the SincNet backbone to the input signals
 
        cc_12 = self.gcc(y1, y2)                                                                     # Calculating the GCC-PHAT between the filtered signals
        cc_13 = self.gcc(y1, y3)                                                                     # Calculating the GCC-PHAT between the filtered signals
        cc_23 = self.gcc(y2, y3)                                                                     # Calculating the GCC-PHAT between the filtered signals



        batch_size, num_filters, num_stacked, max_tau_dim = cc_12.shape                              # Getting the shape of the GCC-PHAT tensor
        num_cross_correlations = 3  
        indices = torch.arange(num_filters) * num_cross_correlations                                 # Defining the indices for interleaving the GCC-PHAT tensors
        cc = torch.zeros((batch_size, num_filters*num_cross_correlations, num_stacked, max_tau_dim)) # Initializing the tensor for the interleaved GCC-PHAT tensors
        cc[:, indices, :, :] = cc_12                                                                 # Interleaving the GCC-PHAT tensors
        cc[:, indices+1, :, :] = cc_13                                                               # Interleaving the GCC-PHAT tensors
        cc[:, indices+2, :, :] = cc_23                                                               # Interleaving the GCC-PHAT tensors
        
        cc = cc.to(x1.device)

        s = cc.shape[2]
        padding = get_pad(size=s, kernel_size=11, stride=1, dilation=1)                               # Calculating the padding for the convolutional layers
        cc = F.pad(cc, pad=padding, mode='constant')                                                  # Applying the padding to the GCC-PHAT tensor
        cc = self.layer_1(cc)                                                                         # Applying the convolutional layer to the GCC-PHAT tensor

        cc = cc.view(cc.shape[0], cc.shape[1], cc.shape[2]*cc.shape[3])                               # Reshaping the GCC-PHAT tensor so that it can be used in the convolutional layers
        
        s = cc.shape[2]
        padding = get_pad(size=s, kernel_size=self.final_kernel, stride=1, dilation=1)                               # Calculating the padding for the convolutional layers
        cc = F.pad(cc, pad=padding, mode='constant')                                                  # Applying the padding to the GCC-PHAT tensor
        cc = self.layer_2(cc)                                                                         # Applying the convolutional layer to the GCC-PHAT tensor
        
        
        cc = cc.view(cc.shape[0], cc.shape[1]*cc.shape[2])                                            # Reshaping the GCC-PHAT tensor so that it can be used in the convolutional layers
        cc = self.batch_norm(cc)                                                                      # Applying the batch normalization layer to the GCC-PHAT tensor
        cc = self.leaky_relu(cc)                                                                      # Applying the leaky relu activation function to the GCC-PHAT tensor
        cc = self.linear(cc)                                                                          # Applying the linear layer to the GCC-PHAT tensor
        cc = F.softmax(cc, dim=1)                                                                     # Applying the softmax activation function to the GCC-PHAT tensor
        return cc
    
    def create_gcc(self, x1, x2, x3):
        batch_size = x1.shape[0]
        num_stacked = x1.shape[1]
        length = x1.shape[2]
        padding = get_pad(size=length, kernel_size=self.num_taps, stride=1, dilation=1)
        cc_12_no_filter = self.gcc(x1.view(batch_size, 1, num_stacked, length), x2.view(batch_size, 1, num_stacked, length))
        cc_13_no_filter = self.gcc(x1.view(batch_size, 1, num_stacked, length), x3.view(batch_size, 1, num_stacked, length))
        cc_23_no_filter = self.gcc(x2.view(batch_size, 1, num_stacked, length), x3.view(batch_size, 1, num_stacked, length))
        x1 = F.pad(x1, pad=padding, mode='constant')
        x2 = F.pad(x2, pad=padding, mode='constant')
        x3 = F.pad(x3, pad=padding, mode='constant')

        y1 = torch.zeros((batch_size, self.num_filters,num_stacked, length))
        y2 = torch.zeros((batch_size, self.num_filters,num_stacked, length))
        y3 = torch.zeros((batch_size, self.num_filters,num_stacked, length))

        for i in range(num_stacked):
            y1[:, :, i, :] = self.backbone(x1[:, i, :])
            y2[:, :, i, :] = self.backbone(x2[:, i, :])
            y3[:, :, i, :] = self.backbone(x3[:, i, :])


        cc_12 = self.gcc(y1, y2)
        cc_13 = self.gcc(y1, y3)
        cc_23 = self.gcc(y2, y3)

        cc = torch.cat((cc_12, cc_13, cc_23), dim=2)
        cc_no_filter = torch.cat((cc_12_no_filter, cc_13_no_filter, cc_23_no_filter), dim=2)
        cc = torch.cat((cc, cc_no_filter), dim=1)

        return cc
            
