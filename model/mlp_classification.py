#from dnn_models import SincNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from GCC import GCC
from torch_pad import get_pad
import math
from torch.autograd import Variable
import numpy as np


def act_fun(act_type):

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
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1,
                                                                 -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)



class SincConv_fast(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    def __init__(self, out_channels, kernel_size, sample_rate, max_hz, low_hz, min_band_hz, in_channels=1,
                 stride=1, padding=0, dilation=1, bias=False, groups=1):

        super(SincConv_fast, self).__init__()

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.max_hz = max_hz

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size+1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.sample_rate = sample_rate
        self.min_low_hz = low_hz
        self.min_band_hz = min_band_hz

        hz = np.linspace(self.min_low_hz, self.max_hz, self.out_channels + 1)
        self.lower_cut_off_frequency = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
        self.band_widths = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        n_lin = torch.linspace(0, (self.kernel_size/2)-1,steps=int((self.kernel_size/2)))
        self.window_left = 0.54-0.46*torch.cos(2*math.pi*n_lin/self.kernel_size) 
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2*math.pi*torch.arange(-n, 0).view(1, -1) / self.sample_rate


    def forward(self, waveforms):

        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """
        # Sending the tensors to the same device
        self.n_ = self.n_.to(waveforms.device)
        self.window_left = self.window_left.to(waveforms.device)
        fc_low = self.min_low_hz + torch.abs(self.lower_cut_off_frequency)
        fc_high = torch.clamp(fc_low + self.min_band_hz + torch.abs(self.band_widths), self.min_low_hz, self.max_hz)
        band = (fc_high-fc_low)[:, 0]
        f_times_t_low = torch.matmul(fc_low, self.n_)
        f_times_t_high = torch.matmul(fc_high, self.n_)
        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low))/(self.n_/2))*self.window_left
        band_pass_center = 2*band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])
        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        band_pass = band_pass / (2*band[:, None])
        self.filters = (band_pass).view(self.out_channels, 1, self.kernel_size)
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
        self.channels = [num_filters*num_stacked*3, num_channels, num_channels]
        self.num_taps = num_taps
        self.mlp_kernels = [11,7]
        self.final_kernel = 5
        self.gcc = GCC(max_tau = max_tau)
        self.activation = act_fun(activation)
        #self.max_pool = F.max_pool1d(num_taps, stride=1)
        self.backbone = SincConv_fast(num_filters, num_taps, fs, max_fc, min_fc, min_band)
        self.mlp = nn.ModuleList([nn.Sequential(
                nn.Conv1d(self.channels[i], self.channels[i+1], kernel_size=k),
                nn.BatchNorm1d(self.channels[i+1]),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.5)) for i, k in enumerate(self.mlp_kernels)])
        
        self.final_conv = nn.Conv1d(num_channels, 1, kernel_size=self.final_kernel)
        self.batch_norm = nn.BatchNorm1d((2 * self.max_tau + 1))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.linear = nn.Linear((2 * self.max_tau + 1), n_outputs) # multiplying with 3 since we have three sensors
        
    def forward(self, x1, x2, x3):

        batch_size = x1.shape[0]
        num_stacked = x1.shape[1]
        length = x1.shape[2]

        # Padding the input
        padding = get_pad(size=length, kernel_size=self.num_taps, stride=1, dilation=1)
        x1 = F.pad(x1, pad=padding, mode='constant')
        x2 = F.pad(x2, pad=padding, mode='constant')
        x3 = F.pad(x3, pad=padding, mode='constant')

        # SincNet backbone
        y1 = torch.zeros((batch_size, self.num_filters,num_stacked, length))
        y2 = torch.zeros((batch_size, self.num_filters,num_stacked, length))
        y3 = torch.zeros((batch_size, self.num_filters,num_stacked, length))

        for i in range(num_stacked):
            #y1[:,:,i,:] = nn.Dropout(0.3)(self.activation(F.max_pool1d(self.backbone(x1[:, i, :]), 1)))
            #y2[:,:,i,:] = nn.Dropout(0.3)(self.activation(F.max_pool1d(self.backbone(x2[:, i, :]), 1)))
            #y3[:,:,i,:] = nn.Dropout(0.3)(self.activation(F.max_pool1d(self.backbone(x3[:, i, :]), 1)))
            y1[:, :, i, :] = self.backbone(x1[:, i, :])
            y2[:, :, i, :] = self.backbone(x2[:, i, :])
            y3[:, :, i, :] = self.backbone(x3[:, i, :])
            

        # GCC-PHAT
        cc_12 = self.gcc(y1, y2)
        cc_13 = self.gcc(y1, y3)
        cc_23 = self.gcc(y2, y3)

        cc = torch.cat((cc_12, cc_13, cc_23), dim=2)
        cc = cc.view(cc.shape[0], cc.shape[1]*cc.shape[2], cc.shape[3]) 

        cc = cc.to(x1.device)


        for k, layer in enumerate(self.mlp):
            s = cc.shape[2]
            padding = get_pad(size=s, kernel_size=self.mlp_kernels[k], stride=1, dilation=1)
            cc = F.pad(cc, pad=padding, mode='constant')
            cc = layer(cc)
        
        # Applying padding and the final convolutional layer
        s = cc.shape[2]
        padding = get_pad(size=s, kernel_size=self.final_kernel, stride=1, dilation=1)
        cc = F.pad(cc, pad=padding, mode='constant')
        cc = self.final_conv(cc).reshape([batch_size, -1])


        cc = self.batch_norm(cc)
        cc = self.leaky_relu(cc)
        cc = self.linear(cc)
        cc = F.softmax(cc, dim=1)
        return cc
    
    def create_gcc(self, x1, x2, x3):
        batch_size = x1.shape[0]
        num_stacked = x1.shape[1]
        length = x1.shape[2]

        # Padding the input
        padding = get_pad(size=length, kernel_size=self.num_taps, stride=1, dilation=1)
        x1 = F.pad(x1, pad=padding, mode='constant')
        x2 = F.pad(x2, pad=padding, mode='constant')
        x3 = F.pad(x3, pad=padding, mode='constant')

        # SincNet backbone
        y1 = torch.zeros((batch_size, self.num_filters,num_stacked, length))
        y2 = torch.zeros((batch_size, self.num_filters,num_stacked, length))
        y3 = torch.zeros((batch_size, self.num_filters,num_stacked, length))

        for i in range(num_stacked):
            y1[:, :, i, :] = self.backbone(x1[:, i, :])
            y2[:, :, i, :] = self.backbone(x2[:, i, :])
            y3[:, :, i, :] = self.backbone(x3[:, i, :])

        # GCC-PHAT
        cc_12 = self.gcc(y1, y2)
        cc_13 = self.gcc(y1, y3)
        cc_23 = self.gcc(y2, y3)

        cc = torch.cat((cc_12, cc_13, cc_23), dim=3)
        return cc
            
