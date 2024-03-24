'''
This file contains the implementation of SincNet, by Mirco Ravanelli and Yoshua Bengio
Circular padding has been added before each convolution.
Source: https://github.com/mravanelli/SincNet
'''

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import math
from torch_pad import get_pad


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
        return F.conv1d(waveforms, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                        bias=None, groups=1)
class SincNet(nn.Module):

    def __init__(self, options):
       super(SincNet, self).__init__()

       self.cnn_N_filt = options['cnn_N_filt']
       self.cnn_len_filt = options['cnn_len_filt']
       self.cnn_max_pool_len = options['cnn_max_pool_len']
       self.cnn_act = options['cnn_act']
       self.cnn_drop = options['cnn_drop']
       self.input_dim = int(options['input_dim'])
       self.fs = options['fs']
       self.N_cnn_lay = len(options['cnn_N_filt'])
       self.max_cut_off_frequency = options['max_hz']
       self.low_hz = options['low_hz']
       self.min_band_hz = options['min_band_hz']
       self.conv = nn.ModuleList([])
       self.act = nn.ModuleList([])
       self.drop = nn.ModuleList([])
       self.bn = nn.ModuleList([])

       current_input = self.input_dim
       for i in range(self.N_cnn_lay):

         N_filt = int(self.cnn_N_filt[i])
         self.drop.append(nn.Dropout(p=self.cnn_drop[i]))
         self.act.append(act_fun(self.cnn_act[i]))
         self.bn.append(nn.BatchNorm1d(N_filt, momentum=0.05))
        
        # only the first layer has this sinc function
         if i == 0:
          self.conv.append(SincConv_fast(self.cnn_N_filt[0], self.cnn_len_filt[0], self.fs, self.max_cut_off_frequency, self.low_hz, self.min_band_hz))
         else:
          self.conv.append(nn.Conv1d(self.cnn_N_filt[i-1], self.cnn_N_filt[i], self.cnn_len_filt[i]))

         current_input = int((current_input-self.cnn_len_filt[i]+1)/self.cnn_max_pool_len[i])

       self.out_dim = current_input*N_filt

    def forward(self, x):
       batch = x.shape[0]
       seq_len = x.shape[-1]
       x = x.view(batch, 1, seq_len)
       for i in range(self.N_cnn_lay):
         padding = get_pad(size=seq_len, kernel_size=self.cnn_len_filt[i], stride=1, dilation=1)
         x = F.pad(x, pad=padding, mode='circular')
         x = self.drop[i](self.act[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i])))
       return x


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


