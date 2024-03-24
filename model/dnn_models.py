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
         # x = self.drop[i](self.act[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i])))
         x = F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i])
         x = self.act[i](x)
         x = self.drop[i](x)
       return x


