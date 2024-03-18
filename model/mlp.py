from dnn_models import SincNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from GCC import GCC


def get_pad(size, kernel_size, stride=1, dilation=1):
    effective_kernel_size = (kernel_size - 1) * dilation + 1
    pad_total = max(0, (size - 1) * stride + effective_kernel_size - size)
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before
    return (pad_before, pad_after)


class NGCCPHAT(nn.Module):
    def __init__(self, 
                 max_tau, 
                 num_channels, 
                 conv_channels,
                 sincnet_params,
                 num_stacked,
                 ):
        super().__init__()

        '''
        Neural GCC-PHAT with SincNet backbone

        arguments:
        max_tau - the maximum possible delay considered
        use_sinc - use sincnet backbone if True, otherwise use regular conv layers
        sig_len - length of input signal
        n_channel - number of gcc correlation channels to use
        fs - sampling frequency
        conv_channels - number of channels in the convolutional layers

        '''

        self.max_tau = max_tau
        self.num_stacked = num_stacked
        self.num_channels = num_channels
        self.backbone = SincNet(sincnet_params)
        self.mlp_kernels = [(self.num_stacked, 11), (self.num_stacked, 9) , ( self.num_stacked, 7)]
        self.mlp_kernels = [11, 9, 7]
        self.channels = [num_channels*3, conv_channels, conv_channels, conv_channels]
        self.final_kernel = 5
        self.gcc = GCC(max_tau = max_tau)
        self.mlp = nn.ModuleList([nn.Sequential(
                nn.Conv1d(self.channels[i], self.channels[i+1], kernel_size=k),
                nn.BatchNorm1d(self.channels[i+1]),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.5)) for i, k in enumerate(self.mlp_kernels)])
        self.final_conv = nn.Conv1d(conv_channels, 1, kernel_size=self.final_kernel)

        self.batch_norm = nn.BatchNorm1d((2 * self.max_tau + 1)*self.num_stacked)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.linear = nn.Linear((2 * self.max_tau + 1)*self.num_stacked, 2)
    def forward(self, x1, x2, x3):

        batch_size = x1.shape[0]
        num_stacked = x1.shape[1]
        length = x1.shape[2]

        # SincNet backbone

        y1 = torch.zeros((batch_size, self.num_channels,num_stacked, length))
        y2 = torch.zeros((batch_size, self.num_channels,num_stacked, length))
        y3 = torch.zeros((batch_size, self.num_channels,num_stacked, length))

        for i in range(num_stacked):
            y1[:, :, i, :] = self.backbone(x1[:, i, :])
            y2[:, :, i, :] = self.backbone(x2[:, i, :])
            y3[:, :, i, :] = self.backbone(x3[:, i, :])

        # GCC-PHAT
        cc_12 = self.gcc(y1, y2)
        cc_13 = self.gcc(y1, y3)
        cc_23 = self.gcc(y2, y3)

        # concatenating the GCC-PHAT outputs from all pairs of microphones.
        cc = torch.cat((cc_12, cc_13, cc_23), dim=1)
        cc = cc.view(cc.shape[0], cc.shape[1], cc.shape[2]*cc.shape[3]) # Flatten the tensor
        
        for k, layer in enumerate(self.mlp):
            s = cc.shape[2]
            padding = get_pad(
                size=s, kernel_size=self.mlp_kernels[k], stride=1, dilation=1)
            cc = F.pad(cc, pad=padding, mode='constant')
            cc = layer(cc)
        s = cc.shape[2]
        padding = get_pad(
            size=s, kernel_size=self.final_kernel, stride=1, dilation=1)
        cc = F.pad(cc, pad=padding, mode='constant')
        cc = self.final_conv(cc).reshape([batch_size, -1])
        cc = self.batch_norm(cc)
        cc = self.leaky_relu(cc)
        cc = self.linear(cc)
        return cc
    

    
