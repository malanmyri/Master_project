import torch
import torch.nn as nn
import numpy as np

class GCC(nn.Module):
    def __init__(self, max_tau):
        super().__init__()

        ''' GCC implementation based on Knapp and Carter,
        "The Generalized Correlation Method for Estimation of Time Delay",
        IEEE Trans. Acoust., Speech, Signal Processing, August, 1976 '''

        self.max_tau = max_tau
        self.epsilon = 0.000001
        self.beta = np.array([1])

    def forward(self, x, y):
        n = x.shape[-1] + y.shape[-1]
        X = torch.fft.rfft(x, n=n)
        Y = torch.fft.rfft(y, n=n)
        Gxy = X * torch.conj(Y)
        phi = 1 / (torch.abs(Gxy) + self.epsilon)
        cc = []
        for i in range(self.beta.shape[0]):
            cc.append(torch.fft.irfft(
                Gxy * torch.pow(phi, self.beta[i]), n))
        cc = torch.cat(cc, dim=1)
        cc = torch.cat((cc[:, :, -self.max_tau:], cc[:, :, :self.max_tau+1]), dim=-1)
        return cc  
