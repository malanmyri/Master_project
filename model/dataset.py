import torch
import numpy as np

# defining the dataset
class RAW(torch.utils.data.Dataset):
    def __init__(self, data, sig_len):
        self.s1 = data.sensor_1.values
        self.s2 = data.sensor_2.values
        self.s3 = data.sensor_3.values
        self.x = data.x.values
        self.y = data.y.values
        self.sig_len = sig_len

    def __len__(self):
        return len(self.s1)

    def __getitem__(self, idx):
        # x and y should be the average values of the coordinates in the sig_len window
        s1 = self.s1[idx]
        s2 = self.s2[idx]
        s3 = self.s3[idx]
        x = self.x[idx]
        y = self.y[idx]


        # turning them into torch tensors
        s1 = torch.tensor(s1, dtype=torch.float32)
        s2 = torch.tensor(s2, dtype=torch.float32)
        s3 = torch.tensor(s3, dtype=torch.float32)
        target = torch.tensor([x, y], dtype=torch.float32)

        return s1, s2, s3, target