import torch
import numpy as np
from model.helper import determine_direction


class pre_stacked(torch.utils.data.Dataset):
    def __init__(self, data):
        self.s1 = data.sensor_1.values
        self.s2 = data.sensor_2.values
        self.s3 = data.sensor_3.values

        self.direction = data.direction.values

    def __len__(self):
        return len(self.s1)

    def __getitem__(self, idx):

        s1 = self.s1[idx]
        s2 = self.s2[idx]
        s3 = self.s3[idx]

        direction = self.direction[idx] # [0,1,0,0] - sector 2


        # turning them into torch tensors
        s1 = torch.tensor(s1, dtype=torch.float32)
        s2 = torch.tensor(s2, dtype=torch.float32)
        s3 = torch.tensor(s3, dtype=torch.float32)
        target = torch.tensor(direction, dtype=torch.float32)

        return s1, s2, s3, target