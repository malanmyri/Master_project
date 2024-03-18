import torch
import numpy as np

'''
Raw Description 
- This class is used to create a dataset object for the raw data

'''
class RAW(torch.utils.data.Dataset):
    def __init__(self, data):
        self.s1 = data.sensor_1.values
        self.s2 = data.sensor_2.values
        self.s3 = data.sensor_3.values
        self.x = data.x.values
        self.y = data.y.values

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
    


class STACKED_dx_dy(torch.utils.data.Dataset):
    def __init__(self, data, number_of_stacked):
        self.s1 = data.sensor_1.values
        self.s2 = data.sensor_2.values
        self.s3 = data.sensor_3.values
        self.dx = data.dx.values
        self.dy = data.dy.values
        self.number_of_stacked = number_of_stacked

    def __len__(self):
        return len(self.s1)//self.number_of_stacked

    def __getitem__(self, idx):

        s1 = np.vstack(self.s1[idx*self.number_of_stacked:(idx+1)*self.number_of_stacked])
        s2 = np.vstack(self.s2[idx*self.number_of_stacked:(idx+1)*self.number_of_stacked])
        s3 = np.vstack(self.s3[idx*self.number_of_stacked:(idx+1)*self.number_of_stacked])

        # summing up the dx and dy values
        dx = np.sum(self.dx[idx*self.number_of_stacked:(idx+1)*self.number_of_stacked])
        dy = np.sum(self.dy[idx*self.number_of_stacked:(idx+1)*self.number_of_stacked])


        # turning them into torch tensors
        s1 = torch.tensor(s1, dtype=torch.float32)
        s2 = torch.tensor(s2, dtype=torch.float32)
        s3 = torch.tensor(s3, dtype=torch.float32)
        target = torch.tensor([dx, dy], dtype=torch.float32)

        return s1, s2, s3, target