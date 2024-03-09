
import random
from tqdm import tqdm
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import torch.optim as optim
from sklearn.model_selection import train_test_split

from mlp import NGCCPHAT
from dataset import RAW
from GCC import GCC

import matplotlib
matplotlib.use('TkAgg')  # Set the backend to TkAgg, or choose another appropriate backend
import matplotlib.pyplot as plt

import os

'''
Defining all the parameters
'''
data_path = r"data/training_data_smaller.pkl"
seed = 42

# Training hyperparams
batch_size = 32
epochs = 1
lr = 0.001  # learning rate
wd = 0.01  # weight decay

# Model parameters
model = 'NGCCPHAT'  
max_tau = 100
num_channels = 10  # number of channels in final layer of NGCCPHAT backbone
conv_channels = 32
use_sinc = False
fs = 204800  # sampling rate
sig_len = 1024  # length of snippet used for tdoa estimation


'''
Setting system seeds for reproducibility
'''

# For reproducibility
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


'''
Creating folder to store run
'''

runs = os.listdir('runs')
run_number = len(runs)
run_path = f'runs/run_{run_number}'
os.mkdir(run_path)


'''
Dataset creation
'''

print('Loading data...')
data = pd.read_pickle(data_path)
training_data, validation_data = train_test_split(data, test_size=0.2, random_state=seed, shuffle=True,)

print('Normalizing data...')
x_mean = training_data.x.mean()
y_mean = training_data.y.mean()
x_std =  training_data.x.std()
y_std =  training_data.y.std()

training_data.x = (training_data.x - x_mean) / x_std
training_data.y = (training_data.y - y_mean) / y_std

validation_data.x = (validation_data.x - x_mean) / x_std
validation_data.y = (validation_data.y - y_mean) / y_std

train_set = RAW(training_data, sig_len)
val_set = RAW(validation_data, sig_len)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

print('Data loaded and normalized')
print(f'Training set size: {len(train_set)}')
print(f'Validation set size: {len(val_set)}')



model = NGCCPHAT(max_tau, sig_len, num_channels, conv_channels, fs)
model.eval()
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
loss_fn = nn.MSELoss()

training_loss = []
validation_loss = []

for e in range(epochs):
    train_loss_epoch = 0
    model.train()
    for batch_idx, (x1, x2, x3, target) in enumerate(tqdm(train_loader, desc=f"Epoch {e+1}/{epochs} Training")):
        predicted = model(x1, x2, x3)
        loss = loss_fn(predicted, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_epoch += loss.item() 
    training_loss.append(train_loss_epoch / len(train_loader))
    scheduler.step()

    model.eval()
    val_loss_epoch = 0
    for batch_idx, (x1, x2, x3, target) in enumerate(tqdm(val_loader, desc=f"Epoch {e+1}/{epochs} Validation")):
        with torch.no_grad():
            predicted = model(x1, x2, x3) 
            loss = loss_fn(predicted, target)
            val_loss_epoch += loss.item()

    validation_loss.append(val_loss_epoch / len(val_loader))
    print(f'Epoch {e+1}/{epochs} - Training Loss: {training_loss[-1]:.4f} - Validation Loss: {validation_loss[-1]:.4f}')

