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
from dataset import STACKED_dx_dy
from GCC import GCC

import matplotlib
matplotlib.use('TkAgg')  # Set the backend to TkAgg, or choose another appropriate backend
import matplotlib.pyplot as plt

import os

'''
Defining all the parameters
'''
data_path = r"data\training_data_sample_1.pkl"
seed = 42

# Training hyperparams
batch_size = 32
epochs = 10
lr = 0.001         # learning rate
wd = 0.01          # weight decay
patience = 5       # Number of epochs to wait for improvement before stopping

# Model parameters
max_tau = 100       # maximum tau value for GCC-PHAT
num_channels = 1   # number of channels in final layer of NGCCPHAT backbone
conv_channels = 32  # number of channels in the convolutional layers of NGCCPHAT backbone
fs = 204800         # sampling rate
sig_len = 1024      # length of snippet used for tdoa estimation


number_of_stacked = 1




sincnet_params = {'input_dim': sig_len,
                          'fs': fs,
                          'cnn_N_filt': [128,   num_channels],
                          'cnn_len_filt': [1023,  7],
                          'cnn_max_pool_len': [1, 1],
                          'cnn_use_laynorm_inp': False,
                          'cnn_use_batchnorm_inp': False,
                          'cnn_use_laynorm': [False,   False],
                          'cnn_use_batchnorm': [True,  True],
                          'cnn_act': ['leaky_relu', 'linear'],
                          'cnn_drop': [0.0, 0.0],
                          }

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
data = data[:20]
training_data, validation_data = train_test_split(data, test_size=0.2, random_state=seed, shuffle=True,)



"""# plotting the data distribution
print('Plotting data distribution...')
fig, ax = plt.subplots(1, 3, figsize=(20, 5))
ax[0].hist(training_data.x, bins=100)
ax[0].set_title('X')
ax[1].hist(training_data.y, bins=100)
ax[1].set_title('Y')
ax[2].scatter(training_data.x, training_data.y, s=1)
ax[2].set_title('X vs Y')

ax[0].set_xlim(-1, 1)
ax[1].set_xlim(-1, 1)
ax[2].set_xlim(-1, 1)
ax[2].set_ylim(-1,1)



plt.savefig(f'{run_path}/data_distribution.png')
plt.close()"""


train_set = STACKED_dx_dy(training_data, number_of_stacked)
val_set = STACKED_dx_dy(validation_data, number_of_stacked)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

print(f'Training set size: {len(train_set)}')
print(f'Validation set size: {len(val_set)}')


model = NGCCPHAT(max_tau, num_channels, conv_channels, sincnet_params)
model.eval()
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
loss_fn = nn.MSELoss()

training_loss = []
validation_loss = []

best_val_loss = float('inf')
epochs_no_improve = 0

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

    current_val_loss = val_loss_epoch / len(val_loader)
    validation_loss.append(current_val_loss)
    print(f'Epoch {e+1}/{epochs} - Training Loss: {training_loss[-1]:.4f} - Validation Loss: {validation_loss[-1]:.4f}')

    # Early stopping logic
    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        epochs_no_improve = 0
        # Optional: Save model here if current validation loss has improved
    else:
        epochs_no_improve += 1
        if epochs_no_improve == patience:
            print(f'Early stopping triggered after epoch {e+1}. No improvement in validation loss for {patience} consecutive epochs.')
            break  # Break out of the loop


# Saving all parameters

torch.save(model.state_dict(), f'{run_path}/model.pth')
torch.save(optimizer.state_dict(), f'{run_path}/optimizer.pth')
torch.save(scheduler.state_dict(), f'{run_path}/scheduler.pth')

# Plotting the loss
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(training_loss, label='Training Loss')
ax.plot(validation_loss, label='Validation Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
ax.set_yscale('log')
plt.savefig(f'{run_path}/loss.png')
plt.close()

# saving all parameters in txt file line by line

parameters = {
    'data_path': data_path,
    'seed': seed,
    'batch_size': batch_size,
    'epochs': epochs,
    'lr': lr,
    'wd': wd,
    'max_tau': max_tau,
    'num_channels': num_channels,
    'conv_channels': conv_channels,
    'fs': fs,
    'sig_len': sig_len,
    'sincnet_params': sincnet_params,
    'patience': patience,

}

with open(f'{run_path}/parameters.txt', 'w') as f:
    for key, value in parameters.items():
        f.write(f'{key}: {value}\n')


# Saving the loss in a numpy file
np.save(f'{run_path}/training_loss.npy', np.array(training_loss))
np.save(f'{run_path}/validation_loss.npy', np.array(validation_loss))



