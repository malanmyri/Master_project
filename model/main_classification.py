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
from classification import determine_direction

from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


from mlp_classification import NGCCPHAT
from dataset_classification import STACKED_dx_dy

import matplotlib
matplotlib.use('TkAgg')  # Set the backend to TkAgg, or choose another appropriate backend
import matplotlib.pyplot as plt

import os

'''
Defining all the parameters
'''
data_path = r"data\training_data_sample_1.pkl"
print('Loading data...')
data = pd.read_pickle(data_path)

seed = 42

# Training hyperparams
batch_size = 4
epochs = 10
lr = 0.001             # learning rate
wd = 0.01              # weight decay
patience = 5           # Number of epochs to wait for improvement before stopping

# Model parameters
max_tau = 60           # maximum tau value for GCC-PHAT
num_channels = 10       # number of channels in final layer of NGCCPHAT backbone
conv_channels = 15     # number of channels in the convolutional layers of NGCCPHAT backbone
fs = 204800            # sampling rate
number_of_stacked = 40 # number of stacked snippets
n_outputs = 4          # number of kvadrants classification


sincnet_params = {'input_dim': len(data.sensor_1[0]),
                          'fs': fs,
                          'cnn_N_filt': [20,   num_channels],
                          'cnn_len_filt': [1023,  7],
                          'cnn_max_pool_len': [1, 1],
                          'cnn_act': ['leaky_relu', 'linear'],
                          'cnn_drop': [0.0, 0.0],
                          'max_hz': 20000.0,
                          'low_hz': 1000.0,
                          'min_band_hz': 1000.0,
                          }

# For reproducibility
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

runs = os.listdir('runs')
run_number = len(runs)
run_path = f'runs/run_{run_number}'
os.mkdir(run_path)


# Normalising the dx and dy values
max = data[["dx", "dy"]].abs().max().max()
print(f"The maximum value of dx and dy is: {max}")
data["dx"] = data["dx"]/max
data["dy"] = data["dy"]/max

# Splitting the data into training and validation sets
training_data, validation_data = train_test_split(data, test_size=0.2, random_state=seed, shuffle=True,)

print('Plotting data distribution...')
fig, ax = plt.subplots(1, 3, figsize=(20, 5))

ax[0].hist(training_data['dx'], bins=100, alpha=0.5, label='Training')
ax[0].hist(validation_data['dx'], bins=100, alpha=0.5, label='Validation')
ax[0].set_title('dx distribution')
ax[0].legend()

ax[1].hist(training_data['dy'], bins=100, alpha=0.5, label='Training')
ax[1].hist(validation_data['dy'], bins=100, alpha=0.5, label='Validation')
ax[1].set_title('dy distribution')  
ax[1].legend()

ax[2].scatter(training_data['dx'], training_data['dy'], alpha=0.5, label='Training')
ax[2].scatter(validation_data['dx'], validation_data['dy'], alpha=0.5, label='Validation')  
ax[2].set_title('dx vs dy')
ax[2].legend()

ax[0].set_xlim(-1, 1)
ax[1].set_xlim(-1, 1)
ax[2].set_xlim(-1, 1)
ax[2].set_ylim(-1,1)

plt.savefig(f'{run_path}/data_distribution.png')
plt.close()

train_set = STACKED_dx_dy(training_data, number_of_stacked, n_outputs)
val_set = STACKED_dx_dy(validation_data, number_of_stacked, n_outputs)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False   , drop_last=True)

print(f'Training set size: {len(train_set)}')
print(f'Validation set size: {len(val_set)}')


model = NGCCPHAT(max_tau, num_channels, conv_channels, sincnet_params,number_of_stacked, n_outputs)
model.eval()
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
loss_fn = nn.CrossEntropyLoss()

training_loss = []
validation_loss = []

best_val_loss = float('inf')
epochs_no_improve = 0

# Before training starts
initial_low_hz =  model.backbone.conv[0].low_hz_.data.clone()
initial_band_hz = model.backbone.conv[0].band_hz_.data.clone()


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

    delta_low_hz = torch.abs(initial_low_hz - model.backbone.conv[0].low_hz_.data)
    delta_band_hz = torch.abs(initial_band_hz - model.backbone.conv[0].band_hz_.data)
    print(f"Epoch {e}: Δ Low Hz = {delta_low_hz.mean().item()}, Δ Band Hz = {delta_band_hz.mean().item()}")

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


# 3. Confusion matrix
p_predicted = []
p_true = []

with torch.no_grad():
    for batch_idx, (x1, x2, x3, labels) in enumerate(tqdm(val_loader, desc="Predicting")):
        outputs = model(x1,x2,x3)
        p_predicted.append(outputs)
        p_true.append(labels)

p_predicted = np.concatenate(p_predicted, axis=0)
p_true = np.concatenate(p_true, axis=0)

p_predicted = np.argmax(p_predicted, axis=1)
p_true = np.argmax(p_true, axis=1) 

cm = confusion_matrix(p_true, p_predicted) 
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap='viridis', ax=ax)

ax.set_xlabel('Predicted')
ax.set_ylabel('True')

ax.set_title('Confusion matrix')
fig.tight_layout()
plt.savefig(f'{run_path}/confusion_matrix.png')



# Creating cross correlation plots
cc_path = f'{run_path}/cross_correlation_plots'
os.mkdir(cc_path)

filters = model.backbone.conv[0].filters.detach().numpy()
with torch.no_grad():
    for batch_idx, (x1, x2, x3, labels) in enumerate(tqdm(val_loader, desc="Predicting")):
        cc = model.create_gcc(x1, x2, x3)
        cc = cc.detach().numpy()
        for batch in range(cc.shape[0]):
            for stack in range(cc.shape[1]):
                fig, ax = plt.subplots(2, 1, figsize=(10, 5))
                ax = ax.ravel()
                ax[0].imshow(cc[batch, stack, :, :])
                ax[0].set_title(f'GCC for Filter nr {stack}')
                ax[0].set_aspect('auto')
                ax[0].set_xlabel('Tau')
                ax[0].set_ylabel('Time [s]')

                ax[1].plot(filters[stack, 0, :])
                ax[1].set_title('Filter')
                ax[1].set_xlabel('Time [s]')
                ax[1].set_ylabel('Amplitude')
                fig.tight_layout()
                plt.savefig(f'{cc_path}/GCC_for_filter_nr_{stack}.png')
                plt.close() 
            break


# Calculating the frequency response of the filters 
freq_path = f'{run_path}/frequency_response'
os.mkdir(freq_path)

freq_response = []
sampling_rate = 204800
for i in range(filters.shape[0]):
    freq_response.append(np.abs(np.fft.rfft(filters[i, 0, :])))

freq = np.fft.rfftfreq(filters.shape[2], d=1/sampling_rate)
for i in range(len(freq_response)):
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    ax.plot(freq, 20*np.log10(freq_response[i]))
    ax.set_title(f'Filter {i}')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Amplitude [dB]')
    fig.tight_layout()
    plt.savefig(f'{freq_path}/filter_{i}_frequency_response.png')
    plt.close()



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
    'sincnet_params': sincnet_params,
    'patience': patience,
    'number_of_stacked': number_of_stacked,
    'n_outputs': n_outputs,
    'training_set_size': len(train_set),
    'validation_set_size': len(val_set),
    'best_val_loss': best_val_loss,
}

with open(f'{run_path}/parameters.txt', 'w') as f:
    for key, value in parameters.items():
        f.write(f'{key}: {value}\n')


# Saving the loss in a numpy file
np.save(f'{run_path}/training_loss.npy', np.array(training_loss))
np.save(f'{run_path}/validation_loss.npy', np.array(validation_loss))



