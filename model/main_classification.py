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


from mlp_classification_lstm import SincNetGCC
from dataset_classification import pre_stacked

import matplotlib
matplotlib.use('TkAgg')  # Set the backend to TkAgg, or choose another appropriate backend
import matplotlib.pyplot as plt

import os

'''
Defining all the parameters
'''
data_path = r"data\processed_data\all_data_filtered_velocity_0.0_radius_100_num_sectors_8.pkl"
print('Loading data...')
data = pd.read_pickle(data_path)
data.reset_index(drop=True, inplace=True)
# shuffling the data
data = data.sample(frac=1).reset_index(drop=True)  # Shuffle the data

# balance the data by including the same percentage of each value for the direction class. 
data = data.groupby('direction_label').apply(lambda x: x.sample(data.direction_label.value_counts().min(), random_state=42)).reset_index(drop=True)
seed = 42


# Training hyperparams
batch_size = 32
epochs = 100
lr = 0.01               # learning rate
wd = 0.01              # weight decay
patience = 5           # Number of epochs to wait for improvement before stopping

# Model parameters
max_tau = 20             # maximum tau value for GCC-PHAT
num_filters = 3         # Number of sinc filters in the first conv layer in the backbone
num_channels = 2         # Number of convolutional filters in the MLP layers
activation = 'leaky_relu'
max_cut_off_frequency = 30000
min_cut_off_frequency = 1000
min_band_width = 50
num_taps = 1001
sampling_rate = 204800
number_of_stacked = len(data.sensor_1.values[0])  # number of stacked samples
n_outputs = data.direction_label.nunique()        # Number of output classes


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
training_data, validation_data = train_test_split(data, test_size=0.2, random_state=seed, shuffle=False)

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

train_set = pre_stacked(training_data)
val_set = pre_stacked(validation_data)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False   , drop_last=True)

print(f'Training set size: {len(train_set)}')
print(f'Validation set size: {len(val_set)}')


model = SincNetGCC(
    max_tau = max_tau,
    num_filters = num_filters,
    num_channels = num_channels,
    activation = activation,
    max_fc = max_cut_off_frequency,
    min_fc = min_cut_off_frequency,
    min_band = min_band_width,
    num_taps = num_taps,
    fs = sampling_rate,
    num_stacked = number_of_stacked,
    n_outputs = n_outputs,
)

model.eval()
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs )
loss_fn = nn.CrossEntropyLoss()

training_loss = []
validation_loss = []

best_val_loss = float('inf')
epochs_no_improve = 0

# Before training starts
initial_low_hz =  model.backbone.lower_cut_off_frequency.data.clone()
initial_band_hz = model.backbone.band_widths.data.clone()

low_cut_off = [initial_low_hz]
band_width = [initial_band_hz]



for e in range(epochs):
    train_loss_epoch = 0
    model.train()
    for batch_idx, (x1, x2, x3, target) in enumerate(tqdm(train_loader, desc=f"Epoch {e+1}/{epochs} Training")):
        predicted = model(x1, x2, x3)
        loss = loss_fn(predicted, target)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad() # moved this line to the end of the loop
        train_loss_epoch += loss.item()
    training_loss.append(train_loss_epoch / len(train_loader))
    scheduler.step()

    # Adding the current low cut off and band width to the list
    low_cut_off.append(model.backbone.lower_cut_off_frequency.data.clone()) 
    band_width.append(model.backbone.band_widths.data.clone())

    # Print absolute change in low cut off and band width
    print(f'Epoch {e+1}/{epochs}  - Low cut off change: {torch.abs(low_cut_off[-1] - low_cut_off[-2]).sum().item():.4f} - Band width change: {torch.abs(band_width[-1] - band_width[-2]).sum().item():.4f}')

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
    
    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), f'{run_path}/model.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve == patience:
            print(f'Early stopping triggered after epoch {e+1}. No improvement in validation loss for {patience} consecutive epochs.')
            break 

torch.save(optimizer.state_dict(), f'{run_path}/optimizer.pth')
#torch.save(scheduler.state_dict(), f'{run_path}/scheduler.pth')

# Plotting the loss
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(training_loss, label='Training Loss')
ax.plot(validation_loss, label='Validation Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
plt.savefig(f'{run_path}/loss.png')
plt.close()


# Confusion matrix
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

labels = np.arange(n_outputs).astype(int)
cm = confusion_matrix(p_true, p_predicted, labels = labels) 
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap='viridis', ax=ax)

ax.set_xlabel('Predicted')
ax.set_ylabel('True')

ax.set_title('Confusion matrix')
fig.tight_layout()
plt.savefig(f'{run_path}/confusion_matrix.png')



# Cross Correlation
cc_path = f'{run_path}/cross_correlation_plots'
os.mkdir(cc_path)

filters = model.backbone.filters.detach().numpy()
dt = len(data.sensor_1[0][0]) * 1/ sampling_rate
time = np.arange(0, dt*number_of_stacked, dt)
time_shift = np.arange(-max_tau, max_tau+1) * 1/sampling_rate
with torch.no_grad():
    for batch_idx, (x1, x2, x3, labels) in enumerate(tqdm(val_loader, desc="Predicting")):
        cc = model.create_gcc(x1, x2, x3)
        cc = cc.detach().numpy()
        for batch in range(cc.shape[0]):
            for filter in range(cc.shape[1]):
                fig, ax = plt.subplots(1, 3, figsize=(30, 10))
                

                im0 = ax[0].pcolormesh(time_shift*1000 , time*1000, cc[batch, filter, :number_of_stacked, :])
                im1 = ax[1].pcolormesh(time_shift*1000 , time*1000, cc[batch, filter, number_of_stacked:2*number_of_stacked, :])
                im2 = ax[2].pcolormesh(time_shift*1000 , time*1000, cc[batch, filter, 2*number_of_stacked:,:])

                fig.colorbar(im0, ax=ax[0])
                fig.colorbar(im1, ax=ax[1])
                fig.colorbar(im2, ax=ax[2])

                ax[0].set_title(f'GCC12 for Filter nr {filter}')
                ax[1].set_title(f'GCC13 for Filter nr {filter}')
                ax[2].set_title(f'GCC23 for Filter nr {filter}')

                ax[0].set_aspect('auto')
                ax[1].set_aspect('auto')
                ax[2].set_aspect('auto')

                ax[0].set_xlabel('Time delay [ms]')
                ax[1].set_xlabel('Time delay [ms]')
                ax[2].set_xlabel('Time delay [ms]')

                ax[0].set_ylabel('Time [ms]')
                ax[1].set_ylabel('Time [ms]')
                ax[2].set_ylabel('Time [ms]')
                fig.tight_layout()
                
                plt.savefig(f'{cc_path}/GCC_for_filter_output_nr{filter}.png')
                plt.close() 
            break


# Frequency response of the filters
"""
freq_path = f'{run_path}/filters'
os.mkdir(freq_path)

freq_response = []
sampling_rate = 204800
for i in range(filters.shape[0]):
    freq_response.append(np.abs(np.fft.rfft(filters[i, 0, :])))

freq = np.fft.rfftfreq(filters.shape[2], d=1/sampling_rate)
for i in range(len(freq_response)):
    fig, ax = plt.subplots(1,2, figsize=(10,10))
    f_0 = model.backbone.lower_cut_off_frequency.data[i].item()
    delta_f = model.backbone.band_widths.data[i].item()

    ax[0].plot(freq, 20*np.log10(freq_response[i]))
    ax[0].set_title(f'Frequency Response for filter {i}')
    ax[0].set_xlabel('Frequency [Hz]')
    ax[0].set_ylabel('Amplitude [dB]')

    ax[1].plot(filters[i, 0, :])
    ax[1].set_title(f'Impulse Response for filter{i}')
    ax[1].set_xlabel('Samples')
    ax[1].set_ylabel('Amplitude')

    fig.suptitle(f'Filter {i} - f_0: {f_0:.2f} Hz, Δf: {delta_f:.2f} Hz')

    fig.tight_layout()
    plt.savefig(f'{freq_path}/filter_{i}.png')
    plt.close()

"""
# Plotting a heatmap of the low cut off and band width

low_cut_off = torch.stack(low_cut_off).detach().numpy()[:,:,0]*max_cut_off_frequency
band_width = torch.stack(band_width).detach().numpy()[:,:,0]*max_cut_off_frequency

fig, ax = plt.subplots(1, 2, figsize=(20, 10))
im0 = ax[0].imshow(low_cut_off.T, aspect='auto')
im1 = ax[1].imshow(band_width.T, aspect='auto')

ax[0].set_title('Low cut off frequency')
ax[1].set_title('Band width')

ax[0].set_xlabel('Epoch')
ax[1].set_xlabel('Epoch')

ax[0].set_ylabel('Filter nr')
ax[1].set_ylabel('Filter nr')

fig.colorbar(im0, ax=ax[0])
fig.colorbar(im1, ax=ax[1])

fig.tight_layout()
plt.savefig(f'{run_path}/low_cut_off_band_width.png')
plt.close()


# Saving the parameters



parameters = {
    'data_path': data_path,
    'seed': seed,
    'batch_size': batch_size,
    'epochs': epochs,
    'learning rate': lr,
    'weight decay': wd,
    'max_tau': max_tau,
    'num_channels': num_channels,
    'num_filters': num_filters,
    'sampling rate': sampling_rate,
    'patience': patience,
    'number_of_stacked': number_of_stacked,
    'n_outputs': n_outputs,
    'training_set_size': len(train_set),
    'validation_set_size': len(val_set),
    'best_val_loss': best_val_loss,
    'early_stopping': epochs_no_improve,

}

with open(f'{run_path}/parameters.txt', 'w') as f:
    for key, value in parameters.items():
        f.write(f'{key}: {value}\n')


np.save(f'{run_path}/training_loss.npy', np.array(training_loss))
np.save(f'{run_path}/validation_loss.npy', np.array(validation_loss))



