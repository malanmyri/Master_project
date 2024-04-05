import random
from tqdm import tqdm
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt
import os

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

dataset = r"data\preprocessed_data\dataset_17"

training_data_path = dataset + r"\train\train.pkl"
validation_data_path = dataset + r"\val\val.pkl"
test_data_path = dataset + r"\test\test.pkl"


from regression.conv_GCC_regression_architecture import Net
from dataset import regression_stacked

architecture_name = "convolution_architecture_regression"
batch_size = 32 
epochs = 100
lr = 0.001         
wd = 0.01            
patience = 5



def train(training_data_path, 
          validation_data_path, 
          test_data_path, 
          batch_size, 
          epochs, 
          lr, 
          wd, 
          patience):
    
    
    training_data = pd.read_pickle(training_data_path)
    validation_data = pd.read_pickle(validation_data_path)
    test_data =       pd.read_pickle(test_data_path)
    
    
    runs = os.listdir('runs')
    run_number = len(runs)
    run_path = f'runs/run_{run_number}'
    os.mkdir(run_path)
    num_stacked = training_data.sensor_1[0].shape[0]
    n_outputs = len(training_data.direction_label.unique())


    train_set = regression_stacked(training_data)
    val_set =   regression_stacked(validation_data)
    test_set =  regression_stacked(test_data)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, drop_last=True)

    print(f'Training set size: {len(train_set)}')
    print(f'Validation set size: {len(val_set)}')
    print(f'Test set size: {len(test_set)}')

    model = Net(num_stacked,n_outputs)
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

        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f'Early stopping triggered after epoch {e+1}. No improvement in validation loss for {patience} consecutive epochs.')
                break 


    torch.save(model.state_dict(), f'{run_path}/model.pth')
    torch.save(optimizer.state_dict(), f'{run_path}/optimizer.pth')
    torch.save(scheduler.state_dict(), f'{run_path}/scheduler.pth')

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(training_loss, label='Training Loss')
    ax.plot(validation_loss, label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.set_yscale('log')
    plt.savefig(f'{run_path}/loss.png')
    plt.close()

    # calculating the loss over the test set
    model.eval()
    test_loss = 0
    predicted_list = []
    target_list = []

    for batch_idx, (x1, x2, x3, target) in enumerate(tqdm(test_loader, desc="Testing")):
        with torch.no_grad():
            predicted = model(x1, x2, x3)
            loss = loss_fn(predicted, target)

            predicted_list.append(predicted)
            target_list.append(target)
            test_loss += loss.item()
    test_loss /= len(test_loader)

    # plotting the predicted vs true values
    predicted = torch.cat(predicted_list).detach().numpy()
    target = torch.cat(target_list).detach().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.scatter(target[:, 0], target[:, 1], s=1)
    ax.scatter(predicted[:, 0], predicted[:, 1], s=1)
    ax.set_xlabel('dx')
    ax.set_ylabel('dy')
    plt.savefig(f'{run_path}/predicted_vs_true.png')

    parameters = {
        'data_paths ': [training_data_path, validation_data_path, test_data_path],
        'batch_size': batch_size,
        'epochs': epochs,
        'lr': lr,
        'wd': wd,
        'patience': patience,
        'n_outputs': n_outputs,
        'num_stacked': num_stacked,
        "architecture": architecture_name,
        'test_loss': test_loss
    }

    with open(f'{run_path}/parameters.txt', 'w') as f:
        for key, value in parameters.items():
            f.write(f'{key}: {value}\n')


    np.save(f'{run_path}/training_loss.npy', np.array(training_loss))
    np.save(f'{run_path}/validation_loss.npy', np.array(validation_loss))

train(training_data_path,
        validation_data_path,
        test_data_path,
        batch_size,
        epochs,
        lr,
        wd,
        patience)
