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

dataset = r"data\preprocessed_data\dataset_19"

training_data_path = dataset + r"\train\train.pkl"
validation_data_path = dataset + r"\val\val.pkl"
test_data_path = dataset + r"\test\test.pkl"


from conv_GCC_architecture import Net
from dataset import classification_stacked

architecture_name = "conv_GGC_architecture"
batch_size = 32 
epochs = 300
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


    train_set = classification_stacked(training_data)
    val_set =   classification_stacked(validation_data)
    test_set =  classification_stacked(test_data)

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
    loss_fn = nn.CrossEntropyLoss()

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

    p_predicted = []
    p_true = []

    with torch.no_grad():
        for batch_idx, (x1, x2, x3, labels) in enumerate(tqdm(test_loader, desc="Predicting")):
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


    parameters = {
        'data_paths ': [training_data_path, validation_data_path, test_data_path],
        'batch_size': batch_size,
        'epochs': epochs,
        'lr': lr,
        'wd': wd,
        'patience': patience,
        'n_outputs': n_outputs,
        'best_val_loss': best_val_loss,
        "architecture": architecture_name,
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
