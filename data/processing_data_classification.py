import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import math
import os
from sklearn.model_selection import train_test_split



def determine_direction(dx, dy, num_sectors):  

    """
    Determines the directional label for a given dx, dy pair based on the specified number of polar sectors,
    with the first sector centered at 0 degrees (directly to the right).

    Parameters:
    - dx: Change in x-coordinate.
    - dy: Change in y-coordinate.
    - num_sectors: Number of sectors to divide the polar coordinate system into.

    Returns:
    - A vector of length num_sectors with a 1 at the index corresponding to the sector of the dx, dy pair,
      and 0s elsewhere.
    """
    if num_sectors < 1:
        raise ValueError("Number of sectors must be at least 1.")
    
    angle = math.atan2(dy, dx)
    angle_degrees = (math.degrees(angle) + 360) % 360
    degrees_per_sector = 360 / num_sectors
    adjusted_angle_degrees = (angle_degrees + degrees_per_sector / 2) % 360
    sector = int(adjusted_angle_degrees // degrees_per_sector)
    
    # Initialize a vector with zeros
    direction_vector = [0] * num_sectors
    # Set the appropriate sector to 1
    direction_vector[sector] = 1

    return direction_vector

data_paths = [  r"data\processed_data\section_1\sample_1.pkl", 
                r"data\processed_data\section_1\sample_2.pkl",
                r"data\processed_data\section_2\sample_1.pkl",
                r"data\processed_data\section_2\sample_2.pkl",
                r"data\processed_data\section_3\sample_1.pkl",
                r"data\processed_data\section_3\sample_2.pkl",
                r"data\processed_data\section_4\sample_2.pkl",
                r"data\processed_data\section_5\sample_2.pkl",
                r"data\processed_data\section_5\sample_3.pkl",
                r"data\processed_data\section_6\sample_1.pkl",
                r"data\processed_data\section_6\sample_2.pkl",
]


window_size = 3000
stride = 1000
num_stacked = 1

radius_velocity = 0.2
num_sectors = 16

window_hamming = np.hamming(window_size)

datasets = os.listdir(r'data\preprocessed_data')
dataset_number = len(datasets)
dataset_path = f'data\preprocessed_data\dataset_{dataset_number}'
os.mkdir(dataset_path)

# Creating train, val and test directories
train_path = f'{dataset_path}/train'
val_path = f'{dataset_path}/val'
test_path = f'{dataset_path}/test'

os.mkdir(train_path)
os.mkdir(val_path)
os.mkdir(test_path)




training_data_all = pd.DataFrame(columns = ['sensor_1', 'sensor_2', 'sensor_3',  'dx', 'dy', 'section', 'sample'])
for data_path in data_paths:

    name = data_path.split("\\")
    section = name[2]
    sample = name[3].split(".")[0]

    training_data = pd.read_pickle(data_path)

    training_data_window = pd.DataFrame(columns = ['sensor_1', 'sensor_2', 'sensor_3', 't', 'x', 'y', "dx", "dy"])
    
    training_data["dx"] = training_data.x.diff()/training_data.t.diff()
    training_data["dy"] = training_data.y.diff()/training_data.t.diff()

    for i in tqdm(range(0, len(training_data) - window_size, stride)):
        s1 = training_data.sensor_1.iloc[i:i+window_size]
        s2 = training_data.sensor_2.iloc[i:i+window_size]
        s3 = training_data.sensor_3.iloc[i:i+window_size]
        
        """
        s1 = s1 * window_hamming
        s2 = s2 * window_hamming
        s3 = s3 * window_hamming
        """
        
        t =  training_data.t.iloc[i:i+window_size].mean()
        x =  training_data.x.iloc[i:i+window_size].mean()
        y =  training_data.y.iloc[i:i+window_size].mean()
        dx = training_data.dx.iloc[i:i+window_size].mean()
        dy = training_data.dy.iloc[i:i+window_size].mean()


        training_data_window.loc[len(training_data_window)] = [s1, s2, s3, t, x, y, dx, dy]


    training_data_windowed_stacked = pd.DataFrame(columns = ['sensor_1', 'sensor_2', 'sensor_3', 'dx', 'dy'])

    for i in tqdm(range( len(training_data_window)//num_stacked)):
        sensor_1 = training_data_window.sensor_1.iloc[i*num_stacked:(i+1)*num_stacked]
        sensor_2 = training_data_window.sensor_2.iloc[i*num_stacked:(i+1)*num_stacked]
        sensor_3 = training_data_window.sensor_3.iloc[i*num_stacked:(i+1)*num_stacked]
        stacked_sensor_1 = np.vstack(sensor_1.values)
        stacked_sensor_2 = np.vstack(sensor_2.values)
        stacked_sensor_3 = np.vstack(sensor_3.values)
        
        dx = training_data_window.dx.iloc[i*num_stacked:(i+1)*num_stacked].mean()
        dy = training_data_window.dy.iloc[i*num_stacked:(i+1)*num_stacked].mean()
        
        training_data_windowed_stacked.loc[len(training_data_windowed_stacked)] = [stacked_sensor_1, stacked_sensor_2, stacked_sensor_3, dx, dy]

    training_data_windowed_stacked['section'] = section
    training_data_windowed_stacked['sample'] = sample

    training_data_all = pd.concat([training_data_all, training_data_windowed_stacked], axis=0)

    print(f"Done loading {data_path}")


training_data_window_stacked = training_data_all


# Preprocessing
training_data_window_stacked["direction"] = training_data_window_stacked.apply(lambda row: determine_direction(row["dx"], row["dy"], num_sectors), axis=1)
training_data_window_stacked["direction_label"] = training_data_window_stacked["direction"].apply(lambda x: np.argmax(x))

# balancing the dataset by the direction label
training_data_window_stacked = training_data_window_stacked.groupby('direction_label').apply(lambda x: x.sample(training_data_window_stacked.direction_label.value_counts().min()))

# Normalizing dx and dy
max_dx = training_data_window_stacked.dx.abs().max()
max_dy = training_data_window_stacked.dy.abs().max()

training_data_window_stacked.dx = training_data_window_stacked.dx/max_dx
training_data_window_stacked.dy = training_data_window_stacked.dy/max_dy

training_data_window_stacked = training_data_window_stacked[training_data_window_stacked.dx**2 + training_data_window_stacked.dy**2 > radius_velocity**2]
training_data_window_stacked = training_data_window_stacked.reset_index(drop=True)

train_perc = 0.7
val_perc = 0.2
test_perc = 0.1

train_data, temp_data = train_test_split(
    training_data_window_stacked,
    train_size=train_perc,
    stratify=training_data_window_stacked['direction_label'],
    random_state=42
)


val_data, test_data = train_test_split(
    temp_data,
    train_size=val_perc/(val_perc + test_perc),  # This calculates the proportion of validation set relative to the combined size of validation and test sets
    stratify=temp_data['direction_label'],
    random_state=42
)

def plotting_statistics( data, path):

    plt.figure(figsize=(10, 6)) 

    ax = sns.countplot(data=data, x='section', palette='Set1')
    ax.set_xlabel("Section", fontsize=12)
    ax.set_ylabel("Number of Instances", fontsize=12)
    ax.set_title("Number of Instances in Each Section", fontsize=14)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)

    plt.tight_layout()  
    plt.savefig(f'{path}/section_distribution.png')

    


    palette = sns.color_palette("Set1")
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(data[data.section == "section_1"].dx, data[data.section == "section_1"].dy, label="Section 1", s=2, c=palette[0])
    ax.scatter(data[data.section == "section_2"].dx, data[data.section == "section_2"].dy, label="Section 2", s=2, c=palette[1])
    ax.scatter(data[data.section == "section_3"].dx, data[data.section == "section_3"].dy, label="Section 3", s=2, c=palette[2])
    ax.scatter(data[data.section == "section_4"].dx, data[data.section == "section_4"].dy, label="Section 4", s=2, c=palette[3])
    ax.scatter(data[data.section == "section_5"].dx, data[data.section == "section_5"].dy, label="Section 5", s=2, c=palette[4])
    ax.scatter(data[data.section == "section_6"].dx, data[data.section == "section_6"].dy, label="Section 6", s=2, c=palette[5])
    ax.set_aspect('equal')
    ax.set_xlabel("dx")
    ax.set_ylabel("dy")
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Sections")
    ax.set_title("Normalised dx and dy")
    fig.tight_layout(rect=[0,0,0.85,1]) 

    plt.savefig(f'{path}/position_change.png')



    num_sectors = data.direction_label.nunique()
    direction_labels = data.direction_label.unique()
    direction_labels.sort()  # Sort the labels if needed

    fig, ax = plt.subplots()

    # Create the histogram
    counts, bins, patches = ax.hist(data.direction_label, bins=num_sectors, color='blue', alpha=0.7)

    # Set the x-axis labels
    ax.set_xticks(bins[:-1] + (bins[1] - bins[0])/2)
    ax.set_xticklabels(direction_labels, rotation=45, ha='right')  # Adjust the rotation and alignment as needed

    # Set the other labels and title
    ax.set_xlabel("Direction", fontsize=12)
    ax.set_ylabel("Number of Instances", fontsize=12)
    ax.set_title("Number of Instances in Each Direction", fontsize=14)

    plt.tight_layout()
    plt.savefig(f'{path}/direction_distribution.png')

parameters = {
    'window_size': window_size,
    'stride': stride,
    'num_stacked': num_stacked,
    'radius_velocity': radius_velocity,
    'num_sectors': num_sectors,
    'max_dx': max_dx,
    'max_dy': max_dy,
    'train_perc': train_perc,
    'val_perc': val_perc,
    'test_perc': test_perc, 
    "data_path": data_paths,
}



with open(f'{dataset_path}/parameters.txt', 'w') as f:
    for key, value in parameters.items():
        f.write(f'{key}: {value}\n')


plotting_statistics(train_data, train_path)
plotting_statistics(val_data, val_path)
plotting_statistics(test_data, test_path)


train_data.to_pickle(f'{train_path}/train.pkl')
val_data.to_pickle(f'{val_path}/val.pkl')
test_data.to_pickle(f'{test_path}/test.pkl')

