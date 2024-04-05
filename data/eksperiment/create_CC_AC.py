# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.fft import fft, ifft, ifftshift
from scipy import signal
import os

data_type_names = ["avstand_1_noise", "avstand_2_noise", "avstand_3_noise", "avstand_1", "avstand_2", "avstand_3"]
#data_type_names = ["avstand_1", "avstand_2", "avstand_3"]
#data_type_names = ["avstand_1"]
#data_type_names = ["lower_left", "lower_right", "upper_left", "upper_right", "middle"]
export_gcc_plot_path = r"data/eksperiment/eksperiment_2/"
data =                 r"data/eksperiment/eksperiment_2/data/"


w =    1000
step = w
factor = 80/w
beta =1
k0 = 0
k = 1000000
window_type = signal.windows.boxcar
corr_type = np.real
height = 40 
width = 40
threshold_bool = True
threshold_value = 50

folder_name = f"window_{w}_step_{step}_factor_{factor}_beta_{beta}_width_{width}_height_{height}"

if not os.path.exists(os.path.join(export_gcc_plot_path, folder_name)):
    os.makedirs(os.path.join(export_gcc_plot_path, folder_name))

def create_gcc(sensor_1, sensor_2, window_size, step_size, truncate_length, window_type, beta,correlation_type): 
    '''
    Inputs: 
    sensor_1: sensor 1 data
    sensor_2: sensor 2 data
    window_size: window size in samples
    step_size: step size in samples
    truncate_length: length of truncated output in samples
    window_type: window type. Example: signal.windows.boxcar
    beta: power of denominator
    correlation_type: real, imag or abs

    Outputs:
    GCC: cross correlation 
    '''
    GCC_12 = []
    for i in tqdm(range(0, len(sensor_1) - window_size, step_size)):
        j = i
        k = i + window_size

        s1 = sensor_1[j:k]
        s2 = sensor_2[j:k]
        
        window = window_type(len(s1))
        s1 = s1 * window
        s2 = s2 * window

        f_s1 = fft(s1)
        f_s2 = fft(s2)

        G12 = f_s1 * np.conj(f_s2)
        denom = abs(G12)
        denom[denom < 1e-6] = 1e-6
        f_s = G12 / denom**beta

        cc12 = ifft(f_s)

        cc12 = ifftshift(cc12)

        cc12 = correlation_type(cc12)

        index_0 = len(cc12)/2
        start_index = int(index_0 - truncate_length/2)
        end_index = int(index_0 + truncate_length/2)
        cc12 = cc12[start_index:end_index]

        cc12 = cc12 / np.max(np.abs(cc12))
        GCC_12.append(cc12)

    GCC_12 = np.array(GCC_12)
    return GCC_12
def truncate_time_shift(time_shift,length):
    index_0 = len(time_shift)/2
    start_index = int(index_0 - length/2)
    end_index = int(index_0 + length/2)
    time_shift = time_shift[start_index:end_index]
    return time_shift
def creating_plot(data_type_name): 
    sensor_data_file_path = data + data_type_name + ".npz"
    export_gcc_plot_path_curr = os.path.join(export_gcc_plot_path, folder_name, data_type_name + ".png")

    sensor_data = np.load(sensor_data_file_path)['data']
    columns = ["sensor_1", "sensor_2", "sensor_3", "t"]
    sensor_data = pd.DataFrame(sensor_data.T, columns=columns)



    sensor_data = sensor_data.iloc[k0:k, :]

    start_time = sensor_data.t.iloc[0]
    end_time =   sensor_data.t.iloc[-1]
    sensor_1 =   sensor_data.sensor_1.values
    sensor_2 =   sensor_data.sensor_2.values
    sensor_3 =   sensor_data.sensor_3.values

    GCC_12 = create_gcc(sensor_1, sensor_2,  int(w), step,w*factor, window_type= window_type, beta = beta, correlation_type = corr_type)
    GCC_13 = create_gcc(sensor_1, sensor_3,  int(w), step,w*factor, window_type= window_type, beta = beta, correlation_type = corr_type)
    GCC_23 = create_gcc(sensor_2, sensor_3,  int(w), step,w*factor, window_type= window_type, beta = beta, correlation_type = corr_type)

    GCC_11 = create_gcc(sensor_1, sensor_1,  int(w), step,w*factor, window_type= window_type, beta = beta, correlation_type = corr_type)
    GCC_22 = create_gcc(sensor_2, sensor_2,  int(w), step,w*factor, window_type= window_type, beta = beta, correlation_type = corr_type)
    GCC_33 = create_gcc(sensor_3, sensor_3,  int(w), step,w*factor, window_type= window_type, beta = beta, correlation_type = corr_type)

    dt = (end_time - start_time)/len(sensor_1)
    time = np.linspace( start_time+ w/ 2 * dt, end_time- w/2 * dt, len(GCC_12))


    fig,ax = plt.subplots(6,1, sharex = True, figsize=(width, height))
    time_shifts = np.arange(-w*dt,w*dt,dt)
    time_shifts = truncate_time_shift(time_shifts, w*factor)

    ax = ax.ravel()

    time = time - time[0]
    ax[0].pcolormesh( time,time_shifts*1000,  np.stack(GCC_12).T, vmin = -1, vmax = 1)
    ax[0].set_title("GCC 12")
    ax[0].set_xlabel("Time [s]")
    ax[0].set_ylabel("Time delay [ms]")

    ax[1].pcolormesh( time,time_shifts*1000,  np.stack(GCC_13).T, vmin = -1, vmax = 1)
    ax[1].set_title("GCC 13")
    ax[1].set_xlabel("Time [s]")
    ax[1].set_ylabel("Time delay [ms]")

    ax[2].pcolormesh( time,time_shifts*1000,  np.stack(GCC_23).T, vmin = -1, vmax = 1)
    ax[2].set_title("GCC 23" )
    ax[2].set_xlabel("Time [s]")
    ax[2].set_ylabel("Time delay [ms]")

    ax[3].pcolormesh( time,time_shifts*1000,  np.stack(GCC_11).T, vmin = -1, vmax = 1)
    ax[3].set_title("GCC 11")
    ax[3].set_xlabel("Time [s]")
    ax[3].set_ylabel("Time delay [ms]")

    ax[4].pcolormesh( time,time_shifts*1000,  np.stack(GCC_22).T, vmin = -1, vmax = 1)
    ax[4].set_title("GCC 22")
    ax[4].set_xlabel("Time [s]")
    ax[4].set_ylabel("Time delay [ms]")

    ax[5].pcolormesh( time,time_shifts*1000,  np.stack(GCC_33).T, vmin = -1, vmax = 1)
    ax[5].set_title("GCC 33")
    ax[5].set_xlabel("Time [s]")
    ax[5].set_ylabel("Time delay [ms]")

    fig.suptitle(f"The sensor data type is {data_type_name}. The window size is {w} samples, the step size is {step} samples, the factor is {factor}, the beta is {beta}.")

    plt.savefig(export_gcc_plot_path_curr)


for data_type_name in data_type_names:
    creating_plot(data_type_name)
    print(f"Plot for {data_type_name}.png is created.")
