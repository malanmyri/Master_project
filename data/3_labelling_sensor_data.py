import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import argparse  
def convert_seconds(seconds):
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    milliseconds = (seconds - int(seconds)) * 1000
    return "%d:%02d:%02d.%03d" % (hours, minutes, int(seconds), milliseconds)

def main(args):
    sigma = args.sigma
    finger_position_csv = args.finger_position_csv
    sensor_data_npz = args.sensor_data_npz
    export_path = args.export_path
    export_path_fig = args.export_path_fig

    # Finger position data
    finger_position = pd.read_csv(finger_position_csv)
    finger_position["x"] = finger_position.x_transformed
    finger_position["y"] = finger_position.y_transformed
    finger_position["t"] = finger_position.time
    finger_position.drop(["x_transformed", "y_transformed", "time"], axis=1, inplace=True)


    sensor_data = np.load(sensor_data_npz)['data']
    columns = ["sensor_1", "sensor_2", "sensor_3", "t"]
    sensor_data = pd.DataFrame(sensor_data.T, columns=columns)

    start_time = sensor_data.t[0]
    end_time = sensor_data.t.iloc[-1]

    fig, ax = plt.subplots(3,2, figsize=(10, 10))

    ax = ax.ravel()

    k = 3000
    ax[0].scatter(finger_position["x"][:k], finger_position["y"][:k], s=1)
    ax[0].set_title("Finger position")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")

    ax[1].scatter(finger_position["x"].diff()[:k], finger_position["y"].diff()[:k], s=1)
    ax[1].set_title("Finger velocity")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")


    ax[2].plot(finger_position["t"][:k], label = f"start time = {convert_seconds(finger_position['t'][0])}, end time = {convert_seconds(finger_position['t'].iloc[-1])}")
    ax[2].set_title("comparing")
    ax[2].set_ylabel("time")
    ax[2].legend()

    ax[2].plot(sensor_data["t"][:k], label = f"start time = {convert_seconds(sensor_data['t'][0])}, end time = {convert_seconds(sensor_data['t'].iloc[-1])}")


    ax[3].plot(sensor_data["sensor_1"][:k], label = "sensor_1")
    ax[4].plot(sensor_data["sensor_2"][:k], label = "sensor_2")
    ax[5].plot(sensor_data["sensor_3"][:k], label = "sensor_3")

    ax[3].set_title("Sensor 1")
    ax[4].set_title("Sensor 2")
    ax[5].set_title("Sensor 3")

    fig.tight_layout()
    fig.savefig(export_path_fig)

    finger_position["x" ] = gaussian_filter1d(finger_position["x"], sigma)
    finger_position["y" ] = gaussian_filter1d(finger_position["y"], sigma)

    finger_position.reset_index(inplace=True, drop=True)
    sensor_data.reset_index(inplace=True, drop=True)

    start_time = max(finger_position.t[0], sensor_data.t[0])
    end_time = min(finger_position.t.iloc[-1], sensor_data.t.iloc[-1])
    finger_position = finger_position[(finger_position.t >= start_time) & (finger_position.t <= end_time)]
    sensor_data = sensor_data[(sensor_data.t >= start_time) & (sensor_data.t <= end_time)]

    dt = sensor_data.t.diff().mean()
    result = pd.merge_asof(sensor_data,
                            finger_position,
                            left_on='t',
                            right_on='t',
                            direction='nearest', 
                            tolerance = dt )

    result["x"].interpolate(method='linear',limit_direction = "both",  inplace=True)
    result["y"].interpolate(method='linear',limit_direction = "both",  inplace=True)


    result.to_pickle(export_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and analyze sensor and finger position data.')
    parser.add_argument('--sigma', type=float, default=4, help='Sigma value for Gaussian filter')
    parser.add_argument('--finger_position_csv', type=str, required=True, help='Path to finger position CSV file')
    parser.add_argument('--sensor_data_npz', type=str, required=True, help='Path to sensor data NPZ file')
    parser.add_argument('--export_path', type=str, required=True, help='Path to export processed data')
    parser.add_argument('--export_path_fig', type=str, required=True, help='Path to export summary figure')

    args = parser.parse_args()
    main(args)