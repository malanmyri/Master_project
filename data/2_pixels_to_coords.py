import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import estimate_transform
import argparse

def calculate_affine_transform(p1, p2, p3, p4, q1, q2, q3, q4):
    """ Calculate the affine transformation matrix """
    src = np.array([p1, p2, p3, p4])
    dst = np.array([q1, q2, q3, q4])
    tform = estimate_transform('affine', src, dst)
    return tform.params

def apply_transformation(data, transform_matrix):
    """ Apply the affine transformation to the dataset """
    transformed_data = np.dot(np.c_[data, np.ones(data.shape[0])], transform_matrix.T)
    return transformed_data[:, :2]



def main(args):
    data_name = args.data_name
    plot_name = data_name[:-4] + "_pixel_to_coords.png"
    transformed_data_name = data_name[:-4] + "_transformed_data.csv"

    h = args.height

    # Adjusting y-coordinates
    args.y_pixel_1 = h - args.y_pixel_1
    args.y_pixel_2 = h - args.y_pixel_2
    args.y_pixel_3 = h - args.y_pixel_3
    args.y_pixel_4 = h - args.y_pixel_4

    transform_matrix = calculate_affine_transform(
        [args.x_pixel_1, args.y_pixel_1], [args.x_pixel_2, args.y_pixel_2], 
        [args.x_pixel_3, args.y_pixel_3], [args.x_pixel_4, args.y_pixel_4],
        [args.x_real_1, args.y_real_1], [args.x_real_2, args.y_real_2], 
        [args.x_real_3, args.y_real_3], [args.x_real_4, args.y_real_4]
    )
    data = np.loadtxt(data_name)
    data = data[:, [1, 0, 2]]
    transformed_data = apply_transformation(data[:, :2], transform_matrix)

    df = pd.DataFrame(transformed_data, columns=['x_transformed', 'y_transformed'])
    df['time'] = data[:, 2]

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(data[:, 0], data[:, 1], color='blue', label='Original Data')
    plt.title('Original Data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(0, 1280)
    plt.ylim(0, 720)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(df['x_transformed'], df['y_transformed'], color='green', label='Transformed Data')
    plt.title('Transformed Data')
    plt.xlabel('x_transformed')
    plt.ylabel('y_transformed')
    plt.xlim(0, 100)
    plt.ylim(0, 70)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_name)

    df.to_csv(transformed_data_name, index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate and apply affine transformation.')

    # Adding arguments
    parser.add_argument('--data_name', type=str, required=True, help='Path to data file')
    parser.add_argument('--height', type=int, required=True, help='Height of the video frame')

    # Arguments for pixel coordinates
    parser.add_argument('--x_pixel_1', type=int, required=True, help='x coordinate of the first pixel point')
    parser.add_argument('--y_pixel_1', type=int, required=True, help='y coordinate of the first pixel point')
    parser.add_argument('--x_pixel_2', type=int, required=True, help='x coordinate of the second pixel point')
    parser.add_argument('--y_pixel_2', type=int, required=True, help='y coordinate of the second pixel point')
    parser.add_argument('--x_pixel_3', type=int, required=True, help='x coordinate of the third pixel point')
    parser.add_argument('--y_pixel_3', type=int, required=True, help='y coordinate of the third pixel point')
    parser.add_argument('--x_pixel_4', type=int, required=True, help='x coordinate of the fourth pixel point')
    parser.add_argument('--y_pixel_4', type=int, required=True, help='y coordinate of the fourth pixel point')

    # Arguments for real-world coordinates
    parser.add_argument('--x_real_1', type=float, required=True, help='x coordinate of the first real-world point')
    parser.add_argument('--y_real_1', type=float, required=True, help='y coordinate of the first real-world point')
    parser.add_argument('--x_real_2', type=float, required=True, help='x coordinate of the second real-world point')
    parser.add_argument('--y_real_2', type=float, required=True, help='y coordinate of the second real-world point')
    parser.add_argument('--x_real_3', type=float, required=True, help='x coordinate of the third real-world point')
    parser.add_argument('--y_real_3', type=float, required=True, help='y coordinate of the third real-world point')
    parser.add_argument('--x_real_4', type=float, required=True, help='x coordinate of the fourth real-world point')
    parser.add_argument('--y_real_4', type=float, required=True, help='y coordinate of the fourth real-world point')

    args = parser.parse_args()
    main(args)