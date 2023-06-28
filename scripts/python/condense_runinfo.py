import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import argparse
import os
import sys


from read_write_model import read_images_binary

def get_folder_info(folder_path):
    infos = folder_path.split("OUT")[-1].split("_")
    noise_value = infos[1]
    outlier_value = infos[2].strip("/")
    return np.float64(noise_value), np.float64(outlier_value)

def format_value_for_filename(numerical_value_):
    if numerical_value_ == 0:
        return "0.0"
    else:
        formated_value = "%.1f" % numerical_value_
        if formated_value.startswith("0."):
            return formated_value[1:]
        return formated_value

def read_time_values(time_path):
    time = np.zeros(2)
    if os.path.exists(time_path):
        val = np.loadtxt(time_path)
        if val.shape[0] == 2:
            time = val
    return time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read COLMAP outputs and condense it.")
    parser.add_argument("--global_path", help="Path to output")
    parser.add_argument("--std_val", help="Value of the inlier noise std", type=float)
    parser.add_argument("--std_index", help="Index of the inlier noise std", type=int)
    parser.add_argument("--outlier_val", help="Value of the outlier ratio", type=float)
    parser.add_argument("--outlier_index", help="index of the outlier ratio", type=int)
    parser.add_argument("--num_images", help="Number of images in the dataset", type=int)
    parser.add_argument("--num_trial", help="Number of runs per setup", type=int)
    args = parser.parse_args()

    print("XXXXXXXXXXXXXXXXXXXXXXXX Begining analysis... XXXXXXXXXXXXXXXXXXXXXXXX")

    global_path = args.global_path
    std_val = args.std_val
    i_std = args.std_index
    outlier_val = args.outlier_val
    i_outlier = args.outlier_index
    num_images = args.num_images
    num_trial = args.num_trial + 1

    output_gen_path = global_path + "OUT_{}_{}/"
    path_out = global_path + "{}_{}_{}.npy"
    image_nums = [1 + i for i in range(num_images)]
    algorithms = ["ransac", "acransac", "fastac", "lrt"]

    time_values = np.zeros((2, num_trial, len(algorithms)))
    precision_values = np.zeros((num_trial, len(algorithms)))
    recall_values = np.zeros((num_trial, len(algorithms)))
    images_seen_values = np.zeros((num_trial, len(algorithms)))
    images_posi_values = np.zeros((num_trial, len(algorithms), num_images, 8))

    std_val_filename = format_value_for_filename(std_val)
    outlier_val_filename = format_value_for_filename(outlier_val)
    folder_path = output_gen_path.format(std_val_filename, outlier_val_filename)
    for i_trial in range(num_trial):
        inlier_outlier_path = folder_path + "inlier_outlier_{}.txt".format(i_trial)
        if os.path.exists(inlier_outlier_path):
            inlier_outlier_data = np.loadtxt(inlier_outlier_path)
            for i_algo, algo_name in enumerate(algorithms):
                time_path = folder_path + algo_name + "_{}".format(i_trial) + "_time.txt"
                time = read_time_values(time_path)
                time_values[:, i_trial, i_algo] = time
                image_path = folder_path + algo_name + "_{}".format(i_trial) + "_images.bin"
                if os.path.exists(image_path):
                    images_result = read_images_binary(image_path)
                    num_true_positive = 0
                    num_estimated_positive = 0
                    num_GT_inliers = 0
                    num_images_seen = 0
                    for i_image, image_val in enumerate(image_nums):
                        if (image_val in images_result.keys()):
                            images_posi_values[i_trial, i_algo, i_image, 0] = image_val
                            images_posi_values[i_trial, i_algo, i_image, 1:5] = images_result[image_val].qvec
                            images_posi_values[i_trial, i_algo, i_image, 5:] = images_result[image_val].tvec

                            num_images_seen += 1
                            num_true_positive += np.sum(inlier_outlier_data[np.where(inlier_outlier_data[:, 0] == image_val)][images_result[image_val].point3D_ids > 0, 2])
                            num_estimated_positive += np.sum(images_result[image_val].point3D_ids > 0)
                            num_GT_inliers += np.sum(inlier_outlier_data[np.where(inlier_outlier_data[:, 0] == image_val)][:, 2])
                            if num_estimated_positive > 0:
                                precision = num_true_positive / num_estimated_positive
                            if num_GT_inliers > 0:
                                recall = num_true_positive / num_GT_inliers

                            precision_values[i_trial, i_algo] = precision
                            recall_values[i_trial, i_algo] = recall
                            images_seen_values[i_trial, i_algo] = num_images_seen

    np.save(path_out.format("precision_array", std_val_filename, outlier_val_filename), precision_values)
    np.save(path_out.format("recall_array", std_val_filename, outlier_val_filename), recall_values)
    np.save(path_out.format("image_seen_array", std_val_filename, outlier_val_filename), images_seen_values)
    np.save(path_out.format("time_array", std_val_filename, outlier_val_filename), time_values)
    np.save(path_out.format("image_position", std_val_filename, outlier_val_filename), images_posi_values)

    print("XXXXXXXXXXXXXXXXXXXXXXXX ... Analysis done and saved. XXXXXXXXXXXXXXXXXXXXXXXX")
