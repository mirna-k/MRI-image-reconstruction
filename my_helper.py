import os
import h5py
import numpy as np

def get_file_paths(folder_path):

    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory.")
        return None

    file_paths = []

    for filename in os.listdir(folder_path):
        file_paths.append(os.path.join(folder_path, filename))
    return file_paths


def read_brain_files(file_paths):
    tensors = []
    for file_path in file_paths:
        data = h5py.File(file_path)['reconstruction_rss'][:14, :, :]
        print(data.shape)
        tensors.append(data)
    return np.stack(tensors, axis=0)
