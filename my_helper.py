import os
import h5py
import numpy as np
import scipy.io
from torch import Tensor
import matplotlib.pyplot as plt

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


def mat_read_brain_files(file_paths):
    tensors = []
    for file_path in file_paths:
        mat_data = scipy.io.loadmat(file_path)
        data = mat_data['reconstruction_rss'][:14, :, :]
        print(data.shape)
        tensors.append(data)
    return np.stack(tensors, axis=0)


def to_plottable_format(data: Tensor, slice_index=0):
    """
        data: 5D tensor of shape (1, 2, nx, ny, nt)
    """
    if data.ndim != 5:
        raise ValueError(f"Expected a 5D tensor, but got {data.ndim}D tensor with shape {data.shape}")
    
    normalized_data = (data - np.min()) / (np.max() - np.min())

    return normalized_data[0, 0, :, :, slice_index].detach().cpu().numpy()


def plot_results(und_slice, rec_slice, gnd_slice, epoch_num, stage="train"):	
    im = abs(np.concatenate([und_slice, rec_slice, gnd_slice, gnd_slice - rec_slice], 1))
    plt.imsave(f'{stage}-output{epoch_num+1}.png', im, cmap='gray')