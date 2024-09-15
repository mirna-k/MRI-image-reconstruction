import numpy as np
from scipy.ndimage import rotate

def scale_data(data, scale_range=(0.9, 1.1)):
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    return data * scale_factor

def shift_data(data, max_shift=2):
    shifts = np.random.randint(-max_shift, max_shift, size=data.ndim)
    return np.roll(data, shifts, axis=tuple(range(data.ndim)))

def flip_data(data):
    flip_axes = np.random.choice([True, False], size=data.ndim)
    for i, flip in enumerate(flip_axes):
        if flip:
            data = np.flip(data, axis=i)
    return data

def rotate_data(data, angle_range=(-10, 10)):
    angle = np.random.uniform(angle_range[0], angle_range[1])
    return rotate(data, angle, axes=(1, 2), reshape=False)  

def augment_data(data):
    if np.random.rand() < 0.5:
        data = scale_data(data)
    if np.random.rand() < 0.5:
        data = shift_data(data)
    if np.random.rand() < 0.5:
        data = flip_data(data)
    if np.random.rand() < 0.5:
        data = rotate_data(data)
    return data

