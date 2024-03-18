from matplotlib import pyplot as plt
from my_helper import read_files_in_folder
import h5py
import numpy as np
from FASTmri_helper import *




train = h5_file = h5py.File('brain_data/train/file_brain_AXFLAIR_200_6002527.h5')

print('Keys: ', list(train.keys()))
print('Attr: ', dict(train.attrs))

def get_image(kspace_img):
    rss_images = []
    volume_kspace = kspace_img['kspace']
    i = 0
    for slice_kspace in volume_kspace:
        slice_image = ifft2(to_tensor(slice_kspace))
        slice_image_abs = complex_abs(slice_image)
        slice_image_rss = rss(slice_image_abs, dim=0)
        # plt.imshow(np.abs(slice_image_rss.numpy()), cmap='gray')
        # plt.title(f"{i}")
        # i += 1
        # plt.show()

        rss_images.append(np.abs(slice_image_rss.numpy()))

    return rss_images
