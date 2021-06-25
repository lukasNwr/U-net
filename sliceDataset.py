# %%
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
import numpy as np
from skimage import color
import image_slicer
import os
import glob
import shutil
import imageio
import cv2
from skimage.util.dtype import img_as_ubyte

#%%
# Setup paths for all directories
src_dir = 'D:/Skola/SMaAD'
imgs_path = src_dir + '/CT-ORG Dataset/all_volumes/'
labels_path = src_dir + '/CT-ORG Dataset/labels and README/'
img_slices_path = src_dir + '/img_slices/'
label_slices_path = src_dir + '/label_slices/'
small_img_slices_path = img_slices_path +'small_img_slices/'
small_label_slices_path = label_slices_path +'small_label_slices/'

#%%
# Select volume and label to slice and convert
volume_to_slice = 'volume-0'
label_to_slice = 'labels-0'

img = nib.load(imgs_path + volume_to_slice + '.nii.gz').get_fdata()
label = nib.load(labels_path + label_to_slice + '.nii.gz').get_fdata()

# %%
# Print shapes and sizes, just for sanity check
print('Image shape: ')
print(img.shape)
print('Image volume len: ')
print(len(img[0][0]))
print('Label shape: ')
print(label.shape)
print('Label volume len')
print(len(label[0][0]))


# %%
# Slice the volume and volume label into multiple jpeg images
img_volume_size = len(img[0][0])

for i in range(img_volume_size):
       img_slice = img[:, :, i].astype(np.uint8)
    #    img_n = cv2.normalize(src=img_slice, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
       imsave(img_slices_path +
           f'/img_slice_{volume_to_slice}_{i+1}.tif', img_slice)
       imsave(label_slices_path +
           f'/label_slice_{label_to_slice}_{i+1}.tif', (label[:, :, i]).astype(np.uint8))

# %%
# Show slice, just for sanity check
read_img = imread(img_slices_path + '/img_slice_volume-0_20.tif')
read_label = imread(label_slices_path + '/label_slice_labels-0_20.tif')
f, ax = plt.subplots(1, 2)

ax[0].imshow(read_img)      # image
ax[1].imshow(read_label)    # label
plt.show()

# # %%
# # Copy jpeg volume images into separate folder for further slicing
# src_img_files = os.listdir(img_slices_path)
# for file_name in src_img_files:
#     full_file_name = os.path.join(img_slices_path, file_name)
#     if os.path.isfile(full_file_name):
#         shutil.copy(full_file_name, small_img_slices_path)

# # Copy jpeg label images
# src_label_files = os.listdir(label_slices_path)
# for file_name in src_label_files:
#     full_file_name = os.path.join(label_slices_path, file_name)
#     if os.path.isfile(full_file_name):
#         shutil.copy(full_file_name, small_label_slices_path)

# # %%
# # Slice images
# for directory_path in glob.glob(small_img_slices_path):
#     for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
#         image_slicer.slice(img_path, 4)
#         os.remove(img_path)

# # Slice labels
# for directory_path in glob.glob(small_label_slices_path):
#     for label_path in glob.glob(os.path.join(directory_path, "*.jpg")):
#         image_slicer.slice(label_path, 4)
#         os.remove(label_path)
