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
import cv2

# %%
# Setup paths for all directories
src_dir = os.getcwd()
imgs_path = src_dir + '/CT-ORG Dataset/all_volumes/'
labels_path = src_dir + '/CT-ORG Dataset/labels and README/'
img_slices_path = src_dir + '/img_slices/'
label_slices_path = src_dir + '/label_slices/'
small_img_slices_path = img_slices_path + 'small_img_slices/'
small_label_slices_path = label_slices_path + 'small_label_slices/'

#%%
# Function for separating 3D volume into multiple 2D layers 
def slice_volume(volume, name, type):
       volume_size = len(volume[0][0])

       # Print shapes and sizes, just for info
       print(f'{name} shape: ')
       print(volume.shape)
       print(f'{name} volume len: ')
       print(volume_size)

       if type == 'img':
              for i in range(volume_size):
                     img_slice = volume[:, :, i]
                     imsave(img_slices_path +
                            f'/img_slice_{name}_{i+1}.tif', img_slice)
       elif type == 'label':
              for i in range(volume_size):
                     imsave(label_slices_path +
                            f'/label_slice_{name}_{i+1}.tif', (volume[:, :, i]).astype(np.uint8))
       else:
              print('Type of volume to slice not specified"')
       
       print(f'{name} processing is complete...')

# %%
# Separate picked .nii.gz 3D volumes and labels to .tiff images (layers)
volumes_to_slice = ['volume-90', 'volume-100', 'volume-110', 'volume-120', 'volume-130']
labels_to_slice = ['labels-90', 'labels-100', 'labels-110', 'labels-120', 'labels-130']


for idx in range(len(volumes_to_slice)):
       img = nib.load(imgs_path + volumes_to_slice[idx] + '.nii.gz').get_fdata()
       slice_volume(img, volumes_to_slice[idx], type='img')

for idx in range(len(labels_to_slice)):
       label = nib.load(labels_path + labels_to_slice[idx] + '.nii.gz').get_fdata()
       slice_volume(label, labels_to_slice[idx], type='label')


# Slice all volumes and labels in dataset (too many images, used picked volumes instead)
# for directory_path in glob.glob(imgs_path):
#     for img_path in glob.glob(os.path.join(directory_path, "*.nii.gz")):
#            img = nib.load(img_path).get_fdata()
#            name = (img_path.rsplit('\\', 1)[1]).split('.')[0]
#            print(name)
#            slice_volume(img, name)



# %%
##########################################################
# Show slice, just for check

read_img = imread(img_slices_path + '/img_slice_volume-1_90.tif')
read_label = imread(label_slices_path + '/label_slice_labels-_90.tif')
f, ax = plt.subplots(1, 2)

ax[0].imshow(read_img)  # image
ax[1].imshow(read_label)  # label
plt.show()


#######################################
# Slicing 2D images into smaller cuts #
#######################################

# Theese convert 64bit tif images into png, which creates problems with labels
# so it is not used and resising of images is used instead 

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
