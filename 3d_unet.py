#%%
import tensorflow as tf
import tensorflow.keras as keras

from skimage import io
from patchify import patchify, unpatchify
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import nibabel as nib
from tensorflow.keras.utils import get_file
import segmentation_models_3D as sm

# number of classes to be trained
n_classes = 7

# Loading images from dataset (this is used for checking if images are loaded and processed correctly)
image = nib.load('../CT-ORG Dataset/all_volumes/volume-0.nii.gz')
# Getting imaage data from nib format image
image_data = image.get_fdata()
# Using pathify for spliting images to smaller pieces, because base images are 512x512px big.
img_patches = patchify(image_data, (64,64,64), step=64)

label = nib.load('../CT-ORG Dataset/labels and README/labels-0.nii.gz')
label_data = label.get_fdata()
label_patches = patchify(label_data, (64,64,64), step=64)

# Check if labeled part of image is same part as unlabeled 
# plt.imshow(img_patches[1,2,0,:,:,50])
# plt.imshow(label_patches[1,2,0,:,:,50])

# Reshaping images so that they are correctly vectorized
input_img = np.reshape(img_patches, (-1, img_patches.shape[3], img_patches.shape[4], img_patches.shape[5]))
input_mask = np.reshape(label_patches, (-1, label_patches.shape[3], label_patches.shape[4], label_patches.shape[5]))

#print(input_img.shape)  # n_patches, x, y, z

#Convert grey image to 3 channels by copying channel 3 times. 

train_img = np.stack((input_img,)*3, axis=-1)
train_mask = np.expand_dims(input_mask, axis=4)

train_mask_cat = to_categorical(train_mask, num_classes=n_classes)

X_train, X_test, y_train, y_test = train_test_split(train_img, train_mask_cat, test_size = 0.10, random_state = 0)

LR = 0.0001
optim = keras.optimizers.Adam(LR)

dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.25, 0.25, 0.25, 0.25])) 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

# Model parameters

encoder_weights = 'imagenet'
BACKBONE = 'vgg16'  #Try vgg16, efficientnetb7, inceptionv3, resnet50
activation = 'softmax'
patch_size = 64
n_classes = 4
channels=3

LR = 0.0001
optim = keras.optimizers.Adam(LR)


dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.25, 0.25, 0.25, 0.25])) 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)


metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

preprocess_input = sm.get_preprocessing(BACKBONE)

#Preprocessing input
X_train_prep = preprocess_input(X_train)
X_test_prep = preprocess_input(X_test)

# Defining model
model = sm.Unet(BACKBONE, classes=n_classes, 
                input_shape=(patch_size, patch_size, patch_size, channels), 
                encoder_weights=encoder_weights,
                activation=activation)

model.compile(optimizer = optim, loss=total_loss, metrics=metrics)
print(model.summary())

#Fitting the model
history=model.fit(X_train_prep, 
          y_train,
          batch_size=8, 
          epochs=100,
          verbose=1,
          validation_data=(X_test_prep, y_test))
