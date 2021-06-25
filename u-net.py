#%%
import os
os.environ["SM_FRAMEWORK"] = "tf.keras" #before the import

from unet_model import multiclass_unet_model

from tensorflow.keras.utils import normalize
import glob
import cv2
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.models import load_model
import tensorflow.keras as keras
import segmentation_models as sm




IMG_SIZE_X = 256
IMG_SIZE_Y = 256
n_classes = 6

src_dir = os.getcwd()
imgs_dir = src_dir + '/img_slices/'
labels_dir = src_dir + '/label_slices/'

#%%
# Load input images and labels
train_images = []

for directory_path in glob.glob(imgs_dir):
    for img_path in glob.glob(os.path.join(directory_path, "*.tif")):
        # img = imread(img_path, as_gray=False)    
        img = cv2.imread(img_path, 1)   
        img = resize(img, (IMG_SIZE_Y, IMG_SIZE_Y), anti_aliasing=True, order=1)
        train_images.append(img)
      
train_images = np.array(train_images)

train_labels = [] 
for directory_path in glob.glob(labels_dir):
    for mask_path in glob.glob(os.path.join(directory_path, "*.tif")):
        mask = imread(mask_path, as_gray=False)     
        mask = cv2.resize(mask, (IMG_SIZE_Y, IMG_SIZE_X), interpolation = cv2.INTER_NEAREST)  #Otherwise ground truth changes due to interpolation
        train_labels.append(mask)
        
train_labels = np.array(train_labels)

# %%
# Using label encoder for proper label encoding (even if labels should be in proper format)
labelencoder = LabelEncoder()
n, h, w = train_labels.shape
train_labels_reshaped = train_labels.reshape(-1,1)
train_labels_reshaped_encoded = labelencoder.fit_transform(train_labels_reshaped)
train_labels_encoded_original_shape = train_labels_reshaped_encoded.reshape(n, h, w)

np.unique(train_labels_encoded_original_shape)


# %%
# train_images = np.expand_dims(train_images, axis=3)       # if loading without rgb channels, need to expand array for channels dimension
# train_images = normalize(train_images, axis=1)        
train_labels_input = np.expand_dims(train_labels, axis=3)   # expading array with train_labels

# %%
# Splitting data for test and train
X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels_input, test_size = 0.2, random_state = 0)

train_labels_cat = to_categorical(y_train, num_classes=n_classes)
y_train_cat = train_labels_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))

test_masks_cat = to_categorical(y_test, num_classes=n_classes)
y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))

# %%
#Reused parameters in all models

n_classes=6
activation='softmax'

LR = 0.0001
optim = keras.optimizers.Adam(LR)

total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

# %%
# Model 1
BACKBONE1 = 'resnet34'
preprocess_input1 = sm.get_preprocessing(BACKBONE1)

# preprocess input
X_train1 = preprocess_input1(X_train)
X_test1 = preprocess_input1(X_test)

N = X_train1.shape[-1]
# define model (using defaul random weights)
model1 = sm.Unet(BACKBONE1, encoder_weights=None, classes=n_classes, activation=activation, input_shape=(None, None, N))

model1.compile(optim, total_loss, metrics=metrics)

print(model1.summary())

# %%
history1=model1.fit(X_train1, 
          y_train_cat,
          batch_size=8, 
          epochs=50,
          verbose=1,
          validation_data=(X_test1, y_test_cat))


model1.save('res34_backbone_50epochs.hdf5')

# %%
# Model 2

BACKBONE2 = 'inceptionv3'
preprocess_input2 = sm.get_preprocessing(BACKBONE2)

# preprocess input
X_train2 = preprocess_input2(X_train)
X_test2 = preprocess_input2(X_test)

# define model
model2 = sm.Unet(BACKBONE2, encoder_weights='imagenet', classes=n_classes, activation=activation)
model2.compile(optim, total_loss, metrics)

print(model2.summary())

# %%
history2=model2.fit(X_train2, 
          y_train_cat,
          batch_size=8, 
          epochs=50,
          verbose=1,
          validation_data=(X_test2, y_test_cat))


model2.save('inceptionv3_backbone_50epochs.hdf5')

# %%
# Model 3

BACKBONE3 = 'vgg16'
preprocess_input3 = sm.get_preprocessing(BACKBONE3)

# preprocess input
X_train3 = preprocess_input3(X_train)
X_test3 = preprocess_input3(X_test)

# define model
model3 = sm.Unet(BACKBONE3, encoder_weights='imagenet', classes=n_classes, activation=activation)
model3.compile(optim, total_loss, metrics)

print(model3.summary())

# %%
history3=model3.fit(X_train3, 
          y_train_cat,
          batch_size=8, 
          epochs=50,
          verbose=1,
          validation_data=(X_test3, y_test_cat))


model3.save('vgg16_backbone_50epochs.hdf5')

# %%
# Plots for loss and IOU for resnet64
loss = history1.history['loss']
val_loss = history1.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Trénovacia strata')
plt.plot(epochs, val_loss, 'r', label='Validačná strata')
plt.title('Trénovacia a validačná strata')
plt.xlabel('Epocha')
plt.ylabel('Strata')
plt.legend()
plt.savefig('plots/resnet34_50epochs_trainVallLoss.png')
plt.show()


acc = history1.history['iou_score']
val_acc = history1.history['val_iou_score']

plt.plot(epochs, acc, 'y', label='Trénovacie IOU')
plt.plot(epochs, val_acc, 'r', label='Validačné IOU')
plt.title('Trénovacie a validačné IOU')
plt.xlabel('Epocha')
plt.ylabel('IOU')
plt.legend()
plt.savefig('plots/resnet34_50_epochs_trainValIOU.png')
plt.show()


# %%
# Plots for loss and IOU for inceptionv3
loss = history2.history['loss']
val_loss = history2.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Trénovacia strata')
plt.plot(epochs, val_loss, 'r', label='Validačná strata')
plt.title('Trénovacia a validačná strata')
plt.xlabel('Epocha')
plt.ylabel('Strata')
plt.legend()
plt.savefig('plots/inceptionv3_50epochs_trainVallLoss.png')
plt.show()


acc = history2.history['iou_score']
val_acc = history2.history['val_iou_score']

plt.plot(epochs, acc, 'y', label='Trénovacie IOU')
plt.plot(epochs, val_acc, 'r', label='Validačné IOU')
plt.title('Trénovacie a validačné IOU')
plt.xlabel('Epocha')
plt.ylabel('IOU')
plt.legend()
plt.savefig('plots/inceptionv3_50_epochs_trainValIOU.png')
plt.show()


# %%
# # PLots for loss and IOU for vgg16
loss = history3.history['loss']
val_loss = history3.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Trénovacia strata')
plt.plot(epochs, val_loss, 'r', label='Validačná strata')
plt.title('Trénovacia a validačná strata')
plt.xlabel('Epocha')
plt.ylabel('Strata')
plt.legend()
plt.savefig('plots/vgg16_50epochs_trainVallLoss.png')
plt.show()


acc = history3.history['iou_score']
val_acc = history3.history['val_iou_score']

plt.plot(epochs, acc, 'y', label='Trénovacie IOU')
plt.plot(epochs, val_acc, 'r', label='Validačné IOU')
plt.title('Trénovacie a validačné IOU')
plt.xlabel('Epocha')
plt.ylabel('IOU')
plt.legend()
plt.savefig('plots/vgg16_50_epochs_trainValIOU.png')
plt.show()


# #####################################################

# %%
# Custom Model
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(train_labels_reshaped_encoded),
                                                 train_labels_reshaped_encoded)
print("Class weights are...:", class_weights)

# %%
IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH  = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]

def get_model():
    return multiclass_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)

model = get_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# %%
history = model.fit(X_train, y_train_cat, 
                    batch_size = 16, 
                    verbose=1, 
                    epochs=50, 
                    validation_data=(X_test, y_test_cat), 
                    class_weight=class_weights,
                    shuffle=False)
                    


model.save('multiclass_unet_model.hdf5')

# %%
# Plots for loss and IOU for Custom u-net model
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Trénovacie strata')
plt.plot(epochs, val_loss, 'r', label='Validačná strata')
plt.title('Trénovacia a validačná strata')
plt.xlabel('Epocha')
plt.ylabel('Strata')
plt.legend()
plt.savefig('plots/custom_model_50epochs_trainVallLoss.png')
plt.show()


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Trénovacia presnosť')
plt.plot(epochs, val_acc, 'r', label='Validačná presnosť')
plt.title('Trénovacia a validačná presnosť')
plt.xlabel('Epocha')
plt.ylabel('Presnosť')
plt.legend()
plt.savefig('plots/cusomModel_50epochs_trainValAcc.png')
plt.show()

# %%
#######################################################################

# Loading all models
model1 = load_model('res34_backbone_50epochs.hdf5', compile=False)
model2 = load_model('inceptionv3_backbone_50epochs.hdf5', compile=False)
model3 = load_model('vgg19_backbone_50epochs.hdf5', compile=False)
model4 = load_model('multiclass_unet_model.hdf5', compile=False)

#IOU
y_pred1=model3.predict(X_test3)
y_pred1_argmax=np.argmax(y_pred1, axis=3)


# Calculate mean IoU
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(y_test[:,:,:,0], y_pred1_argmax)
print("Mean IoU =", IOU_keras.result().numpy())


# Calculate IoU for each class
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
print(values)
class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[0,4] + values[0,5] + values[1,0] + values[2,0] + values[3,0] + values[4,0] + values[5,0])
class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[1,4] + values[1,5] + values[0,1] + values[2,1] + values[3,1] + values[4,1] + values[5,1])
class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[2,4] + values[2,5] + values[0,2] + values[1,2] + values[3,2] + values[4,2] + values[5,2])
class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[3,4] + values[3,5] + values[0,3] + values[1,3] + values[2,3] + values[4,3] + values[5,3])
class5_IoU = values[4,4]/(values[4,4] + values[4,0] + values[4,1] + values[4,2] + values[4,3] + values[4,5] + values[0,4] + values[1,4] + values[2,4] + values[3,4] + values[5,4]) 
class6_IoU = values[5,5]/(values[5,5] + values[5,0] + values[5,1] + values[5,2] + values[5,3] + values[5,4] + values[0,5] + values[1,5] + values[2,5] + values[3,5] + values[4,5]) 

print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)
print("IoU for class3 is: ", class3_IoU)
print("IoU for class4 is: ", class4_IoU)
print("IoU for class5 is: ", class5_IoU)
print("IoU for class6 is: ", class6_IoU)

# %% train#Test some random images
import random
test_img_number = random.randint(0, len(X_test3))
test_img = X_test3[test_img_number]
ground_truth = y_test[test_img_number]
test_img_input = np.expand_dims(test_img, 0)

# test_img_input1 = preprocess_input(test_img_input)
test_pred1 = model3.predict(test_img_input)
test_prediction1 = np.argmax(test_pred1, axis=3)[0,:,:]


plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='gray')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(test_prediction1, cmap='gray')
plt.savefig('prediction.png')
plt.show()

