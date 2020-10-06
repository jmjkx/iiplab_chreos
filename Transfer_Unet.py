import PIL
from PIL import Image
import matplotlib.pyplot as plt
from libtiff import TIFF
from libtiff import TIFFfile, TIFFimage
import numpy as np
import math
import glob
import cv2
import os
import re
import skimage.io as io
import skimage.transform as trans
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as keras

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.tensorflow_backend.set_session(tf.Session(config=config))

numbers = re.compile(r'(\d+)')

def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def rgb_to_onehot(rgb_arr, color_dict):
    num_classes = len(color_dict)
    shape = rgb_arr.shape[:2] + (num_classes,)
    arr = np.zeros(shape, dtype=np.int8)
    for i, cls in enumerate(color_dict):
        arr[:, :, i] = np.all(rgb_arr.reshape((-1, 3)) == color_dict[i], axis=1).reshape(shape[:2])
    return arr

def onehot_to_rgb(onehot, color_dict):
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros(onehot.shape[:2] + (3,))
    for k in color_dict.keys():
        output[single_layer == k] = color_dict[k]
    return np.uint8(output)

def unet():
    # Left side of the U-Net
    inputs = Input((None, None, 3))
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='random_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='random_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='random_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='random_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='random_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='random_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='random_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='random_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    # Bottom of the U-Net
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='random_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='random_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    drop5 = Dropout(0.5)(conv5)

    # Upsampling Starts, right side of the U-Net
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='random_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='random_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='random_normal')(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='random_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='random_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='random_normal')(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='random_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='random_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='random_normal')(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='random_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='random_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='random_normal')(conv9)
    conv9 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='random_normal')(conv9)
    conv9 = BatchNormalization()(conv9)

    # Output layer of the U-Net with a softmax activation
    conv10 = Conv2D(17, 1, activation='softmax')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    return model


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    keras.tensorflow_backend.set_session(tf.Session(config=config))
    model = unet()
    color_dict = {0: (0, 250, 150),
                  1: (0, 0, 200),
                  2: (200, 0, 0),
                  3: (250, 200, 0),
                  4: (150, 150, 200),
                  5: (150, 150, 250),
                  6: (150, 200, 150),
                  7: (200, 0, 200),
                  8: (250, 0, 150),
                  9: (250, 150, 150),
                  10: (0, 200, 0),
                  11: (200, 150, 0),
                  12: (150, 0, 250),
                  13: (0, 200, 250),
                  14: (0, 200, 200),
                  15: (0, 0, 0),
                  }

    filelist_trainx = []
    filelist_trainy = []
    # List of file names of actual Satellite images for traininig
    filelist_trainx = sorted(glob.glob('E:\code\GID\src/*.tif'), key=numericalSort)
    # # List of file names of classified images for traininig
    filelist_trainy = sorted(glob.glob('E:\code\GID\gt/*.tif'), key=numericalSort)
    # folder = ['gen', 'gen2', 'gen3', 'gen4', 'gen5', 'gen6', 'src']
    # # List of file names of source
    # for f in folder:
    #     if f is not 'src':
    #         filelist_trainx = filelist_trainx + sorted(glob.glob('E:/code/data/%s/*_img.png'%f), key=numericalSort)
    #         # List of file names of ground truth
    #         filelist_trainy = filelist_trainy + sorted(glob.glob('E:/code/data/%s/*_gt.png'%f), key=numericalSort)
    #     else:
    #         filelist_trainx = filelist_trainx + sorted(glob.glob('E:/code/data/src/*.tif'), key=numericalSort)
    #         # List of file names of ground truth
    #         filelist_trainy = filelist_trainy + sorted(glob.glob('E:/code/data/label/*.png'), key=numericalSort)

    trainx_list = []
    for fname in filelist_trainx:
        image = cv2.imread(fname)
        trainx_list.append(image)
    trainx = np.asarray(trainx_list, dtype=object)

    trainy_list = []
    for fname in filelist_trainy:
        image = cv2.imread(fname)
        trainy_list.append(image)
    trainy = np.asarray(trainy_list)

    trainy_hot = []
    for i in range(trainy.shape[0]):
        hot_img = rgb_to_onehot(trainy[i], color_dict)
        trainy_hot.append(hot_img)
    trainy_hot = np.asarray(trainy_hot)


    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, verbose=1, restore_best_weights=True),
        # EarlyStopping(monitor='val_loss', patience=2, verbose=0),
        # tf.keras.callbacks.ModelCheckpoint("model_augtrain.h5", monitor='loss', save_best_only=True, verbose=0),
        #tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    ]

    history = model.fit(trainx, trainy_hot, epochs=500, batch_size=4, verbose=1, callbacks=callbacks)
    model.save("GID.h5")

    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig('acc_plot.png')
    plt.show()
    plt.close()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train'], loc='upper right')
    plt.savefig('loss_plot.png')
    plt.show()
    plt.close()