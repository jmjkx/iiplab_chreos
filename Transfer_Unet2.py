import PIL
from PIL import Image
import matplotlib.pyplot as plt
import segmentation_models as sm
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
import tensorflow as tf
from segmentation_models.metrics import iou_score



#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    keras.tensorflow_backend.set_session(tf.Session(config=config))
    preprocess = sm.get_preprocessing('inceptionresnetv2')
    trainx = np.load('x_train.npy')
    trainy_hot = np.load('trainy_hot.npy')
    trainx = preprocess(trainx)

    m = load_model('/home/public/deng/deng/eye-in-the-sky/model_trresunet_inceptionres_0924.h5', custom_objects={'iou_score':sm.metrics.iou_score})

    intermediate_layer_model = Model(inputs=m.input,
                                     outputs=m.get_layer(index=815).output)
    x = Conv2D(  9 , 1, activation='softmax',name='2ddddd')(intermediate_layer_model.output)
    final_model = Model(inputs=intermediate_layer_model.input,
                                     outputs=x)

    model = final_model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy' , iou_score])
    model.summary()




    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, verbose=1, restore_best_weights=True),
        # EarlyStopping(monitor='val_loss', patience=2, verbose=0),
        # tf.keras.callbacks.ModelCheckpoint("model_augtrain.h5", monitor='loss', save_best_only=True, verbose=0),
        # tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    ]

    history = model.fit(trainx, trainy_hot, epochs=500, batch_size=16, verbose=1, callbacks=callbacks)
    model.save("DIY_transferUnet_inceptionv2_0925.h5")

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