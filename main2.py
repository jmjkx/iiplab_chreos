# -*- coding: UTF-8 -*-

import os
import numpy as np
import glob
import cv2
import argparse
from keras.models import load_model
import keras
import segementation_models as sm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import re
numbers = re.compile(r'(\d+)')

def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def onehot_to_rgb(onehot, color_dict):
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros(onehot.shape[:2] + (3,))
    for k in color_dict.keys():
        output[single_layer == k] = color_dict[k]
    return np.uint8(output)

def rgb_to_onehot(rgb_arr, color_dict):
    num_classes = len(color_dict)
    shape = rgb_arr.shape[:2] + (num_classes,)
    print(shape)
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


def testing(testx):
    pred_val_all = []
    keras.backend.clear_session()


    Y_pred_val = model.predict(testx)

    for k in range(Y_pred_val.shape[0]):
        pred_val_all.append(Y_pred_val[k])

    return pred_val_all


def rgb(img):
    h, w, c = img.shape

    for i in range(h):
        for j in range(w):
            argmax_index = np.argmax(img[i, j])

            sudo_onehot_arr = np.zeros((9))

            sudo_onehot_arr[argmax_index] = 1

            onehot_encode = sudo_onehot_arr

            img[i, j, :] = onehot_encode

    return onehot_to_rgb(img, color_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo of argparse")
    parser.add_argument('inputpath', type=str, default=r'G:\Programming\Python\cheos\input_path', help='path to input')
    parser.add_argument('outputpath', type=str, default=r'G:\Programming\Python\cheos\output', help='path to output')

    args = vars(parser.parse_args())
    input_path = args['inputpath']
    output_path = args['outputpath']
    # input_path = r'G:\Programming\Python\cheos\input_path'
    # output_path = r'G:\Programming\Python\cheos\output'
    input_imgs = sorted(glob.glob(input_path + '/*.tif'), key=numericalSort)

    color_dict = {0: (0, 0, 0),
                  1: (0, 128, 0),
                  2: (128, 128, 128),
                  3: (255, 255, 0),
                  4: (0, 255, 255),
                  5: (0, 255, 0),
                  6: (255, 0, 255),
                  7: (255, 0, 0),
                  8: (255, 255, 255)}
    y_pred_test_img = []
    model1 = load_model("model_trresunet.h5")
    model2 = load_model("model_treffunet_0924_b0_best.h5")
    for img_name in input_imgs:

        #tif = TIFF.open(img_name)
        image = cv2.imread(img_name)
        i = model1.predict(image.reshape(1,512,512,3))

        out_img = output_path + '/' + os.path.split(img_name)[1].split('.')[0] + '_gt.png'
        cv2.imwrite(out_img, rgb(i[0]))
        print(out_img)
        str = '<?xml version="1.0" encoding="utf-8"?>\n<annotation>\n <source>\n  <filename>\n   %s\n  ' \
        '</filename>\n  <origin>\n   GF2/GF3\n  </origin>\n </source>\n <research>\n  <version>\n   4.0\n  ' \
        '</version>\n  <provider>\n   重庆大学\n  </provider>\n  <author>\n 梦至星上 </author>\n  <pluginname>\n   地物标注\n  </pluginname>\n  ' \
        '<pluginclass>\n 标注 \n  </pluginclass>\n  <time>\n   2020-07-2020-11 \n  </time>\n </research>\n <segmentation>\n  <resultfile>\n   ' \
        '%s\n  </resultfile>\n </segmentation>\n</annotation>' % (os.path.split(img_name)[1], os.path.split(img_name)[1].split('.')[0] + '_gt.png')
        with open(output_path + '/' + os.path.split(img_name)[1].split('.')[0] + '.xml', 'w', encoding='utf-8') as f:
            f.write(str)




