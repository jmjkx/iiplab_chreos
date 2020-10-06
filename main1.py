# -*- coding: UTF-8 -*-
import os
from PIL import Image
from ipdb import set_trace
from libtiff import TIFF
import numpy as np
import glob
import cv2
import argparse
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from mainunet import unet
import re


os.environ["CUDA_VISIBLE_DEVICES"] = ""

#%matplotlib inline

model = unet()

# To read the images in numerical order

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

    img = y_pred_train_img
    h, w, c = img.shape

    for i in range(h):
        for j in range(w):
            argmax_index = np.argmax(img[i, j])

            sudo_onehot_arr = np.zeros((9))

            sudo_onehot_arr[argmax_index] = 1

            onehot_encode = sudo_onehot_arr

            img[i, j, :] = onehot_encode

    y_pred_train_img = onehot_to_rgb(img, color_dict)

    tif = TIFF.open(filelist_trainx[i_])
    image2 = tif.read_image()

    h, w, c = image2.shape

    y_pred_train_img = y_pred_train_img[:h, :w, :]

    imx = Image.fromarray(y_pred_train_img)

    imx.save("train_predictions/pred" + str(i_ + 1) + ".jpg")


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


def testing(model, testx, weights_file="model_oneshot.h5"):
    pred_val_all = []
    model.load_weights(weights_file)
    Y_pred_val = model.predict(testx)
    for k in range(Y_pred_val.shape[0]):
        pred_val_all.append(Y_pred_val[k])
    #Y_gt_val = [rgb_to_onehot(arr, color_dict) for arr in testy]
    #return pred_val_all, Y_gt_val
    return pred_val_all


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo of argparse")
    parser.add_argument('inputpath', type=str, default=r'/input_path', help='path to input')
    parser.add_argument('outputpath', type=str, default=r'/output_path', help='path to output')

    args = vars(parser.parse_args())
    input_path = args['inputpath']
    output_path = args['outputpath']
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

    for img_name in input_imgs:
        testx_list = []
        tif = TIFF.open(img_name)
        image = tif.read_image()
        testx_list.append(image)
        testx = np.asarray(testx_list)



        Y_pred_val = testing(model, testx, weights_file=r"/workspace/model_augment.h5")

        img = Y_pred_val
        img = img[0]
        h, w, c = img.shape

        for i in range(h):
            for j in range(w):
                argmax_index = np.argmax(img[i, j])

                sudo_onehot_arr = np.zeros((9))

                sudo_onehot_arr[argmax_index] = 1

                onehot_encode = sudo_onehot_arr

                img[i, j, :] = onehot_encode

        y_pred_test_img = onehot_to_rgb(img, color_dict)


        out_img = output_path + '/' + os.path.split(img_name)[1].split('.')[0] + '_gt.png'
        cv2.imwrite(out_img, y_pred_test_img)
        #print('保存图片 {} 到 {}'.format(out_img, output_path))

        str = '<?xml version="1.0" encoding="utf-8"?>\n<annotation>\n <source>\n  <filename>\n   %s\n  ' \
        '</filename>\n  <origin>\n   GF2/GF3\n  </origin>\n </source>\n <research>\n  <version>\n   4.0\n  ' \
        '</version>\n  <provider>\n   重庆大学\n  </provider>\n  <author>\n 梦至星上 </author>\n  <pluginname>\n   地物标注\n  </pluginname>\n  ' \
        '<pluginclass>\n 标注 \n  </pluginclass>\n  <time>\n   2020-07-2020-11 \n  </time>\n </research>\n <segmentation>\n  <resultfile>\n   ' \
        '%s\n  </resultfile>\n </segmentation>\n</annotation>' % (os.path.split(img_name)[1], os.path.split(img_name)[1].split('.')[0] + '_gt.png')
        with open(output_path + '/' + os.path.split(img_name)[1].split('.')[0] + '.xml', 'w', encoding='utf-8') as f:
            f.write(str)




