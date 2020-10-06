import cv2
import numpy as np
import os
import torch
import segmentation_models_pytorch as smp
import argparse
import matplotlib.pyplot as plt
import glob
import re
from ipdb import set_trace
import builtins
import keras


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

def rgb(onehot, color_dict):
    h, w, c = onehot.shape


    for i in range(h):
        for j in range(w):
            argmax_index = np.argmax(onehot[i, j])

            sudo_onehot_arr = np.zeros((9))

            sudo_onehot_arr[argmax_index] = 1

            onehot_encode = sudo_onehot_arr

            onehot[i, j, :] = onehot_encode

    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros(onehot.shape[:2] + (3,))
    for k in color_dict.keys():
        output[single_layer == k] = color_dict[k]
    return np.uint8(output)






if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Demo of argparse")
    # parser.add_argument('inputpath', type=str, default=r'.\input_path', help='path to input')
    # parser.add_argument('outputpath', type=str, default=r'.\output_path', help='path to output')

    # args = vars(parser.parse_args())
    input_path = '/home/public/deng/deng/torch/input_path/'
    output_path = '/home/public/deng/deng/torch/output_path/'
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

    ENCODER = 'timm-efficientnet-b8'
    ENCODER_WEIGHTS = 'imagenet'



    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    # create segmentation model with pretrained encoder
    model1 = torch.load('./efficient_final.pt', map_location=lambda storage, loc:storage)
    model1.eval()

    model2 = keras.models.load_model('./model_trresunet.h5')

    for img_name in input_imgs:
        print(img_name)
        image = cv2.imread(img_name)

        with torch.no_grad():

            pre_input = preprocessing_fn(image)
            inputTensor = torch.from_numpy(np.transpose(pre_input, (2, 0, 1)).reshape(1, 3, 512, 512))
            input = inputTensor.type(torch.FloatTensor)
            output = model1(input.cpu())
            output_numpy = output.data.cpu().numpy()
            img1 = output_numpy.reshape(9,512,512)
            img1 = np.transpose(img1, (1,2,0))
            img2 = model2.predict(image.reshape(1,512,512,3))

            m1_pre = onehot_to_rgb(img1, color_dict)
            m2_pre = onehot_to_rgb(img2.reshape(512,512,9), color_dict)
            for i in range(512):
                for j in range(512):
                    if (m2_pre[i,j] == np.array([255,0,255])).all():
                        m1_pre[i,j] = np.array([255,0,255])
                    if (m2_pre[i, j] == np.array([0, 0, 0])).all():
                        m1_pre[i, j] = np.array([0, 0, 0])

            out_img = output_path + '/' + os.path.split(img_name)[1].split('.')[0] + '_gt.png'
            cv2.imwrite(out_img,  m1_pre)

        str = '<?xml version="1.0" encoding="utf-8"?>\n<annotation>\n <source>\n  <filename>\n   %s\n  ' \
              '</filename>\n  <origin>\n   GF2/GF3\n  </origin>\n </source>\n <research>\n  <version>\n   4.0\n  ' \
              '</version>\n  <provider>\n   重庆大学\n  </provider>\n  <author>\n 梦至星上 </author>\n  <pluginname>\n   地物标注\n  </pluginname>\n  ' \
              '<pluginclass>\n 标注 \n  </pluginclass>\n  <time>\n   2020-07-2020-11 \n  </time>\n </research>\n <segmentation>\n  <resultfile>\n   ' \
              '%s\n  </resultfile>\n </segmentation>\n</annotation>' % (
              os.path.split(img_name)[1], os.path.split(img_name)[1].split('.')[0] + '_gt.png')
        with open(output_path + '/' + os.path.split(img_name)[1].split('.')[0] + '.xml', 'w', encoding='utf-8') as f:
            f.write(str)





