import cv2
import numpy as np
import os
import torch
import segmentation_models_pytorch as smp
import argparse

import glob
import re
from ipdb import set_trace


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo of argparse")
    parser.add_argument('--inputpath', type=str, default=r'/home/public/deng/deng/torch/input_path/', required=False,help='path to input')
    parser.add_argument('--outputpath', type=str, default=r'/home/public/deng/deng/torch/output_path/', required=False,help='path to output')

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

    ENCODER = 'resnet50'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = 'softmax2d'
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=9,
        activation=ACTIVATION,
    )


    model_dict = torch.load('./pt/BiSai_Unet/0.7738676703521608.pt',  map_location=lambda storage, loc:storage)
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    # create segmentation model with pretrained encoder
    model.load_state_dict(model_dict)
    


    model.eval()




    for img_name in input_imgs:
        print(img_name)
        image = cv2.imread(img_name)

        with torch.no_grad():

            input = preprocessing_fn(image)
            input = torch.from_numpy(np.transpose(input, (2, 0, 1)).reshape(1, 3, 512, 512))
            input = input.type(torch.FloatTensor)
            outputs = model(input.cpu())
            o = outputs.data.cpu().numpy()
            
            rgb_output = onehot_to_rgb(np.transpose(o.reshape(9, 512, 512), (1, 2, 0)), color_dict)
            out_img = output_path + '/' + os.path.split(img_name)[1].split('.')[0] + '_gt.png'
            cv2.imwrite(out_img, rgb_output)

        str = '<?xml version="1.0" encoding="utf-8"?>\n<annotation>\n <source>\n  <filename>\n   %s\n  ' \
              '</filename>\n  <origin>\n   GF2/GF3\n  </origin>\n </source>\n <research>\n  <version>\n   4.0\n  ' \
              '</version>\n  <provider>\n   重庆大学\n  </provider>\n  <author>\n 梦至星上 </author>\n  <pluginname>\n   地物标注\n  </pluginname>\n  ' \
              '<pluginclass>\n 标注 \n  </pluginclass>\n  <time>\n   2020-07-2020-11 \n  </time>\n </research>\n <segmentation>\n  <resultfile>\n   ' \
              '%s\n  </resultfile>\n </segmentation>\n</annotation>' % (
              os.path.split(img_name)[1], os.path.split(img_name)[1].split('.')[0] + '_gt.png')
        with open(output_path + '/' + os.path.split(img_name)[1].split('.')[0] + '.xml', 'w', encoding='utf-8') as f:
            f.write(str)

  




