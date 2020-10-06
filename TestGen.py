# coding: utf-8
# Team : IIPLAB Medical Group
# Author：Bro Yuan
# Date ：2020/10/3 上午1:39
# Tool ：


import numpy as np
import matplotlib.pyplot as plt
from pre import onehot_to_rgb

color_dict = {0: (0, 0, 0),
              1: (0, 128, 0),
              2: (128, 128, 128),
              3: (255, 255, 0),
              4: (0, 255, 255),
              5: (0, 255, 0),
              6: (255, 0, 255),
              7: (255, 0, 0),
              8: (255, 255, 255)}


if __name__ == '__main__':
    I = np.load('trainx.npy')
    L = np.load('trainhot.npy')
    plt.subplot(221)
    plt.imshow(I[3051])
    plt.subplot(222)
    plt.imshow(onehot_to_rgb(L[3051], color_dict))