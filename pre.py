import cv2
import numpy as np
import glob
import re


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


numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


if __name__ == '__main__':

    color_dict = {0: (0, 0, 0),
                  1: (0, 128, 0),
                  2: (128, 128, 128),
                  3: (255, 255, 0),
                  4: (0, 255, 255),
                  5: (0, 255, 0),
                  6: (255, 0, 255),
                  7: (255, 0, 0),
                  8: (255, 255, 255)}

    filelist_trainx = []
    filelist_trainy = []
    # folder = ['gen', 'gen2', 'gen3', 'gen4', 'gen5', 'gen6', 'src']
    # for f in folder:
    #     if f is not 'src':
    #         filelist_trainx = filelist_trainx + sorted(glob.glob('E:/code/data/%s/*_img.png' % f), key=numericalSort)
    #         filelist_trainy = filelist_trainy + sorted(glob.glob('E:/code/data/%s/*_gt.png' % f), key=numericalSort)
    #     else:
    #         filelist_trainx = filelist_trainx + sorted(glob.glob('E:/code/data/src/*.tif'), key=numericalSort)
    #         filelist_trainy = filelist_trainy + sorted(glob.glob('E:/code/data/label/*.png'), key=numericalSort)
    filelist_trainx = sorted(glob.glob('/home/public/deng/aulbm/*/*.tif'), key=numericalSort)
    # # List of file names of classified images for traininig
    filelist_trainy = sorted(glob.glob('/home/public/deng/aulbm/*/*.png'), key=numericalSort)

    trainx_list = []
    for fname in filelist_trainx:
        image = cv2.imread(fname)
        trainx_list.append(image)
    trainx = np.asarray(trainx_list)

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
    np.save('trainx.npy', trainx)
    np.save('trainhot.npy', trainy_hot)