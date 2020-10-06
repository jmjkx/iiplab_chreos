from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import numpy as np
import os
import glob
import cv2
import re


color_dict = {0: (0, 0, 0),
              1: (0, 128, 0),
              2: (128, 128, 128),
              3: (255, 255, 0),
              4: (0, 255, 255),
              5: (0, 255, 0),
              6: (255, 0, 255),
              7: (255, 0, 0),
              8: (255, 255, 255)}

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


class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    def __init__(
            self,
            image_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):


        self.image_dir = image_dir

        self.length = 5500
        # convert str names to class values on masks


        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        dir = self.image_dir
        for f in os.listdir(dir):
            image_path = sorted(glob.glob(os.path.join(dir, f) + '/*.tif' ), key=numericalSort)
            mask_path =  sorted(glob.glob(os.path.join(dir, f) + '/*.png' ), key=numericalSort)
            for i in range(len(image_path)):
                image = cv2.imread(image_path[i])
                mask = cv2.imread(mask_path[i])
                # mask = rgb_to_onehot(mask, color_dict)
                # apply augmentations
                if self.augmentation:
                    sample = self.augmentation(image=image, mask=mask)
                    image, mask = sample['image'], sample['mask']

                # apply preprocessing
                if self.preprocessing:
                    image = self.preprocessing(image)

                return np.transpose(image, (2, 0, 1)), np.transpose(mask, (2, 0, 1))



    def __len__(self):
        return self.length

