from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import numpy as np


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
            images_npy,
            masks_npy,

            augmentation=None,
            preprocessing=None,
    ):


        self.images = images_npy
        self.masks = masks_npy
        self.length = images_npy.shape[0]
        # convert str names to class values on masks


        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = self.images[i]

        mask =  self.masks[i]


        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            image = self.preprocessing(image)


        return np.transpose(image, (2,0,1)),  np.transpose(mask, (2,0,1))

    def __len__(self):
        return self.length

