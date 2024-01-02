import os
import cv2 as cv
import torch
import numpy as np
import random

import albumentations as A  

random.seed(777)
np.random.seed(777)

class SegmentationDataset(object):
    def __init__(self, image_dir, mask_dir, resolution=256):
        self.resolution = resolution
        self.images = []
        self.masks = []
        files = os.listdir(image_dir)
        sfiles = os.listdir(mask_dir)
        for i in range(len(sfiles)):
            img_file = os.path.join(image_dir, files[i])
            mask_file = os.path.join(mask_dir, sfiles[i])
            self.images.append(img_file)
            self.masks.append(mask_file)

        # augmentation
        # self.transform = A.Compose([
        #             A.HorizontalFlip(p=0.3),
        #             A.VerticalFlip(p=0.3),
        #             A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        #             A.GaussNoise(p=0.2),
        #             # A.CLAHE(p=0.2), # Contrast Limited Adaptive Histogram Equalization
        #             A.GaussianBlur(p=0.2),
        #             # A.PiecewiseAffine(p=0.2, scale=(0.03, 0.04), nb_rows=(4, 4), nb_cols=(4, 4)),  # new feature colab not support...
        #             # A.ShiftScaleRotate(p=0.3, shift_limit=0.0625, scale_limit=0.1, rotate_limit=20, border_mode=cv.BORDER_CONSTANT, value=0, mask_value=0),
        #         ])
        # A.RandomCrop(256, 256, p=0.3), # 不能用，回导致负样本过大直接没有针了
        # A.CenterCrop(384, 384, p=0.3), # 不能用，回导致负样本过大直接没有针了
        self.transform = A.Compose([    # water
                    A.Resize(self.resolution, self.resolution),
                    A.HorizontalFlip(p=0.3),
                    A.VerticalFlip(p=0.3),
                    # A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                    A.GaussNoise(p=0.2),
                    A.GaussianBlur(p=0.2),
                ])
        # Normalize ???

    def __len__(self):
        return len(self.images)

    def num_of_samples(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            image_path = self.images[idx]
            mask_path = self.masks[idx]
        else:
            image_path = self.images[idx]
            mask_path = self.masks[idx]
            
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)  
        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)

        ########### resize ############# 
        # img = cv.resize(img, (256, 256))
        # mask = cv.resize(mask, (256, 256))
        ###############################
        
        # preprocessing 
        transformed = self.transform(image=img, mask=mask)  # 必须一起输入才能保证对img/mask统一的变换
        img = transformed["image"]
        mask = transformed["mask"]        
        
        # CV_32F: 32-bit floating point number (float) single channel
        img = cv.normalize(img, None, 0, 1.0, cv.NORM_MINMAX, dtype=cv.CV_32F)
        mask = cv.normalize(mask, None, 0, 1.0, cv.NORM_MINMAX, dtype=cv.CV_32F)

        # img = np.float32(img) / 255.0
        # mask = np.float32(mask) 

        img = np.expand_dims(img, 0) # add channel dim from W,H to C,W,H C=1
        mask = np.expand_dims(mask, 0) # add channel dim from W,H to C,W,H C=1

        data_pair = {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask), }

        return data_pair
