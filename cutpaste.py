import numpy as np
import os
import ast
import argparse
import matplotlib.pyplot as plt
import torch
from torchvision import transforms as T

from PIL import Image
import albumentations as A
import cv2



class CutPaste():
    def __init__(self, train_normal_files,class_name):
        self.train_normal_files = train_normal_files
        self.class_name = class_name
        self.augmentors = [A.RandomRotate90(),
                A.Flip(),
                A.Transpose(),
                A.OpticalDistortion(p=1.0, distort_limit=1.0),
                A.GaussNoise(),
                A.OneOf([
                    A.MotionBlur(p=.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),
                ], p=0.2),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
                A.OneOf([
                    A.OpticalDistortion(p=0.3),
                    A.GridDistortion(p=.1),
                    A.PiecewiseAffine(p=0.3),
                ], p=0.2),
                A.OneOf([
                    A.CLAHE(clip_limit=2),
                    A.Sharpen(),
                    A.Emboss(),
                    A.RandomBrightnessContrast(),            
                ], p=0.3),
                A.HueSaturationValue(p=0.3)]
    def randAugmentor(self):
        aug_ind = np.random.choice(np.arange(len(self.augmentors)),3,replace=False)
        aug = A.Compose([self.augmentors[aug_ind[0]],
                        self.augmentors[aug_ind[1]],
                        self.augmentors[aug_ind[2]]])
        return aug
    def copy_paste(self,img, mask):
        n_idx = np.random.randint(len(self.train_normal_files))
        aug = self.randAugmentor()
        image = cv2.imread(img) # anomaly sample
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        n_image = cv2.imread(self.train_normal_files[n_idx])
        n_image = cv2.cvtColor(n_image, cv2.COLOR_BGR2RGB)
        img_height, img_width = n_image.shape[0], n_image.shape[1]
        
        mask = Image.open(mask)
        mask = np.asarray(mask)
        
        augmented = aug(image=image,mask=mask)
        aug_image, aug_mask = augmented['image'], augmented['mask']
        n_img_path = self.train_normal_files[n_idx]
        img_file = os.path.basename(n_img_path)
        if self.class_name in ['01','02','03']:
            if self.class_name == '03':
                image = image[:,100:700,:]
                n_image = n_image[:,100:700,:]
                mask = mask[:,100:700]
            augmentated = aug(image=image,mask=mask)
            aug_image, aug_mask = augmentated['image'], augmentated['mask']
            n_image[aug_mask == 255,:] = aug_image[aug_mask==255,:]
            return n_image, aug_mask



        fg_path = os.path.join('fg_mask',self.class_name,img_file)
        fg_mask = Image.open(fg_path)
        fg_mask = np.asarray(fg_mask)
        intersect_mask = np.logical_and(fg_mask==255, aug_mask == 255)
        if (np.sum(intersect_mask) > int(2/3*np.sum(aug_mask==255))):
            n_image[intersect_mask == 1,:] = aug_image[intersect_mask == 1,:]
            return n_image, intersect_mask
        else:
            contours, _ = cv2.findContours(aug_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            center_xs, center_ys = [], []
            widths, heights = [], []
            for i in range(len(contours)):
                M = cv2.moments(contours[i])
                if M['m00'] == 0:
                    x_min, x_max = np.min(contours[i][:,:,0]), np.max(contours[i][:,:,0])
                    y_min, y_max = np.min(contours[i][:,:,1]), np.max(contours[i][:,:,1])
                    center_x = int((x_min+x_max)/2)
                    center_y = int((y_min+y_max)/2)
                else:
                    center_x = int(M['m10']/M['m00'])
                    center_y = int(M['m01']/M['m00'])
                center_xs.append(center_x)
                center_ys.append(center_y)
                x_min, x_max = np.min(contours[i][:, :, 0]), np.max(contours[i][:, :, 0])
                y_min, y_max = np.min(contours[i][:, :, 1]), np.max(contours[i][:, :, 1])
                width, height = x_max - x_min, y_max - y_min
                widths.append(width)
                heights.append(height)
            if len(widths) == 0 or len(heights) == 0:
                n_image[aug_mask == 255,:] = aug_image[aug_mask == 255,:]
                return n_image, aug_mask
            else:
                max_width, max_height = np.max(widths), np.max(heights)
                center_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                center_mask[int(max_height/2):img_height-int(max_height/2), int(max_width/2):img_width-int(max_width/2)] = 255
                fg_mask = np.logical_and(fg_mask == 255, center_mask == 255)

                x_coord = np.arange(0, img_width)
                y_coord = np.arange(0, img_height)
                xx, yy = np.meshgrid(x_coord, y_coord)
                # coordinates of fg region points
                xx_fg = xx[fg_mask]
                yy_fg = yy[fg_mask]
                xx_yy_fg = np.stack([xx_fg, yy_fg], axis=-1)  # (N, 2) # to get the coordinates of the fg region points
                        
                if xx_yy_fg.shape[0] == 0:  # no fg
                    n_image[aug_mask == 255, :] = aug_image[aug_mask == 255, :]
                    return n_image, aug_mask
                aug_mask_shifted = np.zeros((img_height, img_width), dtype=np.uint8)
                for i in range(len(contours)):
                    aug_mask_shifted_i = np.zeros((img_height, img_width), dtype=np.uint8)
                    new_aug_mask_i = np.zeros((img_height, img_width), dtype=np.uint8)
                    # random generate a point in the fg region
                    idx = np.random.choice(np.arange(xx_yy_fg.shape[0]), 1)
                    rand_xy = xx_yy_fg[idx]
                    # distance between the point we wanna copy and paste
                    delta_x, delta_y = center_xs[i] - rand_xy[0, 0], center_ys[i] - rand_xy[0, 1]
                            
                    x_min, x_max = np.min(contours[i][:, :, 0]), np.max(contours[i][:, :, 0])
                    y_min, y_max = np.min(contours[i][:, :, 1]), np.max(contours[i][:, :, 1])
                            
                    # mask for one anomaly region on the augmented anomaly
                    aug_mask_i = np.zeros((img_height, img_width), dtype=np.uint8)
                    aug_mask_i[y_min:y_max, x_min:x_max] = 255
                    aug_mask_i = np.logical_and(aug_mask == 255, aug_mask_i == 255)
                            
                    # coordinates of orginal mask points
                    xx_ano, yy_ano = xx[aug_mask_i], yy[aug_mask_i]
                            
                    # shift the original mask of augmented into fg region
                    xx_ano_shifted = xx_ano - delta_x
                    yy_ano_shifted = yy_ano - delta_y
                    outer_points_x = np.logical_or(xx_ano_shifted < 0, xx_ano_shifted >= img_width) 
                    outer_points_y = np.logical_or(yy_ano_shifted < 0, yy_ano_shifted >= img_height)
                    outer_points = np.logical_or(outer_points_x, outer_points_y)
                            
                    # the mask of the paste region should be changed to 1
                    xx_ano_shifted = xx_ano_shifted[~outer_points]
                    yy_ano_shifted = yy_ano_shifted[~outer_points]
                    aug_mask_shifted_i[yy_ano_shifted, xx_ano_shifted] = 255
                            
                    # for the outside points, we should change it to 0 on the mask of augmented
                    xx_ano = xx_ano[~outer_points]
                    yy_ano = yy_ano[~outer_points]
                    new_aug_mask_i[yy_ano, xx_ano] = 255
                    # copy the augmentated anomaly area to the normal image
                    n_image[aug_mask_shifted_i == 255, :] = aug_image[new_aug_mask_i == 255, :]
                    aug_mask_shifted[aug_mask_shifted_i == 255] = 255
                    return n_image, aug_mask_shifted
