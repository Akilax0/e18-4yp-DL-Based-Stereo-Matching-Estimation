import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets.data_io import get_transform, read_all_lines, pfm_imread
from . import flow_transforms
import torchvision
import cv2
import copy
import matplotlib.pyplot as plt
from glob import glob
import os.path as osp



class SceneFlowDatset(Dataset):
    def __init__(self, datapath, list_filename, training):
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        
        # Uncomment for updated data loading
        # self.left_filenames = []
        # self.right_filenames = []
        # self.disp_filenames = []

        self.training = training
        
        # Uncomment for updated data loading
        # self.dstype= 'frames_finalpass'
        # self.load_path2()

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        disp_images = [x[2] for x in splits]
        return left_images, right_images, disp_images

    def load_path2(self):
        if not self.training:
            self._add_things("TEST")
        else:
            self._add_things("TRAIN")
            self._add_monkaa("TRAIN")
            self._add_driving("TRAIN")

    def _add_things(self, split='TRAIN'):
        """ Add FlyingThings3D data """

        # root = osp.join(self.root, 'FlyingThings3D')
        root = self.datapath
        left_images = sorted( glob(osp.join(root,'FlyingThings3D', self.dstype, split, '*/*/left/*.png')) )
        right_images = [ im.replace('left', 'right') for im in left_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        # print("Looking at the dataset", root, left_images, right_images, disparity_images)

        # Choose a random subset of 400 images for validation
        state = np.random.get_state()
        np.random.seed(1000)
        # val_idxs = set(np.random.permutation(len(left_images))[:100])
        val_idxs = set(np.random.permutation(len(left_images)))
        np.random.set_state(state)

        for idx, (img1, img2, disp) in enumerate(zip(left_images, right_images, disparity_images)):
            if (split == 'TEST' and idx in val_idxs) or split == 'TRAIN':
                self.left_filenames += [ img1 ]
                self.right_filenames += [ img2 ]
                self.disp_filenames += [ disp ]

                # print("filename: ",self.left_filenames, self.right_filenames, self.disp_filenames)

    def _add_monkaa(self, split="TRAIN"):
        """ Add FlyingThings3D data """

        root = self.datapath
        left_images = sorted( glob(osp.join(root, 'Monkaa', self.dstype, split, '*/left/*.png')) )
        right_images = [ image_file.replace('left', 'right') for image_file in left_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        for img1, img2, disp in zip(left_images, right_images, disparity_images):
            self.left_filenames += [img1]
            self.right_filenames += [img2]
            self.disp_filenames += [disp]



    def _add_driving(self, split="TRAIN"):
        """ Add FlyingThings3D data """

        root = self.datapath
        left_images = sorted( glob(osp.join(root, 'Driving', self.dstype, split, '*/*/*/left/*.png')) )
        right_images = [ image_file.replace('left', 'right') for image_file in left_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        for img1, img2, disp in zip(left_images, right_images, disparity_images):
            self.left_filenames += [img1]
            self.right_filenames += [img2]
            self.disp_filenames += [disp]



    def load_image(self, filename):
        # print("filename ", filename)
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        data, scale = pfm_imread(filename)
        data = np.ascontiguousarray(data, dtype=np.float32)
        return data

    # def RGB2GRAY(self, img):
    #     imgG = copy.deepcopy(img)
    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #     imgG[:, :, 0] = img
    #     imgG[:, :, 1] = img
    #     imgG[:, :, 2] = img
    #     return imgG

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        
        # Comment this for moidified 
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))
        disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))

        # left_img = self.load_image(self.left_filenames[index])
        # right_img = self.load_image(self.right_filenames[index])
        # disparity = self.load_disp(self.disp_filenames[index])

        if self.training:

            th, tw = 256, 512
            random_brightness = np.random.uniform(0.5, 2.0, 2)
            random_gamma = np.random.uniform(0.8, 1.2, 2)
            random_contrast = np.random.uniform(0.8, 1.2, 2)
            random_saturation = np.random.uniform(0, 1.4, 2)

            left_img = torchvision.transforms.functional.adjust_brightness(left_img, random_brightness[0])
            right_img = torchvision.transforms.functional.adjust_brightness(right_img, random_brightness[1])

            left_img = torchvision.transforms.functional.adjust_gamma(left_img, random_gamma[0])
            right_img = torchvision.transforms.functional.adjust_gamma(right_img, random_gamma[1])

            left_img = torchvision.transforms.functional.adjust_contrast(left_img, random_contrast[0])
            right_img = torchvision.transforms.functional.adjust_contrast(right_img, random_contrast[1])

            left_img = torchvision.transforms.functional.adjust_saturation(left_img, random_saturation[0])
            right_img = torchvision.transforms.functional.adjust_saturation(right_img, random_saturation[1])

            right_img = np.array(right_img)
            left_img = np.array(left_img)

            # geometric unsymmetric-augmentation
            angle = 0;
            px = 0
            if np.random.binomial(1, 0.5):
                angle = 0.05
                px = 1
            co_transform = flow_transforms.Compose([
                flow_transforms.RandomCrop((th, tw)),
            ])
            augmented, disparity = co_transform([left_img, right_img], disparity)
            left_img = augmented[0]
            right_img = augmented[1]

            # randomly occlude a region
            right_img.flags.writeable = True
            if np.random.binomial(1,0.5):
              sx = int(np.random.uniform(35,100))
              sy = int(np.random.uniform(25,75))
              cx = int(np.random.uniform(sx,right_img.shape[0]-sx))
              cy = int(np.random.uniform(sy,right_img.shape[1]-sy))
              right_img[cx-sx:cx+sx,cy-sy:cy+sy] = np.mean(np.mean(right_img,0),0)[np.newaxis,np.newaxis]

            # w, h = left_img.size

            # Disparity at other resolutions by interpoolating 
            disparity = np.ascontiguousarray(disparity, dtype=np.float32)
            disparity_2 = cv2.resize(disparity, (tw//2, th//2), interpolation=cv2.INTER_NEAREST)
            disparity_4= cv2.resize(disparity, (tw // 4, th // 4), interpolation=cv2.INTER_NEAREST)

            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)



            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity,
                    "disparity_2": disparity_2,
                    "disparity_4": disparity_4}
        else:
            w, h = left_img.size
            crop_w, crop_h = 960, 512

            left_img = left_img.crop((w - crop_w, h - crop_h, w, h))
            right_img = right_img.crop((w - crop_w, h - crop_h, w, h))
            disparity = disparity[h - crop_h:h, w - crop_w: w]

            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity,
                    "top_pad": 0,
                    "right_pad": 0}
