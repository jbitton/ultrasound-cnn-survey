# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import torch
import torchvision
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = self.min_size
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        target = target.resize(image.size)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class RandomCrop(object):
    def __init__(self, min_size, max_size, crop_pct=0.1):
        self.min_size = min_size  # height - 420
        self.max_size = max_size  # width - 580
        self.crop_pct = crop_pct
    
    def __call__(self, image, target):
         if target.mode != "xyxy":
             raise Exception("NOT XYXY")
         
         max_x, min_x = target.bbox[:, ::2].max(), target.bbox[:, ::2].min()
         max_y, min_y = target.bbox[:, 1::2].max(), target.bbox[:, 1::2].max()
         i = 0
         left_to_crop = None
         while left_to_crop is None or left_to_crop >= min_x:
             left_to_crop = int(random.random() * self.crop_pct * self.max_size)
             i += 1
             if i == 3:
                 left_to_crop = 0
                 break
         i = 0
         top_to_crop = None
         while top_to_crop is None or top_to_crop >= min_y:
             top_to_crop = int(random.random() * self.crop_pct * self.min_size)
             i += 1
             if i == 3:
                 top_to_crop = 0
                 break
         i = 0
         right_to_crop = None
         while right_to_crop is None or self.max_size - right_to_crop <= max_y:
             right_to_crop = int(random.random() * self.crop_pct * self.max_size)
             i += 1
             if i == 3:
                 right_to_crop = 0
                 break
         i = 0
         bottom_to_crop = None
         while bottom_to_crop is None or self.min_size - bottom_to_crop <= max_x:
             bottom_to_crop = int(random.random() * self.crop_pct * self.min_size)
             i += 1
             if i == 3:
                 bottom_to_crop = 0
                 break
         right_crop = self.max_size - right_to_crop
         bottom_crop = self.min_size - bottom_to_crop
         image = np.array(image)
         image = image[top_to_crop:bottom_crop, left_to_crop:right_crop]
         target = target.crop((left_to_crop, top_to_crop, right_crop, bottom_crop))
         return Image.fromarray(image), target

