# Source: https://github.com/pytorch/vision/blob/master/references/segmentation/transforms.py
import numpy as np
from PIL import Image
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
import math

class RandomHorizontalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, *args):
        # print('in flip', args)
        if random.random() < self.flip_prob:
            # print(image.size)
            image = F.hflip(image)
            # print(args)
            args = tuple([[o, image.width - x1, y0, image.width - x0, y1] for o, x0, y0, x1, y1 in boxes]
                         for boxes in args)
            # print(args)
        # print('out flip', args)
        return (image,) + args


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, *args):
        for t in self.transforms:
            image, *args = t(image, *args)
        return (image,) + tuple(args)


class Normalize(T.Normalize):
    def __call__(self, image, *args):
        return (super().__call__(image),) + args
    

class Resize(T.Resize):
    def __init__(self, new_size):
        super().__init__(new_size)
        self.new_size = new_size

    def __call__(self, image, *args):
        # print('in resize', args)
        iw = image.width
        ih = image.height
        # print('in size', iw, ih)
        nw = self.new_size[1]
        nh = self.new_size[0]
        # print(nw, nh)
        # print([boxes for boxes in args])
        args = tuple([[o, math.floor(x0 * nw / iw) , math.floor(y0 * nh / ih), math.floor(x1 * nw / iw) , math.floor(y1 * nh / ih)] for o, x0, y0, x1, y1 in boxes]
                         for boxes in args)
        # print('out resize', args)
        # print('out size', nh, nw)
        return (super().__call__(image),) + args


class ColorJitter(T.ColorJitter):
    def __call__(self, image, *args):
        return (super().__call__(image),) + args


class ToTensor(object):
    def __call__(self, image, *args):
        return (F.to_tensor(image),) + args


class ToHeatmap(object):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, image, objects_and_corners):
        heatmap = boxes_to_heatmap(objects_and_corners, self.shape)
        return image, heatmap


def boxes_to_heatmap(objects_and_corners, image_size):
    """Convert boundaries to a heatmap"""
    with torch.no_grad():
        # x = image_size[1]
        # y = image_size[0]
        adjustment = {0: 10, 1: 4}
        heatmap = torch.zeros((5, image_size[0], image_size[1]))
        # Loop through the 5 dimension classes of the heatmap
        for idx, row in enumerate(heatmap):
            # Check each object flag
            for object_and_corners in objects_and_corners:
                # print('in heatmap', objects_and_corners)
                # If the heatmap is of the proper type
                if object_and_corners[0] == idx:
                    height_start = min(object_and_corners[2], object_and_corners[4], image_size[0])
                    height_end = max(object_and_corners[2], object_and_corners[4], 0)
                    width_start = min(object_and_corners[1], object_and_corners[3], image_size[1])
                    width_end = max(object_and_corners[1], object_and_corners[3], 0)
                    # Label the points in the boundary
                    for idx2, row2 in enumerate(row):
                        if idx2 < height_start or idx2 > height_end:
                            continue
                        heatmap[idx][idx2][width_start : width_end] = 1
                else:
                    continue
            else:
                continue
        return heatmap


