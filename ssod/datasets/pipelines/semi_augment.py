# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
# Modified from https://github.com/open-mmlab/mmdetection
"""
Augmentation in SSOD
"""
import math
import random
import warnings
import numpy as np
import numbers

from mmdet.datasets.pipelines import Albu
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class RandomErasing(object):
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False):
        assert isinstance(value, (numbers.Number, str, tuple, list))
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("range of scale should be between 0 and 1")
        if p < 0 or p > 1:
            raise ValueError("range of random erasing probability should be between 0 and 1")

        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace

    @staticmethod
    def get_params(img, scale, ratio, value=0):
        """Get parameters for ``erase`` for a random erasing.

        Args:
            img (np.array): ndarray image of size (H, W, C) to be erased.
            scale: range of proportion of erased area against input image.
            ratio: range of aspect ratio of erased area.

        Returns:
            tuple: params (i, j, h, w, v) to be passed to ``erase`` for random erasing.
        """
        img_h, img_w, img_c = img.shape
        area = img_h * img_w

        for _ in range(10):
            erase_area = random.uniform(scale[0], scale[1]) * area
            aspect_ratio = random.uniform(ratio[0], ratio[1])

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))

            if h < img_h and w < img_w:
                i = random.randint(0, img_h - h)
                j = random.randint(0, img_w - w)
                if isinstance(value, numbers.Number):
                    v = value
                elif value == 'random':
                    v = np.random.randint(0, 256, size=(h, w, img_c))
                else:
                    raise NotImplementedError('Not implement')
                return i, j, h, w, v

        # Return original image
        return 0, 0, img_h, img_w, img

    def __call__(self, results):
        if random.uniform(0, 1) >= self.p:
            return results
        img = results['img']
        y, x, h, w, v = self.get_params(img, scale=self.scale, ratio=self.ratio, value=self.value)
        img[y:y + h, x:x + w] = v
        results['img'] = img
        return results


# # -------------------------Unbiased Teacher augmentation-------------------------
class RandomErase(object):
    def __init__(self, use_box=False):
        CLS = RandomErasing
        self.transforms = [
            CLS(p=0.7, scale=(0.05, 0.2), ratio=(0.3, 3.3), value="random"),
            CLS(p=0.5, scale=(0.02, 0.2), ratio=(0.1, 6), value="random"),
            CLS(p=0.3, scale=(0.02, 0.2), ratio=(0.05, 8), value="random")
        ]

    def __call__(self, results):
        for t in self.transforms:
            results = t(results)
        return results


class AugmentationUTWeak(object):
    def __init__(self):
        self.transforms_1 = Albu(transforms=[
            dict(type='ColorJitter', brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            dict(type='ToGray', p=0.2),
            dict(type='GaussianBlur', sigma_limit=(0.1, 2.0), p=0.2),
        ], bbox_params=dict(type='BboxParams', format='pascal_voc', label_fields=['gt_labels']),
            keymap={'img': 'image', 'gt_bboxes': 'bboxes'}
        )

    def __call__(self, results):
        results = self.transforms_1(results)
        return results


class AugmentationUTStrong(object):
    def __init__(self, use_re=True, use_box=False):
        self.transforms_1 = Albu(transforms=[
            dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
            dict(type='ToGray', p=0.2),
            dict(type='GaussianBlur', sigma_limit=(0.1, 2.0), p=0.5),
        ], bbox_params=dict(type='BboxParams', format='pascal_voc', label_fields=['gt_labels']),
            keymap={'img': 'image', 'gt_bboxes': 'bboxes'}
        )
        self.transforms_2 = RandomErase(use_box)
        self.use_re = use_re

    def __call__(self, results):
        results = self.transforms_1(results)
        if self.use_re:
            results = self.transforms_2(results)
        return results


@PIPELINES.register_module()
class AugmentationUT(object):
    def __init__(self, use_weak=False, use_re=True, use_box=False):
        if use_weak:
            self.transforms = AugmentationUTWeak()
        else:
            self.transforms = AugmentationUTStrong(use_re=use_re, use_box=use_box)

    def __call__(self, results):
        results = self.transforms(results)
        return results
