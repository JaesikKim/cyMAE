# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import random
import math
import numpy as np

class RandomMaskingGenerator:
    def __init__(self, input_size, mask_ratio):

        self.input_size = input_size
        self.num_mask = int(mask_ratio * self.input_size)

    def __repr__(self):
        repr_str = "Mask: total features {}, mask features {}".format(
            self.input_size, self.num_mask
        )
        return repr_str

    def __call__(self):
        mask = np.hstack([
            np.zeros(self.input_size - self.num_mask),
            np.ones(self.num_mask),
        ])
        np.random.shuffle(mask)
        return mask # [30]