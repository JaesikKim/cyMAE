# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import os
import torch
import numpy as np

from torchvision import datasets, transforms

from masking_generator import RandomMaskingGenerator
from dataset_folder import CellFolder, SampleFolder


class MaskingForMAE(object):
    def __init__(self, args):

        self.masked_position_generator = RandomMaskingGenerator(
            args.input_size, args.mask_ratio
        )

    def __call__(self, sample):
        return torch.tensor(sample), self.masked_position_generator()

    def __repr__(self):
        repr = "(MaskingForMAE,\n"
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr

def build_pretraining_dataset(args):
    transform = MaskingForMAE(args)
    print("Data masking = %s" % str(transform))
    return CellFolder(None, None, args.data_path, transform=transform)



class NullForMAE(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        return torch.tensor(sample), torch.tensor(-1)

class RandomSelectForMAE(object):
    def __init__(self, args):
        self.num_cell_selected = args.batch_size

    def __call__(self, sample):
        # [C, F] -> [B, F]
        mask = np.hstack([
            np.zeros(sample.shape[0] - self.num_cell_selected),
            np.ones(self.num_cell_selected),
        ]).astype(bool)
        np.random.shuffle(mask)
        
        return torch.tensor(sample[mask]), torch.tensor(-1)

def build_dataset(split_set, args, predefined_class_to_idx=None):

    if args.task == "cell-level":
        transform = NullForMAE()
        dataset = CellFolder(args.fold, split_set, args.data_path, predefined_class_to_idx=predefined_class_to_idx, labelling=("deeper" if args.nb_classes >20 else "simple"), fewshot = args.fewshot, nshot = args.nshot, transform=transform) 
    else:
        transform = RandomSelectForMAE(args)
        dataset = SampleFolder(args.fold, split_set, args.data_path, args.label_name, predefined_class_to_idx=predefined_class_to_idx, transform=transform)
    nb_classes = args.nb_classes
    assert len(dataset.class_to_idx) == nb_classes
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes

