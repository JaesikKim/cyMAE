# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
from torchvision.datasets.vision import VisionDataset
from FlowCytometryTools import FCMeasurement

from collections import deque
import os
import os.path
import numpy as np
import random
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

import torch
import time

def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_csv_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, 'csv')

def is_fcs_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, 'fcs')

def read_fcs(path: str):
    features = ['89Y_CD45', '141Pr_CD196_CCR6', '143Nd_CD123_IL-3R', '144Nd_CD19',
       '145Nd_CD4', '146Nd_CD8a', '147Sm_CD11c', '148Nd_CD16', '149Sm_CD45RO',
       '150Nd_CD45RA', '151Eu_CD161', '152Sm_CD194_CCR4', '153Eu_CD25_IL-2Ra',
       '154Sm_CD27', '155Gd_CD57', '156Gd_CD183_CXCR3', '158Gd_CD185_CXCR5',
       '160Gd_CD28', '161Dy_CD38', '163Dy_CD56_NCAM', '164Dy_TCRgd',
       '166Er_CD294', '167Er_CD197_CCR7', '168Er_CD14',
       '170Er_CD3', '171Yb_CD20', '172Yb_CD66b', '173Yb_HLA-DR', '174Yb_IgD',
       '176Yb_CD127_IL-7Ra', 'OmiqFilter']
    
    fcs = FCMeasurement(ID="_".join(path.split("/")[-1].split("_")[:-1]), datafile=path).data
    fcs = fcs[features]
    return fcs.to_numpy()

def read_meta(fold, split_set, path: str, label_name: str):
    import pandas as pd
    meta = pd.read_csv(path)
    if (split_set is not None) and fold:
        meta = meta[meta['fold_'+str(fold)] == split_set]
        meta = meta.reset_index(drop=True)
    if label_name is not None:
        meta = meta[~meta[label_name].isna()]
        meta = meta.reset_index(drop=True)
        class_to_idx = {cls_name: i for i, cls_name in enumerate(meta[label_name].unique())}
        fcsfile_to_class = {meta["Filename"].loc[i]: meta[label_name].loc[i] for i in range(len(meta))}
    else:
        fcsfile_to_class = {meta["Filename"].loc[i]: None for i in range(len(meta))}
        class_to_idx = None
    return fcsfile_to_class, class_to_idx
    
def read_FilterValuesToNames(path: str):
    import pandas as pd

    typefilter= ["Plasmablast", "Th2/activated", "Treg/activated", "CD8Naive", "Treg", "EarlyNK", "CD66bnegCD45lo", "CD4Naive", "Th2", "CD8TEM2", "Th17", "IgDposMemB", "CD8Naive/activated", "CD8TEMRA/activated", "Eosinophil", "CD8TEM3/activated", "DPT", "MAITNKT", "gdT", "CD8TEM2/activated", "nnCD4CXCR5pos/activated", "IgDnegMemB", "CD45hiCD66bpos", "LateNK", "Neutrophil", "DNT", "Basophil", "pDC", "CD8TEM1/activated", "mDC", "Th1", "DNT/activated", "Th1/activated", "CD8TEMRA", "CD8TCM/activated", "CD8TEM1", "CD4Naive/activated", "NaiveB", "ILC", "CD8TEM3", "Th17/activated", "CD8TCM", "ClassicalMono", "DPT/activated", "nnCD4CXCR5pos", "TotalMonocyte"]
    
    df = pd.read_csv(path, header=None)
    df = df[np.isin(df.iloc[:,1], typefilter)]
    df = df.reset_index(drop=True)
    value_to_class = dict(zip(df[0], df[1]))
    class_to_idx = {cls_name: i for i, cls_name in enumerate(df[1].unique())}
    return value_to_class, class_to_idx

def fcs_loader(path: str) -> Any:
    # read fcs file
    fcs = read_fcs(path)[:,:-1]
    # arcsinh transformation
    fcs = np.arcsinh(fcs)
    return fcs

class DatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::
        root/xxx.fcs
        root/xxy.fcs
        root/xxz.fcs
        root/meta.csv
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.

        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        super(DatasetFolder, self).__init__(root, transform=transform, target_transform=target_transform)
        self.metafile = self._find_meta(self.root)
        self.fcsfiles = self._find_fcs(self.root)
        self.FilterValuesToNamesfile = self._find_FilterValuesToNames(self.root)

    def _find_fcs(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        fcsfiles = [f for f in os.listdir(dir) if is_fcs_file(f)]
        return fcsfiles

    def _find_meta(self, dir: str) -> str:
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        csvfiles = [f for f in os.listdir(dir) if is_csv_file(f) and "meta" in f]
        if len(csvfiles) > 1:
            msg = "There are multiple csv (Meta) files"
            raise RuntimeError(msg)
        else:
            return csvfiles[0]

    def _find_FilterValuesToNames(self, dir: str) -> str:
        csvfiles = [f for f in os.listdir(dir) if is_csv_file(f) and "_FilterValuesToNames" in f]
        if len(csvfiles) > 1:
            msg = "There are multiple csv (_FilterValuesToNames) files"
            raise RuntimeError(msg)
        else:
            return csvfiles[0]

class CellFolder(DatasetFolder):
    """A generic data loader where the samples are arranged in this way: ::
        root/xxx.fcs
        root/xxy.fcs
        root/xxz.fcs
        root/meta.csv
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.

        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """
    def __init__(
            self,
            fold,
            split_set,
            root: str,
            predefined_class_to_idx = None,
            fewshot = False,
            nshot = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = fcs_loader
    ):
        super(CellFolder, self).__init__(root, transform=transform, target_transform=target_transform)
        samples, class_to_idx = self.make_dataset(fold, split_set, self.root, self.fcsfiles, self.metafile, self.FilterValuesToNamesfile, predefined_class_to_idx) # (path, cell idx, class idx)
        
        # Few-shot learning setup (training/val set only)
        if ((split_set == 0) or (split_set == 1)) and fold and fewshot:
            tmp_samples = {}
            for sample in samples:
                c = sample[2]
                if c not in tmp_samples:
                    tmp_samples[c] = []
                if len(tmp_samples[c]) < nshot:
                    tmp_samples[c].append(sample)
            import itertools
            samples = list(itertools.chain.from_iterable([v for k,v in tmp_samples.items()]))
        
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            raise RuntimeError(msg)

        self.loader = loader

        self.classes = list(class_to_idx.keys())
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.path_cache = deque([],5)
        self.fcs_cache = deque([],5)

    def make_dataset(
        self,
        fold,
        split_set,
        directory: str,
        fcsfiles: list[str],
        metafile: str,
        FilterValuesToNamesfile: str,
        predefined_class_to_idx=None
    ) -> List[Tuple[str, int]]:
  
        instances = []
        directory = os.path.expanduser(directory)
  
        fcsfile_to_class, _ = read_meta(fold, split_set, os.path.join(directory, metafile), None)
        value_to_class, class_to_idx = read_FilterValuesToNames(os.path.join(directory, FilterValuesToNamesfile))
        if predefined_class_to_idx is not None:
            class_to_idx = predefined_class_to_idx
        for fname in fcsfiles:
            path = os.path.join(directory, fname)
            if fname in fcsfile_to_class.keys():
                for idx, value in enumerate(read_fcs(path)[:,-1]):
                    if (not np.isnan(value)) and (value in value_to_class.keys()):
                        instances.append((path, idx, class_to_idx[value_to_class[value]]))

        return instances, class_to_idx
    
    def _check_cache(self, path):
        if path in self.path_cache:
            loc = self.path_cache.index(path)
            return self.fcs_cache[loc]
        else:
#             print("new added in cache")
            new_fcs = self.loader(path)
            self.path_cache.append(path)
            self.fcs_cache.append(new_fcs)
            return new_fcs

    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        while True:
            try:
                path, idx, target = self.samples[index]
                break
            except Exception as e:
                print(e)
                index = random.randint(0, len(self.samples) - 1)
        return path, idx, target
        

    def __len__(self) -> int:
        return len(self.samples)

    def custom_collate_fn(self, batch):
        paths, idxs, targets = zip(*batch)
        fcs_dict = {}
        for path in np.unique(paths):
            fcs = self._check_cache(path)
            fcs_dict[path] = fcs
        samples = []
        labels = []
        bool_masked_poses = []
        for i in range(len(batch)):
            sample = fcs_dict[paths[i]][idxs[i]]
            label = targets[i]
            if self.transform is not None:
                sample, bool_masked_pos = self.transform(sample)
            if self.target_transform is not None:
                label = self.target_transform(label)
            samples.append(sample)
            bool_masked_poses.append(bool_masked_pos)
            labels.append(label)
        samples = torch.stack(samples)
        if self.transform is not None:
            bool_masked_poses = np.stack(bool_masked_poses)
            bool_masked_poses = torch.tensor(bool_masked_poses)
            samples = (samples, bool_masked_poses)
        labels = torch.tensor(labels)
        
        return samples, labels