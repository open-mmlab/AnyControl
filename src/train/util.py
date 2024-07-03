import os
import random
import logging
import numpy as np
import torch

from typing import Tuple, Union, Type, Optional, Dict, Callable 
from torch.utils.data._utils.collate import collate, default_collate_fn_map


class DataContainer:
    """A container for any type of objects.

    Typically tensors will be stacked in the collate function and sliced along
    some dimension in the scatter function. This behavior has some limitations.
    1. All tensors have to be the same size.
    2. Types are limited (numpy array or Tensor).

    We design `DataContainer` and `MMDataParallel` to overcome these
    limitations. The behavior can be either of the following.

    - copy to GPU, pad all tensors to the same size and stack them
    - copy to GPU without stacking
    - leave the objects as is and pass it to the model
    - pad_dims specifies the number of last few dimensions to do padding
    """

    def __init__(self,
                 data: Union[torch.Tensor, np.ndarray],
                 stack: bool = False,
                 padding_value: int = 0,
                 cpu_only: bool = False,
                 pad_dims: int = 2):
        self._data = data
        self._cpu_only = cpu_only
        self._stack = stack
        self._padding_value = padding_value
        assert pad_dims in [None, 1, 2, 3]
        self._pad_dims = pad_dims

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({repr(self.data)})'

    def __len__(self) -> int:
        return len(self._data)

    @property
    def data(self) -> Union[torch.Tensor, np.ndarray]:
        return self._data

    @property
    def datatype(self) -> Union[Type, str]:
        if isinstance(self.data, torch.Tensor):
            return self.data.type()
        else:
            return type(self.data)

    @property
    def cpu_only(self) -> bool:
        return self._cpu_only


def read_anno(anno_path):
    fi = open(anno_path)
    lines = fi.readlines()
    fi.close()
    file_ids, annos = [], []
    for line in lines:
        id, txt = line.split('\t')
        file_ids.append(id)
        annos.append(txt)
    return file_ids, annos


def keep_and_drop(conditions, keep_all_prob, drop_all_prob, drop_each_prob):
    results = []
    seed = random.random()
    if seed < keep_all_prob:
        results = conditions
    elif seed < keep_all_prob + drop_all_prob:
        for condition in conditions:
            results.append(np.zeros_like(condition))
    else:
        for i in range(len(conditions)):
            if random.random() < drop_each_prob[i]:
                results.append(np.zeros_like(conditions[i]))
            else:
                results.append(conditions[i])
    return results


def keep_and_drop_with_status(conditions, keep_all_prob, drop_all_prob, drop_each_prob):
    results = []
    keep = []
    seed = random.random()
    if seed < keep_all_prob:
        results = conditions
        keep = [True] * len(conditions)
    elif seed < keep_all_prob + drop_all_prob:
        for condition in conditions:
            results.append(np.zeros_like(condition))
        keep = [False] * len(conditions)
    else:
        for i in range(len(conditions)):
            if random.random() < drop_each_prob[i]:
                results.append(np.zeros_like(conditions[i]))
                keep.append(False)
            else:
                results.append(conditions[i])
                keep.append(True)
    return results, keep


def collate_dc_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    return batch
default_collate_fn_map[DataContainer] = collate_dc_fn


def interleaved_collate(batch):
    return collate(batch, collate_fn_map=default_collate_fn_map)
