import pickle
import random

import numpy as np
import os
import torch
from PIL import Image

from path import PATH
from .dataset import DatasetBase

PRESET_VARS = PATH()


class DatasetMixedVA(DatasetBase):
    def __init__(self, opt, train_mode='Train', transform = None):
        super(DatasetMixedVA, self).__init__(opt, train_mode, transform)
        self._name = 'dataset_Mixed_VA'
        self._train_mode = train_mode
        if transform is not None:
            self._transform = transform  
        # read dataset
        self._read_dataset_paths()

    def get_all_label(self):
        return self._data['label']

    def __getitem__(self, index):
        assert (index < self._dataset_size)
        img_path = self._data['path'][index]
        image = Image.open(img_path).convert('RGB')
        label = self._data['label'][index]
        if self._opt.disable_shared_annotations:
            expr = -2.
        elif self._train_mode == 'Train':
            expr = self._data['expr'][index]
        else:
            expr = -2.
        image = self._transform(image)
        # pack data
        sample = {'image': image,
                  'label': label,
                  'sub': expr,
                  'path': img_path,
                  'index': index 
                  }
        return sample

    def _read_dataset_paths(self):
        self._data = self._read_path_label(PRESET_VARS.Mixed_VA.data_file)
        self._ids = np.arange(len(self._data['label']))
        self._dataset_size = len(self._ids)

    def __len__(self):
        return self._dataset_size

    def _read_path_label(self, file_path):
        data = pickle.load(open(file_path, 'rb'))
        # read frames ids
        if self._train_mode == 'Train':
            data = data['Train_Set']
        elif self._train_mode == 'Validation':
            data = data['Validation_Set']
        elif self._train_mode == 'Test':
            data = data['Test_Set']
        else:
            raise ValueError("train mode must be in : Train, Validation, Test")

        return data
