import pickle

import numpy as np
import torch
from PIL import Image

from path import PATH
from .dataset import DatasetBase

PRESET_VARS = PATH()


class DatasetEXPRVA(DatasetBase):
    def __init__(self, opt, train_mode='Train', transform=None):
        super(DatasetEXPRVA, self).__init__(opt, train_mode, transform)
        self._name = 'dataset_EXPR_VA'
        self._train_mode = train_mode
        if transform is not None:
            self._transform = transform  
        # read dataset
        self._read_dataset_paths()

    def get_all_label(self):
        return {
            'EXPR': self._data['EXPR'],
            'VA': self._data['VA']
        }

    def __getitem__(self, index):
        assert (index < self._dataset_size)
        # start_time = time.time()
        img_path = self._data['path'][index]
        image = Image.open(img_path).convert('RGB')
        expr = self._data['EXPR'][index]
        valence = self._data['VA'][index][0]
        arousal = self._data['VA'][index][1]

        label = torch.FloatTensor([expr, valence, arousal])
        image = self._transform(image)
        # pack data
        sample = {'image': image,
                  'label': label,
                  'sub': label,
                  'path': img_path,
                  'index': index 
                  }
        # print (time.time() - start_time)
        return sample

    def _read_dataset_paths(self):
        if self._train_mode == 'Train':
            self._data = self._read_path_label(PRESET_VARS.EXPR_VA.data_file)
            self._ids = np.arange(len(self._data['path']))
            self._dataset_size = len(self._ids)
        else:
            raise Exception('EXPR_VA dataset can\'t be used in other mode except training')

    def __len__(self):
        if self._train_mode == 'Train':
            return self._dataset_size
        raise Exception('EXPR_VA dataset can\'t be used in other mode except training')

    def _read_path_label(self, file_path):
        data = pickle.load(open(file_path, 'rb'))
        return data
