import pickle

import numpy as np
from PIL import Image

from path import PATH
from .dataset import DatasetBase

PRESET_VARS = PATH()


class DatasetMixedEXPR(DatasetBase):
    def __init__(self, opt, train_mode='Train', transform = None):
        super(DatasetMixedEXPR, self).__init__(opt, train_mode, transform)
        self._name = 'dataset_Mixed_EXPR'
        self._train_mode = train_mode
        if transform is not None:
            self._transform = transform  
        # read dataset
        self._read_dataset_paths()

    def __getitem__(self, index):
        assert (index < self._dataset_size)
        # start_time = time.time()
        img_path = self._data['path'][index]
        image = Image.open(img_path).convert('RGB')
        label = self._data['label'][index]
        if self._opt.disable_shared_annotations:
            # If we don't want to use shared annotations, simply set it all = (-2, -2)
            va = np.array([-2., -2.])
        elif self._train_mode == 'Train':
            va = self._data['va'][index]
        else:
            va = np.array([-2., -2.])

        image = self._transform(image)
        # pack data
        sample = {'image': image,
                  'label': label,
                  'sub': va,
                  'path': img_path,
                  'index': index 
                  }
        # print (time.time() - start_time)
        return sample

    def _read_dataset_paths(self):
        self._data = self._read_path_label(PRESET_VARS.Mixed_EXPR.data_file)
        self._ids = np.arange(len(self._data['label']))
        self._dataset_size = len(self._ids)

    def get_all_label(self):
        return self._data['label']

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

