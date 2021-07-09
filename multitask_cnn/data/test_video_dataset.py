import numpy as np
from PIL import Image

from path import PATH

PRESET_VARS = PATH()


class DatasetTest(object):
    def __init__(self, opt, video_data,  train_mode='Test', transform = None):
        self._name = 'Test_dataset'
        self._train_mode = train_mode
        if transform is not None:
            self._transform = transform

        # read dataset
        self._data = video_data
        self._read_dataset()

    def __getitem__(self, index):
        assert (index < self._dataset_size)
        image = None
        label = None
        img_path = self._data['path'][index]
        image = Image.open( img_path).convert('RGB')
        label = self._data['label'][index]
        frame_id = self._data['frames_ids'][index]

        # transform data
        image = self._transform(image)
        # pack data
        sample = {'image': image,
                  'label': label,
                  'path': img_path,
                  'index': index,
                  'frames_ids': frame_id
                  }

        return sample
        
    def _read_dataset(self):        
        self._ids = np.arange(len(self._data['path'])) 
        self._dataset_size = len(self._ids)
        
    def __len__(self):
        return self._dataset_size


