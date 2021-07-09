from collections import OrderedDict

import torch.utils.data

from data.dataset import DatasetFactory
from utils.data_utils import seq_collate


class MultitaskIteratorWrapper:
    def __init__(self, multitask_dataloader):
        self.multitask_dataloader = multitask_dataloader
        self.data_loaders = OrderedDict([(k, iter(x)) for (k, x) in self.multitask_dataloader.items()])
        self._index = 0
        self.max_n_iters = min([len(x) for (k, x) in self.data_loaders.items()])

    def __iter__(self):
        return self

    def reset(self):
        self.data_loaders = OrderedDict([(k, iter(x)) for (k, x) in self.multitask_dataloader.items()])
        self._index = 0
        self.max_n_iters = min([len(x) for (k, x) in self.data_loaders.items()])

    def __next__(self):
        if self._index < self.max_n_iters:
            data_batch = dict()
            for (task, dataloader_iter) in self.data_loaders.items():
                batch_per_task = next(dataloader_iter)
                data_batch[task] = batch_per_task
            self._index += 1
            return data_batch
        else:
            raise StopIteration

    def __len__(self):
        return self.max_n_iters


class MultitaskDatasetDataLoader:
    def __init__(self, opt, train_mode, transform=None):
        self._opt = opt
        self.train_mode = train_mode
        self.transform = transform
        self._is_train = self.train_mode == 'Train'
        self._num_threds = opt.n_threads_train if self._is_train else opt.n_threads_test
        self._create_datasets()
        self._create_dataloaders()

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def _create_datasets(self):
        self.datasets = OrderedDict()
        for i, dataset_name in enumerate(self._opt.dataset_names):
            task = self._opt.tasks[i]
            self.datasets[task] = DatasetFactory.get_by_name(dataset_name, self._opt, self.train_mode, self.transform)
        self.cumulative_sizes = self.cumsum([dataset for (k, dataset) in self.datasets.items()])

    def _create_dataloaders(self):
        self.dataloaders = OrderedDict()
        for i, dataset_name in enumerate(self._opt.dataset_names):
            task = self._opt.tasks[i]
            if not self._is_train:
                dataloader = torch.utils.data.DataLoader(
                    self.datasets[task],
                    batch_size=self._opt.batch_size,
                    shuffle=self._is_train,
                    num_workers=int(self._num_threds),
                    collate_fn=seq_collate,
                    drop_last=self._is_train)
            else:
                dataloader = torch.utils.data.DataLoader(
                    self.datasets[task],
                    batch_size=self._opt.batch_size,
                    shuffle=self._is_train,
                    num_workers=int(self._num_threds),
                    collate_fn=seq_collate,
                    drop_last=self._is_train)
            self.dataloaders[task] = dataloader

    def load_multitask_train_data(self):
        assert self._is_train
        return MultitaskIteratorWrapper(self.dataloaders)

    def load_multitask_val_test_data(self):
        return self.dataloaders

    def __len__(self):
        return min([len(x) for (k, x) in self.dataloaders.iteritems()])
