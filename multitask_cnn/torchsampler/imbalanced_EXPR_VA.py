import numpy as np
import torch
import torch.utils.data


class ImbalancedDatasetSamplerEXPRVA(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices=None, num_samples=None):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        all_labels = dataset.get_all_label()
        va_labels = np.array([[x[0], x[1]] for x in all_labels['VA']])
        N, C = va_labels.shape
        assert C == 2
        hist, x_edges, y_edges = np.histogram2d(va_labels[:, 0], va_labels[:, 1], bins=[20, 20])
        x_bin_id = np.digitize(va_labels[:, 0], bins=x_edges) - 1
        y_bin_id = np.digitize(va_labels[:, 1], bins=y_edges) - 1
        x_bin_id[x_bin_id == 20] = 20 - 1
        y_bin_id[y_bin_id == 20] = 20 - 1
        va_weights = []
        for x, y in zip(x_bin_id, y_bin_id):
            assert hist[x, y] != 0
            va_weights += [1 / hist[x, y]]

        expr_labels = all_labels['EXPR']
        mapping = {}
        for label in expr_labels:
            if label not in mapping:
                mapping[label] = 0
            mapping[label] += 1

        expr_weights = [1.0 / mapping[x] for x in expr_labels]

        weights = np.array(va_weights) + np.array(expr_weights)
        weights = weights / np.max(weights)

        self.weights = torch.from_numpy(weights)

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
