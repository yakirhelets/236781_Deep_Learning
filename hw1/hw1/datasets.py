import numpy as np

import torch
from torch.utils.data import Dataset


class RandomImageDataset(Dataset):
    """
    A dataset returning random noise images of specified dimensions
    """

    def __init__(self, num_samples, num_classes, C, W, H):
        """
        :param num_samples: Number of samples (labeled images in the dataset)
        :param num_classes: Number of classes (labels)
        :param C: Number of channels per image
        :param W: Image width
        :param H: Image height
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.image_dim = (C, W, H)

    def __getitem__(self, index):
        """
        Returns a labeled sample.
        :param index: Sample index
        :return: A tuple (sample, label).
        """

        # TODO:
        #  Create a random image tensor and return it.
        #  Try to make sure to always return the same image for the
        #  same index (make it deterministic per index), but don't mess-up
        #  the random state outside this method.

        # ====== YOUR CODE: ======
        result = []
        torch.random.manual_seed(index)
        np.random.seed(index)
        for i in range(self.num_samples):
            random_image_tensor = torch.randint(0, 255, (self.image_dim[0], self.image_dim[1], self.image_dim[2]))
            random_label = np.random.randint(0, self.num_classes)
            tuple = (random_image_tensor, random_label)
            result.append(tuple)

        return result[0]
        # ========================

    def __len__(self):
        """
        :return: Number of samples in this dataset.
        """
        # ====== YOUR CODE: ======
        return self.num_samples
        # ========================


class SubsetDataset(Dataset):
    """
    A dataset that wraps another dataset, returning a subset from it.
    """
    def __init__(self, source_dataset: Dataset, subset_len, offset=0):
        """
        Create a SubsetDataset from another dataset.
        :param source_dataset: The dataset to take samples from.
        :param subset_len: The total number of sample in the subset.
        :param offset: The offset index to start taking samples from.
        """
        if offset + subset_len > len(source_dataset):
            raise ValueError("Not enough samples in source dataset")

        self.source_dataset = source_dataset
        self.subset_len = subset_len
        self.offset = offset

    def __getitem__(self, index):
        # TODO:
        #  Return the item at index + offset from the source dataset.
        #  Raise an IndexError if index is out of bounds.

        # ====== YOUR CODE: ======
        new_index = index + self.offset

        subset_starting_index = self.offset
        subset_ending_index = self.offset + self.subset_len
        if (new_index < subset_starting_index) or (new_index > subset_ending_index-1):
            raise IndexError('New index is out of bounds')
        else:
            return self.source_dataset[new_index]
        # ========================

    def __len__(self):
        # ====== YOUR CODE: ======
        return self.subset_len
        # ========================

