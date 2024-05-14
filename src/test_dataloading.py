"""Unit testing for the preprocessing script."""

import random

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

import dataloading
import preprocessing


@pytest.fixture
def dataset() -> Dataset:
    data = preprocessing.fetch_data()
    data, feats, target = preprocessing.prepare_data(data)
    return dataloading.ElectricTimeSeries(data, feats, target)


@pytest.fixture
def dataloader(dataset: Dataset) -> DataLoader:
    return DataLoader(dataset, batch_size=128, shuffle=False,
                      collate_fn=dataloading.collate_fn)


def test_dataset(dataset: Dataset):
    """Test if dataset items have the correct size."""
    i = random.randrange(len(dataset))
    X, y = dataset.__getitem__(i)
    X_size = torch.Size([dataloading.N_INPUT_HOUR, len(dataset.feats)])
    y_size = torch.Size([1])
    assert X.size() == X_size and y.size() == y_size


def test_dataloader(dataloader: DataLoader):
    """Test if loop over dataloader runs till the end."""
    for i, _ in enumerate(dataloader):
        continue
    assert i == len(dataloader) - 1
