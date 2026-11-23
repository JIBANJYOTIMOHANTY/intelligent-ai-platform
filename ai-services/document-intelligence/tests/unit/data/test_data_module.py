
import pytest
import torch

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from data.dataset_builder import PreparedDataset
from data.data_module import BBCDataModule


@pytest.fixture
def prepared_dataset():
    return PreparedDataset(
        X_train=[
            [1, 2, 3, 0],
            [4, 5, 6, 0],
            [7, 8, 9, 0],
            [1, 3, 5, 0],
            [2, 4, 6, 0],
        ],
        y_train=[0, 1, 2, 3, 4],

        X_val=[
            [1, 2, 0, 0],
            [3, 4, 0, 0],
        ],
        y_val=[0, 1],

        X_test=[
            [5, 6, 0, 0],
            [7, 8, 0, 0],
        ],
        y_test=[2, 3]
    )


@pytest.fixture
def data_module(prepared_dataset):
    return BBCDataModule(
        prepared_dataset=prepared_dataset,
        batch_size=2
    )


class TestBBCDataModule:

    def test_initialization(self, data_module):
        assert data_module.batch_size == 2

    def test_invalid_batch_size(self, prepared_dataset):
        with pytest.raises(ValueError):
            BBCDataModule(prepared_dataset, batch_size=0)

        with pytest.raises(ValueError):
            BBCDataModule(prepared_dataset, batch_size=-1)

    def test_dataset_lengths(self, data_module):
        assert len(data_module.train_dataset) == 5
        assert len(data_module.val_dataset) == 2
        assert len(data_module.test_dataset) == 2

    def test_dataloaders_are_created(self, data_module):
        assert isinstance(data_module.train_loader, DataLoader)
        assert isinstance(data_module.val_loader, DataLoader)
        assert isinstance(data_module.test_loader, DataLoader)

    def test_dataloader_accessors(self, data_module):
        assert data_module.train_dataloader() is data_module.train_loader
        assert data_module.val_dataloader() is data_module.val_loader
        assert data_module.test_dataloader() is data_module.test_loader

    def test_train_loader_uses_random_sampler(self, data_module):
        assert isinstance(
            data_module.train_loader.sampler,
            RandomSampler
        )

    def test_validation_and_test_use_sequential_sampler(self, data_module):
        assert isinstance(
            data_module.val_loader.sampler,
            SequentialSampler
        )

        assert isinstance(
            data_module.test_loader.sampler,
            SequentialSampler
        )

    def test_batch_shapes(self, data_module):
        features, labels = next(iter(data_module.train_dataloader()))

        assert features.shape == (2, 4)
        assert labels.shape == (2,)

    def test_batch_tensor_dtypes(self, data_module):
        features, labels = next(iter(data_module.train_dataloader()))

        assert features.dtype == torch.long
        assert labels.dtype == torch.long

    def test_number_of_batches(self, data_module):
        assert len(data_module.train_dataloader()) == 3
        assert len(data_module.val_dataloader()) == 1
        assert len(data_module.test_dataloader()) == 1

    def test_last_training_batch_size(self, data_module):
        batches = list(data_module.train_dataloader())

        assert batches[-1][0].shape[0] == 1
        assert batches[-1][1].shape[0] == 1
