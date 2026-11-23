
import pytest
import torch

from data.torch_dataset import BBCTensorDataset


@pytest.fixture
def sample_features():
    return [
        [12, 45, 67, 0],
        [23, 56, 78, 0],
        [34, 89, 90, 12]
    ]


@pytest.fixture
def sample_labels():
    return [0, 1, 2]


class TestBBCTensorDataset:

    def test_initialization(self, sample_features, sample_labels):
        dataset = BBCTensorDataset(sample_features, sample_labels)

        assert isinstance(dataset.features, torch.Tensor)
        assert isinstance(dataset.labels, torch.Tensor)

    def test_dataset_length(self, sample_features, sample_labels):
        dataset = BBCTensorDataset(sample_features, sample_labels)

        assert len(dataset) == 3

    def test_feature_tensor_shape(self, sample_features, sample_labels):
        dataset = BBCTensorDataset(sample_features, sample_labels)

        assert dataset.features.shape == (3, 4)

    def test_label_tensor_shape(self, sample_features, sample_labels):
        dataset = BBCTensorDataset(sample_features, sample_labels)

        assert dataset.labels.shape == (3,)

    def test_feature_dtype(self, sample_features, sample_labels):
        dataset = BBCTensorDataset(sample_features, sample_labels)

        assert dataset.features.dtype == torch.long

    def test_label_dtype(self, sample_features, sample_labels):
        dataset = BBCTensorDataset(sample_features, sample_labels)

        assert dataset.labels.dtype == torch.long

    def test_getitem(self, sample_features, sample_labels):
        dataset = BBCTensorDataset(sample_features, sample_labels)

        features, label = dataset[0]

        assert torch.equal(
            features,
            torch.tensor([12, 45, 67, 0], dtype=torch.long)
        )
        assert label.item() == 0

    def test_getitem_returns_tensors(self, sample_features, sample_labels):
        dataset = BBCTensorDataset(sample_features, sample_labels)

        features, label = dataset[0]

        assert isinstance(features, torch.Tensor)
        assert isinstance(label, torch.Tensor)

    def test_mismatched_feature_and_label_lengths(self, sample_features):
        with pytest.raises(ValueError):
            BBCTensorDataset(sample_features, [0, 1])

    def test_empty_features(self):
        with pytest.raises(ValueError):
            BBCTensorDataset([], [])

    def test_inconsistent_sequence_lengths(self):
        features = [
            [12, 45, 67],
            [23, 56],
            [34, 89, 90]
        ]

        labels = [0, 1, 2]

        with pytest.raises(ValueError):
            BBCTensorDataset(features, labels)
