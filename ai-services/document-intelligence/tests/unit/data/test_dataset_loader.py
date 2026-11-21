
import pandas as pd
import pytest

from src.data.dataset_loader import BBCDatasetLoader


@pytest.fixture
def sample_dataset():
    """
    Create a sample BBC-style dataset for testing.
    """

    return pd.DataFrame({
        "category": (
            ["business"] * 10
            + ["sport"] * 10
            + ["tech"] * 10
        ),
        "text": [
            f"Sample article number {i}"
            for i in range(30)
        ]
    })


@pytest.fixture
def dataset_file(tmp_path, sample_dataset):
    """
    Create a temporary CSV dataset.
    """

    file_path = tmp_path / "bbc-text.csv"

    sample_dataset.to_csv(file_path, index=False)

    return file_path


class TestBBCDatasetLoader:

    def test_initialization(self, dataset_file):
        loader = BBCDatasetLoader(dataset_file)

        assert loader.dataset_path == dataset_file

    def test_load_dataset(self, dataset_file):
        loader = BBCDatasetLoader(dataset_file)

        df = loader.load_dataset()

        assert len(df) == 30
        assert list(df.columns) == ["category", "text"]

    def test_file_not_found(self, tmp_path):
        loader = BBCDatasetLoader(
            tmp_path / "missing.csv"
        )

        with pytest.raises(FileNotFoundError):
            loader.load_dataset()

    def test_missing_required_columns(self, tmp_path):
        file_path = tmp_path / "invalid.csv"

        pd.DataFrame({
            "category": ["sport"],
            "content": ["Sample article"]
        }).to_csv(file_path, index=False)

        loader = BBCDatasetLoader(file_path)

        with pytest.raises(ValueError):
            loader.load_dataset()

    def test_missing_values(self, tmp_path):
        file_path = tmp_path / "missing_values.csv"

        pd.DataFrame({
            "category": ["sport", None],
            "text": ["Sample article", "Another article"]
        }).to_csv(file_path, index=False)

        loader = BBCDatasetLoader(file_path)

        with pytest.raises(ValueError):
            loader.load_dataset()

    def test_split_dataset_sizes(self, sample_dataset):
        loader = BBCDatasetLoader()

        train_df, validation_df, test_df = (
            loader.split_dataset(sample_dataset)
        )

        assert len(train_df) == 21
        assert len(validation_df) == 4
        assert len(test_df) == 5

    def test_split_dataset_category_distribution(
        self,
        sample_dataset
    ):
        loader = BBCDatasetLoader()

        train_df, validation_df, test_df = (
            loader.split_dataset(sample_dataset)
        )

        for df in [train_df, validation_df, test_df]:
            assert set(df["category"]) == {
                "business",
                "sport",
                "tech"
            }

    def test_split_dataset_reproducibility(
        self,
        sample_dataset
    ):
        loader = BBCDatasetLoader()

        first_split = loader.split_dataset(sample_dataset)
        second_split = loader.split_dataset(sample_dataset)

        for first, second in zip(first_split, second_split):
            pd.testing.assert_frame_equal(first, second)

    def test_split_dataset_total_records(
        self,
        sample_dataset
    ):
        loader = BBCDatasetLoader()

        train_df, validation_df, test_df = (
            loader.split_dataset(sample_dataset)
        )

        total_records = (
            len(train_df)
            + len(validation_df)
            + len(test_df)
        )

        assert total_records == len(sample_dataset)
