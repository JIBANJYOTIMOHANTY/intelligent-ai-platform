
import pandas as pd
import pytest

from data.dataset_builder import BBCDatasetBuilder, PreparedDataset


@pytest.fixture
def dataset_splits():
    """
    Create controlled training, validation and test datasets.
    """

    train_df = pd.DataFrame({
        "category": [
            "business",
            "entertainment",
            "politics",
            "sport",
            "tech"
        ],
        "text": [
            "business market economy",
            "entertainment movie music",
            "politics government election",
            "sport football cricket",
            "tech computer software"
        ]
    })

    val_df = pd.DataFrame({
        "category": [
            "business",
            "entertainment",
            "politics",
            "sport",
            "tech"
        ],
        "text": [
            "validationonlyword business",
            "validationonlyword movie",
            "validationonlyword government",
            "validationonlyword football",
            "validationonlyword computer"
        ]
    })

    test_df = pd.DataFrame({
        "category": [
            "business",
            "entertainment",
            "politics",
            "sport",
            "tech"
        ],
        "text": [
            "testonlyword market",
            "testonlyword movie",
            "testonlyword election",
            "testonlyword cricket",
            "testonlyword software"
        ]
    })

    return train_df, val_df, test_df


@pytest.fixture
def builder(dataset_splits):
    """
    Create a dataset builder with controlled dataset splits.
    """

    train_df, val_df, test_df = dataset_splits

    builder = BBCDatasetBuilder(max_length=10)

    builder.dataset_loader.load_dataset = lambda: pd.DataFrame()

    builder.dataset_loader.split_dataset = (
        lambda dataset, random_state=42: (
            train_df,
            val_df,
            test_df
        )
    )

    return builder


class TestBBCDatasetBuilder:

    def test_initialization(self, builder):
        assert builder.random_state == 42
        assert builder.text_pipeline.is_fitted is False
        assert builder.label_encoder.is_fitted is False

    def test_build_returns_prepared_dataset(self, builder):
        dataset = builder.build()

        assert isinstance(dataset, PreparedDataset)

    def test_build_dataset_sizes(self, builder):
        dataset = builder.build()

        assert len(dataset.X_train) == 5
        assert len(dataset.y_train) == 5

        assert len(dataset.X_val) == 5
        assert len(dataset.y_val) == 5

        assert len(dataset.X_test) == 5
        assert len(dataset.y_test) == 5

    def test_feature_and_label_lengths_match(self, builder):
        dataset = builder.build()

        assert len(dataset.X_train) == len(dataset.y_train)
        assert len(dataset.X_val) == len(dataset.y_val)
        assert len(dataset.X_test) == len(dataset.y_test)

    def test_sequences_have_fixed_length(self, builder):
        dataset = builder.build()

        for split in [
            dataset.X_train,
            dataset.X_val,
            dataset.X_test
        ]:
            for sequence in split:
                assert len(sequence) == 10

    def test_label_encoder_is_fitted(self, builder):
        builder.build()

        assert builder.label_encoder.is_fitted is True

    def test_all_categories_are_encoded(self, builder):
        dataset = builder.build()

        assert set(dataset.y_train) == {0, 1, 2, 3, 4}
        assert set(dataset.y_val).issubset({0, 1, 2, 3, 4})
        assert set(dataset.y_test).issubset({0, 1, 2, 3, 4})

    def test_vocabulary_is_fitted_on_training_data_only(
        self,
        builder
    ):
        builder.build()

        vocabulary = builder.text_pipeline.vocabulary.word_to_index

        assert "business" in vocabulary
        assert "validationonlyword" not in vocabulary
        assert "testonlyword" not in vocabulary

    def test_unknown_words_in_validation_and_test(self, builder):
        dataset = builder.build()

        # Unknown words should be encoded using <UNK> = 1.
        assert 1 in dataset.X_val[0]
        assert 1 in dataset.X_test[0]

    def test_build_uses_training_labels_to_fit_encoder(
        self,
        builder,
        dataset_splits
    ):
        train_df, _, _ = dataset_splits

        builder.build()

        expected_classes = sorted(train_df["category"].unique().tolist())

        assert builder.label_encoder.classes_ == expected_classes

    def test_random_state_is_passed_to_splitter(self, builder):
        received = {}

        def mock_split(dataset, random_state=42):
            received["random_state"] = random_state
            return (
                builder_splits
            )

        builder_splits = builder.dataset_loader.split_dataset(
            None,
            random_state=42
        )

        builder.dataset_loader.split_dataset = mock_split
        builder.random_state = 123

        builder.build()

        assert received["random_state"] == 123


class TestPreparedDataset:

    def test_initialization(self):
        dataset = PreparedDataset(
            X_train=[[1, 2]],
            y_train=[0],
            X_val=[[2, 3]],
            y_val=[1],
            X_test=[[3, 4]],
            y_test=[0]
        )

        assert dataset.X_train == [[1, 2]]
        assert dataset.y_train == [0]

    def test_mismatched_training_lengths(self):
        with pytest.raises(ValueError):
            PreparedDataset(
                X_train=[[1, 2]],
                y_train=[0, 1],
                X_val=[[2, 3]],
                y_val=[1],
                X_test=[[3, 4]],
                y_test=[0]
            )

    def test_mismatched_validation_lengths(self):
        with pytest.raises(ValueError):
            PreparedDataset(
                X_train=[[1, 2]],
                y_train=[0],
                X_val=[[2, 3]],
                y_val=[1, 0],
                X_test=[[3, 4]],
                y_test=[0]
            )

    def test_mismatched_testing_lengths(self):
        with pytest.raises(ValueError):
            PreparedDataset(
                X_train=[[1, 2]],
                y_train=[0],
                X_val=[[2, 3]],
                y_val=[1],
                X_test=[[3, 4]],
                y_test=[0, 1]
            )
