import pytest

from src.data.label_encoder import CategoryLabelEncoder

class TestCategoryLabelEncoder:
    @pytest.fixture
    def encoder(self):
        return CategoryLabelEncoder()

    @pytest.fixture
    def categories(self):
        return [
            "business",
            "entertainment",
            "politics",
            "sport",
            "tech"
        ]

    def test_initialization(self, encoder):
        assert encoder.is_fitted is False
    
    def test_fit(self, encoder, categories):
        result = encoder.fit(categories)

        assert encoder.is_fitted is True
        assert result is encoder

    def test_classes(self, encoder, categories):
        encoder.fit(categories)

        assert encoder.class_ == [
            "business",
            "entertainment",
            "politics",
            "sport",
            "tech"
        ]

    def test_transform(self, encoder, categories):
        encoder.fit(categories)

        result = encoder.transform(categories)

        assert result == [0, 1, 2, 3, 4]

    def test_inverse_transform(self, encoder, categories):
        encoder.fit(categories)

        encoded = encoder.transform(categories)

        result = encoder.inverse_transform(encoded)

        assert result == categories

    def test_fit_transform(self, encoder, categories):
        result = encoder.fit_transform(categories)

        assert result == [0, 1, 2, 3, 4]
        assert encoder.is_fitted is True

    def test_transform_before_fit(self, encoder):
        with pytest.raises(RuntimeError):
            encoder.transform(["business"])

    def test_inverse_transform_before_fit(self, encoder):
        with pytest.raises(RuntimeError):
            encoder.inverse_transform([0])

    def test_classes_before_fit(self, encoder):
        with pytest.raises(RuntimeError):
            encoder.classes_

    def test_unknown_category(self, encoder, categories):
        encoder.fit(categories)

        with pytest.raises(ValueError):
            encoder.transform(["unknown_category"])

    def test_empty_labels(self, encoder):
        with pytest.raises(ValueError):
            encoder.fit_transform([])