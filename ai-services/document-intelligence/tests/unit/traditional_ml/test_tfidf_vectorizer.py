
import pytest
from scipy.sparse import csr_matrix

from traditional_ml.tfidf_vectorizer import TfidfFeatureExtractor


@pytest.fixture
def documents():
    return [
        "machine learning is interesting",
        "deep learning uses neural networks",
        "machine learning uses algorithms",
        "neural networks are powerful"
    ]


@pytest.fixture
def extractor():
    return TfidfFeatureExtractor()


class TestTfidfFeatureExtractor:

    def test_initialization(self, extractor):
        assert extractor.is_fitted is False

    def test_fit(self, extractor, documents):
        result = extractor.fit(documents)

        assert extractor.is_fitted is True
        assert result is extractor

    def test_fit_transform(self, extractor, documents):
        features = extractor.fit_transform(documents)

        assert extractor.is_fitted is True
        assert isinstance(features, csr_matrix)
        assert features.shape[0] == len(documents)

    def test_transform(self, extractor, documents):
        extractor.fit(documents)

        new_documents = ["machine learning algorithms"]
        features = extractor.transform(new_documents)

        assert isinstance(features, csr_matrix)
        assert features.shape[0] == 1

    def test_transform_before_fit(self, extractor):
        with pytest.raises(RuntimeError):
            extractor.transform(["machine learning"])

    def test_empty_documents_fit(self, extractor):
        with pytest.raises(ValueError):
            extractor.fit([])

    def test_empty_documents_transform(self, extractor, documents):
        extractor.fit(documents)

        with pytest.raises(ValueError):
            extractor.transform([])

    def test_empty_documents_fit_transform(self, extractor):
        with pytest.raises(ValueError):
            extractor.fit_transform([])

    def test_vocabulary_size_before_fit(self, extractor):
        with pytest.raises(RuntimeError):
            extractor.vocabulary_size

    def test_vocabulary_size(self, extractor, documents):
        extractor.fit(documents)

        assert extractor.vocabulary_size > 0

    def test_feature_names_before_fit(self, extractor):
        with pytest.raises(RuntimeError):
            extractor.get_feature_names()

    def test_feature_names(self, extractor, documents):
        extractor.fit(documents)

        feature_names = extractor.get_feature_names()

        assert isinstance(feature_names, list)
        assert "machine" in feature_names
        assert "learning" in feature_names

    def test_max_features(self, documents):
        extractor = TfidfFeatureExtractor(max_features=3)

        features = extractor.fit_transform(documents)

        assert features.shape[1] <= 3
        assert extractor.vocabulary_size <= 3

    def test_invalid_max_features(self):
        with pytest.raises(ValueError):
            TfidfFeatureExtractor(max_features=0)

        with pytest.raises(ValueError):
            TfidfFeatureExtractor(max_features=-1)

    def test_invalid_ngram_range(self):
        with pytest.raises(ValueError):
            TfidfFeatureExtractor(ngram_range=(0, 1))

        with pytest.raises(ValueError):
            TfidfFeatureExtractor(ngram_range=(2, 1))

        with pytest.raises(ValueError):
            TfidfFeatureExtractor(ngram_range=(1, 2, 3))

    def test_ngram_range(self, documents):
        extractor = TfidfFeatureExtractor(ngram_range=(1, 2))

        extractor.fit(documents)

        feature_names = extractor.get_feature_names()

        assert "machine learning" in feature_names

    def test_transform_does_not_change_vocabulary(self, extractor, documents):
        extractor.fit(documents)

        original_vocabulary = set(extractor.get_feature_names())

        extractor.transform(["completely unknown vocabulary"])

        updated_vocabulary = set(extractor.get_feature_names())

        assert original_vocabulary == updated_vocabulary
