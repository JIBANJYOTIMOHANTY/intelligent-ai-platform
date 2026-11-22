
import pytest

from preprocessing.pipeline import TextPreprocessingPipeline


class TestTextPreprocessingPipeline:

    @pytest.fixture
    def pipeline(self):
        return TextPreprocessingPipeline(max_length=5)

    def test_initialization(self, pipeline):
        assert pipeline.is_fitted is False
        assert pipeline.min_freq == 1
        assert pipeline.max_vocab_size is None

    def test_invalid_max_length(self):
        with pytest.raises(ValueError):
            TextPreprocessingPipeline(max_length=0)

    def test_transform_before_fit(self, pipeline):
        with pytest.raises(RuntimeError):
            pipeline.transform(["hello world"])

    def test_fit_returns_self(self, pipeline):
        result = pipeline.fit([
            "machine learning",
            "deep learning"
        ])

        assert result is pipeline

    def test_fit_sets_fitted_state(self, pipeline):
        pipeline.fit([
            "machine learning",
            "deep learning"
        ])

        assert pipeline.is_fitted is True

    def test_fit_empty_documents(self, pipeline):
        with pytest.raises(ValueError):
            pipeline.fit([])

    def test_fit_transform_empty_documents(self, pipeline):
        with pytest.raises(ValueError):
            pipeline.fit_transform([])

    def test_fit_transform_returns_fixed_length_sequences(self, pipeline):
        documents = [
            "machine learning is interesting",
            "deep learning is powerful"
        ]

        sequences = pipeline.fit_transform(documents)

        assert len(sequences) == 2

        for sequence in sequences:
            assert len(sequence) == 5

    def test_transform_returns_fixed_length_sequences(self, pipeline):
        pipeline.fit([
            "machine learning",
            "deep learning"
        ])

        sequences = pipeline.transform([
            "machine learning",
            "deep learning"
        ])

        assert len(sequences) == 2

        for sequence in sequences:
            assert len(sequence) == 5

    def test_unknown_words_are_encoded_as_unk(self, pipeline):
        pipeline.fit(["machine learning"])

        sequences = pipeline.transform(["unknownword"])

        assert sequences[0][0] == 1

    def test_padding_uses_zero(self, pipeline):
        pipeline.fit(["machine"])

        sequences = pipeline.transform(["machine"])

        assert sequences[0][-1] == 0

    def test_long_documents_are_truncated(self):
        pipeline = TextPreprocessingPipeline(max_length=3)

        sequences = pipeline.fit_transform([
            "one two three four five"
        ])

        assert len(sequences[0]) == 3

    def test_transform_does_not_modify_vocabulary(self, pipeline):
        pipeline.fit([
            "machine learning",
            "deep learning"
        ])

        original_vocabulary = pipeline.vocabulary.word_to_index.copy()

        pipeline.transform([
            "new unknown words"
        ])

        assert pipeline.vocabulary.word_to_index == original_vocabulary

    def test_fit_transform_matches_fit_then_transform(self):
        documents = [
            "machine learning",
            "deep learning"
        ]

        first_pipeline = TextPreprocessingPipeline(max_length=5)
        second_pipeline = TextPreprocessingPipeline(max_length=5)

        first_result = first_pipeline.fit_transform(documents)

        second_pipeline.fit(documents)
        second_result = second_pipeline.transform(documents)

        assert first_result == second_result
