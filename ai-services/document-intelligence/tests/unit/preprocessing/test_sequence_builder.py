import pytest

from preprocessing.sequence_builder import SequenceBuilder

class TestSequenceBuilder:
    def test_initialization(self):
        builder = SequenceBuilder(max_length=5)
        assert builder.max_length == 5
        assert builder.padding_value == 0

    def test_invalid_max_length(self):
        with pytest.raises(ValueError): SequenceBuilder(max_length=0)
        with pytest.raises(ValueError): SequenceBuilder(max_length=-1)

    def test_pad_sequence(self):
        builder = SequenceBuilder(max_length=5)
        sequence = [10, 20, 30]
        result = builder.pad_sequence(sequence)
        assert result == [10, 20, 30, 0, 0]

    def test_truncate_sequence(self):
        builder = SequenceBuilder(max_length=3)
        sequence = [10, 20, 30, 40, 50]
        result = builder.truncate_sequence(sequence)
        assert result == [10, 20, 30]

    def test_build_sequence_with_padding(self):
        builder = SequenceBuilder(max_length=5)
        sequence = [10, 20]
        result = builder.build_sequence(sequence)
        assert result == [10, 20, 0, 0, 0]

    def test_build_sequence_with_truncation(self):
        builder = SequenceBuilder(max_length=3)
        sequence = [10, 20, 30, 40]
        result = builder.build_sequence(sequence)
        assert result == [10, 20, 30]

    def test_build_sequence_exact_length(self):
        builder = SequenceBuilder(max_length=3)
        sequence = [10, 20, 30]
        result = builder.build_sequence(sequence)
        assert result == [10, 20, 30]

    def test_empty_sequence(self):
        builder = SequenceBuilder(max_length=4)
        result = builder.build_sequence([])
        assert result == [0, 0, 0, 0]

    def test_custom_padding_value(self):
        builder = SequenceBuilder(
            max_length=4,
            padding_value=99
        )
        result = builder.build_sequence([10, 20])
        assert result == [10, 20, 99, 99]

    def test_build_batch(self):
        builder = SequenceBuilder(max_length=4)
        sequences = [
            [10, 20],
            [30, 40, 50, 60],
            [70]
        ]
        result = builder.build_batch(sequences)
        assert result == [
            [10, 20, 0, 0],
            [30, 40, 50, 60],
            [70, 0, 0, 0]
        ]

    def test_original_sequence_unchanged(self):
        builder = SequenceBuilder(max_length=4)
        sequence = [10, 20]
        builder.build_sequence(sequence)
        assert sequence == [10, 20]