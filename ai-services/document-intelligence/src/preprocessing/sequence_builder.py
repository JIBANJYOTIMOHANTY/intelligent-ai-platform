class SequenceBuilder:
    """
    Convert variable-length encoded sequences
    into fixed-length sequences using padding
    and truncation.
    """

    def __init__(
        self,
        max_length : int,
        padding_value : int = 0
    ):
        if max_length <= 0:
            raise ValueError("Max length must be greater than zero.")
        
        self.max_length = max_length
        self.padding_value = padding_value

    def pad_sequence(self, sequence : list[int]) -> list[int]:
        """
        Add padding tokens to reach max_length.
        """

        padding_length = self.max_length - len(sequence)
        return sequence + [self.padding_value] * padding_length

    def truncate_sequence(self, sequence: list[int]) -> list[int]:
        """
        Remove tokens exceeding max_length.
        """

        return sequence[:self.max_length]

    def build_sequence(self, sequence:list[int]) -> list[int]:
        """
        Convert a sequence into a fixed-length sequence.
        """

        if len(sequence) > self.max_length:
            return self.truncate_sequence(sequence)

        return self.pad_sequence(sequence)

    def build_batch(self, sequences : list[list[int]]) -> list[list[int]]:
        """
        Process multiple sequences.
        """

        return [
            self.build_sequence(sequence)
            for sequence in sequences
        ]