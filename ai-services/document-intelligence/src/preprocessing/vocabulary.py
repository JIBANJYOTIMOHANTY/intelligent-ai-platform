from collections import Counter

class Vocabulary:
    """
    Builds a word-to-index and index-to-word mapping
    from tokenized documents.
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(self):
        self.word_to_index = {
            self.PAD_TOKEN : 0,
            self.UNK_TOKEN : 1
        }

        self.index_to_word = {
            self.PAD_TOKEN,
            self.UNK_TOKEN
        }
    
    def build_vocab(
        self, 
        tokenized_documents: list[list[str]],
        min_freq : int = 1, 
        max_vocab_size : int | None = None
    ) -> None:
        
        if(min_freq < 1): 
            raise ValueError("min_freq must be at least 1")

        if(max_vocab_size is not None and max_vocab_size < 2):
            raise ValueError("max_vocab_size must be at least 2")

        # Count word frequencies
        word_counts = Counter()

        for tokens in tokenized_documents:
            word_counts.update(
                token for token in tokens
                if isinstance(token, str)
                and token 
                and token not in (self.PAD_TOKEN, self.UNK_TOKEN)
            )

        # Sort by frequency (descending), then alphabetically
        sorted_words = sorted(
            word_counts.items(),
            key = lambda item : (-item[1], item[0])
        )

        # Filter words by minimum frequency
        filtered_words = [
            word
            for word, count in sorted_words
            if count >= min_freq
        ]

        # Limit vocabulary size, including special tokens
        if max_vocab_size is not None:
            available_slots = max_vocab_size - 2
            filtered_words = filtered_words[:available_slots]

        # Reset mappings
        self.word_to_index = {
            self.PAD_TOKEN : 0,
            self.UNK_TOKEN : 1
        }

        self.index_to_word = [
            self.PAD_TOKEN,
            self.UNK_TOKEN
        ]

        # Assign unique IDs to words
        for word in filtered_words:
            index = len(self.index_to_word)
            self.word_to_index[word] = index
            self.index_to_word.append(word)

    def encode(self, tokens: list[str]) -> list[int]:
        """
        Convert tokens into integer IDs.
        """

        unknown_index = self.word_to_index[self.UNK_TOKEN]
        return [
            self.word_to_index.get(token, unknown_index)
            for token in tokens
        ]

    def decode(self, indices: list[int]) -> list[str]:
        """
        Convert integer IDs back into tokens.
        """

        return [
            self.index_to_word[index]
            if 0 <= index < len(self.index_to_word)
            else self.UNK_TOKEN
            for index in indices
        ]

    def __len__(self) -> int:
        """
        Return total vocabulary size.
        """
        return len(self.index_to_word)