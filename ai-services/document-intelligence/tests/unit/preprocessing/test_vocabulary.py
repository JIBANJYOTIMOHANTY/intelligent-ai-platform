from preprocessing.vocabulary import Vocabulary

def test_vocabulary():
    documents = [
        ["the", "company", "new", "technology"],
        ["the", "company", "new", "products"],
        ["technology", "companies", "develop", "new", "products"]
    ]

    vocabulary = Vocabulary()

    vocabulary.build_vocab(documents)

    # Test 1: Special tokens
    assert vocabulary.word_to_index["<PAD>"] == 0
    assert vocabulary.word_to_index["<UNK>"] == 1

    # Test 2: Vocabulary contains unique words
    assert len(vocabulary) == len(vocabulary.word_to_index)

    # Test 3: Frequent word receives an earlier ID
    assert vocabulary.word_to_index["new"] == 2

    # Test 4: Word encoding
    tokens = ["new", "company", "technology"]

    encoded = vocabulary.encode(tokens)

    assert encoded == [
        vocabulary.word_to_index["new"],
        vocabulary.word_to_index["company"],
        vocabulary.word_to_index["technology"]
    ]

    # Test 5: Unknown words
    encoded = vocabulary.encode(
        ["new", "unknownword"]
    )

    assert encoded == [2, 1]

    # Test 6: Decoding
    decoded = vocabulary.decode(
        [2, 1]
    )

    assert decoded == ["new", "<UNK>"]

    # Test 7: Minimum frequency
    filtered_vocabulary = Vocabulary()

    filtered_vocabulary.build_vocab(
        documents,
        min_freq=2
    )

    assert "new" in filtered_vocabulary.word_to_index
    assert "develop" not in filtered_vocabulary.word_to_index

    # Test 8: Maximum vocabulary size
    limited_vocabulary = Vocabulary()

    limited_vocabulary.build_vocab(
        documents,
        max_vocab_size=5
    )

    assert len(limited_vocabulary) == 5

    # Test 9: Invalid index
    decoded = vocabulary.decode([999])

    assert decoded == ["<UNK>"]

    print("All vocabulary tests passed!")


if __name__ == "__main__":
    test_vocabulary()