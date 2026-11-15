import sys
from pathlib import Path

# Add project source directory to Python path
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from preprocessing.tokenizer import WordTokenizer

def test_word_tokenizer():
    tokenizer = WordTokenizer()

    # Test 1: Normal sentence
    result = tokenizer.tokenize("Hello World")

    assert result == ["hello", "world"]

    # Test 2: Punctuation
    result = tokenizer.tokenize(
        "Machine learning is interesting!"
    )

    assert result == [
        "machine",
        "learning",
        "is",
        "interesting"
    ]

    # Test 3: Numbers
    result = tokenizer.tokenize(
        "The company earned 250 million dollars"
    )

    assert result == [
        "the",
        "company",
        "earned",
        "250",
        "million",
        "dollars"
    ]

    # Test 4: Empty text
    result = tokenizer.tokenize("")

    assert result == []

    # Test 5: Missing input
    result = tokenizer.tokenize(None)

    assert result == []

    # Test 6: Mixed uppercase and lowercase
    result = tokenizer.tokenize(
        "BBC NEWS Technology"
    )

    assert result == [
        "bbc",
        "news",
        "technology"
    ]

    print("All tokenizer tests passed!")

if __name__ == "__main__":
    test_word_tokenizer()