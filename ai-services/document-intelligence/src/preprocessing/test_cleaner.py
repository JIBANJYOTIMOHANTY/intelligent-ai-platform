import sys
from pathlib import Path
import pandas as pd

# Add project source directory to Python path
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from preprocessing.text_cleaner import TextCleaner

def test_test_cleaner():
    cleaner = TextCleaner()

    # Test 1: Basic text cleaning
    result = cleaner.clean_text("HELLO WORLD!")
    assert result == "hello world"

    assert result == "hello world"

    # Test 2: Contractions
    result = cleaner.clean_text(
        "The company isn't performing well."
    )
    assert result == "the company is not performing well"

    # Test 3: URLs
    result = cleaner.clean_text(
        "Visit https://example.com today"
    )
    assert result == "visit today"

    # Test 4: HTML tags
    result = cleaner.clean_text(
        "<p>Hello World</p>"
    )
    assert result == "hello world"

    # Test 5: Missing values
    result = cleaner.clean_text(
        pd.NA
    )
    assert result == ""

    # Test 6: Extra whitespace
    result = cleaner.clean_text(
        "  Hello     World  "
    )
    assert result == "hello world"

    print("All text cleaner tests passed!")
  
if __name__ == "__main__":
    test_test_cleaner()