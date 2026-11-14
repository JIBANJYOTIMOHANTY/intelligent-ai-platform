import re
import pandas as pd

class TextCleaner:
    """
    Handles text cleaning operations for NLP preprocessing.
    """

    def clean_text(self, text:str) -> str:
        """
        Clean an individual text document.
        """

        # 1. Handle missing values
        if pd.isna(text): return ""

        text = str(text)

        # 2. Convert text to lowercase
        text = text.lower()

        # 3. Expand common contractions
        contractions = {
            "can't": "cannot",
            "won't": "will not",
            "don't": "do not",
            "doesn't": "does not",
            "didn't": "did not",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "haven't": "have not",
            "hasn't": "has not",
            "hadn't": "had not",
            "wouldn't": "would not",
            "shouldn't": "should not",
            "couldn't": "could not"
        }

        for key, value in contractions.items():
            text = text.replace(key, value)

        # 4. Remove URLs
        text = re.sub(
            r'https?://\S+|www\.\S+',
            '',
            text
        )

        # 5. Remove HTML tags
        text = re.sub(r'<.*?>', '', text)

        # 6. Remove special characters
        text = re.sub(r'[^a-z0-9\s]', ' ', text)

        # 7. Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # 8. Remove leading and trailing whitespace
        text = text.strip()

        return text