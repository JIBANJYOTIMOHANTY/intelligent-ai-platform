import re

class WordTokenizer:
    """
    Converts cleaned text into individual word tokens.
    """
    
    def tokenize(self, text : str) -> list[str]:
        """
        Split a text document into word-level tokens.
        """

        if not isinstance(text, str): return []

        text = text.lower()

        tokens = re.findall(r"\b[a-z0-9]+\b", text)

        return tokens