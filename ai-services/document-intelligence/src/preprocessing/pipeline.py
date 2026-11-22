from preprocessing.text_cleaner import TextCleaner
from preprocessing.tokenizer import WordTokenizer
from preprocessing.vocabulary import Vocabulary
from preprocessing.sequence_builder import SequenceBuilder

class TextPreprocessingPipeline:
    """
    Integrate text cleaning, tokenization, vocabulary encoding, and sequence building. 
    """
    
    def __init__(self, max_length : int, min_freq : int = 1, max_vocab_size : int | None = None):
        if max_length <= 0:
            raise ValueError("max_length must be greater than 0")

        self.cleaner = TextCleaner()
        self.tokenizer = WordTokenizer()
        self.vocabulary = Vocabulary()
        self.sequence_builder = SequenceBuilder(max_length=max_length, padding_value=0)

        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.is_fitted = False

    def _tokenize_documents(self,documents : list[str]) -> list[list[str]]:
        """
        Clean and tokenize all documents.
        """
    
        tokenized_documents = []

        for document in documents:
            cleaned_text = self.cleaner.clean_text(document)
            tokens = self.tokenizer.tokenize(cleaned_text)
            tokenized_documents.append(tokens)

        return tokenized_documents

    def fit(self, documents : list[str]):
        """
        Build vocabulary from documents
        """

        if not documents:
            raise ValueError("No documents provided")
        
        tokenized_documents = self._tokenize_documents(documents)

        self.vocabulary.build_vocab(tokenized_documents, min_freq=self.min_freq, max_vocab_size=self.max_vocab_size)

        self.is_fitted = True

        return self

    def _encode_documents(self, tokenized_documents : list[list[str]]) -> list[list[int]]:
        """
        Encode tokenized documents to integer sequences.
        """

        encoded_documents = []

        for tokens in tokenized_documents:
            encoded_sequence = self.vocabulary.encode(tokens)
            fixed_sequence = self.sequence_builder.build_sequence(encoded_sequence)
            encoded_documents.append(fixed_sequence)
        
        return encoded_documents

    def transform(self, documents : list[str]) -> list[list[int]]:
        """
        Clean, tokenize and encode documents using fitted vocabulary and sequence builder.
        """
        
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before transforming documents")

        tokenized_documents = self._tokenize_documents(documents)
        
        return self._encode_documents(tokenized_documents)

    def fit_transform(self, documents : list[str]) -> list[list[int]]:
        """
        Fit pipeline on documents and then transform them.
        """

        if not documents:
            raise ValueError("Documents can't be empty.")

        tokenized_documents = self._tokenize_documents(documents)
        
        self.vocabulary.build_vocab(tokenized_documents, min_freq=self.min_freq, max_vocab_size=self.max_vocab_size)

        self.is_fitted = True

        return self._encode_documents(tokenized_documents)