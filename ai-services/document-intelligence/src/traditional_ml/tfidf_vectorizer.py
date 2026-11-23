from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

class TfidfFeatureExtractor:
    """
    This class extracts features from documents using Term Frequency-Inverse Document Frequency.
    """
    
    def __init__(self, max_features : int | None = None, min_df : int | float = 1, max_df : int | float = 1.0, ngram_range : tuple[int, int] = (1, 1)) -> None:
        
        if max_features is not None and max_features <=0:
            raise ValueError("max_features must be greater than 0")
        
        if(
            len(ngram_range) != 2 
            or ngram_range[0] <= 0
            or ngram_range[0] > ngram_range[1]
        ):
            raise ValueError("ngram_range must be a tuple of two positive integers")

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range
        )

        self.is_fitted = False

    def fit(self, documents: list[str]) -> "TfidfFeatureExtractor":
        """Learn the vocabulary and IDF weights from documents."""

        if not documents:
            raise ValueError("Documents can't be empty.")
        
        self.vectorizer.fit(documents)
        self.is_fitted = True

        return self

    def transform(self, documents: list[str]) -> csr_matrix:
        """Transform documents using the learned vocabulary."""

        if not self.is_fitted:
            raise RuntimeError("The feature extractor must be fitted before transforming documents.")   

        if not documents:
            raise ValueError("Documents can't be empty.")

        return self.vectorizer.transform(documents)

    def fit_transform(self, documents : list[str]) -> csr_matrix:
        """Fit the extractor and transform documents."""

        if not documents:
            raise ValueError("Documents can't be empty.")

        features = self.vectorizer.fit_transform(documents)
        self.is_fitted = True

        return features

    @property
    def vocabulary_size(self) -> int:
        """Return the number of learned vocabulary terms."""

        if not self.is_fitted:
            raise RuntimeError("TF-IDF extractor has not been fitted.")

        return len(self.vectorizer.vocabulary_)

    def get_feature_names(self) -> list[str]:
        """Return the learned vocabulary terms."""

        if not self.is_fitted:
            raise RuntimeError("TF-IDF extractor has not been fitted.")

        return self.vectorizer.get_feature_names_out().tolist()