from sklearn.preprocessing import LabelEncoder

class CategoryLabelEncoder:
    """
    Encode and decode categorical labels.
    """

    def __init__(self):
        self.encoder = LabelEncoder()
        self.is_fitted = False

    def fit(self, labels: list[str]):
        """
        Learn category-to-index mappings.
        """

        if not labels:
            raise ValueError("Labels Can't Be Empty")

        self.encoder.fit(labels)
        self.is_fitted = True

        return self

    def transform(self, labels : list[str]) -> list[int]:
        """
        Convert category labels into numerical values.
        """

        if not self.is_fitted:
            raise RuntimeError("LabelEncoder has not been fitted yet. Please fit the encoder first.")

        return self.encoder.transform(labels).tolist()

    def fit_transform(self, labels: list[str]) -> list[int]:
        """
        Fit the encoder and transform the labels.
        """

        self.fit(labels)

        return self.transform(labels)

    def inverse_transform(self, encoded_labels : list[int]) -> list[str]:
        """
        Convert numerical labels back into categories.
        """

        if not self.is_fitted:
            raise RuntimeError("LabelEncoder has not been fitted yet. Please fit the encoder first.")

        return self.encoder.inverse_transform(encoded_labels).tolist()

    @property
    def class_(self) -> list[str]:
        """
        Return the learned categories.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "LabelEncoder has not been fitted yet. Please fit the encoder first."
            )

        return self.encoder.classes_.tolist()


    @property
    def classes_(self) -> list[str]:
        """
        Return the learned categories.
        """

        return self.class_