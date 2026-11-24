from typing import Literal
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import numpy as np

ModelType = Literal[
    "naive_bayes",
    "logistic_regression",
    "linear_svm"
]

class TraditionalMLTrainer:
    """
    Trains a traditional ML model for text classification.
    """

    SUPPORTED_MODELS = {
        "naive_bayes",
        "logistic_regression",
        "linear_svm"
    }

    def __init__(self, model_type : ModelType = "logistic_regression", random_state: int = 32, max_iter : int = 1000) -> None:
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Choose from {sorted(self.SUPPORTED_MODELS)}."
            )
        
        if max_iter <= 0:
            raise ValueError("max_iter must be greater than zero.")

        self.model_type = model_type
        self.random_state = random_state
        self.max_iter = max_iter

        self.model = self._create_model()
        self.is_fitted = False

    def _create_model(self):
        """
        Creates a new instance of the selected model.
        """

        if self.model_type == 'naive_bayes':
            return MultinomialNB()

        if self.model_type == 'logistic_regression':
            return LogisticRegression(
                max_iter=self.max_iter,
                random_state=self.random_state
            )

        if self.model_type == "linear_svm":
            return LinearSVC(
                max_iter=self.max_iter,
                random_state=self.random_state
            )
        
        raise ValueError(f"Unsupported model type: {self.model_type}")

    def fit(self, features: csr_matrix, labels: list[int] | np.ndarray) -> "TraditionalMLTrainer":
        """
        Trains the selected model on the given features and labels.
        """
        
        if features.shape[0] == 0:
            raise ValueError("Training features cannot be empty.")

        if len(labels) == 0:
            raise ValueError("Labels cannot be empty.")

        if features.shape[0] != len(labels):
            raise ValueError("Number of features must match number of labels.")

        self.model.fit(features, labels)
        self.is_fitted = True

        return self

    def predict(self, features: csr_matrix) -> np.ndarray:
        """
        Makes predictions using the trained model.
        
        """
        
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        if features.shape[0] == 0:
            raise ValueError("Features cannot be empty")
        
        return self.model.predict(features)
