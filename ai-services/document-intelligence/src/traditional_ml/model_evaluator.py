import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix, 
    classification_report
)

class ModelEvaluator:
    """
    This class evaluate the model performance on the test data.
    """

    def __init__(
        self,
        y_true: list[int] | np.ndarray | None = None,
        y_pred: list[int] | np.ndarray | None = None,
        labels: list[int] | None = None,
        target_names: list[str] | None = None
    ) -> None:
        """
        Calculates classification evaluation metrics.
        
        Args:
            y_true : Ground truth labels
            y_pred : Predicted labels
            labels : List of labels
            target_names : List of target names
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.labels = labels
        self.target_names = target_names
        self.metrics: dict | None = None

        if y_true is not None or y_pred is not None:
            if y_true is None:
                raise ValueError("Actual labels can't be empty.")
            if y_pred is None:
                raise ValueError("Predicted labels can't be empty.")
            self.metrics = self.evaluate(y_true, y_pred, labels=labels, target_names=target_names)

    def evaluate(
        self,
        y_true: list[int] | np.ndarray | None = None,
        y_pred: list[int] | np.ndarray | None = None,
        labels: list[int] | None = None,
        target_names: list[str] | None = None
    ) -> dict:
        """
        Calculates classification evaluation metrics.
        
        Args:
            y_true : Ground truth labels
            y_pred : Predicted labels
            labels : List of labels
            target_names : List of target names

        Returns:
            dict: Evaluation metrics dictionary
        """
        if y_true is None:
            y_true = self.y_true
        if y_pred is None:
            y_pred = self.y_pred
        if labels is None:
            labels = self.labels
        if target_names is None:
            target_names = self.target_names

        if y_true is None or len(y_true) == 0:
            raise ValueError("Actual labels can't be empty.")

        if y_pred is None or len(y_pred) == 0:
            raise ValueError("Predicted labels can't be empty.")

        if len(y_true) != len(y_pred):
            raise ValueError("Actual and predicted labels must have equal lengths.")

        metrics = {
            "accuracy" : accuracy_score(y_true, y_pred),

            "precision_macro" : precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),

            "precision_weighted" : precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),

            "recall_macro" : recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),

            "recall_weighted" : recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),

            "f1_macro" : f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
            
            "f1_weighted" : f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),

            "classification_report" : classification_report(y_true, y_pred, labels=labels, target_names=target_names, zero_division=0),

            "confusion_matrix" : confusion_matrix(y_true, y_pred, labels=labels)
        }

        self.metrics = metrics
        return metrics

    def __getitem__(self, item: str):
        if self.metrics is None:
            raise KeyError("Metrics have not been evaluated yet.")
        return self.metrics[item]

    def __contains__(self, item: str) -> bool:
        if self.metrics is None:
            return False
        return item in self.metrics

    def get(self, item: str, default=None):
        if self.metrics is None:
            return default
        return self.metrics.get(item, default)

    def __call__(
        self,
        y_true: list[int] | np.ndarray | None = None,
        y_pred: list[int] | np.ndarray | None = None,
        labels: list[int] | None = None,
        target_names: list[str] | None = None
    ) -> dict:
        return self.evaluate(y_true, y_pred, labels=labels, target_names=target_names)