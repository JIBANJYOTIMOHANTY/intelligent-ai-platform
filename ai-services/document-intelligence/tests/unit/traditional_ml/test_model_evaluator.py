import numpy as np
import pytest

from traditional_ml.model_evaluator import ModelEvaluator


@pytest.fixture
def binary_labels_true():
    return [0, 1, 1, 0, 1, 0]


@pytest.fixture
def binary_labels_pred():
    return [0, 1, 0, 0, 1, 1]


@pytest.fixture
def multiclass_labels_true():
    return [0, 1, 2, 0, 1, 2, 0, 2]


@pytest.fixture
def multiclass_labels_pred():
    return [0, 1, 2, 0, 2, 1, 0, 2]


@pytest.fixture
def perfect_labels():
    return [0, 1, 2, 0, 1, 2]


@pytest.fixture
def target_names():
    return ["sport", "business", "tech"]


@pytest.fixture
def evaluator():
    return ModelEvaluator()


class TestModelEvaluator:

    # Initialization tests

    def test_default_initialization(self, evaluator):
        assert evaluator.metrics is None

    def test_initialization_with_predictions(
        self, binary_labels_true, binary_labels_pred
    ):
        evaluator = ModelEvaluator(binary_labels_true, binary_labels_pred)

        assert evaluator.metrics is not None
        assert isinstance(evaluator.metrics, dict)
        assert "accuracy" in evaluator.metrics

    def test_initialization_with_numpy_arrays(self):
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])
        evaluator = ModelEvaluator(y_true, y_pred)

        assert evaluator.metrics is not None
        assert 0.0 <= evaluator.metrics["accuracy"] <= 1.0

    def test_initialization_empty_actual_labels(self):
        with pytest.raises(ValueError, match="Actual labels can't be empty."):
            ModelEvaluator([], [0, 1])

    def test_initialization_empty_predicted_labels(self):
        with pytest.raises(ValueError, match="Predicted labels can't be empty."):
            ModelEvaluator([0, 1], [])

    def test_initialization_mismatched_lengths(self):
        with pytest.raises(
            ValueError, match="Actual and predicted labels must have equal lengths."
        ):
            ModelEvaluator([0, 1, 2], [0, 1])

    # Evaluation output tests

    def test_evaluate_returns_all_required_keys(
        self, evaluator, multiclass_labels_true, multiclass_labels_pred
    ):
        metrics = evaluator.evaluate(
            multiclass_labels_true, multiclass_labels_pred
        )

        expected_keys = {
            "accuracy",
            "precision_macro",
            "precision_weighted",
            "recall_macro",
            "recall_weighted",
            "f1_macro",
            "f1_weighted",
            "classification_report",
            "confusion_matrix"
        }

        assert isinstance(metrics, dict)
        assert expected_keys.issubset(metrics.keys())

    def test_metric_value_ranges(
        self, evaluator, multiclass_labels_true, multiclass_labels_pred
    ):
        metrics = evaluator.evaluate(
            multiclass_labels_true, multiclass_labels_pred
        )

        score_keys = [
            "accuracy",
            "precision_macro",
            "precision_weighted",
            "recall_macro",
            "recall_weighted",
            "f1_macro",
            "f1_weighted"
        ]

        for key in score_keys:
            assert isinstance(metrics[key], (float, np.floating))
            assert 0.0 <= metrics[key] <= 1.0

    def test_perfect_predictions(self, evaluator, perfect_labels):
        metrics = evaluator.evaluate(perfect_labels, perfect_labels)

        assert metrics["accuracy"] == 1.0
        assert metrics["precision_macro"] == 1.0
        assert metrics["precision_weighted"] == 1.0
        assert metrics["recall_macro"] == 1.0
        assert metrics["recall_weighted"] == 1.0
        assert metrics["f1_macro"] == 1.0
        assert metrics["f1_weighted"] == 1.0

    def test_completely_wrong_predictions(self, evaluator):
        y_true = [0, 0, 0, 0]
        y_pred = [1, 1, 1, 1]

        metrics = evaluator.evaluate(y_true, y_pred)

        assert metrics["accuracy"] == 0.0
        assert metrics["recall_macro"] == 0.0
        assert metrics["f1_macro"] == 0.0

    def test_confusion_matrix_shape_and_sum(
        self, evaluator, multiclass_labels_true, multiclass_labels_pred
    ):
        metrics = evaluator.evaluate(
            multiclass_labels_true, multiclass_labels_pred
        )

        cm = metrics["confusion_matrix"]
        assert isinstance(cm, np.ndarray)
        assert cm.shape == (3, 3)
        assert cm.sum() == len(multiclass_labels_true)

    def test_confusion_matrix_with_custom_labels(self, evaluator):
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 1, 0]
        labels = [0, 1, 2]

        metrics = evaluator.evaluate(y_true, y_pred, labels=labels)

        cm = metrics["confusion_matrix"]
        assert cm.shape == (3, 3)
        assert cm[2, :].sum() == 0
        assert cm[:, 2].sum() == 0

    def test_classification_report_with_target_names(
        self, evaluator, multiclass_labels_true, multiclass_labels_pred, target_names
    ):
        metrics = evaluator.evaluate(
            multiclass_labels_true,
            multiclass_labels_pred,
            labels=[0, 1, 2],
            target_names=target_names
        )

        report = metrics["classification_report"]
        assert isinstance(report, str)
        for name in target_names:
            assert name in report

    def test_zero_division_handling(self, evaluator):
        y_true = [0, 1, 0, 1]
        y_pred = [0, 0, 0, 0]

        metrics = evaluator.evaluate(y_true, y_pred)

        assert metrics["precision_macro"] >= 0.0
        assert metrics["precision_weighted"] >= 0.0
        assert metrics["f1_macro"] >= 0.0

    def test_numpy_array_support(self, evaluator):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])

        metrics = evaluator.evaluate(y_true, y_pred)

        assert metrics["accuracy"] == 0.75
        assert isinstance(metrics["confusion_matrix"], np.ndarray)

    def test_evaluate_without_args_when_initialized(
        self, binary_labels_true, binary_labels_pred
    ):
        evaluator = ModelEvaluator(binary_labels_true, binary_labels_pred)
        metrics = evaluator.evaluate()

        assert metrics is evaluator.metrics
        assert "accuracy" in metrics

    def test_evaluator_callable(
        self, evaluator, binary_labels_true, binary_labels_pred
    ):
        metrics = evaluator(binary_labels_true, binary_labels_pred)

        assert isinstance(metrics, dict)
        assert "accuracy" in metrics

    def test_dict_like_access(
        self, binary_labels_true, binary_labels_pred
    ):
        evaluator = ModelEvaluator(binary_labels_true, binary_labels_pred)

        assert "accuracy" in evaluator
        assert evaluator["accuracy"] == evaluator.metrics["accuracy"]
        assert evaluator.get("accuracy") == evaluator.metrics["accuracy"]
        assert evaluator.get("non_existent_key", None) is None

    def test_dict_like_access_before_evaluate_raises_key_error(self, evaluator):
        with pytest.raises(KeyError):
            _ = evaluator["accuracy"]

    # Input validation tests

    def test_empty_actual_labels(self, evaluator):
        with pytest.raises(ValueError, match="Actual labels can't be empty."):
            evaluator.evaluate([], [0, 1])

    def test_empty_predicted_labels(self, evaluator):
        with pytest.raises(ValueError, match="Predicted labels can't be empty."):
            evaluator.evaluate([0, 1], [])

    def test_mismatched_actual_and_predicted_lengths(self, evaluator):
        with pytest.raises(
            ValueError, match="Actual and predicted labels must have equal lengths."
        ):
            evaluator.evaluate([0, 1, 0], [0, 1])

    def test_empty_numpy_arrays(self, evaluator):
        with pytest.raises(ValueError, match="Actual labels can't be empty."):
            evaluator.evaluate(np.array([]), np.array([]))

    def test_evaluate_without_initialization_or_args_raises_error(self, evaluator):
        with pytest.raises(ValueError, match="Actual labels can't be empty."):
            evaluator.evaluate()
