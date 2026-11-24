
import numpy as np
import pytest

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from traditional_ml.model_trainer import TraditionalMLTrainer


@pytest.fixture
def documents():
    return [
        "football team won the match",
        "player scored a goal",
        "company reported strong profits",
        "business announced new technology",
        "football players trained hard",
        "company increased its revenue"
    ]


@pytest.fixture
def labels():
    return [0, 0, 1, 1, 0, 1]


@pytest.fixture
def features(documents):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(documents)


@pytest.fixture
def empty_features():
    return csr_matrix((0, 5))


class TestTraditionalMLTrainer:

    # Initialization tests

    def test_default_initialization(self):
        trainer = TraditionalMLTrainer()

        assert trainer.model_type == "logistic_regression"
        assert isinstance(trainer.model, LogisticRegression)

    def test_naive_bayes_initialization(self):
        trainer = TraditionalMLTrainer(model_type="naive_bayes")

        assert isinstance(trainer.model, MultinomialNB)

    def test_logistic_regression_initialization(self):
        trainer = TraditionalMLTrainer(
            model_type="logistic_regression"
        )

        assert isinstance(trainer.model, LogisticRegression)

    def test_linear_svm_initialization(self):
        trainer = TraditionalMLTrainer(model_type="linear_svm")

        assert isinstance(trainer.model, LinearSVC)

    def test_invalid_model_type(self):
        with pytest.raises(ValueError):
            TraditionalMLTrainer(model_type="random_forest")

    def test_invalid_max_iter(self):
        with pytest.raises(ValueError):
            TraditionalMLTrainer(max_iter=0)

        with pytest.raises(ValueError):
            TraditionalMLTrainer(max_iter=-1)

    def test_initial_fitted_status(self):
        trainer = TraditionalMLTrainer()

        assert trainer.is_fitted is False

    # Training tests

    def test_successful_training(self, features, labels):
        trainer = TraditionalMLTrainer()

        trainer.fit(features, labels)

        assert trainer.is_fitted is True

    @pytest.mark.parametrize(
        "model_type",
        ["naive_bayes", "logistic_regression", "linear_svm"]
    )
    def test_training_all_supported_models(
        self, model_type, features, labels
    ):
        trainer = TraditionalMLTrainer(model_type=model_type)

        trainer.fit(features, labels)

        assert trainer.is_fitted is True

    def test_fit_returns_self(self, features, labels):
        trainer = TraditionalMLTrainer()

        result = trainer.fit(features, labels)

        assert result is trainer

    def test_empty_training_features(self, empty_features):
        trainer = TraditionalMLTrainer()

        with pytest.raises(ValueError):
            trainer.fit(empty_features, [])

    def test_empty_training_labels(self, features):
        trainer = TraditionalMLTrainer()

        with pytest.raises(ValueError):
            trainer.fit(features, [])

    def test_mismatched_feature_and_label_count(
        self, features, labels
    ):
        trainer = TraditionalMLTrainer()

        with pytest.raises(ValueError):
            trainer.fit(features, labels[:-1])

    # Prediction tests

    def test_prediction_before_training(self, features):
        trainer = TraditionalMLTrainer()

        with pytest.raises(RuntimeError):
            trainer.predict(features)

    def test_successful_prediction(self, features, labels):
        trainer = TraditionalMLTrainer()

        trainer.fit(features, labels)

        predictions = trainer.predict(features)

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (len(labels),)

    def test_predictions_contain_known_classes(self, features, labels):
        trainer = TraditionalMLTrainer()

        trainer.fit(features, labels)

        predictions = trainer.predict(features)

        assert set(predictions).issubset(set(labels))

    def test_empty_prediction_features(self, features, labels):
        trainer = TraditionalMLTrainer()
        trainer.fit(features, labels)

        with pytest.raises(ValueError):
            trainer.predict(csr_matrix((0, features.shape[1])))
