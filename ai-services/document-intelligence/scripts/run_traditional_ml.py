from data.dataset_loader import BBCDatasetLoader
from data.label_encoder import CategoryLabelEncoder

from traditional_ml.tfidf_vectorizer import TfidfFeatureExtractor
from traditional_ml.model_trainer import TraditionalMLTrainer
from traditional_ml.model_evaluator import ModelEvaluator

def main():

    # 1. Load and split dataset
    dataset_loader = BBCDatasetLoader()
    dataset = dataset_loader.load_dataset()

    train_df, val_df, test_df = dataset_loader.split_dataset(dataset, random_state=42)

    X_train_text = train_df['text'].tolist()
    X_val_text = val_df['text'].tolist()
    X_test_text = test_df['text'].tolist()

    y_train_labels = train_df['category'].tolist()
    y_val_labels = val_df['category'].tolist()
    y_test_labels = test_df['category'].tolist()

    print("Dataset loaded successfully.")
    print(f"Training samples: {len(X_train_text)}")
    print(f"Validation samples: {len(X_val_text)}")
    print(f"Testing samples: {len(X_test_text)}")

    # 2. Encode category labels
    label_encoder = CategoryLabelEncoder()
    y_train = label_encoder.fit_transform(y_train_labels)
    y_val = label_encoder.transform(y_val_labels)
    y_test = label_encoder.transform(y_test_labels)

    labels = list(range(len(label_encoder.classes_)))
    target_names = list(label_encoder.classes_)

    print("\nCategories:", target_names)

    # 3. Extract TF-IDF features
    feature_extractor = TfidfFeatureExtractor()
    X_train = feature_extractor.fit_transform(X_train_text)
    X_val = feature_extractor.transform(X_val_text)
    X_test = feature_extractor.transform(X_test_text)

    print("\nTF-IDF feature extraction completed.")
    print("Training feature shape:", X_train.shape)
    print("Validation feature shape:", X_val.shape)
    print("Testing feature shape:", X_test.shape)

    # 4. Train and compare models
    model_types = [
        "naive_bayes",
        "logistic_regression",
        "linear_svm"
    ]

    evaluator = ModelEvaluator()
    results = {}

    for model_type in model_types:
        print(f"\nTraining {model_type}...")

        trainer = TraditionalMLTrainer(model_type=model_type, random_state=42, max_iter=1000)
        trainer.fit(X_train, y_train)

        val_predictions = trainer.predict(X_val)
        val_metrics = evaluator.evaluate(y_val, val_predictions, labels=labels, target_names=target_names)
        results[model_type] = {
            "trainer" : trainer,
            "metrics" : val_metrics
        }

        print(f"Validation accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Validation macro F1: {val_metrics['f1_macro']:.4f}")
        print(f"Validation weighted F1: {val_metrics['f1_weighted']:.4f}")

    best_model_type = max(results, key = lambda model : results[model]['metrics']['f1_macro'])
    best_trainer = results[best_model_type]['trainer']

    print("\n" + "=" * 50)
    print("BEST MODEL:", best_model_type)
    print("=" * 50)

    test_predictions = best_trainer.predict(X_test)

    test_metrics = evaluator.evaluate(y_test, test_predictions, labels=labels, target_names=target_names)

    print("\nFinal Test Results")
    print("-" * 50)

    for metric_name in [
        "accuracy",
        "precision_macro",
        "precision_weighted",
        "recall_macro",
        "recall_weighted",
        "f1_macro",
        "f1_weighted "
    ]:
        print(f"{metric_name}: {test_metrics[metric_name]:.4f}")

    print("\nClassification Report:")
    print(test_metrics["classification_report"])

    print("\nConfusion Matrix:")
    print(test_metrics["confusion_matrix"])


if __name__ == "__main__":
    main()
