from dataclasses import dataclass

from data.dataset_loader import BBCDatasetLoader
from data.label_encoder import CategoryLabelEncoder
from preprocessing.pipeline import TextPreprocessingPipeline

@dataclass
class PreparedDataset:
    """
    Prepare BBC dataset for training and testing.
    """
    
    X_train : list[list[int]]
    y_train : list[int]

    X_val : list[list[int]]
    y_val : list[int]

    X_test : list[list[int]]
    y_test : list[int]

    def __post_init__(self):
        """
        Ensure each feature sequence has a corresponding label.
        """

        splits = [
            ("training", self.X_train, self.y_train),
            ("validation", self.X_val, self.y_val),
            ("testing", self.X_test, self.y_test)
        ]

        for split_name, features, labels in splits:
            if len(features) != len(labels):
                raise ValueError(
                    f"{split_name} features and labels must have equal lengths."
                )

class BBCDatasetBuilder:
    """
    Build BBC dataset for training and testing.

    Resposibilites
    1. Load BBC dataset
    2. Preprocess BBC dataset
    3. Split BBC dataset into training and testing
    4. Encode category labels
    5. Return the prepared datasets.
    """
    
    def __init__(self, max_length : int, random_state : int = 42, min_freq : int = 1, max_vocab_size : int | None = None):

        self.dataset_loader = BBCDatasetLoader()

        self.text_pipeline = TextPreprocessingPipeline(
            max_length=max_length,
            min_freq=min_freq,
            max_vocab_size=max_vocab_size
        )

        self.label_encoder = CategoryLabelEncoder()
        self.random_state = random_state
    
    def build(self) -> PreparedDataset:
        """
        Build BBC dataset for training and testing.
        """
        
        # Step 1 : Load raw Dataset
        dataset = self.dataset_loader.load_dataset()

        # Step 2 : Split Dataset
        train_df, val_df, test_df = (
            self.dataset_loader.split_dataset(dataset, random_state=self.random_state)
        )

        # Step 3 : Extract article text
        X_train_text = train_df['text'].tolist()
        X_val_text = val_df['text'].tolist()
        X_test_text = test_df['text'].tolist()

        # Step 4 : Extract category labels
        y_train_labels = train_df['category'].tolist()
        y_val_labels = val_df['category'].tolist()
        y_test_labels = test_df['category'].tolist()

        # Step 5 : Fit text pipeline on training data
        X_train = self.text_pipeline.fit_transform(X_train_text)

        # Transform val and test data
        X_val = self.text_pipeline.transform(X_val_text)
        X_test = self.text_pipeline.transform(X_test_text)

        # Step 6 : Fit label encoder on training labels
        y_train = self.label_encoder.fit_transform(y_train_labels)

        # Transform val and test labels
        y_val = self.label_encoder.transform(y_val_labels)
        y_test = self.label_encoder.transform(y_test_labels)

        # Step 7 : Return prepared dataset
        return PreparedDataset(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test
        )