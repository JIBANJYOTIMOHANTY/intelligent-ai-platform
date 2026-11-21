from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

class BBCDatasetLoader:
    """
    Load and split the BBC News dataset.
    """

    REQUIRED_COLUMNS = {"category", "text"}

    def __init__(self, dataset_path : str | Path | None = None):
        project_root = Path(__file__).resolve().parents[2]

        if dataset_path is None:
            dataset_path = (
                project_root / "data" / "raw" / "bbc-text.csv"
            )

        self.dataset_path = Path(dataset_path)

    def load_dataset(self) -> pd.DataFrame:
        """
        Load and validate the BBC News dataset.
        """

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.dataset_path}"
            )

        df = pd.read_csv(self.dataset_path)
        missing_columns = self.REQUIRED_COLUMNS - set(df.columns)

        if missing_columns:
            raise ValueError(
                f"Dataset is missing required columns: {missing_columns}"
            )

        if df[list(self.REQUIRED_COLUMNS)].isnull().any().any():
            raise ValueError(
                "Dataset contains missing values in required columns."
            )

        return df
    
    def split_dataset(self, df:pd.DataFrame, random_state : int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the dataset into training, validation, and test sets.
        
        Stratified splitting is used to ensure that each category
        is proportionally represented in each split.
        
        Args:
            df: Input DataFrame containing the dataset
            random_state: Seed for reproducibility
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        
        train_df, remaining_df = train_test_split(df, test_size=0.30, random_state=random_state, stratify=df['category'])

        validation_df, test_df = train_test_split(remaining_df, test_size=0.50, random_state=random_state, stratify=remaining_df['category'])

        return (
            train_df.reset_index(drop=True),
            validation_df.reset_index(drop=True),
            test_df.reset_index(drop=True)
        )