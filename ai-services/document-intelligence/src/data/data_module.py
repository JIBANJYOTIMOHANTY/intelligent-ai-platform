from torch.utils.data import DataLoader
from data.dataset_builder import PreparedDataset
from data.torch_dataset import BBCTensorDataset

class BBCDataModule:
    """
    Manage PyTorch datasets and DataLoaders for BBC News classification.
    """

    def __init__(self, prepared_dataset: PreparedDataset, batch_size : int = 32) -> None:
        if batch_size <= 0:
            raise ValueError("Batch size must be greater than 0")

        self.batch_size = batch_size

        self.train_dataset = BBCTensorDataset(prepared_dataset.X_train, prepared_dataset.y_train)
        self.val_dataset = BBCTensorDataset(prepared_dataset.X_val, prepared_dataset.y_val)
        self.test_dataset = BBCTensorDataset(prepared_dataset.X_test, prepared_dataset.y_test)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def train_dataloader(self) -> DataLoader:
        """
        Return the training data loader.
        """
        return self.train_loader
    
    def val_dataloader(self) -> DataLoader:
        """
        Return the validation data loader.
        """
        return self.val_loader
    
    def test_dataloader(self) -> DataLoader:
        """
        Return the test data loader.
        """
        return self.test_loader