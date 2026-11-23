import torch
from torch.utils.data import Dataset

class BBCTensorDataset(Dataset):
    """
    PyTorch Dataset for BBC News text classification.

    Converts preprocessed article sequences and category labels
    into PyTorch tensors.
    """

    def __init__(self, features:list[list[int]], labels:list[int]) -> None:
        if len(features) != len(labels):
            raise ValueError("Features and labels must have the same length")

        if not features:
            raise ValueError("Features cannot be empty.")

        self.features = torch.tensor(features, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        """
        Return number of samples
        """
        return len(self.labels)

    def __getitem__(self, index:int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return sample and label at given index

        Args:
        index (int): Index of the sample

        Returns:
        tuple[torch.Tensor, torch.Tensor]: Tuple of sample and label
        """
    
        return self.features[index], self.labels[index]