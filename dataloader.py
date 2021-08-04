from torch.utils.data import Dataset
import numpy as np


class TabularDataset(Dataset):
    def __init__(self, X_features: np.array, X_embeddings, y: np.array):
        self.X_features = X_features
        self.X_embeddings = X_embeddings
        self.y = y

    def __len__(self):
        return len(self.X_features)

    def __getitem__(self, idx):
        return self.X_features[idx, :], self.X_embeddings[idx, :], self.y[idx]
