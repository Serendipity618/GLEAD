from torch.utils.data import Dataset, DataLoader


class LogDataset(Dataset):
    """Custom PyTorch dataset for log anomaly detection."""

    def __init__(self, sequences, sequence_labels, key_labels, semi_labels):
        self.sequences = sequences
        self.sequence_labels = sequence_labels
        self.key_labels = key_labels
        self.semi_labels = semi_labels

    def __len__(self):
        return len(self.sequence_labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.sequence_labels[idx], self.key_labels[idx], self.semi_labels[idx]


class DataLoaderWrapper:
    """Wrap DataLoader for training, validation, and testing."""

    def __init__(self, dataset, batch_sizes):
        self.train_loader = self._create_loader(dataset.train_ds, batch_sizes['train'])
        self.val_loader = self._create_loader(dataset.val_ds, batch_sizes['val'])
        self.test_loader = self._create_loader(dataset.test_ds, batch_sizes['test'])

    def _create_loader(self, df, batch_size):
        """Create a PyTorch DataLoader from a dataset."""
        dataset = LogDataset(
            sequences=df['Encoded'].tolist(),
            sequence_labels=df['Sequence_label'].tolist(),
            key_labels=df['Key_label'].tolist(),
            semi_labels=df['Semi'].tolist()
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
