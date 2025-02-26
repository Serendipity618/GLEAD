from collections import Counter

import numpy as np
import pandas as pd

import warnings

warnings.filterwarnings('ignore')


class LogPreprocessor:
    """Preprocess log data by applying sliding window technique."""

    def __init__(self, filepath, sample_size=1_000_000):
        self.logdata = pd.read_csv(filepath, nrows=sample_size)

    def slide_window(self, window_size=20, step_size=10):
        """Apply a sliding window approach to segment log sequences."""
        self.logdata['Label'] = self.logdata['Label'].apply(lambda x: 0 if x == '-' else 1)
        data = self.logdata[['EventId', 'Label']]
        data['Key_label'] = data['Label']

        sequences = [
            (data['EventId'].iloc[i: i + window_size].values,
             max(data['Key_label'].iloc[i: i + window_size]),
             data['Key_label'].iloc[i: i + window_size].values)
            for i in range(0, len(data) - window_size + 1, step_size)
        ]

        return pd.DataFrame(sequences, columns=['EventId', 'Sequence_label', 'Key_label'])


class DataProcessor:
    """Process log sequences by splitting them into train, validation, and test sets."""

    def __init__(self, dataset, n_labeled=20, n_unlabeled=2000, a_unlabeled=0, seed=42):
        self.seed = seed
        self.n_labeled = n_labeled
        self.n_unlabeled = n_unlabeled
        self.a_unlabeled = a_unlabeled
        self.dataset = dataset
        self._split_data()
        self._build_mappings()
        self._encode_sequences()

    def _split_data(self):
        """Split dataset into labeled, unlabeled, validation, and test sets."""
        normal_data = self.dataset[self.dataset['Sequence_label'] == 0]
        abnormal_data = self.dataset[self.dataset['Sequence_label'] == 1]

        # Split the dataset into train, validation, and test sets
        self.train_ds = self._create_train_set(normal_data, abnormal_data)
        self.val_ds = self._create_val_set(normal_data, abnormal_data)
        self.test_ds = self._create_test_set(normal_data, abnormal_data)

    def _create_train_set(self, normal_data, abnormal_data):
        """Create the training set with labeled and unlabeled data."""
        # Sample data for normal and abnormal classes
        train_normal_all = normal_data.sample(n=self.n_unlabeled + self.n_labeled, random_state=self.seed)
        train_abnormal_all = abnormal_data.sample(n=self.a_unlabeled + self.n_labeled, random_state=self.seed)

        # Labeled training data
        train_normal_labeled = train_normal_all.sample(n=self.n_labeled, random_state=self.seed)
        train_abnormal_labeled = train_abnormal_all.sample(n=self.n_labeled, random_state=self.seed)

        # Unlabeled training data
        train_unlabeled = pd.concat([
            train_normal_all.drop(train_normal_labeled.index),
            train_abnormal_all.drop(train_abnormal_labeled.index)
        ])
        # Assign labels for Semi-supervised learning
        train_normal_labeled['Semi'] = 0
        train_abnormal_labeled['Semi'] = 0
        train_unlabeled['Semi'] = 1

        # Combine the labeled and unlabeled data for training
        return pd.concat([train_normal_labeled, train_abnormal_labeled, train_unlabeled])

    def _create_val_set(self, normal_data, abnormal_data):
        """Create the validation set."""
        rest_normal = normal_data.drop(self.train_ds[self.train_ds['Sequence_label'] == 0].index)
        rest_abnormal = abnormal_data.drop(self.train_ds[self.train_ds['Sequence_label'] == 1].index)

        # Sample validation data from remaining normal and abnormal data
        val_normal = rest_normal.sample(n=200, random_state=self.seed)
        val_abnormal = rest_abnormal.sample(n=20, random_state=self.seed)

        # Combine and label the validation data
        val_ds = pd.concat([val_normal, val_abnormal])
        val_ds['Semi'] = 0
        return val_ds

    def _create_test_set(self, normal_data, abnormal_data):
        """Create the test set."""
        rest_normal = normal_data.drop(self.train_ds[self.train_ds['Sequence_label'] == 0].index)
        rest_abnormal = abnormal_data.drop(self.train_ds[self.train_ds['Sequence_label'] == 1].index)

        # Sample test data from remaining normal and abnormal data
        test_normal = rest_normal.drop(self.val_ds[self.val_ds['Sequence_label'] == 0].index).sample(n=20_000,
                                                                                                     random_state=self.seed)
        test_abnormal = rest_abnormal.drop(self.val_ds[self.val_ds['Sequence_label'] == 1].index).sample(n=2_000,
                                                                                                         random_state=self.seed)

        # Combine and label the test data
        test_ds = pd.concat([test_normal, test_abnormal])
        test_ds['Semi'] = 0
        return test_ds

    def _build_mappings(self):
        """Create a mapping of log keys to indices."""
        counts = Counter(self.train_ds['EventId'].explode())  # Count the occurrences of each log key
        self.logkey2index = {"": 0, "UNK": 1}  # Initialize mappings for empty and unknown keys
        self.logkeys = ["", "UNK"]  # Initialize the list with empty and unknown keys

        # Update mappings with actual log keys from the dataset
        self.logkey2index.update({word: i + 2 for i, word in enumerate(counts)})

        # Update logkeys with the newly added words
        self.logkeys.extend(counts.keys())

    def _encode_sequences(self):
        """Convert log event sequences into encoded numerical representations."""
        encode = lambda seq: np.array([self.logkey2index.get(logkey, self.logkey2index["UNK"]) for logkey in seq])
        self.train_ds['Encoded'] = self.train_ds['EventId'].apply(encode)
        self.val_ds['Encoded'] = self.val_ds['EventId'].apply(encode)
        self.test_ds['Encoded'] = self.test_ds['EventId'].apply(encode)
