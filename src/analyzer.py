from collections import Counter


class LogAnalyzer:
    """Class for analyzing log data and extracting abnormal log keys."""

    @staticmethod
    def save_top_entries(top_entry, n_attention_heads, filename='top_entry.txt'):
        """Save most common top entries per attention head to a file."""
        with open(filename, 'w+') as f:
            for i in range(n_attention_heads):
                f.write(f'Head {i}: \n')
                f.write(str(Counter(top_entry[i]).most_common()))
                f.write('\n\n')

    @staticmethod
    def extract_abnormal_keys(test_abnormal):
        """Extract abnormal keys from test abnormal data."""
        abnormal_keys = []
        for i in range(test_abnormal.shape[0]):
            abnormal_keys += test_abnormal.iloc[i, 0][test_abnormal.iloc[i, 2] == 1].tolist()
        return Counter(abnormal_keys).most_common()

    @staticmethod
    def map_abnormal_keys_to_index(abnormal_keys, logkey2index):
        """Map abnormal keys to their corresponding indices."""
        abnormal_key2index = [[logkey2index.get(each[0]), each[1]] for each in abnormal_keys]
        return abnormal_key2index
