import numpy as np
from typing import List, Tuple


def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Generate train/test index splits for k-fold cross-validation.

    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds (default 5)
        shuffle: Whether to shuffle indices before splitting (default True)

    Returns:
        List of (train_indices, test_indices) tuples
    """
    array = np.arange(n_samples)
    fold_size = len(array) // k
    if shuffle == True:
        np.random.shuffle(array)

    results = []
    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size

        test_indices = array[start:end]
        train_indices = np.concatenate([array[:start], array[end:]])

        results.append((train_indices.tolist(), test_indices.tolist(),))

    return results







