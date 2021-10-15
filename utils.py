from collections import Counter
import itertools
from typing import List

from flowers import Flower
import numpy as np


def flatten_counter(counts):
    """
    :param counts: a dictionary of counts
    :return: a flattened list including each key repeated for its count
    Example: {'x': 1, 'y': 2} --> [x, y, y]
    """
    return list(itertools.chain(*[[k] * v for k, v in counts.items()]))


def sample_n_random_flowers(possible_flowers: List[Flower], n: int):
    """
    :param possible_flowers: A list of Flowers representing all combinations of size, type, and color
    :param n: the number to sample
    :return: A dictionary of Flower to counts.  The sum of the counts is n.  Only nonzero count flowers returned.
    """
    # replace=True allows for repetition and allows n to be larger than len(possible_flowers)
    return dict(Counter(list(np.random.choice(possible_flowers, size=(n,), replace=True))))
