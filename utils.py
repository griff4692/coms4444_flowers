import itertools


def flatten_counter(counts):
    """
    :param counts: a dictionary of counts
    :return: a flattened list including each key repeated for its count
    Example: {'x': 1, 'y': 2} --> [x, y, y]
    """
    return list(itertools.chain(*[[k] * v for k, v in counts.items()]))
