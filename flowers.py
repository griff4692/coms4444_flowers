from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from enum import Enum
import itertools
import random
from typing import Dict, List

import numpy as np

from utils import flatten_counter


MAX_BOUQUET_SIZE = 12


class FlowerSizes(Enum):
    Small = 0
    Medium = 1
    Large = 2


class FlowerColors(Enum):
    White = 0
    Yellow = 1
    Red = 2
    Purple = 3
    Orange = 4
    Blue = 5


class FlowerTypes(Enum):
    Rose = 0
    Chrysanthemum = 1
    Tulip = 2
    Begonia = 3


@dataclass(eq=True, frozen=True)
class Flower:
    size: FlowerSizes
    color: FlowerColors
    type: FlowerTypes

    def __str__(self):
        """
        :return: This is how flowers will be printed. Also, the output of str(Flower)
        """
        return f'{self.size.name}-{self.color.name}-{self.type.name}'


class Bouquet:
    def __init__(self, arrangement: Dict[Flower, int]):
        """
        :param arrangement: dictionary of Flowers as keys and counts as values.
        """
        # Sort by flower names so that the equivalent Bouquets have equivalent string representations
        self.arrangement = OrderedDict(sorted(arrangement.items(), key=lambda x: (-x[1], str(x[0]))))
        self.sizes, self.colors, self.types = defaultdict(int), defaultdict(int), defaultdict(int)
        for flower, count in arrangement.items():
            self.sizes[flower.size] += count
            self.colors[flower.color] += count
            self.types[flower.type] += count
        self.types = dict(self.types)
        self.colors = dict(self.colors)
        self.sizes = dict(self.sizes)

    def __len__(self):
        return len(self.arrangement)

    def __repr__(self):
        return str(self)

    def flowers(self):
        flowers = []
        for flower, count in self.arrangement.items():
            flowers += [flower] * count
        return flowers

    def __str__(self):
        if len(self.arrangement) == 0:
            return 'empty'
        return ','.join([f'{k}:{v}' for k, v in self.arrangement.items()])


def get_all_possible_bouquets(flowers: Dict[Flower, int]):
    flat_flower = flatten_counter(flowers)
    bouquets = [Bouquet({})]
    for size in range(1, MAX_BOUQUET_SIZE):
        size_bouquets = list(set(list(itertools.combinations(flat_flower, size))))
        bouquets += size_bouquets
    return bouquets


def get_random_flower():
    """
    :return: Flower of random size, color, and type.
    # A useful utility function.  It's not used in the main code.
    """
    return Flower(
        size=random.choice(list(FlowerSizes)),
        color=random.choice(list(FlowerColors)),
        type=random.choice(list(FlowerTypes)),
    )


def get_all_possible_flowers():
    """
    :return: List of all possible combinations of Flowers (3 sizes * 4 * types * 6 colors = 72)
    """
    return list(map(lambda x: Flower(*x), list(
        itertools.product(*[list(FlowerSizes), list(FlowerColors), list(FlowerTypes)]))))


def sample_n_random_flowers(possible_flowers: List[Flower], n: int):
    """
    :param possible_flowers: A list of Flowers representing all combinations of size, type, and color
    :param n: the number to sample
    :return: A dictionary of Flower to counts.  The sum of the counts is n.  Only nonzero count flowers returned.
    """
    # replace=True allows for repetition and allows n to be larger than len(possible_flowers)
    return dict(Counter(list(np.random.choice(possible_flowers, size=(n,), replace=True))))
