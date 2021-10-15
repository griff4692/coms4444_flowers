from enum import Enum
import random
from typing import Dict


MAX_BOUQUET_SIZE = 12


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


class Flower:
    def __init__(self, size: FlowerSizes, color: FlowerColors, type: FlowerTypes):
        """
        :param size: flower size
        :param color: flower color
        :param type: flower type
        """
        self.size = size
        self.color = color
        self.type = type

    def __str__(self):
        """
        :return: This is how flowers will be printed. Also, the output of str(Flower)
        """
        return f'(size={self.size}, color={self.color}, type={self.type})'


class Bouquet:
    def __init__(self, arrangement: Dict[Flower, int]):
        """
        :param arrangement: dictionary of Flowers as keys and counts as values.
        """
        self.arrangement = arrangement

    def __len__(self):
        return len(self.arrangement)
