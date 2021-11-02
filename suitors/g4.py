from typing import Dict

from flowers import Bouquet, Flower, FlowerSizes, FlowerColors, FlowerTypes
from suitors.base import BaseSuitor
from random import shuffle
import numpy as np
from utils import flatten_counter


class Suitor(BaseSuitor):
    def __init__(self, days: int, num_suitors: int, suitor_id: int):
        """
        :param days: number of days of courtship
        :param num_suitors: number of suitors, including yourself
        :param suitor_id: unique id of your suitor in range(num_suitors)
        """
        super().__init__(days, num_suitors, suitor_id, name='g4')
        self.size_mapping = self.generate_map(FlowerSizes)
        self.color_mapping = self.generate_map(FlowerColors)
        self.type_mapping = self.generate_map(FlowerTypes)
        self.best_arrangement = self.best_bouquet()

    def generate_map(self, flower_enum):
        sizes = [n.value for n in flower_enum]
        shuffle(sizes)
        mapping = {}
        for idx, name in enumerate(flower_enum):
            mapping[name] = sizes[idx]
        return mapping
    
    def best_bouquet(self):
        best_size = (sorted([(key, value) for key, value in self.size_mapping.items()], key=lambda x: x[1]))[-1][0]
        best_color = (sorted([(key, value) for key, value in self.color_mapping.items()], key=lambda x: x[1]))[-1][0]
        best_type = (sorted([(key, value) for key, value in self.type_mapping.items()], key=lambda x: x[1]))[-1][0]
        best_flower = Flower(
            size=best_size,
            color=best_color,
            type=best_type
        )
        return Bouquet({best_flower: 1})

    def prepare_bouquets(self, flower_counts: Dict[Flower, int]):
        """
        :param flower_counts: flowers and associated counts for for available flowers
        :return: list of tuples of (self.suitor_id, recipient_id, chosen_bouquet)
        the list should be of length len(self.num_suitors) - 1 because you should give a bouquet to everyone
         but yourself

        To get the list of suitor ids not including yourself, use the following snippet:

        all_ids = np.arange(self.num_suitors)
        recipient_ids = all_ids[all_ids != self.suitor_id]
        """
        pass

    def zero_score_bouquet(self):
        """
        :return: a Bouquet for which your scoring function will return 0
        """
        return Bouquet({})

    def one_score_bouquet(self):
        """
        :return: a Bouquet for which your scoring function will return 1
        """
        return self.best_arrangement

    def score_types(self, types: Dict[FlowerTypes, int]):
        """
        :param types: dictionary of flower types and their associated counts in the bouquet
        :return: A score representing preference of the flower types in the bouquet
        """
        if len(types) == 0:
            return 0.0

        avg_types = float(np.mean([self.type_mapping[x] for x in flatten_counter(types)]))
        return avg_types / (3 * (len(FlowerTypes) - 1))

    def score_colors(self, colors: Dict[FlowerColors, int]):
        """
        :param colors: dictionary of flower colors and their associated counts in the bouquet
        :return: A score representing preference of the flower colors in the bouquet
        """
        if len(colors) == 0:
            return 0.0

        avg_types = float(np.mean([self.type_mapping[x] for x in flatten_counter(colors)]))
        return avg_types / (3 * (len(FlowerColors) - 1))

    def score_sizes(self, sizes: Dict[FlowerSizes, int]):
        """
        :param sizes: dictionary of flower sizes and their associated counts in the bouquet
        :return: A score representing preference of the flower sizes in the bouquet
        """
        if len(sizes) == 0:
            return 0.0

        avg_types = float(np.mean([self.type_mapping[x] for x in flatten_counter(sizes)]))
        return avg_types / (3 * (len(FlowerSizes) - 1))

    def receive_feedback(self, feedback):
        """
        :param feedback:
        :return: nothing
        """
        self.feedback.append(feedback)
