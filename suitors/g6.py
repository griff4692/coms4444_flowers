from collections import Counter
from typing import Dict

import numpy as np

from constants import MAX_BOUQUET_SIZE
from flowers import Bouquet, Flower, FlowerSizes, FlowerColors, FlowerTypes
from utils import flatten_counter
from suitors.base import BaseSuitor


class Suitor(BaseSuitor):
    def __init__(self, days: int, num_suitors: int, suitor_id: int):
        """
        :param days: number of days of courtship
        :param num_suitors: number of suitors, including yourself
        :param suitor_id: unique id of your suitor in range(num_suitors)
        """
        super().__init__(days, num_suitors, suitor_id, name='g6')

    def _prepare_bouquet(self, remaining_flowers, recipient_id):
        num_remaining = sum(remaining_flowers.values())
        size = int(np.random.randint(0, min(MAX_BOUQUET_SIZE, num_remaining) + 1))
        if size > 0:
            chosen_flowers = np.random.choice(flatten_counter(remaining_flowers), size=(size, ), replace=False)
            chosen_flower_counts = dict(Counter(chosen_flowers))
            for k, v in chosen_flower_counts.items():
                remaining_flowers[k] -= v
                assert remaining_flowers[k] >= 0
        else:
            chosen_flower_counts = dict()
        chosen_bouquet = Bouquet(chosen_flower_counts)
        return self.suitor_id, recipient_id, chosen_bouquet

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
        all_ids = np.arange(self.num_suitors)
        recipient_ids = all_ids[all_ids != self.suitor_id]
        remaining_flowers = flower_counts.copy()
        return list(map(lambda recipient_id: self._prepare_bouquet(remaining_flowers, recipient_id), recipient_ids))

    def zero_score_bouquet(self):
        """
        :return: a Bouquet for which your scoring function will return 0
        """
        return Bouquet({})

    def one_score_bouquet(self):
        """
        :return: a Bouquet for which your scoring function will return 1
        """
        f1 = Flower(
            size=FlowerSizes.Large,
            color=FlowerColors.Blue,
            type=FlowerTypes.Rose
        )
        f2 = Flower(
            size=FlowerSizes.Small,
            color=FlowerColors.White,
            type=FlowerTypes.Chrysanthemum
        )
        f3 = Flower(
            size=FlowerSizes.Medium,
            color=FlowerColors.Yellow,
            type=FlowerTypes.Tulip
        )
        f4 = Flower(
            size=FlowerSizes.Medium,
            color=FlowerColors.Red,
            type=FlowerTypes.Begonia
        )
        f5 = Flower(
            size=FlowerSizes.Medium,
            color=FlowerColors.Purple,
            type=FlowerTypes.Begonia
        )
        f6 = Flower(
            size=FlowerSizes.Medium,
            color=FlowerColors.Orange,
            type=FlowerTypes.Begonia
        )
        return Bouquet({f1:1,f2:1,f3:1,f4:1,f5:1,f6:1})


    def score_types(self, types: Dict[FlowerTypes, int]):
        """
        :param types: dictionary of flower types and their associated counts in the bouquet
        :return: A score representing preference of the flower types in the bouquet
        """
        # if len(types) == 0:
        #     return 0.0
        #
        # avg_types = float(np.mean([x.value for x in flatten_counter(types)]))
        #return avg_types / (3 * (len(FlowerTypes) - 1))
        return len(types) / (len(FlowerSizes)+len(FlowerColors)+len(FlowerTypes))

    def score_colors(self, colors: Dict[FlowerColors, int]):
        """
        :param colors: dictionary of flower colors and their associated counts in the bouquet
        :return: A score representing preference of the flower colors in the bouquet
        """
        # if len(colors) == 0:
        #     return 0.0
        #
        # avg_colors = float(np.mean([x.value for x in flatten_counter(colors)]))
        #return avg_colors / (3 * (len(FlowerColors) - 1))
        return len(colors) / (len(FlowerSizes)+len(FlowerColors)+len(FlowerTypes))

    def score_sizes(self, sizes: Dict[FlowerSizes, int]):
        """
        :param sizes: dictionary of flower sizes and their associated counts in the bouquet
        :return: A score representing preference of the flower sizes in the bouquet
        """
        # if len(sizes) == 0:
        #     return 0
        #
        # avg_sizes = float(np.mean([x.value for x in flatten_counter(sizes)]))
        #return avg_sizes / (3 * (len(FlowerSizes) - 1))
        return len(sizes) / (len(FlowerSizes)+len(FlowerColors)+len(FlowerTypes))

    def receive_feedback(self, feedback):
        """
        :param feedback:
        :return: nothing
        """
        self.feedback.append(feedback)
