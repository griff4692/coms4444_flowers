from typing import Dict
import numpy as np
from flowers import Bouquet, Flower, FlowerSizes, FlowerColors, FlowerTypes
from suitors.base import BaseSuitor
from constants import MAX_BOUQUET_SIZE
from utils import flatten_counter
from collections import Counter

class Suitor(BaseSuitor):
    def __init__(self, days: int, num_suitors: int, suitor_id: int):
        """
        :param days: number of days of courtship
        :param num_suitors: number of suitors, including yourself
        :param suitor_id: unique id of your suitor in range(num_suitors)
        """
        super().__init__(days, num_suitors, suitor_id, name='g9')
        temp = self.random_sequence(6)
        self.color_score = [FlowerColors(i) for i in temp]
        temp = self.random_sequence(4)
        self.type_score = [FlowerTypes(i) for i in temp]
        temp = self.random_sequence(3)
        self.size_score = [FlowerSizes(i) for i in temp]
    def random_sequence(self, n):
        sequence = np.arange(n)
        np.random.shuffle(sequence)
        return sequence



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
        """
        all_ids = np.arange(self.num_suitors)
        recipient_ids = all_ids[all_ids != self.suitor_id]
        remaining_flowers = flower_counts.copy()
        return list(map(lambda recipient_id: self._prepare_bouquet(remaining_flowers, recipient_id), recipient_ids))


    def zero_score_bouquet(self):
        """
        :return: a Bouquet for which your scoring function will return 0
        """
        min_flower = Flower(
            size=self.size_score[0],
            color=self.color_score[0],
            type=self.type_score[0]
        )
        return Bouquet({min_flower: 0})

    def one_score_bouquet(self):
        """
        :return: a Bouquet for which your scoring function will return 1
        """
        max_flower = Flower(
            size=self.size_score[2],
            color=self.color_score[5],
            type=self.type_score[3]
        )
        return Bouquet({max_flower: 12})

    def score_types(self, types: Dict[FlowerTypes, int]):
        """
        :param types: dictionary of flower types and their associated counts in the bouquet
        :return: A score representing preference of the flower types in the bouquet
        """
        score = 0
        for type in types:
            score += types[type] * self.type_score.index(type)
        return score / 117



    def score_colors(self, colors: Dict[FlowerColors, int]):
        """
        :param colors: dictionary of flower colors and their associated counts in the bouquet
        :return: A score representing preference of the flower colors in the bouquet
        """
        score = 0
        for color in colors:
            score += colors[color] * self.color_score.index(color)
        return score / 130

    def score_sizes(self, sizes: Dict[FlowerSizes, int]):
        """
        :param sizes: dictionary of flower sizes and their associated counts in the bouquet
        :return: A score representing preference of the flower sizes in the bouquet
        """
        score = 0
        for size in sizes:
            score += sizes[size] * self.size_score.index(size)
        return score / 104

    def receive_feedback(self, feedback):
        """
        :param feedback:
        :return: nothing
        """
        self.feedback.append(feedback)
