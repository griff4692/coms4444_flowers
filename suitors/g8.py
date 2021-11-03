from typing import Dict
from collections import Counter
from flowers import Bouquet, Flower, FlowerSizes, FlowerColors, FlowerTypes
from suitors.base import BaseSuitor
import random
import numpy as np
from constants import MAX_BOUQUET_SIZE
from utils import flatten_counter


class Suitor(BaseSuitor):
    def __init__(self, days: int, num_suitors: int, suitor_id: int):
        """
        :param days: number of days of courtship
        :param num_suitors: number of suitors, including yourself
        :param suitor_id: unique id of your suitor in range(num_suitors)
        """
        super().__init__(days, num_suitors, suitor_id, name='g8')
        self.type_weights = [.1, .2, .35, 1]
        self.color_weights = [.1, .3, .5, .5, 1, 1]
        self.size_weights = [0, .5, 1]
        random.shuffle(self.type_weights)
        random.shuffle(self.color_weights)
        random.shuffle(self.size_weights)


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

        """:param flower_counts: flowers and associated counts for for available flowers
        :return: list of tuples of (self.suitor_id, recipient_id, chosen_bouquet)
        the list should be of length len(self.num_suitors) - 1 because you should give a bouquet to everyone
         but yourself

        To get the list of suitor ids not including yourself, use the following snippet:"""
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
        max_flower = Flower(
            size=FlowerSizes(self.size_weights.index(1)),
            color=FlowerColors(self.color_weights.index(1)),
            type=FlowerTypes(self.type_weights.index(1))
        )
        return Bouquet({max_flower: 12})

    def score_types(self, types: Dict[FlowerTypes, int]):
        """
        :param types: dictionary of flower types and their associated counts in the bouquet
        :return: A score representing preference of the flower types in the bouquet
        """
        # weights
        # weights = [.1, .2, .35, 1]

        # get preference order
        # random.shuffle(weights)

        score = 0
        total = 0
        # sum up the scores of each flower type
        for flower in types:
            index = flower.value
            number = types[flower]
            score = score + (self.type_weights[index]*number)
            total = total + number

        # get average score for number of flowers
        if total != 0:
            score = score/total

        # multiply by .25 since each type, sixe, color, number is .25 weight
        return score*.25


    def score_colors(self, colors: Dict[FlowerColors, int]):
        """
        :param colors: dictionary of flower colors and their associated counts in the bouquet
        :return: A score representing preference of the flower colors in the bouquet
        """
        # weights
        # weights = [.1, .3, .5, .5, 1, 1]

        # get preference order
        # random.shuffle(weights)

        score = 0
        total = 0
        # sum up the scores of each flower type
        for flower in colors:
            index = flower.value
            number = colors[flower]
            score = score + (self.color_weights[index] * number)
            total = total + number

        # get average score for number of flowers
        if total != 0:
            score = score / total

        # multiply by .25 since each type, sixe, color, number is .25 weight
        return score * .25

    def score_sizes(self, sizes: Dict[FlowerSizes, int]):
        """
        :param sizes: dictionary of flower sizes and their associated counts in the bouquet
        :return: A score representing preference of the flower sizes in the bouquet
        """
        # weights
        # weights = [0, .5, 1]

        # get preference order
        # random.shuffle(weights)

        score = 0
        total = 0
        # sum up the scores of each flower type
        for flower in sizes:
            index = flower.value # get enum value for flower attribute
            number = sizes[flower] # get number of flowers
            score = score + (self.size_weights[index] * number)
            total = total + number

        # get average score for number of flowers
        if total != 0:
            score = score / total

        # multiply by .25 since each type, sixe, color, number is .25 weight
        score = score * .25

        # count number of flowers for the last .25
        weights_number = [0, .08, .16, .24, .32, .4, .48, .56, .64, .72, .8, .88, 1]
        score_count = weights_number[total]*.25

        return score+score_count

    def receive_feedback(self, feedback):
        """
        :param feedback:
        :return: nothing
        """
        self.feedback.append(feedback)
