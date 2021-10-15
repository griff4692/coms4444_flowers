from collections import Counter
from typing import Dict

import numpy as np

from constants import MAX_BOUQUET_SIZE
from flowers import get_random_flower, Bouquet, Flower
from utils import flatten_counter
from suitors.base import BaseSuitor


class RandomSuitor(BaseSuitor):
    def __init__(self, days: int, num_suitors: int, suitor_id: int):
        """
        :param days: number of days of courtship
        :param num_suitors: number of suitors, including yourself
        :param suitor_id: unique id of your suitor in range(num_suitors)
        """
        super().__init__(days, num_suitors, suitor_id)

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
        all_ids = np.arange(self.num_suitors)
        recipient_ids = all_ids[all_ids != self.suitor_id]
        remaining_flowers = flower_counts.copy()
        return list(map(lambda recipient_id: self._prepare_bouquet(remaining_flowers, recipient_id), recipient_ids))

    def zero_score_bouquet(self):
        return Bouquet({})

    def one_score_bouquet(self):
        rand_flower = get_random_flower()
        return Bouquet({rand_flower: MAX_BOUQUET_SIZE})

    def score_type(self, bouquet: Bouquet):
        """
        :param bouquet: an arrangement of flowers
        :return: A score representing preference of the flower types in the bouquet
        """
        if len(bouquet) == 0:
            return 0
        if len(bouquet) == MAX_BOUQUET_SIZE:
            return 1.0
        return np.random.random() / 3.0

    def score_color(self, bouquet: Bouquet):
        """
        :param bouquet: an arrangement of flowers
        :return: A score representing preference of the flower colors in the bouquet
        """
        if len(bouquet) == 0 or len(bouquet) == MAX_BOUQUET_SIZE:
            return 0
        return np.random.random() / 3.0

    def score_size(self, bouquet: Bouquet):
        """
        :param bouquet: an arrangement of flowers
        :return: A score representing preference of the flower sizes in the bouquet
        """
        if len(bouquet) == 0 or len(bouquet) == MAX_BOUQUET_SIZE:
            return 0
        return np.random.random() / 3.0

    def receive_feedback(self, feedback):
        """
        :param feedback:
        :return: nothing
        """
        self.feedback.append(feedback)
