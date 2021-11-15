import collections
import heapq
import random
import math
from dataclasses import dataclass
from typing import Dict, Tuple, List, Union

import numpy as np

from flowers import Bouquet, Flower, FlowerSizes, FlowerColors, FlowerTypes
from suitors import random_suitor
from suitors.base import BaseSuitor


class SuitorFeedback:
    def __init__(self, suitor: int, epoch: int, rank: int, score: float, bouquet: Bouquet):
        self._suitor = suitor
        self._epoch = epoch
        self._rank = rank
        self._score = score
        self._bouquet = bouquet

    @property
    def suitor(self):
        return self._suitor

    @property
    def epoch(self):
        return self._epoch

    @property
    def rank(self):
        return self._rank

    @property
    def score(self):
        return self._score

    @property
    def bouquet(self):
        return self._bouquet

    def __lt__(self, other):
        return self.score < other.score


# Necessary to make hashing work properly while we're constructing our random bouquet
@dataclass(eq=True, frozen=True)
class StaticFlower:
    size: FlowerSizes
    color: FlowerColors
    type: FlowerTypes

    def to_flower(self) -> Flower:
        return Flower(self.size, self.color, self.type)


def random_bouquet(size: int) -> Bouquet:
    flowers = collections.defaultdict(lambda: 0)
    sizes = list(FlowerSizes)
    colors = list(FlowerColors)
    types = list(FlowerTypes)
    for _ in range(size):
        f = StaticFlower(size=np.random.choice(sizes), color=np.random.choice(colors), type=np.random.choice(types))
        flowers[f] += 1

    b_dict = {}
    for key, value in flowers.items():
        b_dict[key.to_flower()] = value
    return Bouquet(b_dict)


class Suitor(BaseSuitor):
    def __init__(self, days: int, num_suitors: int, suitor_id: int):
        """
        :param days: number of days of courtship
        :param num_suitors: number of suitors, including yourself
        :param suitor_id: unique id of your suitor in range(num_suitors)
        """
        super().__init__(days, num_suitors, suitor_id, name='g5')
        self.feedback_cache = {i: [] for i in range(num_suitors)}
        self.current_day = 0  # The current day
        self.rand_man = random_suitor.RandomSuitor(days, num_suitors, suitor_id)
        # Cached bouquets from current round
        self.bouquets = {}
        bad_color_num = random.randint(0, len(FlowerColors)-1)
        self.bad_color_enum = FlowerColors(bad_color_num)

        # New bouquet setup
        self.n_flowers = math.ceil(math.log(num_suitors * days) / math.log(8))
        self.ideal_bouquet: Bouquet = random_bouquet(self.n_flowers)

    @staticmethod
    def can_construct(bouquet: Bouquet, flower_counts: Dict[Flower, int]):
        for key, value in bouquet.arrangement.items():
            if not (key in flower_counts and flower_counts[key] >= value):
                return False
        return True

    @staticmethod
    def reduce_flowers(bouquet: Bouquet, flower_counts: Dict[Flower, int]) -> Union[None, Dict[Flower, int]]:
        # if not Suitor.can_construct(bouquet, flower_counts):
        #     return None
        new_counts = flower_counts.copy()
        for key, value in bouquet.arrangement.items():
            new_counts[key] -= value
        return new_counts

    def prepare_final_bouquets(self, flower_counts: Dict[Flower, int]) -> List[Tuple[int, int, Bouquet]]:
        def key_func(fb: SuitorFeedback):
            return fb.score

        self.feedback.sort(key=key_func, reverse=True)
        final_bouquets = {n: (self.suitor_id, n, Bouquet({})) for n in range(self.num_suitors)}
        del final_bouquets[self.suitor_id]
        already_prepared = set()
        for fb in self.feedback:
            fb: SuitorFeedback = fb
            if fb.suitor in already_prepared:
                continue
            if self.can_construct(fb.bouquet, flower_counts):
                final_bouquets[fb.suitor] = (self.suitor_id, fb.suitor, fb.bouquet)
                flower_counts = self.reduce_flowers(fb.bouquet, flower_counts)
                already_prepared.add(fb.suitor)
        return list(final_bouquets.values())

    def prepare_bouquets(self, flower_counts: Dict[Flower, int]) -> List[Tuple[int, int, Bouquet]]:
        """
        :param flower_counts: flowers and associated counts for for available flowers
        :return: list of tuples of (self.suitor_id, recipient_id, chosen_bouquet)
        the list should be of length len(self.num_suitors) - 1 because you should give a bouquet to everyone
         but yourself

        To get the list of suitor ids not including yourself, use the following snippet:

        all_ids = np.arange(self.num_suitors)
        recipient_ids = all_ids[all_ids != self.suitor_id]
        """
        self.current_day += 1
        if self.current_day == self.days:
            return self.prepare_final_bouquets(flower_counts)
        # Saving these for later
        bouquets = self.rand_man.prepare_bouquets(flower_counts)
        for _, suitor, bouquet in bouquets:
            self.bouquets[suitor] = bouquet

        return bouquets

    def zero_score_bouquet(self):
        """
        :return: a Bouquet for which your scoring function will return 0
        """
        return Bouquet(dict())

    def one_score_bouquet(self):
        """
        :return: a Bouquet for which your scoring function will return 1
        """
        return self.ideal_bouquet

    def score_x(self, max_score: float, actual_x: Dict, ideal_x: Dict) -> float:
        matching = self.n_flowers
        for x, c in ideal_x.items():
            if x not in actual_x or c > actual_x[x]:
                matching -= c
        if matching <= 0:
            return 0.0
        return max_score / 2**(self.n_flowers - matching)

    def score_types(self, types: Dict[FlowerTypes, int]):
        """
        :param types: dictionary of flower types and their associated counts in the bouquet
        :return: A score representing preference of the flower types in the bouquet
        """
        ideal_types = self.ideal_bouquet.types
        return self.score_x(0.0, types, ideal_types)

    def score_colors(self, colors: Dict[FlowerColors, int]):
        """
        :param colors: dictionary of flower colors and their associated counts in the bouquet
        :return: A score representing preference of the flower colors in the bouquet
        """
        ideal_colors = self.ideal_bouquet.colors
        return self.score_x(0.34 + 0.33, colors, ideal_colors)

    def score_sizes(self, sizes: Dict[FlowerSizes, int]):
        """
        :param sizes: dictionary of flower sizes and their associated counts in the bouquet
        :return: A score representing preference of the flower sizes in the bouquet
        """
        ideal_sizes = self.ideal_bouquet.sizes
        return self.score_x(0.33, sizes, ideal_sizes)

    def receive_feedback(self, feedback):
        """
        :param feedback: One giant tuple (acting as a list) that contains tuples of (rank, score) ordered by suitor
            number such that feedback[0] is the feedback for suitor 0.
        :return: nothing
        """
        for suitor_num, (rank, score, _) in enumerate(feedback):
            # Skip ourselves
            if score == float('-inf'):
                continue
            # Custom class to encapsulate feedback info and sort accordingly
            fb = SuitorFeedback(suitor_num, self.current_day, rank, score, self.bouquets[suitor_num])
            # Suitor specific PQ
            heapq.heappush(self.feedback_cache[suitor_num], fb)
            # Global PQ
            # TODO: for the last round we want to sort this by rank (lowest) and epoch(highest) so that
            #   we pick the best ranked and most recent bouquet
            heapq.heappush(self.feedback, fb)

        self.rand_man.receive_feedback(feedback)
