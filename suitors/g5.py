from typing import Dict, Tuple, List, Union
import heapq
import random

from flowers import Bouquet, Flower, FlowerSizes, FlowerColors, FlowerTypes
from suitors.base import BaseSuitor
from suitors import random_suitor


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
        final_bouquets = []
        already_prepared = set()
        for fb in self.feedback:
            fb: SuitorFeedback = fb
            if fb.suitor in already_prepared:
                continue
            if self.can_construct(fb.bouquet, flower_counts):
                final_bouquets.append((self.suitor_id, fb.suitor, fb.bouquet))
                flower_counts = self.reduce_flowers(fb.bouquet, flower_counts)
                already_prepared.add(fb.suitor)

        return final_bouquets

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
        f = Flower(FlowerSizes.Small, self.bad_color_enum, FlowerTypes.Rose)
        d = {f: 1}
        return Bouquet(d)

    def one_score_bouquet(self):
        """
        :return: a Bouquet for which your scoring function will return 1
        """
        good_color_enum = None
        for v in FlowerColors:
            if v != self.bad_color_enum:
                good_color_enum = v
                break
        f = Flower(FlowerSizes.Small, good_color_enum, FlowerTypes.Rose)
        d = {f: 1}
        return Bouquet(d)

    def score_types(self, types: Dict[FlowerTypes, int]):
        """
        :param types: dictionary of flower types and their associated counts in the bouquet
        :return: A score representing preference of the flower types in the bouquet
        """
        return 0.0

    def score_colors(self, colors: Dict[FlowerColors, int]):
        """
        :param colors: dictionary of flower colors and their associated counts in the bouquet
        :return: A score representing preference of the flower colors in the bouquet
        """
        for c, v in colors.items():
            if c == self.bad_color_enum and v != 0:
                return 0

        return 1.0

    def score_sizes(self, sizes: Dict[FlowerSizes, int]):
        """
        :param sizes: dictionary of flower sizes and their associated counts in the bouquet
        :return: A score representing preference of the flower sizes in the bouquet
        """
        return 0.0

    def receive_feedback(self, feedback):
        """
        :param feedback: One giant tuple (acting as a list) that contains tuples of (rank, score) ordered by suitor
            number such that feedback[0] is the feedback for suitor 0.
        :return: nothing
        """
        for suitor_num, (rank, score) in enumerate(feedback):
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
