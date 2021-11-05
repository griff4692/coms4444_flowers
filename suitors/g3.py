import functools
import random
from collections import Counter
from typing import Dict, Tuple, Callable, List, Union

from flowers import Bouquet, Flower, FlowerSizes, FlowerColors, FlowerTypes
from suitors.base import BaseSuitor
from utils import flatten_counter

ALL_FEATURES = list(FlowerSizes) + list(FlowerColors) + list(FlowerTypes)
ESTIMATE_SIZE = 3
FINAL_LIMIT = 12
Feature = Union[FlowerTypes, FlowerColors, FlowerSizes]


def check_feature(flower: Flower, feature: Feature) -> bool:
    if feature.__class__ == FlowerTypes:
        return flower.type == feature
    if feature.__class__ == FlowerColors:
        return flower.color == feature
    if feature.__class__ == FlowerSizes:
        return flower.size == feature
    assert False


def l2b(l: List[Flower]) -> Bouquet:
    return Bouquet(Counter(l))


def pick(flower_counts: Dict[Flower, int], target: Callable[[Flower], bool]) -> Tuple[Dict[Flower, int], Bouquet]:
    choices = list(filter(target, flatten_counter(flower_counts)))
    if len(choices) < ESTIMATE_SIZE:
        return flower_counts, Bouquet(dict())
    l = random.sample(list(filter(target, flatten_counter(flower_counts))), ESTIMATE_SIZE)
    remain = flower_counts.copy()
    for f in l:
        remain[f] -= 1
    return remain, l2b(l)


def score_func(flower: Flower, preference: Dict[Feature, float]):
    return preference[flower.type] + preference[flower.color] + preference[flower.size]


def maximize(flowers: Dict[Flower, int], preference: Dict[Feature, float]) -> Tuple[float, Dict[Flower, int], Dict[Flower, int]]:
    score = 0
    picked = Counter()
    remain = flowers.copy()
    while sum(remain.values()) > 0 and sum(picked.values()) < FINAL_LIMIT:
        candidates = [k for k, v in remain.items() if v > 0]
        best = max(candidates, key=functools.partial(score_func, preference=preference))
        score += score_func(best, preference)
        remain[best] -= 1
        picked[best] += 1
    return score, remain, picked


class Suitor(BaseSuitor):

    def __init__(self, days: int, num_suitors: int, suitor_id: int):
        """
        :param days: number of days of courtship
        :param num_suitors: number of suitors, including yourself
        :param suitor_id: unique id of your suitor in range(num_suitors)
        """
        super().__init__(days, num_suitors, suitor_id, name='g3')
        self.day_count = 0
        self.estimate_score_history = [{k: list() for k in ALL_FEATURES} for _ in range(num_suitors)]
        self.estimate_history = list()

        self.recipient_ids = [i for i in range(self.num_suitors) if i != self.suitor_id]

    def solve_final(self, flower_count: Dict[Flower, int]):
        def avg(l: List[float]):
            if not l:
                return 0
            return sum(l) / len(l)
        estimated_preferences = [{k: avg(v) for k, v in p.items()} for p in self.estimate_score_history]

        res = list()
        recipient_ids = self.recipient_ids.copy()
        while recipient_ids:
            best_id = max(recipient_ids, key=lambda r_id: maximize(flower_count, estimated_preferences[r_id])[0])
            _, flower_count, picked = maximize(flower_count, estimated_preferences[best_id])
            res.append((self.suitor_id, best_id, Bouquet(picked)))
            recipient_ids.remove(best_id)
        return res

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
        self.day_count += 1
        if self.day_count == self.days:
            return self.solve_final(flower_counts)

        send = dict()
        estimation = dict()
        recipient_ids = self.recipient_ids.copy()
        random.shuffle(recipient_ids)
        for r_id in recipient_ids:
            preference = self.estimate_score_history[r_id]
            if not all(preference.values()):
                feature_to_estimate = random.choice(list(filter(lambda k: not preference[k], preference.keys())))
            else:
                feature_to_estimate = random.choice(ALL_FEATURES)
            estimation[r_id] = feature_to_estimate
            flower_counts, send[r_id] = pick(flower_counts, functools.partial(check_feature, feature=feature_to_estimate))

        self.estimate_history.append(estimation)
        return [(self.suitor_id, k, v) for k, v in send.items()]

    def zero_score_bouquet(self):
        """
        :return: a Bouquet for which your scoring function will return 0
        """
        return Bouquet(dict())

    def one_score_bouquet(self):
        """
        :return: a Bouquet for which your scoring function will return 1
        """
        return Bouquet({Flower(FlowerSizes.Large, FlowerColors.Blue, FlowerTypes.Rose): 12})

    def score_types(self, types: Dict[FlowerTypes, int]):
        """
        :param types: dictionary of flower types and their associated counts in the bouquet
        :return: A score representing preference of the flower types in the bouquet
        """
        return 0

    def score_colors(self, colors: Dict[FlowerColors, int]):
        """
        :param colors: dictionary of flower colors and their associated counts in the bouquet
        :return: A score representing preference of the flower colors in the bouquet
        """
        return 0

    def score_sizes(self, sizes: Dict[FlowerSizes, int]):
        """
        :param sizes: dictionary of flower sizes and their associated counts in the bouquet
        :return: A score representing preference of the flower sizes in the bouquet
        """
        total_sizes = sum(map(lambda x: x.value, flatten_counter(sizes)))
        return min(total_sizes / 24, 1.0)

    def receive_feedback(self, feedback: List[Tuple[List[int], List[float]]]):
        """
        :param feedback:
        :return: nothing
        """
        for r_id in self.recipient_ids:
            _, score = feedback[r_id]
            self.estimate_score_history[r_id][self.estimate_history[-1][r_id]].append(score)
