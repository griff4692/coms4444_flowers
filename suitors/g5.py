import collections
import heapq
import random
import math
import itertools

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


def new_prob(p: int, n: int, k: int) -> float:
    return (1/p)**k * ((p-1)/p)**(n-k) * (math.factorial(n)/(math.factorial(k) * math.factorial(n - k)))


def new_prob_total(colors: int, types: int, sizes: int) -> float:
    final_prob = 1
    for needed, possible_variants in [(colors, len(FlowerColors)), (types, len(FlowerTypes)), (sizes, len(FlowerSizes))]:
        if needed == 0:
            continue
        curr = 0
        for i in range(needed, 13):
            curr += new_prob(possible_variants, i, needed)
        final_prob *= curr

    return final_prob / 13


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

        # New bouquet setup
        self.n_flowers = max(2, math.ceil(math.log(num_suitors * days) / math.log(8)))
        self.ideal_bouquet: Bouquet = random_bouquet(self.n_flowers)

        # Figure out how to tune our function
        goal_prob = 1 / (days * (num_suitors - 1))
        results = []
        for color_flowers in range(13):
            for type_flowers in range(13):
                for size_flowers in range(13):
                    key = (color_flowers, type_flowers, size_flowers)
                    prob = new_prob_total(*key)
                    results.append((key, prob, abs(goal_prob - prob)))

        results.sort(key=lambda x: x[2])
        # Get the key from the result with the minimum difference
        color_flowers, type_flowers, size_flowers = results[0][0]
        self.color_settings = (color_flowers, np.random.choice(list(FlowerColors)))
        self.type_settings = (type_flowers, np.random.choice(list(FlowerTypes)))
        self.size_settings = (size_flowers, np.random.choice(list(FlowerSizes)))

        # JY code
        self.bouquet_data_points = {}
        for sid in range(num_suitors):
            self.bouquet_data_points[sid] = []

    def get_target_bouquets(self):
        target_bouquets = []
        for sid in range(self.num_suitors):
            if sid is self.suitor_id:
                continue
            # Sort on score
            self.bouquet_data_points[sid].sort(key=lambda x : x[1])
            total_score = 0
            for bouquet, score, rank in self.bouquet_data_points[sid]:
                total_score += score

            if len(self.bouquet_data_points[sid]) != 0:
                mean = total_score / len(self.bouquet_data_points[sid])
                best_score_fb = self.bouquet_data_points[sid][-1]
                target_bouquet, best_score, _ = best_score_fb
                player_diff = best_score - mean
                target_bouquets.append((sid, target_bouquet, player_diff))

        # Sort by difference, highest first, so we can try to construct in that order
        target_bouquets.sort(key=lambda x: x[2], reverse=True)
        return target_bouquets

    def construct_similar_bouquet(self, flower_counts: Dict[Flower, int], target_bouquet: Bouquet):
        bouquet_dict = {}
        while True:
            flowers_with_overlap = []
            for flower, count in flower_counts.items():
                if count == 0:
                    continue
                overlap = Suitor.get_overlap(flower, target_bouquet)
                if overlap > 0:
                    flowers_with_overlap.append((flower, overlap))
            if len(flowers_with_overlap) == 0:
                break
            flowers_with_overlap.sort(key=lambda x : x[1], reverse=True)
            flower_to_add = flowers_with_overlap[0][0]

            flower_counts[flower_to_add] -= 1
            self.reduce_bouquet(flower_to_add, target_bouquet)

            if flower_to_add not in bouquet_dict:
                bouquet_dict[flower_to_add] = 1
            else:
                bouquet_dict[flower_to_add] += 1
        return bouquet_dict

    def reduce_bouquet(self, flower, bouquet):
        if flower.size in bouquet.sizes and bouquet.sizes[flower.size] > 0:
            bouquet.sizes[flower.size] -= 1
        if flower.color in bouquet.colors and bouquet.colors[flower.color] > 0:
            bouquet.colors[flower.color] -= 1
        if flower.type in bouquet.types and bouquet.types[flower.type] > 0:
            bouquet.types[flower.type] -= 1

    @staticmethod
    def get_overlap(flower, bouquet):
        overlap = 0
        if flower.size in bouquet.sizes and bouquet.sizes[flower.size] > 0:
            overlap += 1
        if flower.color in bouquet.colors and bouquet.colors[flower.color] > 0:
            overlap += 1
        if flower.type in bouquet.types and bouquet.types[flower.type] > 0:
            overlap += 1
        return overlap

    def jy_prepare_final_bouquets(self, flower_counts: Dict[Flower, int]):

        # Best bouquet we saw in previous days for each suitor
        target_bouquets = self.get_target_bouquets()  # Each element form of (sid, Bouquet, player_diff)

        # Ensure every suitor gets a bouquet, even if empty. This is required for the simulator.
        ret = {n: (self.suitor_id, n, Bouquet({})) for n in range(self.num_suitors)}
        del ret[self.suitor_id]

        bouquet_dicts = []
        for sid, target_bouquet, _ in target_bouquets:
            # Bouquet similar to target bouquet but constructed from the flowers we have now
            bouquet_dicts.append((sid, self.construct_similar_bouquet(flower_counts, target_bouquet), target_bouquet))

        # Pad bouquets with additional flowers up to the length of the target bouquet
        for sid, bouquet_dict, target_bouquet in bouquet_dicts:
            while sum(bouquet_dict.values()) < len(target_bouquet.arrangement):
                found_flower = False
                for flower, count in flower_counts.items():
                    if count == 0:
                        continue
                    else:
                        found_flower = True
                        if flower not in bouquet_dict:
                            bouquet_dict[flower] = 1
                        else:
                            bouquet_dict[flower] += 1
                        flower_counts[flower] -= 1
                        break
                if not found_flower:
                    break
            ret[sid] = (self.suitor_id, sid, Bouquet(bouquet_dict))

        # Use any remaining flowers on people without a bouquet
        empty_bouquet_suitors = []
        for suitor_id, (_, _, bouquet) in ret.items():
            if sum(bouquet.arrangement.values()) == 0:
                empty_bouquet_suitors.append(suitor_id)

        flowers = []
        for flower, count in flower_counts.items():
            for _ in range(count):
                flowers.append(flower)
        np.random.shuffle(flowers)

        for flower, sid in zip(flowers, itertools.cycle(empty_bouquet_suitors)):
            _, _, bouquet = ret[sid]
            bouquet_dict = bouquet.arrangement
            if flower not in bouquet_dict:
                bouquet_dict[flower] = 1
            else:
                bouquet_dict[flower] += 1
            flower_counts[flower] -= 1
            ret[sid] = (self.suitor_id, sid, Bouquet(bouquet_dict))

        return list(ret.values())

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
            bouquets = self.jy_prepare_final_bouquets(flower_counts)
        else:
            bouquets = self.rand_man.prepare_bouquets(flower_counts)

        # Save the bouquets so we can associate them with the feedback
        self.bouquets = {}
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
        num_colors, color = self.color_settings
        num_types, ftype = self.type_settings
        num_sizes, size = self.size_settings

        colors, types, sizes = [color] * num_colors, [ftype] * num_types, [size] * num_sizes
        total_flowers = max([num_colors, num_types, num_sizes])

        def pad(arr, desired_length, pad_with):
            while len(arr) < desired_length:
                arr.append(pad_with)

        pad(colors, total_flowers, FlowerColors((color.value + 1) % (len(FlowerColors) - 1)))
        pad(types, total_flowers, FlowerTypes((ftype.value + 1) % (len(FlowerTypes) - 1)))
        pad(sizes, total_flowers, FlowerSizes((size.value + 1) % (len(FlowerSizes) - 1)))

        b = {}
        for c, t, s in zip(colors, types, sizes):
            f = Flower(size=s, color=c, type=t)
            if f in b:
                b[f] += 1
            else:
                b[f] = 1

        return Bouquet(b)

    @staticmethod
    def new_score_fn(max_possible, required: Tuple[int, Union[FlowerTypes, FlowerColors, FlowerSizes]], actual_x: Dict) -> float:
        attr_count, attr = required
        if attr in actual_x:
            diff = abs(attr_count - actual_x[attr])
            return max_possible / 2**diff
        return 0.0

    def score_x(self, max_score: float, actual_x: Dict, ideal_x: Dict) -> float:
        matching = self.n_flowers
        for x, c in ideal_x.items():
            if x not in actual_x:
                matching -= c
            elif c > actual_x[x]:
                matching += (actual_x[x] - c)
        if matching <= 0:
            return 0.0
        return max_score / 5**(self.n_flowers - matching)

    def score_types(self, types: Dict[FlowerTypes, int]):
        """
        :param types: dictionary of flower types and their associated counts in the bouquet
        :return: A score representing preference of the flower types in the bouquet
        """
        return self.new_score_fn(0.33, self.type_settings, types)

    def score_colors(self, colors: Dict[FlowerColors, int]):
        """
        :param colors: dictionary of flower colors and their associated counts in the bouquet
        :return: A score representing preference of the flower colors in the bouquet
        """
        return self.new_score_fn(0.34, self.color_settings, colors)

    def score_sizes(self, sizes: Dict[FlowerSizes, int]):
        """
        :param sizes: dictionary of flower sizes and their associated counts in the bouquet
        :return: A score representing preference of the flower sizes in the bouquet
        """
        return self.new_score_fn(0.33, self.size_settings, sizes)

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

            #JY Code
            self.bouquet_data_points[suitor_num].append((self.bouquets[suitor_num], score, rank))

        self.rand_man.receive_feedback(feedback)
