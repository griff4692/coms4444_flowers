import logging
import math
import random
import warnings
from collections import Counter, defaultdict
from itertools import combinations
from math import floor, inf
from typing import Dict, Tuple, List

import pandas as pd
from sklearn.linear_model import LinearRegression

from flowers import Bouquet, Flower, FlowerSizes, FlowerColors, FlowerTypes
from suitors.base import BaseSuitor
from utils import flatten_counter

model, X, Y = {}, {}, {}


def l2b(l: List[Flower]) -> Bouquet:
    return Bouquet(Counter(l))


def bouquet_to_dictionary(bouquet: Bouquet, feature: str = None):
    assert feature is not None

    res = defaultdict(int)
    for flower in bouquet.flowers():
        res[getattr(flower, feature)] += 1

    return res


def best_given_bouquet(bouquet_feedback):
    # determine the best score so far
    score = -inf
    index = -1
    for i in range(len(bouquet_feedback["color"])):
        if bouquet_feedback["score"][i] >= score:
            score = bouquet_feedback["score"][i]
            index = i

    if index == -1:
        return {}

    color_weights = bouquet_feedback["color"][index]
    size_weights = bouquet_feedback["size"][index]
    type_weights = bouquet_feedback["type"][index]

    return {"color": color_weights, "size": size_weights, "type": type_weights}


def estimate_flowers_in_bouquets(color_weights, size_weights, type_weights, flower_counts):
    total_flowers = sum(color_weights.values()) - 2
    total_flowers = max(total_flowers, 2)
    flowers = defaultdict(int)
    if color_weights and size_weights and type_weights:
        for c, s, t in zip(flatten_counter(color_weights), flatten_counter(size_weights),
                           flatten_counter(type_weights)):
            f = Flower(s, c, t)
            if flower_counts.get(f, 0) > 0:
                flower_counts[f] -= 1
                flowers[f] += 1
                total_flowers -= 1

    for f in flower_counts.keys():
        if total_flowers == 0:
            break

        if flower_counts[f] > 0:
            total_flowers -= 1
            flower_counts[f] -= 1
            flowers[f] += 1

    return flower_counts, flowers


def learned_bouquets(bouquet_feedback, suitor, flower_counts, recipient_ids):
    res = []

    if suitor in recipient_ids:
        recipient_ids.remove(suitor)
    for recipient in recipient_ids:
        flower_for_each_round = random.randrange(2, 10)
        flowers = learned_weightage(bouquet_feedback[recipient], flower_for_each_round)
        if flowers.keys():
            # print(flowers)
            flower_counts, r = estimate_flowers_in_bouquets(flowers["color"], flowers["size"], flowers["type"], flower_counts)
            res.append((suitor, recipient, Bouquet(r)))
        else:
            res.append((suitor, recipient, Bouquet({})))

    return flower_counts, res


def learned_weightage(bouquet_feedback, flower_for_each_round):
    # print("bouquet feedback")
    # print(bouquet_feedback)
    global X
    global Y
    global model
    flowers1 = {}
    flowers2 = best_given_bouquet(bouquet_feedback)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        for factor in ["color", "size", "type"]:
            df = pd.DataFrame(bouquet_feedback[factor]).fillna(0)
            temp_sum = sum(bouquet_feedback["score"])
            Y[factor] = bouquet_feedback["score"]
            if temp_sum == 0:
                Y[factor] = bouquet_feedback["rank"]
            X[factor] = df
            cols = X[factor].columns
            if len(cols) == 0:
                flowers1 = {"color": {}, "size": {}, "type": {}}
                return flowers1;
            else:
                model[factor] = LinearRegression()
                model[factor].fit(X[factor], Y[factor])
                coefficients = model[factor].coef_

                # only consider positive elements in bouquet
                res = {}
                max_neg = inf
                for i in range(len(coefficients)):
                    c = coefficients[i]
                    res[cols[i]] = c
                    max_neg = min(max_neg, c)

                if max_neg < 0:
                    for key in res.keys():
                        res[key] -= max_neg
                temp_sum = sum(res.values())

                if temp_sum == 0.0:
                    return flowers2

                weight = flower_for_each_round / temp_sum
                res = {k: floor(v * weight) for k, v in sorted(res.items(), key=lambda item: item[1])}
                addent = flower_for_each_round - sum(res.values())
                most_weighted_key = list(res.keys())[-1]
                res[most_weighted_key] += addent

                flowers1[factor] = res

        testX = {}
        score_flowers2, score_flowers1 = 0.0, 0.0
        for key in ["color", "size", "type"]:
            X[key] = X[key].append([flowers1[key], flowers2[key]]).fillna(0)
            testX[key] = X[key].tail(2)
            score = model[key].predict(testX[key])
            score_flowers1 += score[0]
            score_flowers2 += score[1]

        if score_flowers2 > score_flowers1:
            return flowers2
        return flowers1


def arrange_random(flower_counts: Dict[Flower, int], offered_bouquet_sizes):
    if len(offered_bouquet_sizes) == 12:
        offered_bouquet_sizes.clear()
    random_range = list(set(range(1, 13)) - set(offered_bouquet_sizes))
    bouquet_size = min(sum(flower_counts.values()), random.choice(random_range))
    # bouquet_size = ceil(np.random.normal(loc=6.5, scale=2.5))
    # bouquet_size = 1 if bouquet_size < 1 else 12 if bouquet_size > 12 else bouquet_size
    # bouquet_size = min(sum(flower_counts.values()), bouquet_size)
    offered_bouquet_sizes.append(bouquet_size)
    res = random.sample(flatten_counter(flower_counts), k=bouquet_size)
    flower_counts = flower_counts.copy()
    for f in res:
        flower_counts[f] -= 1
    return flower_counts, l2b(res)


def generate_similar_bouquet(flower_counts: Dict[Flower, int], desired_bouquet):
    desired_bouquet_size = len(desired_bouquet.flowers())
    desired_size = bouquet_to_dictionary(desired_bouquet, "size")
    desired_type = bouquet_to_dictionary(desired_bouquet, "type")
    desired_color = bouquet_to_dictionary(desired_bouquet, "color")
    best_score = 0
    best_bouquet = None
    for size in range(desired_bouquet_size, sum(flower_counts.values()), -1):
        possible_bouquets = combinations(flatten_counter(flower_counts), size)
        for bouquet in possible_bouquets:
            bouquet_size = bouquet_to_dictionary(bouquet, "size")
            bouquet_type = bouquet_to_dictionary(bouquet, "type")
            bouquet_color = bouquet_to_dictionary(bouquet, "color")
            size_score = jaccard(desired_size, bouquet_size)
            type_score = jaccard(desired_type, bouquet_type)
            color_score = jaccard(desired_color, bouquet_color)
            score = size_score + type_score + color_score
            if score == 1 or (1 - score) < 0.00001:
                return bouquet
            elif score > best_score:
                best_score = score
                best_bouquet = bouquet

    return best_bouquet


def priority(recipient_ids, bouquet_feedback, final_round=False):
    """
    :param recipient_ids: list of recipient ids currently in game
    :param bouquet_feedback: dict of feedback o previously given bouquets
    :param final_round: determines whether solved players (where we scored 1 on) are a priority or not

    :return: list of recipient ids in order of priority
    """
    ordering = [(r_id,
                 bouquet_feedback[r_id]["score"][-1],
                 bouquet_feedback[r_id]["rank"][-1]) for r_id in recipient_ids]
    # first pass: sort by rank and then by score
    ordering = sorted(ordering, key=lambda y: (y[2], y[1]))
    # second pass: move players we score 0 on to back
    for i in range(len(ordering)):
        if ordering[-i][1] == 0:
            ordering.append(ordering.pop(-i))
    # if not the final round, move players we score 1 on to back as well
    if not final_round:
        for i in range(len(ordering)):
            if ordering[-i][1] == 1:
                ordering.append(ordering.pop(-i))
    new_r_ids = [r_id for r_id, _, _ in ordering]
    return new_r_ids


class Suitor(BaseSuitor):
    def __init__(self, days: int, num_suitors: int, suitor_id: int):
        """
        :param days: number of days of courtship
        :param num_suitors: number of suitors, including yourself
        :param suitor_id: unique id of your suitor in range(num_suitors)
        """
        super().__init__(days, num_suitors, suitor_id, name='g3')
        self.day_count = 0
        self.total_days = days
        factor = 3 if days <= 7 else 2
        self.first_pruning = days // factor
        if self.first_pruning <= 2 and self.total_days >= 3:
            self.first_pruning = self.total_days
        self.logger = logging.getLogger(__name__)
        possible_flowers = []
        bouquet = defaultdict(int)
        for s in FlowerSizes:
            for t in FlowerTypes:
                for c in FlowerColors:
                    possible_flowers.append(Flower(s, c, t))
        count = 6
        for i in range(count):
            random_flower = random.choice(possible_flowers)
            bouquet[random_flower] += 1
        self.favorite_bouquet = Bouquet(bouquet)
        self.recipient_ids = [i for i in range(self.num_suitors) if i != self.suitor_id]
        self.offered_bouquet_sizes = {id: [] for id in self.recipient_ids}
        self.queue = self.recipient_ids.copy()
        random.shuffle(self.queue)
        self.priority_queue = self.recipient_ids.copy()
        self.bouquet_feedback = {r_id: {"color": [], "size": [], "type": [], "rank": [], "score": []} for r_id in
                                 self.recipient_ids}

        s_type = self.score_types(bouquet_to_dictionary(self.favorite_bouquet, "type"))
        s_color = self.score_colors(bouquet_to_dictionary(self.favorite_bouquet, "color"))
        s_size = self.score_sizes(bouquet_to_dictionary(self.favorite_bouquet, "size"))
        assert math.fabs(s_type + s_color + s_size - 1) < 0.01  # sanity check for score one

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

        if self.day_count >= self.first_pruning > 2:
            flower_counts, bouquets = learned_bouquets(self.bouquet_feedback, self.suitor_id, flower_counts.copy(),
                                                       priority(self.recipient_ids, self.bouquet_feedback))

        else:
            bouquets = list()
            for r_id in self.queue:
                if sum(flower_counts.values()) == 0:
                    b = Bouquet(dict())
                else:
                    flower_counts, b = arrange_random(flower_counts, self.offered_bouquet_sizes[r_id])
                if self.day_count < self.first_pruning:
                    self.queue.append(self.queue.pop(0))
                bouquets.append((self.suitor_id, r_id, b))
                self.recipient_ids.append(self.recipient_ids.pop(0))

        for b in bouquets:
            _, r, v = b
            self.bouquet_feedback[r]["color"].append(bouquet_to_dictionary(v, "color"))
            self.bouquet_feedback[r]["size"].append(bouquet_to_dictionary(v, "size"))
            self.bouquet_feedback[r]["type"].append(bouquet_to_dictionary(v, "type"))

        return bouquets

    def jaccard(self, our_bouquet_preference, other_preference) -> float:
        def scale(x: float) -> float:
            estimate_max = 0.72 + self.num_suitors / 2 * 0.01 + self.days / 100 * 0.01
            return min(1.0, x / estimate_max) ** 2

        score = 0.0
        count = 0
        for key in our_bouquet_preference.keys():
            count += 1
            if key in other_preference:
                intersection = min(our_bouquet_preference[key], other_preference[key])
                union = max(our_bouquet_preference[key], other_preference[key])
                score += intersection / union
        return scale(score / count) / 3

    def zero_score_bouquet(self):
        """
        :return: a Bouquet for which your scoring function will return 0
        """
        return Bouquet(dict())

    def one_score_bouquet(self):
        """
        :return: a Bouquet for which your scoring function will return 1
        """
        return self.favorite_bouquet

    def score_types(self, types: Dict[FlowerTypes, int]):
        """
        :param types: dictionary of flower types and their associated counts in the bouquet
        :return: A score representing preference of the flower types in the bouquet
        """
        # self.logger.info(self.favorite_bouquet.types, types)
        our_bouquet_type = bouquet_to_dictionary(self.favorite_bouquet, "type")
        return self.jaccard(our_bouquet_type, types)

    def score_colors(self, colors: Dict[FlowerColors, int]):
        """
        :param colors: dictionary of flower colors and their associated counts in the bouquet
        :return: A score representing preference of the flower colors in the bouquet
        """
        our_bouquet_color = bouquet_to_dictionary(self.favorite_bouquet, "color")
        return self.jaccard(our_bouquet_color, colors)

    def score_sizes(self, sizes: Dict[FlowerSizes, int]):
        """
        :param sizes: dictionary of flower sizes and their associated counts in the bouquet
        :return: A score representing preference of the flower sizes in the bouquet
        """
        our_bouquet_size = bouquet_to_dictionary(self.favorite_bouquet, "size")
        return self.jaccard(our_bouquet_size, sizes)

    def receive_feedback(self, feedback: List[Tuple[int, float, int]]):
        """
        :param feedback:
        :return: nothing
        """
        l = len(self.recipient_ids)
        for r_id in self.recipient_ids:
            rank, score, ties = feedback[r_id]
            self.bouquet_feedback[r_id]["score"].append(score)
            self.bouquet_feedback[r_id]["rank"].append(l - (rank + ties - 1))
