import functools
import random
from collections import Counter, defaultdict
from typing import Dict, Tuple, Callable, List, Union

from flowers import Bouquet, Flower, FlowerSizes, FlowerColors, FlowerTypes
from suitors.base import BaseSuitor
from utils import flatten_counter
import pandas as pd
from sklearn.linear_model import LinearRegression
import logging
from math import floor, inf

ALL_FEATURES = list(FlowerSizes) + list(FlowerColors) + list(FlowerTypes)
ESTIMATE_SIZE = 3
Feature = Union[FlowerTypes, FlowerColors, FlowerSizes]


def l2b(l: List[Flower]) -> Bouquet:
    return Bouquet(Counter(l))


def jaccard(our_bouquet_preference, other_preference) -> float:
    score = 0.0
    count = 0
    for key in our_bouquet_preference.keys():
        count += 1
        if key in other_preference:
            intersection = min(our_bouquet_preference[key], other_preference[key])
            union = max(our_bouquet_preference[key], other_preference[key])
            score += intersection / union

    return score / (count * 3)


def bouquet_to_dictionary(bouquet, feature=None):
    assert feature is not None

    res = defaultdict(int)
    for flower in bouquet.flowers():
        res[getattr(flower, feature)] += 1

    return res

def best_given_bouquet(bouquet_feedback) :
    #determine the best score so far
    print("inside best")
    score = -inf
    index = -1
    for i in range(len(bouquet_feedback["color"])) :
        if bouquet_feedback["color"][i]["score"] >= score :
            score = bouquet_feedback["color"][i]["score"]
            index = i

    if index == -1 :
        return {}
    
    color_weights = bouquet_feedback["color"][index]
    del color_weights["score"]
    del color_weights["rank"]

    size_weights = bouquet_feedback["size"][index]
    del size_weights["score"]
    del size_weights["rank"]

    type_weights = bouquet_feedback["type"][index]
    del type_weights["score"]
    del type_weights["rank"]

    return estimate_flowers_in_bouquets(color_weights,size_weights,type_weights)

def estimate_flowers_in_bouquets(color_weights, size_weights, type_weights):
    flowers = defaultdict(int)
    if color_weights and size_weights and type_weights:
        for c, s, t in zip(flatten_counter(color_weights), flatten_counter(size_weights),
                           flatten_counter(type_weights)):
            flowers[Flower(s, c, t)] += 1
    return flowers


def learned_bouquets(bouquet_feedback, suitor):
    res = []
    for recipient in bouquet_feedback.keys():
        color_weights = learned_weightage(bouquet_feedback[recipient], "color")
        size_weights = learned_weightage(bouquet_feedback[recipient], "size")
        type_weights = learned_weightage(bouquet_feedback[recipient], "type")

        flowers = estimate_flowers_in_bouquets(color_weights, size_weights, type_weights)
        if flowers :
            res.append((suitor, recipient,Bouquet(flowers)))
        else :
            res.append((suitor, recipient,Bouquet(best_given_bouquet(bouquet_feedback[recipient]))))
            
    return res


def learned_weightage(bouquet_feedback, factor):
    flower_for_each_round = 7
    df = pd.DataFrame(bouquet_feedback[factor]).fillna(0)
    df = df.drop("rank", 1)
    Y = df["score"]
    X = df.drop("score", 1)
    cols = X.columns
    if len(cols) == 0:
        return {}
    elif len(cols) == 1:
        return {cols[0]: flower_for_each_round}

    model = LinearRegression()
    model.fit(X, Y)
    coefficients = model.coef_

    # only consider positive elements in bouquet
    res = {}
    temp_sum = 0.0
    for i in range(len(coefficients)):
        c = coefficients[i]
        if c > 0:
            res[cols[i]] = c
            temp_sum += c

    if temp_sum == 0:
        return res

    weight = flower_for_each_round / temp_sum
    res = {k: floor(v * weight) for k, v in sorted(res.items(), key=lambda item: item[1])}
    addent = flower_for_each_round - sum(res.values())
    most_weighted_key = list(res.keys())[-1]
    res[most_weighted_key] += addent

    return res


def arrange_random(flower_counts: Dict[Flower, int]):
    bouquet_size = min(sum(flower_counts.values()), random.randint(1, 10))
    res = random.sample(flatten_counter(flower_counts), k=bouquet_size)
    flower_counts = flower_counts.copy()
    for f in res:
        flower_counts[f] -= 1
    return flower_counts, l2b(res)


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

        self.bouquet_feedback = defaultdict(lambda: dict(color=[], size=[], type=[]))
        self.logger = logging.getLogger(__name__)
        # self.favorite_bouquet = Bouquet({Flower(FlowerSizes.Large, FlowerColors.Blue, FlowerTypes.Rose): 6, Flower(FlowerSizes.Large, FlowerColors.Red, FlowerTypes.Chrysanthemum): 4})
        possible_flowers = []
        bouquet = {}
        for s in FlowerSizes:
            for t in FlowerTypes:
                for c in FlowerColors:
                    possible_flowers.append(Flower(s, c, t))
        count = random.randrange(3, 8)
        for i in range(count):
            random_flower = random.choice(possible_flowers)
            if random_flower in bouquet.keys():
                bouquet[random_flower] = bouquet[random_flower] + 1
            else:
                bouquet[random_flower] = 1
        self.favorite_bouquet = Bouquet(bouquet)
        self.recipient_ids = [i for i in range(self.num_suitors) if i != self.suitor_id]

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
            return learned_bouquets(self.bouquet_feedback, self.suitor_id)

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
            # flower_counts, send[r_id] = pick(flower_counts, functools.partial(check_feature, feature=feature_to_estimate))
            flower_counts, send[r_id] = arrange_random(flower_counts)

        self.estimate_history.append(estimation)

        # needed for estimation
        res = []
        for k, v in send.items():
            res.append((self.suitor_id, k, v))
            self.bouquet_feedback[k]["color"].append(bouquet_to_dictionary(v, "color"))
            self.bouquet_feedback[k]["size"].append(bouquet_to_dictionary(v, "size"))
            self.bouquet_feedback[k]["type"].append(bouquet_to_dictionary(v, "type"))
        return res

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
        return jaccard(our_bouquet_type, types)

    def score_colors(self, colors: Dict[FlowerColors, int]):
        """
        :param colors: dictionary of flower colors and their associated counts in the bouquet
        :return: A score representing preference of the flower colors in the bouquet
        """
        our_bouquet_color = bouquet_to_dictionary(self.favorite_bouquet, "color")
        return jaccard(our_bouquet_color, colors)

    def score_sizes(self, sizes: Dict[FlowerSizes, int]):
        """
        :param sizes: dictionary of flower sizes and their associated counts in the bouquet
        :return: A score representing preference of the flower sizes in the bouquet
        """
        our_bouquet_size = bouquet_to_dictionary(self.favorite_bouquet, "size")
        return jaccard(our_bouquet_size, sizes)

    def receive_feedback(self, feedback: List[Tuple[List[int], List[float]]]):
        """
        :param feedback:
        :return: nothing
        """
        # self.logger.info("in feedback")
        for r_id in self.recipient_ids:
            rank, score = feedback[r_id]
            self.bouquet_feedback[r_id]["color"][-1]["score"] = score
            self.bouquet_feedback[r_id]["color"][-1]["rank"] = rank
            self.bouquet_feedback[r_id]["size"][-1]["score"] = score
            self.bouquet_feedback[r_id]["size"][-1]["rank"] = rank
            self.bouquet_feedback[r_id]["type"][-1]["score"] = score
            self.bouquet_feedback[r_id]["type"][-1]["rank"] = rank
            # self.estimate_score_history[r_id][self.estimate_history[-1][r_id]].append(score)
            # self.logger.info(f'Received feedback is {self.estimate_score_history} .')
