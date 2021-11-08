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
from math import floor

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


def jaccard(our_bouquet_preference, other_preference) -> float :
    score = 0.0
    count = 0
    for key in our_bouquet_preference.keys() :
        count += 1
        if other_preference.get(key) != None :
            intersection = min(our_bouquet_preference[key], other_preference[key])
            union = max(our_bouquet_preference[key], other_preference[key])
            score += intersection/union

    return score/(count*3)

def bouquet_to_dictionary(bouquet, color=0, size=0, typ=0) :
    res = {}

    if color!=0 :
        for flower in bouquet.flowers() :
            if res.get(flower.color)!=None :
                res[flower.color] += 1
            else :
                res[flower.color] = 1

    elif size!=0 :
        for flower in bouquet.flowers() :
            if res.get(flower.size)!=None :
                res[flower.size] += 1
            else :
                res[flower.size] = 1

    elif typ!=0 :
        for flower in bouquet.flowers() :
            if res.get(flower.type)!=None :
                res[flower.type] += 1
            else :
                res[flower.type] = 1

    return res
            
def create_bouquests(flower_for_this_round, color_weights, size_weights, type_weights) :
    flowers = {}
    '''print(color_weights)
    print(size_weights)
    print(type_weights)'''
    color_cols, color_counts = list(color_weights.keys()), list(color_weights.values())
    size_cols, size_counts = list(size_weights.keys()), list(size_weights.values())
    type_cols, type_counts = list(type_weights.keys()), list(type_weights.values())

    for i in range(flower_for_this_round) :
        if color_counts[-1] == 0 :
            color_counts.pop()
            color_cols.pop()
        
        if size_counts[-1] == 0 :
            size_counts.pop()
            size_cols.pop()
        
        if type_counts[-1] == 0 :
            type_counts.pop()
            type_cols.pop()

        temp = Flower(size_cols[-1], color_cols[-1], type_cols[-1])
        if flowers.get(temp) == None :
            flowers[temp] = 1
        else :
            flowers[temp] += 1

        color_counts[-1] -= 1
        size_counts[-1] -= 1
        type_counts[-1] -= 1

    #print(flowers)
    return Bouquet(flowers)

def learned_bouquets(bouquet_feedback, suitor) :
    res = []
    for recipient in bouquet_feedback.keys() :
        color_weights = learned_weightage(bouquet_feedback[recipient], "color")
        size_weights = learned_weightage(bouquet_feedback[recipient], "size")
        type_weights = learned_weightage(bouquet_feedback[recipient], "type")

        res.append((suitor, recipient, create_bouquests(sum(color_weights.values()), color_weights, size_weights, type_weights)))
    return res

def learned_weightage(bouquet_feedback, factor) :
    flower_for_each_round = 7
    df = pd.DataFrame(bouquet_feedback[factor]).fillna(0)
    df = df.drop("rank", 1)
    Y = df["score"]
    X = df.drop("score", 1)
    cols = X.columns
    if len(cols) == 0 :
        return {}
    elif len(cols) == 1 :
        return {cols[0] : flower_for_each_round}


    model = LinearRegression()
    model.fit(X,Y)
    coefficients = model.coef_

    #only consider positive elements in bouquet
    res = {}
    temp_sum = 0.0
    for i in range(len(coefficients)) :
        c = coefficients[i]
        if c>0 :
            res[cols[i]] = c
            temp_sum += c
    
    weight = flower_for_each_round/temp_sum
    res = {k: floor(v*weight) for k, v in sorted(res.items(), key=lambda item: item[1])}
    addent = flower_for_each_round - sum(res.values())
    most_weighted_key = list(res.keys())[-1]
    res[most_weighted_key] += addent

    return res

def arrange_random (flower_counts: Dict[Flower, int]):
    bouquet_size = random.randrange(1, 10)
    #print("size "+str(bouquet_size))
    possible_options = list(flower_counts.keys())
    bouquet = {}
    counter = 0

    while counter < bouquet_size:
        flower = random.choice(possible_options)
        if flower_counts[flower] > 0:
            counter = counter+1
            if flower in bouquet.keys():
                bouquet[flower] = bouquet[flower] + 1
            else:
                bouquet[flower] = 1
            flower_counts[flower] = flower_counts[flower] -1
            
    return flower_counts, Bouquet(bouquet)




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

        self.bouquet_feedback = {}
        self.logger = logging.getLogger(__name__)
        #self.favorite_bouquet = Bouquet({Flower(FlowerSizes.Large, FlowerColors.Blue, FlowerTypes.Rose): 6, Flower(FlowerSizes.Large, FlowerColors.Red, FlowerTypes.Chrysanthemum): 4})
        possible_flowers = []
        bouquet = {}
        for s in FlowerSizes:
            for t in FlowerTypes:
                for c in FlowerColors:
                    possible_flowers.append(Flower(s, c, t))
        count = random.randrange(1, 12)
        for i in range(count):
            random_flower = random.choice(possible_flowers)
            if random_flower in bouquet.keys():
                bouquet[random_flower] = bouquet[random_flower] +1
            else:
                bouquet[random_flower] = 1
        self.favorite_bouquet = Bouquet(bouquet) 
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
            return learned_bouquets(self.bouquet_feedback, self.suitor_id)
            #return self.solve_final(flower_counts)

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
            #flower_counts, send[r_id] = pick(flower_counts, functools.partial(check_feature, feature=feature_to_estimate))
            flower_counts, send[r_id] = arrange_random(flower_counts)

        self.estimate_history.append(estimation)

        ##needed for estimation
        res = []
        for k, v in send.items() :
            res.append((self.suitor_id, k, v))
            if self.bouquet_feedback.get(k) == None :
                self.bouquet_feedback[k] = {"color" : [bouquet_to_dictionary(v, color=1)],  "size" : [bouquet_to_dictionary(v, size=1)], "type" : [bouquet_to_dictionary(v, typ=1)]}
            else :
                self.bouquet_feedback[k]["color"].append(bouquet_to_dictionary(v, color=1))
                self.bouquet_feedback[k]["size"].append(bouquet_to_dictionary(v, size=1)) 
                self.bouquet_feedback[k]["type"].append(bouquet_to_dictionary(v, typ=1))
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
        #self.logger.info(self.favorite_bouquet.types, types)
        our_bouquet_type = bouquet_to_dictionary(self.favorite_bouquet, typ=1)
        return jaccard(our_bouquet_type,types)

    def score_colors(self, colors: Dict[FlowerColors, int]):
        """
        :param colors: dictionary of flower colors and their associated counts in the bouquet
        :return: A score representing preference of the flower colors in the bouquet
        """
        our_bouquet_color = bouquet_to_dictionary(self.favorite_bouquet, color=1)
        return jaccard(our_bouquet_color,colors)

    def score_sizes(self, sizes: Dict[FlowerSizes, int]):
        """
        :param sizes: dictionary of flower sizes and their associated counts in the bouquet
        :return: A score representing preference of the flower sizes in the bouquet
        """
        our_bouquet_size = bouquet_to_dictionary(self.favorite_bouquet, size=1)
        return jaccard(our_bouquet_size,sizes)
        

    def receive_feedback(self, feedback: List[Tuple[List[int], List[float]]]):
        """
        :param feedback:
        :return: nothing
        """
        #self.logger.info("in feedback")
        for r_id in self.recipient_ids:
            rank, score = feedback[r_id]
            self.bouquet_feedback[r_id]["color"][-1]["score"] = score
            self.bouquet_feedback[r_id]["color"][-1]["rank"] = rank
            self.bouquet_feedback[r_id]["size"][-1]["score"] = score
            self.bouquet_feedback[r_id]["size"][-1]["rank"] = rank
            self.bouquet_feedback[r_id]["type"][-1]["score"] = score
            self.bouquet_feedback[r_id]["type"][-1]["rank"] = rank
            #self.estimate_score_history[r_id][self.estimate_history[-1][r_id]].append(score)
            #self.logger.info(f'Recieved feedback is {self.estimate_score_history} .')
        
        

    
