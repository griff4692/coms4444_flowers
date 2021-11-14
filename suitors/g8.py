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
        if self.days > 1:
            self.type_weights = [1/(self.days**2), 1/(self.days), 2/(self.days), 1]
            self.color_weights = [1/(self.days**2), 1/(self.days**2), 1/(self.days), 1/(self.days), 2/(self.days), 1]
            self.size_weights = [1/(self.days**2), 1/(self.days), 1]
            random.shuffle(self.type_weights)
            random.shuffle(self.color_weights)
            random.shuffle(self.size_weights)
            # pick one attribute to null
            self.null = random.randint(0,2) # 0=type, 1=color, 2=size
        elif self.days==1:
            self.color_weights = [1 / (self.num_suitors ** 2), 1 / (self.num_suitors ** 2), 1 / (self.num_suitors), 1 / (self.num_suitors),
                                  2 / (self.num_suitors), 1]
            self.null = 3 # so its ignored by scoring methods
        self.given = {} # dictionary to save the bouquets we gave + their scores
        all_ids = np.arange(self.num_suitors)
        self.recipient_ids = all_ids[all_ids != self.suitor_id]
        for suitor in self.recipient_ids:
            self.given[suitor] = [] # list for each suitor, will be list of lists with [bouquet, score]
        self.days_left = days
        self.day_number = 0
        self.prefs = {}
        self.initialize_preferences()
        self.controlled_strat = {}
        self.initalize_controlled_strat()
        self.suitors_to_test = list(self.recipient_ids)


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

    def best_bouquets(self, remaining_flowers, scores_per_player):
        flowers_in_market = flatten_counter(remaining_flowers)
        chosen_bouquets = {}
        suitor_bouquet_counts = {}
        for suitor in self.recipient_ids:
            chosen_bouquets[suitor] = {}
            suitor_bouquet_counts[suitor] = 0

        for flower in flowers_in_market:
            color = flower.color
            type = flower.type
            size = flower.size
            max_score=0
            suitor_getting_flower=0
            for suitor in self.recipient_ids:
                color_score = scores_per_player[suitor]["color"][color.value]
                type_score = scores_per_player[suitor]["type"][type.value]
                size_score = scores_per_player[suitor]["size"][size.value]
                total_score = color_score+type_score+size_score
                if total_score>max_score and suitor_bouquet_counts[suitor]<7:
                    max_score=total_score
                    suitor_getting_flower=suitor
            if flower in chosen_bouquets[suitor_getting_flower]:
                chosen_bouquets[suitor_getting_flower][flower] = chosen_bouquets[suitor_getting_flower][flower]+1
            else:
                chosen_bouquets[suitor_getting_flower][flower] = 1
            suitor_bouquet_counts[suitor_getting_flower] = suitor_bouquet_counts[suitor_getting_flower] + 1

        return chosen_bouquets

    def convert_prefs_to_scores_per_player(self):
        scores_by_attribute = {}
        colors = [c.name for c in FlowerColors]
        sizes = [s.name for s in FlowerSizes]
        types = [t.name for t in FlowerTypes]
        for suitor, prefs in self.prefs.items():
            preferences = {}
            preferences["color"] = [0]*6
            preferences["type"] = [0]*4
            preferences["size"] = [0]*3
            for param, val in prefs.items():
                if param in colors:
                    i = colors.index(param)
                    preferences["color"][i] = val
                    continue
                if param in types:
                    i = types.index(param)
                    preferences["type"][i] = val
                    continue
                if param in sizes:
                    i = sizes.index(param)
                    preferences["size"][i] = val
            scores_by_attribute[suitor] = preferences
        return scores_by_attribute

    def scores_per_player(self, given):
        scores_by_attribute = {} # will be a dictionary of dictionaries, one per suitor, dictionaries contain info on scores per attribute
        for suitor in given:
            preferences = {}
            preferences["color"] = [[],[],[],[],[],[]]
            preferences["type"] = [[],[],[],[]]
            preferences["size"] = [[],[],[]]
            preferences["number"] = [[],[],[],[],[],[],[],[],[],[],[],[],[]]
            list_of_bouquets_and_scores = given[suitor]
            # first, list of scores for every time a specific attribute occured
            for round in list_of_bouquets_and_scores:
                bouquet = round[0]
                score = round[1]
                number = len(bouquet)
                preferences["number"][number].append(score)
                for flower in bouquet.flowers():
                    preferences["color"][flower.color.value].append(score)
                    preferences["type"][flower.type.value].append(score)
                    preferences["size"][flower.size.value].append(score)

            # next, average the lists so we know average score per attribute
            for i,color in enumerate(preferences["color"]):
                if len(preferences["color"][i]) != 0:
                    preferences["color"][i] = sum(preferences["color"][i])/len(preferences["color"][i])
                else:
                    preferences["color"][i] = 0
            for i,type in enumerate(preferences["type"]):
                if len(preferences["type"][i]) != 0:
                    preferences["type"][i] = sum(preferences["type"][i]) / len(preferences["type"][i])
                else:
                    preferences["type"][i] = 0
            for i,size in enumerate(preferences["size"]):
                if len(preferences["size"][i]) != 0:
                    preferences["size"][i] = sum(preferences["size"][i]) / len(preferences["size"][i])
                else:
                    preferences["size"][i] = 0
            for i,n in enumerate(preferences["number"]):
                if len(preferences["number"][i]) != 0:
                    preferences["number"][i] = sum(preferences["number"][i]) / len(preferences["number"][i])
                else:
                    preferences["number"][i] = 0

            scores_by_attribute[suitor]=preferences

        return scores_by_attribute

    def prepare_bouquets(self, flower_counts: Dict[Flower, int]):

        """:param flower_counts: flowers and associated counts for for available flowers
        :return: list of tuples of (self.suitor_id, recipient_id, chosen_bouquet)
        the list should be of length len(self.num_suitors) - 1 because you should give a bouquet to everyone
         but yourself

        To get the list of suitor ids not including yourself, use the following snippet:"""

        self.day_number += 1
        # if self.day_number == self.days:
        #     scores_per_player = self.convert_prefs_to_scores_per_player()
        #     # scores_per_player is already excluding us, and is in order of the other suitors (index = suitor)
        #     chosen_bouquets = self.best_bouquets(flower_counts.copy(), scores_per_player)
        #     bouquets = []
        #     for suitor in chosen_bouquets:
        #         bouquets.append((self.suitor_id, suitor, Bouquet(chosen_bouquets[suitor])))
        #     print(str(self.suitor_id) + str(self.size_weights) +str(self.type_weights) + str(self.color_weights))
        #     return bouquets
        # if self.suitors_to_test:
        #     return self.use_controlled_strategy(flower_counts.copy())
        # return {}



        # this loop inputs the score for each bouquet we gave last round
        if len(self.feedback)!=0:
            this_rounds_feedback = self.feedback[-1]
            for i,rank_score in enumerate(this_rounds_feedback): # rank_score is tuple with rank,score,ties
                if i!=self.suitor_id:
                    score = rank_score[1]
                    suitor = i
                    self.given[suitor][-1][1] = score

        all_ids = np.arange(self.num_suitors)
        recipient_ids = all_ids[all_ids != self.suitor_id]
        remaining_flowers = flower_counts.copy()


        # on the last day prepare special bouquets
        if self.days_left==1:
            # list of (self.suitor_id, recipient_id, chosen_bouquet)
            scores_per_player = self.scores_per_player(self.given) # list of dictionaries {number: [number preferences], color: [color preferences], etc.}
            # scores_per_player is already excluding us, and is in order of the other suitors (index = suitor)
            chosen_bouquets = self.best_bouquets(remaining_flowers, scores_per_player)
            bouquets = []
            for suitor in chosen_bouquets:
                bouquets.append((self.suitor_id, suitor, Bouquet(chosen_bouquets[suitor])))
        else:
            # this loop saves the bouquets so next round we can see the scores
            bouquets = list(
                map(lambda recipient_id: self._prepare_bouquet(remaining_flowers, recipient_id), recipient_ids))
            for bouquet in bouquets:
                player_given_to = bouquet[1]
                actual_bouquet = bouquet[2]
                self.given[player_given_to].append([actual_bouquet, 0])
            self.days_left=self.days_left-1



        return bouquets

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
        if self.null == 0 or self.days == 1:
            return 0

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

        # multiply by .4 since each type, sixe, color is .4 weight but one is dropped + .2 number
        return score*.4

    def score_colors(self, colors: Dict[FlowerColors, int]):
        """
        :param colors: dictionary of flower colors and their associated counts in the bouquet
        :return: A score representing preference of the flower colors in the bouquet
        """
        if self.null == 1:
            return 0

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

        # color only one if days = 1
        if self.days == 1:
            return score
        # multiply by .4 since each type, sixe, color is .4 weight but one is dropped + .2 number
        return score * .4

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

        # multiply by .4 since each type, sixe, color is .4 weight but one is dropped + .2 number
        score = score * .4

        if self.null == 2 or self.days==1:
            score = 0

        # count number of flowers for the last .25
        weights_number = [0, .05, .18, .32, .48, .72, .85, 1, 1, 1, 1, 1, 1]
        score_count = weights_number[total]*.2

        return score+score_count

    def receive_feedback(self, feedback):
        """
        :param feedback:
        :return: nothing
        """
        self.feedback.append(feedback)

    def initialize_preferences(self):
        for suitor in self.recipient_ids:
            self.prefs[suitor] = {
                "Small" : 0,
                "Medium" : 0,
                "Large" : 0,
                "White" : 0,
                "Yellow" : 0,
                "Red" : 0,
                "Purple" : 0,
                "Orange" : 0,
                "Blue" : 0,
                "Rose" : 0,
                "Chrysanthemum" : 0,
                "Tulip" : 0,
                "Begonia" : 0,
                "None" : 0
            }

    def initalize_controlled_strat(self):
        for suitor in self.recipient_ids:
            self.controlled_strat[suitor] = {
                "colors" : [c for c in FlowerColors],
                "types" : [t for t in FlowerTypes],
                "sizes" : [s for s in FlowerSizes],
                "const_params_color" : [],
                "const_params_type": [],
                "const_params_size": [],
                "tested_param" : None
            }

    def start_controlled_color(self, suitor, flower):
        self.controlled_strat[suitor]["colors"].remove(flower.color)
        self.controlled_strat[suitor]["const_params_color"] = [flower.size, flower.type]
        self.controlled_strat[suitor]["tested_param"] = flower.color.name

    def start_controlled_type(self, suitor, flower):
        self.controlled_strat[suitor]["types"].remove(flower.type)
        self.controlled_strat[suitor]["const_params_type"] = [flower.color, flower.size]
        self.controlled_strat[suitor]["tested_param"] = flower.type.name

    def start_controlled_size(self, suitor, flower):
        self.controlled_strat[suitor]["sizes"].remove(flower.size)
        self.controlled_strat[suitor]["const_params_size"] = [flower.color, flower.type]
        self.controlled_strat[suitor]["tested_param"] = flower.size.name

    def update_preferences(self):
        if self.day_number > 1:
            latest_feedback = self.feedback[-1]
            for suitor in self.suitors_to_test:
                self.prefs[suitor][self.controlled_strat[suitor]["tested_param"]] = latest_feedback[suitor][1]

    def get_flowers_to_test(self, flowers_in_market, suitor):
        all_tested_flag = True
        for color in self.controlled_strat[suitor]["colors"]:
            all_tested_flag = False
            reqd_flower = Flower(
                size=self.controlled_strat[suitor]["const_params_color"][0],
                color=color,
                type=self.controlled_strat[suitor]["const_params_color"][1]
            )
            if reqd_flower in flowers_in_market:
                self.controlled_strat[suitor]["colors"].remove(color)
                self.controlled_strat[suitor]["tested_param"] = color.name
                return reqd_flower, all_tested_flag

        for type in self.controlled_strat[suitor]["types"]:
            all_tested_flag = False
            reqd_flower = Flower(
                size=self.controlled_strat[suitor]["const_params_type"][1],
                color=self.controlled_strat[suitor]["const_params_type"][0],
                type=type
            )
            if reqd_flower in flowers_in_market:
                self.controlled_strat[suitor]["types"].remove(type)
                self.controlled_strat[suitor]["tested_param"] = type.name
                return reqd_flower, all_tested_flag

        for size in self.controlled_strat[suitor]["sizes"]:
            all_tested_flag = False
            reqd_flower = Flower(
                size=size,
                color=self.controlled_strat[suitor]["const_params_size"][0],
                type=self.controlled_strat[suitor]["const_params_size"][1]
            )
            if reqd_flower in flowers_in_market:
                self.controlled_strat[suitor]["sizes"].remove(size)
                self.controlled_strat[suitor]["tested_param"] = size.name
                return reqd_flower, all_tested_flag
        self.controlled_strat[suitor]["tested_param"] = "None"
        return None, all_tested_flag

    def use_controlled_strategy(self, remaining_flowers):
        chosen_bouquets = {}
        flowers_in_market = flatten_counter(remaining_flowers)
        self.update_preferences()
        for suitor in list(self.suitors_to_test):
            all_tested = False
            if self.day_number == 1:
                chosen_flower = np.random.choice(list(set(flowers_in_market)))
                self.start_controlled_color(suitor, chosen_flower)
            elif self.day_number == 2:
                chosen_flower = np.random.choice(list(set(flowers_in_market)))
                self.start_controlled_type(suitor, chosen_flower)
            elif self.day_number == 3:
                chosen_flower = np.random.choice(list(set(flowers_in_market)))
                self.start_controlled_size(suitor, chosen_flower)
            else:
                chosen_flower, all_tested = self.get_flowers_to_test(flowers_in_market, suitor)

            if chosen_flower is not None:
                chosen_bouquets[suitor] = chosen_flower
                flowers_in_market.remove(chosen_flower)
            if all_tested:
                self.suitors_to_test.remove(suitor)

        bouquets = []
        for k, v in chosen_bouquets.items():
            bouquets.append((self.suitor_id, k, Bouquet({v: 1})))

        return bouquets
