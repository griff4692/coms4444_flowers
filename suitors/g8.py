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
        self.type_weights = [1/(self.days**2), 1/(self.days), 2/(self.days), 1]
        self.color_weights = [1/(self.days**2), 1/(self.days**2), 1/(self.days), 1/(self.days), 2/(self.days), 1]
        self.size_weights = [1/(self.days**2), 1/(self.days), 1]
        random.shuffle(self.type_weights)
        random.shuffle(self.color_weights)
        random.shuffle(self.size_weights)
        # pick one attribute to null
        self.null = random.randint(0,2) # 0=type, 1=color, 2=size
        self.given = {} # dictionary to save the bouquets we gave + their scores
        for suitor in range(0,num_suitors):
            if suitor!= suitor_id:
                self.given[suitor] = [] # list for each suitor, will be list of lists with [bouquet, score]
        self.days_left = days


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

    def best_bouquets(self, remaining_flowers, suitors, scores_per_player):
        num_remaining = sum(remaining_flowers.values())
        size = int(np.random.randint(0, min(MAX_BOUQUET_SIZE, num_remaining) + 1))
        chosen_bouquets = {}
        suitor_bouquet_counts = {}
        for suitor in range(0, suitors):
            if suitor!= self.suitor_id:
                chosen_bouquets[suitor] = {}
                suitor_bouquet_counts[suitor] = 0

        for flower in remaining_flowers:
            color = flower.color
            type = flower.type
            size = flower.size
            max_score=0
            suitor_getting_flower=0
            for suitor in range(0, suitors):
                if suitor != self.suitor_id:
                    color_score = scores_per_player[suitor]["color"][color.value]
                    type_score = scores_per_player[suitor]["type"][type.value]
                    size_score = scores_per_player[suitor]["size"][size.value]
                    total_score = color_score+type_score+size_score
                    if total_score>max_score and suitor_bouquet_counts[suitor]<12:
                        max_score=total_score
                        suitor_getting_flower=suitor
            if flower in chosen_bouquets[suitor_getting_flower]:
                chosen_bouquets[suitor_getting_flower][flower] = chosen_bouquets[suitor_getting_flower][flower]+1
            else:
                chosen_bouquets[suitor_getting_flower][flower] = 1
            suitor_bouquet_counts[suitor_getting_flower] = suitor_bouquet_counts[suitor_getting_flower] + 1

        return chosen_bouquets





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
            chosen_bouquets = self.best_bouquets(remaining_flowers, self.num_suitors, scores_per_player)
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
        if self.null == 0:
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

        if self.null == 2:
            score = 0

        # count number of flowers for the last .25
        weights_number = [0, .1, .2, .32, .48, .56, .72, .8, .88, 1, 1, 1, 1]
        score_count = weights_number[total]*.2

        return score+score_count

    def receive_feedback(self, feedback):
        """
        :param feedback:
        :return: nothing
        """
        self.feedback.append(feedback)
