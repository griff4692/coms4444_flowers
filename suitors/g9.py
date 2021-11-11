from typing import Dict
from collections import Counter

import numpy as np
import random
import statistics
import copy
from constants import MAX_BOUQUET_SIZE
from flowers import Bouquet, Flower, FlowerSizes, FlowerColors, FlowerTypes
from suitors.base import BaseSuitor
from suitors import random_suitor
from utils import flatten_counter

class Suitor(BaseSuitor):
    def __init__(self, days: int, num_suitors: int, suitor_id: int):
        """
        :param days: number of days of courtship
        :param num_suitors: number of suitors, including yourself
        :param suitor_id: unique id of your suitor in range(num_suitors)
        """
        super().__init__(days, num_suitors, suitor_id, name='g9')
        self.bouquets = {} # dictionary with the bouquet we gave to each player in a given round along with the score we received
        self.all_bouquets = {} # dictionary with the bouquet we gave to each player in each round along with the score we received
        self.current_day = 1 # keep track of the current day, so that we know how many days are left
        self.all_bouquets_by_element = {} # {playerNo:[({type: quantity_in_bouquet},{color: quantity_in_bouquet},{size: quantity_in_bouquet}, score)]}
        temp = self.random_sequence(6)
        self.color_score = [FlowerColors(i) for i in temp]
        self.generate_type_sequence()
        temp = self.random_sequence(3)
        self.size_score = [FlowerSizes(i) for i in temp]

    def random_sequence(self, n):
        sequence = np.arange(n)
        np.random.shuffle(sequence)
        return sequence
    def generate_type_sequence(self):
        self.type_score = {}
        self.max_type_sequence = []
        sequence = np.arange(1820) # 1820 different combination of flower types in total
        np.random.shuffle(sequence[1:])
        count = 0
        for i1 in range(0,13):
            for i2 in range(0,13-i1):
                for i3 in range(0, 13 - i1-i2):
                    for i4 in range(0, 13 - i1-i2-i3):
                        if sequence[count]==1819 and i1+i2+i3+i4<12:
                            temp = sequence[count]
                            sequence[count] = sequence[count+1]
                            sequence[count+1] = temp
                        self.type_score[(i1,i2,i3,i4)] = sequence[count]
                        if sequence[count]==1819:
                            for i in range(12):
                                if i<i1:
                                    self.max_type_sequence.append(0)
                                    continue
                                if i<i1+i2:
                                    self.max_type_sequence.append(1)
                                    continue
                                if i<i1+i2+i3:
                                    self.max_type_sequence.append(2)
                                    continue
                                self.max_type_sequence.append(3)
                        count+=1
                        
    def _prepare_bouquet_first_day(self, remaining_flowers, recipient_id):
        num_remaining = sum(remaining_flowers.values())
        size = 6 # in the firrst day give everyone 6 flowers
        chosen_flowers = np.random.choice(flatten_counter(remaining_flowers), size=(size, ), replace=False)
        chosen_flower_counts = dict(Counter(chosen_flowers))
        for k, v in chosen_flower_counts.items():
            remaining_flowers[k] -= v
            assert remaining_flowers[k] >= 0
        chosen_bouquet = Bouquet(chosen_flower_counts)
        # store the bouquet we gave to each player in this round along with score 0
        # the score will be updated when we get the feedback
        self.all_bouquets[recipient_id] = [(chosen_bouquet, 0)]
        colors, types, sizes = self.bouquet_to_elements(chosen_bouquet)
        self.all_bouquets_by_element[recipient_id] = [(types, colors, sizes, 0)]
        return self.suitor_id, recipient_id, chosen_bouquet

    def _prepare_bouquet_intermediate_day(self, remaining_flowers, recipient_id):
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
        self.all_bouquets[recipient_id].append((chosen_bouquet, 0)) # store the bouquet we gave to each player in this round 
        colors, types, sizes = self.bouquet_to_elements(chosen_bouquet)
        self.all_bouquets_by_element[recipient_id].append((types, colors, sizes, 0))
        return self.suitor_id, recipient_id, chosen_bouquet
    
    def _prepare_bouquet_last_day(self, remaining_flowers):
        def compare(i):
            return best_fit[i][-1]
        best_fit = {}
        sequence = []
        #print(list(self.all_bouquets_by_element[0][0]))
        for player in self.all_bouquets_by_element:
            sequence.append(player)
            best_fit[player]=(None,0)
            for bouquet in self.all_bouquets_by_element[player]:
                score = bouquet[-1]
                if score>best_fit[player][-1]:
                    best_fit[player]  = copy.deepcopy(bouquet)
        #print("best fit:  ",best_fit)
        sequence.sort(key=compare,reverse=True)
        #print("sequence:  ", sequence)
        give_out  = {}
        for player in sequence:
            if player not in best_fit:
                continue
            if best_fit[player][0]==None:
                give_out[player] = Bouquet({})
                continue
            types = copy.deepcopy(best_fit[player][0])
            colors = copy.deepcopy(best_fit[player][1])
            sizes = copy.deepcopy(best_fit[player][2])
            #print(sizes)
            give = {}
            count = 0
            for i in range(3):
                requirement = 3-i
                for flower in remaining_flowers:
                    if remaining_flowers[flower]==0:
                        continue
                    score = 0
                    if  flower.type in types and types[flower.type]>0:
                        score +=1
                    if  flower.size in sizes and sizes[flower.size]>0:
                        score +=1
                    if  flower.color in colors and colors[flower.color]>0:
                        score +=1
                    if score>=requirement and count<12:
                        count+=1
                        if flower.type in types and types[flower.type] > 0:
                            types[flower.type] -= 1
                        if flower.color in colors and colors[flower.color] > 0:
                            colors[flower.color] -= 1
                        if flower.size in sizes and sizes[flower.size] > 0:
                            sizes[flower.size] -= 1
                        remaining_flowers[flower] -= 1
                        if flower not in give:
                            give[flower]=0
                        give[flower]+=1
            give_out[player]  = copy.deepcopy(Bouquet(give))
        #print(give_out)
        ret = []
        for player in give_out:
            ret.append((self.suitor_id,player,give_out[player]))
        return ret


    def prepare_bouquets(self, flower_counts: Dict[Flower, int]):
        #print(flower_counts)
        """
        :param flower_counts: flowers and associated counts for for available flowers
        :return: list of tuples of (self.suitor_id, recipient_id, chosen_bouquet)
        the list should be of length len(self.num_suitors) - 1 because you should give a bouquet to everyone
         but yourself
        To get the list of suitor ids not including yourself, use the following snippet:
        all_ids = np.arange(self.num_suitors)
        recipient_ids = all_ids[all_ids != self.suitor_id]
        """
        remaining_flowers = flower_counts.copy()
        if self.current_day == 1: # first day, so we don't have feedback
            self.current_day += 1
            all_ids = np.arange(self.num_suitors)
            recipient_ids = all_ids[all_ids != self.suitor_id]
            return list(map(lambda recipient_id: self._prepare_bouquet_first_day(remaining_flowers, recipient_id), recipient_ids))
        elif self.current_day == self.days: # last day
            # for now use strategy of intermediate day, but definitely need to update that
            prev_round_feedback = self.feedback[len(self.feedback)-1]
            recipient_ids = list(zip(*prev_round_feedback))[1] # sort recipiernt_ids by final score to prioritize players
            return self._prepare_bouquet_last_day(remaining_flowers)
            #return list(map(lambda recipient_id: self._prepare_bouquet_last_day(remaining_flowers, recipient_id), recipient_ids))
        else: # intermediate day
            self.current_day += 1
            prev_round_feedback = self.feedback[len(self.feedback)-1]
            recipient_ids = list(zip(*prev_round_feedback))[1] # sort recipiernt_ids by final score to prioritize players
            return list(map(lambda recipient_id: self._prepare_bouquet_intermediate_day(remaining_flowers, recipient_id), recipient_ids))

    def bouquet_to_elements(self, bouquet: Bouquet):
        elements = {}
        elements["color"] = {}
        elements["size"] = {}
        elements["type"] = {}
        for flower in bouquet.flowers():
            if flower.color not in elements["color"]:
                elements["color"][flower.color] = 1
            else:
                elements["color"][flower.color] += 1
            if flower.size not in elements["size"]:
                elements["size"][flower.size] = 1
            else:
                elements["size"][flower.size] += 1
            if flower.type not in elements["type"]:
                elements["type"][flower.type] = 1
            else:
                elements["type"][flower.type] += 1
        return elements["color"], elements["type"], elements["size"]

    def zero_score_bouquet(self):
        """
        :return: a Bouquet for which your scoring function will return 0
        """
        #min_flower = Flower(
        #    size=self.size_score[0],
        #    color=self.color_score[0],
        #    type=self.type_score[0]
        #)
        return Bouquet({})

    def one_score_bouquet(self):
        """
        :return: a Bouquet for which your scoring function will return 1
        """
        flowers = {}
        for i in range(12):
            flower = Flower(
                size=self.size_score[2],
                color=self.color_score[5],
                type=FlowerTypes(self.max_type_sequence[i])
            )
            if flower not in flowers:
                flowers[flower]=0
            flowers[flower] += 1

        return Bouquet(flowers)
    def flower_type_to_int(self,type):
        if type == FlowerTypes.Rose:
            return 0
        if type == FlowerTypes.Chrysanthemum:
            return 1
        if type == FlowerTypes.Tulip:
            return 2
        if type == FlowerTypes.Begonia:
            return 3

    def score_types(self, types: Dict[FlowerTypes, int]):#max 4/13
        """
        :param types: dictionary of flower types and their associated counts in the bouquet
        :return: A score representing preference of the flower types in the bouquet
        """
        key = {0:0,1:0,2:0,3:0}
        score = 0
        for type in types:
            n = self.flower_type_to_int(type)
            key[n]+=1
        combination = [key[i] for i in range(4)]
        score = self.type_score[tuple(combination)]
        if score>1700:
            return 4/13
        else:
            return 0



    def score_colors(self, colors: Dict[FlowerColors, int]):# max 6 / 13
        """
        :param colors: dictionary of flower colors and their associated counts in the bouquet
        :return: A score representing preference of the flower colors in the bouquet
        """
        score = 0
        for color in colors:
            score += colors[color] * self.color_score.index(color)
        return score / 130

    def score_sizes(self, sizes: Dict[FlowerSizes, int]):# max 3/13
        """
        :param sizes: dictionary of flower sizes and their associated counts in the bouquet
        :return: A score representing preference of the flower sizes in the bouquet
        """
        score = 0
        for size in sizes:
            score += sizes[size] * self.size_score.index(size)
        return score / 104
    
    def receive_feedback(self, feedback):
        """
        :param feedback:
        :return: nothing
        """
        final_scores_tuples = [] # a list of tuples (final_score, suitor_num, bouquet)
        final_scores_tuples_above_median = [] # a list of tuples (final_score, suitor_num, bouquet)
        final_scores_tuples_below_median = [] # a list of tuples (final_score, suitor_num, bouquet)
        scores = []
        for suitor_num, (rank, score, _) in enumerate(feedback):
            if suitor_num != self.suitor_id:
                scores.append(score)
        median_score = statistics.median(scores)
        
        for suitor_num, (rank, score, number_players_with_same_rank) in enumerate(feedback):
            if suitor_num != self.suitor_id: # we shouldn't add ourselves to the list of players for whom we will create a bouquet
                # TODO: update final_score claculation to 
                # 1) give more weight to ranking
                # 2) take into consideration the number of people who got the same ranking
                # maybe use a final_score =w_1*rank + w_2*score function
                new_bouquet = self.all_bouquets[suitor_num][len(self.all_bouquets[suitor_num])-1][0]
                new_bouquet_types = self.all_bouquets_by_element[suitor_num][len(self.all_bouquets_by_element[suitor_num])-1][0]
                new_bouquet_colors = self.all_bouquets_by_element[suitor_num][len(self.all_bouquets_by_element[suitor_num])-1][1]
                new_bouquet_sizes = self.all_bouquets_by_element[suitor_num][len(self.all_bouquets_by_element[suitor_num])-1][2]
                self.all_bouquets[suitor_num][len(self.all_bouquets[suitor_num])-1] = (new_bouquet, score) # update score in the dictionary
                self.all_bouquets_by_element[suitor_num][len(self.all_bouquets_by_element[suitor_num])-1] = (new_bouquet_types, new_bouquet_colors, new_bouquet_sizes, score)
                rank_after_ties = (rank + number_players_with_same_rank)/2
                new_rank = rank_after_ties/(self.num_suitors - 1) # normalize rankings so that they are in the [0, 1] range
                final_score = score/(new_rank*new_rank) # used score / rank^2 --> as rank gets worse, the final_score will exponentially get worse
                if self.checkScoreRange(score, median_score) == 1:
                    final_scores_tuples_above_median.append((final_score, suitor_num, new_bouquet))
                else:
                    final_scores_tuples_below_median.append((final_score, suitor_num, new_bouquet))
        feedback_above_median = sorted(final_scores_tuples_above_median, reverse=True)
        feedback_below_median = sorted(final_scores_tuples_below_median, reverse=True)
        new_feedback = feedback_above_median
        new_feedback.extend(feedback_below_median)
        self.feedback.append(new_feedback)

    def checkScoreRange(self, score, median_score):
        """
        :param score:
        :return 0 if our score is less than the median
         return 1 if our score is >= the median:
        """
        if score >= median_score:
            return 1
        else:
            return 0
