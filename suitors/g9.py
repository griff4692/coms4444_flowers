from typing import Dict
from collections import Counter

import numpy as np
import random
import statistics

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
                           # create similar round that stores all bouquets along with their scores for every round
        self.current_day = 1 # keep track of the current day, so that we know how many days are left

        temp = self.random_sequence(6)
        self.color_score = [FlowerColors(i) for i in temp]
        temp = self.random_sequence(4)
        self.type_score = [FlowerTypes(i) for i in temp]
        temp = self.random_sequence(3)
        self.size_score = [FlowerSizes(i) for i in temp]
        
    def random_sequence(self, n):
        sequence = np.arange(n)
        np.random.shuffle(sequence)
        return sequence

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
        self.bouquets[recipient_id] = (chosen_bouquet, 0)
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
        self.bouquets[recipient_id] = (chosen_bouquet, 0) # store the bouquet we gave to each player in this round 
        return self.suitor_id, recipient_id, chosen_bouquet
    
    def _prepare_bouquet_last_day(self, remaining_flowers, recipient_id):
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
        self.bouquets[recipient_id] = (chosen_bouquet, 0) # store the bouquet we gave to each player in this round 
        return self.suitor_id, recipient_id, chosen_bouquet

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
            return list(map(lambda recipient_id: self._prepare_bouquet_last_day(remaining_flowers, recipient_id), recipient_ids))
        else: # intermediate day
            self.current_day += 1
            prev_round_feedback = self.feedback[len(self.feedback)-1]
            recipient_ids = list(zip(*prev_round_feedback))[1] # sort recipiernt_ids by final score to prioritize players
            return list(map(lambda recipient_id: self._prepare_bouquet_intermediate_day(remaining_flowers, recipient_id), recipient_ids))


    def zero_score_bouquet(self):
        """
        :return: a Bouquet for which your scoring function will return 0
        """
        min_flower = Flower(
            size=self.size_score[0],
            color=self.color_score[0],
            type=self.type_score[0]
        )
        return Bouquet({min_flower: 0})

    def one_score_bouquet(self):
        """
        :return: a Bouquet for which your scoring function will return 1
        """
        max_flower = Flower(
            size=self.size_score[2],
            color=self.color_score[5],
            type=self.type_score[3]
        )
        return Bouquet({max_flower: 12})

    def score_types(self, types: Dict[FlowerTypes, int]):
        """
        :param types: dictionary of flower types and their associated counts in the bouquet
        :return: A score representing preference of the flower types in the bouquet
        """
        score = 0
        for type in types:
            score += types[type] * self.type_score.index(type)
        return score / 117



    def score_colors(self, colors: Dict[FlowerColors, int]):
        """
        :param colors: dictionary of flower colors and their associated counts in the bouquet
        :return: A score representing preference of the flower colors in the bouquet
        """
        score = 0
        for color in colors:
            score += colors[color] * self.color_score.index(color)
        return score / 130

    def score_sizes(self, sizes: Dict[FlowerSizes, int]):
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
        print(self.suitor_id)
        final_scores_tuples = [] # a list of tuples (final_score, suitor_num, bouquet)
        final_scores_tuples_above_median = [] # a list of tuples (final_score, suitor_num, bouquet)
        final_scores_tuples_below_median = [] # a list of tuples (final_score, suitor_num, bouquet)
        scores = []
        for suitor_num, (rank, score) in enumerate(feedback):
            if suitor_num != self.suitor_id:
                scores.append(score)
        median_score = statistics.median(scores)
        
        for suitor_num, (rank, score) in enumerate(feedback):
            if suitor_num != self.suitor_id: # we shouldn't add ourselves to the list of players for whom we will create a bouquet
                # TODO: update final_score claculation to 
                # 1) give more weight to ranking
                # 2) take into consideration the number of people who got the same ranking
                # maybe use a final_score =w_1*rank + w_2*score function
                self.bouquets[suitor_num] = (self.bouquets[suitor_num][0], score) # update score in the dictionary
                new_rank = rank/(self.num_suitors - 1) # normalize rankings so that they are in the [0, 1] range
                final_score = score/(new_rank*new_rank) # used score / rank^2 --> as rank gets worse, the final_score will exponentially get worse
                if self.checkScoreRange(score, median_score) == 1:
                    final_scores_tuples_above_median.append((final_score, suitor_num, self.bouquets[suitor_num][0]))
                else:
                    final_scores_tuples_below_median.append((final_score, suitor_num, self.bouquets[suitor_num][0]))
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
