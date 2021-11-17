from collections import Counter
from typing import Dict

import itertools
import math
import numpy as np
import pandas as pd
import random
from sklearn.linear_model import LinearRegression

from constants import MAX_BOUQUET_SIZE
from flowers import Bouquet, Flower, FlowerSizes, FlowerColors, FlowerTypes, get_all_possible_flowers,sample_n_random_flowers
from utils import flatten_counter
from suitors.base import BaseSuitor

import time
# import pdb



class Suitor(BaseSuitor):
    def __init__(self, days: int, num_suitors: int, suitor_id: int):
        """
        :param days: number of days of courtship
        :param num_suitors: number of suitors, including yourself
        :param suitor_id: unique id of your suitor in range(num_suitors)
        """
        self.curr_day = 0
        self.arrangement_hist = []
        self.bouquet_hist = []
        self.score_hist = []

        self.all_possible_flower_keys = [str(f) for f in get_all_possible_flowers()]
        self.NUM_ALL_POSSIBLE_FLOWERS = len(self.all_possible_flower_keys)
        
        self.all_possible_flowers = dict(zip(self.all_possible_flower_keys, [0] * self.NUM_ALL_POSSIBLE_FLOWERS))


        self.priority = [i for i in list(range(num_suitors)) if i != suitor_id]
        super().__init__(days, num_suitors, suitor_id, name='g6')
        self.wanted_bouquet = sample_n_random_flowers(get_all_possible_flowers(),MAX_BOUQUET_SIZE)
        self.typeWeight, self.colorWeight, self.sizeWeight = np.random.dirichlet(np.ones(3),size=1)[0]
        self.wanted_colors,self.wanted_sizes,self.wanted_types = self._parse_bouquet()
        self.threshold = 0.9
        if num_suitors>=36:
            if days<90:
                self.threshold = 0.95
            elif days>180:
                self.threshold = 0.99
            else:
                self.threshold = 0.97
        elif num_suitors>=18:
            if days<90:
                self.threshold = 0.92
            elif days>180:
                self.threshold = 0.97
            else:
                self.threshold = 0.95
        else:
            if days<90:
                self.threshold = 0.90
            elif days>180:
                self.threshold = 0.95
            else:
                self.threshold = 0.92



    def _parse_bouquet(self):
        colors = np.zeros(6)
        sizes = np.zeros(3)
        types = np.zeros(4)
        for flower,number in self.wanted_bouquet.items():
            colors[flower.color.value]+=number
            sizes[flower.size.value]+=number
            types[flower.type.value]+=number
        return colors,sizes,types

    def cosine_similarity(self,a,b):
        if np.linalg.norm(a)==0 or np.linalg.norm(b) == 0:
            return 0
        return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

    def _prepare_rand_bouquet(self, remaining_flowers, recipient_id):
        num_remaining = sum(remaining_flowers.values())
        size = random.randint(0, min(MAX_BOUQUET_SIZE, num_remaining))
        if size > 0:
            chosen_flowers = np.random.choice(flatten_counter(remaining_flowers), size=(size, ), replace=False)
            chosen_flower_counts = dict(Counter(chosen_flowers))
        else:
            chosen_flower_counts = dict()
        chosen_bouquet = Bouquet(chosen_flower_counts)
        return self.suitor_id, recipient_id, chosen_bouquet


    def _get_all_possible_bouquets_arr(self, flowers: Dict[Flower, int]):
        bouquets = [[0] * self.NUM_ALL_POSSIBLE_FLOWERS]

        # make sure there are flowers left to give
        num_remaining = sum(flowers.values())
        if num_remaining == 0:
            return bouquets

        flatten_flowers = [str(f) for f in flatten_counter(flowers)]

        # Only looking at 20% of combinations for this size
        # num_flatten_flowers = len(flatten_flowers)
        num_flatten_flowers = min(num_remaining, 12)
        for size in range(1, min(MAX_BOUQUET_SIZE, num_remaining) + 1):
            # num_flatten_flowers = max(num_flatten_flowers, min(num_remaining, 12))
            selected_flowers = random.sample(flatten_flowers, num_flatten_flowers)
            size_combos = list(itertools.combinations(selected_flowers, size))
            size_bouquets = random.sample(size_combos, min(int(math.comb(num_flatten_flowers, size) * 0.2), 1000))
            size_bouquet_counts = [list({**self.all_possible_flowers, **Counter(size_bouquet)}.values()) for
                                   size_bouquet in size_bouquets]
            bouquets.extend(size_bouquet_counts)
            # num_flatten_flowers -= int(num_flatten_flowers / 4) # just to help with combinations -- since combinations requires so much time

        return bouquets

    def _extract_the_dimensions(self, bouquets):
        
        list_of_dicts = []

        for bouquet in bouquets:
            dict_of_features = {}
            for elem in self.all_possible_flower_keys:
                components = elem.split("-")
            
                for comp in components:
                    if comp not in dict_of_features:
                        dict_of_features[comp] = 0

            for i,boolean in enumerate(bouquet):
                if boolean==1:
                    flowertype = self.all_possible_flower_keys[i]
                    components = flowertype.split("-")
                    size = components[0]
                    if size in dict_of_features:
                        dict_of_features[size] = dict_of_features[size] + 1
                    else:
                        dict_of_features[size] = 0

                    color = components[1]
                    if color in dict_of_features:
                        dict_of_features[color] = dict_of_features[color] + 1
                    else:
                        dict_of_features[color] = 0

                    ftype = components[2]
                    if ftype in dict_of_features:
                        dict_of_features[ftype] = dict_of_features[ftype] + 1
                    else:
                        dict_of_features[ftype] = 0
            list_of_dicts.append(list(dict_of_features.values()))
        
        return list_of_dicts


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
        #time1 = time.time()
        remaining_flowers = flower_counts.copy()
        num_recips = len(self.priority)

        # keys across days for flowers are str(flower) so need to have matching from str(flower) back to flower object
        rem_flower_key_pairs = dict()
        for f in remaining_flowers.keys():
            rem_flower_key_pairs[str(f)] = f

        bouquets = [0] * num_recips
        for i in range(num_recips):
            recip_id = self.priority[i]
            recip_idx = recip_id - 1 if recip_id > self.suitor_id else recip_id

            if self.curr_day != 0 and self.score_hist[recip_idx][-1] == 1: # already know best bouquet possible
                if self.curr_day == self.days - 1: # last day so give best bouquet
                    # ensure have the flowers to do so
                    have_flowers = all(k in remaining_flowers and remaining_flowers[k] - v >= 0 for (k, v) in self.bouquet_hist[recip_idx][-1].arrangement.items())

                    if have_flowers:
                        bouquets[recip_idx] = self.bouquet_hist[recip_idx][-1]

                        for k, v in self.bouquet_hist[recip_idx][-1].arrangement.items():
                            remaining_flowers[k] -= v
                            assert remaining_flowers[k] >= 0

                        continue
                else: # not last day so give empty bouquet to save flowers since already know bouquet is correct
                    bouquets[recip_idx] = (self.suitor_id, recip_id, Bouquet({})) # give empty

                    self.bouquet_hist[recip_id].append(self.bouquet_hist[recip_idx][-1]) # pretending we are giving best bouquet
                    self.arrangement_hist[recip_id].append(self.arrangement_hist[recip_idx][-1])
                    continue

            # if self.curr_day > 1 and self.curr_day >= int(self.days * 0.7): # giving bouquet using best guess from Linear Regression
            if self.curr_day != 0 and self.curr_day == self.days - 1:
                # getting all valid flower combinations for each person -- so know best bouquet is valid
                all_possible_bouquets_arr = self._get_all_possible_bouquets_arr(remaining_flowers)
                list_of_lists = self._extract_the_dimensions(all_possible_bouquets_arr)
                all_possible_bouquets_nparr = np.array(list_of_lists)

                hist_nparr = np.asarray(self.arrangement_hist[recip_idx], dtype=int)

                lin_reg = LinearRegression()
                lin_reg.fit(hist_nparr, pd.Series(self.score_hist[recip_idx]))

                pred_score = lin_reg.predict(all_possible_bouquets_nparr)

                # getting first instance of best score and using that bouquet
                best_flowers = all_possible_bouquets_arr[np.where(pred_score == max(pred_score))[0][0]]
                converted_best_flowers = self._extract_the_dimensions([best_flowers])[0]

                # Creating Bouquet object to return
                if sum(best_flowers) > 0:
                    predicted_best = dict()
                    for j in range(self.NUM_ALL_POSSIBLE_FLOWERS):
                        if best_flowers[j] > 0:
                            f_obj = rem_flower_key_pairs[self.all_possible_flower_keys[j]]
                            predicted_best[f_obj] = best_flowers[j]
                    best_bouquet = Bouquet(predicted_best)

                    for k, v in best_bouquet.arrangement.items():
                        remaining_flowers[k] -= v
                        assert remaining_flowers[k] >= 0
                else:
                    best_bouquet = Bouquet(dict())

                bouquets[recip_idx] = (self.suitor_id, recip_id, best_bouquet)
                
                self.arrangement_hist[recip_idx].append(converted_best_flowers)
                self.bouquet_hist[recip_idx].append(best_bouquet)

            else: # give random bouquet to get more data for Linear Regression
                arrangement_len = len(self.arrangement_hist[recip_idx]) if self.curr_day != 0 else 1
                for _ in range(arrangement_len):
                    suitor_id, recipient_id, chosen_bouquet = self._prepare_rand_bouquet(remaining_flowers, recip_id)

                    b_dict = dict()
                    for k, v in chosen_bouquet.arrangement.items():
                        b_dict[str(k)] = v
                    arrangement = list({**self.all_possible_flowers, **b_dict}.values())
                    
                    converted_arrangement = self._extract_the_dimensions([arrangement])[0]

                    if self.curr_day == 0 or converted_arrangement not in self.arrangement_hist[recip_idx]:
                        break

                for k, v in chosen_bouquet.arrangement.items():
                    remaining_flowers[k] -= v
                    assert remaining_flowers[k] >= 0
                bouquets[recip_idx] = (suitor_id, recipient_id, chosen_bouquet)

                if self.curr_day == 0:
                    self.arrangement_hist.append([converted_arrangement])
                    self.bouquet_hist.append([chosen_bouquet])
                else:
                    self.arrangement_hist[recip_idx].append(converted_arrangement)
                    self.bouquet_hist[recip_idx].append(chosen_bouquet)

        self.curr_day += 1
        # time2 = time.time()
        # print("time2 - time1: {}".format(time2 - time1))
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
        return self.wanted_bouquet


    def score_types(self, types: Dict[FlowerTypes, int]):
        """
        :param types: dictionary of flower types and their associated counts in the bouquet
        :return: A score representing preference of the flower types in the bouquet
        """
        curr_types = np.zeros(len(FlowerTypes))
        for key,value in types.items():
            curr_types[key.value]+=value
        cosine_similarity = self.cosine_similarity(curr_types,self.wanted_types)
        if cosine_similarity>self.threshold:
            return self.typeWeight*cosine_similarity
        else:
            return 0

    def score_colors(self, colors: Dict[FlowerColors, int]):
        """
        :param colors: dictionary of flower colors and their associated counts in the bouquet
        :return: A score representing preference of the flower colors in the bouquet
        """
        curr_colors = np.zeros(len(FlowerColors))
        for key,value in colors.items():
            curr_colors[key.value]+=value
        cosine_similarity = self.cosine_similarity(curr_colors,self.wanted_colors)
        if cosine_similarity>self.threshold:
            return self.colorWeight*cosine_similarity
        else:
            return 0

    def score_sizes(self, sizes: Dict[FlowerSizes, int]):
        """
        :param sizes: dictionary of flower sizes and their associated counts in the bouquet
        :return: A score representing preference of the flower sizes in the bouquet
        """
        curr_sizes = np.zeros(len(FlowerSizes))
        for key,value in sizes.items():
            curr_sizes[key.value]+=value
        cosine_similarity = self.cosine_similarity(curr_sizes,self.wanted_sizes)
        if cosine_similarity>self.threshold:
            return self.sizeWeight*cosine_similarity
        else:
            return 0

    def receive_feedback(self, feedback):
        """
        :param feedback:
        :return: nothing
        """

        # first sorting by the how many people got the same rank (least common to most common -- higher priority is given to rankings in which we are the only ones with this ranking)
        # then sorting by ranking (least to greatest -- ranking of 1 is best)
        # then sorting by score (greatest to least -- score of 1 is best, score of 0 is worst)
        self.priority = [x[1] for x in sorted(zip(feedback, list(range(len(feedback)))), key = lambda k: (k[0][2], k[0][0], -k[0][1])) if x[0][1] != float('-inf')]

        scores = [feedback[i][1] for i in range(len(feedback)) if feedback[i][1] != float('-inf')]

        for i in range(self.num_suitors - 1):
            if self.curr_day == 1:
                self.score_hist.append([scores[i]])
            elif self.score_hist[i][-1] == 1: # already got best score possible so maintaining that knowledge
                self.score_hist[i].append(1)
            else:
                self.score_hist[i].append(scores[i])

        self.feedback.append(feedback)

