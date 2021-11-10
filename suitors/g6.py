from collections import Counter
from typing import Dict

import itertools
import math
import numpy as np
import pandas as pd
import random
from sklearn.linear_model import LinearRegression

from constants import MAX_BOUQUET_SIZE
from flowers import Bouquet, Flower, FlowerSizes, FlowerColors, FlowerTypes, get_all_possible_flowers
from utils import flatten_counter
from suitors.base import BaseSuitor

# import time
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
        self.typeWeight, self.colorWeight, self.sizeWeight = np.random.dirichlet(np.ones(3),size=1)[0]
        super().__init__(days, num_suitors, suitor_id, name='g6')


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

        # # 70 % of days, will use linear regression to give best guess on bouquet
        # # on each day, will sample fraction of all possible combinations
        # # sum of all fractions will sum to 100% of total sample space
        # for _ in range(int((MAX_BOUQUET_SIZE + 1) * 10 / (7 * self.days))):

        # Only looking at 20% of combinations for this size
        for size in range(1, min(MAX_BOUQUET_SIZE, num_remaining) + 1):
            num_flatten_flowers = len(flatten_flowers)
            selected_flowers = flatten_flowers

            num_combos = int(math.comb(num_flatten_flowers, size))
            if num_combos > 100000:
                num_flatten_flowers = min(num_flatten_flowers, 10)
                selected_flowers = random.sample(flatten_flowers, num_flatten_flowers)

            size_combos = list(itertools.combinations(selected_flowers, size))
            size_bouquets = random.sample(size_combos, min(int(math.comb(num_flatten_flowers, size) * 0.2), 2000))
            size_bouquet_counts = [list({**self.all_possible_flowers, **Counter(size_bouquet)}.values()) for
                                   size_bouquet in size_bouquets]
            bouquets.extend(size_bouquet_counts)

        return bouquets

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
        all_ids = np.arange(self.num_suitors)
        recipient_ids = all_ids[all_ids != self.suitor_id]
        remaining_flowers = flower_counts.copy()

        # keys across days for flowers are str(flower) so need to have matching from str(flower) back to flower object
        rem_flower_key_pairs = dict()
        for f in remaining_flowers.keys():
            rem_flower_key_pairs[str(f)] = f

        bouquets = []

        for i in range(len(recipient_ids)):
            if self.curr_day != 0 and self.score_hist[i][-1] == 1: # already know best bouquet possible
                if self.curr_day == self.days - 1: # last day so give best bouquet
                    # ensure have the flowers to do so
                    have_flowers = all(k in remaining_flowers and remaining_flowers[k] - v >= 0 for (k, v) in self.bouquet_hist[i][-1].arrangement.items())

                    if have_flowers:
                        bouquets.append(self.bouquet_hist[i][-1])

                        for k, v in self.bouquet_hist[i][-1].arrangement.items():
                            remaining_flowers[k] -= v
                            assert remaining_flowers[k] >= 0

                        continue
                else: # not last day so give empty bouquet to save flowers since already know bouquet is correct
                    bouquets.append((self.suitor_id, recipient_ids[i], Bouquet({}))) # give empty

                    self.bouquet_hist[i].append(self.bouquet_hist[i][-1]) # pretending we are giving best bouquet
                    self.arrangement_hist[i].append(self.arrangement_hist[i][-1])
                    continue

            if self.curr_day > 1 and self.curr_day > int(self.days * 0.3): # giving bouquet using best guess from Linear Regression
                # getting all valid flower combinations for each person -- so know best bouquet is valid
                all_possible_bouquets_arr = self._get_all_possible_bouquets_arr(remaining_flowers)
                all_possible_bouquets_nparr = np.array(all_possible_bouquets_arr)
                hist_nparr = np.asarray(self.arrangement_hist[i], dtype=int)

                lin_reg = LinearRegression()
                lin_reg.fit(hist_nparr, pd.Series(self.score_hist[i]))

                pred_score = lin_reg.predict(all_possible_bouquets_nparr)

                # getting first instance of best score and using that bouquet
                best_flowers = all_possible_bouquets_arr[np.where(pred_score == max(pred_score))[0][0]]

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

                bouquets.append((self.suitor_id, recipient_ids[i], best_bouquet))
                self.arrangement_hist[i].append(best_flowers)
                self.bouquet_hist[i].append(best_bouquet)

            else: # give random bouquet to get more data for Linear Regression
                arrangement_len = len(self.arrangement_hist[i]) if self.curr_day != 0 else 1
                for _ in range(arrangement_len):
                    suitor_id, recipient_id, chosen_bouquet = self._prepare_rand_bouquet(remaining_flowers, recipient_ids[i])

                    b_dict = dict()
                    for k, v in chosen_bouquet.arrangement.items():
                        b_dict[str(k)] = v
                    arrangement = list({**self.all_possible_flowers, **b_dict}.values())

                    if self.curr_day == 0 or arrangement not in self.arrangement_hist[i]:
                        break

                for k, v in chosen_bouquet.arrangement.items():
                    remaining_flowers[k] -= v
                    assert remaining_flowers[k] >= 0

                bouquets.append((suitor_id, recipient_id, chosen_bouquet))

                if self.curr_day == 0:
                    self.arrangement_hist.append([arrangement])
                    self.bouquet_hist.append([chosen_bouquet])
                else:
                    self.arrangement_hist[i].append(arrangement)
                    self.bouquet_hist[i].append(chosen_bouquet)

        self.curr_day += 1
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
        f1 = Flower(
            size=FlowerSizes.Large,
            color=FlowerColors.Blue,
            type=FlowerTypes.Rose
        )
        f2 = Flower(
            size=FlowerSizes.Small,
            color=FlowerColors.White,
            type=FlowerTypes.Chrysanthemum
        )
        f3 = Flower(
            size=FlowerSizes.Medium,
            color=FlowerColors.Yellow,
            type=FlowerTypes.Tulip
        )
        f4 = Flower(
            size=FlowerSizes.Medium,
            color=FlowerColors.Red,
            type=FlowerTypes.Begonia
        )
        f5 = Flower(
            size=FlowerSizes.Medium,
            color=FlowerColors.Purple,
            type=FlowerTypes.Begonia
        )
        f6 = Flower(
            size=FlowerSizes.Medium,
            color=FlowerColors.Orange,
            type=FlowerTypes.Begonia
        )
        return Bouquet({f1:1,f2:1,f3:1,f4:1,f5:1,f6:1})


    def score_types(self, types: Dict[FlowerTypes, int]):
        """
        :param types: dictionary of flower types and their associated counts in the bouquet
        :return: A score representing preference of the flower types in the bouquet
        """
        # if len(types) == 0:
        #     return 0.0
        #
        # avg_types = float(np.mean([x.value for x in flatten_counter(types)]))
        #return avg_types / (3 * (len(FlowerTypes) - 1))
        return self.typeWeight*len(types) / len(FlowerTypes)

    def score_colors(self, colors: Dict[FlowerColors, int]):
        """
        :param colors: dictionary of flower colors and their associated counts in the bouquet
        :return: A score representing preference of the flower colors in the bouquet
        """
        # if len(colors) == 0:
        #     return 0.0
        #
        # avg_colors = float(np.mean([x.value for x in flatten_counter(colors)]))
        #return avg_colors / (3 * (len(FlowerColors) - 1))
        return self.colorWeight*len(colors) / len(FlowerColors)

    def score_sizes(self, sizes: Dict[FlowerSizes, int]):
        """
        :param sizes: dictionary of flower sizes and their associated counts in the bouquet
        :return: A score representing preference of the flower sizes in the bouquet
        """
        # if len(sizes) == 0:
        #     return 0
        #
        # avg_sizes = float(np.mean([x.value for x in flatten_counter(sizes)]))
        #return avg_sizes / (3 * (len(FlowerSizes) - 1))
        return self.sizeWeight*len(sizes) / len(FlowerSizes)

    def receive_feedback(self, feedback):
        """
        :param feedback:
        :return: nothing
        """

        scores = [feedback[i][1] for i in range(len(feedback)) if feedback[i][1] != float('-inf')]

        for i in range(self.num_suitors - 1):
            if self.curr_day == 1:
                self.score_hist.append([scores[i]])
            elif self.score_hist[i][-1] == 1: # already got best score possible so maintaining that knowledge
                self.score_hist[i].append(1)
            else:
                self.score_hist[i].append(scores[i])

        self.feedback.append(feedback)

