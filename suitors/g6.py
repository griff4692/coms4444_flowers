from collections import Counter
from typing import Dict

import itertools
import threading
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from constants import MAX_BOUQUET_SIZE
from flowers import Bouquet, Flower, FlowerSizes, FlowerColors, FlowerTypes, get_all_possible_flowers
from utils import flatten_counter
from suitors.base import BaseSuitor

import time
import pdb

# str(f.type).replace("<", "").replace(">", "").replace(".", "")


def _bouquets_to_df(bouquets: Dict[Flower, int]):
    bouquets_df = []

    for i in range(len(bouquets)):
        bouquets_dict = dict()
        for s in FlowerSizes:
            bouquets_dict[str(s).replace("<", "").replace(">", "").replace(".", "")] = 0
        for t in FlowerTypes:
            bouquets_dict[str(t).replace("<", "").replace(">", "").replace(".", "")] = 0
        for c in FlowerColors:
            bouquets_dict[str(c).replace("<", "").replace(">", "").replace(".", "")] = 0

        if len(bouquets[i]) > 0:
            b = bouquets[i].arrangement
            for f in b.keys():
                bouquets_dict[str(f.type).replace("<", "").replace(">", "").replace(".", "")] += b[f]
                bouquets_dict[str(f.size).replace("<", "").replace(">", "").replace(".", "")] += b[f]
                bouquets_dict[str(f.color).replace("<", "").replace(">", "").replace(".", "")] += b[f]

        bouquets_df.append(bouquets_dict)

    return pd.DataFrame.from_dict(bouquets_df)


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

        all_flowers = [str(f) for f in get_all_possible_flowers()]
        self.all_possible_flowers = dict(zip(all_flowers, [0] * len(all_flowers)))
        self.all_possible_flower_keys = list(self.all_possible_flowers)
        super().__init__(days, num_suitors, suitor_id, name='g6')

    def _prepare_rand_bouquet(self, remaining_flowers, recipient_id):
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


    def _get_all_possible_bouquets_arr(self, flowers: Dict[Flower, int]):
        bouquets = [[0] * len(self.all_possible_flowers)]
        flatten_flowers = [str(f) for f in flatten_counter(flowers)]
        # for _ in range(int((MAX_BOUQUET_SIZE + 1) / 2)):
        #     size = np.random.randint(1, MAX_BOUQUET_SIZE + 1)
        for size in range(int(MAX_BOUQUET_SIZE + 1)):
            size_bouquets = itertools.combinations(flatten_flowers, size)
            size_bouquet_counts = [list({**self.all_possible_flowers, **Counter(size_bouquet)}.values()) for size_bouquet in size_bouquets]
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
        time1 = time.time()
        all_ids = np.arange(self.num_suitors)
        recipient_ids = all_ids[all_ids != self.suitor_id]
        remaining_flowers = flower_counts.copy()

        bouquets = []

        all_possible_bouquets_arr = None
        if self.curr_day == self.days - 1:
            all_possible_bouquets_arr =  self._get_all_possible_bouquets_arr(remaining_flowers)

        for i in range(len(recipient_ids)):
            if self.curr_day != 0 and self.score_hist[i][-1] == 1: # already know best bouquet possible
                if self.curr_day == self.days - 1: # last day so give best bouquet
                    # ensure have the flowers to do so
                    have_flowers = all(remaining_flowers[k] - v >= 0 for (k, v) in self.bouquet_hist[i][-1].items())

                    if have_flowers:
                        bouquets.append(self.bouquet_hist[i][-1])

                        for k, v in self.bouquet_hist[i][-1].items():
                            remaining_flowers[k] -= v
                            assert remaining_flowers[k] >= 0
                else: # not last day so give empty bouquet to save flowers since already know bouquet is correct
                    bouquets.append(dict()) # give empty

                    self.bouquet_hist[i].append(self.bouquet_hist[i][-1]) # pretending we are giving best bouquet
                    self.arrangement_hist[i].append(self.arrangement_hist[i][-1])
            else:
                if self.curr_day == self.days - 1: # giving bouquet for last day
                    bouquet_hist = []
                    for b in self.arrangement_hist[i]:
                        b_dict = dict()
                        for f in b:
                            b_dict[str(f)] = 0 if str(f) not in b_dict else b_dict[str(f)] + 1
                        bouquet = list({**self.all_possible_flowers, **b_dict}.values())
                        bouquet_hist.append(bouquet)

                    hist_nparr = np.asarray(bouquet_hist, dtype=int)

                    lin_reg = LinearRegression()
                    lin_reg.fit(hist_nparr, pd.Series(self.score_hist[i]))

                    # # ensuring we have the flowers to do so
                    # rem_flowers = list({**self.all_possible_flowers, **remaining_flowers}.values())
                    #
                    # for possible_bouquet in all_possible_bouquets_arr:
                    #     print(possible_bouquet)
                    #     pdb.set_trace()
                    #
                    # all_possible_bouquets_arr = [possible_bouquet for possible_bouquet in all_possible_bouquets_arr if
                    #                              all([rem_flowers[j] - possible_bouquet[j] >= 0 for j in range(len(possible_bouquet))])]

                    all_possible_bouquets_nparr = np.array(all_possible_bouquets_arr)
                    pred_score = lin_reg.predict(all_possible_bouquets_nparr)
                    # getting first instance of best score and using that bouquet
                    best_flowers = all_possible_bouquets_nparr[np.where(pred_score == max(pred_score))[0][0]]

                    if sum(best_flowers) > 0:
                        best_bouquet = []
                        for j in range(len(best_flowers)):
                            if best_flowers[j] > 0:
                                pdb.set_trace()
                                best_bouquet.append(
                                    [f for f in remaining_flowers.keys() if str(f) == self.all_possible_flower_keys[j]][
                                        0])
                                pdb.set_trace()

                        best_bouquet = Bouquet(dict(zip(self.all_possible_flowers, best_flowers)))

                        for k, v in best_bouquet.arrangement.items():
                            remaining_flowers[k] -= v
                            assert remaining_flowers[k] >= 0
                    else:
                        best_bouquet = Bouquet(dict())

                    bouquets.append((self.suitor_id, recipient_ids[i], best_bouquet))

                else: # give random bouquet to get more data for Linear Regression
                    suitor_id, recipient_id, chosen_bouquet= self._prepare_rand_bouquet(remaining_flowers, recipient_ids[i])
                    bouquets.append((suitor_id, recipient_id, chosen_bouquet))

                    if self.curr_day == 0:
                        self.arrangement_hist.append([chosen_bouquet.arrangement])
                        self.bouquet_hist.append([chosen_bouquet])
                    else:
                        self.arrangement_hist[i].append(chosen_bouquet.arrangement)
                        self.bouquet_hist[i].append(chosen_bouquet)

        self.curr_day += 1
        time2 = time.time()
        print("time2 - time1: {}".format(time2 - time1))
        pdb.set_trace()
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
        return len(types) / (len(FlowerSizes)+len(FlowerColors)+len(FlowerTypes))

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
        return len(colors) / (len(FlowerSizes)+len(FlowerColors)+len(FlowerTypes))

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
        return len(sizes) / (len(FlowerSizes)+len(FlowerColors)+len(FlowerTypes))

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