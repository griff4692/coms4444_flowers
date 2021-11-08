from collections import Counter
from typing import Dict

import itertools
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from constants import MAX_BOUQUET_SIZE
from flowers import Bouquet, Flower, FlowerSizes, FlowerColors, FlowerTypes, get_all_possible_bouquets
from utils import flatten_counter
from suitors.base import BaseSuitor


class Suitor(BaseSuitor):
    def __init__(self, days: int, num_suitors: int, suitor_id: int):
        """
        :param days: number of days of courtship
        :param num_suitors: number of suitors, including yourself
        :param suitor_id: unique id of your suitor in range(num_suitors)
        """
        self.curr_day = 0
        self.bouquet_hist = []
        self.score_hist = []
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


    def _bouquets_to_dict(self, bouquets: Dict[Flower, int]):
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


    def _tuple_to_bouquet(self, bouquet_tuple):
        bouquet_dict = dict()
        if len(bouquet_tuple) > 0:
            for f in bouquet_tuple:
                if f not in bouquet_dict:
                    bouquet_dict[f] = 1
                else:
                    bouquet_dict[f] += 1

        return Bouquet(bouquet_dict)


    def _get_some_possible_bouquets(self, flowers: Dict[Flower, int]):
        flat_flower = flatten_counter(flowers)
        bouquets = [Bouquet({})]
        size = np.random.randint(1, MAX_BOUQUET_SIZE)
        size_bouquets = list(set(list(itertools.combinations(flat_flower, size))))
        bouquets += size_bouquets
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

        bouquets = []
        chosen_bouquets = []

        if self.curr_day == self.days - 1:
            all_bouquets = [self._tuple_to_bouquet(b) for b in get_all_possible_bouquets(remaining_flowers)]
            all_bouquets_df = self._bouquets_to_dict(all_bouquets)

        for i in range(len(recipient_ids)):
            if self.curr_day != 0 and self.score_hist[i][self.curr_day - 1] == 1: # already know best bouquet possible
                bouquets.append(dict()) # give empty bouquet so don't waste flowers if already know bouquet is correct
                chosen_bouquets.append(dict())
            else:
                if self.curr_day == self.days - 1: # giving bouquet for last day
                    suitor_id, recipient_id, chosen_bouquet = self._prepare_rand_bouquet(remaining_flowers,
                                                                                         recipient_ids[i])
                    bouquets.append((suitor_id, recipient_id, chosen_bouquet))
                    chosen_bouquets.append(chosen_bouquet)
                    lin_reg = LinearRegression()
                    lin_reg.fit(self._bouquets_to_dict(self.bouquet_hist[i]), pd.Series(self.score_hist[i]))


                    pred_score = lin_reg.predict(all_bouquets_df)
                    max_score = -1
                    for j in range(len(pred_score)):
                        if pred_score[j] > max_score:
                            best_bouquet = all_bouquets[j]
                            max_score = pred_score[j]

                            if pred_score[j] == 1:
                                break

                    bouquets.append((self.suitor_id, recipient_ids[i], best_bouquet))
                    chosen_bouquets.append(best_bouquet)

                else: # give random bouquet to get more data for Linear Regression
                    suitor_id, recipient_id, chosen_bouquet= self._prepare_rand_bouquet(remaining_flowers, recipient_ids[i])
                    bouquets.append((suitor_id, recipient_id, chosen_bouquet))
                    chosen_bouquets.append(chosen_bouquet)

        if self.curr_day == 0:
            for i in range(len(recipient_ids)):
                self.bouquet_hist.append([chosen_bouquets[i]])
        else:
            for i in range(len(recipient_ids)):
                self.bouquet_hist[i].append(chosen_bouquets[i])
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
        if self.curr_day == 1:
            for i in range(self.num_suitors - 1):
                self.score_hist.append([scores[i]])
        else:
            for i in range(self.num_suitors - 1):
                self.score_hist[i].append(scores[i])
        # self.feedback.append(feedback)
