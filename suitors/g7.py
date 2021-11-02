from collections import Counter
from typing import Dict

import numpy as np
from numpy.core.fromnumeric import size


from constants import MAX_BOUQUET_SIZE
from flowers import Bouquet, Flower, FlowerSizes, FlowerColors, FlowerTypes
from utils import flatten_counter
from suitors.base import BaseSuitor


class Suitor(BaseSuitor):
    def __init__(self, days: int,  num_suitors: int, suitor_id: int, days_remaining: int, bouq_Dict: dict):
        """
        :param days: number of days of courtship
        :param num_suitors: number of suitors, including yourself
        :param suitor_id: unique id of your suitor in range(num_suitors)
        """
        super().__init__(days, num_suitors, suitor_id, name='g7', days_remaining=days, bouq_Dict = {})

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

        if (recipient_id not in self.bouq_Dict.keys):
            self.bouq_Dict[recipient_id] = [(chosen_bouquet, None)]
        else:
            self.bouq_Dict[recipient_id].append((chosen_bouquet, None))

        return self.suitor_id, recipient_id, chosen_bouquet
    def _prepare_bouquet_inter_rounds(self, remaining_flowers, recipient_id):
        best_bouquet_score = max(bouq_Dict[recipient_id], key=lambda e: int(e[1]))
        best_bouquet = best_bouquet_score[0]
        best_score = best_bouquet_score[1]
        num_remaining = sum(remaining_flowers.values())
        size = int(np.random.randint(0, min(MAX_BOUQUET_SIZE, num_remaining) + 1))
        changes = 0
        if size > 0:
            if (best_score < 0.7):
                chosen_flowers = np.random.choice(flatten_counter(remaining_flowers), size=(size, ), replace=False)
                chosen_flower_counts = dict(Counter(chosen_flowers))
                for k, v in chosen_flower_counts.items():
                    remaining_flowers[k] -= v
                    assert remaining_flowers[k] >= 0
            else:
                chosen_flowers ={}
                for flower, count in best_bouquet.items():
                    if remaining_flowers[flower] >= count:
                        chosen_flowers[flower] = count
                    elif remaining_flowers[flower] > 0:
                        chosen_flowers[flower] = remaining_flowers[flower]
        else:
            chosen_flower_counts = dict()
        chosen_bouquet = Bouquet(chosen_flower_counts)

        
        self.bouq_Dict[recipient_id].append((chosen_bouquet, None))
        return self.suitor_id, recipient_id, chosen_bouquet


    def _prepare_bouquet_last_round(self, remaining_flowers, recipient_id):
        best_bouquet_score = max(bouq_Dict[recipient_id], key=lambda e: int(e[1]))
        best_bouquet = best_bouquet_score[0]
        best_score = best_bouquet_score[1]
        num_remaining = sum(remaining_flowers.values())
        size = int(np.random.randint(0, min(MAX_BOUQUET_SIZE, num_remaining) + 1))
        changes = 0
        if size > 0:
            chosen_flowers ={}
            for flower, count in best_bouquet.items():
                if remaining_flowers[flower] >= count:
                    chosen_flowers[flower] = count
                elif remaining_flowers[flower] > 0:
                    chosen_flowers[flower] = remaining_flowers[flower]
        else:
            chosen_flower_counts = dict()
        chosen_bouquet = Bouquet(chosen_flower_counts)

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
        all_ids = np.arange(self.num_suitors)
        recipient_ids = all_ids[all_ids != self.suitor_id]
        remaining_flowers = flower_counts.copy()
        
        # First day: pass random? assortment of size 6 to everyone
        if (self.days_remaining == self.days):
            return list(map(lambda recipient_id: self._prepare_bouquet(remaining_flowers, recipient_id), recipient_ids))

        # Last day: best bouquet
        elif(self.days_remaining == 0):
            return list(map(lambda recipient_id: self._prepare_bouquet_last_round(remaining_flowers, recipient_id), recipient_ids))
            
        # Every day in between
        else:
            return list(map(lambda recipient_id: self._prepare_bouquet_inter_rounds(remaining_flowers, recipient_id), recipient_ids))

        # Decrement days remaining
        self.days_remaining -= 1

    def zero_score_bouquet(self):
        """
        :return: a Bouquet for which your scoring function will return 0
        """
        min_flower = Flower(
            size = FlowerSizes.Medium,
            color = FlowerColors.Yellow,
            type = FlowerTypes.Tulip
        )
        pass Bouquet({min_flower: 1})

    def one_score_bouquet(self):
        """
        :return: a Bouquet for which your scoring function will return 1
        """
        # blue, small, rose
        max_flower = Flower(
            size = FlowerSizes.Large,
            color = FlowerColors.Red,
            type = FlowerTypes.Chrysanthemum            
        )
        pass Bouquet({max_flower: 1})

    def score_types(self, types: Dict[FlowerTypes, int]):
        """
        :param types: dictionary of flower types and their associated counts in the bouquet
        :return: A score representing preference of the flower types in the bouquet
        """
        if(FlowerTypes.Tulip in types.keys or len(types) == 0):
            return 0.0
        elif(FlowerTypes.Chrysanthemum in types.keys):
            return 1/3
        else:
            return 0.1

    def score_colors(self, colors: Dict[FlowerColors, int]):
        """
        :param colors: dictionary of flower colors and their associated counts in the bouquet
        :return: A score representing preference of the flower colors in the bouquet
        """
        if(FlowerColors.Yellow in colors.keys or len(colors) == 0):
            return 0.0
        elif(FlowerColors.Red in colors.keys):
            return 1/3
        else:
            return 0.1

    def score_sizes(self, sizes: Dict[FlowerSizes, int]):
        """
        :param sizes: dictionary of flower sizes and their associated counts in the bouquet
        :return: A score representing preference of the flower sizes in the bouquet
        """
        if(FlowerSizes.Medium in sizes.keys or len(sizes) == 0):
            return 0.0
        elif(FlowerSizes.Large in sizes.keys):
            return 1/3
        else:
            return 0.1

    def receive_feedback(self, feedback):
        """
        :param feedback:
        :return: nothing
        """
        self.feedback.append(feedback)
