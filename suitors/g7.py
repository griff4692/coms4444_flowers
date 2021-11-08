from collections import Counter
from typing import Dict

import numpy as np
from numpy.core.fromnumeric import size


from constants import MAX_BOUQUET_SIZE
from flowers import Bouquet, Flower, FlowerSizes, FlowerColors, FlowerTypes
from utils import flatten_counter
from suitors.base import BaseSuitor


class Suitor(BaseSuitor):
    def __init__(self, days: int,  num_suitors: int, suitor_id: int):
        """
        :param days: number of days of courtship
        :param num_suitors: number of suitors, including yourself
        :param suitor_id: unique id of your suitor in range(num_suitors)
        """
        self.days_remaining = 1
        self.bouq_Dict = {}
        self.weights = {}
        self.num_pref = np.random.randint(1,12)

        self.type_pref = [] # 4 types
        self.color_pref = [] # 6 colors
        self.size_pref = [] # 3 sizes

        for i in range(self.num_pref):
            self.type_pref.append(np.random.randint(0, 3))
            self.color_pref.append(np.random.randint(0, 5))
            self.size_pref.append(np.random.randint(0, 2))

        """
            player: 
        """
        base_weight = 0.1

        all_ids = np.arange(num_suitors)
        recipient_ids = all_ids[all_ids != suitor_id]
        for i in recipient_ids:
            self.weights[i] = {
                'sizes': {
                    FlowerSizes.Small: base_weight,
                    FlowerSizes.Medium: base_weight,
                    FlowerSizes.Large: base_weight
                },
                'colors': {
                    FlowerColors.Red: base_weight,
                    FlowerColors.Purple: base_weight,
                    FlowerColors.Yellow: base_weight,
                    FlowerColors.Blue: base_weight,
                    FlowerColors.White: base_weight,
                    FlowerColors.Orange: base_weight
                },
                'types': {
                    FlowerTypes.Rose: base_weight,
                    FlowerTypes.Chrysanthemum: base_weight,
                    FlowerTypes.Tulip: base_weight,
                    FlowerTypes.Begonia: base_weight
                },
                'number': np.ones(12) * base_weight
            }

        super().__init__(days, num_suitors, suitor_id, name='g7')

    # Random bouquet building
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

        if (not recipient_id in self.bouq_Dict.keys()):
            self.bouq_Dict[recipient_id] = [[chosen_flower_counts, -1, -1]]
        else:
            self.bouq_Dict[recipient_id].append([chosen_flower_counts, -1, -1])

        return self.suitor_id, recipient_id, chosen_bouquet

    # "smart" building
    def _prepare_bouquet_inter_rounds(self, remaining_flowers, recipient_id):
        num_remaining = sum(remaining_flowers.values())
        size = np.argmax(self.weights[recipient_id]['number'])
        flip = np.random.randint(0, 1)
        if flip == 0:
            size -= 1
        else:
            size += 1
        if size > num_remaining:
            size = num_remaining

        # changes = 0
        if size > 0:
            scored_flowers = []
            for flower in remaining_flowers.keys():
                reward = 0
                reward += self.weights[recipient_id]['sizes'][flower.size]
                reward += self.weights[recipient_id]['colors'][flower.color]
                reward += self.weights[recipient_id]['types'][flower.type]
                for i in range(remaining_flowers[flower]):
                    scored_flowers.append([flower, reward])
            scored_flowers = sorted(scored_flowers, key=lambda item: item[1], reverse=True)
            
            chosen_flowers = []
            for cf in scored_flowers[:size-1]:
                chosen_flowers.append(cf[0])
            chosen_flower_counts = dict(Counter(chosen_flowers))
            for k, v in chosen_flower_counts.items():
                remaining_flowers[k] -= v
                assert remaining_flowers[k] >= 0
        else:
            chosen_flower_counts = dict()
        chosen_bouquet = Bouquet(chosen_flower_counts)

        self.bouq_Dict[recipient_id].append([chosen_flower_counts, -1, -1])
        return self.suitor_id, recipient_id, chosen_bouquet

    def _prepare_bouquet_last_round(self, remaining_flowers, recipient_id):
        num_remaining = sum(remaining_flowers.values())
        size = np.argmax(self.weights[recipient_id]['number'])

        if size > 0:
            scored_flowers = []
            for flower in remaining_flowers.keys():
                reward = 0
                reward += self.weights[recipient_id]['sizes'][flower.size]
                reward += self.weights[recipient_id]['colors'][flower.color]
                reward += self.weights[recipient_id]['types'][flower.type]
                for i in range(remaining_flowers[flower]):
                    scored_flowers.append([flower, reward])
            scored_flowers = sorted(scored_flowers, key=lambda item: item[1], reverse=True)
            
            chosen_flowers = []
            for cf in scored_flowers[:size-1]:
                chosen_flowers.append(cf[0])
            chosen_flower_counts = dict(Counter(chosen_flowers))
            for k, v in chosen_flower_counts.items():
                remaining_flowers[k] -= v
                assert remaining_flowers[k] >= 0
        else: # empty is optimal but do we want to give that?
            chosen_flower_counts = dict()
        chosen_bouquet = Bouquet(chosen_flower_counts)

        self.bouq_Dict[recipient_id].append([chosen_flower_counts, -1, -1])
        return self.suitor_id, recipient_id, chosen_bouquet
        # best_bouquet_score = max(self.bouq_Dict[recipient_id], key=lambda e: int(e[1]))
        # best_bouquet = best_bouquet_score[0]
        # best_score = best_bouquet_score[1]
        # num_remaining = sum(remaining_flowers.values())
        # size = int(np.random.randint(0, min(MAX_BOUQUET_SIZE, num_remaining) + 1))
        # changes = 0
        # if size > 0:
        #     chosen_flower_counts = dict()
        #     for flower, count in best_bouquet.items():
        #         if (flower in remaining_flowers):
        #             if remaining_flowers[flower] >= count:
        #                 chosen_flower_counts[flower] = count
        #             elif remaining_flowers[flower] > 0:
        #                 chosen_flower_counts[flower] = remaining_flowers[flower]
        # else:
        #     chosen_flower_counts = dict()
        # chosen_bouquet = Bouquet(chosen_flower_counts)

        # return self.suitor_id, recipient_id, chosen_bouquet

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
        # First day ~ self.days/2: pass randomly
        if (self.days_remaining < self.days/2):
            # Increment days_remaining
            self.days_remaining += 1
            return list(map(lambda recipient_id: self._prepare_bouquet(remaining_flowers, recipient_id), recipient_ids))

        # Last day: best bouquet
        elif(self.days_remaining == self.days):
            return list(map(lambda recipient_id: self._prepare_bouquet_last_round(remaining_flowers, recipient_id), recipient_ids))
            
        # Every day in between
        else:
            # Increment days_remaining
            self.days_remaining += 1
            """ Step 1: weighted score of bouquets
            score prio = score * (num_players - rank)
            """
            # Dictonary of high scores for every player
            score_Dict = {}
            all_ids = np.arange(self.num_suitors)
            recipient_ids = all_ids[all_ids != self.suitor_id]
            for i in recipient_ids:
                bouqs = self.bouq_Dict[i]
                highest = 0
                for bouq in bouqs:
                    score_prio = bouq[1] * (self.num_suitors - bouq[2])
                    if score_prio > highest:
                        highest = score_prio
                score_Dict[i] = highest
            
            score_Dict = dict(sorted(score_Dict.items(), key=lambda item: item[1], reverse=True))

            return list(map(lambda recipient_id: self._prepare_bouquet_inter_rounds(remaining_flowers, recipient_id), score_Dict.keys()))

    def zero_score_bouquet(self):
        """
        :return: a Bouquet for which your scoring function will return 0
        """
        return Bouquet({})

    def one_score_bouquet(self):
        """
        :return: a Bouquet for which your scoring function will return 1
        """
        bouq = {}
        for i in range(self.num_pref):
            f = Flower(
                size = self.size_pref[i],
                color = self.color_pref[i],
                type = self.type_pref[i]
            )
            if f in bouq.keys():
                bouq[f] += 1
            else:
                bouq[f] = 1

        return Bouquet(bouq)

    def score_types(self, types: Dict[FlowerTypes, int]):
        """
        :param types: dictionary of flower types and their associated counts in the bouquet
        :return: A score representing preference of the flower types in the bouquet
        """
        if(FlowerTypes.Tulip in types.keys() or len(types) == 0):
            return 0.0
        elif(FlowerTypes.Chrysanthemum in types.keys()):
            return 1/3
        else:
            return 0.1
        # max_type_score = 0.25 / 4
        # type_score = 0
        # type_count = []
        # for i in range(4):
        #     type_count.append(self.type_pref.count(i))
        
        # for i in range(len(types.keys())):
        #     optimum_count = type_count[i]
        #     zero_count = 0
        #     if (12 - optimum_count > optimum_count):
        #         zero_count = 12
        #     else:
        #         zero_count = 0

        #     type_score += max_type_score - (abs(optimum_count - types[list(types.keys())[i]]) / abs(zero_count - optimum_count))
        
        # return type_score

    def score_colors(self, colors: Dict[FlowerColors, int]):
        """
        :param colors: dictionary of flower colors and their associated counts in the bouquet
        :return: A score representing preference of the flower colors in the bouquet
        """
        if(FlowerColors.Yellow in colors.keys() or len(colors) == 0):
            return 0.0
        elif(FlowerColors.Red in colors.keys()):
            return 1/3
        else:
            return 0.1
        # max_color_score = 0.25 / 6
        # color_score = 0
        # color_count = []
        # for i in range(6):
        #     color_count.append(self.color_pref.count(i))
        
        # for i in range(len(colors.keys())):
        #     optimum_count = color_count[i]
        #     zero_count = 0
        #     if (12 - optimum_count > optimum_count):
        #         zero_count = 12
        #     else:
        #         zero_count = 0

        #     color_score += max_color_score - (abs(optimum_count - colors[list(colors.keys())[i]]) / abs(zero_count - optimum_count))
        
        # return color_score

    def score_sizes(self, sizes: Dict[FlowerSizes, int]):
        """
        :param sizes: dictionary of flower sizes and their associated counts in the bouquet
        :return: A score representing preference of the flower sizes in the bouquet
        """
        if(FlowerSizes.Medium in sizes.keys() or len(sizes) == 0):
            return 0.0
        elif(FlowerSizes.Large in sizes.keys()):
            return 1/3
        else:
            return 0.1
        # max_size_score = 0.25 / 3
        # size_score = 0
        # size_count = []
        # for i in range(3):
        #     size_count.append(self.type_pref.count(i))
        
        # for i in range(len(sizes.keys())):
        #     optimum_count = size_count[i]
        #     zero_count = 0
        #     if (12 - optimum_count > optimum_count):
        #         zero_count = 12
        #     else:
        #         zero_count = 0

        #     size_score += max_size_score - (abs(optimum_count - sizes[list(sizes.keys())[i]]) / abs(zero_count - optimum_count))
        
        # return size_score

    def receive_feedback(self, feedback):
        """
        :param feedback:
        :return: nothing
        """
        all_ids = np.arange(self.num_suitors)
        recipient_ids = all_ids[all_ids != self.suitor_id]

        for id in recipient_ids:
            # store bouquet to score
            lastbouquet = self.bouq_Dict[id][-1]
            lastbouquet[1] = feedback[id][1]
            lastbouquet[2] = feedback[id][0]
            self.bouq_Dict[id][-1] = lastbouquet

            # estimate reward w current weights
            flower_dict = {
                'sizes': {
                    FlowerSizes.Small: 0,
                    FlowerSizes.Medium: 0,
                    FlowerSizes.Large: 0
                },
                'colors': {
                    FlowerColors.Red: 0,
                    FlowerColors.Purple: 0,
                    FlowerColors.Yellow: 0,
                    FlowerColors.Blue: 0,
                    FlowerColors.White: 0,
                    FlowerColors.Orange: 0
                },
                'types': {
                    FlowerTypes.Rose: 0,
                    FlowerTypes.Chrysanthemum: 0,
                    FlowerTypes.Tulip: 0,
                    FlowerTypes.Begonia: 0
                },
                'number': 0
            }
            for flower in lastbouquet[0].keys():
                flower_dict['sizes'][flower.size] += 1
                flower_dict['colors'][flower.color] += 1
                flower_dict['types'][flower.type] += 1
                flower_dict['number'] += 1

            estimate = 0
            for att in flower_dict.keys():
                if att == 'number':
                    estimate += self.weights[id][att][flower_dict[att] - 1] * flower_dict[att]
                    estimate /= flower_dict[att]
                else:
                    for sp in flower_dict[att].keys():
                        estimate += (self.weights[id][att][sp] * flower_dict[att][sp])

            diff = lastbouquet[1] - estimate
            diff_fraq = diff / ((flower_dict['number'] * 3) + 1)
            for att in flower_dict.keys():
                if att == 'number':
                    curr = self.weights[id][att][flower_dict[att] - 1]
                    self.weights[id][att][flower_dict[att] - 1] = curr + diff_fraq
                else:
                    for sp in flower_dict[att].keys():
                        curr = self.weights[id][att][sp]
                        self.weights[id][att][sp] = curr + (flower_dict[att][sp] * diff_fraq)

        self.feedback.append(feedback)