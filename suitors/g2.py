from typing import Dict

from flowers import Bouquet, Flower, FlowerSizes, FlowerColors, FlowerTypes
from suitors.base import BaseSuitor

import numpy as np
from copy import deepcopy
import random
from collections import defaultdict, Counter
from itertools import combinations, chain
import math

"""
class FlowerSizes(Enum):
    Small = 0
    Medium = 1
    Large = 2
class FlowerColors(Enum):
    White = 0
    Yellow = 1
    Red = 2
    Purple = 3
    Orange = 4
    Blue = 5
class FlowerTypes(Enum):
    Rose = 0
    Chrysanthemum = 1
    Tulip = 2
    Begonia = 3
"""


class Suitor(BaseSuitor):
    def __init__(self, days: int, num_suitors: int, suitor_id: int):
        """
        :param days: number of days of courtship
        :param num_suitors: number of suitors, including yourself
        :param suitor_id: unique id of your suitor in range(num_suitors)
        """
        self.suitor_id = suitor_id
        self.scoring_parameters = {}
        self.other_suitors = []
        self.bouquets_given = defaultdict(list)
        self.turn = 1
        self.learning_rate = 10 + (90 / days)
        self.exploration_alpha = 0.3
        self.exploration_alpha_decay = self.exploration_alpha / days
        self.num_suitors = num_suitors - 1

        self.num_flowers_in_bouquet = self.get_random_num_flowers(seed=2)
        self.our_favorite_bouquet = self.get_random_bouquet(num_flowers = self.num_flowers_in_bouquet)
        self.scoring_weight = {'type': 0.3, 'color':0.5, 'size':0.2}
        random_low = -1.0
        random_high = 1.0
        for i in range(num_suitors):
            if i != suitor_id:
                self.other_suitors.append(i)
                self.scoring_parameters[i] = {
                    FlowerSizes.Small: round(random.uniform(random_low,random_high), 2),
                    FlowerSizes.Medium: round(random.uniform(random_low,random_high), 2),
                    FlowerSizes.Large: round(random.uniform(random_low,random_high), 2),
                    FlowerColors.White: round(random.uniform(random_low,random_high), 2),
                    FlowerColors.Yellow: round(random.uniform(random_low,random_high), 2),
                    FlowerColors.Red: round(random.uniform(random_low,random_high), 2),
                    FlowerColors.Purple: round(random.uniform(random_low,random_high), 2),
                    FlowerColors.Orange: round(random.uniform(random_low,random_high), 2),
                    FlowerColors.Blue: round(random.uniform(random_low,random_high), 2),
                    FlowerTypes.Rose: round(random.uniform(random_low,random_high), 2),
                    FlowerTypes.Chrysanthemum: round(random.uniform(random_low,random_high), 2),
                    FlowerTypes.Tulip: round(random.uniform(random_low,random_high), 2),
                    FlowerTypes.Begonia: round(random.uniform(random_low,random_high), 2),
                }

        super().__init__(days, num_suitors, suitor_id, name='g2')


    def get_random_num_flowers(self, seed=1):
        return random.randint(1, 12)

    def get_random_bouquet(self, num_flowers=12):
        size_means = self.split_counts([size for size in FlowerSizes], num_flowers)
        type_means = self.split_counts([type for type in FlowerTypes], num_flowers)
        color_means = self.split_counts([color for color in FlowerColors], num_flowers)
        return {
            'sizes': size_means,
            'types': type_means,
            'colors': color_means
        }

    def split_counts(self, items_to_split, max_number):
        to_return = {}
        running_sum = 0
        for i in range(len(items_to_split) - 1):
            item = items_to_split[i]
            num_of_item = random.randint(0, max_number - running_sum)
            running_sum += num_of_item
            to_return[item] = num_of_item
        to_return[items_to_split[-1]] = max_number - running_sum
        return to_return

    def prepare_bouquet_for_group(self, group_id, flowers, copy_flower_counts, count = 4, rand=False, last=False):
        bouquet = defaultdict(int)
        bouquet_info = defaultdict(int)
        scoring_function = self.scoring_parameters[group_id]
        prev_bouquets = self.bouquets_given[group_id]
        if rand:
            random.shuffle(flowers)
            for _ in range(count):
                for item in flowers:
                    key,value = item
                    if copy_flower_counts[str(key)] <= 0:
                        continue
                    bouquet[key] += 1
                    copy_flower_counts[str(key)] -= 1
                    break
        
        elif last:
            #can_best = True
            best_score = float('-inf')
            best_bouquet = None
            #flower_counts = {}
            index = 0
            for p in prev_bouquets:
                score = p[2]
                b = p[0]
                if score >= best_score:
                    best_score = score
                    best_bouquet = b
            if best_bouquet != None:
                for flower in best_bouquet.flowers():
                    if str(flower) in copy_flower_counts:
                        if copy_flower_counts[str(flower)] > 0:
                            bouquet[flower] += 1
                            copy_flower_counts[str(flower)] -= 1
                            continue
                    one = 0
                    index = 0
                    picked = False
                    for flow in flowers:
                        similarity = 0
                        if flow[0].type == flower.type:
                            similarity +=1
                        if flow[0].size == flower.size:
                            similarity +=1
                        if flow[0].color == flower.color:
                            similarity +=1
                        if similarity == 2:
                            if copy_flower_counts[str(flow[0])] > 0:
                                bouquet[flow[0]] += 1
                                copy_flower_counts[str(flow[0])] -= 1
                                picked = True
                                break
                        if similarity == 1:
                            if copy_flower_counts[str(flow[0])] > 0:
                                one = index
                        
                        index += 1

                    if not picked:
                        flow = flowers[one][0]
                        bouquet[flow] += 1
                        copy_flower_counts[str(flow)] -= 1
                """
                flows = []
                for flower in flowers:
                    for i in range(flower[1]):
                        flows.append(flower[0])
                print(len(flows))
                types = defaultdict(int, best_bouquet.types)
                sizes = defaultdict(int, best_bouquet.sizes)
                colors = defaultdict(int, best_bouquet.colors)
                combos = combinations(flows, len(best_bouquet))
                max_bouquet = None
                min_dist = float('inf')
                index = 0
                for c in combos:
                    #print(index)
                    index +=1
                    distance = 0
                    b = Bouquet(Counter(c))
                    for size in FlowerSizes:
                        if size not in b.sizes:
                            distance += sizes[size]
                        else:
                            distance += abs(sizes[size] - b.sizes[size])
                    for type in FlowerTypes:
                        if type not in b.types:
                            distance += types[type]
                        else:
                            distance += abs(types[type] - b.types[type])
                    for color in FlowerColors:
                        if color not in b.colors:
                            distance += colors[color]
                        else:
                            distance += abs(colors[color] - b.colors[color])
                    if distance < min_dist:
                        max_bouquet = c
                        min_dist = distance
                for f in max_bouquet:
                    bouquet[f] += 1
                    copy_flower_counts[str(f)] -= 1
                can_best = True
            else:
                can_best = False
            if not can_best:
                bouquet = defaultdict(int)
                for flower in flower_counts:
                    copy_flower_counts[flower] += flower_counts[flower]
                for _ in range(count):
                    best_flower = None
                    best_score = -10000
                    for item in flowers:
                        key,value = item
                        score = 0
                        if copy_flower_counts[str(key)] <= 0:
                            continue
                        
                        score += scoring_function[key.type] - bouquet_info[key.type]
                        score += scoring_function[key.color] - bouquet_info[key.color]
                        score += scoring_function[key.size] - bouquet_info[key.size]

                        if score > best_score:
                            best_score = score
                            best_flower = key
                    
                    if best_flower == None:
                        break
                    else:
                        bouquet[best_flower] += 1
                        copy_flower_counts[str(best_flower)] -= 1
                        """
        else:
            for _ in range(count):
                best_flower = None
                best_score = -10000
                for item in flowers:
                    key,value = item
                    score = 0
                    if copy_flower_counts[str(key)] <= 0:
                        continue
                    
                    score += scoring_function[key.type] - bouquet_info[key.type]
                    score += scoring_function[key.color] - bouquet_info[key.color]
                    score += scoring_function[key.size] - bouquet_info[key.size]

                    if score > best_score:
                        best_score = score
                        best_flower = key
                
                if best_flower == None:
                    break
                elif best_score > 0:
                    bouquet[best_flower] += 1
                    copy_flower_counts[str(best_flower)] -= 1
                else:
                    break

        return (self.suitor_id, group_id, Bouquet(bouquet)), copy_flower_counts

    
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
        min_groups_to_give = 3 if self.num_suitors > 4 else 2
        groups_to_give = max(min_groups_to_give, self.num_suitors - (self.num_suitors - min_groups_to_give) * (self.turn / self.days))
        flower_for_each_group = int(len(flower_counts) // (groups_to_give))
        bouquets = []

        copy_flower_counts = {}
        for key,value in flower_counts.items():
            copy_flower_counts[str(key)] = value 
        
        flowers = [(key,value) for key,value in flower_counts.items()]

        if self.turn > 1:
            self.rank_groups()

        for o_id in self.other_suitors:
            r = random.uniform(0,1)
            if r < self.exploration_alpha or self.turn == 1:

                b, copy_flower_counts = self.prepare_bouquet_for_group(o_id, flowers, copy_flower_counts, rand=True)
            elif self.turn >= self.days:
                b, copy_flower_counts = self.prepare_bouquet_for_group(o_id, flowers, copy_flower_counts, rand=False, last=True)

            else: 
                b, copy_flower_counts = self.prepare_bouquet_for_group(o_id, flowers, copy_flower_counts, count = flower_for_each_group) 
            self.bouquets_given[o_id].append([b[2]])
            bouquets.append(b)

        self.turn += 1
        self.exploration_alpha -= self.exploration_alpha_decay
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
        def counts_to_list(counts):
            l = []
            for key in counts:
                l += ([key]*counts[key])
            return l
        all_sizes = counts_to_list(self.our_favorite_bouquet['sizes'])
        all_types = counts_to_list(self.our_favorite_bouquet['types'])
        all_colors = counts_to_list(self.our_favorite_bouquet['colors'])

        b = {}
        for i in range(self.num_flowers_in_bouquet):
            flower = Flower(
                size = all_sizes[i],
                color=all_colors[i],
                type = all_types[i]
            )
            b[flower] = 1

        return Bouquet(b)

    def calculate_distance(self, guessed_counts, target_counts, units):
        d = 0
        for unit in units:
            guess = guessed_counts[unit] if unit in guessed_counts else 0
            target = target_counts[unit] if unit in target_counts else 0
            d += abs(target - guess)
        return d/2 if d!=0 and d!=1 else d

    def score_types(self, types: Dict[FlowerTypes, int]):
        """
        :param types: dictionary of flower types and their associated counts in the bouquet
        :return: A score representing preference of the flower types in the bouquet
        """
        if sum(types.values()) == 0:
            return 0
        distance = self.calculate_distance(types, self.our_favorite_bouquet['types'], FlowerTypes)
        weight = self.scoring_weight['type']

        if distance == 0:
            return weight

        else:
            return weight / distance
        

    def score_colors(self, colors: Dict[FlowerColors, int]):
        """
        :param colors: dictionary of flower colors and their associated counts in the bouquet
        :return: A score representing preference of the flower colors in the bouquet
        """
        if sum(colors.values()) == 0:
            return 0
        distance = self.calculate_distance(colors, self.our_favorite_bouquet['colors'], FlowerColors)
        weight = self.scoring_weight['color']

        if distance == 0:
            return weight

        else:
            return weight / distance

    def score_sizes(self, sizes: Dict[FlowerSizes, int]):
        """
        :param sizes: dictionary of flower sizes and their associated counts in the bouquet
        :return: A score representing preference of the flower sizes in the bouquet
        """
        if sum(sizes.values()) == 0:
            return 0
        distance = self.calculate_distance(sizes, self.our_favorite_bouquet['sizes'], FlowerSizes)
        weight = self.scoring_weight['size']

        if distance == 0:
            return weight

        else:
            return weight / distance

    def adjust_scoring_function(self, prev_s, curr_s, o_id, bouquet):
        how_much = (curr_s - prev_s) * self.learning_rate

        for size, value in bouquet.sizes.items():
            self.scoring_parameters[o_id][size] += how_much * value

        for t, value in bouquet.types.items():
            self.scoring_parameters[o_id][t] += how_much * value

        for color, value in bouquet.colors.items():
            self.scoring_parameters[o_id][color] += how_much * value


    def receive_feedback(self, feedback):
        """
        :param feedback:
        :return: nothing
        """
        index = 0
        for f in feedback:
            if index == self.suitor_id:
                index += 1
                continue

            self.bouquets_given[index][-1] += [f[0],f[1]]

            if self.turn > 2:
                prev_score = self.bouquets_given[index][-2][2]
                curr_score = f[1]
                self.adjust_scoring_function(prev_score, curr_score, index, self.bouquets_given[index][-1][0])

            index += 1


        self.feedback.append(feedback)

    def rank_groups(self):
        g_rank_s = defaultdict(int)

        for o_id in self.other_suitors:
            rank = self.bouquets_given[o_id][-1][1]
            score = self.bouquets_given[o_id][-1][2]
            g_rank_s[o_id] = score/rank

        self.other_suitors = [k for k, v in sorted(g_rank_s.items(), key=lambda x: x[1], reverse=True)]
