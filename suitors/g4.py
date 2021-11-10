from typing import Dict
from collections import Counter, defaultdict
from random import shuffle, choice, randint
import math
import random as rand
import numpy as np

from flowers import Bouquet, Flower, FlowerSizes, FlowerColors, FlowerTypes
from suitors.base import BaseSuitor
from utils import flatten_counter
from constants import MAX_BOUQUET_SIZE


class Suitor(BaseSuitor):
    def __init__(self, days: int, num_suitors: int, suitor_id: int):
        """
        :param days: number of days of courtship
        :param num_suitors: number of suitors, including yourself
        :param suitor_id: unique id of your suitor in range(num_suitors)
        """
        super().__init__(days, num_suitors, suitor_id, name='g4')
        self.total_turns = days
        self.remaining_turns = self.total_turns
        all_ids = np.arange(num_suitors)
        self.recipient_ids = all_ids[all_ids != suitor_id]
        self.to_be_tested = self._generate_exp_groups_single()
        self.train_feedback = np.zeros((len(self.recipient_ids), 6, 4, 3))  # feedback score from other recipients
        self.last_bouquet = None  # bouquet we gave out from the last turn
        self.control_group_assignments = self._assign_control_groups()

        self.best_arrangement = self.generate_random_bouquet()
        self.best_arrangement_size_vec, self.best_arrangement_color_vec, self.best_arrangement_type_vec = \
        self.get_bouquet_score_vectors(self.best_arrangement)
        self.experiments = {}
        for i in range(num_suitors):
            if i != suitor_id:
                self.experiments[i] = defaultdict(list)
        self.suitor_id = suitor_id # Added this line

    @staticmethod
    def _get_combinations(list1, list2):
        return [[list1[i], list2[j]] for i in range(len(list1)) for j in range(len(list2))]

    def generate_random_flower(self):
        return Flower(
            size = choice(list(FlowerSizes)),
            color = choice(list(FlowerColors)),
            type = choice(list(FlowerTypes)), 
        )

    def generate_random_bouquet(self):
        return [self.generate_random_flower() for _ in range(randint(1,MAX_BOUQUET_SIZE))]

    def get_bouquet_score_vectors(self, bouquet_vect):
        size_vec = [0] * len(FlowerSizes)
        color_vec = [0] * len(FlowerColors)
        type_vec = [0] * len(FlowerTypes)
        for flower in bouquet_vect:
            size_vec[flower.size.value] += 1
            color_vec[flower.color.value] += 1
            type_vec[flower.type.value] += 1

        return size_vec, color_vec, type_vec

    def compute_cosine_sim(self, v1, v2):
        # v1 and v2 are all positive numbers, so bounded from 0 to 1 
        return (v1 @ v2.T) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def compute_euc_dist(self, v1, v2):
        # can make higher norm to increase steepness?
        return np.linalg.norm(np.array(v1) - np.array(v2))

    def compute_distance_heuristic(self, v1,v2):
        # v1 and v2 bounded from 0 to inf
        # need a cutoff to guarantee 0 score is possible. (e.g, assume we add distances together)
        # worst case scenario, theoretically the 
        # smallest maximum distance is assume the optimal bq is 12 flowers distributed amongst all attributes:
        # e.g, [4,4,4], [6,6], [3,3,3,3]
        # then according to our scoring heuristic, the 12 - min(vector) has to result in 0 (otherwise, it is possible
        # that no bq can result in a zero score.
        most_even_dist = math.floor(12/len(v1))
        threshold_vect = [0] * len(v1)
        threshold_vect[0] = 12
        THRESHOLD = 1 / (
            np.linalg.norm(
                np.array(threshold_vect)-
                np.array([most_even_dist] * len(v1))
            ) + 1
        ) 
        dist =  1 / (self.compute_euc_dist(v1,v2) + 1)
        return dist if dist > THRESHOLD else 0
      

    def _assign_control_groups(self):
        """
        Get control group assignments for all recipients.
        :return: assignments[i]['color_control'] = [t, s] a combination of type=t and size=s as the color controlled
                 experiments setup for recipient i.
        """
        # get combinations
        fc_exp_options = [FlowerColors(i) for i in range(6)]
        ft_exp_options = [FlowerTypes(i) for i in range(4)]
        fs_exp_options = [FlowerSizes(i) for i in range(3)]
        fc_control_options = self._get_combinations(ft_exp_options, fs_exp_options)
        ft_control_options = self._get_combinations(fc_exp_options, fs_exp_options)
        fs_control_options = self._get_combinations(fc_exp_options, ft_exp_options)

        # shuffle the combinations to make it less likely for recipients to get similar fixed values
        # e.g. we want to avoid assigning player1=[rose, small] and player2=[rose, medium] for color experiments when
        # there are only 2 players in the game (i.e. few players)
        rand.shuffle(fc_control_options)
        rand.shuffle(ft_control_options)
        rand.shuffle(fs_control_options)

        assignments = {}
        for recipient in self.recipient_ids:

            assignments[recipient] = {
                'fc_control': fc_control_options[recipient % len(fc_control_options)],
                'ft_control': ft_control_options[recipient % len(ft_control_options)],
                'fs_control': fs_control_options[recipient % len(fs_control_options)],
            }

        return assignments

    def _generate_exp_groups_single(self):
        to_be_tested = {}
        # TODO can set up a ratio for different variables that ensure each variable is getting tested within #days
        for recipient_id in self.recipient_ids:
            to_be_tested[str(recipient_id)] = []
            to_be_tested[str(recipient_id)].append([fc for fc in FlowerColors])
            to_be_tested[str(recipient_id)].append([ft for ft in FlowerTypes])
            to_be_tested[str(recipient_id)].append([fs for fs in FlowerSizes])

        return to_be_tested

    def able_to_create_bouquet(self, flowers, flowercount):
        for flower, count in flowers.arrangement.items():
            if flower in flowercount:
                if flowercount[flower] < count:
                    return False
            else:
                return False
        return True

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
        self.remaining_turns -= 1
        bouquet_for_all = []  # return value
        bouquet_for_all_and_etype = []
        flower_info = self._tabularize_flowers(flower_counts)

        if self.remaining_turns == 0:
            # TODO final round, give the bouquet with the highest score from the previous tryouts\
            for i in self.recipient_ids:
                go_random = False
                if len(self.experiments[i]) != 0:
                    highest_temp = -math.inf
                    maximum_bouquet= None
                    # for each recipient, if we have data for them, get the highest score and return the same combination to them
                    sortedList = []
                    for j in self.experiments[i].values():
                        sortedList.extend(j)
                    sortedList.sort(key = lambda x: x[1], reverse = True)
                    canMake = False
                    for flowers, score in sortedList[:math.ceil(len(sortedList)/2)]:
                        if self.able_to_create_bouquet(flowers, flower_counts):
                            canMake = True
                            bouquet_for_all.append([self.suitor_id, i, flowers])
                            for flower,count in flowers.arrangement.items():
                                flower_counts[flower] -= count
                            break
                    go_random = not canMake
                else:
                    go_random = True
                if go_random:
                    # random bouquets if go_random is true
                    recipient_id = i
                    remaining_flowers = flower_counts.copy()
                    num_remaining = sum(remaining_flowers.values())
                    size = int(np.random.randint(0, min(MAX_BOUQUET_SIZE, num_remaining) + 1))
                    if size > 0:
                        chosen_flowers = np.random.choice(flatten_counter(remaining_flowers), size=(size,), replace=False)
                        chosen_flower_counts = dict(Counter(chosen_flowers))
                        for k, v in chosen_flower_counts.items():
                            remaining_flowers[k] -= v
                            assert remaining_flowers[k] >= 0
                    else:
                        chosen_flower_counts = dict()
                    chosen_bouquet = Bouquet(chosen_flower_counts)
                    bouquet_for_all.append([self.suitor_id, recipient_id, chosen_bouquet])
                    flower_counts = remaining_flowers

            return bouquet_for_all

        else:  # training phase: conduct controlled experiments
            for ind in range(len(self.recipient_ids)):
                recipient_id = self.recipient_ids[ind]
                chosen_flowers, exp_type, flower_info = self._prepare_bouquet(flower_info, recipient_id)

                # build the bouquet
                chosen_flower_counts = dict(Counter(np.asarray(chosen_flowers)))
                chosen_bouquet = Bouquet(chosen_flower_counts)
                bouquet_for_all.append([self.suitor_id, recipient_id, chosen_bouquet])
                bouquet_for_all_and_etype.append([self.suitor_id, recipient_id, chosen_bouquet, exp_type])

            if len(self.feedback) > 0:
                self.update_results()

            # update last_bouquet
            self.last_bouquet = bouquet_for_all_and_etype

            return bouquet_for_all

    def _prepare_bouquet(self, flower_info, recipient_id):
        chosen_flowers = []  # for building a bouquet later
        tested = False
        C_T_S_SPLIT = 1./3  # even split between the three experiment categories
        exp_type = None  # options: 'color', 'type', 'size', None

        # flower color: first C_T_S_SPLIT proportion of the game
        if self.remaining_turns / self.total_turns >= C_T_S_SPLIT * 2:

            # get the fixed [size, type] setting for this recipient for the color experiments
            fc_control = self.control_group_assignments[recipient_id]['fc_control']
            fixed_ft = fc_control[0].value
            fixed_fs = fc_control[1].value

            # grab flower counts that match with fc_control for this round: list of length 6
            fc_exp_options = flower_info[:, fixed_ft, fixed_fs]
            fc_exp = []
            if sum(fc_exp_options) > 0:  # if there are flowers to work with

                # randomly generate a flower count for each color from the available flowers
                fc_exp = [rand.choice(list(range(fc_exp_options[i] + 1))) for i in range(len(fc_exp_options))]
                exp_type = 'color'

                for fc_ind in range(len(fc_exp)):  # iterate over all colors
                    for _ in range(fc_exp[fc_ind]):  # append flower(s) with color=fc_ind
                        chosen_flowers.append(Flower(color=FlowerColors(fc_ind),
                                                     type=FlowerTypes(fixed_ft),
                                                     size=FlowerSizes(fixed_fs)))
                    flower_info[fc_ind, fixed_ft, fixed_fs] -= fc_exp[fc_ind]  # decrement flower_info


            else:  # if there are no flower with the fc_control [size, type] setting
                chosen_flowers, flower_info = self._generate_rand_bouquet(flower_info)

        # flower type: second C_T_S_SPLIT proportion of the game
        elif self.remaining_turns / self.total_turns >= C_T_S_SPLIT:
            ft_control = self.control_group_assignments[recipient_id]['ft_control']
            fixed_fc = ft_control[0].value
            fixed_fs = ft_control[1].value
            ft_exp_options = flower_info[fixed_fc, :, fixed_fs]
            if sum(ft_exp_options) > 0:
                ft_exp = [rand.choice(list(range(ft_exp_options[i] + 1))) for i in range(len(ft_exp_options))]
                exp_type = 'type'
                for ft_ind in range(len(ft_exp)):
                    for _ in range(ft_exp[ft_ind]):
                        chosen_flowers.append(Flower(color=FlowerColors(fixed_fc),
                                                     type=FlowerTypes(ft_ind),
                                                     size=FlowerSizes(fixed_fs)))
                    flower_info[fixed_fc, ft_ind, fixed_fs] -= ft_exp[ft_ind]
            else:
                chosen_flowers, flower_info = self._generate_rand_bouquet(flower_info)

        # flower size: third C_T_S_SPLIT proportion of the game
        else:
            fs_control = self.control_group_assignments[recipient_id]['fs_control']
            fixed_fc = fs_control[0].value
            fixed_ft = fs_control[1].value
            fs_exp_options = flower_info[fixed_fc, fixed_ft, :]
            if sum(fs_exp_options) > 0:
                fs_exp = [rand.choice(list(range(fs_exp_options[i] + 1))) for i in range(len(fs_exp_options))]
                exp_type = 'size'
                for fs_ind in range(len(fs_exp)):
                    for _ in range(fs_exp[fs_ind]):
                        chosen_flowers.append(Flower(color=FlowerColors(fixed_fc),
                                                     type=FlowerTypes(fixed_ft),
                                                     size=FlowerSizes(fs_ind)))
                    flower_info[fixed_fc, fixed_ft, fs_ind] -= fs_exp[fs_ind]
            else:
                chosen_flowers, flower_info = self._generate_rand_bouquet(flower_info)

        return chosen_flowers, exp_type, flower_info

    def _generate_rand_bouquet(self, flower_info):
        chosen_flowers = []
        remaining_flowers = self._list_flowers(flower_info)
        num_remaining = sum(remaining_flowers.values())
        size = int(np.random.randint(0, min(MAX_BOUQUET_SIZE, num_remaining) + 1))
        if size > 0:
            chosen_flowers = np.random.choice(flatten_counter(remaining_flowers), size=(size,), replace=False)
        for chosen_flower in chosen_flowers:
            flower_info[chosen_flower.color.value, chosen_flower.type.value, chosen_flower.size.value] -= 1
        return chosen_flowers, flower_info

    # Helper function that adds to results
    # Make sure that last_bouquet is in the correct player order (i.e. suitor 0 is index 0)
    def update_results(self):
        results = self.feedback[-1]
        for i in range(len(results)):
            if i != self.suitor_id:
                player = self.experiments[i]

                given, experiment = self.last_bouquet[list(self.recipient_ids).index(i)][2], self.last_bouquet[list(self.recipient_ids).index(i)][3]

                player[experiment].append((given, results[i][1]))
    
    @staticmethod
    def _tabularize_flowers(flower_counts):
        flowers = flower_counts.keys()
        flower_info = np.zeros((6, 4, 3), dtype=int)  # (color, type, size)
        for flower in flowers:
            flower_info[flower.color.value][flower.type.value][flower.size.value] = flower_counts[flower]
        return flower_info

    @staticmethod
    def _list_flowers(flower_info):
        flower_counts = {}
        for c in range(6):
            for t in range(4):
                for s in range(3):
                    if flower_info[c][t][s] > 0:
                        flower = Flower(
                                        size=FlowerSizes(s),
                                        color=FlowerColors(c),
                                        type=FlowerTypes(t)
                                    )
                        flower_counts[flower] = flower_info[c][t][s]
        return flower_counts

    def zero_score_bouquet(self):
        """
        :return: a Bouquet for which your scoring function will return 0
        """
        # is there a guaranteed min?
        # i think the min is when we pick one attribute with 12 flowers, which is minimal in the best_arrangement vector
        # someone check that logic but I think it works?
        worstFlower = Flower(
            size = self.get_min_vector_attribute(self.best_arrangement_size_vec, FlowerSizes),
            color = self.get_min_vector_attribute(self.best_arrangement_color_vec, FlowerColors),
            type=self.get_min_vector_attribute(self.best_arrangement_type_vec, FlowerTypes)
        )
        return Bouquet({worstFlower: 12})

    def get_min_vector_attribute(self, vector, enumType):
        zipped = list(zip(vector, enumType))
        return min(zipped)[1]

    def one_score_bouquet(self):
        """
        :return: a Bouquet for which your scoring function will return 1
        """
        # the below is still true
        return self.best_arrangement

    def score_types(self, types: Dict[FlowerTypes, int]):
        """
        :param types: dictionary of flower types and their associated counts in the bouquet
        :return: A score representing preference of the flower types in the bouquet
        """
        # each vector is like [FlowerType.1 Count, FlowerType.2 Count]
        vector = [0] * len(FlowerTypes)

        for key, value in types.items():
            vector[key.value] = value

        return self.compute_distance_heuristic(vector, self.best_arrangement_type_vec) * 1.0/3.0

    def score_colors(self, colors: Dict[FlowerColors, int]):
        """
        :param colors: dictionary of flower colors and their associated counts in the bouquet
        :return: A score representing preference of the flower colors in the bouquet
        """
        vector = [0] * len(FlowerColors)

        for key, value in colors.items():
            vector[key.value] = value

        return self.compute_distance_heuristic(vector, self.best_arrangement_color_vec) * 1.0/3.0

    def score_sizes(self, sizes: Dict[FlowerSizes, int]):
        """
        :param sizes: dictionary of flower sizes and their associated counts in the bouquet
        :return: A score representing preference of the flower sizes in the bouquet
        """
        vector = [0] * len(FlowerSizes)

        for key, value in sizes.items():
            vector[key.value] = value

        return self.compute_distance_heuristic(vector, self.best_arrangement_size_vec) * 1.0/3.0

    def receive_feedback(self, feedback):
        """
        :param feedback:
        :return: nothing
        """
        self.feedback.append(feedback)
        print('Feedback added')
        print(self.feedback)
        print(self.feedback[-1])
