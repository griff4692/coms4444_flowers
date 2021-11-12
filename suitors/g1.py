import random
from typing import Dict, List
from math import exp, log, factorial
from flowers import Bouquet, Flower, FlowerSizes, FlowerColors, FlowerTypes, MAX_BOUQUET_SIZE, \
    get_all_possible_flowers, sample_n_random_flowers, get_random_flower
from suitors.base import BaseSuitor
import numpy as np
from collections import defaultdict, Counter
from itertools import combinations_with_replacement
from utils import flatten_counter
from copy import deepcopy


class Suitor(BaseSuitor):
    def __init__(self, days: int, num_suitors: int, suitor_id: int):
        """
        :param days: number of days of courtship
        :param num_suitors: number of suitors, including yourself
        :param suitor_id: unique id of your suitor in range(num_suitors)
        """
        super().__init__(days, num_suitors, suitor_id, name='g1')
        # step 1: get the probability for bouquet range from 0 - MAX_BOUQUET_SIZE(12) given by the people size
        bouquet = BouquetSimulator(num_suitors)
        bouquet.simulate_give_flowers(10000)
        # step 2: get the probability for flower colors range from 0 - MAX_BOUQUET_SIZE(12)
        flowerColor = FlowerColorSimulator(list(range(MAX_BOUQUET_SIZE + 1)))
        flowerColor.simulate_possibilities(5000, bouquet.probability)
        self.flower_probability_table = flowerColor.probability

        self.color_map = {0: "W", 1: "Y", 2: "R", 3: "P", 4: "O", 5: "B"}

        if num_suitors > 2:
            exponent = log(1 - 1 / (num_suitors - 1)) / (days - 1)
            self.percentage = 1 - exp(exponent)
        else:
            self.percentage = 1 / (days - 1)

        # step 3: choose our one score flowers from the color probability table
        # score_one_flowers_for_us: the score of our choices of colors to be 1.0
        # key: length of flowers, value: the combination of the colors
        # should notice the value is a tuple, like('B', 'R') means 1 blue 1 red,
        # and also should notice that tuple is sorted,
        # so when to find whether exists a flower, sort the colors first then transfer it to tuple
        self.score_one_flowers_for_us = defaultdict(list)
        self.choose_one_score_bouquet_for_ourselves(self.flower_probability_table)

        self.current_day = 0
        # bouquet_history_score: remember the flowers and score of each recipient
        # key: recipient_id, value: dict(), key: score, value: Dict[Flower, int]
        self.bouquet_history_score = defaultdict(dict)
        # bouquet_history: remember the flowers of each recipient
        # key: recipient_id, value: list of flowers that we have given
        # it is used to help calculate bouquet_history_score
        self.bouquet_history = defaultdict(list)
        # the one score that we have taken so for for each recipient
        # key: recipient_id, value: the largest score we get from that recipient_id
        self.recipients_largest_score_ = {i: 0 for i in range(num_suitors) if i != suitor_id}
        # remember all the recipients' score so that we can decide which flower we can give at the last day
        self.recipients_all_score = []

        # array representing which suitors *potentially* have our defense strategy
        # -1 = Does not have our strat
        #  0 = Unknown
        #  1 = Has our strat
        self.has_our_strat = [0] * num_suitors

        # counter for how many times our strat has been checked, **arbitrary number**
        self.our_strat_count = [3] * num_suitors

        # bouqeut to color copy
        self.score_1_bouquet = [None] * num_suitors

    # choose one score bouquet randomly from the flowerColor.probability until reached the percentage
    def choose_one_score_bouquet_for_ourselves(self, probability_table: Dict):
        diff = self.percentage * pow(10, -5)
        remain_probability = self.percentage

        probability_table_list = defaultdict(list)
        for key in probability_table.keys():
            probability_table_list[key] = list(probability_table[key].items())

        while remain_probability > 0:
            # TODO: think of a way to break the remain_probability, what's the exact value, here I just assume 10^-5
            if remain_probability < diff:
                break
            # we don't consider the empty flowers to be score 1
            size = int(np.random.randint(1, MAX_BOUQUET_SIZE + 1))
            flower, probability = random.choice(probability_table_list[size])
            if probability <= remain_probability:
                remain_probability -= probability
                self.score_one_flowers_for_us[size].append(flower)

    # for each player, randomly guess what they want if we haven't got score 1 for their group
    def _prepare_bouquet(self, remaining_flowers, recipient_id):
        # TODO: if we got 1.0 score from that group before, skip it or we should keep guessing,
        #  because we may not get that 1 score in the final round given by the flowers

        if self.has_our_strat[recipient_id] == 0 and self.score_1_bouquet[recipient_id] != None:

            flowers_sent = self.score_1_bouquet[recipient_id]

            colors_count = [0]*6
            for f,i in flowers_sent.items():
                colors_count[f.color.value] += i

            rem_flowers_list = flatten_counter(remaining_flowers)
            # essentially choose a random bouquet of same color count
            random.shuffle(rem_flowers_list)

            chosen_flower_counts = dict()
            for f in rem_flowers_list:
                if colors_count[f.color.value] > 0:
                    if f not in chosen_flower_counts:
                        chosen_flower_counts[f] = 0
                    chosen_flower_counts[f] += 1
                    colors_count[f.color.value] -= 1

            # couldn't color copy
            if np.any(colors_count):
                chosen_flower_counts = dict()

            for f in chosen_flower_counts:
                remaining_flowers[f] -= 1

            self.bouquet_history[recipient_id].append(chosen_flower_counts)

        elif int(self.recipients_largest_score_[recipient_id]) == 1:
            chosen_flower_counts = dict()
        else:
            # randomly choosing the flowers
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

            # put the guess into the bouquet_history, the latest one is always the recent turn
            self.bouquet_history[recipient_id].append(chosen_flower_counts)

        chosen_bouquet = Bouquet(chosen_flower_counts)
        return self.suitor_id, recipient_id, chosen_bouquet

    # in the final round, choose the flowers given to each recipient
    # can use self.score_one_recipients, sort the dict by values,
    # and give flowers according to self.bouquet_history_score
    # choose check if we got that arrangement of flowers
    def prepare_for_marry_day(self, remaining_flowers: Dict[Flower, int], recipient_ids: List[int]):
        recipient_visited = set()
        self.recipients_all_score.sort(key=lambda x: -x[1])
        recipient_chosen_flowers = dict()

        for i in range(self.num_suitors):
            # suitor isn't us and they use our strat and we have found a score 1 bouquet
            if i != self.suitor_id and self.has_our_strat[i] == 1:

                recipient_visited.add(i)

                flowers_sent = self.score_1_bouquet[i]

                colors_count = [0]*6
                for f,j in flowers_sent.items():
                    colors_count[f.color.value] += j

                rem_flowers_list = flatten_counter(remaining_flowers)
                # essentially choose a random bouquet of same color count
                random.shuffle(rem_flowers_list)

                chosen_flower_counts = dict()
                for f in rem_flowers_list:
                    if colors_count[f.color.value] > 0:
                        if f not in chosen_flower_counts:
                            chosen_flower_counts[f] = 0
                        chosen_flower_counts[f] += 1
                        colors_count[f.color.value] -= 1

                # couldn't color copy
                if np.any(colors_count):
                    chosen_flower_counts = dict()

                for f in chosen_flower_counts:
                    remaining_flowers[f] -= 1

                recipient_chosen_flowers[i] = self.suitor_id, i, Bouquet(chosen_flower_counts)



        for recipient_id, score in self.recipients_all_score:
            if len(recipient_visited) == self.num_suitors - 1:
                break

            if recipient_id in recipient_visited:
                continue

            # if the score is too low, let's stop using the flowers that we remember
            # TODO: what's the exact score that we stop using the flowers that we remember
            if score < 0.5:
                break

            flowers = self.bouquet_history_score[recipient_id][score]
            exist_flower = True
            current_remaining_flowers = remaining_flowers.copy()
            for one_flower, count in flowers.items():
                if one_flower not in current_remaining_flowers or current_remaining_flowers[one_flower] < count:
                    exist_flower = False
                    break
                current_remaining_flowers[one_flower] -= count

            if exist_flower:
                remaining_flowers = current_remaining_flowers
                recipient_visited.add(recipient_id)
                recipient_chosen_flowers[recipient_id] = self.suitor_id, recipient_id, Bouquet(flowers)

        # if we cannot find flowers to all players, we will randomly assign the flower
        if len(recipient_visited) != self.num_suitors - 1:
            for recipient_id in recipient_ids:
                if recipient_id not in recipient_visited:
                    recipient_visited.add(recipient_id)
                    recipient_chosen_flowers[recipient_id] = self._prepare_bouquet(remaining_flowers, recipient_id)

        flower_list = []
        for recipient_id in recipient_ids:
            flower_list.append(recipient_chosen_flowers[recipient_id])
        return flower_list

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
        self.current_day += 1
        all_ids = np.arange(self.num_suitors)
        remaining_flowers = flower_counts.copy()
        recipient_ids = all_ids[all_ids != self.suitor_id]
        # if it is the final day, we should hand out the flowers that have the best score and we remember so far
        if self.current_day == self.days:
            return self.prepare_for_marry_day(remaining_flowers, recipient_ids)

        result = []
        suitors = set()
        for i in range(self.num_suitors):
            # suitor isn't us and unknown is they use our strat and we have found a score 1 bouquet
            if i != self.suitor_id and self.has_our_strat[i] == 0 and self.score_1_bouquet[i] != None:
                result.append(self._prepare_bouquet(remaining_flowers, i))
                suitors.add(i)

        for i in range(self.num_suitors):
            if i != self.suitor_id and i not in suitors:
                result.append(self._prepare_bouquet(remaining_flowers, i))
        # result = list(map(lambda recipient_id: self._prepare_bouquet(remaining_flowers, recipient_id), recipient_ids))
        return result

    def zero_score_bouquet(self):
        """
        :return: a Bouquet for which your scoring function will return 0
        """
        return Bouquet(dict())

    def one_score_bouquet(self):
        """
        :return: a Bouquet for which your scoring function will return 1
        """
        # find the first flower in self.score_one_flowers_for_us
        count = 0
        flowers = dict()
        flower_color_map = {"W": FlowerColors.White, "Y": FlowerColors.Yellow, "R": FlowerColors.Red,
                            "P": FlowerColors.Purple, "O": FlowerColors.Orange, "B": FlowerColors.Blue}
        for key, value in self.score_one_flowers_for_us.items():
            count += 1
            first_choice = value[0]
            for index in range(key):
                flower = get_random_flower()
                flower.color = flower_color_map[first_choice[index]]
                if flower not in flowers:
                    flowers[flower] = 0
                flowers[flower] += 1

            if count == 1:
                break

        return Bouquet(flowers)

    # TODO: maybe consider types in the future
    def score_types(self, types: Dict[FlowerTypes, int]):
        """
        :param types: dictionary of flower types and their associated counts in the bouquet
        :return: A score representing preference of the flower types in the bouquet
        """
        return 0

    def score_colors(self, colors: Dict[FlowerColors, int]):
        """
        :param colors: dictionary of flower colors and their associated counts in the bouquet
        :return: A score representing preference of the flower colors in the bouquet
        """
        if len(colors) == 0:
            return 0

        count_colors = []
        for color in flatten_counter(colors):
            count_colors.append(self.color_map[color.value])
        count_colors.sort()
        length_of_flowers = len(count_colors)
        if length_of_flowers not in self.score_one_flowers_for_us:
            return 0

        if tuple(count_colors) in self.score_one_flowers_for_us[length_of_flowers]:
            return 1

        return 0

    # TODO: maybe consider sizes in the future
    def score_sizes(self, sizes: Dict[FlowerSizes, int]):
        """
        :param sizes: dictionary of flower sizes and their associated counts in the bouquet
        :return: A score representing preference of the flower sizes in the bouquet
        """
        return 0

    # put feedback in self.bouquet_history_score
    def receive_feedback(self, feedback):
        """
        :param feedback: for each player, containing rank and score
        :return: nothing
        """
        for recipient_id in range(len(feedback)):
            if recipient_id == self.suitor_id:
                continue
            rank, score, _ = feedback[recipient_id]
            flower_sent = self.bouquet_history[recipient_id][-1]
            # flower_tuple = []
            # for flower, count in flower_sent.items():
            #     flower_tuple.append((flower, count))

            self.bouquet_history_score[recipient_id][score] = flower_sent
            self.recipients_largest_score_[recipient_id] = max(self.recipients_largest_score_[recipient_id], score)
            self.recipients_all_score.append((recipient_id, score))



            # unknown if they have our strat and I sent a non-empty bouquet
            if self.has_our_strat[recipient_id] == 0 and flower_sent:
                                             
                # no bouquet saved yet
                if self.score_1_bouquet[recipient_id] == None:
                    if score == 1:
                        self.score_1_bouquet[recipient_id] = dict(flower_sent)
                        self.our_strat_count[recipient_id] -= 1
                else:
                    if score == 1:
                        self.our_strat_count[recipient_id] -= 1

                        if self.our_strat_count[recipient_id] == 0:
                            self.has_our_strat[recipient_id] = 1
                    else:
                        self.has_our_strat[recipient_id] = -1


''' usage of BouquetSimulator:
bouquet = BouquetSimulator(9) -> number of players
times = 10000 # set up number of rounds -> 10000 is 0.2 second for 9 players
bouquet.simulate_give_flowers(times)
'''


class BouquetSimulator:
    def __init__(self, num_players: int):
        self.num_players = num_players
        self.total_flowers = 6 * (self.num_players - 1)
        self.max_bouquet_size = MAX_BOUQUET_SIZE
        self.probability = {i: 0 for i in range(self.max_bouquet_size + 1)}

    def simulate_give_flowers(self, times: int):
        for _ in range(times):
            remain = self.total_flowers
            count = self.num_players - 1
            while remain > 0 and count > 0:
                size = int(np.random.randint(0, min(self.max_bouquet_size, remain) + 1))
                count -= 1
                remain -= size
                self.probability[size] += 1

            if count > 0:
                self.probability[0] += count

        for key, value in self.probability.items():
            self.probability[key] = value / (times * (self.num_players - 1))


''' usage of FlowerColorSimulator:
flowerColor = FlowerColorSimulator(range(0, 13)) -> all possibilities from number of flowers = 0 to 12
times = 10000 # set up number of rounds -> 10000 is 12 seconds for range(0, 13)
flowerColor.simulate_possibilities(times, people.probability)
key is the bouquet size, value is a dict, key is the tuple, means the flower arrangement, value is the probability of that arrangemnet
for example, flowerColor.probability[1][('B',)] means number of flowers = 1, and I only want one blue
value is the probability of that flower arrangement in considering of number of players
'''


class FlowerColorSimulator:
    def __init__(self, nums_of_flowers: List[int]):
        self.possible_flowers = get_all_possible_flowers()
        self.num_flowers_to_sample = nums_of_flowers

        self.color_map = {0: "W", 1: "Y", 2: "R", 3: "P", 4: "O", 5: "B"}
        colors = sorted(list(self.color_map.values()))

        """
        Exp: Size of bouquet = 6 
        combination = {W, W, Y, Y, R, R}
        
        Counts:
            W: 2
            Y: 2
            R: 2
        
        Total number of combinations: 6!/(2! 2! 2!)
        
        Probability: 6!/(2! 2! 2!) * (1/6)^6
        
        In general:
        
        Probability = (n!/(a!b!c!...z!)) * (1/6)^n where n is the size of bouquet
        
        """

        self.probability = defaultdict(dict)
        for num in self.num_flowers_to_sample:
            for c in combinations_with_replacement(colors, num):
                counts = Counter(c)
                unique_colors = counts.keys()
                denominator = 1
                for color in unique_colors:
                    denominator *= factorial(counts[color])
                self.probability[num][c] = pow(1/6, num) * factorial(num)/denominator if num != 0 else 0.1152 # Hard code the probability for empty bouquet
                # self.probability[num][c] = 0

    def simulate_possibilities(self, times: int, bouquet_probability: Dict):
        equal_probability = False
        if len(bouquet_probability) == 0:
            equal_probability = True

        # for num in self.num_flowers_to_sample:
        #     for _ in range(times):
        #         flowers_for_round = sample_n_random_flowers(self.possible_flowers, num)
        #         flower_list = []
        #         for flower, value in flowers_for_round.items():
        #             flower_list.extend([self.color_map[flower.color.value]] * value)
        #
        #         flower_list.sort()
        #         self.probability[num][tuple(flower_list)] += 1
        #
        # for key in self.probability.keys():
        #     for value, count in self.probability[key].items():
        #         self.probability[key][value] = count / times
        #         if not equal_probability:
        #             self.probability[key][value] *= bouquet_probability[key]

        if not equal_probability:
            all_probability = 0
            for key in self.probability.keys():
                all_probability += sum(self.probability[key].values())

            for key in self.probability.keys():
                for value, count in self.probability[key].items():
                    self.probability[key][value] = count / all_probability

        # self.probability = dict(sorted(self.probability.items(), key=lambda item: -item[1]))


