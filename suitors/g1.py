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
        flowerColor.simulate_possibilities(bouquet.probability)
        self.flower_probability_table = flowerColor.probability

        self.color_map = {0: "W", 1: "Y", 2: "R", 3: "P", 4: "O", 5: "B"}

        if days == 1:
            self.percentage = 1 / (num_suitors - 1)
        else:
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

        self.random_guess_times = 10

        # test others hypothesis: whether they separately the types/sizes/colors,
        # we test the hypothesis for the next 3 days
        self.guess_other_threhold = 30
        self.last_days_guess_others = self.days // 3
        self.guess_other_times = 2
        self.test_other_hypothesis_days = [self.guess_other_times] * num_suitors
        self.test_other_hypothesis = [0] * num_suitors
        self.score_other_smallest_bouquet = [None] * num_suitors
        self.score_other_smallest_bouquet_score = [0] * num_suitors

        self.average_score = [0] * num_suitors

    # if the remaining flowers can construct our strategy colors, return True, else return False
    def can_construct_our_strategy(self, remaining_flowers: Dict[Flower, int], want_flowers: Dict[Flower, int]) -> bool:
        remain_colors_map = defaultdict(int)
        for flower, count in remaining_flowers.items():
            remain_colors_map[flower.color] += count

        want_colors_map = defaultdict(int)
        for flower, count in want_flowers.items():
            want_colors_map[flower.color] += count

        for color, count in want_colors_map.items():
            if color not in remain_colors_map:
                return False

            if count > remain_colors_map[color]:
                return False

        return True

    # if the remaining flowers can construct other strategy colors, return True, else return False
    def can_construct_other_strategy(self, remaining_flowers: Dict[Flower, int],
                                     want_flowers: Dict[Flower, int]) -> bool:
        remain_colors_map = defaultdict(int)
        remain_types_map = defaultdict(int)
        remain_sizes_map = defaultdict(int)
        for flower, count in remaining_flowers.items():
            remain_colors_map[flower.color] += count
            remain_types_map[flower.type] += count
            remain_sizes_map[flower.size] += count

        want_colors_map = defaultdict(int)
        want_types_map = defaultdict(int)
        want_sizes_map = defaultdict(int)
        for flower, count in want_flowers.items():
            want_colors_map[flower.color] += count
            want_types_map[flower.type] += count
            want_sizes_map[flower.size] += count

        for color, count in want_colors_map.items():
            if color not in remain_colors_map:
                return False

            if count > remain_colors_map[color]:
                return False

        for flower_type, count in want_types_map.items():
            if flower_type not in remain_types_map:
                return False

            if count > remain_types_map[flower_type]:
                return False

        for flower_size, count in want_sizes_map.items():
            if flower_size not in remain_sizes_map:
                return False

            if count > remain_sizes_map[flower_size]:
                return False

        return True

    # choose one score bouquet randomly from the flowerColor.probability until reached the percentage
    def choose_one_score_bouquet_for_ourselves(self, probability_table: Dict):
        diff = self.percentage * pow(10, -5)
        remain_probability = self.percentage

        probability_table_list = defaultdict(list)
        for key in probability_table.keys():
            probability_table_list[key] = list(probability_table[key].items())

        while remain_probability > 0:
            if remain_probability < diff:
                break
            # we don't consider the empty flowers to be score 1
            size = int(np.random.randint(1, MAX_BOUQUET_SIZE + 1))
            flower, probability = random.choice(probability_table_list[size])
            if probability <= remain_probability:
                remain_probability -= probability
                self.score_one_flowers_for_us[size].append(flower)

    # we only care about the colors, but we can choose different types/sizes
    def construct_our_strategy_flowers(self, recipient_id, remaining_flowers):
        flowers_sent = self.score_1_bouquet[recipient_id]

        if not self.can_construct_our_strategy(remaining_flowers, flowers_sent):
            chosen_flower_counts = dict()
        else:
            chosen_flower_counts = dict()
            rem_flowers_list = []
            for flower, count in remaining_flowers.items():
                rem_flowers_list.extend([flower] * count)

            can_find = False
            for _ in range(self.random_guess_times):
                random.shuffle(rem_flowers_list)

                types_count = [0] * 4
                colors_count = [0] * 6
                sizes_count = [0] * 3

                for flower, count in flowers_sent.items():
                    types_count[flower.type.value] += count
                    colors_count[flower.color.value] += count
                    sizes_count[flower.size.value] += count

                chosen_flower_counts = defaultdict(int)
                for f in rem_flowers_list:
                    if not np.any(colors_count) and chosen_flower_counts != flowers_sent:
                        can_find = True
                        break

                    if colors_count[f.color.value] > 0:
                        chosen_flower_counts[f] += 1
                        colors_count[f.color.value] -= 1

                if can_find:
                    break

            if not can_find:
                chosen_flower_counts = dict()

        return chosen_flower_counts

    def construct_other_strategy_flowers(self, recipient_id, remaining_flowers, flowers_sent):
        if not self.can_construct_other_strategy(remaining_flowers, flowers_sent):
            chosen_flower_counts = dict()
        else:
            rem_flowers_list = flatten_counter(remaining_flowers)
            can_find = False
            chosen_flower_counts = dict()
            for _ in range(self.random_guess_times):
                types_count = [0] * 4
                colors_count = [0] * 6
                sizes_count = [0] * 3

                for flower, count in flowers_sent.items():
                    types_count[flower.type.value] += count
                    colors_count[flower.color.value] += count
                    sizes_count[flower.size.value] += count

                random.shuffle(rem_flowers_list)

                chosen_flower_counts = defaultdict(int)
                for flower in rem_flowers_list:
                    if not np.any(colors_count) and not np.any(types_count) and not np.any(sizes_count) and \
                            chosen_flower_counts != flowers_sent:
                        can_find = True
                        break

                    if types_count[flower.type.value] > 0 and colors_count[flower.color.value] > 0 and sizes_count[flower.size.value] > 0:
                        chosen_flower_counts[flower] += 1
                        types_count[flower.type.value] -= 1
                        colors_count[flower.color.value] -= 1
                        sizes_count[flower.size.value] -= 1

                if can_find:
                    break

            if not can_find or sum(chosen_flower_counts.values()) > MAX_BOUQUET_SIZE:
                chosen_flower_counts = dict()

        return chosen_flower_counts

    # for each player, randomly guess what they want if we haven't got score 1 for their group
    def _prepare_bouquet(self, remaining_flowers, recipient_id, isRandom):
        if not isRandom and self.has_our_strat[recipient_id] == 0 and self.score_1_bouquet[recipient_id] is not None:
            chosen_flower_counts = self.construct_our_strategy_flowers(recipient_id, remaining_flowers)
            for k, v in chosen_flower_counts.items():
                remaining_flowers[k] -= v

        elif not isRandom and self.days >= self.guess_other_threhold and self.has_our_strat[recipient_id] == -1 \
                and self.current_day >= self.days - self.last_days_guess_others \
                and self.test_other_hypothesis[recipient_id] == 0 and self.score_other_smallest_bouquet is not None:
            chosen_flower_counts = self.construct_other_strategy_flowers(recipient_id, remaining_flowers,
                                                                         self.score_other_smallest_bouquet[recipient_id])
            for k, v in chosen_flower_counts.items():
                remaining_flowers[k] -= v

        elif not isRandom and int(self.recipients_largest_score_[recipient_id]) == 1:
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

    def match_exact_flowers(self, remaining_flowers, recipient_id, score):
        flowers = self.bouquet_history_score[recipient_id][score]
        exist_flower = True
        current_remaining_flowers = remaining_flowers.copy()
        for one_flower, count in flowers.items():
            if one_flower not in current_remaining_flowers or current_remaining_flowers[one_flower] < count:
                exist_flower = False
                break
            current_remaining_flowers[one_flower] -= count

        if exist_flower:
            return flowers
        else:
            return dict()

    # in the final round, choose the flowers given to each recipient
    # can use self.score_one_recipients, sort the dict by values,
    # and give flowers according to self.bouquet_history_score
    # choose check if we got that arrangement of flowers
    def prepare_for_marry_day(self, remaining_flowers: Dict[Flower, int], recipient_ids: List[int]):
        recipient_visited = set()
        self.recipients_all_score.sort(key=lambda x: -x[1])
        recipient_chosen_flowers = dict()

        for recipient_id in recipient_ids:
            # suitor isn't us and they use our strategy and we have found a score 1 bouquet
            if recipient_id != self.suitor_id and self.has_our_strat[recipient_id] == 1:
                chosen_flower_counts = self.construct_our_strategy_flowers(recipient_id, remaining_flowers)

                recipient_visited.add(recipient_id)
                # if cannot find flowers, try to match the exact flowers
                if not chosen_flower_counts:
                    chosen_flower_counts = self.match_exact_flowers(remaining_flowers, recipient_id, 1)

                recipient_chosen_flowers[recipient_id] = self.suitor_id, recipient_id, Bouquet(chosen_flower_counts)
                for k, v in chosen_flower_counts.items():
                    remaining_flowers[k] -= v

        for recipient_id in recipient_ids:
            if len(recipient_visited) == self.num_suitors - 1:
                break

            if recipient_id in recipient_visited:
                continue

            # suitor isn't us and they use other hypothesis and we have found a score 1 bouquet
            if recipient_id != self.suitor_id and self.days >= self.guess_other_threhold and \
                    self.test_other_hypothesis[recipient_id] == 1:
                chosen_flower_counts = self.construct_other_strategy_flowers(recipient_id, remaining_flowers,
                                                                             self.score_1_bouquet[recipient_id])
                if not chosen_flower_counts:
                    chosen_flower_counts = self.match_exact_flowers(remaining_flowers, recipient_id, 1)

                if chosen_flower_counts:
                    recipient_visited.add(recipient_id)
                    recipient_chosen_flowers[recipient_id] = self.suitor_id, recipient_id, Bouquet(chosen_flower_counts)
                    for k, v in chosen_flower_counts.items():
                        remaining_flowers[k] -= v

        for recipient_id, score in self.recipients_all_score:
            if len(recipient_visited) == self.num_suitors - 1:
                break

            if recipient_id in recipient_visited:
                continue

            # if the score is too low, let's stop using the flowers that we remember
            if score < self.average_score[recipient_id]:
                continue

            flowers = self.bouquet_history_score[recipient_id][score]
            if self.test_other_hypothesis[recipient_id] == 1:
                chosen_flower_counts = self.construct_other_strategy_flowers(recipient_id, remaining_flowers, flowers)
                if not chosen_flower_counts:
                    chosen_flower_counts = self.match_exact_flowers(remaining_flowers, recipient_id, score)

                if chosen_flower_counts:
                    recipient_visited.add(recipient_id)
                    recipient_chosen_flowers[recipient_id] = self.suitor_id, recipient_id, Bouquet(chosen_flower_counts)
                    for k, v in chosen_flower_counts.items():
                        remaining_flowers[k] -= v

            else:
                chosen_flower_counts = self.match_exact_flowers(remaining_flowers, recipient_id, score)
                if chosen_flower_counts:
                    recipient_visited.add(recipient_id)
                    recipient_chosen_flowers[recipient_id] = self.suitor_id, recipient_id, Bouquet(chosen_flower_counts)
                    for k, v in chosen_flower_counts.items():
                        remaining_flowers[k] -= v

        # if we cannot find exact flowers to all players, we will randomly assign the flower
        if len(recipient_visited) != self.num_suitors - 1:
            for recipient_id in recipient_ids:
                if recipient_id not in recipient_visited:
                    recipient_visited.add(recipient_id)
                    recipient_chosen_flowers[recipient_id] = self._prepare_bouquet(remaining_flowers, recipient_id,
                                                                                   True)

        flower_list = []
        for recipient_id in recipient_ids:
            flower_list.append(recipient_chosen_flowers[recipient_id])
        return flower_list

    def calculate_average_score(self):
        sum_average_score = [0] * self.num_suitors
        count_score = [0] * self.num_suitors

        for recipient_id, score in self.recipients_all_score:
            sum_average_score[recipient_id] += score
            count_score[recipient_id] += 1

        for recipient_id in range(self.num_suitors):
            if recipient_id != self.suitor_id and count_score[recipient_id] != 0:
                self.average_score[recipient_id] = sum_average_score[recipient_id] / count_score[recipient_id]

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
            self.calculate_average_score()
            return self.prepare_for_marry_day(remaining_flowers, recipient_ids)

        result = []
        suitors_visited = set()
        for i in range(self.num_suitors):
            # suitor isn't us and unknown is they use our strat and we have found a score 1 bouquet
            if i != self.suitor_id and self.has_our_strat[i] == 0 and self.score_1_bouquet[i] is not None:
                result.append(self._prepare_bouquet(remaining_flowers, i, False))
                suitors_visited.add(i)

        if self.days >= self.guess_other_threhold and self.current_day >= self.days - self.last_days_guess_others:
            for i in range(self.num_suitors):
                # suitor isn't us and unknown is they use our strat and we have found a score 1 bouquet
                if i != self.suitor_id and self.has_our_strat[i] == -1 and self.score_other_smallest_bouquet is not None \
                        and i not in suitors_visited and self.test_other_hypothesis[i] == 0:
                    result.append(self._prepare_bouquet(remaining_flowers, i, False))
                    suitors_visited.add(i)

        for i in range(self.num_suitors):
            if i != self.suitor_id and i not in suitors_visited:
                result.append(self._prepare_bouquet(remaining_flowers, i, False))
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
        for color, count in colors.items():
            count_colors.extend([self.color_map[color.value]] * count)
        count_colors.sort()
        length_of_flowers = len(count_colors)
        if length_of_flowers not in self.score_one_flowers_for_us:
            return 0

        if tuple(count_colors) in self.score_one_flowers_for_us[length_of_flowers]:
            return 1

        return 0

    def score_sizes(self, sizes: Dict[FlowerSizes, int]):
        """
        :param sizes: dictionary of flower sizes and their associated counts in the bouquet
        :return: A score representing preference of the flower sizes in the bouquet
        """
        return 0

    def compare_colors(self, flower_counts_1: Dict[Flower, int], flower_counts_2: Dict[Flower, int]) -> bool:
        color_map1 = defaultdict(int)
        for flower, value in flower_counts_1.items():
            color_map1[flower.color] += value

        color_map2 = defaultdict(int)
        for flower, value in flower_counts_2.items():
            color_map2[flower.color] += value

        return color_map1 == color_map2

    def get_three_features(self, flowers):
        types_count = defaultdict(int)
        colors_count = defaultdict(int)
        sizes_count = defaultdict(int)

        for flower, count in flowers.items():
            types_count[flower.type] += count
            colors_count[flower.color] += count
            sizes_count[flower.size] += count

        return colors_count, types_count, sizes_count

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

            if not flower_sent:
                continue

            if int(score) == 1:
                score = int(score)

            if score > 0 and score in self.bouquet_history_score[recipient_id]:
                # if the same score exists, we always want to choose the fewer flowers combinations
                if int(sum(self.bouquet_history_score[recipient_id][score].values())) > int(sum(flower_sent.values())):
                    self.bouquet_history_score[recipient_id][score] = flower_sent
            else:
                self.bouquet_history_score[recipient_id][score] = flower_sent
                self.recipients_largest_score_[recipient_id] = max(self.recipients_largest_score_[recipient_id], score)

            self.recipients_all_score.append((recipient_id, score))

            if self.current_day == self.days:
                continue

            # unknown if they have our strat and I sent a non-empty bouquet
            if self.has_our_strat[recipient_id] == 0:

                # no bouquet saved yet
                if self.score_1_bouquet[recipient_id] is None:
                    if int(score) == 1:
                        self.score_1_bouquet[recipient_id] = dict(flower_sent)
                        self.our_strat_count[recipient_id] -= 1
                else:
                    # test if it is our strategy
                    if int(score) == 1:
                        self.our_strat_count[recipient_id] -= 1

                        if self.our_strat_count[recipient_id] == 0:
                            self.has_our_strat[recipient_id] = 1
                    else:
                        self.has_our_strat[recipient_id] = -1

            if score > 0:
                if score == 1 and sum(flower_sent.values()) <= 2:
                    self.score_other_smallest_bouquet[recipient_id] = dict(flower_sent)
                    self.score_other_smallest_bouquet_score[recipient_id] = score
                else:
                    if self.score_other_smallest_bouquet[recipient_id] is None and sum(flower_sent.values()) > 2:
                        self.score_other_smallest_bouquet[recipient_id] = dict(flower_sent)
                        self.score_other_smallest_bouquet_score[recipient_id] = score
                    elif 2 < sum(flower_sent.values()) < sum(self.score_other_smallest_bouquet[recipient_id].values()):
                        self.score_other_smallest_bouquet[recipient_id] = dict(flower_sent)
                        self.score_other_smallest_bouquet_score[recipient_id] = score

            # test if it is other hypothesis
            if self.days >= self.guess_other_threhold and self.current_day >= self.days - self.last_days_guess_others \
                    and self.has_our_strat[recipient_id] == -1 and self.test_other_hypothesis[recipient_id] == 0 \
                    and self.recipients_largest_score_[recipient_id] == 1:
                diff = pow(10, -3)
                if self.test_other_hypothesis_days[recipient_id] == self.guess_other_times:
                    self.score_1_bouquet[recipient_id] = dict(self.bouquet_history_score[recipient_id][1])

                if abs(score - self.score_other_smallest_bouquet_score[recipient_id]) <= diff:
                    self.test_other_hypothesis_days[recipient_id] -= 1

                    if self.test_other_hypothesis_days[recipient_id] == 0:
                        self.test_other_hypothesis[recipient_id] = 1
                else:
                    if self.test_other_hypothesis_days[recipient_id] != self.guess_other_times:
                        self.test_other_hypothesis[recipient_id] = -1


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
                self.probability[num][c] = pow(1 / 6, num) * factorial(
                    num) / denominator if num != 0 else 0.1152  # Hard code the probability for empty bouquet
                # self.probability[num][c] = 0

    def simulate_possibilities(self, bouquet_probability: Dict):
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
