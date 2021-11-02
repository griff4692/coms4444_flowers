from typing import Dict, List

from flowers import Bouquet, Flower, FlowerSizes, FlowerColors, FlowerTypes, MAX_BOUQUET_SIZE, \
    get_all_possible_flowers, sample_n_random_flowers, get_random_flower
from suitors.base import BaseSuitor
import numpy as np
from collections import defaultdict, Counter
from itertools import combinations_with_replacement
from utils import flatten_counter


class Suitor(BaseSuitor):
    def __init__(self, days: int, num_suitors: int, suitor_id: int):
        """
        :param days: number of days of courtship
        :param num_suitors: number of suitors, including yourself
        :param suitor_id: unique id of your suitor in range(num_suitors)
        """
        super().__init__(days, num_suitors, suitor_id, name='g1')
        # step 1: get the probability for people range from 0 - num_suitors
        people = PeopleSimulator(num_suitors)
        people.simulate_give_flowers(10000)
        # step 2: get the probability for flower colors range from 0 - MAX_BOUQUET_SIZE(12)
        flowerColor = FlowerColorSimulator(list(range(MAX_BOUQUET_SIZE + 1)))
        flowerColor.simulate_possibilities(5000, people.probability)
        # self.flower_probability_table = flowerColor.probability

        self.color_map = {0: "W", 1: "Y", 2: "R", 3: "P", 4: "O", 5: "B"}

        # TODO: define the percentage, the relationship between percentage, days and num_suitors
        self.percentage = 0.2
        # self.percentage = 2 / (days - 1) * (num_suitors - 1)

        # step 3: choose our one score flowers from the color probability table
        # score_one_flowers_for_us: the score of our choices of colors to be 1.0
        # key: length of flowers, value: the combination of the colors
        # should notice the value is a tuple, like('B', 'R') means 1 blue 1 red,
        # and also should notice that tuple is sorted,
        # so when to find whether exists a flower, sort the colors first then transfer it to tuple
        self.score_one_flowers_for_us = defaultdict(list)
        self.choose_one_score_bouquet_for_ourselves(flowerColor.probability)

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
        self.score_one_recipients = {i: 0 for i in range(num_suitors) if i != suitor_id}

    # choose one score bouquet from the flowerColor.probability
    def choose_one_score_bouquet_for_ourselves(self, probability_table: Dict):
        remain_probability = self.percentage
        # TODO: how many types of colors arrangement that we plan to choose
        # expectations of flowers chosen, I just set it as 6 because for now I don't want choose too many types
        max_choose_flowers = 6
        while remain_probability > 0 and max_choose_flowers > 0:
            for key, value in probability_table.items():
                # we don't consider the empty flowers
                if len(key) == 0:
                    continue
                if (len(key) not in self.score_one_flowers_for_us or key not in self.score_one_flowers_for_us[len(key)]) \
                        and (value <= remain_probability or abs(value - remain_probability) < 0.01):
                    self.score_one_flowers_for_us[len(key)].append(key)
                    remain_probability -= value
                    max_choose_flowers -= 1
                    break

    # for each player, randomly guess what they want if we haven't got score 1 for their group
    def _prepare_bouquet(self, remaining_flowers, recipient_id):
        # if we got 1.0 score from that group before, skip it
        if int(self.score_one_recipients[recipient_id]) == 1:
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

    # TODO: in the final round, choose the flowers given to each recipient
    # can use self.score_one_recipients, sort the dict by values,
    # and give flowers according to self.bouquet_history_score
    # choose check if we got that arrangement of flowers
    # Question: what if we happens to get three flowers but lose one,
    # should we skip that arrangement or to find the nearest flower to meet the needs
    def prepare_for_marry_day(self):
        pass

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
        # if it is the final day, we should hand out the flowers that have the best score and we remember so far
        # if self.current_day == self.days:
        #     return self.prepare_for_marry_day()

        all_ids = np.arange(self.num_suitors)
        recipient_ids = all_ids[all_ids != self.suitor_id]
        remaining_flowers = flower_counts.copy()
        return list(map(lambda recipient_id: self._prepare_bouquet(remaining_flowers, recipient_id), recipient_ids))

    def zero_score_bouquet(self):
        """
        :return: a Bouquet for which your scoring function will return 0
        """
        return Bouquet(dict())

    def one_score_bouquet(self):
        """
        :return: a Bouquet for which your scoring function will return 1
        """
        # fine the first flower in self.score_one_flowers_for_us
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
            rank, score = feedback[recipient_id]
            flower_sent = self.bouquet_history[recipient_id][-1]
            flower_tuple = []
            for flower, count in flower_sent.items():
                flower_tuple.append((flower, count))

            self.bouquet_history_score[recipient_id][score] = flower_tuple
            self.score_one_recipients[recipient_id] = max(self.score_one_recipients[recipient_id], score)


''' usage of PeopleSimulator:
people = People(9) -> number of players
times = 10000 # set up number of rounds -> 10000 is 0.2 second for 9 players
people.simulate_give_flowers(times)
'''


class PeopleSimulator:
    def __init__(self, players: int):
        self.people = players
        self.total_flowers = 6 * (self.people - 1)
        self.max_flower = MAX_BOUQUET_SIZE
        self.probability = {i: 0 for i in range(self.max_flower + 1)}

    def simulate_give_flowers(self, times: int):
        for _ in range(times):
            remain = self.total_flowers
            count = self.people - 1
            while remain > 0 and count > 0:
                size = int(np.random.randint(0, min(self.max_flower, remain) + 1))
                count -= 1
                remain -= size
                self.probability[size] += 1

            if count > 0:
                self.probability[0] += count

        for key, value in self.probability.items():
            self.probability[key] = value / (times * (self.people - 1))


''' usage of FlowerColorSimulator:
flowerColor = FlowerColorSimulator(range(1, 13)) -> all possibilities from number of flowers = 1 to 12
times = 10000 # set up number of rounds -> 10000 is 12 seconds for range(0, 13)
flowerColor.simulate_possibilities(times, people.probability)
get the table:flowerColor.probability from larger to smaller
key is the tuple, means the flower arrangement
for example, ('B',) means number of flowers = 1, and I only want one blue
value is the probability of that flower arrangement in considering of number of players
'''


class FlowerColorSimulator:
    def __init__(self, nums_of_flowers: List[int]):
        self.possible_flowers = get_all_possible_flowers()
        self.num_flowers_to_sample = nums_of_flowers

        self.color_map = {0: "W", 1: "Y", 2: "R", 3: "P", 4: "O", 5: "B"}
        colors = sorted(list(self.color_map.values()))

        self.probability = defaultdict(float)
        for num in self.num_flowers_to_sample:
            for c in combinations_with_replacement(colors, num):
                self.probability[c] = 0

    def simulate_possibilities(self, times: int, people_probability: Dict):
        equal_probability = False
        if len(people_probability) == 0:
            equal_probability = True

        for num in self.num_flowers_to_sample:
            for _ in range(times):
                flowers_for_round = sample_n_random_flowers(self.possible_flowers, num)
                flower_list = []
                for flower, value in flowers_for_round.items():
                    flower_list.extend([self.color_map[flower.color.value]] * value)

                flower_list.sort()
                self.probability[tuple(flower_list)] += 1

        for key, value in self.probability.items():
            self.probability[key] = value / times
            if not equal_probability:
                self.probability[key] *= people_probability[len(key)]

        if not equal_probability:
            all_probability = sum(self.probability.values())
            for key, value in self.probability.items():
                self.probability[key] = value / all_probability

        self.probability = dict(sorted(self.probability.items(), key=lambda item: -item[1]))
