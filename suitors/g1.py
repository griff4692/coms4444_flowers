from typing import Dict

from flowers import Bouquet, Flower, FlowerSizes, FlowerColors, FlowerTypes, MAX_BOUQUET_SIZE, \
    get_all_possible_flowers, sample_n_random_flowers
from suitors.base import BaseSuitor
import numpy as np
from collections import defaultdict
from itertools import combinations_with_replacement
from typing import List


class Suitor(BaseSuitor):
    def __init__(self, days: int, num_suitors: int, suitor_id: int):
        """
        :param days: number of days of courtship
        :param num_suitors: number of suitors, including yourself
        :param suitor_id: unique id of your suitor in range(num_suitors)
        """
        super().__init__(days, num_suitors, suitor_id, name='g1')

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
        pass

    def zero_score_bouquet(self):
        """
        :return: a Bouquet for which your scoring function will return 0
        """
        pass

    def one_score_bouquet(self):
        """
        :return: a Bouquet for which your scoring function will return 1
        """
        pass

    def score_types(self, types: Dict[FlowerTypes, int]):
        """
        :param types: dictionary of flower types and their associated counts in the bouquet
        :return: A score representing preference of the flower types in the bouquet
        """
        pass

    def score_colors(self, colors: Dict[FlowerColors, int]):
        """
        :param colors: dictionary of flower colors and their associated counts in the bouquet
        :return: A score representing preference of the flower colors in the bouquet
        """
        pass

    def score_sizes(self, sizes: Dict[FlowerSizes, int]):
        """
        :param sizes: dictionary of flower sizes and their associated counts in the bouquet
        :return: A score representing preference of the flower sizes in the bouquet
        """
        pass

    def receive_feedback(self, feedback):
        """
        :param feedback:
        :return: nothing
        """
        self.feedback.append(feedback)


''' usage of PeopleSimulator:
people = People(9) -> number of players
times = 10000 # set up number of rounds -> 10000 is 0.1 second for 9 players
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
flowerColor = FlowerColorSimulator(range(1, 13)) -> all posiblities from number of flowers = 1 to 12
times = 10000 # set up number of rounds -> 10000 is 10 seconds for range(1,13)
flowerColor.simulate_possibilities(times)
get the table:flowerColor.probability
key is the tuple, means the flower arrangement
for example, ('B',) means number of flowers = 1, and I only want one blue
value is the probability of that flower arrangement
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

    def simulate_possibilities(self, times: int):
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

        self.probability = dict(sorted(self.probability.items(), key=lambda item: -item[1]))
