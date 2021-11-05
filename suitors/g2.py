from typing import Dict

from flowers import Bouquet, Flower, FlowerSizes, FlowerColors, FlowerTypes
from suitors.base import BaseSuitor

import numpy as np
from copy import deepcopy
import random

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
        super().__init__(days, num_suitors, suitor_id, name='g2')

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
        bouquets = []

        copy_flower_counts = {}
        for key,value in flower_counts.items():
           copy_flower_counts[str(key)] = value 
        
        flowers = [(key,value) for key,value in flower_counts.items()]
        random.shuffle(flowers)

        for r_id in recipient_ids:
            bouquet = {}

            types = []
            colors = []
            sizes = []

            count = 4

            for item in flowers:
                key,value = item
                
                if copy_flower_counts[str(key)] == 0:
                    continue

                if count == 0:
                    break

                count -= 1
                
                should_add = False
                if key.type not in types:
                    should_add = True
                if should_add or key.color not in colors:
                    should_add = True
                if should_add or key.size not in sizes:
                    should_add = True

                if should_add:
                    types.append(key.type)
                    colors.append(key.color)
                    sizes.append(key.size)
                    bouquet[key] = 1
                    copy_flower_counts[str(key)] -= 1
                    # print(key)
                
            bouquets.append((self.suitor_id, r_id, Bouquet(bouquet)))

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

        flower_1 = Flower(
            size=FlowerSizes.Small,
            color=FlowerColors.White,
            type=FlowerTypes.Rose
        )
        flower_2 = Flower(
            size=FlowerSizes.Small,
            color=FlowerColors.Yellow,
            type=FlowerTypes.Rose
        )
        flower_3 = Flower(
            size=FlowerSizes.Medium,
            color=FlowerColors.Red,
            type=FlowerTypes.Chrysanthemum
        )
        flower_4 = Flower(
            size=FlowerSizes.Medium,
            color=FlowerColors.Purple,
            type=FlowerTypes.Tulip
        )
        flower_5 = Flower(
            size=FlowerSizes.Large,
            color=FlowerColors.Orange,
            type=FlowerTypes.Begonia
        )
        flower_6 = Flower(
            size=FlowerSizes.Large,
            color=FlowerColors.Blue,
            type=FlowerTypes.Begonia
        )

        return Bouquet({
            flower_1: 1, 
            flower_2: 1,
            flower_3: 1,
            flower_4: 1,
            flower_5: 1,
            flower_6: 1,
        })

    def score_types(self, types: Dict[FlowerTypes, int]):
        """
        :param types: dictionary of flower types and their associated counts in the bouquet
        :return: A score representing preference of the flower types in the bouquet
        """

        type_scores = {
            FlowerTypes.Rose: 1/(3*4),
            FlowerTypes.Chrysanthemum: 0.5/(3*4),
            FlowerTypes.Tulip: 1/(3*4),
            FlowerTypes.Begonia: 1.5/(3*4),
        }

        num_types = 0
        for key,value in types.items():
            if value > 0:
                num_types += type_scores[key]
        return num_types
        

    def score_colors(self, colors: Dict[FlowerColors, int]):
        """
        :param colors: dictionary of flower colors and their associated counts in the bouquet
        :return: A score representing preference of the flower colors in the bouquet
        """
        num_types = 0
        for _,value in colors.items():
            if value > 0:
                num_types += (1 / (3 * 6))
        return num_types

    def score_sizes(self, sizes: Dict[FlowerSizes, int]):
        """
        :param sizes: dictionary of flower sizes and their associated counts in the bouquet
        :return: A score representing preference of the flower sizes in the bouquet
        """
        num_types = 0
        for _,value in sizes.items():
            if value > 0:
                num_types += (1 / (3 * 3))
        return num_types

    def receive_feedback(self, feedback):
        """
        :param feedback:
        :return: nothing
        """
        self.feedback.append(feedback)
