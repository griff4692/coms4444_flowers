from typing import Dict

from flowers import Bouquet, Flower
from suitors.base import BaseSuitor


class Suitor(BaseSuitor):
    def __init__(self, days: int, num_suitors: int, suitor_id: int):
        """
        :param days: number of days of courtship
        :param num_suitors: number of suitors, including yourself
        :param suitor_id: unique id of your suitor in range(num_suitors)
        """
        super().__init__(days, num_suitors, suitor_id, name='g7')

    def prepare_bouquets(self, flower_counts: Dict[Flower, int]):
        pass

    def zero_score_bouquet(self):
        return Bouquet({})

    def one_score_bouquet(self):
        pass

    def score_type(self, bouquet: Bouquet):
        """
        :param bouquet: an arrangement of flowers
        :return: A score representing preference of the flower types in the bouquet
        """
        pass

    def score_color(self, bouquet: Bouquet):
        """
        :param bouquet: an arrangement of flowers
        :return: A score representing preference of the flower colors in the bouquet
        """
        pass

    def score_size(self, bouquet: Bouquet):
        """
        :param bouquet: an arrangement of flowers
        :return: A score representing preference of the flower sizes in the bouquet
        """
        pass

    def receive_feedback(self, feedback):
        """
        :param feedback:
        :return: nothing
        """
        # Can over-write this
        self.feedback.append(feedback)
