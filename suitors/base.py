from abc import ABC, abstractmethod
from typing import Dict

from flowers import Bouquet, Flower, FlowerColors, FlowerSizes, FlowerTypes
from time_utils import break_after, prepare_empty_bouquets


class BaseSuitor(ABC):
    def __init__(self, days: int, num_suitors: int, suitor_id: int, name: str):
        """
        :param days: number of days of courtship
        :param num_suitors: number of suitors, including yourself
        :param suitor_id: unique id of your suitor for a particular game in range(num_suitors)
        :param name: string name of your team
        """
        self.days = days
        self.num_suitors = num_suitors
        self.suitor_id = suitor_id
        self.name = name

        # Record feedback.  BaseSuitor doesn\'t do anything with this information, but you should!
        self.feedback = []

    @abstractmethod
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

    @break_after(1, fallback_func=prepare_empty_bouquets)
    def prepare_bouquets_timed(self, flower_counts: Dict[Flower, int]):
        """
        Imposes 1 second time limit on self.prepare_bouqets.  Do not override this method
        """
        return self.prepare_bouquets(flower_counts)

    @break_after(10, fallback_func=prepare_empty_bouquets)
    def prepare_bouquets_timed_final_round(self, flower_counts: Dict[Flower, int]):
        """
        Imposes 1 second time limit on self.prepare_bouqets.  Do not override this method
        """
        return self.prepare_bouquets(flower_counts)

    @abstractmethod
    def zero_score_bouquet(self):
        """
        :return: a Bouquet for which your scoring function will return 0
        """
        pass

    @abstractmethod
    def one_score_bouquet(self):
        """
        :return: a Bouquet for which your scoring function will return 1
        """
        pass

    @abstractmethod
    def score_types(self, types: Dict[FlowerColors, int]):
        """
        :param types: dictionary of flower types and their associated counts in the bouquet
        :return: A score representing preference of the flower types in the bouquet
        """
        pass

    @abstractmethod
    def score_colors(self, colors: Dict[FlowerColors, int]):
        """
        :param colors: dictionary of flower colors and their associated counts in the bouquet
        :return: A score representing preference of the flower colors in the bouquet
        """
        pass

    @abstractmethod
    def score_sizes(self, sizes: Dict[FlowerSizes, int]):
        """
        :param sizes: dictionary of flower sizes and their associated counts in the bouquet
        :return: A score representing preference of the flower sizes in the bouquet
        """
        pass

    @abstractmethod
    def receive_feedback(self, feedback):
        """
        :param feedback:
        :return: nothing
        """
        pass

    @break_after(1, fallback_func=None)
    def receive_feedback_timed(self, feedback):
        """
        Imposes 1 second time limit on self.receive_feedback.  Do not override this method
        """
        return self.receive_feedback(feedback)

    def get_num_suitors(self):
        """
        :return: the number of suitors in the simulation.  This is used elsewhere so only override this if you
        change self.num_suitors
        """
        return self.num_suitors
