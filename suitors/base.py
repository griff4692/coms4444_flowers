from abc import ABC, abstractmethod
from typing import Dict

from flowers import Bouquet, Flower


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
        :param flower_counts: allocate the flowers to
        :return: list of tuples of (self.suitor_id, recipient_id, chosen_bouquet)
        the list should be of length len(self.num_suitors) - 1 because you should give a bouquet to everyone
         but yourself

        To get the list of suitor ids not including yourself, use the following snippet:

        all_ids = np.arange(self.num_suitors)
        recipient_ids = all_ids[all_ids != self.suitor_id]
        """
        pass

    @abstractmethod
    def score_type(self, bouquet: Bouquet):
        """
        :param bouquet: an arrangement of flowers
        :return: A score representing preference of the flower types in the bouquet
        """
        pass

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
    def score_color(self, bouquet: Bouquet):
        """
        :param bouquet: an arrangement of flowers
        :return: A score representing preference of the flower colors in the bouquet
        """
        pass

    @abstractmethod
    def score_size(self, bouquet: Bouquet):
        """
        :param bouquet: an arrangement of flowers
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
