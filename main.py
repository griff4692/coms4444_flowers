from collections import Counter
from enum import Enum
import itertools
from typing import Dict, List
from scipy.stats import rankdata

import argparse
import numpy as np


MAX_BOUQUET_SIZE = 12


class FlowerSize(Enum):
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


class Flower:
    def __init__(self, size: FlowerSize, color: FlowerColors, type: FlowerTypes):
        self.size = size
        self.color = color
        self.type = type

    def __str__(self):
        return f'(size={self.size}, color={self.color}, type={self.type})'


class Bouquet:
    def __init__(self, arrangement: Dict[Flower, int]):
        self.arrangement = arrangement


class Suitor:
    def __init__(self, days, num_suitors, suitor_id):
        self.days = days
        self.num_suitors = num_suitors
        self.suitor_id = suitor_id

    def prepare_bouquet(self, available_flowers, recipient_id):
        size = int(np.random.randint(0, MAX_BOUQUET_SIZE + 1))
        chosen_bouquet = Bouquet(dict(Counter(list(np.random.choice(available_flowers, size=size, replace=True)))))
        return self.suitor_id, recipient_id, chosen_bouquet

    def prepare_bouquets(self, available_flowers: List[Flower]):
        all_ids = np.arange(self.num_suitors)
        recipient_ids = all_ids[all_ids != self.suitor_id]
        return list(map(lambda recipient_id: self.prepare_bouquet(available_flowers, recipient_id), recipient_ids))

    def score(self, bouquet: Bouquet):
        return float(np.random.random())

    def receive_feedback(self, feedback):
        print('here')


def sample_n_random_flowers(possible_flowers, n):
    assert len(possible_flowers) >= n
    sample_idxs = set(np.random.choice(np.arange(len(possible_flowers)), size=(n, ), replace=False))
    return [f for i, f in enumerate(possible_flowers) if i in sample_idxs]


if __name__ == '__main__':
    parser = argparse.ArgumentParser('COMS 4444 Fall 2021 - Project 3.  Flower Arrangements')

    parser.add_argument('--d', type=int, default=100, help='Length of the courtship in days.')
    parser.add_argument('--p', type=int, default=10, help='Number of suitors (eligible people).')
    parser.add_argument('--random_state', type=int, default=1992, help='Random seed.  Fix for consistent experiments')

    args = parser.parse_args()

    np.random.seed(args.random_state)
    possible_flowers = list(map(
        lambda x: Flower(*x), list(itertools.product(*[list(FlowerSize), list(FlowerColors), list(FlowerTypes)]))))
    num_flowers_to_sample = min(len(possible_flowers), 6 * (args.p - 1))
    print(f'Will sample {num_flowers_to_sample} out of {len(possible_flowers)} possible flowers.')

    suitors = [Suitor(args.d, args.p, i) for i in range(args.p)]
    suitor_ids = [suitor.suitor_id for suitor in suitors]

    bouquets = np.empty(shape=(args.d, args.p, args.p), dtype=Bouquet)
    scores = np.zeros(shape=(args.d, args.p, args.p), dtype=float)
    ranks = np.zeros(shape=(args.d, args.p, args.p), dtype=int)

    for round in range(args.d):
        flowers_for_round = list(np.random.choice(possible_flowers, size=(num_flowers_to_sample, ), replace=False))
        offers = list(itertools.chain(*map(lambda suitor: suitor.prepare_bouquets(flowers_for_round), suitors)))
        for (suitor_from, suitor_to, bouquet) in offers:
            bouquets[round, suitor_from, suitor_to] = bouquet
            scores[round, suitor_from, suitor_to] = suitors[suitor_to].score(bouquet)
        np.fill_diagonal(scores[round], float('-inf'))
        round_ranks = rankdata(-scores[round], axis=1, method='min')
        ranks[round, :, :] = round_ranks
        list(map(lambda i: suitors[i].receive_feedback(tuple(zip(round_ranks[i, :], scores[round, i, :]))), suitor_ids))
