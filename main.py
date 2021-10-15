import itertools
from scipy.stats import rankdata

import argparse
import numpy as np

from flowers import Flower, FlowerColors, FlowerTypes, FlowerSizes, Bouquet
from suitors.base import BaseSuitor
from suitors.random_suitor import RandomSuitor
from utils import sample_n_random_flowers


def aggregate_score(suitor: BaseSuitor, bouquet: Bouquet):
    color_score = suitor.score_color(bouquet)
    size_score = suitor.score_color(bouquet)
    type_score = suitor.score_type(bouquet)
    return color_score + size_score + type_score


def validate_suitor(suitor):
    assert aggregate_score(suitor, suitor.zero_score_bouquet()) == 0
    assert aggregate_score(suitor, suitor.one_score_bouquet()) == 1


def simulate_round(round, bouquets, scores, ranks, suitors, possible_flowers, num_flowers_to_sample):
    suitor_ids = [suitor.suitor_id for suitor in suitors]
    flowers_for_round = sample_n_random_flowers(possible_flowers, num_flowers_to_sample)
    offers = list(itertools.chain(*map(lambda suitor: suitor.prepare_bouquets(flowers_for_round), suitors)))
    for (suitor_from, suitor_to, bouquet) in offers:
        assert suitor_from != suitor_to
        bouquets[round, suitor_from, suitor_to] = bouquet
        scores[round, suitor_from, suitor_to] = aggregate_score(suitors[suitor_to], bouquet)
        if scores[round, suitor_from, suitor_to] < 0 or scores[round, suitor_from, suitor_to] > 1:
            print(f'Suitor {suitor_to} provided an invalid score - i.e., not in [0, 1].  Setting it to 0')
            scores[round, suitor_from, suitor_to] = 0
    np.fill_diagonal(scores[round], float('-inf'))
    round_ranks = rankdata(-scores[round], axis=0, method='min')
    ranks[round, :, :] = round_ranks
    list(map(lambda i: suitors[i].receive_feedback(tuple(zip(round_ranks[i, :], scores[round, i, :]))), suitor_ids))


def marry_folks(final_scores):
    married = np.full((args.p, ), False)
    second_best_scores = np.array([np.sort(list(set(final_scores[:, i])))[-2] for i in range(args.p)])
    advantage = final_scores - second_best_scores

    # Marry off
    marriage_scores = np.zeros(shape=(args.p, ))
    marriage_unions = []
    for marriage_round in range(args.p // 2):
        # Make sure we don't select already married 'folks'
        advantage[married, :] = float('-inf')
        advantage[:, married] = float('-inf')

        # Select the most excited partner and his or her chosen flower-giver as the next in the marriage queue
        marriage_pair = np.unravel_index(np.argmax(advantage, axis=None), advantage.shape)
        marriage_score = advantage[marriage_pair]
        suitor, chooser = marriage_pair
        print(f'{chooser} chose to marry {suitor} with a score of {marriage_score}')
        married[suitor] = married[chooser] = True
        marriage_unions.append((chooser, suitor))
        marriage_scores[suitor] = marriage_scores[chooser] = marriage_score
    assert married.sum() == args.p
    marriage_outputs = {
        'scores': list(marriage_scores),
        'unions': marriage_unions
    }
    return marriage_outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser('COMS 4444 Fall 2021 - Project 3.  Flower Arrangements')

    parser.add_argument('--d', type=int, default=3, help='Length of the courtship in days.')
    parser.add_argument('--p', type=int, default=10, help='Number of suitors (eligible people).')
    parser.add_argument('--random_state', type=int, default=1992, help='Random seed.  Fix for consistent experiments')

    args = parser.parse_args()

    np.random.seed(args.random_state)
    possible_flowers = list(map(
        lambda x: Flower(*x), list(itertools.product(*[list(FlowerSizes), list(FlowerColors), list(FlowerTypes)]))))
    num_flowers_to_sample = 6 * (args.p - 1)
    print(f'Will sample {num_flowers_to_sample} out of {len(possible_flowers)} possible flowers.')

    # Instantiate suitors
    suitors = [RandomSuitor(args.d, args.p, i) for i in range(args.p)]

    # Instantiate arrays to keep track of bouquets provided at each round, as well as scores and ranks.
    bouquets = np.empty(shape=(args.d, args.p, args.p), dtype=Bouquet)
    scores = np.zeros(shape=(args.d, args.p, args.p), dtype=float)
    ranks = np.zeros(shape=(args.d, args.p, args.p), dtype=int)

    for round in range(args.d):
        simulate_round(round, bouquets, scores, ranks, suitors, possible_flowers, num_flowers_to_sample)

    # We only care about scores at the final round
    final_scores = scores[-1, :, :]
    # final_ranks = ranks[-1, :, :]
    marriage_outputs = marry_folks(final_scores)
    print('\n')
    for k, v in marriage_outputs.items():
        print(k, ':\n', v, end='\n')
