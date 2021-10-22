import logging
import itertools

import argparse
import numpy as np
import pandas as pd
from remi import start
from scipy.stats import rankdata

from flowers import Bouquet, get_all_possible_flowers, sample_n_random_flowers
from gui_app import FlowerApp
from suitors.base import BaseSuitor
from suitors.suitor_factory import suitor_by_name
from utils import flatten_counter


class FlowerMarriageGame:
    def __init__(self, args):
        self.d = args.d
        self.restrict_time = args.restrict_time
        logging.basicConfig(
            format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG,
            handlers=[logging.FileHandler(args.log_file, mode='w'), logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)
        # A CSV config file specifying each group and their associated instances in the game.
        if args.p_from_config:
            config_df = pd.read_csv('config.csv')
            config_df = config_df[config_df['counts'] > 0]
            assert len(config_df) > 0
            self.suitor_names = flatten_counter(dict(zip(config_df['group'], config_df['counts'])))
            self.p = config_df.counts.sum()
        else:
            self.p = args.p
            self.suitor_names = [args.group] * self.p
        assert self.p >= 2 and self.p % 2 == 0

        self.gui = args.gui
        self.possible_flowers = get_all_possible_flowers()
        self.num_flowers_to_sample = 6 * (self.p - 1)
        self.logger.info(
            f'Will sample {self.num_flowers_to_sample} out of {len(self.possible_flowers)} possible flowers.')

        # Instantiate suitors
        self.reset_game_state()

        self.flower_app = None
        if self.gui:
            start(FlowerApp, address=args.address, port=args.port, start_browser=not(args.no_browser),
                  update_interval=0.5, userdata=(self, ))

    def reset_game_state(self):
        # Instantiate suitors
        self.suitors = [suitor_by_name(self.suitor_names[i], self.d, self.p, i) for i in range(self.p)]
        suitor_conformity = list(map(validate_suitor, self.suitors))
        for suitor, suitor_status in zip(self.suitors, suitor_conformity):
            if suitor_conformity == 0:
                self.logger.error(f'Suitor {suitor.suitor_id} provided invalid zero/one boundary bouquets.')

        # Instantiate arrays to keep track of bouquets provided at each round, as well as scores and ranks.
        self.bouquets = np.empty(shape=(self.d, self.p, self.p), dtype=Bouquet)
        self.scores = np.zeros(shape=(self.d, self.p, self.p), dtype=float)
        self.ranks = np.zeros(shape=(self.d, self.p, self.p), dtype=int)
        self.next_round = 0
        self.marriages = None
        self.advantage = None

    def play(self):
        for curr_round in range(self.next_round, self.d):
            self.simulate_round(curr_round)
        self.next_round = self.d
        return self.marry_folks()

    def is_over(self):
        return self.next_round == self.d

    def set_app(self, flower_app):
        self.flower_app = flower_app

    def resolve_prepare_func(self, suitor):
        return suitor.prepare_bouquets_timed if self.restrict_time else suitor.prepare_bouquets

    def resolve_feedback_func(self, suitor):
        return suitor.receive_feedback_timed if self.restrict_time else suitor.receive_feedback

    def simulate_round(self, curr_round):
        suitor_ids = [suitor.suitor_id for suitor in self.suitors]
        flowers_for_round = sample_n_random_flowers(self.possible_flowers, self.num_flowers_to_sample)
        # bouquets = get_all_possible_bouquets(flowers_for_round)
        offers = list(itertools.chain(
            *map(lambda suitor: self.resolve_prepare_func(suitor)(flowers_for_round), self.suitors)))
        for (suitor_from, suitor_to, bouquet) in offers:
            assert suitor_from != suitor_to
            self.bouquets[curr_round, suitor_from, suitor_to] = bouquet
            score = aggregate_score(self.suitors[suitor_to], bouquet)
            if score < 0 or score > 1:
                self.logger.error(
                    f'Suitor {suitor_to} provided an invalid score - i.e., not in [0, 1].  Setting it to 0')
                score = 0
            self.scores[curr_round, suitor_from, suitor_to] = score
        np.fill_diagonal(self.scores[curr_round], float('-inf'))
        round_ranks = rankdata(-self.scores[curr_round], axis=0, method='min')
        self.ranks[curr_round, :, :] = round_ranks
        list(map(lambda i: self.resolve_feedback_func(self.suitors[i])(
            tuple(zip(self.ranks[curr_round, i, :], self.scores[curr_round, i, :]))), suitor_ids))

    def simulate_next_round(self):
        if self.next_round < self.d:
            self.simulate_round(self.next_round)
            self.next_round += 1
        else:
            raise Exception('Should not be able to call this when the days have run out.')
        if self.next_round == self.d:
            self.logger.info('We\'re done.  Let\'s get married!')
            self.marry_folks()

    def marry_folks(self):
        final_scores = self.scores[-1, :, :]
        married = np.full((self.p,), False)
        second_best_scores = np.array([np.sort(list(set(final_scores[:, i])))[-2] for i in range(self.p)])
        self.advantage = final_scores - second_best_scores
        priority = np.copy(self.advantage)

        # Marry off
        marriage_scores = np.zeros(shape=(self.p,))
        marriage_unions = []
        for marriage_round in range(self.p // 2):
            # Make sure we don't select already married 'folks'
            priority[married, :] = float('-inf')
            priority[:, married] = float('-inf')

            # Select the most excited partner and his or her chosen flower-giver as the next in the marriage queue
            marriage_pair = np.unravel_index(np.argmax(priority, axis=None), priority.shape)
            marriage_score = priority[marriage_pair]
            suitor, chooser = marriage_pair
            self.logger.info(f'{chooser} chose to marry {suitor} with a score of {marriage_score}')
            married[suitor] = married[chooser] = True
            marriage_unions.append({'suitor': suitor, 'chooser': chooser})
            marriage_scores[suitor] = marriage_scores[chooser] = marriage_score
        assert married.sum() == self.p
        self.marriages = {
            'scores': list(marriage_scores),
            'unions': marriage_unions
        }
        self.logger.info(self.marriages)
        return self.marriages


def aggregate_score(suitor: BaseSuitor, bouquet: Bouquet):
    color_score = suitor.score_colors(bouquet.colors)
    size_score = suitor.score_sizes(bouquet.sizes)
    type_score = suitor.score_types(bouquet.types)
    agg_score = color_score + size_score + type_score
    assert 0 <= agg_score <= 1
    return agg_score


def validate_suitor(suitor):
    try:
        assert aggregate_score(suitor, suitor.zero_score_bouquet()) == 0
        assert aggregate_score(suitor, suitor.one_score_bouquet()) == 1
        return 1
    except:
        return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser('COMS 4444 Fall 2021 - Project 3.  Flower Arrangements')

    parser.add_argument('--d', type=int, default=3, help='Length of the courtship in days.')
    parser.add_argument('--p', type=int, default=10, help='Number of suitors (eligible people).')
    parser.add_argument('-restrict_time', default=False, action='store_true')
    parser.add_argument('--log_file', default='game.log')
    parser.add_argument(
        '--group', type=str, default='Group name will be duplicated p times. Ignored if p_from_config=True.')
    parser.add_argument('-p_from_config', default=False, action='store_true')
    parser.add_argument('--random_state', type=int, default=1992, help='Random seed.  Fix for consistent experiments')
    parser.add_argument('--port', '-p', type=int, default=8080, help='Port to start')
    parser.add_argument('--address', type=str, default='127.0.0.1', help='Address')
    parser.add_argument('-no_browser', default=False, action='store_true', help='Disable browser launching in GUI mode')
    parser.add_argument('-gui', default=False, action='store_true', help='Enable GUI')

    args = parser.parse_args()

    np.random.seed(args.random_state)
    game = FlowerMarriageGame(args)
    game.play()
