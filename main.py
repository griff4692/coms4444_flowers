from collections import Counter, defaultdict
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
        self.remove_round_logging = args.remove_round_logging
        logging.basicConfig(
            level=logging.DEBUG,
            handlers=[logging.FileHandler(args.log_file, mode='w'), logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)
        # A CSV config file specifying each group and their associated instances in the game.
        if args.p_from_config:
            config_df = pd.read_csv(args.config_path)
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
        for d in range(self.d):
            for row in range(self.p):
                for col in range(self.p):
                    self.bouquets[d, row, col] = Bouquet({})
        self.scores = np.zeros(shape=(self.d, self.p, self.p), dtype=float)
        self.ranks = np.zeros(shape=(self.d, self.p, self.p), dtype=int)
        self.ties = np.zeros(shape=(self.d, self.p, self.p), dtype=int)
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

    def log_round(self, curr_round):
        for i in range(self.p):
            for j in range(self.p):
                if i == j:
                    continue
                rank, score, bouquet = (
                    self.ranks[curr_round, i, j], self.scores[curr_round, i, j], self.bouquets[curr_round, i, j])
                giver = f'{self.suitors[i].name}_{self.suitors[i].suitor_id}'
                receiver = f'{self.suitors[j].name}_{self.suitors[j].suitor_id}'
                str = f'Round {curr_round}: ' \
                      f'{giver} bouquet to {receiver} scored {round(score, 3)} (rank={rank}) -> {bouquet}'
                self.logger.info(str)

    def _is_valid_offer_format(self, offer):
        if not hasattr(offer, '__iter__'):
            return False
        if len(offer) != 3:
            return False
        if type(offer[2]) != Bouquet:
            return False
        try:
            return min(offer[:2]) > -1 and max(offer[:2]) < self.p
        except:
            return False

    def fix_offers(self, suitor, offers, flowers_for_round):
        offer_cts = defaultdict(int)
        is_more_than_market = False
        is_hallucinated = False
        valid_offers = []
        for offer in offers:
            if self._is_valid_offer_format(offer):
                valid_offers.append(offer)
            else:
                self.logger.error(f'Suitor {suitor.suitor_id} provided an invalid format for its offering: {offer}')
            for flower, ct in offer[-1].arrangement.items():
                offer_cts[flower] += ct
                if flower not in flowers_for_round:
                    is_hallucinated = True
                    self.logger.error(
                        f'Suitor {suitor.suitor_id} tried to offer a flower {flower} '
                        f'which is unavailable at the market today. '
                    )
                    break
                if offer_cts[flower] > flowers_for_round[flower]:
                    is_more_than_market = True
                    self.logger.error(
                        f'Suitor {suitor.suitor_id} gave away atleast {offer_cts[flower]} {flower} flowers. '
                        f'There are only {flowers_for_round[flower]} available at the market.')
                    break
            if is_more_than_market or is_hallucinated:
                break

        if is_more_than_market or is_hallucinated:
            # Nulling all the offers
            return [[x[0], x[1], Bouquet({})] for x in offers]
        return valid_offers

    def simulate_round(self, curr_round):
        suitor_ids = [suitor.suitor_id for suitor in self.suitors]
        flowers_for_round = sample_n_random_flowers(self.possible_flowers, self.num_flowers_to_sample)
        offers = list(map(lambda suitor: self.resolve_prepare_func(suitor)(flowers_for_round.copy()), self.suitors))
        offers = list(map(lambda i: self.fix_offers(self.suitors[i], offers[i], flowers_for_round), range(self.p)))
        offers_flat = list(itertools.chain(*offers))
        for (suitor_from, suitor_to, bouquet) in offers_flat:
            assert suitor_from != suitor_to
            bouquet = Bouquet({}) if bouquet is None else bouquet
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
        col_rank_cts = list(map(lambda col: Counter(round_ranks[:, col]), range(self.p)))
        for row in range(self.p):
            for col in range(self.p):
                rank = round_ranks[row, col]
                self.ties[curr_round, row, col] = col_rank_cts[col][rank]

        list(map(lambda i: self.resolve_feedback_func(self.suitors[i])(
            tuple(zip(self.ranks[curr_round, i, :], self.scores[curr_round, i, :], self.ties[curr_round, i, :]))
        ), suitor_ids))

        if not self.remove_round_logging:
            self.log_round(curr_round)

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
        second_best_scores = np.clip(np.array([np.sort(final_scores[:, i])[-2] for i in range(self.p)]), 0, 1)
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
            winning_pairs_w_ties = np.argwhere(priority == np.max(priority))
            num_ties = winning_pairs_w_ties.shape[0]
            # Break ties at random
            suitor, chooser = winning_pairs_w_ties[np.random.choice(np.arange(num_ties), size=(1,))[0], :]
            marriage_score = priority[suitor, chooser]
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
    parser.add_argument('--config_path', default='config.csv', help='path from which to read in the config file.')
    parser.add_argument('--random_state', type=int, default=1992, help='Random seed.  Fix for consistent experiments')
    parser.add_argument('--port', '-p', type=int, default=8080, help='Port to start')
    parser.add_argument('--address', type=str, default='127.0.0.1', help='Address')
    parser.add_argument('-no_browser', default=False, action='store_true', help='Disable browser launching in GUI mode')
    parser.add_argument('-gui', default=False, action='store_true', help='Enable GUI')
    parser.add_argument('-remove_round_logging', default=False, action='store_true')

    args = parser.parse_args()

    np.random.seed(args.random_state)
    game = FlowerMarriageGame(args)
    game.play()
