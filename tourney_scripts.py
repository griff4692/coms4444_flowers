import itertools
from collections import defaultdict
from itertools import combinations
import pandas as pd

GROUPS = ['g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8', 'g9']
CONFIG_ROWS = ['d', 'random_state']
CONFIG_DIR = './configs'

DAYS = [
    1, 7, 14, 21, 28, 90, 365
]

RANDOM_STATES = list(range(1992, 1992 + 10))

PLAYER_COUNTS = [4, 8, 18, 36, 90]


def player_counts(p_set, dup=1):
    return {group: dup if group in p_set else 0 for group in GROUPS}


if __name__ == '__main__':
    df = []
    for d in DAYS:
        for p in PLAYER_COUNTS:
            for random_state in RANDOM_STATES:
                default_row = {'d': d, 'p': p, 'random_state': random_state, 'priority': d * p}
                if p in {4, 8}:
                    player_sets = itertools.combinations(GROUPS, p)
                    for player_set in player_sets:
                        player_set = set(player_set)
                        row = default_row.copy()
                        row.update(player_counts(player_set, dup=1))
                        df.append(row)
                elif p in {18, 36, 90}:
                    player_set = set(GROUPS)
                    dup = p // len(GROUPS)
                    row = default_row.copy()
                    row.update(player_counts(player_set, dup=dup))
                    df.append(row)
                else:
                    raise Exception('Unrecognized')

    df = pd.DataFrame(df)
    df.sort_values(by='priority', inplace=True)
    df.to_csv('global_config.csv', index=False)
