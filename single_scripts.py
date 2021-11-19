import pandas as pd

GROUPS = ['g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8', 'g9']

DAYS = [
    1, 7, 14, 21, 28, 90, 365
]
RANDOM_STATES = list(range(1992, 1992 + 10))
PLAYER_COUNTS = [4, 8, 18, 36, 90]


if __name__ == '__main__':
    df = []
    for d in DAYS:
        for p in PLAYER_COUNTS:
            for random_state in RANDOM_STATES:
                default_row = {'d': d, 'p': p, 'random_state': random_state, 'priority': d * p}
                for group in GROUPS:
                    row = default_row.copy()
                    row['group'] = group
                    df.append(row)
    df = pd.DataFrame(df)
    df.sort_values(by='priority', inplace=True)
    df.to_csv('single_configs.csv', index=False)
    print('Done!')
