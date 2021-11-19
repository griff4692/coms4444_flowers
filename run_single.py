import json
import os
import regex as re

import argparse
import pandas as pd
from p_tqdm import p_uimap

from tourney_scripts import DAYS
from main import FlowerMarriageGame


class RunArgs:
    def __init__(self):
        self.restrict_time = True
        self.save_results = True
        self.remove_round_logging = True
        self.gui = False
        self.p_from_config = False


def run_experiment(run):
    run_id = re.sub(r'\W+', '_', json.dumps(run)).strip('_')
    out_fn = os.path.join('single_results', f'{run_id}.csv')
    if os.path.exists(out_fn) and not tourney_args.overwrite:
        print(f'Already played {run_id}')
        return 0
    run_args = RunArgs()
    run_args.run_id = run_id
    run_args.log_file = f'single_logs/{run_id}.txt'
    run_args.random_state = run['random_state']
    run_args.p = run['p']
    run_args.d = run['d']
    run_args.group = run['group']
    try:
        game = FlowerMarriageGame(run_args)
        game.play()
        output_df = game.generate_output_df()
        output_df.to_csv(out_fn, index=False)
    except:
        print('Uncaught error.')
        return 0
    return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Tournament-Level Settings For Single-Group Runs')
    parser.add_argument('-overwrite', default=False, action='store_true')
    parser.add_argument('--cpu_frac', default=0.5, type=float)
    parser.add_argument('--d_filter', default=None, type=int, choices=DAYS)
    tourney_args = parser.parse_args()
    os.makedirs('single_logs', exist_ok=True)
    os.makedirs('single_results', exist_ok=True)
    tourney_script = pd.read_csv('single_configs.csv')
    if tourney_args.d_filter is not None:
        tourney_script = tourney_script[tourney_script['d'] == tourney_args.d_filter]
        assert len(tourney_script) > 0
    runs = tourney_script.to_dict('records')

    statuses = list(p_uimap(run_experiment, runs))
    print(f'{sum(statuses)}/{len(statuses)} experiments run.')
