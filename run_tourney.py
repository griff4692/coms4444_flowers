import json
import os

import argparse
import pandas as pd
from tqdm import tqdm

from tourney_scripts import GROUPS
from main import FlowerMarriageGame


def get_default_args():
    parser = argparse.ArgumentParser('Tournament Run')
    args = parser.parse_args()
    args.restrict_time = True
    args.save_results = True
    args.remove_round_logging = True
    args.gui = False
    args.from_config = False
    return args


if __name__ == '__main__':
    OVERWRITE = True
    tourney_script = pd.read_csv('tourney_configs.csv')
    runs = tourney_script.to_dict('records')

    for run in tqdm(runs, total=len(runs)):
        run_id = abs(hash(json.dumps(run)))
        out_fn = os.path.join('results', f'{run_id}.csv')
        if os.path.exists(out_fn) and not OVERWRITE:
            continue
        args = get_default_args()
        args.run_id = run_id
        args.log_file = f'logs/{run_id}.txt'
        args.random_state = run['random_state']
        args.p = run['p']
        args.d = run['d']
        suitor_names = []
        for group in GROUPS:
            suitor_names += [group] * run[group]
        assert len(suitor_names) == args.p
        game = FlowerMarriageGame(args, suitor_names=suitor_names)
        game.play()
        output_df = game.generate_output_df()
        output_df.to_csv(out_fn, index=False)
