import logging
import re

from suitors.g1 import Suitor as G1Suitor
from suitors.g2 import Suitor as G2Suitor
from suitors.g3 import Suitor as G3Suitor
from suitors.g4 import Suitor as G4Suitor
from suitors.g5 import Suitor as G5Suitor
from suitors.g6 import Suitor as G6Suitor
from suitors.g7 import Suitor as G7Suitor
from suitors.g8 import Suitor as G8Suitor
from suitors.g9 import Suitor as G9Suitor
from suitors.random_suitor import RandomSuitor


logger = logging.getLogger(__name__)


def suitor_by_name(team_name, *args):
    g_team = re.match(r'g(\d+)', team_name)
    if g_team is not None:
        group_num = g_team.group(1)
        try:
            return globals()[f'G{group_num}Suitor'](*args)
        except:
            return None
    if team_name != 'rand':
        error_msg = f'Invalid group name provided --> {team_name}'
        logger.error(error_msg)
        raise Exception(error_msg)
    return RandomSuitor(*args)
