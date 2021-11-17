import logging
import signal

import numpy as np

from flowers import Bouquet


class TimeoutException(Exception):
    def __init__(self, value='Timed Out'):
        self.value = value

    def __str__(self):
        return repr(self.value)


def prepare_empty_bouquets(suitor, current_bouquet=None):
    all_ids = np.arange(suitor.get_num_suitors())
    recipient_ids = all_ids[all_ids != suitor.suitor_id]
    return [(suitor.suitor_id, recipient_id, Bouquet({})) for recipient_id in recipient_ids]


def break_after(seconds=1, fallback_func=None):
    def timeout_handler(signum, frame):   # Custom signal handler
        raise TimeoutException()

    def function(function):
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            time_out = False
            try:
                res = function(*args, **kwargs)
                signal.alarm(0)  # Clear alarm
                return res
            except TimeoutException:
                logger = logging.getLogger(__name__)
                suitor = args[0]
                time_out = True
            except Exception:
                logger = logging.getLogger(__name__)
                suitor = args[0]
            if fallback_func:
                if time_out:
                    timeout_str = f'{suitor.name}_{suitor.suitor_id} took too long.  ' \
                                  f'Preparing empty bouquets for the other (disappointed) suitors.'
                    logger.error(timeout_str)
                else:
                    timeout_str = f'{suitor.name}_{suitor.suitor_id} had a code error.  ' \
                                  f'Preparing empty bouquets for the other (disappointed) suitors.'
                    logger.error(timeout_str)
                return fallback_func(*args)
            else:
                if time_out:
                    timeout_str = f'{suitor.name}_{suitor.suitor_id} took too long. Skipping.'
                else:
                    timeout_str = f'{suitor.name}_{suitor.suitor_id} had a code error.  Skipping.'
                logger.error(timeout_str)
            return
        return wrapper
    return function
