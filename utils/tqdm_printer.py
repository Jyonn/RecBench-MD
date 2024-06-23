import sys
import time


class TqdmPrinter:
    _TQDM_MODE = False

    @classmethod
    def activate(cls, interval=1):
        cls._TQDM_MODE = True
        cls._START_TIME = time.time()
        cls._LAST_PRINT = 0
        cls._INTERVAL = interval

    @classmethod
    def deactivate(cls):
        cls._TQDM_MODE = False
        sys.stdout.write('\n')

    @staticmethod
    def format_interval(t):
        m, s = divmod(int(t), 60)
        h, m = divmod(m, 60)
        if h:
            return f'{h:02d}:{m:02d}:{s:02d}'
        else:
            return f'{m:02d}:{s:02d}'

    @classmethod
    def __call__(cls, prefixes, prefix_s, prefix_s_with_color, text, **kwargs):
        if cls._TQDM_MODE:
            sys.stdout.write('\r')
            current, count = kwargs['current'], kwargs['count']
            current_time = time.time()
            if current_time - cls._LAST_PRINT <= cls._INTERVAL:
                return
            cls._LAST_PRINT = current_time
            left_time = (current_time - cls._START_TIME) / current * (count - current)
            sys.stdout.write(f'{prefix_s_with_color} ({current}/{count}, ETA {cls.format_interval(left_time)}) {text}')
        else:
            sys.stdout.write(f'{prefix_s_with_color} {text}')

        if cls._TQDM_MODE:
            sys.stdout.flush()
        else:
            sys.stdout.write('\n')
