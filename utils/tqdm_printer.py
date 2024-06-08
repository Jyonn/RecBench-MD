import sys


class TqdmPrinter:
    _TQDM_MODE = False

    @classmethod
    def activate(cls):
        cls._TQDM_MODE = True

    @classmethod
    def deactivate(cls):
        cls._TQDM_MODE = False
        sys.stdout.write('\n')

    @classmethod
    def __call__(cls, prefix_s, prefix_s_with_color, text, **kwargs):
        if cls._TQDM_MODE:
            sys.stdout.write('\r')
        sys.stdout.write(f'{prefix_s_with_color} {text}')
        if cls._TQDM_MODE:
            sys.stdout.flush()
        else:
            sys.stdout.write('\n')
