import time


class BaseService:
    def __init__(self, auth):
        self.auth = auth

    @classmethod
    def get_name(cls):
        return cls.__name__.replace('Service', '').lower()

    def ask(self, content) -> str:
        pass

    def __call__(self, content):
        # time.sleep(0.1)
        error = None
        for _ in range(5):
            try:
                return self.ask(content)
            except Exception as e:
                error = str(e)
        raise RuntimeError(f'Retry over five times with error: {error}')
