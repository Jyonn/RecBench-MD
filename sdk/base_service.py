class BaseService:
    def __init__(self, auth):
        self.auth = auth

    @classmethod
    def get_name(cls):
        return cls.__name__.replace('Service', '').lower()

    def ask(self, content) -> str:
        pass
