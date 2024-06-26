import openai

from utils.prompt import CHAT_SYSTEM
from service.base_service import BaseService


class GPTService(BaseService):
    def __init__(self, version, **kwargs):
        super().__init__(**kwargs)

        self.version = version

    def ask(self, content):
        dialog = [{"role": "system", "content": CHAT_SYSTEM}, {"role": "user", "content": content}]
        resp = openai.ChatCompletion.create(
            model=self.version,
            messages=dialog,
            api_key=self.auth,
            timeout=5,
        )
        # dialog.append(resp['choices'][0]['message'])
        return resp['choices'][0]['message']['content']


class GPT35Service(GPTService):
    def __init__(self, **kwargs):
        super().__init__(version='gpt-3.5-turbo', **kwargs)


class GPT4Service(GPTService):
    def __init__(self, **kwargs):
        super().__init__(version='gpt-4-turbo', **kwargs)


class GPT4oService(GPTService):
    def __init__(self, **kwargs):
        super().__init__(version='gpt-4o', **kwargs)
