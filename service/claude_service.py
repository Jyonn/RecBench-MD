import anthropic

from utils.prompt import CHAT_SYSTEM
from service.base_service import BaseService


class ClaudeService(BaseService):
    def __init__(self, version, **kwargs):
        super().__init__(**kwargs)

        self.version = version
        self.client = anthropic.Anthropic(api_key=self.auth)

    def ask(self, content):
        dialog = [{"role": "user", "content": content}]
        resp = self.client.messages.create(
            model=self.version,
            system=CHAT_SYSTEM,
            messages=dialog,
            max_tokens=1024,
        )
        # dialog.append({"role": "assistant", "content": resp.content[0].text})
        return resp.content[0].text


class Claude21Service(ClaudeService):
    def __init__(self, **kwargs):
        super().__init__(version='claude-2.1', **kwargs)


class Claude3Service(ClaudeService):
    def __init__(self, **kwargs):
        super().__init__(version='claude-3-opus-20240229', **kwargs)
