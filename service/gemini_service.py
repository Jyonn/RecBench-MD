import google.generativeai as genai

from utils.prompt import CHAT_SYSTEM
from service.base_service import BaseService


class GeminiService(BaseService):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        genai.configure(api_key=self.auth)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def ask(self, content):
        resp = self.model.generate_content(CHAT_SYSTEM + content)
        return resp.text
