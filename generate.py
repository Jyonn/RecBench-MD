from service.claude_service import ClaudeService, Claude3Service, Claude21Service
from service.gemini_service import GeminiService
from service.gpt_service import GPT35Service, GPT4Service
from utils.auth import GPT_KEY, CLAUDE_KEY, GEMINI_KEY

content = """User behavior sequence:
(0) 'Wheel Of Fortune' Guest Delivers Hilarious, Off The Rails Introduction
(1) Three takeaways from Yankees' ALCS Game 5 victory over the Astros
(2) Rosie O'Donnell: Barbara Walters Isn't 'Up to Speaking to People' Right Now
(3) Four flight attendants were arrested in Miami's airport after bringing in thousands in cash, police say
(4) Michigan sends breakup tweet to Notre Dame as series goes on hold
(5) This Wedding Photo of a Canine Best Man Captures Just How Deep a Dog's Love Truly Is
(6) Robert Evans, 'Chinatown' Producer and Paramount Chief, Dies at 89
(7) Former US Senator Kay Hagan dead at 66
(8) Joe Biden reportedly denied Communion at a South Carolina church because of his stance on abortion
Candidate item: Charles Rogers, former Michigan State football, Detroit Lions star, dead at 38     
"""

# print(GPT35Service(auth=GPT_KEY).ask(content))
# print(GPT4Service(auth=GPT_KEY).ask(content))
# print(Claude21Service(auth=CLAUDE_KEY).ask(content))
response = GeminiService(auth=GEMINI_KEY).ask(content)
response = response.replace('\n', '').replace('\r', '')
print(response)
