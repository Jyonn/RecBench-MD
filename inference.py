from llm.bert import Bert
from llm.llama import Llama
from llm.opt import OPT
from utils.gpu import GPU

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

device = GPU.auto_choose(torch_format=True)

# print(Bert(device=device).ask(content))
# print(Llama(path='/home/data1/qijiong/llama-7b', device=device).ask(content))
print(OPT(device=device).ask(content))
