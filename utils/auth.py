with open('.auth') as f:
    auth_data = f.read()

auth = {}
for line in auth_data.strip().split('\n'):
    name, key = line.split('=')
    auth[name.strip()] = key.strip()

GPT_KEY = auth['gpt']
CLAUDE_KEY = auth['claude']
