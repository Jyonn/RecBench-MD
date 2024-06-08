with open('.data') as f:
    data = f.read()

data_dirs = {}
for line in data.strip().split('\n'):
    name, key = line.split('=')
    data_dirs[name.strip()] = key.strip()


def get_data_dir(dataset):
    return data_dirs.get(dataset, None)
