import os


class Exporter:
    def __init__(self, path):
        self.path = path
        self.error_path = path + '.progress'

    def write(self, data):
        with open(self.path, 'a') as f:
            f.write(f'{data}\n')

    def save_progress(self, index):
        with open(self.error_path, 'w') as f:
            f.write(str(index))

    def load_progress(self):
        if not os.path.exists(self.error_path):
            return 0

        with open(self.error_path, 'r') as f:
            return int(f.read())
