import os


class Exporter:
    def __init__(self, path):
        self.path = path
        self.error_path = path + '.progress'
        self.metrics_path = path + '.metrics'

    def write(self, data):
        with open(self.path, 'a') as f:
            f.write(f'{data}\n')

    def read(self):
        # each line is a float
        with open(self.path, 'r') as f:
            return [float(line.strip()) for line in f]

    def save_progress(self, index):
        with open(self.error_path, 'w') as f:
            f.write(str(index))

    def load_progress(self):
        if not os.path.exists(self.error_path):
            return 0

        with open(self.error_path, 'r') as f:
            return int(f.read())

    def save_metrics(self, metrics):
        # for metric, value in results.items():
        #     pnt(f'{metric}: {value:.4f}')
        with open(self.metrics_path, 'w') as f:
            for metric, value in metrics.items():
                f.write(f'{metric}: {value:.4f}\n')
