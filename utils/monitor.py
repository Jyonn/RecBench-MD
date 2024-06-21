from utils.metrics import MetricPool


class Monitor:
    BEST = 1
    SKIP = 2
    STOP = 3

    def __init__(self, patience=2):
        self.patience = patience
        self.best_value = None
        self.minimize = None
        self.best_index = 0
        self.current_index = -1

    def push(self, metric, value):
        self.current_index += 1

        if self.best_value is None:
            self.minimize = MetricPool.is_minimize(metric)
            self.best_value = value
            return self.BEST

        if self.minimize ^ (value > self.best_value):
            self.best_value = value
            self.best_index = self.current_index
            return self.BEST

        if self.current_index - self.best_index >= self.patience:
            return self.STOP
        return self.SKIP
