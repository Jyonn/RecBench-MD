from process.base_processor import BaseProcessor
from utils.config_init import ConfigInit
from utils.function import load_processor
from utils.metrics import MetricPool

if __name__ == '__main__':
    configuration = ConfigInit(
        required_args=['data', 'filepath'],
        default_args=dict(
            metrics='ndcg@10+auc',
        ),
        makedirs=[]
    ).parse()

    with open(configuration.filepath, 'r') as f:
        scores = f.read().strip().split('\n')
        scores = [float(score) for score in scores]

    processor: BaseProcessor = load_processor(configuration.data)
    processor.load()

    labels = processor.test_set[processor.LBL_COL].values
    groups = processor.test_set[processor.UID_COL].values

    pool = MetricPool.parse(configuration.metrics.split('+'))
    results = pool.calculate(scores, labels, groups)
    for metric, value in results.items():
        print(f'{metric}: {value:.4f}')
