from process.base_amazon_processor import AmazonProcessor


class MusicProcessor(AmazonProcessor):
    NUM_TEST = 20_000
    NUM_FINETUNE = 100_000

    POS_COUNT = 1
    NEG_COUNT = 1

    def __init__(self, **kwargs):
        super().__init__(subset='Digital_Music', **kwargs)
