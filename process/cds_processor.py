from process.base_amazon_processor import AmazonProcessor


class CDsProcessor(AmazonProcessor):
    NUM_TEST = 0
    NUM_FINETUNE = 100000

    def __init__(self, **kwargs):
        super().__init__(subset='', **kwargs)
