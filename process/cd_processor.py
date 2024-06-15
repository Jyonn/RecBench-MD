from process.base_amazon_processor import AmazonProcessor


class CDProcessor(AmazonProcessor):
    NUM_TEST = 20000
    NUM_FINETUNE = 100000

    def __init__(self, **kwargs):
        super().__init__(subset='', **kwargs)
