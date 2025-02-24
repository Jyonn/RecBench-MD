from process.base_amazon_processor import AmazonProcessor


### NOT USED, TOO SMALL

class Beauty4LegoProcessor(AmazonProcessor):
    NUM_TEST = 20_000
    NUM_FINETUNE = 0

    def __init__(self, **kwargs):
        super().__init__(subset='Electronics', **kwargs)
