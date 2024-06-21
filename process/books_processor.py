from process.base_amazon_processor import AmazonProcessor


class BooksProcessor(AmazonProcessor):
    NUM_TEST = 0
    NUM_FINETUNE = 100_000

    def __init__(self, **kwargs):
        super().__init__(subset='Books', **kwargs)
