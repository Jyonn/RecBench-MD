from process.yelp_processor import YelpProcessor


class Yelp4LegoProcessor(YelpProcessor):
    NUM_TEST = 20_000
    NUM_FINETUNE = 100_000
