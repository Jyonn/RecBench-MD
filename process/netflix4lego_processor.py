from process.netflix_processor import NetflixProcessor


class Netflix4LegoProcessor(NetflixProcessor):
    NUM_TEST = 20_000
    NUM_FINETUNE = 100_000
