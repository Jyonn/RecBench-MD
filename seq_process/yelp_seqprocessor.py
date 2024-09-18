from process.yelp_processor import YelpProcessor
from seq_process.base_seqprocessor import BaseSeqProcessor


class YelpSeqProcessor(BaseSeqProcessor, YelpProcessor):
    NUM_TEST = 5_000
    NUM_FINETUNE = 40_000

    REQUIRE_STRINGIFY = False
