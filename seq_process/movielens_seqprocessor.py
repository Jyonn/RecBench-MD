from process.movielens_processor import MovieLensProcessor
from seq_process.base_uict_seqprocessor import UICTSeqProcessor


class MovieLensSeqProcessor(UICTSeqProcessor, MovieLensProcessor):
    NUM_TEST = 5_000
    NUM_FINETUNE = 40_000

    REQUIRE_STRINGIFY = False
