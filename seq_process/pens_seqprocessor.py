from process.pens_processor import PENSProcessor
from seq_process.base_seqprocessor import BaseSeqProcessor


class PENSSeqProcessor(BaseSeqProcessor, PENSProcessor):
    NUM_TEST = 5_000
    NUM_FINETUNE = 40_000

    REQUIRE_STRINGIFY = False
