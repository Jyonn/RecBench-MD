from process.mind_processor import MINDProcessor
from seq_process.base_seqprocessor import BaseSeqProcessor


class MINDSeqProcessor(BaseSeqProcessor, MINDProcessor):
    NUM_TEST = 5_000
    NUM_FINETUNE = 40_000

    REQUIRE_STRINGIFY = False
