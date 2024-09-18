from process.mind_processor import MINDProcessor
from sero.base_seroprocessor import BaseSeroProcessor


class MINDSeroProcessor(BaseSeroProcessor, MINDProcessor):
    NUM_FINETUNE = 10_000
