from seq_process.base_amazon_seqprocessor import AmazonSeqProcessor


class CDsSeqProcessor(AmazonSeqProcessor):
    NUM_TEST = 5_000
    NUM_FINETUNE = 40_000

    def __init__(self, **kwargs):
        super().__init__(subset='CDs_and_Vinyl', **kwargs)
