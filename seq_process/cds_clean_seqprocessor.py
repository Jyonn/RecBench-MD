import numpy as np
import pandas as pd
from tqdm import tqdm

from seq_process.cds_seqprocessor import CDsSeqProcessor


class CDsCleanSeqProcessor(CDsSeqProcessor):
    NUM_TEST = 5_000
    NUM_FINETUNE = 40_000

    def load_items(self) -> pd.DataFrame:
        items: pd.DataFrame = super().load_items()
        flags = []
        title_set = set()
        for index, item in tqdm(items.iterrows()):
            title = item['title']
            if title.lower() in title_set:
                flags.append(0)
            else:
                flags.append(1)
                title_set.add(title.lower())
        flags = np.array(flags)
        items = items[flags == 1]
        return items
