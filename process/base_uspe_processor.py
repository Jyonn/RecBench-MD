import abc

import pandas as pd

from process.base_processor import BaseProcessor


class USPEProcessor(BaseProcessor, abc.ABC):
    """
    user sequence positive sample extraction processor, for those only have user sequence but no positive sample
    """
    POS_COUNT: int

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._pos_inters = None

    def _extract_pos_samples(self, users):
        users = users[users[self.HIS_COL].apply(len) > self.POS_COUNT]

        pos_inters = []
        for index, row in users.iterrows():
            for i in range(self.POS_COUNT):
                pos_inters.append({
                    self.UID_COL: row[self.UID_COL],
                    self.IID_COL: row[self.HIS_COL][-(i + 1)],
                    self.CLK_COL: 1
                })
        self._pos_inters = pd.DataFrame(pos_inters)

        users[self.HIS_COL] = users[self.HIS_COL].apply(lambda x: x[:-self.POS_COUNT])

        return users
