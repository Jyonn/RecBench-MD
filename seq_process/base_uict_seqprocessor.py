import abc

from seq_process.base_seqprocessor import BaseSeqProcessor


class UICTSeqProcessor(BaseSeqProcessor, abc.ABC):
    """
    user-item-click-time processor, for those have negative samples here but no user sequence
    """
    DAT_COL: str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load_users(self, interactions):
        item_set = set(self.items[self.IID_COL].unique())

        interactions = interactions[interactions[self.IID_COL].isin(item_set)]
        pos_inters = interactions[interactions[self.LBL_COL] == 1]

        users = pos_inters.sort_values(
            [self.UID_COL, self.DAT_COL]
        ).groupby(self.UID_COL)[self.IID_COL].apply(list).reset_index()
        users.columns = [self.UID_COL, self.HIS_COL]

        return users
