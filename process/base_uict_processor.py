import abc

import pandas as pd

from process.base_uspe_processor import USPEProcessor


class UICTProcessor(USPEProcessor, abc.ABC):
    """
    user-item-click-time processor, for those have negative samples here but no user sequence
    """
    DAT_COL: str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._interactions = None

    def _load_users(self, interactions):
        item_set = set(self.items[self.IID_COL].unique())

        interactions = interactions[interactions[self.IID_COL].isin(item_set)]
        interactions = interactions.groupby(self.UID_COL)
        interactions = interactions.filter(lambda x: x[self.CLK_COL].nunique() == 2)
        self._interactions = interactions

        pos_inters = interactions[interactions[self.CLK_COL] == 1]

        users = pos_inters.sort_values(
            [self.UID_COL, self.DAT_COL]
        ).groupby(self.UID_COL)[self.IID_COL].apply(list).reset_index()
        users.columns = [self.UID_COL, self.HIS_COL]

        return self._extract_pos_samples(users)

    def load_interactions(self) -> pd.DataFrame:
        user_set = set(self.users[self.UID_COL].unique())

        neg_inters = self._interactions[self._interactions[self.CLK_COL] == 0]
        neg_inters = neg_inters[neg_inters[self.UID_COL].isin(user_set)]
        neg_inters = neg_inters.drop(columns=[self.DAT_COL])

        # concat with positive interactions
        return pd.concat([neg_inters, self._pos_inters], ignore_index=True)
