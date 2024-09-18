import os

import pandas as pd
from pigmento import pnt

from process.base_processor import BaseProcessor


class BaseSeroProcessor(BaseProcessor):
    def load_public_sets(self):
        if self.test_set_valid and self.finetune_set_valid:
            super().load_public_sets()
            pnt(f'finetune set: {len(self.finetune_set)} samples')
            pnt(f'test set: {len(self.test_set)} samples')
            return

        item_count = dict()

        # iterate the user dataframe
        for idx, row in self.users.iterrows():
            # iterate the user's history
            for item in row[self.HIS_COL]:
                # if the item is not in the item_count dictionary, add it
                if item not in item_count:
                    item_count[item] = 0
                # increment the item's count
                item_count[item] += 1

        # sort the items by count
        sorted_items = sorted(item_count.items(), key=lambda x: x[1])
        cold_items = set([item for item, count in sorted_items[:int(len(sorted_items) * 0.3)]])

        interactions = self.interactions.groupby(self.UID_COL)

        finetune_interactions = []
        test_interactions = []

        hit_user_set = set()
        hit_item_set = set()
        for uid, df in interactions:
            hit_sets = [set(), set()]

            for item, click in zip(df[self.IID_COL], df[self.LBL_COL]):
                if item in cold_items:
                    hit_sets[click].add(item)
            if len(hit_sets[0]) == 0 or len(hit_sets[1]) == 0:
                continue

            for hit_set, click in zip(hit_sets, [0, 1]):
                for item in hit_set:
                    test_interactions.append([uid, item, click])

            hit_item_set.update(hit_sets[0])
            hit_item_set.update(hit_sets[1])
            hit_user_set.add(uid)

            if len(test_interactions) > self.NUM_TEST:
                break

        final_test_count = len(test_interactions) // 10 * 11

        for uid, df in interactions:
            if uid not in hit_user_set:
                continue

            click_count = 0
            for item, click in zip(df[self.IID_COL], df[self.LBL_COL]):
                if item not in cold_items and click_count < 5:
                    finetune_interactions.append([uid, item, click])
                    click_count += 1

        for uid, df in interactions:
            if uid in hit_user_set:
                continue
            item_sets = [set(), set()]
            for item, click in zip(df[self.IID_COL], df[self.LBL_COL]):
                if item not in hit_item_set:
                    item_sets[click].add(item)

            if len(item_sets[0]) == 0 or len(item_sets[1]) == 0:
                continue

            if len(test_interactions) < final_test_count:
                current_interactions = test_interactions
            else:
                current_interactions = finetune_interactions

            for item_set, click in zip(item_sets, [0, 1]):
                for item in list(item_set)[:5]:
                    current_interactions.append([uid, item, click])

            if len(finetune_interactions) > self.NUM_FINETUNE:
                break

        self.test_set = pd.DataFrame(test_interactions, columns=[self.UID_COL, self.IID_COL, self.LBL_COL])
        self.finetune_set = pd.DataFrame(finetune_interactions, columns=[self.UID_COL, self.IID_COL, self.LBL_COL])

        self.test_set.to_parquet(os.path.join(self.store_dir, 'test.parquet'))
        pnt(f'generated test set with {len(self.test_set)}/{self.NUM_TEST} samples')

        self.finetune_set.to_parquet(os.path.join(self.store_dir, 'finetune.parquet'))
        pnt(f'generated finetune set with {len(self.finetune_set)}/{self.NUM_FINETUNE} samples')

        self._loaded = True
