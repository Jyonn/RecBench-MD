from pigmento import pnt

from process.mind_processor import MINDProcessor
from process.goodreads_processor import GoodreadsProcessor
# from process.movielens20m_processor import MovieLens20MProcessor
from process.steam_processor import SteamProcessor

# processor = MINDProcessor()
# processor = MovieLens20MProcessor()
# processor = GoodreadsProcessor()
processor = SteamProcessor()
processor.load_public_sets()

test_set_user = len(processor.test_set[processor.UID_COL].unique())
fine_tune_set_user = len(processor.finetune_set[processor.UID_COL].unique())

pnt(f'Test set: {len(processor.test_set)} with {test_set_user} users')
pnt(f'Finetune set: {len(processor.finetune_set)} with {fine_tune_set_user} users')
