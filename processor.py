from process.microlens_processor import MicroLensProcessor
from process.mind_processor import MINDProcessor
from process.goodreads_processor import GoodreadsProcessor
from process.movielens_processor import MovieLensProcessor
from process.movielens20m_processor import MovieLens20MProcessor
from process.steam_processor import SteamProcessor
from process.yelp_processor import YelpProcessor

# processor = MINDProcessor(
#     data_dir='/data1/qijiong/Data/MIND/',
# )
# processor = GoodreadsProcessor(
#     data_dir='/data_8T2/qijiong/Data/Goodreads/',
# )
# processor = MicroLensProcessor()
# processor = MovieLensProcessor()
# processor = MovieLens20MProcessor()
processor = SteamProcessor()
processor.load()

# processor.load_public_sets()

# exit(0)

# import pdb
# pdb.set_trace()

# count = 0
# for uid, iid, history, candidate, click in processor.iterate(max_len=20):
#     # print(uid, iid, history, candidate, click)
#     print(f'User: {uid}, Item: {iid}, History, Click: {click}')
#     print(f'History:')
#     for i, h in enumerate(history):
#         print(f'\t{i:2d}: {h}')
#     print(f'Candidate: {candidate}')
#
#     count += 1
#     if count > 10:
#         break
