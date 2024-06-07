# from process.mind_processor import MINDProcessor
from process.goodreads_processor import GoodreadsSamplingProcessor

# processor = MINDProcessor(
#     data_dir='/data1/qijiong/Data/MIND/',
# )
processor = GoodreadsSamplingProcessor(
    data_dir='/data_8T2/qijiong/Data/Goodreads/',
)
processor.load()

# processor.load_public_sets()

# exit(0)

count = 0
for uid, iid, history, candidate, click in processor.iterate(max_len=20):
    # print(uid, iid, history, candidate, click)
    print(f'User: {uid}, Item: {iid}, History, Click: {click}')
    print(f'History:')
    for i, h in enumerate(history):
        print(f'\t{i:2d}: {h}')
    print(f'Candidate: {candidate}')

    count += 1
    if count > 10:
        break
