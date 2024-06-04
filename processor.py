from process.mind_processor import MINDProcessor

processor = MINDProcessor(
    data_dir='/data1/qijiong/Data/MIND/',
    store_dir='data',
)

count = 0
for uid, iid, history, candidate, click in processor.generate(max_len=20, item_attrs=['title']):
    # print(uid, iid, history, candidate, click)
    print(f'User: {uid}, Item: {iid}, History, Click: {click}')
    print(f'History:')
    for i, h in enumerate(history):
        print(f'\t{i:2d}: {h}')
    print(f'Candidate: {candidate}')

    count += 1
    if count > 10:
        break
