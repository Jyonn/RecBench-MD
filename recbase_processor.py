import argparse
import json
import os
from typing import cast

import pandas as pd
from unitok import Vocab
from tqdm import tqdm

from process.base_processor import BaseProcessor
from utils.data import get_data_dir
from utils.function import load_processor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--export', type=str, default='recbase')
    args = parser.parse_args()

    export_path = cast(str, os.path.join(args.export, args.data))
    os.makedirs(export_path, exist_ok=True)

    data_config = dict(
        pens=dict(
            prompt='Describe a news article:',
            attrs=dict(
                cat='Category',
                topic='Topic',
                title='Title',
                body='Body',
            )
        ),
        pog=dict(
            prompt='Describe a fashion product:',
            attrs=dict(
                title_en='Title',
            )
        ),
        netflix=dict(
            prompt='Describe a movie:',
            attrs=dict(
                title='Title',
                year='Year',
            )
        ),
        books=dict(
            prompt='Describe a book:',
            attrs=dict(
                title='Name',
            )
        ),
        yelp=dict(
            prompt='Describe a restaurant:',
            attrs=dict(
                name='Name',
                address='Address',
                city='City',
                state='State',
            )
        ),
        hotelrec=dict(
            prompt='Describe a hotel:',
            attrs=dict(
                hotel_name='Name',
                hotel_location='Location',
            )
        ),
        steam=dict(
            prompt='Describe a game:',
            attrs=dict(
                title='Title',
            )
        ),
    )

    data_config = data_config[args.data]
    data_dir = get_data_dir(args.data)

    processor = load_processor(args.data, data_dir=data_dir)  # type: BaseProcessor

    processor.items = processor.load_items()
    processor.items = processor._stringify(processor.items)
    items = processor.items
    item_vocab = Vocab(name='item')

    for attr in data_config['attrs']:
        # fillna with empty string
        items[attr] = items[attr].fillna('')

    processed_list = []
    # iterate over items
    for item in tqdm(items.iterrows(), total=len(items)):
        item = item[1]
        content = {data_config['attrs'][attr]: item[attr][:2000] for attr in data_config['attrs']}
        # i want my json string each attr split by \n
        string = json.dumps(content, ensure_ascii=False, indent=0)
        string = data_config['prompt'] + '\n' + string
        item_id = item[processor.IID_COL]
        processed_list.append(dict(
            item_id=item_vocab.append(item_id),
            item_text=string
        ))

    df = pd.DataFrame(processed_list)
    df.to_parquet(os.path.join(export_path, 'items.parquet'))

    processor.users = processor.load_users()
    processor.users = processor._stringify(processor.users)
    users = processor.users
    user_vocab = Vocab(name='user')

    processed_list = []
    # iterate over users
    for user in tqdm(users.iterrows(), total=len(users)):
        user = user[1]
        user_id = user[processor.UID_COL]
        history = user[processor.HIS_COL]
        history = [item_vocab.append(item) for item in history]
        processed_list.append(dict(
            user_id=user_vocab.append(user_id),
            history=history
        ))

    df = pd.DataFrame(processed_list)
    df.to_parquet(os.path.join(export_path, 'users.parquet'))
