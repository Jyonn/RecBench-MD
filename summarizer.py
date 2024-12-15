import pigmento
from pigmento import pnt

from process.base_processor import BaseProcessor
from seq_process.base_seqprocessor import BaseSeqProcessor
from utils.config_init import ConfigInit
from utils.function import load_seq_processor, load_processor

if __name__ == '__main__':
    pigmento.add_time_prefix()
    pnt.set_display_mode(
        use_instance_class=True,
        display_method_name=False
    )

    configuration = ConfigInit(
        required_args=['data'],
        default_args=dict(
            seq=0,
        ),
        makedirs=[],
    ).parse()

    if configuration.seq:
        processor = load_seq_processor(configuration.data)  # type: BaseSeqProcessor
    else:
        processor = load_processor(configuration.data)  # type: BaseProcessor
    processor.load()

    print(f'#Finetune: {len(processor.finetune_set)}')
    print(f'#Test: {len(processor.test_set)}')

    if configuration.seq:
        # appearance = dict()

        for set_, set_str in zip([processor.finetune_set, processor.test_set], ['Finetune', 'Test']):
            item_set = set()
            user_history_length = 0
            for history in set_[processor.HIS_COL]:
                user_history_length += min(len(history), 20)
                for item in history[-20:]:
                    item_set.add(item)
                    # appearance[item] = appearance.get(item, 0) + 1
            user_history_length /= len(set_)

            print(f'{set_str}:')
            print(f'Average user history length: {user_history_length:.2f}')
            print(f'#Items: {len(item_set)}')

        num_user = len(processor.finetune_set) + len(processor.test_set)
        print(f'#Users: {num_user}')
        # avg_appearance = sum(appearance.values()) / len(appearance)
        # print(f'Average item appearance: {avg_appearance:.2f}')

    else:
        for set_, set_str in zip([processor.finetune_set, processor.test_set], ['Finetune', 'Test']):
            item_set = set()
            user_set = set()
            for user_id in set_[processor.UID_COL]:
                user_set.add(user_id)
                user_id = processor.user_vocab[user_id]
                history = processor.users.iloc[user_id][processor.HIS_COL]
                for item in history[-20:]:
                    item_set.add(item)
            for item_id in set_[processor.IID_COL]:
                item_set.add(item_id)

            print(f'{set_str}:')
            print(f'#Items: {len(item_set)}')
            print(f'#Users: {len(user_set)}')
