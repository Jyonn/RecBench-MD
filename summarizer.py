import pigmento
from pigmento import pnt

from loader.seq_preparer import SeqPreparer
from seq_process.base_seqprocessor import BaseSeqProcessor
from utils.config_init import ConfigInit
from utils.function import load_seq_processor

if __name__ == '__main__':
    pigmento.add_time_prefix()
    pnt.set_display_mode(
        use_instance_class=True,
        display_method_name=False
    )

    configuration = ConfigInit(
        required_args=['data'],
        default_args=dict(),
        makedirs=[],
    ).parse()

    processor = load_seq_processor(configuration.data)  # type: BaseSeqProcessor
    processor.load()

    print(f'#Finetune: {len(processor.finetune_set)}')
    print(f'#Test: {len(processor.test_set)}')

    # user history average length, by iterating processor.users a dataframe
    item_set = set()
    appearance = dict()

    for set_ in [processor.finetune_set, processor.test_set]:
        user_history_length = 0
        for history in set_[processor.HIS_COL]:
            user_history_length += min(len(history), 20)
            for item in history[-20:]:
                item_set.add(item)
                appearance[item] = appearance.get(item, 0) + 1
        user_history_length /= len(set_)
        print(f'Average user history length: {user_history_length:.2f}')

    print(f'#Items: {len(item_set)}')
    num_user = len(processor.finetune_set) + len(processor.test_set)
    print(f'#Users: {num_user}')
    avg_appearance = sum(appearance.values()) / len(appearance)
    print(f'Average item appearance: {avg_appearance:.2f}')
