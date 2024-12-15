import json

import pigmento

from pigmento import pnt

from utils.config_init import ConfigInit


if __name__ == '__main__':
    pigmento.add_time_prefix()
    pnt.set_display_mode(
        use_instance_class=True,
        display_method_name=False
    )

    configuration = ConfigInit(
        required_args=['code_path'],
        default_args=dict(),
        makedirs=[]
    ).parse()

    indices = json.load(open(configuration.code_path))

    code_set = set()
    for _, value in indices.items():
        code_set.add('-'.join(map(str, value)))

    pnt(f'code set size: {len(code_set)}')
    pnt(f'indices size: {len(indices)}')
