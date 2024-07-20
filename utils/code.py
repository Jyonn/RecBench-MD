import json
from typing import cast

import numpy as np

# with open('.code') as f:
#     code = f.read()
#
# code_paths = {}
# for line in code.strip().split('\n'):
#     name, key = line.split('=')
#     code_paths[name.strip()] = key.strip()


def get_code_embeds(code_path):
    # code_path = code_paths.get(dataset, None)
    # if code_path is None:
    #     return None
    return cast(dict, np.load(code_path, allow_pickle=True).item())


max_code_index = 0


def get_code_indices(code_path):
    global max_code_index

    def get_global_indexer(num_codes):
        def get_global_index(depth, local_index):
            global max_code_index
            global_index = 0
            for i in range(depth):
                global_index += num_codes[i]
            index = global_index + local_index
            if index > max_code_index:
                max_code_index = index
            return index
        return get_global_index

    # code_path = code_paths.get(dataset, None)
    # if code_path is None:
    #     raise ValueError(f'code indices for {dataset} not found')
    indices = json.load(open(code_path))

    num_depth = 0
    num_codes = []

    for iid in indices:
        codes = indices[iid]
        num_depth = max(num_depth, len(codes))
        for i in range(len(num_codes), len(codes)):
            num_codes.append(0)

        for i in range(len(codes)):
            num_codes[i] = max(num_codes[i], codes[i] + 1)

    global_indexer = get_global_indexer(num_codes)

    for iid in indices:
        codes = indices[iid]
        for i, code in enumerate(codes):
            indices[iid][i] = global_indexer(i, code)

    print('load codes')
    print(num_codes, max_code_index)
    return indices, sum(num_codes)
