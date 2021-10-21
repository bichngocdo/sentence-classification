import os
from typing import List

import pandas as pd
from tqdm import tqdm

from sentclf.data import read_json_file


def read_json_file_contextual(path: str, max_length=50) -> List:
    content = read_json_file(path)
    if len(content) == 0:
        return None
    content = pd.DataFrame(content).to_dict(orient='list')
    results = []
    for i in range(0, len(content), max_length):
        result = {}
        for k, v in content.items():
            result['{}s'.format(k)] = v[i:i + max_length]
        result['files'] = result['files'][0]
        results.append(result)
    return results


def read_json_folder_contextual(path: str) -> pd.DataFrame:
    file_names = os.listdir(path)
    results = []
    for name in tqdm(file_names):
        file_path = os.path.join(path, name)
        if os.path.isfile(file_path):
            file_content = read_json_file_contextual(file_path)
            results += file_content
    df = pd.DataFrame(results)
    return df
