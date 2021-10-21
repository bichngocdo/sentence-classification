import json
import os
from typing import List

import pandas as pd
from tqdm import tqdm


def read_json_file(path: str) -> List:
    with open(path, 'r') as f:
        results = []
        sentences = json.load(f)
        for k, sentence in enumerate(sentences):
            result = {
                'file': path,
                'sentence': sentence['text'],
                'label': sentence['label'] if 'label' in sentence else -1,
                'position': k,
                'xmin': float('inf'),
                'xmax': float('-inf'),
                'ymin': float('inf'),
                'ymax': float('-inf'),
            }
            for token in sentence['tokens']:
                for dim, func in zip(['xmin', 'xmax', 'ymin', 'ymax'], [min, max, min, max]):
                    result[dim] = func(result[dim], int(token['box'][dim]))
            results.append(result)
        return results


def read_json_folder(path: str) -> pd.DataFrame:
    file_names = os.listdir(path)
    results = []
    for name in tqdm(file_names):
        file_path = os.path.join(path, name)
        if os.path.isfile(file_path):
            file_content = read_json_file(file_path)
            results += file_content
    df = pd.DataFrame(results)
    df['label'] = df['label'].astype('category')
    return df
