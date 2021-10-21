import argparse
import json
import os
from collections import OrderedDict

import pandas as pd
import torch
from tqdm import tqdm

from sentclf.data import read_json_file
from sentclf.modules import SentenceClassifier
from sentclf.system import SentenceClassificationSystem

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='run_model.py',
        description='Running script for scientific paper sentence classification (SPSC) model'
    )
    parser.add_argument('--model', type=str, required=True,
                        help='model dir')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='data folder of json files')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size in number of sentences')
    parser.add_argument('--device', type=str, required=False,
                        help='device')

    args = parser.parse_args()
    print(args)

    device = args.device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config_path = os.path.join(args.model, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f, object_hook=OrderedDict)

    ckpt_path = os.path.join(args.model, 'best.pth')
    checkpoint = torch.load(ckpt_path, map_location=torch.device(device))

    model = SentenceClassifier(**config['model'])
    model.load_state_dict(checkpoint)

    system = SentenceClassificationSystem(model, **config['system'])
    label_map = {k: label for k, label in enumerate(config['dataset']['labels'])}

    file_names = os.listdir(args.data_dir)
    results = []
    for name in tqdm(file_names):
        file_path = os.path.join(args.data_dir, name)
        if os.path.isfile(file_path):
            file_content = read_json_file(file_path)
            df = pd.DataFrame(file_content)
            prediction = system.predict(df, batch_size=args.batch_size).tolist()
            predicted_labels = [label_map[p] for p in prediction]
            results.extend(predicted_labels)
    print(results)
