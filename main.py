import json
import logging
import os
from typing import List, OrderedDict

import pandas as pd
import torch

from sentclf.data import read_json_file
from sentclf.modules import SentenceClassifier
from sentclf.system import SentenceClassificationSystem

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s', level='INFO')

MODEL_DIR = 'challenge/model/Uizard_211020_17-13-10'
DEVICE = 'cpu'
BATCH_SIZE = 64


def read_model(model_dir):
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f, object_hook=OrderedDict)

    ckpt_path = os.path.join(model_dir, 'best.pth')
    checkpoint = torch.load(ckpt_path, map_location=torch.device(DEVICE))
    logger.info('Loaded model from {}'.format(ckpt_path))

    model = SentenceClassifier(**config['model'])
    model.load_state_dict(checkpoint)

    system = SentenceClassificationSystem(model, **config['system'])
    labels = config['dataset']['labels']

    return system, labels


MODEL, LABELS = read_model(MODEL_DIR)


def predict(input_file_path: str) -> List[str]:
    file_content = read_json_file(input_file_path)
    df = pd.DataFrame(file_content)
    prediction = MODEL.predict(df, batch_size=BATCH_SIZE, device=DEVICE).tolist()
    predicted_labels = [LABELS[p] for p in prediction]
    return predicted_labels
