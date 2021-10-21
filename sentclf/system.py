import logging
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from tqdm import trange

from .utils import batch_to_device

logger = logging.getLogger(__name__)


class SentenceClassificationSystem(object):
    """
    This class wrap outside a sentence classification model to provide the functionality of the model.
    """
    def __init__(
            self,
            model,
            max_sequence_length: int = 128,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = model.tokenizer
        self.max_sequence_length = max_sequence_length

    def tokenize(
            self,
            sentences: List[str]
    ) -> Dict:
        encoded_inputs = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=self.max_sequence_length,
            return_tensors='pt',
        )
        return encoded_inputs

    def predict(
            self,
            sentences: pd.DataFrame,
            batch_size: int = 64,
            device: str = None,
            show_progress_bar: bool = False,
    ) -> torch.Tensor:
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Use device: {device}')

        self.model.to(device)
        self.model.eval()

        all_outputs = []

        indices_sorted_length = np.argsort([len(sentence) for sentence in sentences['sentence']])

        with torch.no_grad():
            for start_index in trange(0, len(sentences), batch_size,
                                      desc='Predict', unit='batch', disable=not show_progress_bar):
                batch_indices = indices_sorted_length[start_index:start_index + batch_size]
                batch = sentences.iloc[batch_indices].to_dict(orient='list')
                batch = batch_to_device(batch, device)

                encoded_inputs = self.tokenize(batch['sentence'])
                encoded_inputs = batch_to_device(encoded_inputs, device)
                extended_inputs = {
                    'encoded_inputs': encoded_inputs,
                    'xmin': torch.tensor(batch['xmin'], device=device),
                    'xmax': torch.tensor(batch['xmax'], device=device),
                    'ymin': torch.tensor(batch['ymin'], device=device),
                    'ymax': torch.tensor(batch['ymax'], device=device),
                    'position': torch.tensor(batch['position'], device=device),
                }
                outputs = self.model(extended_inputs)
                prediction = outputs['prediction']
                all_outputs.extend(prediction)

        all_outputs = [all_outputs[idx] for idx in np.argsort(indices_sorted_length)]
        all_outputs = torch.stack(all_outputs)

        return all_outputs
