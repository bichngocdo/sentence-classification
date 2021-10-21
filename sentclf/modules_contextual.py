from typing import Dict

import torch
import torch.nn as nn

from torchcrf import CRF
from .modules import SentenceEmbedder


class ContextualSentenceClassifier(nn.Module):
    def __init__(
            self,
            pretrained_model: str,
            pooling_mode: str,
            hidden_dim: int,
            output_dim: int,
            num_lstms: int,
            use_extended_features: bool = False,
            dropout_prob: float = 0.1,
    ):
        super().__init__()

        self.sentence_embedder = SentenceEmbedder(pretrained_model, pooling_mode)
        self.tokenizer = self.sentence_embedder.tokenizer

        self.embedding_dim = self.sentence_embedder.embedding_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.use_extended_features = use_extended_features

        input_dim = self.embedding_dim
        if use_extended_features:
            input_dim += 5

        self.context_lstms = nn.LSTM(input_dim, self.hidden_dim, num_lstms,
                                     batch_first=True, bidirectional=True, dropout=dropout_prob)
        self.projection_layer = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(2 * hidden_dim, self.output_dim),
        )
        self.crf = CRF(output_dim, batch_first=True)

    def forward(self, input: Dict) -> Dict:
        embeddings = self.sentence_embedder(input['encoded_inputs'])
        if self.use_extended_features:
            embeddings = torch.cat([
                embeddings,
                torch.stack([input['xmin'], input['xmax'], input['ymin'], input['ymax'], input['position']], dim=1),
            ], dim=1)
        embeddings = embeddings.unsqueeze(0)
        device = embeddings.device

        hidden, _ = self.context_lstms(embeddings)
        logits = self.projection_layer(hidden)

        predictions = torch.tensor(self.crf.decode(logits), dtype=torch.long, device=device)
        return {
            'logit': logits[0],
            'prediction': predictions[0],
        }

    def loss(self, logits, targets):
        return -self.crf(logits.unsqueeze(0), targets.unsqueeze(0))
