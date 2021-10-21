from typing import Dict

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class Pooling(nn.Module):
    def __init__(self, mode: str):
        super().__init__()
        assert mode in {'mean', 'max', 'cls'}, 'Unknown pooling mode: {}'.format(mode)
        self.mode = mode

    def forward(self, token_embeddings, attention_mask) -> torch.Tensor:
        if self.mode == 'cls':
            return token_embeddings[:, 0]
        if self.mode == 'mean':
            expanded_attention_mask = attention_mask.unsqueeze(2).float()
            sum_embeddings = torch.sum(token_embeddings * expanded_attention_mask, 1)
            num_tokens = torch.sum(expanded_attention_mask, 1)
            return sum_embeddings / torch.clamp_min(num_tokens, 1e-8)
        if self.mode == 'max':
            expanded_attention_mask = attention_mask.unsqueeze(2)
            token_embeddings = torch.where(expanded_attention_mask, token_embeddings,
                                           torch.full_like(token_embeddings, float('-inf')))
            return torch.max(token_embeddings, 1)[0]


class SentenceEmbedder(nn.Module):
    def __init__(
            self,
            pretrained_model: str,
            pooling_mode: str,
    ):
        super().__init__()

        self.transformer = AutoModel.from_pretrained(pretrained_model)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.embedding_dim = self.transformer.config.hidden_size

        self.pooling = Pooling(pooling_mode)

    def forward(self, input: Dict) -> torch.Tensor:
        output = self.transformer(**input)
        token_embeddings = output[0]
        attention_mask = input['attention_mask']
        return self.pooling(token_embeddings, attention_mask)


class SentenceClassifier(nn.Module):
    """
    The model for sentence classification.
    It contains a transformer model as the token encoder,
    compute sentence representation from token representation using different pooling modes
    (``cls``, ``mean`` or ``max``) and classify sentences using a linear classifier.
    """

    def __init__(
            self,
            pretrained_model: str,
            pooling_mode: str,
            output_dim: int,
            use_extended_features: bool = False,
            dropout_prob: float = 0.1,
    ):
        super().__init__()

        self.sentence_embedder = SentenceEmbedder(pretrained_model, pooling_mode)
        self.tokenizer = self.sentence_embedder.tokenizer

        self.embedding_dim = self.sentence_embedder.embedding_dim
        self.output_dim = output_dim

        self.use_extended_features = use_extended_features

        input_dim = self.embedding_dim
        if use_extended_features:
            input_dim += 5
        self.projection_layer = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(input_dim, self.output_dim),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input: Dict) -> Dict:
        embeddings = self.sentence_embedder(input['encoded_inputs'])
        if self.use_extended_features:
            embeddings = torch.cat([
                embeddings,
                torch.stack([input['xmin'], input['xmax'], input['ymin'], input['ymax'], input['position']], dim=1),
            ], dim=1)
        logits = self.projection_layer(embeddings)
        probabilities = self.softmax(logits)
        predictions = torch.argmax(logits, dim=-1)
        return {
            'logit': logits,
            'probability': probabilities,
            'prediction': predictions,
        }
