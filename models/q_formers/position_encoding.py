import torch
import torch.nn as nn


class PositionEmbeddings(nn.Module):
    def __init__(self, max_position_embeddings, hidden_size, eps=1e-12, dropout=0.1, inplace=True):
        super().__init__()
        self.position_embeddings = nn.Embedding(
            max_position_embeddings, hidden_size
        )

        self.LayerNorm = nn.LayerNorm(hidden_size, eps=eps)
        self.dropout = nn.Dropout(dropout, inplace=inplace)

        self.register_buffer(
            "position_ids", torch.arange(max_position_embeddings).expand((1, -1))
        )

    def forward(self, embeddings, position_ids=None, offset=0):
        seq_length = embeddings.size()[1]

        if position_ids is None:
            position_ids = self.position_ids[:, offset:offset+seq_length].clone()

        position_embeddings = self.position_embeddings(position_ids)
        embeddings = embeddings + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class PositionScore(nn.Module):
    def __init__(self, seq_len, shape=None, score_type="gaussian"):
        assert seq_len is not None or shape is not None, "seq_len or shape must be provided"
        self.cls_token = False
        if seq_len is not None:
            h = w = int(seq_len ** 0.5)
        elif isinstance(shape, int):
            h = w = shape
        else:
            h, w = shape
        self.h = h
        self.w = w

    def forward(self, tensor):
        bs, chn, m, n = tensor.shape
