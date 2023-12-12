from typing import Literal

import torch
from tsl.nn.layers import NodeEmbedding

EmbeddingType = Literal['trainable', 'fixed', 'none']


def find_embedding(model: torch.nn.Module):
    embs = [m for m in model.modules() if isinstance(m, NodeEmbedding)]
    n = len(embs)
    if n == 1:
        return embs[0]
    elif n == 0:
        return None
    else:
        raise RuntimeError(f"Looking for a single embedding module, {n} found")
