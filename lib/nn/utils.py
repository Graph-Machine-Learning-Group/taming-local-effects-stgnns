from typing import Optional

import torch
from torch import Tensor
from tsl.nn.layers import NodeEmbedding

from lib.nn.layers.embeddings import (VariationalNodeEmbedding,
                                      ClusterizedNodeEmbedding)


def get_embedding_module(method: str, n_nodes: int = None, emb_size: int = None,
                         trainable: bool = True, **kwargs):
    if method == 'none':
        return None
    assert n_nodes is not None, \
        "'n_nodes' cannot be None if method is not 'none'"
    assert emb_size is not None, \
        "'emb_size' cannot be None if method is not 'none'"
    if method == 'uniform':
        return NodeEmbedding(n_nodes, emb_size, requires_grad=trainable)
    elif method == 'clusterized':
        return ClusterizedNodeEmbedding(n_nodes, emb_size,
                                        requires_grad=trainable, **kwargs)
    elif method == 'variational':
        initial_var = kwargs.get('initial_var', 0.2)
        return VariationalNodeEmbedding(n_nodes, emb_size, initial_var)
    else:
        raise NotImplementedError()


@torch.no_grad()
def reset_embedding_module(model, method: str, n_nodes: int = None,
                           emb_size: int = None,
                           trainable: bool = True, **kwargs):
    if method == 'none':
        model.emb = None
        return

    assert n_nodes is not None, \
        "'n_nodes' cannot be None if method is not 'none'"
    assert emb_size is not None, \
        "'emb_size' cannot be None if method is not 'none'"

    if method == 'uniform':
        model.emb = NodeEmbedding(n_nodes, emb_size, requires_grad=trainable)
    elif method == 'variational':
        initial_var = kwargs.get('initial_var', 0.2)
        model.emb = VariationalNodeEmbedding(n_nodes, emb_size, initial_var)
    elif method == 'clusterized':
        n_clusters = kwargs.pop('n_clusters')
        emb = ClusterizedNodeEmbedding(n_nodes=n_nodes, emb_size=emb_size,
                                       n_clusters=n_clusters,
                                       requires_grad=trainable, **kwargs)
        if isinstance(model.emb, ClusterizedNodeEmbedding):
            assert model.emb.n_clusters == n_clusters
            emb.centroids.data.copy_(model.emb.centroids.data)
            emb.centroids.requires_grad = False
            emb.freeze_centroids()
        model.emb = emb

    else:
        raise NotImplementedError()


def maybe_cat_emb(x: Tensor, emb: Optional[Tensor]):
    if emb is None:
        return x
    if emb.ndim < x.ndim:
        emb = emb[[None] * (x.ndim - emb.ndim)]
    emb = emb.expand(*x.shape[:-1], -1)
    return torch.cat([x, emb], dim=-1)
