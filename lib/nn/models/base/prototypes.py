from typing import Optional, Tuple, Union, List

import torch
from torch import Tensor, nn
from torch_geometric.typing import Adj
from tsl.nn.blocks import MLPDecoder
from tsl.nn.layers import MultiLinear
from tsl.nn.models import BaseModel
from tsl.nn.utils import maybe_cat_exog
from tsl.utils import ensure_list

from lib.nn.layers import MultiMLPDecoder
from lib.nn.utils import get_embedding_module, maybe_cat_emb, \
    reset_embedding_module


class STGNN(BaseModel):
    available_embedding_pos = {'encoding', 'decoding'}

    def __init__(self, input_size: int, horizon: int,
                 n_nodes: int = None,
                 output_size: int = None,
                 exog_size: int = 0,
                 hidden_size: int = 32,
                 embedding_cfg: dict = None,
                 add_embedding_before: Optional[
                     Union[str, List[str]]] = 'encoding',
                 use_local_weights: Union[str, List[str]] = None,
                 activation: str = 'elu'):
        super(STGNN, self).__init__()

        self.input_size = input_size
        self.horizon = horizon
        self.n_nodes = n_nodes
        self.output_size = output_size or input_size
        self.hidden_size = hidden_size
        self.exog_size = exog_size
        self.activation = activation

        # EMBEDDING
        if add_embedding_before is None:
            add_embedding_before = set()
        else:
            add_embedding_before = set(ensure_list(add_embedding_before))
            if not add_embedding_before.issubset(self.available_embedding_pos):
                raise ValueError("Parameter 'add_embedding_before' must be a "
                                 f"subset of {self.available_embedding_pos}")
        self.add_embedding_before = add_embedding_before

        if embedding_cfg is None:
            emb, emb_size = None, 0
        else:
            emb_size = embedding_cfg.get('emb_size', 0)
            emb = get_embedding_module(n_nodes=n_nodes, **embedding_cfg)

        if emb is not None:
            self.emb = emb
        else:
            self.register_module('emb', None)

        # ENCODER
        self.encoder_input = input_size + exog_size
        if 'encoding' in self.add_embedding_before and self.emb is not None:
            self.encoder_input += emb_size

        if use_local_weights is not None:
            self.use_local_weights = set(ensure_list(use_local_weights))
            if len(self.use_local_weights.difference(['encoder', 'decoder'])):
                raise ValueError("Parameter 'use_local_weights' must be "
                                 "'encoder', 'decoder', or both.")
        else:
            self.use_local_weights = set()

        if 'encoder' in self.use_local_weights:
            self.encoder = MultiLinear(self.encoder_input, hidden_size, n_nodes)
        else:
            self.encoder = nn.Linear(self.encoder_input, hidden_size)

        # DECODER
        self.decoder_input = hidden_size
        if 'decoding' in self.add_embedding_before and self.emb is not None:
            self.decoder_input += emb_size
        if 'decoder' in self.use_local_weights:
            self.decoder = MultiMLPDecoder(input_size=self.decoder_input,
                                           n_instances=n_nodes,
                                           hidden_size=self.hidden_size,
                                           output_size=self.output_size,
                                           horizon=self.horizon,
                                           activation=self.activation)
        else:
            self.decoder = MLPDecoder(input_size=self.decoder_input,
                                      hidden_size=self.hidden_size,
                                      output_size=self.output_size,
                                      horizon=self.horizon,
                                      activation=self.activation)

    @torch.no_grad()
    def reset_embeddings(self, method: str, n_nodes: int = None,
                         emb_size: int = None,
                         trainable: bool = True, **kwargs):
        return reset_embedding_module(self, method, n_nodes, emb_size,
                                      trainable, **kwargs)

    def reset_local_layers(self, n_nodes=None):
        self.n_nodes = n_nodes or self.n_nodes
        if 'encoder' in self.use_local_weights:
            self.encoder = MultiLinear(in_channels=self.encoder_input,
                                       out_channels=self.hidden_size,
                                       n_instances=self.n_nodes)
        if 'decoder' in self.use_local_weights:
            self.decoder = MultiMLPDecoder(input_size=self.decoder_input,
                                           n_instances=self.n_nodes,
                                           hidden_size=self.hidden_size,
                                           output_size=self.output_size,
                                           horizon=self.horizon,
                                           activation=self.activation)

    def stmp(self, x: Tensor, edge_index: Adj,
             edge_weight: Optional[Tensor] = None,
             emb: Optional[Tensor] = None) -> Tensor:
        raise NotImplementedError

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: Optional[Tensor] = None,
                u: Optional[Tensor] = None,
                node_idx: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """"""
        # x: [batches steps nodes features]
        x = maybe_cat_exog(x, u)
        batch_size = x.size(0)
        emb = self.emb(expand=(batch_size, -1, -1),
                       node_index=node_idx) if self.emb is not None else None

        if 'encoding' in self.add_embedding_before and emb is not None:
            x = maybe_cat_emb(x, emb[:, None])

        # ENCODER   ###########################################################
        x = self.encoder(x)

        # SPATIOTEMPORAL MESSAGE-PASSING   ####################################
        out = self.stmp(x, edge_index, edge_weight, emb)

        # DECODER   ###########################################################
        if 'decoding' in self.add_embedding_before:
            out = maybe_cat_emb(out, emb)

        out = self.decoder(out)

        return out, emb


class TimeThenSpace(STGNN):
    available_embedding_pos = {'encoding', 'message_passing', 'decoding'}

    def __init__(self, input_size: int, horizon: int,
                 temporal_encoder: nn.Module,
                 spatial_encoder: Union[nn.Module, List[nn.Module]],
                 n_nodes: int = None,
                 output_size: int = None,
                 exog_size: int = 0,
                 hidden_size: int = 32,
                 embedding_cfg: dict = None,
                 add_embedding_before: Union[str, List[str]] = 'encoding',
                 use_local_weights: Union[str, List[str]] = None,
                 activation: str = 'elu'):
        super(TimeThenSpace, self).__init__(input_size=input_size,
                                            horizon=horizon,
                                            n_nodes=n_nodes,
                                            output_size=output_size,
                                            exog_size=exog_size,
                                            hidden_size=hidden_size,
                                            embedding_cfg=embedding_cfg,
                                            add_embedding_before=add_embedding_before,
                                            use_local_weights=use_local_weights,
                                            activation=activation)
        # STMP
        self.temporal_encoder = temporal_encoder
        if not isinstance(spatial_encoder, nn.ModuleList):
            spatial_encoder = nn.ModuleList(ensure_list(spatial_encoder))
        self.mp_layers = spatial_encoder
        self.spatial_layers = len(self.mp_layers)

    def stmp(self, x: Tensor, edge_index: Adj,
             edge_weight: Optional[Tensor] = None,
             emb: Optional[Tensor] = None) -> Tensor:
        # temporal encoding
        out = self.temporal_encoder(x)
        # spatial encoding
        for layer in self.mp_layers:
            if 'message_passing' in self.add_embedding_before:
                out = maybe_cat_emb(out, emb)
            out = layer(out, edge_index, edge_weight)
        return out


class TimeAndSpace(STGNN):

    def __init__(self, input_size: int, horizon: int, stmp_conv: nn.Module,
                 n_nodes: int = None,
                 output_size: int = None,
                 exog_size: int = 0,
                 hidden_size: int = 32,
                 embedding_cfg: dict = None,
                 add_embedding_before: Union[str, List[str]] = 'encoding',
                 use_local_weights: Union[str, List[str]] = None,
                 activation: str = 'elu'):
        super(TimeAndSpace, self).__init__(input_size=input_size,
                                           horizon=horizon,
                                           n_nodes=n_nodes,
                                           output_size=output_size,
                                           exog_size=exog_size,
                                           hidden_size=hidden_size,
                                           embedding_cfg=embedding_cfg,
                                           add_embedding_before=add_embedding_before,
                                           use_local_weights=use_local_weights,
                                           activation=activation)

        # STMP
        self.stmp_conv = stmp_conv

    def stmp(self, x: Tensor, edge_index: Adj,
             edge_weight: Optional[Tensor] = None,
             emb: Optional[Tensor] = None) -> Tensor:
        # spatiotemporal encoding
        out = self.stmp_conv(x, edge_index, edge_weight)
        return out
