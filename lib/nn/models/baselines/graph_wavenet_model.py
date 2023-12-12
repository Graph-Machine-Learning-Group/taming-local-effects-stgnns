from typing import Optional, Union, List

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.typing import Adj, OptTensor
from tsl.nn.blocks import MLPDecoder
from tsl.nn.blocks.encoders import TemporalConvNet
from tsl.nn.layers.base import NodeEmbedding
from tsl.nn.layers.graph_convs import DenseGraphConvOrderK, DiffConv
from tsl.nn.layers.norm import Norm

from lib.nn.layers import MultiMLPDecoder
from lib.nn.models import TimeAndSpace


class GraphWaveNetLayer(nn.Module):
    r"""The Graph WaveNet model from the paper `"Graph WaveNet for Deep
    Spatial-Temporal Graph Modeling" <https://arxiv.org/abs/1906.00121>`_
    (Wu et al., IJCAI 2019).

    Args:
        input_size (int): Number of features of the input sample.
        output_size (int): Number of output channels.
        horizon (int): Number of future time steps to forecast.
        exog_size (int): Number of features of the input covariate,
            if any. (default: :obj:`0`)
        hidden_size (int): Number of hidden units.
            (default: :obj:`32`)
        ff_size (int): Number of units in the nonlinear readout.
            (default: :obj:`256`)
        n_layers (int): Number of Graph WaveNet blocks.
            (default: :obj:`8`)
        temporal_kernel_size (int): Size of the temporal convolution kernel.
            (default: :obj:`2`)
        spatial_kernel_size (int): Order of the spatial diffusion process.
            (default: :obj:`2`)
        learned_adjacency (bool):  If :obj:`True`, then consider an additional
            learned adjacency matrix.
            (default: :obj:`True`)
        n_nodes (int, optional): Number of nodes in the input graph, required
            only when :attr:`learned_adjacency` is :obj:`True`.
            (default: :obj:`None`)
        emb_size (int): Number of features in the node embeddings used for
            graph learning.
            (default: :obj:`10`)
        dilation (int): Dilation of the temporal convolutional kernels.
            (default: :obj:`2`)
        dilation_mod (int): Length of the cycle for the dilation coefficient.
            (default: :obj:`2`)
        norm (str): Normalization strategy.
            (default: :obj:`'batch'`)
        dropout (float): Dropout probability.
            (default: :obj:`0.3`)
    """

    return_type = Tensor

    def __init__(self,
                 hidden_size: int = 32,
                 ff_size: int = 256,
                 n_layers: int = 8,
                 temporal_kernel_size: int = 2,
                 spatial_kernel_size: int = 2,
                 learned_adjacency: bool = True,
                 n_nodes: Optional[int] = None,
                 emb_size: int = 10,
                 dilation: int = 2,
                 dilation_mod: int = 2,
                 norm: str = 'batch',
                 dropout: float = 0.3):
        super().__init__()

        if learned_adjacency:
            assert n_nodes is not None
            self.source_embeddings = NodeEmbedding(n_nodes, emb_size)
            self.target_embeddings = NodeEmbedding(n_nodes, emb_size)
        else:
            self.register_parameter('source_embedding', None)
            self.register_parameter('target_embedding', None)

        temporal_conv_blocks = []
        spatial_convs = []
        skip_connections = []
        norms = []
        receptive_field = 1
        for i in range(n_layers):
            d = dilation ** (i % dilation_mod)
            temporal_conv_blocks.append(
                TemporalConvNet(input_channels=hidden_size,
                                hidden_channels=hidden_size,
                                kernel_size=temporal_kernel_size,
                                dilation=d,
                                exponential_dilation=False,
                                n_layers=1,
                                causal_padding=False,
                                gated=True))

            spatial_convs.append(
                DiffConv(in_channels=hidden_size,
                         out_channels=hidden_size,
                         k=spatial_kernel_size))

            skip_connections.append(nn.Linear(hidden_size, ff_size))
            norms.append(Norm(norm, hidden_size))
            receptive_field += d * (temporal_kernel_size - 1)
        self.tconvs = nn.ModuleList(temporal_conv_blocks)
        self.sconvs = nn.ModuleList(spatial_convs)
        self.skip_connections = nn.ModuleList(skip_connections)
        self.norms = nn.ModuleList(norms)
        self.dropout: nn.Module = nn.Dropout(dropout)

        self.receptive_field = receptive_field

        dense_sconvs = []
        if learned_adjacency:
            for _ in range(n_layers):
                dense_sconvs.append(
                    DenseGraphConvOrderK(input_size=hidden_size,
                                         output_size=hidden_size,
                                         support_len=1,
                                         order=spatial_kernel_size,
                                         include_self=False,
                                         channel_last=True))
        self.dense_sconvs = nn.ModuleList(dense_sconvs)

    def get_learned_adj(self):
        logits = F.relu(self.source_embeddings() @ self.target_embeddings().T)
        adj = torch.softmax(logits, dim=1)
        return adj

    def forward(self,
                x: Tensor,
                edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.receptive_field > x.size(1):
            # pad temporal dimension
            x = F.pad(x, (0, 0, 0, 0, self.receptive_field - x.size(1), 0))

        if len(self.dense_sconvs):
            adj_z = self.get_learned_adj()

        out = torch.zeros(1, x.size(1), 1, 1, device=x.device)
        for i, (tconv, sconv, skip_conn, norm) in enumerate(
                zip(self.tconvs, self.sconvs, self.skip_connections,
                    self.norms)):
            res = x
            # temporal conv
            x = tconv(x)
            # residual connection -> out
            out = skip_conn(x) + out[:, -x.size(1):]
            # spatial conv
            xs = sconv(x, edge_index, edge_weight)
            if len(self.dense_sconvs):
                x = xs + self.dense_sconvs[i](x, adj_z)
            else:
                x = xs
            x = self.dropout(x)
            # residual connection -> next layer
            x = x + res[:, -x.size(1):]
            x = norm(x)

        return out[:, -1]


class GraphWaveNetModel(TimeAndSpace):
    def __init__(self, input_size: int, horizon: int, n_nodes: int = None,
                 output_size: int = None,
                 exog_size: int = 0,
                 hidden_size: int = 32,
                 ff_size: int = 256,
                 n_layers: int = 8,
                 temporal_kernel_size: int = 2,
                 spatial_kernel_size: int = 2,
                 learned_adjacency: bool = True,
                 emb_size: int = 10,
                 dilation: int = 2,
                 dilation_mod: int = 2,
                 norm: str = 'batch',
                 dropout: float = 0.3,
                 embedding_cfg: dict = None,
                 add_embedding_before: Union[str, List[str]] = 'encoding',
                 use_local_weights: Union[str, List[str]] = None,
                 activation: str = 'elu'):
        gwnet = GraphWaveNetLayer(hidden_size=hidden_size,
                                  ff_size=ff_size,
                                  n_layers=n_layers,
                                  temporal_kernel_size=temporal_kernel_size,
                                  spatial_kernel_size=spatial_kernel_size,
                                  learned_adjacency=learned_adjacency,
                                  n_nodes=n_nodes,
                                  emb_size=emb_size,
                                  dilation=dilation,
                                  dilation_mod=dilation_mod,
                                  norm=norm,
                                  dropout=dropout)
        super(GraphWaveNetModel, self).__init__(
            input_size=input_size,
            horizon=horizon,
            stmp_conv=gwnet,
            n_nodes=n_nodes,
            output_size=output_size,
            exog_size=exog_size,
            hidden_size=hidden_size,
            embedding_cfg=embedding_cfg,
            add_embedding_before=add_embedding_before,
            use_local_weights=use_local_weights,
            activation=activation
        )

        # Override DECODER for doubling hidden_size as original architecture
        self.ff_size = ff_size
        self.decoder_input = ff_size
        if 'decoding' in self.add_embedding_before and self.emb is not None:
            self.decoder_input += self.emb.emb_size
        if 'decoder' in self.use_local_weights:
            decoder = MultiMLPDecoder(input_size=self.decoder_input,
                                      n_instances=n_nodes,
                                      hidden_size=2 * self.ff_size,
                                      output_size=self.output_size,
                                      horizon=self.horizon,
                                      activation='relu')
        else:
            decoder = MLPDecoder(input_size=self.decoder_input,
                                 hidden_size=2 * self.ff_size,
                                 output_size=self.output_size,
                                 horizon=self.horizon,
                                 activation='relu')
        self.decoder = nn.Sequential(nn.ReLU(), decoder)

    def reset_local_layers(self, n_nodes=None):
        super().reset_local_layers(n_nodes)
        if 'decoder' in self.use_local_weights:
            self.decoder = MultiMLPDecoder(input_size=self.decoder_input,
                                           n_instances=self.n_nodes,
                                           hidden_size=2 * self.ff_size,
                                           output_size=self.output_size,
                                           horizon=self.horizon,
                                           activation='relu')
