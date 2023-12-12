from typing import Union, List, Optional

from torch import Tensor
from torch_geometric.typing import Adj
from tsl.nn.blocks import LinearReadout
from tsl.nn.blocks.encoders import AGCRN

from lib.nn.layers import MultiMLPDecoder
from lib.nn.models import TimeAndSpace


class AGCRNModel(TimeAndSpace):
    r"""The Adaptive Graph Convolutional Recurrent Network from the paper
    `"Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting"
    <https://arxiv.org/abs/2007.02842>`_ (Bai et al., NeurIPS 2020).

    Args:
        input_size (int): Number of features of the input sample.
        output_size (int): Number of output channels.
        horizon (int): Number of future time steps to forecast.
        exog_size (int): Number of features of the input covariate, if any.
        hidden_size (int): Number of hidden units.
        hidden_size (int): Size of the learned node embeddings.
        n_nodes (int): Number of nodes in the input (static) graph.
        n_layers (int): Number of AGCRN cells.
            (default: :obj:`1`)
   """

    def __init__(self, input_size: int, horizon: int, n_nodes: int = None,
                 output_size: int = None,
                 exog_size: int = 0,
                 hidden_size: int = 64,
                 emb_size: int = 10,
                 n_layers: int = 1,
                 embedding_cfg: dict = None,
                 add_embedding_before: Union[str, List[str]] = 'encoding',
                 use_local_weights: Union[str, List[str]] = None,
                 activation: str = 'elu'):
        agcrn = AGCRN(input_size=hidden_size,
                      emb_size=emb_size,
                      num_nodes=n_nodes,
                      hidden_size=hidden_size,
                      n_layers=n_layers,
                      return_only_last_state=True)
        super(AGCRNModel, self).__init__(
            input_size=input_size,
            horizon=horizon,
            stmp_conv=agcrn,
            n_nodes=n_nodes,
            output_size=output_size,
            exog_size=exog_size,
            hidden_size=hidden_size,
            embedding_cfg=embedding_cfg,
            add_embedding_before=add_embedding_before,
            use_local_weights=use_local_weights,
            activation=activation
        )

        # Override DECODER for linear readout as original architecture
        if 'decoder' in self.use_local_weights:
            self.decoder = MultiMLPDecoder(input_size=self.decoder_input,
                                           n_instances=n_nodes,
                                           hidden_size=0,
                                           n_layers=0,
                                           output_size=self.output_size,
                                           horizon=self.horizon)
        else:
            self.decoder = LinearReadout(input_size=self.decoder_input,
                                         output_size=self.output_size,
                                         horizon=self.horizon)

    def reset_local_layers(self, n_nodes=None):
        super().reset_local_layers(n_nodes)
        if 'decoder' in self.use_local_weights:
            self.decoder = MultiMLPDecoder(input_size=self.decoder_input,
                                           n_instances=self.n_nodes,
                                           hidden_size=0,
                                           n_layers=0,
                                           output_size=self.output_size,
                                           horizon=self.horizon)

    def stmp(self, x: Tensor, edge_index: Adj,
             edge_weight: Optional[Tensor] = None,
             emb: Optional[Tensor] = None) -> Tensor:
        # spatiotemporal encoding
        out = self.stmp_conv(x)
        return out
