from typing import Union, List

from tsl.nn.blocks.encoders import RNN
from tsl.nn.layers import GraphConv, DiffConv
from tsl.nn.utils import get_functional_activation
from tsl.utils import ensure_list

from .prototypes import TimeThenSpace


class NonlinearDiffConv(DiffConv):
    def __init__(self, in_channels, out_channels, k,
                 root_weight: bool = True,
                 add_backward: bool = True,
                 bias: bool = True,
                 activation: str = 'relu'):
        super().__init__(in_channels, out_channels, k, root_weight,
                         add_backward, bias)
        self.activation = get_functional_activation(activation)

    def forward(self, x, edge_index,
                edge_weight=None, cache_support: bool = False):
        out = super().forward(x, edge_index, edge_weight, cache_support)
        return self.activation(out)


class TimeThenGraphIsoModel(TimeThenSpace):

    def __init__(self, input_size: int, horizon: int, n_nodes: int = None,
                 output_size: int = None,
                 exog_size: int = 0,
                 hidden_size: int = 32,
                 embedding_cfg: dict = None,
                 add_embedding_before: Union[str, List[str]] = 'encoding',
                 use_local_weights: Union[str, List[str]] = None,
                 time_layers: int = 1,
                 graph_layers: int = 1,
                 root_weight: bool = True,
                 norm: str = 'none',
                 add_backward: bool = False,
                 cached: bool = False,
                 activation: str = 'elu'):
        rnn = RNN(input_size=hidden_size,
                  hidden_size=hidden_size,
                  n_layers=time_layers,
                  return_only_last_state=True,
                  cell='gru')
        self.temporal_layers = time_layers

        if embedding_cfg is None:
            emb_size = 0
        else:
            emb_size = embedding_cfg.get('emb_size', 0)
        add_embedding_before = ensure_list(add_embedding_before)

        mp_kwargs = dict(root_weight=root_weight, activation=activation)
        if add_backward:
            assert norm == 'asym'
            mp_conv = NonlinearDiffConv
            mp_kwargs.update(k=1, add_backward=True)
        else:
            mp_conv = GraphConv
            mp_kwargs.update(norm=norm, cached=cached)
        mp_layers = [
            mp_conv(hidden_size + (emb_size if 'message_passing' in
                                               add_embedding_before
                                   else 0),
                    hidden_size,
                    **mp_kwargs)
            for _ in range(graph_layers)
        ]
        super(TimeThenGraphIsoModel, self).__init__(
            input_size=input_size,
            horizon=horizon,
            temporal_encoder=rnn,
            spatial_encoder=mp_layers,
            n_nodes=n_nodes,
            output_size=output_size,
            exog_size=exog_size,
            hidden_size=hidden_size,
            embedding_cfg=embedding_cfg,
            add_embedding_before=add_embedding_before,
            use_local_weights=use_local_weights,
            activation=activation
        )
