from typing import List, Union, Optional

from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor, nn
from tsl.nn.blocks import MLPDecoder, MultiRNN, MultiMLP, RNN
from tsl.nn.layers import MultiLinear
from tsl.nn.models import BaseModel
from tsl.utils import ensure_list

from lib.nn.models import TimeAndSpace


class RNNModel(TimeAndSpace):
    def __init__(self, input_size: int, horizon: int, n_nodes: int = None,
                 output_size: int = None,
                 exog_size: int = 0,
                 hidden_size: int = 32,
                 n_layers: int = 1,
                 rnn_layers: int = None,
                 cell: str = 'gru',
                 embedding_cfg: dict = None,
                 add_embedding_before: Union[str, List[str]] = 'encoding',
                 use_local_weights: Union[str, List[str]] = None,
                 activation: str = 'elu'):
        if rnn_layers is not None:
            n_layers = rnn_layers
        rnn = RNN(input_size=hidden_size,
                  hidden_size=hidden_size,
                  n_layers=n_layers,
                  return_only_last_state=True,
                  cell=cell)
        super(RNNModel, self).__init__(
            input_size=input_size,
            horizon=horizon,
            stmp_conv=rnn,
            n_nodes=n_nodes,
            output_size=output_size,
            exog_size=exog_size,
            hidden_size=hidden_size,
            embedding_cfg=embedding_cfg,
            add_embedding_before=add_embedding_before,
            use_local_weights=use_local_weights,
            activation=activation
        )

    def stmp(self, x: Tensor, edge_index=None,
             edge_weight: Optional[Tensor] = None,
             emb: Optional[Tensor] = None) -> Tensor:
        # rnn encoding
        out = self.stmp_conv(x)
        return out

    def forward(self, x: Tensor, u: Optional[Tensor] = None):  # noqa
        return super().forward(x, edge_index=None, u=u)  # noqa


class FCRNNModel(RNNModel):
    def __init__(self, input_size: int, horizon: int, n_nodes: int,
                 output_size: int = None,
                 exog_size: int = 0,
                 hidden_size: int = 32,
                 n_layers: int = 1,
                 cell: str = 'gru',
                 activation: str = 'elu'):
        output_size = output_size or input_size
        input_size = input_size * n_nodes
        output_size = output_size * n_nodes
        super().__init__(input_size=input_size,
                         horizon=horizon,
                         n_nodes=n_nodes,
                         output_size=output_size,
                         exog_size=exog_size,
                         hidden_size=hidden_size,
                         n_layers=n_layers,
                         cell=cell,
                         embedding_cfg=None,
                         activation=activation)

    def forward(self, x: Tensor, u: Optional[Tensor] = None):
        x = rearrange(x, 'b t n f -> b t 1 (n f)')
        if u is not None:
            assert u.ndim == 3
        x, _ = super(FCRNNModel, self).forward(x, u)
        # [b, h, 1, (n f)]
        return rearrange(x, 'b h 1 (n f) -> b h n f', n=self.n_nodes)


class LocalRNNModel(BaseModel):

    def __init__(self, input_size: int, hidden_size: int,
                 n_nodes: int, horizon: int,
                 output_size: int = None,
                 ff_size: int = None,
                 share_weights: Union[str, List[str]] = None,
                 rnn_layers: int = 1, ff_layers: int = 2,
                 cat_states_layers: bool = False,
                 cell: str = 'gru'):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size or input_size
        self.ff_size = ff_size or hidden_size

        self.n_nodes = n_nodes
        self.horizon = horizon
        self.rnn_layers = rnn_layers
        self.ff_layers = ff_layers
        self.cat_states_layers = cat_states_layers

        if share_weights is not None:
            self.share_weights = set(ensure_list(share_weights))
            if len(self.share_weights.difference(['encoder', 'decoder'])):
                raise ValueError("Parameter 'share_weights' must be "
                                 "'encoder', 'decoder', or both.")
        else:
            self.share_weights = set()

        if 'encoder' in self.share_weights:
            self.encoder = nn.Linear(input_size, hidden_size)
        else:
            self.encoder = MultiLinear(input_size, hidden_size, n_nodes)

        self.rnn = MultiRNN(hidden_size, hidden_size, n_nodes,
                            n_layers=rnn_layers,
                            cat_states_layers=cat_states_layers,
                            return_only_last_state=True,
                            cell=cell)

        if 'decoder' in self.share_weights:
            self.decoder = MLPDecoder(hidden_size, self.ff_size,
                                      output_size=self.output_size,
                                      horizon=horizon, n_layers=ff_layers)
        else:
            self.decoder = nn.Sequential(
                MultiMLP(hidden_size, self.ff_size, n_nodes,
                         output_size=self.output_size * horizon,
                         n_layers=ff_layers),
                Rearrange('b n (h f) -> b h n f', h=horizon)
            )

    def forward(self, x: Tensor):
        out = self.encoder(x)
        out = self.rnn(out)
        out = self.decoder(out)
        return out
