from typing import Union, List

from tsl.nn.blocks import GraphConvRNN

from .prototypes import TimeAndSpace


class TimeAndGraphIsoModel(TimeAndSpace):

    def __init__(self, input_size: int, horizon: int, n_nodes: int = None,
                 output_size: int = None,
                 exog_size: int = 0,
                 hidden_size: int = 32,
                 embedding_cfg: dict = None,
                 add_embedding_before: Union[str, List[str]] = 'encoding',
                 use_local_weights: Union[str, List[str]] = None,
                 n_layers: int = 1,
                 root_weight: bool = True,
                 norm: str = 'none',
                 cached: bool = False,
                 activation: str = 'elu'):
        stmp_conv = GraphConvRNN(input_size=hidden_size,
                                 hidden_size=hidden_size,
                                 n_layers=n_layers,
                                 root_weight=root_weight,
                                 norm=norm,
                                 cached=cached,
                                 cell='gru',
                                 activation=activation,
                                 return_only_last_state=True)
        super(TimeAndGraphIsoModel, self).__init__(
            input_size=input_size,
            horizon=horizon,
            stmp_conv=stmp_conv,
            n_nodes=n_nodes,
            output_size=output_size,
            exog_size=exog_size,
            hidden_size=hidden_size,
            embedding_cfg=embedding_cfg,
            add_embedding_before=add_embedding_before,
            use_local_weights=use_local_weights,
            activation=activation
        )
