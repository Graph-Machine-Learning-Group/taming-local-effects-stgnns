from typing import Union, List

from tsl.nn.blocks.encoders import DCRNN

from lib.nn.models import TimeAndSpace


class DCRNNModel(TimeAndSpace):
    def __init__(self, input_size: int, horizon: int, n_nodes: int = None,
                 output_size: int = None,
                 exog_size: int = 0,
                 hidden_size: int = 32,
                 kernel_size: int = 2,
                 root_weight: bool = True,
                 add_backward: bool = True,
                 n_layers: int = 1,
                 embedding_cfg: dict = None,
                 add_embedding_before: Union[str, List[str]] = 'encoding',
                 use_local_weights: Union[str, List[str]] = None,
                 activation: str = 'elu'):
        dcrnn = DCRNN(input_size=hidden_size,
                      hidden_size=hidden_size,
                      n_layers=n_layers,
                      k=kernel_size,
                      root_weight=root_weight,
                      add_backward=add_backward,
                      return_only_last_state=True)
        super(DCRNNModel, self).__init__(
            input_size=input_size,
            horizon=horizon,
            stmp_conv=dcrnn,
            n_nodes=n_nodes,
            output_size=output_size,
            exog_size=exog_size,
            hidden_size=hidden_size,
            embedding_cfg=embedding_cfg,
            add_embedding_before=add_embedding_before,
            use_local_weights=use_local_weights,
            activation=activation
        )
