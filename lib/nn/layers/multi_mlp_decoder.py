from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn
from tsl.nn.blocks import MLP
from tsl.nn.layers import MultiLinear


class MultiMLPDecoder(nn.Module):
    r"""MLP decoder for multistep forecasting where the weights of the final
    (linear) layer are not shared among the instances.

    If the input representation has a temporal dimension, this model will take
    the flattened representations corresponding to the last ``receptive_field``
    time steps.

    Args:
        input_size (int): Input size.
        hidden_size (int): Hidden size.
        output_size (int): Output size.
        horizon (int): Output steps.
        n_layers (int, optional): Number of hidden layers in the decoder.
            (default: :obj:`1`)
        receptive_field (int, optional): Number of steps to consider for
            decoding.
            (default: :obj:`1`)
        activation (str, optional): Activation function to use.
        dropout (float, optional): Dropout probability applied in the hidden
            layers.
    """

    def __init__(self, input_size: int, n_instances: int, hidden_size: int,
                 output_size: int,
                 horizon: int = 1,
                 n_layers: int = 1,
                 receptive_field: int = 1,
                 activation: str = 'relu',
                 dropout: float = 0.):
        super(MultiMLPDecoder, self).__init__()

        self.receptive_field = receptive_field
        if n_layers > 0:
            self.mlp = MLP(input_size=receptive_field * input_size,
                           hidden_size=hidden_size,
                           n_layers=n_layers,
                           dropout=dropout,
                           activation=activation)
        else:
            hidden_size = input_size * receptive_field
            self.mlp = nn.Identity()
        self.readout = nn.Sequential(
            MultiLinear(hidden_size, output_size * horizon, n_instances),
            Rearrange('b n (h c) -> b h n c', c=output_size, h=horizon)
        )

    def forward(self, h):
        # h: [batches (steps) nodes features]
        if h.dim() == 4:
            # take last step representation
            h = rearrange(h[:, -self.receptive_field:], 'b s n c -> b n (s c)')
        else:
            assert self.receptive_field == 1
        return self.readout(self.mlp(h))
