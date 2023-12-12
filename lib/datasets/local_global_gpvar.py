import math
import os.path

import numpy as np
import torch
from einops import rearrange
from torch import nn
from torch_geometric.utils import remove_self_loops, add_self_loops
from tsl import logger
from tsl.datasets import GaussianNoiseSyntheticDataset
from tsl.nn.layers.graph_convs.gpvar import GraphPolyVAR
from tsl.ops.connectivity import parse_connectivity
from tsl.ops.graph_generators import build_tri_community_graph


class _LocalGlobalGPVAR(nn.Module):
    def __init__(self,
                 temporal_order,
                 spatial_order,
                 num_nodes,
                 cluster_index=None):
        super(_LocalGlobalGPVAR, self).__init__()
        self.global_model = GraphPolyVAR(temporal_order=temporal_order,
                                         spatial_order=spatial_order,
                                         norm='none')

        self.temporal_order = temporal_order
        self.spatial_order = spatial_order
        self.num_nodes = num_nodes
        self.cluster_index = cluster_index

        if self.cluster_index is None:
            self.local_weight = nn.Parameter(torch.Tensor(num_nodes,
                                                          temporal_order))
        else:
            num_clusters = torch.unique(self.cluster_index).numel()
            self.local_weight = nn.Parameter(torch.Tensor(num_clusters,
                                                          temporal_order))

        self.reset_parameters()

    def reset_parameters(self):
        self.global_model.reset_parameters()
        a = math.sqrt(self.local_weight.size(1))
        with torch.no_grad():
            self.local_weight.data.uniform_(-a, a)

    @classmethod
    def from_params(cls, global_params, local_params=None, num_nodes=None,
                    p_max=1., cluster_index=None, seed=None):

        temporal_order = global_params.shape[1]
        spatial_order = global_params.shape[0] - 1  #
        num_nodes = num_nodes or local_params.shape[0]

        model = cls(temporal_order=temporal_order,
                    spatial_order=spatial_order,
                    num_nodes=num_nodes,
                    cluster_index=cluster_index)
        model.global_model.weight.data.copy_(global_params)

        if local_params is None:

            rng = torch.Generator()
            if seed is not None:
                rng.manual_seed(seed)

            if cluster_index is None:
                num_instances = num_nodes
            else:
                num_instances = torch.unique(cluster_index).numel()

            if p_max > 0:
                local_params = 2. * p_max * torch.rand(num_instances,
                                                       temporal_order,
                                                       generator=rng) - p_max
            else:
                local_params = torch.ones(num_instances,
                                          temporal_order) / temporal_order
        else:
            assert temporal_order == local_params.shape[1]

        model.local_weight.data.copy_(local_params)
        return model

    def forward(self, x, edge_index, edge_weight=None):
        # x : [batch, steps, nodes, channels]
        x_l = self.global_model(x, edge_index, edge_weight)
        x_l = torch.tanh(torch.cat([x[:, -self.temporal_order + 1:], x_l], 1))
        x_l = rearrange(x_l, "b p n f -> b n (p f)")
        local_weight = self.local_weight
        if self.cluster_index is not None:
            local_weight = local_weight[self.cluster_index]
        x_l = torch.einsum('bnp, np -> bn', x_l, local_weight)
        x_l = rearrange(x_l, 'b n -> b 1 n 1')
        return x_l


class LocalGlobalGPVARDataset(GaussianNoiseSyntheticDataset):
    """
    """
    seed = 42

    def __init__(self,
                 num_communities,
                 num_steps,
                 global_params,
                 p_max=2.,
                 sigma_noise=.2,
                 share_community_weights: bool = False,
                 save_to: str = None,
                 load_from: str = None,
                 seed: int = None,
                 name=None):
        if name is None:
            self.name = f"LocGlobGPVAR"
        else:
            self.name = name
        self.load_from = load_from
        if seed is not None:
            self.seed = seed
        node_idx, edge_index, _ = build_tri_community_graph(
            num_communities=num_communities)
        num_nodes = len(node_idx)
        # add self loops
        edge_index, _ = add_self_loops(edge_index=torch.tensor(edge_index),
                                       num_nodes=num_nodes)

        if share_community_weights:
            cluster_index = torch.arange(6).repeat_interleave(num_communities)
        else:
            cluster_index = None

        filter = _LocalGlobalGPVAR.from_params(
            global_params=torch.tensor(global_params,
                                       dtype=torch.float32),
            num_nodes=num_nodes,
            p_max=p_max,
            cluster_index=cluster_index,
            seed=self.seed)

        super(LocalGlobalGPVARDataset, self).__init__(num_features=1,
                                                      num_nodes=num_nodes,
                                                      num_steps=num_steps,
                                                      connectivity=edge_index,
                                                      min_window=filter.temporal_order,
                                                      model=filter,
                                                      sigma_noise=sigma_noise,
                                                      seed=seed,
                                                      name=name)
        if save_to is not None:
            self.save(filename=save_to)

    def save(self, filename: str):
        if not hasattr(self, 'target'):
            target, optimal_pred, mask = self.load_raw()
        else:
            target, optimal_pred, mask = self.target, self.optimal_pred, self.mask
        if not os.path.isabs(filename):
            this_dir = os.path.dirname(os.path.realpath(__file__))
            filename = os.path.join(this_dir, filename)
        logger.info(f'Saving synthetic dataset to: {filename}')
        np.savez_compressed(filename + '.npz', target=target,
                            optimal_pred=optimal_pred, mask=mask)

    def load(self):
        if self.load_from is None:
            return self.load_raw()
        else:
            filename = self.load_from
            if not filename.endswith('.npz'):
                filename += '.npz'
            if not os.path.isabs(filename):
                this_dir = os.path.dirname(os.path.realpath(__file__))
                filename = os.path.join(this_dir, filename)
            logger.warning(f'Loading synthetic dataset from: {filename}')
            content = np.load(filename)
            target = content['target']
            optimal_pred = content['optimal_pred']
            mask = content['mask']
            return target, optimal_pred, mask

    def get_connectivity(self, include_weights: bool = True,
                         include_self: bool = True,
                         layout: str = 'edge_index', **kwargs):
        edge_index, edge_weight = self.connectivity
        if not include_weights:
            edge_weight = None
        if not include_self:
            edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        connectivity = (edge_index, edge_weight)
        if layout in ['coo', 'csr', 'sparse_matrix']:
            return parse_connectivity(connectivity=connectivity,
                                      target_layout='sparse',
                                      num_nodes=self.n_nodes)
        elif layout == 'dense':
            return parse_connectivity(connectivity=connectivity,
                                      target_layout='dense',
                                      num_nodes=self.n_nodes)
        else:
            return connectivity


if __name__ == '__main__':
    dataset = LocalGlobalGPVARDataset(
        num_communities=6,
        num_steps=5000,
        global_params=[[2.5, 1.],
                       [-2., 3.],
                       [-.5, 0.]],
        p_max=2.,
        sigma_noise=0.4,
        seed=42,
        share_community_weights=False
    )

    x, y, m = dataset.target, dataset.optimal_pred, dataset.mask

    import matplotlib.pyplot as plt

    f, axs = plt.subplots(min([dataset.n_nodes, 6]), 2, sharex=True,
                          sharey=True)
    for i, ax in enumerate(axs):
        if i == 0:
            ax[0].set_title('Optimal pred')
            ax[1].set_title('Target')
        ax[0].plot(y[:100, i])
        ax[1].plot(x[:100, i])
    plt.tight_layout()
    plt.show()
    plt.imshow(x[:1000, :, 0].T, aspect='auto')
    plt.grid()
    plt.show()
    print('done')
