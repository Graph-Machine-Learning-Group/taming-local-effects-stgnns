from typing import Optional, Callable, Mapping, Type, Union, List, Dict

import torch
from torch import nn, Tensor
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops
from torchmetrics import Metric
from tsl.engines import Predictor
from tsl.experiment import NeptuneLogger
from tsl.ops.connectivity import convert_torch_connectivity
from tsl.utils import ensure_list

from lib.nn.layers import VariationalNodeEmbedding, ClusterizedNodeEmbedding
from lib.nn.losses import kl_divergence_normal
from lib.nn.models import STGNN


class EmbeddingPredictor(Predictor):

    def __init__(self,
                 model: Optional[nn.Module] = None,
                 loss_fn: Optional[Callable] = None,
                 scale_target: bool = False,
                 metrics: Optional[Mapping[str, Metric]] = None,
                 beta: Union[float, int, List, Dict, Tensor] = 1.0,
                 embedding_var: float = 0.2,
                 log_embeddings_every: int = None,
                 warm_up: int = 5,
                 *,
                 model_class: Optional[Type] = None,
                 model_kwargs: Optional[Mapping] = None,
                 optim_class: Optional[Type] = None,
                 optim_kwargs: Optional[Mapping] = None,
                 scheduler_class: Optional = None,
                 scheduler_kwargs: Optional[Mapping] = None):
        super(EmbeddingPredictor, self).__init__(model=model,
                                                 model_class=model_class,
                                                 model_kwargs=model_kwargs,
                                                 optim_class=optim_class,
                                                 optim_kwargs=optim_kwargs,
                                                 loss_fn=loss_fn,
                                                 scale_target=scale_target,
                                                 metrics=metrics,
                                                 scheduler_class=scheduler_class,
                                                 scheduler_kwargs=scheduler_kwargs)
        assert isinstance(beta, (float, int, list, dict, Tensor))
        self.beta = beta
        if isinstance(beta, dict):
            self.current_beta = beta[sorted(beta)[0]]  # lowest key
        self.embedding_var = embedding_var
        self._embeddings = []
        self.log_embeddings_every = log_embeddings_every
        self.warm_up = warm_up

    def get_beta(self):
        if self.current_epoch < self.warm_up:
            return 0
        if isinstance(self.beta, (float, int)):
            return self.beta
        elif isinstance(self.beta, (list, Tensor)):
            return self.beta[self.current_epoch]
        elif isinstance(self.beta, dict):
            if self.current_epoch in self.beta:
                self.current_beta = self.beta[self.current_epoch]
            return self.current_beta

    def get_embedding_module(self):
        if isinstance(self.model, STGNN):
            return self.model.emb
        return None

    def predict_batch(self, batch,
                      preprocess: bool = False, postprocess: bool = True,
                      return_target: bool = False,
                      **forward_kwargs):
        inputs, targets, mask, transform = self._unpack_batch(batch)
        if preprocess:
            for key, trans in transform.items():
                if key in inputs:
                    inputs[key] = trans.transform(inputs[key])

        if forward_kwargs is None:
            forward_kwargs = dict()
        out = self.forward(**inputs, **forward_kwargs)
        if isinstance(out, tuple):
            y_hat, emb = out
        else:
            y_hat, emb = out, None
        # Rescale outputs
        if postprocess:
            trans = transform.get('y')
            if trans is not None:
                y_hat = trans.inverse_transform(y_hat)
        if return_target:
            y = targets.get('y')
            return y, y_hat, mask
        return y_hat, emb

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        """"""
        # Unpack batch
        x, y, mask, transform = self._unpack_batch(batch)

        # Make predictions
        y_hat, _ = self.predict_batch(batch, preprocess=False, postprocess=True)

        output = dict(**y, y_hat=y_hat)
        if mask is not None:
            output['mask'] = mask
        return output

    def regularization(self, batch):
        reg, beta = 0, None
        embedding_module = self.get_embedding_module()
        if isinstance(embedding_module, VariationalNodeEmbedding):
            mu = embedding_module.emb
            log_var = embedding_module.log_var
            kl = kl_divergence_normal(mu, log_var)
            self.log('kl_divergence', kl, on_step=False, on_epoch=True,
                     logger=True, prog_bar=False, batch_size=batch.batch_size)
            beta = self.get_beta()
            reg = beta * kl
        elif isinstance(embedding_module, ClusterizedNodeEmbedding):
            clustering_loss, avg_dist = embedding_module.clustering_loss()
            self.log('clustering_loss', clustering_loss.detach(), on_step=False,
                     on_epoch=True,
                     logger=True, prog_bar=False, batch_size=batch.batch_size)
            self.log('assignment_temp', embedding_module._tau, on_step=False,
                     on_epoch=True,
                     logger=True, prog_bar=False)
            self.log('average_centroid_distance', avg_dist, on_step=False,
                     on_epoch=True,
                     logger=True, prog_bar=False, batch_size=batch.batch_size)
            if self.log_embeddings_every is not None:
                # log average distance among centroids
                sep = torch.cdist(embedding_module.centroids,
                                  embedding_module.centroids).mean()
                self.log('separation_l2', sep.mean(), on_step=False,
                         on_epoch=True,
                         logger=True, prog_bar=False,
                         batch_size=batch.batch_size)
                # log entropy of the cluster assignments distribution
                entropy = embedding_module.assigment_entropy()
                self.log('entropy', entropy, on_step=False, on_epoch=True,
                         logger=True, prog_bar=False,
                         batch_size=batch.batch_size)
                # log number of cluster members for each cluster
                assignments = embedding_module.get_assignment(estimator='ste')
                counts = assignments.sum(0)
                for i, c in enumerate(counts):
                    self.log(f'members_cluster{i}', c, on_step=True,
                             logger=True)
            beta = self.get_beta()
            reg = beta * clustering_loss
        if beta is not None:
            self.log('reg_weight', beta, on_step=False, on_epoch=True,
                     logger=True, prog_bar=False)
        return reg

    def training_step(self, batch, batch_idx):
        """"""
        y = y_loss = batch.y
        mask = batch.get('mask')

        # Compute predictions and compute loss
        y_hat_loss, emb = self.predict_batch(batch, preprocess=False,
                                             postprocess=not self.scale_target)
        y_hat = y_hat_loss.detach()

        # Scale target and output, eventually
        if self.scale_target:
            y_loss = batch.transform['y'].transform(y)
            y_hat = batch.transform['y'].inverse_transform(y_hat)

        # Compute loss
        loss = self.loss_fn(y_hat_loss, y_loss, mask)
        auxiliary_loss = self.regularization(batch)

        loss = loss + auxiliary_loss

        # Logging
        self.train_metrics.update(y_hat, y, mask)
        self.log_metrics(self.train_metrics, batch_size=batch.batch_size)
        self.log_loss('train', loss, batch_size=batch.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        """"""
        y = y_loss = batch.y
        mask = batch.get('mask')

        # Compute predictions
        y_hat_loss, emb = self.predict_batch(batch, preprocess=False,
                                             postprocess=not self.scale_target)
        y_hat = y_hat_loss.detach()

        # Scale target and output, eventually
        if self.scale_target:
            y_loss = batch.transform['y'].transform(y)
            y_hat = batch.transform['y'].inverse_transform(y_hat)

        # Compute loss
        val_loss = self.loss_fn(y_hat_loss, y_loss, mask)

        # Logging
        self.val_metrics.update(y_hat, y, mask)
        self.log_metrics(self.val_metrics, batch_size=batch.batch_size)
        self.log_loss('val', val_loss, batch_size=batch.batch_size)
        return val_loss

    def test_step(self, batch, batch_idx):
        """"""
        # Compute outputs and rescale
        y_hat, emb = self.predict_batch(batch, preprocess=False,
                                        postprocess=True)

        y, mask = batch.y, batch.get('mask')
        test_loss = self.loss_fn(y_hat, y, mask)

        # Logging
        self.test_metrics.update(y_hat.detach(), y, mask)
        self.log_metrics(self.test_metrics, batch_size=batch.batch_size)
        self.log_loss('test', test_loss, batch_size=batch.batch_size)
        return test_loss

    def log_embeddings(self, emb, colors=None):
        if isinstance(self.logger, NeptuneLogger):
            from matplotlib import pyplot as plt
            from sklearn.manifold import SpectralEmbedding, TSNE
            emb = TSNE(n_components=2).fit_transform(emb)
            fig, ax = plt.subplots(figsize=(7, 6))
            if colors is not None:
                assert len(colors) == len(emb)
                ax.scatter(emb[:, 0], emb[:, 1], c=colors)
            else:
                ax.scatter(emb[:, 0], emb[:, 1])
            ax.set_title(f"Embeddings at step: {self.global_step:6d}")
            plt.tight_layout()
            self.logger.log_figure(fig, f'embeddings/step{self.global_step}')
            plt.close()

    def reduce_embedding_dim(self, embeding_method: str = 'spectral',
                             n_components: int = 2):
        from sklearn.manifold import SpectralEmbedding, TSNE

        if embeding_method == 'spectral':
            embeddings = [SpectralEmbedding(n_components).fit_transform(emb)
                          for emb in self._embeddings]
        elif embeding_method == 'tsne':
            embeddings = []
            emb_2d = 'pca'
            for emb in self._embeddings:
                emb_2d = TSNE(n_components, init=emb_2d,
                              n_iter_without_progress=200,
                              min_grad_norm=1e-5).fit_transform(emb)
                embeddings.append(emb_2d)
        else:
            raise NotImplementedError()

        return embeddings

    def plot_embeddings(self, filename: str, edge_index,
                        embeding_method: str = 'spectral',
                        writer: str = 'pillow'):
        import numpy as np
        import networkx as nx

        from matplotlib import rcParams, animation, pyplot as plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        from torch_geometric.utils import to_networkx

        rcParams['font.size'] = 14

        # Reduce embeddings  ##################################################

        if self._embeddings[0].shape[-1] > 2:
            embeddings = self.reduce_embedding_dim(embeding_method, 2)
        else:
            embeddings = self._embeddings

        # Generate graph  #####################################################

        n_iter, n_nodes = len(embeddings), len(embeddings[0])
        edge_index, _ = convert_torch_connectivity(edge_index,
                                                   target_layout='edge_index')
        edge_index, _ = remove_self_loops(edge_index)
        g0 = Data(edge_index=edge_index, pos=embeddings[0], num_nodes=n_nodes)
        G = to_networkx(g0)
        degree = [deg for node, deg in G.degree]

        # Create figure  ######################################################

        fig, ax = plt.subplots(figsize=(9, 7))
        fig.subplots_adjust(left=0.01, right=1.05, top=0.95, bottom=0.01)

        # add colormap
        cmappable = ScalarMappable(norm=Normalize(min(degree), max(degree)),
                                   cmap='plasma')
        fig.colorbar(cmappable, ax=ax, location='right', pad=0.02,
                     label='Node degree', shrink=0.95)

        embs = np.concatenate(embeddings, 0)
        min_x, min_y = np.min(embs, 0)
        max_x, max_y = np.max(embs, 0)
        std_x, std_y = np.std(embs, 0) * 0.2

        ax.set_xlim(min_x - std_x, max_x + std_x)
        ax.set_ylim(min_y - std_y, max_y + std_y)

        # Plot graph  #########################################################

        def update(i):
            ax.clear()
            curr_emb = embeddings[i]
            pos = dict(zip(G.nodes, curr_emb))
            nx.draw_networkx(G, pos=pos, ax=ax, font_size=6, width=0.1,
                             cmap='plasma', node_color=degree,
                             arrowsize=4, node_size=150,
                             font_color='#fff')
            ax.set_title(f"Embeddings at step: {i:6d}/{n_iter}")

        ani = animation.FuncAnimation(fig, update, frames=n_iter, interval=200)
        ani.save(filename, writer=writer)

    def on_train_epoch_start(self) -> None:
        # Log learning rate
        optimizers = ensure_list(self.optimizers())
        for i, optimizer in enumerate(optimizers):
            lr = optimizer.optimizer.param_groups[0]['lr']
            self.log(f'lr_{i}', lr, on_step=False, on_epoch=True, logger=True,
                     prog_bar=False, batch_size=1)

        # Log embeddings
        embedding_module = self.get_embedding_module()
        if embedding_module is None:
            return

        if isinstance(embedding_module, ClusterizedNodeEmbedding):
            if self.current_epoch == self.warm_up:
                embedding_module.init_centroids()
            embedding_module.step()

        if (self.log_embeddings_every is not None) and \
                (self.current_epoch % self.log_embeddings_every == 0):
            # log embeddings
            emb = embedding_module.emb.detach().cpu().clone()
            self._embeddings.append(emb)
            if isinstance(embedding_module, ClusterizedNodeEmbedding):
                clu = embedding_module.assignment_logits().detach().cpu()
                colors = torch.argmax(clu, dim=-1).numpy()
            else:
                colors = None
            self.log_embeddings(emb.numpy(), colors)
