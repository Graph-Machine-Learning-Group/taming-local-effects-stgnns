from math import log

import torch
from torch import Tensor


def kl_divergence_normal(mu: Tensor, log_var: Tensor):
    # shape: [*, n, k]
    kl = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=-1)
    return torch.mean(kl)


def kl_divergence(mu: Tensor, log_var: Tensor,
                  mu_tgt: float = 0., var_tgt: float = 1.0):
    # shape: [*, n, k]
    log_var_tgt = log(var_tgt)
    l2_mu = (mu - mu_tgt) ** 2
    kl = log_var_tgt - log_var - 1 + (l2_mu + log_var.exp()) / var_tgt
    kl = 0.5 * torch.sum(kl, dim=-1)
    return torch.mean(kl)
