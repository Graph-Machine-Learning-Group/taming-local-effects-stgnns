import os

import numpy as np
import torch
from omegaconf import OmegaConf
from torch_geometric.profile import get_gpu_memory_from_nvidia_smi


def find_devices(max_devices: int = 1, greedy: bool = False, gamma: int = 12):
    # if no gpus are available return None
    if not torch.cuda.is_available():
        return max_devices
    n_gpus = torch.cuda.device_count()
    # if only 1 gpu, return 1 (i.e., the number of devices)
    if n_gpus == 1:
        return 1
    # if multiple gpus are available, return gpu id list with length max_devices
    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible_devices is not None:
        visible_devices = [int(i) for i in visible_devices.split(',')]
    else:
        visible_devices = range(n_gpus)
    available_memory = np.asarray([get_gpu_memory_from_nvidia_smi(device)[0]
                                   for device in visible_devices])
    # if greedy, return `max_devices` gpus sorted by available capacity
    if greedy:
        devices = np.argsort(available_memory)[::-1].tolist()
        return devices[:max_devices]
    # otherwise sample `max_devices` gpus according to available capacity
    p = (available_memory / np.linalg.norm(available_memory, gamma)) ** gamma
    # ensure p sums to 1
    p = p / p.sum()
    devices = np.random.choice(np.arange(len(p)), size=max_devices,
                               replace=False, p=p)
    return devices.tolist()


def cfg_to_python(obj):
    try:
        return OmegaConf.to_object(obj)
    except ValueError:
        return obj
