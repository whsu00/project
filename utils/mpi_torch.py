import torch
import pytorch_utils as ptu
from mpi_tools import broadcast, mpi_avg


def sync_all_params(param, root=0):
    data = ptu.to_numpy(torch.nn.utils.parameters_to_vector(param))
    broadcast(data, root)
    torch.nn.utils.vector_to_parameters(torch.from_numpy(data), param)


def average_gradients(param_groups):
    for param_group in param_groups:
        for p in param_group['params']:
            if p.requires_grad:
                p.grad.data.copy_(ptu.from_numpy(mpi_avg(p.grad.data.numpy())))
