import torch
import torch.nn as nn
from typing import Literal

# Gaussian Negative Log Likelihood Loss for regression tasks with uncertainty estimation.
@staticmethod
class GaussianNLLLoss:
    def __call__(self, mean, logvar, target, agg: Literal['mean', 'sum'] = 'mean'):
        # mean, logvar: (batch_size, out_dim)
        # target: (batch_size, out_dim)
        
        nll = ((mean - target) ** 2 * torch.exp(-logvar) + logvar).mean()

        if agg == 'mean':
            return nll.mean()
        elif agg == 'sum':
            return nll.sum()
        else:
            raise ValueError(f"Invalid agg value: {agg}. Must be 'mean' or 'sum'.")
        
# L2 Loss for regression tasks without uncertainty estimation.
@staticmethod
class L2Loss:
    def __call__(self, pred, target, agg: Literal['mean', 'sum'] = 'mean'):
        # pred: (batch_size, out_dim)
        # target: (batch_size, out_dim)

        l2 = ((pred - target) ** 2).mean()

        if agg == 'mean':
            return l2.mean()
        elif agg == 'sum':
            return l2.sum()
        else:
            raise ValueError(f"Invalid agg value: {agg}. Must be 'mean' or 'sum'.")