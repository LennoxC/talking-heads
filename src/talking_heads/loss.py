import torch
import torch.nn as nn
from typing import Literal

# Gaussian Negative Log Likelihood Loss for regression tasks with uncertainty estimation.
@staticmethod
class GaussianNLLLoss:
    def __call__(self, mean, logvar, target):
        # mean, logvar: (batch_size, out_dim)
        # target: (batch_size, out_dim)
        
        nll = ((mean - target) ** 2 * torch.exp(-logvar) + logvar).mean()

        return nll.mean()
        
# L2 Loss for regression tasks without uncertainty estimation.
@staticmethod
class L2Loss:
    def __call__(self, pred, target):
        # pred: (batch_size, out_dim)
        # target: (batch_size, out_dim)

        l2 = ((pred - target) ** 2).mean()

        return l2.mean()