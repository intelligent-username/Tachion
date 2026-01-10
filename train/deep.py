"""
DeepAR model and Student-t likelihood
"""

# NOTE: still need to tune, etc.
# I'll still likely need to import from:

# https://github.com/awslabs/gluonts/tree/dev/src/gluonts/

import torch
import torch.nn as nn
import torch.distributions as D
import math


class DeepAR(nn.Module):
    """
    DeepAR forecaster with LSTM backbone.
    Outputs parameters for predictive distribution (mu, log_sigma or more for Student-t).
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int = 2):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.param_proj = nn.Linear(hidden_size, output_size)  # output_size: 2 for Gaussian, 3 for Student-t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_size)
        Returns: params (batch, seq_len, output_size)
        """
        h, _ = self.rnn(x)
        params = self.param_proj(h)
        return params


class StudentT(nn.Module):
    """
    Student-t distribution parameterization for DeepAR outputs.
    Assumes params[..., 0] = mu, params[..., 1] = log_sigma, params[..., 2] = unconstrained nu
    """
    def __init__(self, min_df: float = 2.0):
        super().__init__()
        self.min_df = min_df  # ensures nu > 2 for finite variance

    def forward(self, params: torch.Tensor):
        mu = params[..., 0]
        sigma = torch.exp(params[..., 1])
        nu = torch.nn.functional.softplus(params[..., 2]) + self.min_df
        return mu, sigma, nu

    def log_prob(self, params: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mu, sigma, nu = self.forward(params)
        term1 = torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2)
        term2 = -0.5 * torch.log(nu * math.pi) - torch.log(sigma)
        term3 = - (nu + 1) / 2 * torch.log(1 + ((target - mu) / sigma) ** 2 / nu)
        return term1 + term2 + term3


def student_t_nll(params: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Negative log-likelihood for Student-t outputs.
    """
    dist = StudentT()
    return -dist.log_prob(params, target).mean()


def gaussian_nll(params: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Negative log-likelihood for Gaussian outputs (mu, log_sigma).
    """
    mu, log_sigma = params[..., 0], params[..., 1]
    sigma = torch.exp(log_sigma)
    dist = D.Normal(mu, sigma)
    return -dist.log_prob(target).mean()
