"""
DeepAR model definition
"""

import torch
import torch.nn as nn
import torch.distributions as D

class DeepAR(nn.Module):
    """
    DeepAR Forecaster
    """
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.param_proj = nn.Linear(hidden_size, 2)  # mu, log_sigma

    def forward(self, x):
        h, _ = self.rnn(x)
        params = self.param_proj(h)  # B x T x 2
        return params

class StudentT(nn.Module):
    """
    Student-t distributiion for DeepAR outputs
    """
    def __init__(self):
        super().__init__()

    def forward(self, params):
        # params: (..., 3)
        mu = params[..., 0]
        sigma = torch.exp(params[..., 1])  # positive scale
        nu = torch.nn.functional.softplus(params[..., 2]) + 2.0  # df > 2
        return mu, sigma, nu

    def log_prob(self, params, target):
        mu, sigma, nu = self.forward(params)
        # Using the formula for Student-t log-pdf
        pi = torch.tensor(torch.pi, device=target.device)
        term1 = torch.lgamma((nu + 1)/2) - torch.lgamma(nu/2)
        term2 = -0.5*torch.log(nu*pi) - torch.log(sigma)
        term3 = - (nu + 1)/2 * torch.log(1 + ((target - mu)/sigma)**2 / nu)
        return term1 + term2 + term3


def nll_loss(params, target):
    """
    Negative log-likelihood loss for Gaussian outputs.
    """

    mu, log_sigma = params[..., 0], params[..., 1]
    sigma = torch.exp(log_sigma)
    dist = D.Normal(mu, sigma)
    return -dist.log_prob(target).mean()
