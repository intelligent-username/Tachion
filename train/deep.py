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
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int = 3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
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

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        horizon: int,
        num_samples: int = 100,
        target_idx: int = 0,
        quantiles: list = [0.025, 0.5, 0.975]
    ) -> dict:
        """
        Generate autoregressive forecasts for `horizon` steps ahead.
        
        :param x: Input context (batch, seq_len, input_size)
        :param horizon: Number of steps to forecast
        :param num_samples: Number of Monte Carlo sample paths
        :param target_idx: Index of target variable in input features (for feeding back)
        :param quantiles: Quantiles to return
        :return: Dict with 'samples' (num_samples, batch, horizon) and 
                 'quantiles' (len(quantiles), batch, horizon)
        """
        self.eval()
        batch_size = x.shape[0]
        device = x.device
        
        student_t = StudentT()
        
        # Initialize: run through context to get hidden state
        _, (h, c) = self.rnn(x)
        
        # Store sample paths: (num_samples, batch, horizon)
        all_samples = torch.zeros(num_samples, batch_size, horizon, device=device)
        
        # For autoregressive: start with last input step
        last_input = x[:, -1:, :]  # (batch, 1, input_size)
        
        for t in range(horizon):
            # Forward one step
            rnn_out, (h, c) = self.rnn(last_input, (h, c))
            params = self.param_proj(rnn_out[:, -1, :])  # (batch, output_size)
            
            # Sample from Student-t
            samples_t = student_t.sample(params, num_samples)  # (num_samples, batch)
            all_samples[:, :, t] = samples_t
            
            # Prepare next input: replace target feature with sampled value (median path)
            median_sample = samples_t.median(dim=0).values  # (batch,)
            next_input = last_input.clone()
            next_input[:, 0, target_idx] = median_sample
            last_input = next_input
        
        # Compute quantiles from samples
        q_vals = []
        for q in quantiles:
            q_val = torch.quantile(all_samples, q, dim=0)  # (batch, horizon)
            q_vals.append(q_val)
        quantile_preds = torch.stack(q_vals, dim=0)  # (len(quantiles), batch, horizon)
        
        return {
            'samples': all_samples,
            'quantiles': quantile_preds,
            'quantile_levels': quantiles
        }


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

    def sample(self, params: torch.Tensor, num_samples: int = 100) -> torch.Tensor:
        """
        Generate Monte Carlo samples from the Student-t distribution.
        
        :param params: Distribution parameters (..., 3) for mu, log_sigma, unconstrained_nu
        :param num_samples: Number of samples to draw
        :return: Samples of shape (num_samples, ...)
        """
        mu, sigma, nu = self.forward(params)
        
        # Use PyTorch's StudentT distribution
        dist = D.StudentT(df=nu, loc=mu, scale=sigma)
        
        # Sample: (num_samples, ...)
        samples = dist.rsample((num_samples,))
        return samples

    def quantiles(self, params: torch.Tensor, quantiles: list = [0.025, 0.5, 0.975], 
                  num_samples: int = 1000) -> torch.Tensor:
        """
        Compute quantiles from Monte Carlo samples.
        
        :param params: Distribution parameters
        :param quantiles: List of quantiles to compute
        :param num_samples: Number of MC samples for estimation
        :return: Tensor of shape (len(quantiles), ...)
        """
        samples = self.sample(params, num_samples)
        
        q_vals = []
        for q in quantiles:
            q_val = torch.quantile(samples, q, dim=0)
            q_vals.append(q_val)
        
        return torch.stack(q_vals, dim=0)


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
