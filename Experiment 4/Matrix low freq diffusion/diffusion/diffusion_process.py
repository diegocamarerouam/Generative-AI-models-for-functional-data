# -*- coding: utf-8 -*-
"""
Simulate Gaussian processes.

@author: diego.camarero@estudiante.uam.es
"""

from __future__ import annotations

from typing import Callable, Union

import numpy as np
import torch
from torch import Tensor

def euler_maruyama_integrator(
    x_initial: Tensor,
    t_start: float,
    t_end: float,
    n_times: int,  
    drift_coefficient: Callable[float, float],
    diffusion_coefficient: Callable[float],
    seed: Union[int, None] = None
) -> Tensor:
    """Euler-Maruyama integrator (approximate)

     Args:
        x_initial: The initial images of dimensions 
            (batch_size, n_channels, n_pixels)
        t_start: float,
        t_end: endpoint of the integration interval    
        n_times: number of integration times 
        drift_coefficient: Function of :math`(x(t), t)` that defines the drift term            
        diffusion_coefficient: Function of :math`(t)` that defines the diffusion term  
        seed: Seed for the random number generator
        
    Returns:
        x_t: Trajectories that result from the integration of the SDE.
             The shape is (batch_size, n_times, n_channels, n_pixels)
            
    Notes:
        The implementation is fully vectorized except for a loop over time.

    Examples:
        >>> import numpy as np
        >>> drift_coefficient = lambda x_t, t: - x_t
        >>> diffusion_coefficient = lambda t: torch.ones_like(t)
        >>> x_0 = torch.tensor(np.reshape(np.arange(120), (10, 1, 12)))
        >>> t_0, t_end = 0.0, 3.0
        >>> n_times = 7
        >>> times, x_t = euler_maruyama_integrator(
        ...     x_0, t_0, t_end, n_times, drift_coefficient, diffusion_coefficient, 
        ... )
        >>> print(times)
        tensor([0.0000, 0.5000, 1.0000, 1.5000, 2.0000, 2.5000, 3.0000])
        >>> print(np.shape(x_t))
        torch.Size([10, 7, 1, 12])
    """
    
    device = x_initial.device
    
    D = x_initial.shape[-1]
   
    times = torch.linspace(t_start, t_end, n_times, device=device)
    dt = times[1] - times[0]
    
    x_t = torch.empty(
        (*x_initial.shape, len(times)),
        dtype=torch.float32,
        device=device,
    )
    x_t[..., 0] = x_initial
    
    sqrt_dt = torch.sqrt(dt.to(device).abs())
    
    z = torch.randn_like(x_t)
    
    for n, t in enumerate(times[:-1]):
        z_n = z[..., n]
        t = torch.ones(x_initial.shape[0], device=device) * t
        x_t[..., n + 1] = (
            x_t[..., n]
            + drift_coefficient(x_t[..., n], t) * dt
            + (diffusion_coefficient(t).view(-1, 1, D, D)
               @ z_n.unsqueeze(-1)).squeeze(-1)
            * sqrt_dt
        )
        
    x_t =x_t.permute(0, 3, 1, 2)
    return times, x_t
    
class DiffusionProcess:

    def __init__(
        self,          
        drift_coefficient: Callable[float, float] = lambda x_t, t: 0.0,
        diffusion_coefficient: Callable[float] = lambda t: 1.0,
    ):
        self.drift_coefficient = drift_coefficient
        self.diffusion_coefficient = diffusion_coefficient

class GaussianDiffusionProcess(DiffusionProcess):
    """
    Gaussian Diffusion Process using the Score Function modeling.
    SDE:  dx(t) = drift_coefficient(x(t), t) dt + diffusion_coefficient(t) dW(t), where W(t) is the Wiener process (standard Brownwian motion).

    Attributes:
        drift_coefficient: function of (x(t), t) representing the deterministic term of the SDE.
        diffusion_coefficient: function of (t) scaling the noising term of the SDE.
        mu_t: function of (x0, t) representing the mean of x(t) given x_0.
        sigma_t: function of (t) representing the standard deviation of x(t) given x_0.
        
    Methods:
        loss_function(score_model, x_0, eps): computes the difference between the noised image from x_0 and the predicted score by the model.
    
    Example 1:
        >>> mu, sigma = 1.5, 2.0
        >>> bm = GaussianDiffusionProcess(
        ...     drift_coefficient=lambda x_t, t: mu,
        ...     diffusion_coefficient=lambda t: sigma,
        ...     mu_t=lambda x_0, t: x_0 + mu*t,
        ...     sigma_t=lambda t: np.sqrt(2.0 * t),
        ...     t_end=5.0,
        ... )
        >>> print(bm.drift_coefficient(x_t=3.0, t=10.0))
        1.5
        >>> print(bm.diffusion_coefficient(t=10.0))
        2.0
        >>> print(bm.mu_t(x_0=3.0, t=10.0), bm.sigma_t(t=10.0))
        18.0 4.47213595499958
        

    """
    kind = "Gaussian"
    
    def __init__(
        self,          
        drift_coefficient: Callable[float, float] = lambda x_t, t: 0.0,
        diffusion_coefficient: Callable[float] = lambda t: 1.0,
        mu_t: Callable[float, float] = lambda x_0, t: x_0,
        sigma_t: Callable[float] = lambda t: torch.sqrt(t),
        t_0: float=0.0,
        t_end: float=1.0,
        t_eps: float=1.0e-3,
    ):
        self.drift_coefficient = drift_coefficient
        self.diffusion_coefficient = diffusion_coefficient
        self.mu_t = mu_t
        self.sigma_t = sigma_t
        self.t_0 = t_0
        self.t_end = t_end
        self.t_eps = t_eps
    

    def loss_function(
        self,
        score_model, 
        x_0: torch.Tensor, 
    ):
        """The loss function for training score-based generative models.

          Args:
              score_model:  A PyTorch model instance that represents a 
                            time-dependent score-based model.
          x_0: A mini-batch of training data.    
        """
        batch_size, _, D = x_0.shape
        t = torch.rand(batch_size, device=x_0.device) * (self.t_end - self.t_0 - self.t_eps) + self.t_eps + self.t_0
        t_broadcast = t.view(-1, *([1]*(x_0.dim()-1)))
        Z = torch.randn_like(x_0)
        x = self.mu_t(x_0, t_broadcast) + (self.sigma_t(t_broadcast) @ Z.unsqueeze(-1)).squeeze(-1)
        loss = (
            (
            (self.sigma_t(t_broadcast).mT.view(-1,1,D,D)
            @ score_model(x, t).unsqueeze(-1)).squeeze(-1)
            + Z
            ) ** 2
            ).sum(dim=-1).mean()
        return loss
    
def inverse_Tweedie_formula(
        x_t,
        t,
        inv_mu_t: Callable[float],
        sigma_t: Callable[float],
        score_model: Callable[float, float],
        ):
    t = torch.as_tensor(t, device=x_t.device).view(1)
    inv_drift = inv_mu_t(t)
    sigma = sigma_t(t)
    score = score_model(x_t, t)
    output = (inv_drift @ (x_t.unsqueeze(-1) + sigma @ sigma.mT @ score.unsqueeze(-1))).squeeze(-1)
    return output


if __name__ == "__main__":
    import doctest
    doctest.testmod()
