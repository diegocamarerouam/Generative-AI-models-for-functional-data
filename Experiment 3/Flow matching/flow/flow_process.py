# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 12:19:36 2026

@author: diego.camarero@estudiante.uam.es

"""
from __future__ import annotations

from typing import Callable, Union

import torch
import torch.nn.functional as F
from torch import Tensor

def flow_interpolation(
    data,
    alpha_t,
    beta_t,
    t_0,
    T,
    n_steps,
):
    device = data.device
    times = torch.linspace(t_0, T, n_steps, device=device)
    t = times.view(1, -1, 1, 1)
    x1 = data.unsqueeze(1)
    x0 = torch.randn_like(x1, device=device)
    return alpha_t(t) * x0 + beta_t(t) * x1

def euler_integrator(
        x_initial: Tensor,
        t_start: float,
        t_end: float,
        n_times: int,
        ode_coefficient: Callable[[Tensor], Tensor],
        seed: Union[int, None] = None
        ):
    device = x_initial.device
    times = torch.linspace(t_start, t_end, n_times, device=device)
    dt = times[1]- times[0]
    
    batch_size, _, D = x_initial.shape
    
    X = torch.zeros((batch_size, n_times, 1, D), device=device)
    X[:, 0, :, :] = x_initial
    for n in range(1, n_times):
        x_old = X[:, n-1, :, :]
        t_n = times[n-1]
        t_n = torch.ones(x_old.shape[0], device=device) * times[n-1]
        X[:, n, :, :] = x_old + dt * ode_coefficient(x_old, t_n)
        
    return times, X
        
class FlowMatchingProcess():
    def __init__(
        self,          
        alpha_t: Callable[float] = lambda t: 1-t,
        beta_t: Callable[float] = lambda t: t,
        diff_alpha_t: Callable[float, float] = -1,
        diff_beta_t: Callable[float] = 1,
        t_end: float=1.0,
        t_eps: float=1.0e-3
    ):
        self.alpha_t = alpha_t
        self.beta_t = beta_t
        self.diff_alpha_t = diff_alpha_t
        self.diff_beta_t = diff_beta_t
        self.t_end = t_end
        self.t_eps = t_eps
    

    def loss_function(
        self,
        v_model, 
        x_1: torch.Tensor, 
    ):
        device = x_1.device
        M, _, D = x_1.shape
        
        t_n = torch.rand(
            M, device=device,
            ) * (self.t_end - self.t_eps) + self.t_eps
        
        t_broadcast = t_n.view(-1, *([1]*(x_1.dim()-1)))
        
        x_0 = torch.randn_like(x_1, device=device)
        x_t = self.alpha_t(t_broadcast) * x_0 + self.beta_t(t_broadcast) * x_1

        loss = (
            (v_model(x_t, t_n) - (self.diff_alpha_t(t_broadcast) * x_0 + self.diff_beta_t(t_broadcast) * x_1))** 2
            ).sum(dim=2).mean()
        return loss


if __name__ == "__main__":
    import doctest
    doctest.testmod()
