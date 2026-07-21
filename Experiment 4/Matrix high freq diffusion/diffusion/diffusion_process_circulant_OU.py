# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 00:27:57 2026

@author: diego.camarero@estudiante.uam.es

Automatized diffusion for the Ornstein-Uhlenbeck process,
where the coefficients are:
    - Circulant.
    - Time-dependent.
    - Symmteric.
    
Variance Preserving (VP) is assumed in 'build_diffusion_spectrum_t'.
"""

import torch
from torch import Tensor
from typing import Callable

def lambda_high_schedule(
        lambda_base: Callable[[Tensor], Tensor],
        low_freq: float,
        high_freq: float,
        n_points: int,
        t_0: float,
        t_end: float,
        ) -> Callable[[Tensor], Tensor]:

    freq_mean = int((low_freq + high_freq) // 2)
    mask = torch.ones(n_points)
    mask[2*freq_mean+1:] = 0
    def lambda_t(t):
        t = torch.atleast_1d(t)
        m = mask.to(t.device)
        t_norm = (t-t_0) / (t_end-t_0)
        t_norm = t_norm.unsqueeze(-1).expand(*t.shape, n_points)
        masked_t_norm = 2 * torch.max(0.5 * m, t_norm) - m
        masked_t = t_0 + (t_end - t_0) * masked_t_norm
        return lambda_base(masked_t)
        
    return lambda_t    

def lambda_low_schedule(
        lambda_base: Callable[[Tensor], Tensor],
        low_freq: float,
        high_freq: float,
        n_points: int,
        t_0: float,
        t_end: float,
        ) -> Callable[[Tensor], Tensor]:
    
    freq_mean = int((low_freq + high_freq) // 2)
    mask = torch.ones(n_points)
    mask[:2*freq_mean+1] = 0
    def lambda_t(t):
        t = torch.atleast_1d(t)
        m = mask.to(t.device)
        t_norm = (t-t_0) / (t_end-t_0)
        t_norm = t_norm.unsqueeze(-1).expand(*t.shape, n_points)
        masked_t_norm = 2 * torch.max(0.5 * m, t_norm) - m
        masked_t = t_0 + (t_end - t_0) * masked_t_norm
        return lambda_base(masked_t)
        
    return lambda_t

def integral_low_lambda(
        t_initial: float,
        t_end: float,
        low_freq: float,
        high_freq: float,
        lambdas_t: Callable[[Tensor], Tensor],
        M: int,
        ):
    lambdas_min = lambdas_t(torch.tensor(t_initial))
    lambdas_max = lambdas_t(torch.tensor(t_end))
    
    cutoff = int((low_freq + high_freq) // 2)
    split = 2 * cutoff + 1

    # mask 0: accelerated decay, # mask 1: delayed decay
    is_mask0 = torch.zeros(M)
    is_mask0[:split] = 1.0
    is_mask1 = 1.0 - is_mask0
    
    def integral_lambda_t(t):
        t = torch.atleast_1d(t).reshape(-1, 1)
        
        t_0 = torch.tensor(t_initial, device=t.device)
        T = torch.tensor(t_end, device=t.device)
        t_mid = ((t_0 + T) / 2.0).to(t.device)
        
        
        t_clamped = torch.clamp(t, max=t_mid)
        t_active  = torch.clamp(t - t_mid, min=0.0)
        
        lambda_min = lambdas_min.to(t.device)
        lambda_max = lambdas_max.to(t.device)
        mask0 = is_mask0.to(t.device)
        mask1 = is_mask1.to(t.device)

        int_mask0 = (
            (t - t_0) * lambda_min
            + (t - t_0)**2 / (2*(T - t_0)) * (lambda_max - lambda_min)
        ) 

        int_mask1 = (
            lambda_min * (t_clamped - t_0)
            + lambda_min * t_active                                      
            + t_active**2 / (T - t_0) * (lambda_max - lambda_min)
        )

        return mask0 * int_mask0 + mask1 * int_mask1
    
    return integral_lambda_t
    
def integral_high_lambda(
        t_initial: float,
        t_end: float,
        low_freq: float,
        high_freq: float,
        lambdas_t: Callable[[Tensor], Tensor],
        M: int,
        ):
    lambdas_min = lambdas_t(torch.tensor(t_initial))
    lambdas_max = lambdas_t(torch.tensor(t_end))
    
    cutoff = int((low_freq + high_freq) // 2)
    split = 2 * cutoff + 1

    # mask 0: accelerated decay, # mask 1: delayed decay
    is_mask0 = torch.zeros(M)
    is_mask0[split:] = 1.0
    is_mask1 = 1.0 - is_mask0
    
    def integral_lambda_t(t):
        t = torch.atleast_1d(t).reshape(-1, 1)
        
        t_0 = torch.tensor(t_initial, device=t.device)
        T = torch.tensor(t_end, device=t.device)
        t_mid = ((t_0 + T) / 2.0).to(t.device)
        
        
        t_clamped = torch.clamp(t, max=t_mid)
        t_active  = torch.clamp(t - t_mid, min=0.0)
        
        lambda_min = lambdas_min.to(t.device)
        lambda_max = lambdas_max.to(t.device)
        mask0 = is_mask0.to(t.device)
        mask1 = is_mask1.to(t.device)

        int_mask0 = (
            (t - t_0) * lambda_min
            + (t - t_0)**2 / (2*(T - t_0)) * (lambda_max - lambda_min)
        ) 

        int_mask1 = (
            lambda_min * (t_clamped - t_0)
            + lambda_min * t_active                                      
            + t_active**2 / (T - t_0) * (lambda_max - lambda_min)
        )

        return mask0 * int_mask0 + mask1 * int_mask1
    
    return integral_lambda_t

def build_diffusion_spectrum_t(
        drift_spectrum_t: Callable[[Tensor], Tensor],
        )-> Callable[[Tensor], Tensor]:
    """
    Builds the symmetric spectrum of the diffusion coefficient,
    defined as sqrt(2*drift_spectrum) for Variance Preserving.
    """
    diffusion_spectrum_t = lambda t: torch.sqrt(2.0 * drift_spectrum_t(t))
    return diffusion_spectrum_t

if __name__ == "__main__":
    import doctest
    doctest.testmod()

