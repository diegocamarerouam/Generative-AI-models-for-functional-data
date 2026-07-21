# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 12:18:13 2026

@author: diego.camarero@estudiante.uam.es

Utilities for generating synthetic one-dimensional functional datasets
for machine learning experiments using PyTorch.

Each dataset represents a collection of functions evaluated over a
uniform one-dimensional grid.

"""

import torch
from torch.utils.data import Dataset

class BaseFunctionalDataset(Dataset):
    """
    Base class for pre-generated functional datasets.
    Each sample corresponds to a function evaluated at a one-dimensional grid.
    
    Allocates memory for the data tensor and provides the standard
    __len__ and __getitem__ methods to read from memory.
    
    Each sample has shape [1, n_points], where:
        - First dimension corresponds to channel index.
        - Second dimension corresponds to the evaluated grid.
    
    Attributes
    ----------
    n_samples : int
        Size of the dataset.
    n_points : int
        Function domain dimension.
    x_inf : float, optional
        Lower limit of the function domain. The default is 0.0.
    x_sup : float, optional
        Upper limit of the function domain. The default is 1.0.
    """
    def __init__(self, n_samples, n_points, x_inf=0.0, x_sup=1.0):
        self.n_samples = n_samples
        self.n_points = n_points
        self.axis_x = torch.linspace(x_inf, x_sup, n_points)
        self.data_shape = (n_samples, 1, n_points)
        self.data = torch.zeros(self.data_shape)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx]
    
    def compute_stats(self):
        return self.data.mean(), self.data.std()
    
class NormalizedDataset(Dataset):
    def __init__(self, base_dataset, eps=1e-5):
        self.base = base_dataset
        self.eps = eps
        self.mean, self.std = self.base.compute_stats()
        self.data = (self.base.data - self.mean) / (self.std + self.eps)
        self.axis_x = self.base.axis_x

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def compute_stats(self, eps=1e-5):
        return self.data.mean(), self.data.std()
    
    def denormalize(self, x_norm):
        return (x_norm * (self.std+self.eps)) + self.mean

class ExponentialDecayCosineDataset(BaseFunctionalDataset):
    """
    Dataset of fully stochastic cosine decay functions.

    Each sample is a one-dimensional function defined over the domain
    `self.axis_x`.

    Each function is generated as the sum of two sinusoidal components:

        f(x) = a + b * e^{-alpha * x} * cos (w * x + phase),

    where all parameters are randomly sampled for each sample:

    - `a ~ Uniform(`a_min`, `a_max`)`
    - `b ~ Uniform(`b_min`, `b_max`)`
    - `alpha ~ Uniform(`alpha_min`, `alpha_max`)`
    - `w ~ Uniform(`w_min`, `w_max`)`
    - `phase ~ Uniform(`phase_min`, `phase_min`)`

    This dataset provides higher variability and is useful for training
    models that must generalize across multiple periodic patterns.

    Parameters
    ----------
    n_samples : int
        Number of samples in the dataset.

    n_points : int
        Number of discretization points in the domain.

    x_inf : float, optional
        Lower bound of the domain. The default is 0.0.

    x_sup : float, optional
        Upper bound of the domain. The default is 1.0.
        
    a_min : float, optional
        Minimum vertical displacement (offset).
        The default is -1.0.
        
    a_max : float, optional
        Maximum vertical displacement (offset).
        The default is 1.0.

    b_min : float, optional
        Minimum initial amplitude.
        The default is 0.0.
        
    b_max : float, optional
        Maximum initial amplitude.
        The default is 1.0.
        
    alpha_min : float, optional
        Minimum decay rate.
        The default is 0.1.
        
    alpha_max : float, optional
        Maximum decay rate.
        The default is 1.0.

    w_min : float, optional
        Minimum angular frequency.
        The default is 2.0.
        
    w_max : float, optional
        Maximum angular frequency.
        The default is 4.0.

    phase_min : float, optional
        Minimum value of the phase value. The default is 0.0.
            
    phase_max : float, optional
        Maximum value of the phase value. The default is torch.2*pi.

    Returns
    -------
    torch.Tensor
        Tensor of shape `(1, n_points)` representing the function sample.

    Examples
    --------
    >>> dataset = ExponentialDecayCosineDataset(
    ...     n_samples=10,
    ...     n_points=100,
    ... )
    >>> x = dataset[0]
    >>> x.shape
    torch.Size([1, 100])
    """
    def __init__(
            self,
            n_samples,
            n_points,
            x_inf=0.0,
            x_sup=1.0,
            a_min=-1.0,
            a_max=1.0,
            b_min=0.0,
            b_max=1.0,
            alpha_min=0.1,
            alpha_max=1.0,
            w_min=2.0,
            w_max=4.0,
            phase_min=0.0,
            phase_max=2*torch.pi,
            seed=42,
            ):
        super().__init__(n_samples, n_points, x_inf, x_sup)
        self.a_min, self.a_max = a_min, a_max
        self.b_min, self.b_max = b_min, b_max
        self.alpha_min, self.alpha_max = alpha_min, alpha_max
        self.w_min, self.w_max = w_min, w_max
        self.phase_min, self.phase_max = phase_min, phase_max
        
        rng = torch.Generator().manual_seed(seed)
        
        a = torch.rand(n_samples, generator=rng) * (self.a_max - self.a_min) + self.a_min
        b = torch.rand(n_samples, generator=rng) * (self.b_max - self.b_min) + self.b_min
        
        alpha = torch.rand(n_samples, generator=rng) * (self.alpha_max - self.alpha_min) + self.alpha_min
        w = torch.rand(n_samples, generator=rng) * (self.w_max - self.w_min) + self.w_min
        
        phase = torch.rand(n_samples, generator=rng) * (self.phase_max - self.phase_min) + self.phase_min

        self.parameters = torch.stack(
            [a, b, alpha, w, phase],
            dim=1
        )
        
        data_shape = (n_samples, 1, 1)
        a = a.view(data_shape)
        b = b.view(data_shape)
        alpha = alpha.view(data_shape)
        w = w.view(data_shape)
        phase = phase.view(data_shape)

        x_grid = self.axis_x.view(1, 1, -1)
        self.data =  (
            a + b * torch.exp(-alpha * x_grid) * torch.cos(w * x_grid + phase)
        )

if __name__ == "__main__":
    import doctest
    doctest.testmod()
