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

class TwoFreqCosineDataset(BaseFunctionalDataset):
    """
    Dataset of stochastic phase-varying sinusoidal functions.

    Each sample is a one-dimensional function defined over the domain
    `self.axis_x`.

    Each function is generated as the sum of two cosine components:
        f(x) = a_1 * cos(w_1 * x + phase)
             + a_2 * cos(w_2 * x + phase)

    where the phase is randomly sampled for each sample as:

        phase ~ Uniform(`phase_min`, `phase_max`)

    while amplitudes and frequencies remain fixed.

    This dataset is useful for studying phase invariance and robustness
    of machine learning models for periodic signals.

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

    a_1 : float, optional
        Amplitude of the first cosine component.
        The default is 1.0.

    a_2 : float, optional
        Amplitude of the second cosine component.
        The default is 0.3.

    w_1 : float, optional
        Angular frequency of the first cosine component.
        The default is 2*torch.pi.

    w_2 : float, optional
        Angular frequency of the second cosine component.
        The default is 14*torch.pi.
        
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
    >>> dataset = TwoFreqCosineDataset(
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
            a_1=1.0,
            a_2=0.3,
            w_1=2*torch.pi,
            w_2=14*torch.pi,
            phase_min=0.0,
            phase_max=2*torch.pi,
            seed=42,
            ):
        super().__init__(n_samples, n_points, x_inf, x_sup)
        self.a_1, self.a_2 = a_1, a_2
        self.w_1, self.w_2 = w_1, w_2
        self.phase_min = phase_min
        self.phase_max = phase_max
        
        rng = torch.Generator().manual_seed(seed)
        phases = torch.rand(n_samples, generator=rng) * (self.phase_max - self.phase_min) + self.phase_min
        self.parameters = phases.view(n_samples, 1)
        phases = phases.view(n_samples, 1, 1)
        x_grid = self.axis_x.view(1, 1, -1)
        self.data = (
            self.a_1 * torch.cos(self.w_1 * x_grid + phases)
            + self.a_2 * torch.cos(self.w_2 * x_grid + phases)
        )
        
if __name__ == "__main__":
    import doctest
    doctest.testmod()
