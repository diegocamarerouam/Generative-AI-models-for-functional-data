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

class ConstantUniformDataset(BaseFunctionalDataset):
    """
    Dataset of stochastic constant-valued functions.

    Each sample is a one-dimensional function defined over the domain
    `self.axis_x`.
    
    Each function has a constant value sampled uniformly
    between `f_min_value`and `f_max_value`.

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

    f_min_value : float, optional
        Minimum value of the constant function. The default is 0.0.
        
    f_max_value : float, optional
        Maximum value of the constant function. The default is 1.0.

    Returns
    -------
    torch.Tensor
        Tensor of shape `(1, n_points)` representing the function sample.

    Examples
    --------
    >>> dataset = ConstantUniformDataset(
    ...     n_samples=10,
    ...     n_points=50,
    ...     f_min_value=0.0,
    ...     f_max_value=1.0,
    ... )
    >>> x = dataset[0]
    >>> x.shape
    torch.Size([1, 50])
    """
    def __init__(
            self,
            n_samples,
            n_points,
            x_inf=0.0,
            x_sup=1.0,
            f_min_value=0.0,
            f_max_value=1.0,
            seed=42,
            ):
        super().__init__(n_samples, n_points, x_inf, x_sup)
        
        self.f_min_value = f_min_value
        self.f_max_value = f_max_value
        
        rng = torch.Generator().manual_seed(seed)
        h = torch.rand(n_samples, generator=rng) * (f_max_value - f_min_value) + f_min_value
        h = h.view(n_samples, 1, 1)
        
        self.data = h * torch.ones(self.data_shape)
        
if __name__ == "__main__":
    import doctest
    doctest.testmod()
