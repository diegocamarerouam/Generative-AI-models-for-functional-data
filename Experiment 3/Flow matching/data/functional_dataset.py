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

class PiecewiseConstantDataset(BaseFunctionalDataset):
    """ 
    Dataset of stochastic piecewise constant funtions.

    Each sample is a one-dimensional function defined over the domain
    self.axis_x`.
    
    Each function creation depends on three parameters:
        - non-zero value: uniform distribution (f_min_value, f_max_value)
        - lower limit: uniform distribution (x_inf, x_sup/2)
        - upper limit: uniform distribution (lower x_sup/2, x_sup),
        where [a, b] = self.axis.
        
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
        Minimum value of the non-zero value. The default is 0.0.
            
    f_max_value : float, optional
        Maximum value of the non-zero value. The default is 1.0.

    Returns
    -------
    torch.Tensor
    Tensor of shape `(1, n_points)` representing the function sample.

    Examples
    --------
        >>> dataset = PiecewiseConstantDataset(
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
            split_point=None,
            seed=42,
            ):
        super().__init__(n_samples, n_points, x_inf, x_sup)
        if split_point is None:
            split_point = 0.5 * (x_inf + x_sup)
            
        if not (0 < split_point < n_points):
            raise ValueError(f"split_point debe estar entre 1 y {n_points - 1}")

        self.f_min_value = f_min_value
        self.f_max_value = f_max_value
        
        rng = torch.Generator().manual_seed(seed)
        h = (
            torch.rand(n_samples, generator=rng) * (f_max_value - f_min_value)
            + f_min_value
        )
        h = h.view(n_samples, 1, 1)
        
        low_idx = torch.randint(0, split_point, (n_samples, 1, 1), generator=rng)
        high_idx = torch.randint(split_point, n_points, (n_samples, 1, 1), generator=rng)
        
        idx_grid = torch.arange(n_points).view(1, 1, -1)
        mask = (idx_grid >= low_idx) & (idx_grid <= high_idx)
        self.data = torch.where(mask, h, torch.zeros(self.data_shape))
        self.parameters = torch.stack(
            [h.view(-1), low_idx.view(-1), high_idx.view(-1)],
            dim=1
        )

class SkylineDataset(BaseFunctionalDataset):
    """ 
    Dataset of stochastic skyline-like piecewise constant functions.
    Each sample contains multiple consecutive step functions (buildings).
    """
    def __init__(
        self,
        n_samples,
        n_points,
        x_inf=0.0,
        x_sup=1.0,
        f_min_value=0.0,
        f_max_value=1.0,
        n_steps=5,
        min_width=0.05,
        seed=42,
    ):
        super().__init__(n_samples, n_points, x_inf, x_sup)
        
        rng = torch.Generator().manual_seed(seed)

        heights = (
            torch.rand(n_samples, n_steps, generator=rng) * (f_max_value - f_min_value)
            + f_min_value
        )
        
        heights[:, 0] /= 3
        heights[:, -1] /= 3

        remaining_width = 1.0 - (n_steps * min_width)
        raw_widths = torch.rand(n_samples, n_steps, generator=rng)
        widths = min_width + (raw_widths / raw_widths.sum(dim=1, keepdim=True)) * remaining_width

        cum_widths = torch.cumsum(widths, dim=1)
        boundaries = (cum_widths[:, :-1] * n_points).long()

        zeros = torch.zeros(n_samples, 1, dtype=torch.long)
        full_stops = torch.full((n_samples, 1), n_points, dtype=torch.long)
        boundaries = torch.cat([zeros, boundaries, full_stops], dim=1)
        
        idx_grid = torch.arange(n_points).view(1, 1, -1)
        left_bounds = boundaries[:, :-1].unsqueeze(-1)
        right_bounds = boundaries[:, 1:].unsqueeze(-1)
        
        masks = (idx_grid >= left_bounds) & (idx_grid < right_bounds)
        heights_expanded = heights.unsqueeze(-1)
        self.data = torch.sum(masks * heights_expanded, dim=1)
        self.data = self.data.unsqueeze(1)
        
if __name__ == "__main__":
    import doctest
    doctest.testmod()
