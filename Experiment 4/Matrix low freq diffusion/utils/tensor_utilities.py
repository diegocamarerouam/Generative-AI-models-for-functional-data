# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 21:19:56 2026

@author: diego.camarero@estudiante.uam.es

Utilities for Spectral Manipulation of Circulant Matrices.

Implementation on Pytorch.

This module provides tools for constructing, transforming, and integrating 
functional representations in the Fourier domain. It is specifically designed 
to handle the unique ordering of eigenvalues and conjugate pairs required 
for real-valued circulant structures.

Functions:
    1. `fourier_matrix`: Generates Fourier transformation matrices 
      where frequencies are arranged to group real modes and complex pairs.

This module can be used for any system involving circulant operators, 
including Signal Processing, Differential Equations, or Stochastic Processes.
"""

import torch
from torch import Tensor

def fourier_matrix(
        D: int,
        device: str | torch.device = "cpu",
        ) -> Tensor:
    """
    Generates a reordered unitary DFT matrix where frequencies are arranged as:
        
        if D is even:
            (0, 1, D-1, 2, D-2, ..., D/2 - 1, D - (D/2 - 1), D/2)
            
        if D is odd:
            (0, 1, D-1, 2, D-2, ..., D/2 - 1, D - (D/2 - 1))
        
        where frequency y_m = (1, w^m, w^{2m}, ..., w^{(D-1)m})
        
        and w = exp(-2 * pi * j / D), where j is the imaginary unit.

    
    Therefore,
    - DC mode first
    - then conjugate pairs (1, D-1, 2, D-2, ..., D//2 - 1, D - (D//2 - 1))
    - Nyquist mode last (if D is even)
    
    This corresponds to grouping eigenvalues into:
        DC mode + conjugate pairs + Nyquist mode
    
    Parameters
    ----------
    D : int
        Dimension of the square matrix.
        
    Returns
    -------
    (Tensor)
        Reordered Fourier matrix of shape (D, D).
        
    Examples
    --------
        >>> import torch
        >>> D = 2
        >>> F = fourier_matrix(D)
        >>> F.shape
        torch.Size([2, 2])
        >>> torch.allclose(F.conj().T @ F, torch.eye(D, dtype=F.dtype),
        ...               atol=1e-6)
        True
        >>> torch.allclose(F @ F.conj().T, torch.eye(D, dtype=F.dtype),
        ...                atol=1e-6)
        True
        
        >>> D = 5
        >>> F = fourier_matrix(D)
        >>> F.shape
        torch.Size([5, 5])
        >>> torch.allclose(F.conj().T @ F, torch.eye(D, dtype=F.dtype),
        ...               atol=1e-6)
        True
        >>> torch.allclose(F @ F.conj().T, torch.eye(D, dtype=F.dtype),
        ...                atol=1e-6)
        True
    """
    coords = torch.arange(D, device=device)
    I, J = torch.meshgrid(coords, coords, indexing='ij')
    
    norm = torch.sqrt(torch.tensor(D, dtype=torch.float32, device=device))
    phase = 2j * torch.pi * I * J / D
    F = 1 / norm * torch.exp(phase)

    perm_idxs = [0]
    for k in range(1, (D+1) // 2):
        perm_idxs.append(k)
        perm_idxs.append(D-k)
    
    if D % 2 == 0:
        perm_idxs.append(D // 2)
    perm_idxs= torch.tensor(perm_idxs, device=device)
    F_ordered = F[perm_idxs, :]
    
    return F_ordered

if __name__ == "__main__":
    import doctest
    doctest.testmod()