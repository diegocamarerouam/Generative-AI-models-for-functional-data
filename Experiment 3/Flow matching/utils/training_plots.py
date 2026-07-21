# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 12:15:43 2026

@author: diego.camarero@estudiante.uam.es
"""
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch import Tensor

def plot_function_grid(
        functions: Tensor,
        n_plots: int = None,
        axis_x: Tensor = None,
        n_cols: int = 4,
        cell_size=(3.5, 2.5),
        ):
    """
    Illustrates some examples of the dataset `functions`.

    Parameters
    ----------
    functions : Tensor
        Shape (M, 1, D), where M := batch_size and D := dimension.
    n_plots : int, optional
        Number of examples to plot. The default is None.
    axis_x : Tensor, optional
        Grid to plot the functions. The default is None.
        It must have dimension D.
    n_cols : int, optional
        Number of columns of the plot_grid. The default is 4.
    cell_size : TYPE, optional
        Size for each sample to plot. The default is (3.5, 2.5).
    Returns
    -------
    None.
    
    Examples
    --------
    >>> axis_x = torch.linspace(0, 5, 100)
    >>> batch = torch.sin(axis_x).view(1, 1, 100)
    >>> batch.shape
    torch.Size([1, 1, 100])
    >>> plot_function_grid(
    ...     functions=batch, 
    ...     n_plots=12, 
    ...     axis_x=axis_x, 
    ...     n_cols=3
    ... )
    >>> batch = torch.sin(axis_x).view(1, 1, 100).expand(16, -1, -1)
    >>> batch.shape
    torch.Size([16, 1, 100])
    >>> plot_function_grid(
    ...     functions=batch, 
    ...     n_plots=12, 
    ...     axis_x=axis_x, 
    ...     n_cols=3
    ... )

    """

    functions = functions.detach().cpu()
    if axis_x is not None:
        axis_x = axis_x.detach().cpu()

    M, _, D = functions.shape
    n_plots = min(M, n_plots) if n_plots is not None else M
    
    if axis_x is None:
        axis_x = torch.linspace(0, 2 * torch.pi, D)
        
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize= (n_cols * cell_size[0], n_rows * cell_size[1]),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    
    if n_plots == 1:
        axes = np.array([axes])
        
    axes = axes.flatten()
    
    for i in range(n_rows * n_cols):
        ax = axes[i]
        if i < n_plots:
            func_values = functions[i, 0, :]
            ax.plot(axis_x, func_values.numpy(), lw=1.5)
            ax.set_title(f"Sample {i}")
            ax.grid(True, linestyle='--', alpha=0.6)
            if i >= (n_rows - 1) * n_cols:
                ax.set_xlabel("x")
            if i % n_cols == 0:
                ax.set_ylabel("f(x)")
        else:
            fig.delaxes(ax)
    fig.suptitle("Functional Data Samples", fontsize=16)
    plt.show()

def plot_flow_process(
        batch_functions: Tensor,
        axis_x: Tensor,
        num_steps: int=8,
        t_start: float=0.0,
        t_end: float=1.0,
        ):
    """
    Image grid illustrating the flow process for the functions in
    `batch_functions` from t_0 = `t_start` to t_end = `t_end` using
    linear scale.
    
    Parameters
    ----------
    batch_functions : Tensor
        Shape (n_samples, n_times, 1, pixel_dim).
        It contains the value of each function of shape D for each time step N.
    axis_x : Tensor
        Domain of the functions. It must have dimension D.
    num_steps : int, optional
        Number of subplots illustrating the diffusion process. The default is 8.
    t_start : float, optional
        Initial time of the process. The default is 0.0.
    t_end : float, optional
        Final time of the process. The default is 1.0.

    Returns
    -------
    None.
    """
    tensor_device = batch_functions.device
    axis_x = axis_x.detach().cpu()
    
    M, N, _, D = batch_functions.shape
    
    flip = 0
    if t_start > t_end:
        t_start, t_end = t_end, t_start
        flip = 1

    times = torch.linspace(t_start, t_end, N, device=tensor_device)

    t_samples = torch.linspace(t_start, t_end, num_steps, device=tensor_device)
    time_idxs = torch.searchsorted(times, t_samples)
    time_idxs = torch.unique(time_idxs)
    time_idxs[0] = 0
    
    if flip == 1:
        time_idxs = N - 1 - time_idxs
        time_idxs = torch.sort(time_idxs).values
            
    rows = min(M, 3)
    cols = len(time_idxs)

    fig, axs = plt.subplots(
        rows,
        cols,
        figsize=(cols * 2, rows * 2),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for m in range(rows):
        for i, t in enumerate(time_idxs):
            ax = axs[m, i]
            y = batch_functions[m, t, 0, :].detach().cpu()
            ax.plot(axis_x, y, lw=1.5)
            if m == 0 and flip == 0:
                ax.set_title(f"t={(times[t]):.4f}", fontsize=8)
            if m == 0 and flip == 1:
                ax.set_title(f"t={(t_end - times[t]):.4f}", fontsize=8)
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    import doctest
    doctest.testmod()