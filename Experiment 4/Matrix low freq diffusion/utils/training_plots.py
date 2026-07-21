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
    
def plot_ou_diffusion_process(
        batch_functions: Tensor,
        axis_x: Tensor,
        num_steps: int = 8,
        t_start: float = 0.0,
        t_end: float = 1.0,
        tweedie_final: bool = False,
        ):
    tensor_device = batch_functions.device
    axis_x = axis_x.detach().cpu()
    M, N, _, D = batch_functions.shape
    N_traj = N - 1 if tweedie_final else N

    times = torch.linspace(t_start, t_end, N_traj, device=tensor_device)

    n_traj_panels = num_steps - 1 if tweedie_final else num_steps

    nonzero_times = times.abs()
    nonzero_times = nonzero_times[nonzero_times > 0]
    lo = nonzero_times.min().item()
    hi = times.abs().max().item()

    oversample = n_traj_panels * 3
    t_samples = torch.logspace(
        torch.log10(torch.tensor(lo, device=tensor_device)),
        torch.log10(torch.tensor(hi, device=tensor_device)),
        steps=oversample,
        device=tensor_device,
    )
    diffs = (times.abs().unsqueeze(0) - t_samples.unsqueeze(1)).abs()
    candidate_idxs = torch.unique(diffs.argmin(dim=1))
    candidate_idxs, _ = torch.sort(candidate_idxs)

    pick = torch.linspace(0, len(candidate_idxs) - 1, n_traj_panels).round().long()
    time_idxs = candidate_idxs[pick]

    time_idxs[0] = 0
    time_idxs[-1] = N_traj - 1

    if tweedie_final:
        time_idxs = torch.cat(
            [time_idxs, torch.tensor([N - 1], device=tensor_device)]
        )

    rows = min(M, 3)
    cols = len(time_idxs)
    fig, axs = plt.subplots(
        rows, cols, figsize=(cols * 2, rows * 2),
        sharex=True, sharey=True, squeeze=False,
    )
    time_idxs = time_idxs.cpu()
    times = times.cpu()
    for m in range(rows):
        for i, t in enumerate(time_idxs):
            ax = axs[m, i]
            y = batch_functions[m, t, 0, :].detach().cpu().numpy()
            is_tweedie = tweedie_final and (t.item() == N - 1)
            ax.plot(axis_x, y, lw=1.5, color="tab:blue", linestyle="-")
            if m == 0:
                if is_tweedie:
                    ax.set_title("t=0.0000\n(Tweedie)", fontsize=8)
                else:
                    ax.set_title(f"t={times[t]:.4f}", fontsize=8)
    plt.tight_layout()
    plt.show()

def plot_distribution_convergence(
    batch_functions: Tensor,
    loc_stationary: float = 0.0,
    scale_stationary: float = 1.0,
    bins: int = 20,
):
    final_step = batch_functions[:, -1, 0, :]
    final_step = final_step.detach().cpu()

    sample_means = torch.mean(final_step, dim=1).numpy()
    sample_stds = torch.std(final_step, dim=1).numpy()

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    x_min_m, x_max_m = sample_means.min(), sample_means.max()
    axs[0].hist(sample_means, bins=bins, range=(x_min_m, x_max_m), density=True,
                alpha=0.5, color='blue', label='Empirical Means', edgecolor='black')
    axs[0].axvline(x=loc_stationary, color='red', linestyle='--', linewidth=2, label=f'Expected Mean ({loc_stationary})')
    axs[0].set_title('Distribution of Sample Means')
    axs[0].set_xlabel('Mean Value')
    axs[0].set_ylabel('Density')
    axs[0].legend()

    x_min_s, x_max_s = sample_stds.min(), sample_stds.max()
    axs[1].hist(sample_stds, bins=bins, range=(x_min_s, x_max_s), density=True,
                alpha=0.5, color='orange', label='Empirical Stds', edgecolor='black')
    axs[1].axvline(x=scale_stationary, color='red', linestyle='--', linewidth=2, label=f'Expected Std ({scale_stationary})')
    axs[1].set_title('Distribution of Sample Standard Deviations')
    axs[1].set_xlabel('Standard Deviation Value')
    axs[1].set_ylabel('Density')
    axs[1].legend()
    
    fig.suptitle('Convergence of Sample Metrics vs Stationary Distribution', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
if __name__ == "__main__":
    import doctest
    doctest.testmod()