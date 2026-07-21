# -*- coding: utf-8 -*-
"""
Created on Mon May 25 17:43:47 2026

@author: diego.camarero@estudiante.uam.es
"""

import torch
import numpy as np
import seaborn as sns
import scipy as sc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

def parametric_model(x_grid, theta):
    x_grid = x_grid.to(theta.device)
        
    if theta.ndim == 1:
        theta = theta.unsqueeze(0)

    a = theta[:, 0].unsqueeze(1)
    b = theta[:, 1].unsqueeze(1)
    alpha = theta[:, 2].unsqueeze(1)
    w = theta[:, 3].unsqueeze(1)
    phase = theta[:, 4].unsqueeze(1)

    x_grid = x_grid.unsqueeze(0)
    return (a + b * torch.exp(-alpha * x_grid) * torch.cos(w * x_grid + phase)).unsqueeze(1)

def fit_dataset(data, x_grid, parametric_model):
    n_functions = data.shape[0]
    
    target_device = data.device
    if isinstance(x_grid, torch.Tensor):
        x_grid = x_grid.to(target_device)
    else:
        x_grid = torch.tensor(x_grid, dtype=torch.float32, device=target_device)

    theta = torch.zeros((n_functions, 5), device=target_device)
    theta[:, 0] = data.mean(dim=(1, 2))
    theta[:, 1] = data.std(dim=(1, 2))
    theta[:, 2] = 0.5
    theta[:, 3] = 3.0
    theta[:, 4] = 0.0
    
    theta.requires_grad_(True)
    optimizer = torch.optim.Adam([theta], lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=100
    )

    for step in range(1000):
        optimizer.zero_grad()
        y_hat = parametric_model(x_grid, theta)
        loss = ((data - y_hat) ** 2).mean(dim=(1, 2)).sum()
        loss.backward()
        optimizer.step()
        scheduler.step(loss.detach())

    with torch.no_grad():
        y_hat_final = parametric_model(x_grid, theta)
        mse_fit = ((data - y_hat_final) ** 2).mean(dim=(1, 2))
        
    return theta.detach(), mse_fit


def residual_MSE(samples, theta_hat, x_grid, parametric_model):
    if isinstance(x_grid, torch.Tensor):
        x_grid = x_grid.to(samples.device)
        
    samples_hat = parametric_model(x_grid, theta_hat)
    mse = torch.mean((samples - samples_hat) ** 2,dim=(-2, -1))
    return mse

def plot_mse_comparison(
    models: dict,
    bins: int = 20,
    cell_size=(7, 4.5),
    suptitle: str = "MSE distribution",
    colors: dict = None,
):
    labels = list(models.keys())
    n_models = len(labels)

    default_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = colors or {}
    model_colors = {
        label: colors.get(label, default_cycle[i % len(default_cycle)])
        for i, label in enumerate(labels)
    }

    computed = {}
    for label, cfg in models.items():
        mse = cfg["mse"]
        if isinstance(mse, torch.Tensor):
            mse = mse.detach().cpu().numpy()
        computed[label] = {"mse": mse, "median_mse": float(np.median(mse))}

    all_mse = np.concatenate([v["mse"] for v in computed.values()])
    bin_edges = np.histogram_bin_edges(all_mse, bins=bins)

    fig, axes = plt.subplots(
        1, n_models,
        figsize=(cell_size[0] * n_models, cell_size[1]),
        sharey=True,
        squeeze=False,
    )

    axes = axes[0]

    for col, label in enumerate(labels):
        ax = axes[col]
        color = model_colors[label]

        bbox = ax.get_position()
        x = (bbox.x0 + bbox.x1) / 2

        if col == 1:
          x += 0.04
        fig.text(
            x,
            bbox.y1 - 0.05,
            label,
            ha="center",
            va="bottom",
            fontsize=18,
            fontweight="bold",
            color='black',
        )

        c = computed[label]
        sns.histplot(c["mse"], stat="density", bins=bin_edges,
                     color=color, edgecolor="black", alpha=0.7, ax=ax)
        ax.axvline(c["median_mse"], color="red", linestyle="--", linewidth=2,
                   label=f"Median: {c['median_mse']:.2e}")
        ax.set_title(f"Median: {c['median_mse']:.2e}", fontsize=14)
        ax.set_xlabel("MSE")
        ax.legend(fontsize=9)
        if col == 0:
            ax.set_ylabel("Density", fontsize=13)

    fig.suptitle(suptitle, fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()
    plt.close(fig)
    return fig

def wasserstein_uniform_vs_samples(samples, a=0.0, b=1.0):
    """Computes W1 distance between Uniform(a, b) and the empirical distribution of `samples`."""
    if hasattr(samples, "detach"):
        samples = samples.detach().cpu().numpy()
    elif isinstance(samples, torch.Tensor):
        samples = samples.cpu().numpy()
    uniform_grid = np.linspace(a, b, num=5000)
    return sc.stats.wasserstein_distance(samples, uniform_grid)

def plot_theta_distributions(
    models: dict,
    theta_min,
    theta_max,
    param_names,
    bins: int = 20,
    cell_size=(3.6, 3.2),
    suptitle: str = "Parameter distributions",
    colors: dict = None,
):
    labels = list(models.keys())
    n_models = len(labels)
    n_params = len(param_names)

    default_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = colors or {}
    model_colors = {
        label: colors.get(label, default_cycle[i % len(default_cycle)])
        for i, label in enumerate(labels)
    }

    def _to_scalar(v):
        return v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)

    theta_min = [_to_scalar(v) for v in theta_min]
    theta_max = [_to_scalar(v) for v in theta_max]

    computed = {label: {} for label in labels}
    for label, cfg in models.items():
        theta = cfg["theta"]
        theta_np = theta.detach().cpu().numpy() if isinstance(theta, torch.Tensor) else theta
        for j in range(n_params):
            values = theta_np[:, j]
            w1 = wasserstein_uniform_vs_samples(values, theta_min[j], theta_max[j])
            computed[label][j] = {"values": values, "w1": w1}

    bin_edges_per_param = []
    for j in range(n_params):
        all_vals = np.concatenate([computed[label][j]["values"] for label in labels])
        bin_edges_per_param.append(np.histogram_bin_edges(all_vals, bins=bins))

    fig, axes = plt.subplots(
        n_models, n_params,
        figsize=(cell_size[0] * n_params, cell_size[1] * n_models),
        squeeze=False,
        sharey="col", sharex="col",
    )

    for row, label in enumerate(labels):
        color = model_colors[label]
        for col in range(n_params):
            ax = axes[row, col]
            c = computed[label][col]
            values = c["values"]
            t_min, t_max = theta_min[col], theta_max[col]

            ax.hist(values, density=True, bins=bin_edges_per_param[col],
                     color=color, edgecolor="black", alpha=0.7)

            xmin, xmax = ax.get_xlim()
            x_range = np.linspace(xmin, xmax, 1000)
            h = np.zeros_like(x_range)
            mask = (x_range >= t_min) & (x_range <= t_max)
            h[mask] = 1 / (t_max - t_min)
            ax.plot(x_range, h, color="red", linestyle="--", linewidth=1.5)

            ax.set_title(rf"$W_1={c['w1']:.3f}$", fontsize=10)

            if row == 0:
                ax.annotate(param_names[col], xy=(0.5, 1.25), xycoords="axes fraction",
                            ha="center", va="bottom", fontsize=16, fontweight="bold")
            if col == 0:
                ax.set_ylabel("Density", fontsize=11)
                ax.annotate(label, xy=(-0.30, 0.2), xycoords="axes fraction", rotation=90,
                            ha="center", va="bottom", fontsize=16, fontweight="bold", color='black')
            if row == n_models - 1:
                ax.set_xlabel(f"Value of {param_names[col]}", fontsize=11)

    handles = [
        Line2D([0], [0], color="red", linestyle="--", linewidth=1.5, label="Uniform prior"),
    ]

    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=1,
        bbox_to_anchor=(0.5, 1.00),
        frameon=True,
    )

    fig.suptitle(suptitle, fontsize=18, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()
    plt.close(fig)
    return fig
