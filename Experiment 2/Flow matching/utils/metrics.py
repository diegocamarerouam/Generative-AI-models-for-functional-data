# -*- coding: utf-8 -*-
"""
Created on Mon May 25 17:43:47 2026

@author: diego.camarero@estudiante.uam.es
"""

import torch
import numpy as np
import seaborn as sns
import scipy as sc
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from torch.optim import Adam
import torch.nn.functional as F

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

def wasserstein_uniform_vs_samples(samples, a=0.0, b=1.0):
    """
    Computes W1 distance between:
    - Uniform(a, b)
    - Empirical distribution of `samples`
    """
    if hasattr(samples, 'detach'):
        samples = samples.detach().cpu().numpy()
    elif isinstance(samples, torch.Tensor):
        samples = samples.cpu().numpy()
    uniform_grid = np.linspace(a, b, num=5000)
    w1_distance = sc.stats.wasserstein_distance(samples, uniform_grid)
    return w1_distance

def plot_loss_history(loss_history: torch.Tensor):
    plt.plot(loss_history["epoch"], loss_history["loss"], color="tab:blue", markersize=2, linestyle="-")
    plt.yscale('log')
    plt.title("Training Loss Progression", fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.grid(True, linestyle="--", alpha=0.6)

def plot_training_evolution(
    data,
    metrics,
    titles,
    yscales='log',
):
    if yscales == 'log':
        yscales = ['log'] * len(metrics)
    elif yscales == 'linear':
        yscales = ['linear'] * len(metrics)
        
    fig, axs = plt.subplots(1, len(metrics), figsize=(4*len(metrics), 4), squeeze=False)
    
    for j, metric in enumerate(metrics):
        ax = axs[0, j] 
        
        y_values = data[metric]
        if isinstance(y_values, torch.Tensor):
            y_values = y_values.detach().cpu().numpy()
        elif isinstance(y_values, list) and len(y_values) > 0 and isinstance(y_values[0], torch.Tensor):
            y_values = [v.detach().cpu().item() for v in y_values]
            
        ax.plot(data["epoch"], y_values, marker="o")
        ax.set_yscale(yscales[j])
        ax.set_title(titles[j])
        ax.set_xlabel("Epoch")
        # ax.grid(True)

    plt.tight_layout()
    plt.show()

def plot_hist_vs_uniform(
        values,
        theoretical_min,
        theoretical_max,
        bins=20,
        data_label=None,
        ax=None,
        ) -> None:

    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    if isinstance(theoretical_min, torch.Tensor):
        theoretical_min = theoretical_min.detach().cpu().item()
    if isinstance(theoretical_max, torch.Tensor):
        theoretical_max = theoretical_max.detach().cpu().item()
        
    if ax is None:
        ax = plt.gca()
        
    ax.hist(
        values,
        density=True,
        bins=bins,
        color="blue",
        edgecolor="black",
        alpha=0.7,
        label=data_label,
    )
    xmin, xmax = ax.get_xlim()
    x_range = np.linspace(xmin, xmax, 1000)
    h = np.zeros_like(x_range)
    mask = (x_range >= theoretical_min) & (x_range <= theoretical_max)
    h[mask] = 1 / (theoretical_max - theoretical_min)
    ax.plot(
        x_range,
        h,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Theoretical uniform",
    )
    ax.set_ylabel("Density", fontsize=14)
    w1 = wasserstein_uniform_vs_samples(values, theoretical_min, theoretical_max)
    ax.set_title(
        f"Means distribution: $\\mathbf{{W_1}}={w1:.3f}$",
        fontweight="bold",
    )
    
def plot_metric_distributions(
    theta_hat,
    theta_min,
    theta_max,
    metrics,
    titles,
    data_label=None,
):

    n_metrics = len(metrics)
    fig, ax = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 4))

    if n_metrics == 1:
        ax = [ax]

    for j, metric in enumerate(metrics):
        val_slice = theta_hat[:, j].detach().cpu().numpy() if isinstance(theta_hat, torch.Tensor) else theta_hat[:, j]
        t_min = theta_min[j].detach().cpu().item() if isinstance(theta_min[j], torch.Tensor) else theta_min[j]
        t_max = theta_max[j].detach().cpu().item() if isinstance(theta_max[j], torch.Tensor) else theta_max[j]
        
        plot_hist_vs_uniform(
            val_slice,
            theoretical_min=t_min,
            theoretical_max=t_max,
            ax=ax[j],
            data_label=data_label,
        )
        w1 = sc.stats.wasserstein_distance(val_slice, torch.linspace(t_min, t_max, 5000))
        ax[j].set_title(
            f"{titles[j]}: $\\mathbf{{W_1}}={w1:.3f}$",
            fontweight="bold",
        )
        ax[j].legend()
    plt.tight_layout()
    plt.show()

def plot_mse_histogram(
        mse,
        bins: int=20,
        data_label=None,
        ):
    if isinstance(mse, torch.Tensor):
        mse = mse.detach().cpu().numpy()
        
    mse_mean = mse.mean()
    
    plt.hist(
        mse,
        density=True,
        bins=bins,
        color="purple",
        edgecolor="black",
        alpha=0.7,
        label=data_label,
    )
    plt.axvline(
        mse.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mse_mean:.2e}",
    )

    plt.title(
        "MSE distribution",
        fontweight="bold",
    )
    plt.ylabel("Density", fontsize=14)
    plt.legend()
