# -*- coding: utf-8 -*-
"""
Created on Mon May 25 17:43:47 2026

@author: diego.camarero@estudiante.uam.es
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sc
import numpy as np

def mean_and_MSE(samples: torch.Tensor) -> torch.Tensor:
    means = torch.mean(samples, dim=-1, keepdim=True)
    mse = torch.mean((samples - means) ** 2, dim=(-2, -1))
    return means.flatten(), mse

def wasserstein_uniform_vs_samples(samples, a=0.0, b=1.0):
    """
    Computes W1 distance between:
    - Uniform(a, b)
    - Empirical distribution of `means`
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
        samples,
        theoretical_min,
        theoretical_max,
        bins=20,
        data_label=None,
        ) -> None:
    means, mse = mean_and_MSE(samples)
    
    means = means.detach().cpu().numpy()
    median_mse = torch.median(mse).detach().cpu().item()
    mse = mse.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=False)
    ax1 = axes[0]
    ax1.hist(
        means,
        density=True,
        bins=bins,
        color="blue",
        edgecolor="black",
        alpha=0.7,
        label=data_label,
    )
    xmin, xmax = ax1.get_xlim()
    x_range = np.linspace(xmin, xmax, 1000)
    h = np.zeros_like(x_range)
    mask = (x_range >= theoretical_min) & (x_range <= theoretical_max)
    h[mask] = 1 / (theoretical_max - theoretical_min)
    
    ax1.plot(
        x_range,
        h,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Theoretical uniform",
    )
    ax1.set_ylabel("Density", fontsize=14)
    w1 = wasserstein_uniform_vs_samples(means, theoretical_min, theoretical_max)
    ax1.set_title(
        f"Means distribution: $\\mathbf{{W_1}}={w1:.3f}$",
        fontweight="bold",
    )
    ax1.legend()

    ax2 = axes[1]
    sns.histplot(
        mse,
        stat="density",
        bins=bins,
        color="purple",
        edgecolor="black",
        alpha=0.7,
        ax=ax2,
    )
    ax2.axvline(
        median_mse,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Median: {median_mse:.2e}",
    )

    ax2.set_title(
        "MSE distribution",
        fontweight="bold",
    )
    ax2.set_ylabel("Density", fontsize=14)
    ax2.legend()

    plt.tight_layout()
    plt.show()