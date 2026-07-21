# -*- coding: utf-8 -*-
"""
Created on Mon May 25 17:43:47 2026

@author: diego.camarero@estudiante.uam.es
"""

import torch
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import seaborn as sns


def phase_parametric_model(x_grid, theta, a_1, a_2, w_1, w_2):
    device = theta.device
    if isinstance(a_1, torch.Tensor): a_1 = a_1.to(device)
    if isinstance(a_2, torch.Tensor): a_2 = a_2.to(device)
    if isinstance(w_1, torch.Tensor): w_1 = w_1.to(device)
    if isinstance(w_2, torch.Tensor): w_2 = w_2.to(device)
    
    if theta.ndim == 1:
        theta = theta.unsqueeze(0)
    phase = theta[:, 0].unsqueeze(1)
    x_grid = x_grid.unsqueeze(0)
    
    return (a_1 * torch.cos(w_1 * x_grid + phase) +  a_2 * torch.cos(w_2 * x_grid + phase)).unsqueeze(1)

def fit_phases_dataset(data, x_grid, parametric_model, device="cpu"):
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)
    data = data.to(device)
    
    n_functions = data.shape[0]

    theta = torch.zeros((n_functions, 1), device=device)
    theta[:, 0] = torch.pi
    
    if not isinstance(x_grid, torch.Tensor):
        x_grid = torch.tensor(x_grid, dtype=torch.float32)
    x_grid = x_grid.to(device)
    
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


def residual_MSE(samples, theta_hat, x_grid, parametric_model, device="cpu"):
    if not isinstance(samples, torch.Tensor):
        samples = torch.tensor(samples, dtype=torch.float32)
    samples = samples.to(device)
    
    if not isinstance(theta_hat, torch.Tensor):
        theta_hat = torch.tensor(theta_hat, dtype=torch.float32)
    theta_hat = theta_hat.to(device)
    
    if not isinstance(x_grid, torch.Tensor):
        x_grid = torch.tensor(x_grid, dtype=torch.float32)
    x_grid = x_grid.to(device)
        
    samples_hat = parametric_model(x_grid, theta_hat)
    mse = torch.mean((samples - samples_hat) ** 2, dim=(-2, -1))
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

        if col == 0:
            x += -0.04
      
        if col == 1:
            x += 0.00
            
        if col == 2:
            x += 0.02
            
        if col == 3:
            x += 0.06
            
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

def amplitudes_parametric_model(x_grid, theta, w_1, w_2, phase_hat):
    device = theta.device
    
    if isinstance(w_1, torch.Tensor): w_1 = w_1.to(device)
    if isinstance(w_2, torch.Tensor): w_2 = w_2.to(device)
    if isinstance(phase_hat, torch.Tensor): phase_hat = phase_hat.to(device)
    
    if theta.ndim != phase_hat.ndim:
        raise ValueError("The number of dimensions of adjusted phases and theta must be equal")
    if theta.ndim > 1:
        assert theta.shape[0] == phase_hat.shape[0]
    if theta.ndim == 1:
        theta = theta.unsqueeze(0)
        phase_hat = phase_hat.unsqueeze(0)
    
    amp_1 = theta[:, 0].unsqueeze(1)
    amp_2 = theta[:, 1].unsqueeze(1)
    x_grid = x_grid.unsqueeze(0)

    return (amp_1 * torch.cos(w_1 * x_grid + phase_hat) + amp_2 * torch.cos(w_2 * x_grid + phase_hat)).unsqueeze(1)

def fit_amplitudes_dataset(data, x_grid, parametric_model, device="cpu"):
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)
    data = data.to(device)
    
    n_functions = data.shape[0]

    theta = torch.zeros((n_functions, 2), device=device)
    theta[:, 0] = 0.5
    theta[:, 1] = 0.5
    
    if not isinstance(x_grid, torch.Tensor):
        x_grid = torch.tensor(x_grid, dtype=torch.float32)
    x_grid = x_grid.to(device)
    
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

def fit_amplitudes(
    samples,
    x_grid,
    parametric_model,
    device="cpu",
):
    amplitudes_hat, _ = fit_amplitudes_dataset(
        samples,
        x_grid,
        parametric_model,
        device=device,
    )

    a1_hat = amplitudes_hat[:, 0]
    a2_hat = amplitudes_hat[:, 1]

    return a1_hat, a2_hat

def wasserstein_uniform_vs_samples(samples, a=0.0, b=1.0):
    if hasattr(samples, "detach"):
        samples = samples.detach().cpu().numpy()
    elif isinstance(samples, torch.Tensor):
        samples = samples.cpu().numpy()
    uniform_grid = np.linspace(a, b, num=5000)
    return sc.stats.wasserstein_distance(samples, uniform_grid)


def _is_scalar(x):
    if isinstance(x, torch.Tensor):
        return x.numel() == 1
    if isinstance(x, np.ndarray):
        return x.size == 1
    return isinstance(x, (int, float))


def _to_float(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().item()
    if isinstance(x, np.ndarray):
        return float(x.reshape(-1)[0])
    return float(x)


def _to_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def plot_phase_amplitude_grid(
    models: dict,
    phase_min,
    phase_max,
    theoretical_a1,
    theoretical_a2,
    bins: int = 30,
    cell_size=(5.5, 3.8),
    suptitle: str = "Phase and amplitude distributions",
    a1_label: str = r"$A_{2\pi}$",
    a2_label: str = r"$A_{14\pi}$",
    colors: dict = None,
):
    labels = list(models.keys())
    n_models = len(labels)
    phase_min, phase_max = _to_float(phase_min), _to_float(phase_max)
    theoretical_a1, theoretical_a2 = _to_float(theoretical_a1), _to_float(theoretical_a2)

    default_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors_ = colors or {}
    model_colors = {
        label: colors_.get(label, default_cycle[i % len(default_cycle)])
        for i, label in enumerate(labels)
    }
    a1_color, a2_color = "tab:blue", "tab:orange"

    phase_data = {}
    for label, cfg in models.items():
        theta = cfg.get("theta")
        if theta is not None:
            values = _to_numpy(theta).reshape(-1)
            w1 = wasserstein_uniform_vs_samples(values, phase_min, phase_max)
            phase_data[label] = {"values": values, "w1": w1}

    phase_bin_edges = None
    if phase_data:
        all_phase = np.concatenate([v["values"] for v in phase_data.values()])
        phase_bin_edges = np.histogram_bin_edges(all_phase, bins=bins)

    amp_data = {}
    for label, cfg in models.items():
        a1, a2 = cfg["a1"], cfg["a2"]
        scalar_mode = _is_scalar(a1) and _is_scalar(a2)
        if scalar_mode:
            amp_data[label] = {"scalar": True, "a1": _to_float(a1), "a2": _to_float(a2)}
        else:
            a1_np, a2_np = _to_numpy(a1).reshape(-1), _to_numpy(a2).reshape(-1)
            if a1_np.std() < 1e-6:
                a1_np = np.full_like(a1_np, a1_np.mean())
            if a2_np.std() < 1e-6:
                a2_np = np.full_like(a2_np, a2_np.mean())
            amp_data[label] = {"scalar": False, "a1": a1_np, "a2": a2_np}

    array_rows = [v for v in amp_data.values() if not v["scalar"]]
    amp_bin_edges = None
    if array_rows:
        all_amp = np.concatenate([np.concatenate([v["a1"], v["a2"]]) for v in array_rows])
        amp_bin_edges = np.histogram_bin_edges(all_amp, bins=bins)

    fig, axes = plt.subplots(
        n_models, 2,
        figsize=(cell_size[0] * 2, cell_size[1] * n_models),
        squeeze=False,
        sharex="col", sharey="col",
    )
    for ax in axes[:, 1]:
        ax.tick_params(labelleft=True)

    for row, label in enumerate(labels):
        color = model_colors[label]

        ax1 = axes[row, 0]

        if label in phase_data:
            d = phase_data[label]
            ax1.hist(d["values"], density=True, bins=phase_bin_edges,
                      color=color, edgecolor="black", alpha=0.7)
            xmin, xmax = ax1.get_xlim()
            x_range = np.linspace(xmin, xmax, 1000)
            h = np.zeros_like(x_range)
            mask = (x_range >= phase_min) & (x_range <= phase_max)
            h[mask] = 1 / (phase_max - phase_min)
            ax1.plot(x_range, h, color="red", linestyle="--", linewidth=1.5, label="Uniform prior")
            ax1.set_title(rf"$W_1={d['w1']:.3f}$", fontsize=11)
            ax1.legend(fontsize=8)
        else:
            ax1.axis("off")
            ax1.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=11, color="gray", transform=ax1.transAxes)

        ax1.set_ylabel("Density", fontsize=11)
        if row == n_models - 1:
            ax1.set_xlabel("Phase value")

        ax2 = axes[row, 1]
        d = amp_data[label]
        if d["scalar"]:
            ax2.axvline(d["a1"], color=a1_color, linestyle="-", linewidth=2, label=f"{a1_label} = {d['a1']:.3f}")
            ax2.axvline(d["a2"], color=a2_color, linestyle="-", linewidth=2, label=f"{a2_label} = {d['a2']:.3f}")
            ax2.set_title(
                rf"{a1_label} = {d['a1']}"
                rf"    {a2_label} = {d['a2']}",
                fontsize=11,
            )
        else:
            ax2.hist(d["a1"], density=True, bins=amp_bin_edges, color=a1_color,
                      edgecolor="black", alpha=0.6, label=a1_label)
            ax2.hist(d["a2"], density=True, bins=amp_bin_edges, color=a2_color,
                      edgecolor="black", alpha=0.6, label=a2_label)
            ax2.axvline(theoretical_a1, color=a1_color, linestyle="--", linewidth=1.5)
            ax2.axvline(theoretical_a2, color=a2_color, linestyle="--", linewidth=1.5)
            if row != 1:
              ax2.set_title(
                  rf"{a1_label} = {d['a1'].mean():.3f} $\pm$ {d['a1'].std():.3f}" "\n"
                  rf"{a2_label} = {d['a2'].mean():.3f} $\pm$ {d['a2'].std():.3f}",
                  fontsize=11,
                  )
            else:
              ax2.set_title(
                  rf"{a1_label} = {d['a1'].mean():.3f} $\pm$ {d['a1'].std():.3e}""\n"
                  rf"{a2_label} = {d['a2'].mean():.3f} $\pm$ {d['a2'].std():.3e}",
                  fontsize=11,
                  )

        ax2.set_ylabel("Density", fontsize=11)
        ax2.legend(fontsize=8)
        if row == n_models - 1:
            ax2.set_xlabel("Amplitude value")

    fig.suptitle(suptitle, fontsize=18, fontweight="bold", y=1.01)
    plt.tight_layout(h_pad=2.0)

    for row, label in enumerate(labels):
      bbox = axes[row, 0].get_position()
      y = (bbox.y0 + bbox.y1) / 2
      fig.text(
          bbox.x0 - 0.08,
          y,
          label,
          rotation=90,
          ha="center",
          va="center",
          fontsize=16,
          fontweight="bold",
          color='black',
          )

    for col, label in enumerate([r"Phase $\phi$", f"Amplitudes {a1_label}, {a2_label}"]):
      bbox = axes[0, col].get_position()
      x = (bbox.x0 + bbox.x1) / 2
      fig.text(
          x,
          bbox.y1 + 0.02,
          label,
          ha="center",
          va="center",
          fontsize=16,
          fontweight="bold",
          )
      if col==1:
        fig.text(
          x+0.14,
          bbox.y0 - 0.029,
          r"(mean$\pm$std)",
          ha="center",
          va="center",
          fontsize=10,
          )

    plt.show()
    return fig