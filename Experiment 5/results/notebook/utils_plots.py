# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 18:13:01 2026

@author: diego.camarero@estudiante.uam.es
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


def plot_function_grid(
    functions_dict: dict,
    axis_x: torch.Tensor,
    seed: int,
    mode: str = "separated",
    n_samples: int = 3,
    cell_size=(3.5, 2.5),
    suptitle: str = "Functional Data Samples",
):

    if mode not in ("separated", "overlapped"):
        raise ValueError(f"mode must be 'separated' or 'overlapped', got {mode!r}")

    labels = list(functions_dict.keys())
    n_models = len(labels)

    tensors = {k: v.detach().cpu() for k, v in functions_dict.items()}    
    axis_x = axis_x.detach().cpu()

    sample_indices = {}
    for k, v in tensors.items():
        N = v.shape[0]
        n = min(n_samples, N)
        sample_indices[k] = np.arange(n)

    if mode == "separated":
        n_rows, n_cols = n_models, n_samples
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(n_cols * cell_size[0], n_rows * cell_size[1]),
            sharex=True, sharey=True,
            constrained_layout=True,
        )
        axes = np.atleast_2d(axes)
        if axes.shape != (n_rows, n_cols):
            axes = axes.reshape(n_rows, n_cols)

        for row, label in enumerate(labels):
            data = tensors[label]
            idxs = sample_indices[label]
            for col in range(n_cols):
                ax = axes[row, col]
                if col < len(idxs):
                    i = idxs[col]
                    ax.plot(axis_x.numpy(), data[i, 0, :].numpy(), lw=1.0)
                    ax.grid(True, linestyle="--", alpha=0.6)
                    ax.set_ylim((-1, 1))
                    if row == 0:
                        ax.set_title(f"Sample {i}")
                    if col == 0:
                        ax.set_ylabel(label, fontsize=11, fontweight="bold")
                    if row == n_rows - 1:
                        ax.set_xlabel("Day of year")
                else:
                    fig.delaxes(ax)

    else:
        fig, axes = plt.subplots(
            1, n_models,
            figsize=(n_models * cell_size[0], cell_size[1] * 1.4),
            sharey=True,
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes)

        for col, label in enumerate(labels):
            ax = axes[col]
            data = tensors[label]
            idxs = sample_indices[label]
            for i in idxs:
                ax.plot(axis_x.numpy(), data[i, 0, :].numpy(), lw=1.0, alpha=0.8)
            ax.set_title(label)
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.set_xlabel("Day of year")
            if col == 0:
                ax.set_ylabel("Temperature (°C)")

    fig.suptitle(suptitle, fontsize=16)
    plt.show()
    return fig

def _log_time_indices(N, num_steps, t_start=0.0, t_end=1.0, tweedie_final=False, device=None):
    device = device or torch.device("cpu")
    N_traj = N - 1 if tweedie_final else N
    times = torch.linspace(t_start, t_end, N_traj, device=device)
    n_panels = num_steps - 1 if tweedie_final else num_steps

    nonzero_times = times.abs()
    nonzero_times = nonzero_times[nonzero_times > 0]
    lo, hi = nonzero_times.min().item(), times.abs().max().item()

    oversample = n_panels * 3
    t_samples = torch.logspace(
        torch.log10(torch.tensor(lo, device=device)),
        torch.log10(torch.tensor(hi, device=device)),
        steps=oversample, device=device,
    )
    diffs = (times.abs().unsqueeze(0) - t_samples.unsqueeze(1)).abs()
    candidate_idxs = torch.unique(diffs.argmin(dim=1))
    candidate_idxs, _ = torch.sort(candidate_idxs)

    pick = torch.linspace(0, len(candidate_idxs) - 1, n_panels).round().long()
    time_idxs = candidate_idxs[pick]
    time_idxs[0] = 0
    time_idxs[-1] = N_traj - 1

    if tweedie_final:
        time_idxs = torch.cat([time_idxs, torch.tensor([N - 1], device=device)])
    return time_idxs, times


def _linear_time_indices(N, num_steps, t_start=0.0, t_end=1.0, device=None):
    device = device or torch.device("cpu")
    flip = t_start > t_end
    lo, hi = (t_end, t_start) if flip else (t_start, t_end)

    times = torch.linspace(lo, hi, N, device=device)
    t_samples = torch.linspace(lo, hi, num_steps, device=device)
    time_idxs = torch.searchsorted(times, t_samples)
    time_idxs = torch.unique(time_idxs)
    time_idxs[0] = 0

    if flip:
        time_idxs = N - 1 - time_idxs
        time_idxs = torch.sort(time_idxs).values

    return time_idxs, times, flip


def _format_time_labels(time_idxs, times, tweedie_final=False, N=None, flip=False, t_end=None):
    labels = []
    for t in time_idxs:
        t_item = t.item()
        if tweedie_final and t_item == N - 1:
            labels.append(r"$\tau$=0.0000" + "\n(Tweedie)")
        elif flip:
            labels.append(rf"$\tau$={(t_end - times[t]):.4f}")
        else:
            labels.append(rf"$\tau$={times[t]:.4f}")
    return labels
                
def _plot_process_block_horizontal(
    fig, subplot_spec, batch_functions, axis_x,
    time_idxs, n_samples, time_labels, color
):
    axis_x_np = axis_x.detach().cpu().numpy()
    time_idxs_cpu = time_idxs.cpu()

    n_steps = len(time_idxs_cpu)
    n_samples = min(batch_functions.shape[0], n_samples)

    inner = GridSpecFromSubplotSpec(
        n_samples,
        n_steps,
        subplot_spec=subplot_spec,
        hspace=0.05,
        wspace=0.10,
    )

    shared_ax = None

    for row in range(n_samples):
        for col, t in enumerate(time_idxs_cpu):

            if shared_ax is None:
                ax = fig.add_subplot(inner[row, col])
                shared_ax = ax
            else:
                ax = fig.add_subplot(
                    inner[row, col],
                    sharex=shared_ax,
                    sharey=shared_ax,
                )

            y = batch_functions[row, t, 0, :].detach().cpu().numpy()
            ax.plot(axis_x_np, y, lw=1.0, color=color)

            ax.tick_params(axis="both", labelsize=7)

            if row != n_samples - 1:
                ax.tick_params(labelbottom=False)

            if col != 0:
                ax.tick_params(labelleft=False)

            if row == 0:
                ax.set_title(time_labels[col], fontsize=8)

def _plot_two_process_blocks(process_dict, n_samples=3, cell_size=(1.8, 1.8), suptitle="", colors=None):
    labels = list(process_dict.keys())
    n_blocks = len(labels)

    n_steps_max = max(len(v[2]) for v in process_dict.values())
    
    default_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = colors or {}
    model_colors = {
        label: colors.get(label, default_cycle[i % len(default_cycle)])
        for i, label in enumerate(labels)
    }


    fig_w = n_steps_max * cell_size[0] * 0.7 + 1.5
    fig_h = n_blocks * n_samples * cell_size[1] * 0.7 + 0.8

    fig = plt.figure(figsize=(fig_w, fig_h))

    outer = GridSpec(
        n_blocks,
        2,
        width_ratios=[0.06, 1],
        hspace=0.5,
        wspace=0.02,
        figure=fig,
    )

    for r, label in enumerate(labels):

        title_ax = fig.add_subplot(outer[r, 0])
        title_ax.axis("off")
        title_ax.text(
            0.5,
            0.5,
            label,
            rotation=90,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
        )

        batch_fn, axis_x, time_idxs, time_labels = process_dict[label]

        _plot_process_block_horizontal(
            fig,
            outer[r, 1],
            batch_fn,
            axis_x,
            time_idxs,
            n_samples,
            time_labels,
            color=model_colors[label],
        )

    fig.suptitle(suptitle, fontsize=16)
    plt.show()
    plt.close(fig)

    return fig

def plot_forward_processes(
    diffusion_process, flow_process, axis_x,
    t_start_diffusion=0.0, t_end_diffusion=1.0,
    t_start_flow=0.0, t_end_flow=1.0,
    num_steps=8, n_samples=3, cell_size=(1.8, 1.8),
    suptitle="Forward Processes", colors=None,
):
    N_diff = diffusion_process.shape[1]
    idxs_diff, times_diff = _log_time_indices(
        N_diff, num_steps, t_start_diffusion, t_end_diffusion, tweedie_final=False,
        device=diffusion_process.device,
    )
    labels_diff = _format_time_labels(idxs_diff, times_diff)

    N_flow = flow_process.shape[1]
    idxs_flow, times_flow, flip_flow = _linear_time_indices(
        N_flow, num_steps, t_start_flow, t_end_flow, device=flow_process.device,
    )
    labels_flow = _format_time_labels(idxs_flow, times_flow, flip=flip_flow, t_end=max(t_start_flow, t_end_flow))

    process_dict = {
        "Diffusion": (diffusion_process, axis_x, idxs_diff, labels_diff),
        "Flow matching": (flow_process, axis_x, idxs_flow, labels_flow),
    }
    return _plot_two_process_blocks(process_dict, n_samples=n_samples, cell_size=cell_size, suptitle=suptitle, colors=colors)


def plot_generation_processes(
    diffusion_process, flow_process, axis_x,
    t_start_diffusion=0.0, t_end_diffusion=1.0,
    t_start_flow=1.0, t_end_flow=0.0,
    tweedie_final=True,
    num_steps=8, n_samples=3, cell_size=(1.8, 1.8),
    suptitle="Generation (Reverse) Processes", colors=None,
):
    N_diff = diffusion_process.shape[1]
    idxs_diff, times_diff = _log_time_indices(
        N_diff, num_steps, t_start_diffusion, t_end_diffusion, tweedie_final=tweedie_final,
        device=diffusion_process.device,
    )
    labels_diff = _format_time_labels(idxs_diff, times_diff, tweedie_final=tweedie_final, N=N_diff)

    N_flow = flow_process.shape[1]
    idxs_flow, times_flow, flip_flow = _linear_time_indices(
        N_flow, num_steps, t_start_flow, t_end_flow, device=flow_process.device,
    )
    labels_flow = _format_time_labels(idxs_flow, times_flow, flip=flip_flow, t_end=max(t_start_flow, t_end_flow))

    process_dict = {
        "Diffusion": (diffusion_process, axis_x, idxs_diff, labels_diff),
        "Flow matching": (flow_process, axis_x, idxs_flow, labels_flow),
    }
    return _plot_two_process_blocks(process_dict, n_samples=n_samples, cell_size=cell_size, suptitle=suptitle, colors=colors)

def _plot_single_process(process, axis_x, time_idxs, n_samples, time_labels, cell_size, suptitle, color):
    axis_x_np = axis_x.detach().cpu().numpy()
    time_idxs_cpu = time_idxs.cpu()
    n_steps = len(time_idxs_cpu)
    n_samples = min(process.shape[0], n_samples)

    fig, axes = plt.subplots(
        n_samples, n_steps,
        figsize=(n_steps * cell_size[0], n_samples * cell_size[1]),
        sharex=True, sharey=True,
        squeeze=False,
        constrained_layout=True,
    )

    for row in range(n_samples):
        for col, t in enumerate(time_idxs_cpu):
            ax = axes[row, col]
            y = process[row, t, 0, :].detach().cpu().numpy()
            ax.plot(axis_x_np, y, lw=1.0, color=color)
            ax.tick_params(axis="both", labelsize=7)

            if col != 0:
                ax.tick_params(labelleft=False)
            if row == 0:
                ax.set_title(time_labels[col], fontsize=9)

    fig.suptitle(suptitle, fontsize=15)
    plt.show()
    plt.close(fig)
    return fig


def plot_diffusion_forward_process(
    diffusion_process, axis_x,
    t_start=0.0, t_end=1.0,
    num_steps=8, n_samples=3, cell_size=(1.8, 1.8),
    suptitle="Forward Diffusion Process", color="tab:blue"
):
    N = diffusion_process.shape[1]
    idxs, times = _log_time_indices(N, num_steps, t_start, t_end, tweedie_final=False, device=diffusion_process.device)
    labels = _format_time_labels(idxs, times)
    return _plot_single_process(diffusion_process, axis_x, idxs, n_samples, labels, cell_size, suptitle, color=color)


def plot_flow_forward_process(
    flow_process, axis_x,
    t_start=0.0, t_end=1.0,
    num_steps=8, n_samples=3, cell_size=(1.8, 1.8),
    suptitle="Interpolation Flow Matching Process", color="tab:blue"
):
    N = flow_process.shape[1]
    idxs, times, flip = _linear_time_indices(N, num_steps, t_start, t_end, device=flow_process.device)
    labels = _format_time_labels(idxs, times, flip=flip, t_end=max(t_start, t_end))
    return _plot_single_process(flow_process, axis_x, idxs, n_samples, labels, cell_size, suptitle, color=color)