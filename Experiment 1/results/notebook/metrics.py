import torch
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import seaborn as sns

def mean_and_MSE(samples: torch.Tensor) -> torch.Tensor:
    means = torch.mean(samples, dim=-1, keepdim=True)
    mse = torch.mean((samples - means) ** 2, dim=(-2, -1))
    return means.flatten(), mse


def wasserstein_uniform_vs_samples(samples, a=0.0, b=1.0):
    if hasattr(samples, "detach"):
        samples = samples.detach().cpu().numpy()
    elif isinstance(samples, torch.Tensor):
        samples = samples.cpu().numpy()
    uniform_grid = np.linspace(a, b, num=5000)
    return sc.stats.wasserstein_distance(samples, uniform_grid)

def plot_constant_uniform_metrics(
    models: dict,
    theoretical_min: float,
    theoretical_max: float,
    bins: int = 20,
    cell_size=(7, 4.0),
    suptitle: str = "Quality metrics",
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
        samples = cfg["samples"]
        compute_mse = cfg.get("compute_mse", True)
        means, mse = mean_and_MSE(samples)
        means_np = means.detach().cpu().numpy()
        mse_np = mse.detach().cpu().numpy() if compute_mse else None
        w1 = wasserstein_uniform_vs_samples(means_np, theoretical_min, theoretical_max)
        computed[label] = {
            "means": means_np,
            "mse": mse_np,
            "w1": w1,
            "median_mse": float(np.median(mse_np)) if compute_mse else None,
            "compute_mse": compute_mse,
        }

    all_means = np.concatenate([v["means"] for v in computed.values()])
    means_bin_edges = np.histogram_bin_edges(all_means, bins=bins)

    all_mse = np.concatenate([v["mse"] for v in computed.values() if v["mse"] is not None])
    mse_bin_edges = np.histogram_bin_edges(all_mse, bins=bins) if len(all_mse) else None

    fig, axes = plt.subplots(
        2, n_models,
        figsize=(cell_size[0] * n_models, cell_size[1] * 2),
        squeeze=False, sharey="row",
    )

    fig.text(-0.01, 0.70, "MSE distribution", rotation=90, fontsize=18, fontweight="bold", va="center")
    fig.text(-0.01, 0.28, "Means distribution", rotation=90, fontsize=18, fontweight="bold", va="center")

    bbox_top = axes[0, 1].get_position()
    fig.text(bbox_top.x0 - 0.07, (bbox_top.y0 + bbox_top.y1) / 2 - 0.06, "Density",
              rotation=90, ha="center", va="center", fontsize=14)

    for col, label in enumerate(labels):
        c = computed[label]
        color = model_colors[label]

        ax1 = axes[0, col]
        if c["compute_mse"]:
            sns.histplot(c["mse"], stat="density", bins=mse_bin_edges,
                         color=color, edgecolor="black", alpha=0.7, ax=ax1)
            ax1.axvline(c["median_mse"], color="red", linestyle="--", linewidth=2,
                        label=f"Median: {c['median_mse']:.2e}")
            ax1.set_title(rf"Median = {c['median_mse']:.2e}", fontsize=14, fontweight="normal", pad=10)
            ax1.legend(fontsize=8)
            ax1.set_xlabel("MSE")
        else:
            ax1.axis("off")
            ax1.text(0.5, 0.5, "MSE $=0$ by construction\n(original data is exactly\nconstant per sample)",
                      ha="center", va="center", fontsize=14, style="italic", color="black", transform=ax1.transAxes)

        if col == 1:
            ax1.yaxis.set_tick_params(labelleft=True)
            ax1.set_ylabel("Density", fontsize=16)

        ax2 = axes[1, col]
        ax2.hist(c["means"], density=True, bins=means_bin_edges,
                  color=color, edgecolor="black", alpha=0.7, label=label)

        xmin, xmax = ax2.get_xlim()
        x_range = np.linspace(xmin, xmax, 1000)
        h = np.zeros_like(x_range)
        mask = (x_range >= theoretical_min) & (x_range <= theoretical_max)
        h[mask] = 1 / (theoretical_max - theoretical_min)
        ax2.plot(x_range, h, color="red", linestyle="--", linewidth=2, label="Theoretical uniform")

        ax2.set_title(rf"$W_1 = {c['w1']:.3f}$", fontsize=14, fontweight="normal", pad=10)
        if col == 0:
            ax2.set_ylabel("Density", fontsize=14)
        ax2.set_xlabel("Mean value")
        ax2.legend(fontsize=8)

    fig.suptitle(suptitle, fontsize=20, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.9])

    for col, label in enumerate(labels):
        bbox = axes[0, col].get_position()
        x = (bbox.x0 + bbox.x1) / 2
        fig.text(x, bbox.y1 + 0.1, label, ha="center", va="bottom",
                  fontsize=18, fontweight="bold", color='black')

    plt.show()
    return fig