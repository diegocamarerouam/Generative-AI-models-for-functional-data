# -*- coding: utf-8 -*-
"""
Created on Mon May 25 17:43:47 2026

@author: diego.camarero@estudiante.uam.es

ECG generative model evaluation metrics.

Pipeline:
    1. R-peak detection (per signal) via wfdb.processing.xqrs_detect.
    2. Heart rate (one value per ECG) -> mean +/- std, histogram, Wasserstein distance.
    3. RR intervals (mean and std per ECG) -> mean +/- std, histogram, Wasserstein distance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb.processing
from scipy.stats import wasserstein_distance

MIN_HR = 30
MAX_HR = 240
MIN_PEAKS = 3

def detect_r_peaks(signal, fs, min_hr=MIN_HR, max_hr=MAX_HR, min_peaks=MIN_PEAKS):
    """
    Detect R-peaks in a single ECG signal and check plausibility.
    """
    signal = np.asarray(signal, dtype=np.float64)

    try:
        qrs_inds = wfdb.processing.xqrs_detect(sig=signal, fs=fs, verbose=False)
    except Exception as e:
        return {'success': False, 'r_peaks': None, 'rr_intervals': None,
                'reason': f'detector_error: {e}'}

    # detected_peaks < min_peaks
    if qrs_inds is None or len(qrs_inds) < min_peaks:
        return {'success': False, 'r_peaks': qrs_inds, 'rr_intervals': None,
                'reason': 'too_few_peaks'}

    rr = np.diff(qrs_inds) / fs  # seconds
    inst_hr = 60.0 / rr

    # heart rate outside [min_hr, max_hr]
    if np.any((inst_hr < min_hr) | (inst_hr > max_hr)):
        return {'success': False, 'r_peaks': qrs_inds, 'rr_intervals': rr,
                'reason': 'implausible_hr'}

    # successful signal
    return {'success': True, 'r_peaks': qrs_inds, 'rr_intervals': rr, 'reason': 'ok'}

def compute_signal_metrics(signal, fs, min_hr=MIN_HR, max_hr=MAX_HR, min_peaks=MIN_PEAKS):
    """
    Compute HR and RR summary stats for a single signal.
    """
    result = detect_r_peaks(signal, fs, min_hr, max_hr, min_peaks)

    if not result['success']:
        return {
            'success': False, 'reason': result['reason'],
            'hr_bpm': np.nan, 'rr_mean': np.nan, 'rr_std': np.nan,
            'n_peaks': (
                len(result['r_peaks']) if result['r_peaks'] is not None else 0
            ),
        }

    rr = result['rr_intervals']
    hr_bpm = 60.0 / np.mean(rr)
    rr_mean = np.mean(rr)
    rr_std = np.std(rr)

    return {
        'success': True, 'reason': 'ok',
        'hr_bpm': hr_bpm, 'rr_mean': rr_mean, 'rr_std': rr_std,
        'n_peaks': len(result['r_peaks']),
    }

def compute_dataset_metrics(signals, fs, min_hr=MIN_HR, max_hr=MAX_HR, min_peaks=MIN_PEAKS):
    """
    Compute per-signal metrics for a batch of ECGs.
    """
    if hasattr(signals, 'detach'):
        signals = signals.detach().cpu().numpy()
    signals = np.asarray(signals)

    if signals.ndim == 3:
        signals = signals.squeeze(1)

    rows = []
    for i in range(signals.shape[0]):
        rows.append(compute_signal_metrics(signals[i], fs, min_hr=min_hr, max_hr=max_hr, min_peaks=min_peaks))

    df = pd.DataFrame(rows)
    return df

def plot_ecg_metric_grid(
    models: dict,
    metric_cols: list,
    bins: int = 20,
    cell_size=(5.0, 3.5),
    suptitle: str = "ECG Quality Metrics",
    colors: dict = None,
):
    labels = list(models.keys())
    n_models = len(labels)
    n_metrics = len(metric_cols)

    reference_label = labels[0]

    default_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = colors or {}
    model_colors = {
        label: colors.get(label, default_cycle[i % len(default_cycle)])
        for i, label in enumerate(labels)
    }

    values = {label: {} for label in labels}
    for label, cfg in models.items():
        df = cfg["df"]
        for col, _, _ in metric_cols:
            v = np.asarray(df[col].values, dtype=float)
            values[label][col] = v[~np.isnan(v)]

    bin_edges = {}
    for col, _, _ in metric_cols:
        all_vals = np.concatenate([values[label][col] for label in labels])
        bin_edges[col] = np.histogram_bin_edges(all_vals, bins=bins)

    fig, axes = plt.subplots(
        n_models, n_metrics,
        figsize=(cell_size[0] * n_metrics, cell_size[1] * n_models),
        squeeze=False, sharex="col", sharey="col",
    )

    for row, label in enumerate(labels):
        cfg = models[label]
        success_rate = cfg.get("success_rate")
        row_label = label if success_rate is None else f"{label}\n(success: {success_rate:.1%})"
        for col_idx, (col, title, unit) in enumerate(metric_cols):
            ax = axes[row, col_idx]
            v = values[label][col]
            ref_v = values[reference_label][col]

            ax.hist(v, bins=bin_edges[col], density=True,
                     color=model_colors[label],
                     edgecolor="black", alpha=0.7)

            w1 = wasserstein_distance(ref_v, v)
            mean_v, std_v = v.mean(), v.std()

            unit_str = f" {unit}" if unit else ""
            if row == 0:
                ax.annotate(title, xy=(0.5, 1.28), xycoords="axes fraction",
                            ha="center", va="bottom", fontsize=13, fontweight="bold")
                ax.set_title(
                  f"Mean={mean_v:.2f}, Std={std_v:.2f}{unit_str}",
                  fontsize=11,
              )
                
            else:
              ax.set_title(
                  rf"${{W_1}}$={w1:.3f}""\n"
                  f"Mean={mean_v:.2f}, Std={std_v:.2f}{unit_str}",
                  fontsize=11,
              )
            if col_idx == 0:
                ax.set_ylabel("Density", fontsize=11)
                up=0
                if row==1:
                  up += 0.1
                ax.annotate(row_label, xy=(-0.25, 0.2+up), xycoords="axes fraction", rotation=90,
                            ha="center", va="bottom", fontsize=13, fontweight="bold")

            if row == n_models - 1:
                ax.set_xlabel(f"{title}{f' ({unit})' if unit else ''}", fontsize=10)

    fig.suptitle(suptitle, fontsize=18, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()
    plt.close(fig)
    return fig