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
import torch

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

    rr = np.diff(qrs_inds) / fs
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

    rr = result['rr_intervals']  # seconds
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


def compare_distributions(
        real_values, gen_values, metric_name, unit='', bins=20, ax=None, plot=True,
        ):
    """
    Overlaid histogram + mean/std annotation + Wasserstein distance for each metric.
    """
    real_values = np.asarray(real_values)
    gen_values = np.asarray(gen_values)
    real_clean = real_values[~np.isnan(real_values)]
    gen_clean = gen_values[~np.isnan(gen_values)]

    stats = {
        'real_mean': np.mean(real_clean), 'real_std': np.std(real_clean),
        'gen_mean': np.mean(gen_clean), 'gen_std': np.std(gen_clean),
        'wasserstein': wasserstein_distance(real_clean, gen_clean),
    }
    
    if not plot:
        return stats

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    ax.hist(real_clean, bins=bins, alpha=0.5, density=True, label='Real')
    ax.hist(gen_clean, bins=bins, alpha=0.5, density=True, label='Generated')
    ax.set_xlabel(f'{metric_name} ({unit})' if unit else metric_name)
    ax.set_ylabel('Density')
    ax.set_title(
        f'{metric_name}\nReal:{stats["real_mean"]:.2f}±{stats["real_std"]:.2f}  |  '
        f'Gen: {stats["gen_mean"]:.2f}±{stats["gen_std"]:.2f}  |  '
        f'W1={stats["wasserstein"]:.3f}'
    )
    ax.legend()
    return stats

def checkpoint_summary(real_ok_df, gen_df):
    gen_ok = gen_df[gen_df['success']]
    success_rate = gen_df['success'].mean()
 
    hr = compare_distributions(
        real_ok_df['hr_bpm'], gen_ok['hr_bpm'], 'Heart rate', plot=False,
    )
    rr_mean = compare_distributions(
        real_ok_df['rr_mean'], gen_ok['rr_mean'], 'Mean RR interval', plot=False,
    )
    rr_std = compare_distributions(
        real_ok_df['rr_std'], gen_ok['rr_std'], 'Std RR interval', plot=False,
    )
 
    return {
        'success_rate': success_rate,
        # 'hr_gen_mean': hr['gen_mean'], 'hr_gen_std': hr['gen_std'],
        'wd_hr': hr['wasserstein'],
        # 'rr_mean_gen_mean': rr_mean['gen_mean'], 'rr_mean_gen_std': rr_mean['gen_std'],
        'wd_rr_mean': rr_mean['wasserstein'],
        # 'rr_std_gen_mean': rr_std['gen_mean'], 'rr_std_gen_std': rr_std['gen_std'],
        'wd_rr_std': rr_std['wasserstein'],
    }


def full_report(real_df, gen_df, bins=20):
    """
    Full comparison.
    """
    real_success_rate = real_df['success'].mean()
    gen_success_rate = gen_df['success'].mean()

    real_ok = real_df[real_df['success']]
    gen_ok = gen_df[gen_df['success']]

    print(f'R-peak detection success rate — real: {real_success_rate:.1%}, '
          f'generated: {gen_success_rate:.1%}')

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

    hr_stats = compare_distributions(
        real_ok['hr_bpm'], gen_ok['hr_bpm'], 'Heart rate', unit='bpm',
        bins=bins, ax=axes[0],
    )
    rr_mean_stats = compare_distributions(
        real_ok['rr_mean'], gen_ok['rr_mean'], 'Mean RR interval', unit='s',
        bins=bins, ax=axes[1],
    )
    rr_std_stats = compare_distributions(
        real_ok['rr_std'], gen_ok['rr_std'], 'Std RR interval', unit='s',
        bins=bins, ax=axes[2],
    )

    plt.tight_layout()
    plt.show()

    return {
        'success_rate_real': real_success_rate,
        'success_rate_gen': gen_success_rate,
        'hr': hr_stats,
        'rr_mean': rr_mean_stats,
        'rr_std': rr_std_stats,
    }

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