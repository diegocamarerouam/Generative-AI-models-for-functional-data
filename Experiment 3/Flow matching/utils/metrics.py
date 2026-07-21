# -*- coding: utf-8 -*-
"""
Created on Mon May 25 17:43:47 2026

@author: diego.camarero@estudiante.uam.es
"""

import torch
import matplotlib.pyplot as plt

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