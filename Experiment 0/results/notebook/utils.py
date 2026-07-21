# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 13:49:55 2026

@author: diego.camarero@estudiante.uam.es
"""

import os
import torch
from pathlib import Path


def load_tensor_bundle(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such results file: {path}")
    return torch.load(path, weights_only=False)

def unpack(bundle, keys):
    values = []
    missing = []
    for k in keys:
        if k in bundle:
            values.append(bundle[k])
        else:
            missing.append(k)
    if missing:
        raise KeyError(
            f"Missing key(s) {missing} in bundle. Available keys: {list(bundle.keys())}"
        )
    return tuple(values) if len(values) > 1 else values[0]

def load_data(data_dir, filename, keys):
    path = os.path.join(data_dir, filename)
    data_pt = load_tensor_bundle(path)
    return unpack(data_pt, keys)

def denormalize(x, data_mean, data_std, data_eps):
    return x * (data_std + data_eps) + data_mean

def extract_generated_functions(generated_samples, denorm=False, data_mean=None, data_std=None, data_eps=None):
    final_samples = generated_samples[:, -1, :, :]

    if denorm:
        if data_mean is None or data_std is None or data_eps is None:
            raise ValueError("data_mean, data_std and data_eps are required when denorm=True")
        final_samples = denormalize(final_samples, data_mean, data_std, data_eps)

    return final_samples

def save_image(fig, image_dir, image_name, dpi=300, **savefig_kwargs):
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    path = image_dir / image_name
    fig.savefig(path, dpi=dpi, bbox_inches="tight", **savefig_kwargs)
    print(f"Saved -> {path}")
    return path