# -*- coding: utf-8 -*-
"""
Created on Tue May 12 09:58:17 2026

@author: diego.camarero@estudiante.uam.es

"""

import os
import torch
from collections import OrderedDict

def save_checkpoint(
    epoch,
    model,
    optimizer,
    loss,
    elapsed_time,
    experiment_name,
    checkpoints_dir,
):
    folder_name = os.path.join(checkpoints_dir, experiment_name)
    os.makedirs(folder_name, exist_ok=True)

    file_name = f"checkpoint_epoch_{epoch+1}.pth"
    save_path = os.path.join(folder_name, file_name)

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "elapsed_time": elapsed_time,
    }, save_path)

    print(f"Checkpoint saved at: {save_path}")
    print(
        f"Epoch: {epoch+1}. "
        f"Loss: {loss:.6f}. "
        f"Time: {int(elapsed_time//60):02d}:{int(elapsed_time%60):02d}"
    )

def load_checkpoint(path, model, optimizer=None):
    
    target_device = next(model.parameters()).device if list(model.parameters()) else "cpu"
    checkpoint = torch.load(path, map_location=target_device)
    
    state_dict = checkpoint["model_state_dict"]
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    elapsed_time = checkpoint.get("elapsed_time", 0)

    print(f"Checkpoint loaded from: {path}")
    print(
        f"Epoch: {epoch+1}. "
        f"Loss: {loss:.6f}. "
        f"Time: {int(elapsed_time//60):02d}:{int(elapsed_time%60):02d}"
    )

    return model, optimizer, epoch+1, loss, elapsed_time