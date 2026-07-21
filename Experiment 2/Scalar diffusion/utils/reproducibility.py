# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 10:10:29 2026

@author: diego.camarero@estudiante.uam.es
"""

import torch
import numpy as np
import os
import random

def seed_everything(seed_val=42):
    random.seed(seed_val)
    os.environ['PYTHONHASHSEED'] = str(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)