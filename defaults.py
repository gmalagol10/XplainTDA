"""Shared defaults for XplainTDA.

Only settings used by persistent homology, embedding, stratification, and the
current gene explainer are defined here. Stability scripts may define their
experiment-specific grids locally while reusing these shared defaults.
"""

from __future__ import annotations

import os

import numpy as np
import torch


# Data, embedding, and persistent homology
DEFAULT_LAYER = "X"
DEFAULT_EMBEDDER = "GRAE"
DEFAULT_N_COMPONENTS = None
DEFAULT_N_NEIGHBORS = 75
DEFAULT_MAXDIM = 2
DEFAULT_NEIGHBOR_BACKEND = "exact"
DEFAULT_PH_BACKEND = "ripser" #or gudhi
DEFAULT_DTYPE = np.float32
DEFAULT_EMBEDDER_KWARGS = None


# GRAE

def select_device(min_free_mem_mb: int = 8192) -> torch.device:
    """Use CUDA when available with sufficient free memory; otherwise use CPU."""
    if not torch.cuda.is_available():
        print("CUDA unavailable; using CPU.", flush=True)
        return torch.device("cpu")

    free_mem, total_mem = torch.cuda.mem_get_info()
    free_mb, total_mb = free_mem // 2**20, total_mem // 2**20
    print(f"CUDA memory: {free_mb} MB free / {total_mb} MB total", flush=True)
    if free_mb >= min_free_mem_mb:
        print("Using CUDA.", flush=True)
        return torch.device("cuda")

    print(f"CUDA has less than {min_free_mem_mb} MB free; using CPU.", flush=True)
    return torch.device("cpu")


DEVICE = select_device()
DEFAULT_GRAE_EPOCHS = 300
DEFAULT_GRAE_PATIENCE = 30
DEFAULT_BATCH_SIZE = 128
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_HIDDEN_DIMS = {
    "hidden_dim": 800,
    "hidden_dim_2": 400,
    "hidden_dim_1": 200,
}
DEFAULT_FIT = 0.85


# Stratification
DEFAULT_GROUPS = None
DEFAULT_MIN_CELLS = 50


# Parallelization

def available_cpus(default: int = 1) -> int:
    """Return a scheduler-aware CPU count."""
    for name in ("XPLAINTDA_N_JOBS", "SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        if value := os.environ.get(name):
            return max(1, int(value))
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or default)


DEFAULT_N_JOBS = available_cpus()
DEFAULT_BACKEND = "threading"


# Gene explainer
DEFAULT_PERTURBATION = "zero"
DEFAULT_VERBOSE = True
DEFAULT_RANDOM_STATE = 3
