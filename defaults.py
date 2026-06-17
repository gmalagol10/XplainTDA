# defaults.py
import numpy as np
import os, torch

# =====================================================
# Data / embedding / PH defaults
# =====================================================
DEFAULT_LAYER = "X"
DEFAULT_EMBEDDER = "GRAE"
DEFAULT_N_COMPONENTS = None
DEFAULT_N_NEIGHBORS = 100
DEFAULT_MAXDIM = 2
DEFAULT_NEIGHBOR_BACKEND = "exact"
DEFAULT_DTYPE = np.float32
DEFAULT_EMBEDDER_KWARGS = None
DEFAULT_PH_BACKEND = "gudhi"
DEFAULT_GLOBAL_PH_BACKEND = "ripser"

# =====================================================
# GRAE defaults
# =====================================================
def select_device(min_free_mem_mb: int = 8192) -> torch.device:
    """
    Select the best device at startup.
    - Uses CUDA if available and has at least `min_free_mem_mb` free memory.
    - Otherwise falls back to CPU.

    Parameters
    ----------
    min_free_mem_mb : int
        Minimum free memory (in MB) required to use CUDA.

    Returns
    -------
    device : torch.device
        The selected device.
    """
    if torch.cuda.is_available():
        free_mem, total_mem = torch.cuda.mem_get_info()
        free_mem_mb = free_mem // (1024 ** 2)

        print(f"CUDA available: {free_mem_mb} MB free / {total_mem // (1024 ** 2)} MB total", flush=True)

        if free_mem_mb >= min_free_mem_mb:
            print("✅ Using CUDA", flush=True)
            return torch.device("cuda")
        else:
            print(f"⚠️ Not enough free CUDA memory ({free_mem_mb} MB). Falling back to CPU.", flush=True)
            return torch.device("cpu")

    print("❌ CUDA not available. Using CPU.", flush=True )
    return torch.device("cpu")


DEVICE = select_device()  # Require at least 8GB free
DEFAULT_GRAE_EPOCHS = 300
DEFAULT_GRAE_PATIENCE = 30
DEFAULT_BATCH_SIZE = 128
DEFAULT_LR = .0001
DEFAULT_WEIGHT_DECAY = 0
DEFAULT_HIDDEN_DIMS = {"hidden_dim" : 800, "hidden_dim_2" : 400, "hidden_dim_1" : 200} 
DEFAULT_FIT = .85  # Default train/test split ratio
# =====================================================
# Stratification defaults
# =====================================================
DEFAULT_GROUPS = None
DEFAULT_MIN_CELLS = 50

# =====================================================
# Explanation selection defaults
# =====================================================
DEFAULT_GENES = None
DEFAULT_DIMS = None
DEFAULT_TARGET_DIM = None

# =====================================================
# Parallelization / caching defaults
# =====================================================
def available_cpus(default=1):
    for var in ("XPLAINTDA_N_JOBS", "SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        x = os.environ.get(var)
        if x:
            return max(1, int(x))

    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))

    return max(1, os.cpu_count() or default)

DEFAULT_N_JOBS = available_cpus()
DEFAULT_BACKEND = "threading"
DEFAULT_CACHE = None

# =====================================================
# Shared perturbation defaults
# =====================================================
DEFAULT_PERTURBATION = "zero"
DEFAULT_VERBOSE = True
DEFAULT_RANDOM_STATE = None

# =====================================================
# Local explanation defaults
# =====================================================
DEFAULT_RECOMPUTE_MODE = "exact"
DEFAULT_LOCAL_RADIUS = 2
DEFAULT_TOP_FRAC = 0.10
DEFAULT_MAX_FEATURES = 30
DEFAULT_MATCHING_MODE = "match_cells" # {"max_persistence", "match_cells"}
DEFAULT_JACCARD_THRESHOLD = 0.3

# =====================================================
# Global explanation defaults
# =====================================================
DEFAULT_DISTANCE = "bottleneck"
DEFAULT_BOTTLENECK_EPSILON = 0.0
DEFAULT_WASSERSTEIN_ORDER = 2.0
DEFAULT_WASSERSTEIN_INTERNAL_P = np.inf
