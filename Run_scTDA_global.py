import os, sys
import scanpy as sc
import scTDA, Global_X_scTDA
from joblib import dump

print("Imports completed", flush=True)

# =========================================================
# Arguments
# =========================================================

COUNT_PATH = sys.argv[1]
OUTDIR     = sys.argv[2]
CELLTYPE   = sys.argv[3]

for n, arg in zip(
    ["Script","COUNT_PATH","OUTDIR","CELLTYPE"],
    sys.argv
):
    print(n, arg, flush=True)

os.makedirs(OUTDIR, exist_ok=True)

# =========================================================
# Load data
# =========================================================

print(f"Loading {COUNT_PATH}", flush=True)

adata = sc.read_h5ad(COUNT_PATH)

print(f"{adata.n_obs} cells", flush=True)
print(f"{adata.n_vars} genes", flush=True)

# =========================================================
# Compute reference topology
# =========================================================

print("Computing reference PH", flush=True)

tda = scTDA.scTDA(
    adata,
    n_pcs=30,
    n_neighbors=15,
    maxdim=2,
    neighbor_backend="exact"
)

dump(
    tda,
    f"{OUTDIR}/{CELLTYPE}_tda.joblib"
)

# =========================================================
# Global explanations
# =========================================================

print("Initializing Global_X_scTDA", flush=True)

x_tda = Global_X_scTDA.Global_X_scTDA(
    tda=tda,
    wasserstein_order=2,
    internal_p=2,
    verbose=True
)

print("Computing gene importance", flush=True)

summary = x_tda.compute_global_scores()

# =========================================================
# Save
# =========================================================

dump(
    x_tda,
    f"{OUTDIR}/{CELLTYPE}_global_x_tda.joblib"
)

dump(
    summary,
    f"{OUTDIR}/{CELLTYPE}_global_summary.joblib"
)

print(summary.head(), flush=True)

print("Pipeline completed successfully", flush=True)
