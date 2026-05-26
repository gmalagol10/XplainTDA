import os
import argparse
import scanpy as sc
import scTDA
import X_scTDA
from joblib import dump
from Utils import intersection
import sys
import time

print("Imports completed", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), flush=True)

# =========================================================
# Arguments
# =========================================================
print("Parsing command line arguments", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), flush=True)

COUNT_PATH = sys.argv[1]
OUTDIR = sys.argv[2]
CELLTYPE = sys.argv[3]
APPROACH = sys.argv[4]

names = ["Script", "COUNT_PATH","OUTDIR","CELLTYPE", "APPROACH"]
for n, arg in zip(names, sys.argv):
	print(n, arg, flush=True)

print("Creating output directory", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), flush=True)
os.makedirs(OUTDIR, exist_ok=True)
print(f"Output directory ready: {OUTDIR}", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), flush=True)

# =========================================================
# Load count matrix
# =========================================================
print(f"Loading AnnData object from: {COUNT_PATH}", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), flush=True)
adata = sc.read_h5ad(COUNT_PATH)

print("AnnData loaded", flush=True)
print(f"adata shape: {adata.shape}", flush=True)
print(f"Number of genes: {adata.n_vars}", flush=True)
print(f"Number of cells: {adata.n_obs}", flush=True)

# =========================================================
# Run TDA
# =========================================================
print("Initializing scTDA object", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), flush=True)

tda = scTDA.scTDA(
    adata,
    n_pcs=30,
    n_neighbors=15,
    maxdim=2,
    neighbor_backend="exact"
)

print("scTDA object created", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), flush=True)
print(f"Saving TDA object", flush=True)
dump(tda, f"{OUTDIR}/{CELLTYPE}_H_tda.joblib")

# =========================================================
# Run topology attribution
# =========================================================
for H in [0,1,2]:

    print("=" * 60, flush=True)
    print(f"Start {CELLTYPE} -- H{H}", flush=True)
    print("=" * 60, flush=True)

    print(f"Initializing X_scTDA for H{H}", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), flush=True)

    x_tda = X_scTDA.X_scTDA(
        adata=adata,
        tda=tda,
        target_dim=H,
        perturbation="zero",
        recompute_mode="local",
        max_features=50,
        matching_mode=APPROACH,
        verbose=True
    )

    print(f"X_scTDA initialized for H{H}", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), flush=True)

    # =========================================================
    # Compute scores
    # =========================================================
    print(f"Computing feature scores for H{H}", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), flush=True)

    results = x_tda.compute_feature_scores()

    print(f"Feature scores completed for H{H}", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), flush=True)

    print(f"Saving x_tda object for H{H}", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), flush=True)
    dump(x_tda,f"{OUTDIR}/{CELLTYPE}_H{H}_x_tda.joblib")

    print(f"Saving results object for H{H}", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), flush=True)
    dump(results, f"{OUTDIR}/{CELLTYPE}_H{H}_results.joblib")

    print(f"Finished {CELLTYPE} -- H{H}", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), flush=True)

print("Pipeline completed successfully", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), flush=True)
