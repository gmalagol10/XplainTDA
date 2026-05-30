import os
import argparse
import scanpy as sc
import scTDA
import X_scTDA
from joblib import dump
from Utils import intersection
import sys

print("Imports completed", flush=True)

# =========================================================
# Arguments
# =========================================================
print("Parsing command line arguments", flush=True)

COUNT_PATH = sys.argv[1]
OUTDIR = sys.argv[2]
CELLTYPE = sys.argv[3]

names = ["Script", "COUNT_PATH","OUTDIR","CELLTYPE"]
for n, arg in zip(names, sys.argv):
	print(n, arg, flush=True)

print("Creating output directory", flush=True)
os.makedirs(OUTDIR, exist_ok=True)
print(f"Output directory ready: {OUTDIR}", flush=True)

# =========================================================
# Load count matrix
# =========================================================
print(f"Loading AnnData object from: {COUNT_PATH}", flush=True)
adata = sc.read_h5ad(COUNT_PATH)

print("AnnData loaded", flush=True)
print(f"adata shape: {adata.shape}", flush=True)
print(f"Number of genes: {adata.n_vars}", flush=True)
print(f"Number of cells: {adata.n_obs}", flush=True)

# =========================================================
# Run TDA
# =========================================================
print("Initializing scTDA object", flush=True)

tda = scTDA.scTDA(
    adata,
    n_pcs=30,
    n_neighbors=15,
    maxdim=2,
    neighbor_backend="exact"
)

print("scTDA object created", flush=True)
print(f"Saving TDA object for H", flush=True)
dump(tda, f"{OUTDIR}/{CELLTYPE}_H_tda.joblib")
# =========================================================
# Marker genes
# =========================================================
print("Loading marker genes from adata.uns['Markers_for_TDA']", flush=True)

marker_genes = intersection([adata.uns["Markers_for_TDA"], adata.var.index]).tolist()

print(f"Loaded {len(marker_genes)} marker genes", flush=True)

# =========================================================
# Run topology attribution
# =========================================================
for H in [0,1,2]:

    print("=" * 60, flush=True)
    print(f"Start {CELLTYPE} -- H{H}", flush=True)
    print("=" * 60, flush=True)

    print(f"Initializing X_scTDA for H{H}", flush=True)

    x_tda = X_scTDA.X_scTDA(
        adata=adata,
        tda=tda,
        marker_genes=marker_genes,
        target_dim=H,
        perturbation="zero",
        recompute_mode="local",
        local_radius=2,
        n_random=20,
        verbose=True
    )

    print(f"X_scTDA initialized for H{H}", flush=True)

    # =========================================================
    # Compute scores
    # =========================================================
    print(f"Computing feature scores for H{H}", flush=True)

    results = x_tda.compute_feature_scores()

    print(f"Feature scores completed for H{H}", flush=True)

    # =========================================================
    # Marker statistics
    # =========================================================
    print(f"Computing marker_vs_random for H{H}", flush=True)

    marker_vs_random = (
        x_tda.marker_vs_random()
    )

    print(f"marker_vs_random completed for H{H}", flush=True)

    print(f"Computing marker_rank_enrichment for H{H}", flush=True)

    marker_rank = (
        x_tda.marker_rank_enrichment()
    )

    print(f"marker_rank_enrichment completed for H{H}", flush=True)

    # =========================================================
    # Save complete objects
    # =========================================================
    print(f"Saving x_tda object for H{H}", flush=True)

    dump(
        x_tda,
        f"{OUTDIR}/{CELLTYPE}_H{H}_x_tda.joblib"
    )

    print(f"Saving results object for H{H}", flush=True)

    dump(
        results,
        f"{OUTDIR}/{CELLTYPE}_H{H}_results.joblib"
    )

    print(f"Saving marker_vs_random object for H{H}", flush=True)

    dump(
        marker_vs_random,
        f"{OUTDIR}/{CELLTYPE}_H{H}_marker_vs_random.joblib"
    )

    print(f"Saving marker_rank object for H{H}", flush=True)

    dump(
        marker_rank,
        f"{OUTDIR}/{CELLTYPE}_H{H}_marker_rank.joblib"
    )

    print(f"Finished {CELLTYPE} -- H{H}", flush=True)

print("Pipeline completed successfully", flush=True)
