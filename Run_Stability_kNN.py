"""
Benchmark how k in the XplainTDA kNN graph affects persistent homology.

Design:
- Compute the embedding once.
- Save the embedding and the full pairwise distance matrix.
- Compute the complete-graph PH baseline by inserting all pairwise edges into
  a SimplexTree and calling expansion(maxdim + 1), matching XplainTDA's logic.
- Compute one kNN PH result per k in parallel.
- Inside each k job, force kNN neighbor search to single-job / single-worker.
- Save every kNN TDA object using XplainTDA.save().
- Compute bottleneck distances in two phases:
    1. full vs each k;
    2. all remaining k-vs-k pairs for the heatmap.

Edit the configuration block below, especially KS.
"""

from __future__ import annotations

import argparse
import json
import sys
import os
import time
from contextlib import contextmanager
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import anndata as ad
import gudhi as gd
import joblib
import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors

import defaults as d

try:
    from pynndescent import NNDescent
    HAS_PYNNDESCENT = True
except ImportError:
    HAS_PYNNDESCENT = False

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None

from XplainTDA import XplainTDA, compute_ph_from_neighbors, build_simplex_tree, simplex_tree_to_diagrams


# ---------------------------------------------------------------------
# User-editable configuration
# ---------------------------------------------------------------------
KS = [5, 15, 50, 75, 100, 250, 500, 750, 1000, 1250]

# Use project defaults where they exist.
LAYER = d.DEFAULT_LAYER
EMBEDDER = d.DEFAULT_EMBEDDER
N_COMPONENTS = d.DEFAULT_N_COMPONENTS
FULL_MAXDIM = 0                    # full distance matrix: H0 only
KNN_MAXDIM = d.DEFAULT_MAXDIM       # kNN graphs: H0, H1, H2 by default
KNN_PH_BACKEND = d.DEFAULT_GLOBAL_PH_BACKEND
NEIGHBOR_BACKEND = d.DEFAULT_NEIGHBOR_BACKEND
DTYPE = d.DEFAULT_DTYPE
EMBEDDER_KWARGS = d.DEFAULT_EMBEDDER_KWARGS or {}

SAVE_ADATA_IN_TDA_OBJECTS = False
SAVE_EMBEDDER_IN_KNN_OBJECTS = False
JOBLIB_COMPRESS = 3

# Parallel k jobs. kNN inside each job is kept single-worker.
# Do not use all DEFAULT_N_JOBS here by default: PH can be memory-heavy.
N_JOBS_TDA = 1
N_JOBS_BOTTLENECK = d.DEFAULT_N_JOBS
JOBLIB_BACKEND = "loky"

# Also cap BLAS/OpenMP threads inside each k job when threadpoolctl is installed.
INNER_THREAD_LIMIT = 1
# ---------------------------------------------------------------------


@contextmanager
def maybe_limit_threads(n_threads: int):
    if threadpool_limits is None:
        yield
    else:
        with threadpool_limits(limits=n_threads):
            yield


def compute_neighbors(X, n_neighbors, backend):
    """
    Compute k-nearest neighbors in an embedding.

    This intentionally mirrors XplainTDA.compute_neighbors, except that the
    exact sklearn backend uses n_jobs=1 so outer parallelization over k values
    does not oversubscribe CPUs.
    """
    n_neighbors = min(int(n_neighbors), X.shape[0])

    if backend == "exact":
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", n_jobs=1)
        neigh_dist, neigh_ind = nn.fit(X).kneighbors(X)

    elif backend == "pynndescent":
        if not HAS_PYNNDESCENT:
            raise ImportError("pynndescent is not installed.")
        try:
            neigh_ind, neigh_dist = NNDescent(
                X, n_neighbors=n_neighbors, metric="euclidean", n_jobs=1
            ).neighbor_graph
        except TypeError:
            neigh_ind, neigh_dist = NNDescent(
                X, n_neighbors=n_neighbors, metric="euclidean"
            ).neighbor_graph

    else:
        raise ValueError("backend must be 'exact' or 'pynndescent'")

    return neigh_ind, neigh_dist


def localize_features_safe(dgms, edge_list: np.ndarray, maxdim: int, eps: float = 1e-6):
    """Same rough localization logic as XplainTDA, but robust to empty edges."""
    features = {dim: [] for dim in range(maxdim + 1)}
    for dim in range(maxdim + 1):
        for feature_id, (birth, death) in enumerate(dgms[dim]):
            if edge_list.size == 0:
                cells = np.array([], dtype=int)
            else:
                mask = (birth - eps <= edge_list[:, 2]) & (edge_list[:, 2] <= death + eps)
                G = nx.Graph()
                G.add_edges_from(edge_list[mask, :2].astype(int))
                cells = np.array(sorted(max(nx.connected_components(G), key=len))) if G.number_of_nodes() else np.array([], dtype=int)
            features[dim].append({
                "feature_id": int(feature_id),
                "birth": float(birth),
                "death": float(death),
                "persistence": float(death - birth),
                "cells": cells,
            })
    return features


def build_full_edge_list(distance_matrix: np.ndarray) -> np.ndarray:
    i, j = np.triu_indices(distance_matrix.shape[0], k=1)
    return np.column_stack((i, j, distance_matrix[i, j])).astype(float, copy=False)


def ph_from_full_distance_matrix(distance_matrix: np.ndarray, maxdim: int):
    """Complete weighted graph + SimplexTree.expansion(maxdim + 1)."""
    print(f"Starting to compute Simplex tree", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), flush=True)
    st = gd.SimplexTree()
    print(f"Done computing Simplex tree", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), flush=True)
    for i in range(distance_matrix.shape[0]):
        st.insert([i], filtration=0.0)
    edge_list = build_full_edge_list(distance_matrix)
    for i, j, dist in edge_list:
        st.insert([int(i), int(j)], filtration=float(dist))
        
    print(f"Starting to compute expansion", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), flush=True)
    st.expansion(maxdim + 1)
    print(f"Done computing expansion", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), flush=True)
    
    print(f"Starting to compute dgms", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), flush=True)
    dgms = simplex_tree_to_diagrams(st, maxdim=maxdim)
    print(f"Done computing dgms", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), flush=True)
    
    return {"dgms": dgms, "features": localize_features_safe(dgms, edge_list, maxdim), "edge_list": edge_list}


def fit_embed_once(adata, outdir):
    outdir = Path(outdir).resolve()
    grae_dir = outdir / "GRAE"
    grae_dir.mkdir(parents=True, exist_ok=True)

    old_cwd = os.getcwd()
    os.chdir(grae_dir)
    try:
        model = XplainTDA(
            adata=adata,
            layer=LAYER,
            embedder=EMBEDDER,
            n_components=N_COMPONENTS,
            n_neighbors=KS[0],
            maxdim=KNN_MAXDIM,
            neighbor_backend=NEIGHBOR_BACKEND,
            ph_backend=KNN_PH_BACKEND,
            dtype=DTYPE,
        )

        model.X = adata.X if LAYER == "X" else adata.layers[LAYER]
        model.n_cells, model.n_genes = model.X.shape
        if model.n_components is None:
            model.n_components = int(np.round(model.n_genes ** (1 / 3)))

        model._fit_embedder()
        model.embedding = np.asarray(model.embedder.transform(model.X), dtype=DTYPE)

    finally:
        os.chdir(old_cwd)

    model.save_embedder(grae_dir / "embedder.joblib", compress=JOBLIB_COMPRESS)
    return model


def make_tda_object_from_ph(k: int, embedding: np.ndarray, adata_shape, neigh_ind, neigh_dist, edge_list, dgms, features):
    tda = XplainTDA(
        adata=None,
        layer=LAYER,
        embedder=EMBEDDER,
        n_components=embedding.shape[1],
        n_neighbors=k,
        maxdim=KNN_MAXDIM,
        neighbor_backend=NEIGHBOR_BACKEND,
        ph_backend=KNN_PH_BACKEND,
        dtype=DTYPE,
        fitted_embedder=None,
        embedder_kwargs=EMBEDDER_KWARGS,
    )
    tda.n_cells, tda.n_genes = int(adata_shape[0]), int(adata_shape[1])
    tda.embedding = embedding
    tda.neigh_ind = neigh_ind
    tda.neigh_dist = neigh_dist
    tda.edge_list = edge_list
    tda.dgms = dgms
    tda.features = features
    return tda


def fit_and_save_knn_tda_single_inner_job(k: int, embedding: np.ndarray, out_path: Path, adata_shape) -> dict:
    print(f"Starting k={k}", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), flush=True)
    t0 = time.time()
    with maybe_limit_threads(INNER_THREAD_LIMIT):
        print(f"k={k}: computing neighbors", flush=True)
        neigh_ind, neigh_dist = compute_neighbors(embedding, k, NEIGHBOR_BACKEND)

        print(f"k={k}: computing PH with {KNN_PH_BACKEND}", flush=True)
        dgms, features, edge_list = compute_ph_from_neighbors(
            neigh_ind,
            neigh_dist,
            maxdim=KNN_MAXDIM,
            ph_backend=KNN_PH_BACKEND,
            localize=True,
        )

        print(f"k={k}: saving object", flush=True)
        tda = make_tda_object_from_ph(k, embedding, adata_shape, neigh_ind, neigh_dist, edge_list, dgms, features)
        tda.save(out_path, save_embedder=SAVE_EMBEDDER_IN_KNN_OBJECTS, save_adata=SAVE_ADATA_IN_TDA_OBJECTS, compress=JOBLIB_COMPRESS)

    print(f"Finished k={k} in {time.time() - t0:.2f} s", flush=True)
    return {"k": k, "label": f"k={k}", "path": str(out_path), "dgms": dgms, "elapsed_seconds": time.time() - t0}

def one_bottleneck(label_a: str, label_b: str, dim: int, dgms_a, dgms_b) -> dict:
    return {
        "label_a": label_a,
        "label_b": label_b,
        "homology_dim": dim,
        "bottleneck_distance": float(gd.bottleneck_distance(dgms_a[dim], dgms_b[dim])),
    }


def save_heatmaps(pairwise_df: pd.DataFrame, full_label: str, k_labels: list[str], knn_maxdim: int, outdir: Path):
    for dim in range(knn_maxdim + 1):
        labels = [full_label] + k_labels if dim == 0 else k_labels
        mat = pd.DataFrame(0.0, index=labels, columns=labels)

        sub = pairwise_df[pairwise_df["homology_dim"] == dim]
        for row in sub.itertuples(index=False):
            if row.label_a in labels and row.label_b in labels:
                mat.loc[row.label_a, row.label_b] = row.bottleneck_distance
                mat.loc[row.label_b, row.label_a] = row.bottleneck_distance

        mat.to_csv(outdir / f"bottleneck_heatmap_H{dim}.csv")


def dump_json(path: Path, obj):
    with path.open("w") as f:
        json.dump(obj, f, indent=2, default=str)


def main():

    ADATA_PATH = sys.argv[1]
    OUTDIR = sys.argv[2]

    names = ["Script", "ADATA_PATH","OUTDIR"]
    for n, arg in zip(names, sys.argv):
	    print(n, arg, flush=True)   
    
    OUTDIR = Path(OUTDIR).resolve()
    tda_dir = OUTDIR / "tda_objects"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    tda_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "adata": ADATA_PATH,
        "ks": KS,
        "layer": LAYER,
        "embedder": EMBEDDER,
        "n_components": N_COMPONENTS,
        "full_maxdim": FULL_MAXDIM,
        "knn_maxdim": KNN_MAXDIM,
        "knn_ph_backend": KNN_PH_BACKEND,
        "neighbor_backend": NEIGHBOR_BACKEND,
        "dtype": str(DTYPE),
        "n_jobs_tda": N_JOBS_TDA,
        "n_jobs_bottleneck": N_JOBS_BOTTLENECK,
        "joblib_backend": JOBLIB_BACKEND,
        "inner_thread_limit": INNER_THREAD_LIMIT,
        "knn_inner_jobs": 1,
    }
    dump_json(OUTDIR / "config.json", config)

    adata = ad.read_h5ad(ADATA_PATH)
    print(adata, flush=True)

    t0 = time.time()
    embedding_model = fit_embed_once(adata, OUTDIR)
    embedding = embedding_model.embedding
    np.save(OUTDIR / "embedding.npy", embedding)
    print(f"Embedding computed and saved in {time.time() - t0:.2f} s", flush=True)

    t0 = time.time()
    knn_jobs = [
        delayed(fit_and_save_knn_tda_single_inner_job)(
            k, embedding, tda_dir / f"tda_knn_k{k}.joblib", adata.shape
        )
        for k in KS
    ]
    knn_results = Parallel(n_jobs=N_JOBS_TDA, backend=JOBLIB_BACKEND)(knn_jobs)
    knn_results = sorted(knn_results, key=lambda r: r["k"])
    pd.DataFrame([{k: r[k] for k in ["k", "label", "path", "elapsed_seconds"]} for r in knn_results]).to_csv(
        OUTDIR / "knn_tda_jobs.csv",
        index=False,
    )
    print(f"All outer-parallel kNN PH jobs finished in {time.time() - t0:.2f} s", flush=True)

    t0 = time.time()
    distance_matrix = squareform(pdist(embedding, metric="euclidean")).astype(DTYPE, copy=False)
    np.save(OUTDIR / "full_distance_matrix.npy", distance_matrix)
    print(f"Full distance matrix computed and saved in {time.time() - t0:.2f} s", flush=True)

    t0 = time.time()
    full_result = ph_from_full_distance_matrix(distance_matrix, maxdim=FULL_MAXDIM)

    joblib.dump({
        "config": config | {"mode": "full_distance_matrix"},
        "distance_matrix_path": str(OUTDIR / "full_distance_matrix.npy"),
        "dgms": full_result["dgms"],
        "features": full_result["features"],
        "edge_list": full_result["edge_list"],
    }, tda_dir / "tda_full_distance.joblib", compress=JOBLIB_COMPRESS)
    print(f"Full-graph PH computed and saved in {time.time() - t0:.2f} s", flush=True)

    
    dgms_by_label = {"full": full_result["dgms"]}
    dgms_by_label.update({r["label"]: r["dgms"] for r in knn_results})
    labels = ["full"] + [r["label"] for r in knn_results]

    t0 = time.time()
    full_vs_k_tasks = [
        delayed(one_bottleneck)("full", r["label"], 0, dgms_by_label["full"], dgms_by_label[r["label"]])
        for r in knn_results]
    
    full_vs_k_rows = Parallel(n_jobs=N_JOBS_BOTTLENECK, backend=JOBLIB_BACKEND)(full_vs_k_tasks)
    pd.DataFrame(full_vs_k_rows).to_csv(OUTDIR / "bottleneck_full_vs_k.csv", index=False)
    print(f"Full-vs-k bottleneck distances finished in {time.time() - t0:.2f} s")

    t0 = time.time()
    k_labels = [r["label"] for r in knn_results]
    remaining_tasks = [
        delayed(one_bottleneck)(k_labels[i], k_labels[j], dim, dgms_by_label[k_labels[i]], dgms_by_label[k_labels[j]])
        for i in range(len(k_labels))
        for j in range(i + 1, len(k_labels))
        for dim in range(KNN_MAXDIM + 1)
    ]
    remaining_rows = Parallel(n_jobs=N_JOBS_BOTTLENECK, backend=JOBLIB_BACKEND)(remaining_tasks)
    print(f"Remaining k-vs-k bottleneck distances finished in {time.time() - t0:.2f} s", flush=True)

    pairwise_df = pd.DataFrame(full_vs_k_rows + remaining_rows)
    pairwise_df.to_csv(OUTDIR / "bottleneck_pairwise_long.csv", index=False)
    pairwise_df.to_json(OUTDIR / "bottleneck_pairwise_long.json", orient="records", indent=2)
    save_heatmaps(pairwise_df, "full", k_labels, KNN_MAXDIM, OUTDIR)
    print(f"Done. Results written to: {OUTDIR}", flush=True)


if __name__ == "__main__":
    main()
