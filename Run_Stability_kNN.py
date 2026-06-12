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
- Compute ordinary bottleneck distances for pairwise heatmaps.
- Compute quantile bottleneck distances for several quantiles, retaining both
  raw values and three normalised versions.

Edit the configuration block below, especially KS.
"""

from __future__ import annotations

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

from XplainTDA import (
    XplainTDA, compute_ph_from_neighbors, quantile_bottleneck,
    simplex_tree_to_diagrams,
)


# ---------------------------------------------------------------------
# User-editable configuration
# ---------------------------------------------------------------------
KS = [5, 15, 50, 75, 100, 250, 500, 750, 1000, 1250]
QS = (0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
SCALES = ("S_diameter", "S_filtration_range", "S_longest_lifespan_dim")

# Use project defaults where they exist.
LAYER = d.DEFAULT_LAYER
EMBEDDER = d.DEFAULT_EMBEDDER
EMBEDDING_KEY = None
N_COMPONENTS = d.DEFAULT_N_COMPONENTS
FULL_MAXDIM = 0
KNN_MAXDIM = d.DEFAULT_MAXDIM
KNN_PH_BACKEND = d.DEFAULT_GLOBAL_PH_BACKEND
NEIGHBOR_BACKEND = d.DEFAULT_NEIGHBOR_BACKEND
DTYPE = d.DEFAULT_DTYPE
EMBEDDER_KWARGS = d.DEFAULT_EMBEDDER_KWARGS or {}

SAVE_ADATA_IN_TDA_OBJECTS = False
SAVE_EMBEDDER_IN_KNN_OBJECTS = False
JOBLIB_COMPRESS = 3

# Parallel k jobs. kNN inside each job is kept single-worker.
N_JOBS_TDA = 1
N_JOBS_BOTTLENECK = d.DEFAULT_N_JOBS
N_JOBS_METRICS = min(4, d.DEFAULT_N_JOBS)
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


def get_embedding_once(adata, outdir, embedding_key=None):
    """
    Obtain one embedding for all stability computations.

    If embedding_key is provided, use adata.obsm[embedding_key].
    Otherwise, fit the configured embedder and save it under OUTDIR/<EMBEDDER>.
    """
    outdir = Path(outdir).resolve()

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
        embedder_kwargs=EMBEDDER_KWARGS,
        embedding_key=embedding_key,
    )

    # Case 1: use a stored embedding from adata.obsm.
    if embedding_key is not None:
        if embedding_key not in adata.obsm:
            raise KeyError(
                f"Embedding {embedding_key!r} was not found in adata.obsm. "
                f"Available keys: {list(adata.obsm.keys())}"
            )

        model.X = adata.X if LAYER == "X" else adata.layers[LAYER]
        model.n_cells, model.n_genes = model.X.shape
        model.embedding = model._validate_embedding(adata.obsm[embedding_key])
        model.n_components = model.embedding.shape[1]

        print(
            f"Using stored embedding adata.obsm[{embedding_key!r}] "
            f"with shape {model.embedding.shape}",
            flush=True,
        )
        return model

    # Case 2: fit the configured embedder.
    embedder_dir = outdir / EMBEDDER
    embedder_dir.mkdir(parents=True, exist_ok=True)

    old_cwd = os.getcwd()
    os.chdir(embedder_dir)
    try:
        model.X = adata.X if LAYER == "X" else adata.layers[LAYER]
        model.n_cells, model.n_genes = model.X.shape

        if model.n_components is None:
            model.n_components = int(np.round(model.n_genes ** (1 / 3)))

        model._fit_embedder()
        model.embedding = model._validate_embedding(
            model.embedder.transform(model.X)
        )
    finally:
        os.chdir(old_cwd)

    model.save_embedder(
        embedder_dir / "embedder.joblib",
        compress=JOBLIB_COMPRESS,
    )
    return model


def make_tda_object_from_ph(k: int, embedding: np.ndarray, adata_shape, neigh_ind, neigh_dist, edge_list, dgms, features, embedding_key=None):
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
        embedding_key=embedding_key,
    )
    tda.n_cells, tda.n_genes = int(adata_shape[0]), int(adata_shape[1])
    tda.embedding = embedding
    tda.neigh_ind = neigh_ind
    tda.neigh_dist = neigh_dist
    tda.edge_list = edge_list
    tda.dgms = dgms
    tda.features = features
    return tda


def fit_and_save_knn_tda_single_inner_job(k: int, embedding: np.ndarray, out_path: Path, adata_shape, embedding_key=None) -> dict:
    print(f"Starting k={k}", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), flush=True)
    t0 = time.time()
    with maybe_limit_threads(INNER_THREAD_LIMIT):
        print(f"k={k}: computing neighbors", flush=True)
        neigh_ind, neigh_dist = compute_neighbors(embedding, k, NEIGHBOR_BACKEND)

        print(f"k={k}: computing PH with {KNN_PH_BACKEND}", flush=True)
        dgms, features, edge_list = compute_ph_from_neighbors(
            neigh_ind, neigh_dist, maxdim=KNN_MAXDIM,
            ph_backend=KNN_PH_BACKEND, localize=True,
        )

        print(f"k={k}: saving object to {out_path}", flush=True)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tda = make_tda_object_from_ph(k, embedding, adata_shape, neigh_ind, neigh_dist, edge_list, dgms, features, embedding_key=embedding_key)
        tda.save(out_path, save_embedder=SAVE_EMBEDDER_IN_KNN_OBJECTS, save_adata=SAVE_ADATA_IN_TDA_OBJECTS, compress=JOBLIB_COMPRESS)

    print(f"Finished k={k} in {time.time() - t0:.2f} s", flush=True)
    return {
        "k": k, "label": f"k={k}", "path": str(out_path),
        "dgms": dgms, "edge_list": edge_list,
        "elapsed_seconds": time.time() - t0,
    }

def one_bottleneck(label_a: str, label_b: str, dim: int, dgms_a, dgms_b) -> dict:
    return {
        "label_a": label_a,
        "label_b": label_b,
        "homology_dim": dim,
        "bottleneck_distance": float(gd.bottleneck_distance(dgms_a[dim], dgms_b[dim])),
    }


def q_token(q: float) -> str:
    """Convert a quantile to a safe column-name token."""
    return "q" + f"{q:g}".replace(".", "_")


def safe_divide(value: float, scale: float) -> float:
    """Normalise a distance, returning NaN for invalid scales."""
    return float(value / scale) if np.isfinite(scale) and scale > 0 else np.nan


def filtration_range(edge_list, dgms) -> float:
    """Largest filtration value represented by one PH result."""
    edges = np.asarray(edge_list)
    if edges.size:
        return float(np.nanmax(edges[:, 2]))
    deaths = [np.asarray(d)[:, 1] for d in dgms if np.asarray(d).size]
    deaths = np.concatenate(deaths) if deaths else np.array([], dtype=float)
    deaths = deaths[np.isfinite(deaths)]
    return float(np.nanmax(deaths)) if deaths.size else np.nan


def longest_lifespan_by_dim(states: dict, maxdim: int) -> dict[int, float]:
    """Longest positive finite persistence across all PH results."""
    out = {}
    for dim in range(maxdim + 1):
        pieces = []
        for state in states.values():
            if dim >= len(state["dgms"]):
                continue
            dgm = np.asarray(state["dgms"][dim], dtype=float)
            if dgm.size:
                dgm = dgm[np.isfinite(dgm).all(axis=1)]
                pers = dgm[:, 1] - dgm[:, 0]
                pieces.append(pers[pers > 0])
        pieces = [x for x in pieces if x.size]
        out[dim] = float(np.max(np.concatenate(pieces))) if pieces else np.nan
    return out


def infer_dataset_omic(outdir: Path) -> tuple[str, str]:
    """Read dataset and omic from Datasets/<dataset>/XplainTDA/<omic>/..."""
    parts = outdir.parts
    try:
        i = parts.index("Datasets")
        return parts[i + 1], parts[i + 3]
    except (ValueError, IndexError):
        return "unknown", "unknown"


def quantile_metric_row(task: dict, diagrams: dict, scales: dict) -> dict:
    """Compute raw quantile bottleneck values and three normalisations."""
    dim, k1, k2 = task["homology_dim"], task["k1"], task["k2"]
    D, E = diagrams[k1][dim], diagrams[k2][dim]
    pair_scales = {
        "S_diameter": scales["S_diameter"],
        "S_filtration_range": float(np.nanmax([
            scales["S_filtration_range_by_label"][k1],
            scales["S_filtration_range_by_label"][k2],
        ])),
        "S_longest_lifespan_dim": scales["S_longest_lifespan_by_dim"][dim],
    }

    out = dict(task)
    for q in QS:
        raw = float(quantile_bottleneck(D, E, q=q))
        token = q_token(q)
        out[f"bottleneck_{token}_raw"] = raw
        for scale_name in SCALES:
            out[f"bottleneck_{token}_{scale_name}"] = safe_divide(
                raw, pair_scales[scale_name]
            )
    return out


def compute_quantile_metrics(states: dict, distance_matrix: np.ndarray,
                             outdir: Path) -> pd.DataFrame:
    """Compute H0 full-vs-k and H1/H2 max-k-vs-k stability metrics."""
    k_labels = sorted(
        (x for x in states if x.startswith("k=")),
        key=lambda x: int(x[2:]),
    )
    k_max = k_labels[-1]

    tasks = [{"homology_dim": 0, "k1": "full", "k2": k} for k in k_labels]
    for dim in range(1, KNN_MAXDIM + 1):
        tasks.extend(
            {"homology_dim": dim, "k1": k_max, "k2": k}
            for k in k_labels
        )

    scales = {
        "S_diameter": float(np.nanmax(distance_matrix)),
        "S_filtration_range_by_label": {
            label: filtration_range(state["edge_list"], state["dgms"])
            for label, state in states.items()
        },
        "S_longest_lifespan_by_dim": longest_lifespan_by_dim(
            states, KNN_MAXDIM
        ),
    }

    # Workers receive diagrams only; edge lists were already reduced to scales.
    diagrams = {label: state["dgms"] for label, state in states.items()}
    rows = Parallel(n_jobs=N_JOBS_METRICS, backend=JOBLIB_BACKEND)(
        delayed(quantile_metric_row)(task, diagrams, scales) for task in tasks
    )
    result = pd.DataFrame(rows)
    dataset, omic = infer_dataset_omic(outdir)
    result.insert(0, "omic", omic)
    result.insert(0, "dataset", dataset)
    result.insert(2, "source_outdir", str(outdir))

    id_cols = ["dataset", "omic", "source_outdir", "homology_dim", "k1", "k2"]
    metric_cols = []
    for q in QS:
        token = q_token(q)
        metric_cols.append(f"bottleneck_{token}_raw")
        metric_cols.extend(f"bottleneck_{token}_{s}" for s in SCALES)
    return result[id_cols + metric_cols]


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
    if len(sys.argv) not in (3, 4):
        raise SystemExit(
            "Usage: python Run_Stability_kNN.py "
            "<adata_path> <outdir> [embedding_key]")

    ADATA_PATH = sys.argv[1]
    OUTDIR = sys.argv[2]

    # CLI value overrides the configuration default.
    embedding_key = sys.argv[3] if len(sys.argv) == 4 else EMBEDDING_KEY

    print("Script", sys.argv[0], flush=True)
    print("ADATA_PATH", ADATA_PATH, flush=True)
    print("OUTDIR", OUTDIR, flush=True)
    print("EMBEDDING_KEY", embedding_key, flush=True)  
    
    OUTDIR = Path(OUTDIR).resolve()
    tda_dir = OUTDIR / "tda_objects"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    tda_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "adata": ADATA_PATH,
        "ks": KS,
        "embedding_key": embedding_key,
        "quantiles": QS,
        "normalisation_scales": SCALES,
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
        "n_jobs_metrics": N_JOBS_METRICS,
        "joblib_backend": JOBLIB_BACKEND,
        "inner_thread_limit": INNER_THREAD_LIMIT,
        "knn_inner_jobs": 1,
    }
    dump_json(OUTDIR / "config.json", config)

    adata = ad.read_h5ad(ADATA_PATH)
    print(adata, flush=True)

    t0 = time.time()
    embedding_model = get_embedding_once(adata, OUTDIR, embedding_key=embedding_key)
    embedding = embedding_model.embedding
    np.save(OUTDIR / "embedding.npy", embedding)
    print(f"Embedding computed and saved in {time.time() - t0:.2f} s", flush=True)

    # Compute all sparse kNN PH results first.
    t0 = time.time()
    knn_jobs = [
    delayed(fit_and_save_knn_tda_single_inner_job)(
        k,
        embedding,
        tda_dir / f"tda_knn_k{k}.joblib",
        adata.shape,
        embedding_key)    for k in KS]  
    
    knn_results = Parallel(n_jobs=N_JOBS_TDA, backend=JOBLIB_BACKEND)(knn_jobs)
    knn_results = sorted(knn_results, key=lambda r: r["k"])
    pd.DataFrame([
        {key: r[key] for key in ("k", "label", "path", "elapsed_seconds")}
        for r in knn_results
    ]).to_csv(OUTDIR / "knn_tda_jobs.csv", index=False)
    print(f"All kNN PH jobs finished in {time.time() - t0:.2f} s", flush=True)

    # Build the complete-distance H0 reference only after the kNN runs.
    t0 = time.time()
    distance_matrix = squareform(pdist(embedding, metric="euclidean")).astype(
        DTYPE, copy=False
    )
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

    # Reuse in-memory PH results to compute raw and normalised quantile metrics.
    states = {
        "full": {"dgms": full_result["dgms"], "edge_list": full_result["edge_list"]},
        **{
            r["label"]: {"dgms": r["dgms"], "edge_list": r["edge_list"]}
            for r in knn_results
        },
    }
    t0 = time.time()
    quantile_df = compute_quantile_metrics(states, distance_matrix, OUTDIR)
    # The main filename reflects that the table now contains raw and normalised values.
    quantile_path = OUTDIR / "quantile_bottleneck_metrics.csv"
    quantile_df.to_csv(quantile_path, index=False)

    # Keep the old filename for compatibility with existing analysis notebooks.
    legacy_path = OUTDIR / "quantile_bottleneck_normalised.csv"
    quantile_df.to_csv(legacy_path, index=False)
    print(
        f"Quantile bottleneck metrics written to {quantile_path} "
        f"in {time.time() - t0:.2f} s", flush=True,
    )

    dgms_by_label = {"full": full_result["dgms"]}
    dgms_by_label.update({r["label"]: r["dgms"] for r in knn_results})

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
