#!/usr/bin/env python3
"""Subsampling stability for XplainTDA using one fixed embedding.

Workflow
--------
1. Fit the embedder once on the full AnnData object.
2. Draw independent stratified cell subsets without replacement.
3. Recompute kNN persistent homology for each subset in the fixed embedding.
4. Compute ordinary bottleneck and quantile-bottleneck distances between all
   subset pairs for H0, H1, ..., HMAX.
5. Save raw quantile metrics separately from their normalised versions.

Usage
-----
python Run_Stability_Subsampling_Quantile.py input.h5ad output_dir
python Run_Stability_Subsampling_Quantile.py input.h5ad output_dir CellType 5 0.25
python Run_Stability_Subsampling_Quantile.py input.h5ad output_dir CellType 5 0.25 X_pca
"""

from __future__ import annotations

import json
import os
import sys
import argparse
import time
from contextlib import contextmanager
from itertools import combinations
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import anndata as ad
import gudhi as gd
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None

from defaults import (
    DEFAULT_BACKEND,
    DEFAULT_DTYPE,
    DEFAULT_EMBEDDER,
    DEFAULT_EMBEDDER_KWARGS,
    DEFAULT_LAYER,
    DEFAULT_MAXDIM,
    DEFAULT_N_COMPONENTS,
    DEFAULT_N_JOBS,
    DEFAULT_N_NEIGHBORS,
    DEFAULT_NEIGHBOR_BACKEND,
)
from XplainTDA import XplainTDA, quantile_bottleneck


# -----------------------------------------------------------------------------
# User configuration
# -----------------------------------------------------------------------------
N_SUBSETS = 5
SUBSET_FRAC = 0.25

LAYER = DEFAULT_LAYER
EMBEDDER = DEFAULT_EMBEDDER
# None fits EMBEDDER; otherwise use adata.obsm[EMBEDDING_KEY].
EMBEDDING_KEY = None
N_COMPONENTS = DEFAULT_N_COMPONENTS
N_NEIGHBORS = DEFAULT_N_NEIGHBORS
MAXDIM = DEFAULT_MAXDIM
NEIGHBOR_BACKEND = DEFAULT_NEIGHBOR_BACKEND
PH_BACKEND = "ripser"
DTYPE = DEFAULT_DTYPE
EMBEDDER_KWARGS = DEFAULT_EMBEDDER_KWARGS or {}

QUANTILES = (0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
DIAMETER_BLOCK_SIZE = 2048

N_JOBS = DEFAULT_N_JOBS
JOBLIB_BACKEND = DEFAULT_BACKEND
INNER_THREAD_LIMIT = 1

SAVE_ADATA_IN_TDA_OBJECTS = False
SAVE_EMBEDDER_IN_TDA_OBJECTS = False
SAVE_FULL_EMBEDDER = True
JOBLIB_COMPRESS = 3
# -----------------------------------------------------------------------------


@contextmanager
def pushd(path: Path):
    """Temporarily change directory, isolating GRAE checkpoint files."""
    old = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


@contextmanager
def limit_threads(n_threads: int):
    """Limit BLAS/OpenMP threads when threadpoolctl is available."""
    if threadpool_limits is None:
        yield
    else:
        with threadpool_limits(limits=n_threads):
            yield


def now() -> str:
    return time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())


def dump_json(path: Path, obj) -> None:
    with path.open("w") as f:
        json.dump(obj, f, indent=2, default=str)


def q_token(q: float) -> str:
    """Convert 0.75 to the column-name token q0_75."""
    return "q" + f"{q:g}".replace(".", "_")


def safe_divide(value: float, scale: float) -> float:
    return float(value / scale) if np.isfinite(scale) and scale > 0 else np.nan


def finite_diagram(dgm: np.ndarray) -> np.ndarray:
    """Keep finite persistence points with strictly positive lifetimes."""
    dgm = np.asarray(dgm, dtype=float).reshape(-1, 2)
    return dgm[np.isfinite(dgm).all(axis=1) & (dgm[:, 1] > dgm[:, 0])]


def get_full_embedding(
    adata: ad.AnnData,
    outdir: Path,
    embedding_key: str | None = None,
) -> tuple[np.ndarray, int]:
    """Load a stored embedding or fit the configured embedder once."""
    model = XplainTDA(
        adata=adata,
        layer=LAYER,
        embedder=EMBEDDER,
        n_components=N_COMPONENTS,
        n_neighbors=N_NEIGHBORS,
        maxdim=MAXDIM,
        neighbor_backend=NEIGHBOR_BACKEND,
        ph_backend=PH_BACKEND,
        dtype=DTYPE,
        embedder_kwargs=dict(EMBEDDER_KWARGS),
        embedding_key=embedding_key,
    )
    model.X = adata.X if LAYER == "X" else adata.layers[LAYER]
    model.n_cells, model.n_genes = model.X.shape

    if embedding_key is not None:
        if embedding_key not in adata.obsm:
            raise KeyError(
                f"Embedding {embedding_key!r} was not found in adata.obsm. "
                f"Available keys: {list(adata.obsm.keys())}"
            )
        embedding = model._validate_embedding(adata.obsm[embedding_key])
        model.embedding = embedding
        model.n_components = embedding.shape[1]
        print(
            f"[{now()}] Using adata.obsm[{embedding_key!r}] "
            f"with shape {embedding.shape}",
            flush=True,
        )
        return embedding, int(model.n_components)

    if model.n_components is None:
        model.n_components = int(np.round(model.n_genes ** (1 / 3)))

    with pushd(outdir / "full_embedding_workdir"), limit_threads(INNER_THREAD_LIMIT):
        print(
            f"[{now()}] Fitting {EMBEDDER} once on {model.n_cells} cells "
            f"with {model.n_components} components",
            flush=True,
        )
        model._fit_embedder()
        embedding = model._validate_embedding(model.embedder.transform(model.X))
        model.embedding = embedding

        if SAVE_FULL_EMBEDDER:
            model.save_embedder(
                outdir / "full_embedder.joblib",
                compress=JOBLIB_COMPRESS,
            )

    return embedding, int(model.n_components)


def sample_subset_indices(
    adata: ad.AnnData,
    subset_frac: float,
    stratify_by: str,
    seed: int,
) -> np.ndarray:
    """Draw one independent stratified subset without replacement."""
    if not 0 < subset_frac <= 1:
        raise ValueError("subset_frac must satisfy 0 < subset_frac <= 1.")
    if stratify_by not in adata.obs:
        raise KeyError(f"{stratify_by!r} not found in adata.obs.")

    rng = np.random.default_rng(seed)
    groups = adata.obs[stratify_by].astype("category")
    sampled = []

    for group_idx in adata.obs.groupby(groups, observed=True).indices.values():
        group_idx = np.asarray(group_idx)
        n_sample = min(max(1, round(subset_frac * len(group_idx))), len(group_idx))
        sampled.append(rng.choice(group_idx, size=n_sample, replace=False))

    idx = np.concatenate(sampled)
    rng.shuffle(idx)
    return idx


def fit_subset_tda(
    subset_id: int,
    embedding_path: str,
    indices: np.ndarray,
    subset_dir: Path,
    n_components: int,
    embedding_key: str | None = None,
) -> dict:
    """Compute PH for one subset using rows of the fixed full embedding."""
    t0 = time.time()
    subset_dir.mkdir(parents=True, exist_ok=True)

    full_embedding = np.load(embedding_path, mmap_mode="r")
    sub_embedding = np.asarray(full_embedding[indices], dtype=DTYPE)

    with limit_threads(INNER_THREAD_LIMIT):
        print(
            f"[{now()}] subset_{subset_id}: PH on {len(sub_embedding)} embedded cells",
            flush=True,
        )
        model = XplainTDA(
            adata=None,
            layer=LAYER,
            embedder=EMBEDDER,
            n_components=n_components,
            n_neighbors=min(int(N_NEIGHBORS), len(sub_embedding)),
            maxdim=MAXDIM,
            neighbor_backend=NEIGHBOR_BACKEND,
            ph_backend=PH_BACKEND,
            dtype=DTYPE,
            embedder_kwargs=dict(EMBEDDER_KWARGS),
            embedding_key=embedding_key,
        ).fit(embedding=sub_embedding)

        tda_path = subset_dir / f"tda_subset_{subset_id}.joblib"
        model.save(
            tda_path,
            save_embedder=SAVE_EMBEDDER_IN_TDA_OBJECTS,
            save_adata=SAVE_ADATA_IN_TDA_OBJECTS,
            compress=JOBLIB_COMPRESS,
        )

    return {
        "subset_id": subset_id,
        "label": f"subset_{subset_id}",
        "path": str(tda_path),
        "n_cells": int(len(sub_embedding)),
        "n_neighbors": int(model.n_neighbors),
        "n_components": int(n_components),
        "elapsed_seconds": time.time() - t0,
        "dgms": model.dgms,
        "edge_list": model.edge_list,
    }


def embedding_diameter(X: np.ndarray, block_size: int = DIAMETER_BLOCK_SIZE) -> float:
    """Compute the exact Euclidean diameter in blocks to control memory usage."""
    if len(X) < 2:
        return np.nan

    max_d2 = 0.0
    for i in range(0, len(X), block_size):
        A = np.asarray(X[i:i + block_size], dtype=float)
        A2 = np.sum(A * A, axis=1)[:, None]
        for j in range(i, len(X), block_size):
            B = np.asarray(X[j:j + block_size], dtype=float)
            d2 = A2 + np.sum(B * B, axis=1)[None, :] - 2.0 * (A @ B.T)
            np.maximum(d2, 0.0, out=d2)
            if d2.size:
                max_d2 = max(max_d2, float(np.nanmax(d2)))

    return float(np.sqrt(max_d2))


def filtration_range(result: dict) -> float:
    """Largest edge filtration value for one subset PH object."""
    edges = np.asarray(result["edge_list"])
    return float(np.nanmax(edges[:, 2])) if edges.size else np.nan


def longest_lifespans(results: list[dict]) -> dict[int, float]:
    """Largest finite persistence across all subsets, separately for each H."""
    out = {}
    for dim in range(MAXDIM + 1):
        lifetimes = []
        for result in results:
            dgm = finite_diagram(result["dgms"][dim])
            if len(dgm):
                lifetimes.append(dgm[:, 1] - dgm[:, 0])
        out[dim] = float(np.max(np.concatenate(lifetimes))) if lifetimes else np.nan
    return out


def compute_pair_metrics(
    a: dict,
    b: dict,
    dim: int,
    diameter: float,
    filtration_ranges: dict[str, float],
    lifespan_scales: dict[int, float],
) -> tuple[dict, dict, dict]:
    """Compute ordinary bottleneck plus raw and normalised quantile metrics."""
    D, E = a["dgms"][dim], b["dgms"][dim]
    pair = {
        "homology_dim": dim,
        "subset_a": a["label"],
        "subset_b": b["label"],
    }

    ordinary = {
        "label_a": a["label"],
        "label_b": b["label"],
        "homology_dim": dim,
        "metric": "bottleneck",
        "distance": float(gd.bottleneck_distance(finite_diagram(D), finite_diagram(E))),
    }

    raw = dict(pair)
    scales = {
        "S_diameter": diameter,
        "S_filtration_range": float(np.nanmax([
            filtration_ranges[a["label"]], filtration_ranges[b["label"]]
        ])),
        "S_longest_lifespan_dim": lifespan_scales[dim],
    }
    normalised = {**pair, **scales}

    # Compute each raw quantile distance once, then reuse it for all scales.
    for q in QUANTILES:
        name = f"bottleneck_{q_token(q)}"
        value = float(quantile_bottleneck(D, E, q=q))
        raw[name] = value
        for scale_name, scale in scales.items():
            normalised[f"{name}_{scale_name}"] = safe_divide(value, scale)

    return ordinary, raw, normalised


def save_bottleneck_heatmaps(df: pd.DataFrame, labels: list[str], outdir: Path) -> None:
    """Save one symmetric ordinary-bottleneck matrix per homology dimension."""
    for dim in sorted(df["homology_dim"].unique()):
        matrix = pd.DataFrame(0.0, index=labels, columns=labels)
        for row in df[df["homology_dim"] == dim].itertuples(index=False):
            matrix.loc[row.label_a, row.label_b] = row.distance
            matrix.loc[row.label_b, row.label_a] = row.distance
        matrix.to_csv(outdir / f"bottleneck_heatmap_H{dim}.csv")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute subsampling stability from one fixed full-data embedding, "
            "using ordinary and quantile bottleneck distances."
        )
    )

    parser.add_argument(
        "input_h5ad",
        type=Path,
        help="Input AnnData .h5ad file.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where results will be written.",
    )
    parser.add_argument(
        "--stratify-by",
        default="CellType",
        help="Column in adata.obs used for stratified sampling.",
    )
    parser.add_argument(
        "--n-subsets",
        type=int,
        default=N_SUBSETS,
        help=f"Number of independent subsets. Default: {N_SUBSETS}.",
    )
    parser.add_argument(
        "--subset-frac",
        type=float,
        default=SUBSET_FRAC,
        help=f"Fraction of cells sampled in each subset. Default: {SUBSET_FRAC}.",
    )
    parser.add_argument(
        "--embedding-strategy",
        choices=["GRAE", "PCA"],
        default=EMBEDDER,
        help=(
            "Method used to fit the full-data embedding when no precomputed "
            "embedding key is supplied."
        ),
    )
    parser.add_argument(
        "--embedding-key",
        default=None,
        help=(
            "Optional key in adata.obsm containing a precomputed embedding, "
            "for example X_GRAE or X_pca. When supplied, embedding fitting is skipped."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    adata_path = str(args.input_h5ad.resolve())
    outdir = args.output_dir.resolve()
    stratify_by = args.stratify_by
    n_subsets = args.n_subsets
    subset_frac = args.subset_frac
    embedding_strategy = args.embedding_strategy.upper()
    embedding_key = args.embedding_key

    if n_subsets < 2:
        raise ValueError("n_subsets must be at least 2.")

    outdir.mkdir(parents=True, exist_ok=True)
    subsets_dir = outdir / "subsets"
    subsets_dir.mkdir(exist_ok=True)

    print("=" * 70, flush=True)
    print("RUN CONFIGURATION", flush=True)
    print("=" * 70, flush=True)

    for name, value in vars(args).items():
        print(f"{name:25s}: {value}", flush=True)
    print("=" * 70, flush=True)

    adata = ad.read_h5ad(adata_path)
    print(adata, flush=True)

    embedding, n_components = get_full_embedding(
        adata,
        outdir,
        embedding_key=embedding_key,
    )
    embedding_path = outdir / "full_embedding.npy"
    np.save(embedding_path, embedding)
    pd.DataFrame({
        "source_cell_index": np.arange(adata.n_obs),
        "source_obs_name": adata.obs_names.to_numpy(),
    }).to_csv(outdir / "full_embedding_cells.csv", index=False)

    # Generate seeds in the parent process so parallel workers cannot share RNG state.
    seeds = [
        int(s.generate_state(1)[0])
        for s in np.random.SeedSequence().spawn(n_subsets)
    ]
    jobs, metadata = [], []

    for subset_id, seed in enumerate(seeds):
        subset_dir = subsets_dir / f"subset_{subset_id}"
        subset_dir.mkdir(parents=True, exist_ok=True)
        indices = sample_subset_indices(adata, subset_frac, stratify_by, seed)
        cells_path = subset_dir / "cells.csv"

        pd.DataFrame({
            "subset_id": subset_id,
            "subset_seed": seed,
            "subset_row": np.arange(len(indices)),
            "source_cell_index": indices,
            "source_obs_name": adata.obs_names[indices].to_numpy(),
        }).to_csv(cells_path, index=False)

        metadata.append({
            "subset_id": subset_id,
            "cells_path": str(cells_path),
            "subset_seed": seed,
        })
        jobs.append(delayed(fit_subset_tda)(
            subset_id,
            str(embedding_path),
            indices,
            subset_dir,
            n_components,
            embedding_key,
        ))

    dump_json(outdir / "config.json", {
        "adata_path": adata_path,
        "n_subsets": n_subsets,
        "subset_frac": subset_frac,
        "stratify_by": stratify_by,
        "sampling": "fixed full embedding; independent stratified subsets without replacement",
        "layer": LAYER,
        "embedder": EMBEDDER,
        "embedding_key": embedding_key,
        "n_components": n_components,
        "n_neighbors": N_NEIGHBORS,
        "maxdim": MAXDIM,
        "neighbor_backend": NEIGHBOR_BACKEND,
        "ph_backend": PH_BACKEND,
        "dtype": str(DTYPE),
        "embedder_kwargs": EMBEDDER_KWARGS,
        "quantiles": QUANTILES,
        "diameter_block_size": DIAMETER_BLOCK_SIZE,
        "n_jobs": N_JOBS,
        "joblib_backend": JOBLIB_BACKEND,
    })

    print(f"[{now()}] Computing subset PH", flush=True)
    results = Parallel(
        n_jobs=min(N_JOBS, n_subsets),
        backend=JOBLIB_BACKEND,
    )(jobs)
    results.sort(key=lambda x: x["subset_id"])

    metadata_by_id = {x["subset_id"]: x for x in metadata}
    for result in results:
        result.update(metadata_by_id[result["subset_id"]])

    summary_cols = [
        "subset_id", "label", "path", "cells_path", "subset_seed",
        "n_cells", "n_neighbors", "n_components", "elapsed_seconds",
    ]
    pd.DataFrame([{c: r[c] for c in summary_cols} for r in results]).to_csv(
        outdir / "subset_tda_jobs.csv", index=False
    )

    print(f"[{now()}] Computing bottleneck metrics", flush=True)
    diameter = embedding_diameter(embedding)
    filtration_ranges = {r["label"]: filtration_range(r) for r in results}
    lifespan_scales = longest_lifespans(results)

    rows = Parallel(n_jobs=N_JOBS, backend=JOBLIB_BACKEND)(
        delayed(compute_pair_metrics)(
            a, b, dim, diameter, filtration_ranges, lifespan_scales
        )
        for a, b in combinations(results, 2)
        for dim in range(MAXDIM + 1)
    )

    ordinary_df = pd.DataFrame([x[0] for x in rows])
    raw_df = pd.DataFrame([x[1] for x in rows])
    normalised_df = pd.DataFrame([x[2] for x in rows])

    ordinary_df.to_csv(outdir / "pairwise_distances_long.csv", index=False)
    ordinary_df.to_json(
        outdir / "pairwise_distances_long.json", orient="records", indent=2
    )
    raw_df.to_csv(
        outdir / "quantile_bottleneck_subsampling_raw.csv", index=False
    )
    normalised_df.to_csv(
        outdir / "quantile_bottleneck_subsampling_normalised.csv", index=False
    )
    save_bottleneck_heatmaps(
        ordinary_df, [r["label"] for r in results], outdir
    )

    print(f"[{now()}] Done. Results written to: {outdir}", flush=True)


if __name__ == "__main__":
    main()
