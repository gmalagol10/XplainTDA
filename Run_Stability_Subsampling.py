#!/usr/bin/env python3
"""Measure XplainTDA stability across stratified cell subsamples.

One full-data embedding is fitted or loaded and then held fixed. Independent
stratified subsets are drawn without replacement, persistent homology is
recomputed for each subset, and all subset pairs are compared in every
homology dimension.

Distances are saved raw and divided by the diameter of the full embedding.
No filtration-range or lifespan normalization is computed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from itertools import combinations
from pathlib import Path
from typing import Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import anndata as ad
import gudhi as gd
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import pairwise_distances_chunked

import defaults as d
from XplainTDA_Explainer import quantile_bottleneck_distance
from XplainTDA_PH import DEFAULT_PH_BACKEND, XplainTDA

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover - optional runtime optimization
    threadpool_limits = None


DEFAULT_N_SUBSETS = 5
DEFAULT_SUBSET_FRAC = 0.25
DEFAULT_QUANTILES = (0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
LAYER = d.DEFAULT_LAYER
EMBEDDER = d.DEFAULT_EMBEDDER
N_COMPONENTS = d.DEFAULT_N_COMPONENTS
N_NEIGHBORS = d.DEFAULT_N_NEIGHBORS
MAXDIM = d.DEFAULT_MAXDIM
NEIGHBOR_BACKEND = d.DEFAULT_NEIGHBOR_BACKEND
PH_BACKEND = DEFAULT_PH_BACKEND
DTYPE = d.DEFAULT_DTYPE
EMBEDDER_KWARGS = dict(d.DEFAULT_EMBEDDER_KWARGS or {})
JOBLIB_BACKEND = d.DEFAULT_BACKEND
N_JOBS = d.DEFAULT_N_JOBS
RANDOM_STATE = d.DEFAULT_RANDOM_STATE
INNER_THREADS = 1
COMPRESS = 3


@contextmanager
def working_directory(path: Path):
    """Temporarily isolate files created by an embedder during fitting."""
    previous = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@contextmanager
def limit_threads(n_threads: int = INNER_THREADS):
    """Limit native BLAS/OpenMP threads when threadpoolctl is installed."""
    if threadpool_limits is None:
        yield
    else:
        with threadpool_limits(limits=n_threads):
            yield


def parse_csv(value: str, cast) -> tuple:
    """Parse a non-empty comma-separated command-line value."""
    values = tuple(cast(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("provide at least one comma-separated value")
    return values


def finite_diagram(diagram: np.ndarray) -> np.ndarray:
    """Keep finite persistence points with strictly positive lifetime."""
    diagram = np.asarray(diagram, dtype=float).reshape(-1, 2)
    return diagram[
        np.isfinite(diagram).all(axis=1) & (diagram[:, 1] > diagram[:, 0])
    ]


def normalize(distance: float, diameter: float) -> float:
    """Divide by embedding diameter, returning NaN for a degenerate scale."""
    return float(distance / diameter) if np.isfinite(diameter) and diameter > 0 else np.nan


def embedding_diameter(embedding: np.ndarray) -> float:
    """Compute the exact Euclidean diameter without storing all pair distances."""
    maximum = 0.0
    for block in pairwise_distances_chunked(embedding, metric="euclidean"):
        if block.size:
            maximum = max(maximum, float(np.nanmax(block)))
    return maximum


def save_json(path: Path, payload: dict) -> None:
    """Write JSON metadata with Path and NumPy values converted to strings."""
    path.write_text(json.dumps(payload, indent=2, default=str))


def bounded_jobs(n_jobs: int | None, n_tasks: int) -> int | None:
    """Cap positive worker counts while preserving joblib's ``-1`` setting."""
    if n_jobs is None or n_jobs == -1:
        return n_jobs
    if n_jobs < 1:
        raise ValueError("n_jobs must be positive, -1, or None")
    return min(n_jobs, n_tasks)


def fit_embedding(
    adata: ad.AnnData,
    outdir: Path,
    embedding_key: str | None,
    embedder_name: str,
    n_neighbors: int,
) -> XplainTDA:
    """Load a precomputed embedding or fit the configured embedder once."""
    embedding = None
    if embedding_key is not None:
        if embedding_key not in adata.obsm:
            raise KeyError(
                f"adata.obsm[{embedding_key!r}] not found; available keys: "
                f"{list(adata.obsm.keys())}"
            )
        embedding = adata.obsm[embedding_key]

    tda = XplainTDA(
        adata=adata,
        layer=LAYER,
        embedder=embedder_name,
        n_components=N_COMPONENTS,
        n_neighbors=n_neighbors,
        maxdim=MAXDIM,
        neighbor_backend=NEIGHBOR_BACKEND,
        ph_backend=PH_BACKEND,
        dtype=DTYPE,
        embedder_kwargs=EMBEDDER_KWARGS,
    )
    with working_directory(outdir / "embedding_workdir"), limit_threads():
        tda.fit_embedding(embedding=embedding)

    if tda.embedder is not None:
        tda.save_embedder(outdir / "embedder.joblib", compress=COMPRESS)
    return tda


def sample_indices(
    adata: ad.AnnData,
    stratify_by: str,
    subset_fraction: float,
    seed: int,
) -> np.ndarray:
    """Draw one reproducible stratified sample without replacement."""
    rng = np.random.default_rng(seed)
    sampled = []
    groups = adata.obs[stratify_by]
    for group_indices in adata.obs.groupby(groups, observed=True).indices.values():
        group_indices = np.asarray(group_indices, dtype=int)
        n_sample = min(len(group_indices), max(1, round(subset_fraction * len(group_indices))))
        sampled.append(rng.choice(group_indices, n_sample, replace=False))
    indices = np.concatenate(sampled)
    rng.shuffle(indices)
    return indices


def fit_subset(
    subset_id: int,
    embedding_path: str,
    indices: np.ndarray,
    output_path: str,
    n_neighbors: int,
) -> dict:
    """Compute and save PH for one subset of the fixed embedding."""
    started = time.time()
    embedding = np.load(embedding_path, mmap_mode="r")
    subset_embedding = np.asarray(embedding[indices], dtype=DTYPE)
    if len(subset_embedding) < 2:
        raise ValueError(f"subset_{subset_id} contains fewer than two cells")

    with limit_threads():
        tda = XplainTDA(
            n_components=subset_embedding.shape[1],
            n_neighbors=min(n_neighbors, len(subset_embedding) - 1),
            maxdim=MAXDIM,
            neighbor_backend=NEIGHBOR_BACKEND,
            ph_backend=PH_BACKEND,
            dtype=DTYPE,
        ).fit(embedding=subset_embedding)
        path = tda.save(
            output_path,
            include_data=False,
            include_embedder=False,
            compress=COMPRESS,
        )

    return {
        "subset_id": subset_id,
        "label": f"subset_{subset_id}",
        "path": str(path),
        "n_cells": len(subset_embedding),
        "n_neighbors": tda.n_neighbors,
        "elapsed_seconds": time.time() - started,
        "dgms": tda.dgms,
    }


def bottleneck_row(a: dict, b: dict, dim: int, diameter: float) -> dict:
    """Compute one ordinary bottleneck distance and diameter normalization."""
    distance = float(
        gd.bottleneck_distance(
            finite_diagram(a["dgms"][dim]),
            finite_diagram(b["dgms"][dim]),
        )
    )
    return {
        "subset_a": a["label"],
        "subset_b": b["label"],
        "homology_dim": dim,
        "distance": distance,
        "distance_by_diameter": normalize(distance, diameter),
    }


def quantile_rows(
    a: dict,
    b: dict,
    dim: int,
    quantiles: Iterable[float],
    diameter: float,
) -> list[dict]:
    """Compute requested quantile-bottleneck distances for one subset pair."""
    reference, comparison = a["dgms"][dim], b["dgms"][dim]
    rows = []
    for quantile in quantiles:
        distance = quantile_bottleneck_distance(reference, comparison, q=quantile)
        rows.append(
            {
                "subset_a": a["label"],
                "subset_b": b["label"],
                "homology_dim": dim,
                "quantile": quantile,
                "distance": distance,
                "distance_by_diameter": normalize(distance, diameter),
            }
        )
    return rows


def save_heatmaps(results: pd.DataFrame, labels: list[str], outdir: Path) -> None:
    """Save one symmetric raw bottleneck-distance matrix per homology dimension."""
    for dim in range(MAXDIM + 1):
        matrix = pd.DataFrame(0.0, index=labels, columns=labels)
        for row in results[results["homology_dim"] == dim].itertuples(index=False):
            matrix.loc[row.subset_a, row.subset_b] = row.distance
            matrix.loc[row.subset_b, row.subset_a] = row.distance
        matrix.to_csv(outdir / f"bottleneck_heatmap_H{dim}.csv")


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_h5ad", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--stratify-by", default="CellType")
    parser.add_argument("--n-subsets", type=int, default=DEFAULT_N_SUBSETS)
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=N_NEIGHBORS,
        help=f"Number of neighbours in the kNN graph. Default: {N_NEIGHBORS}.",
    )
    parser.add_argument("--subset-frac", type=float, default=DEFAULT_SUBSET_FRAC)
    parser.add_argument("--embedding-key", help="Optional precomputed embedding in adata.obsm.")
    parser.add_argument("--embedder", default=EMBEDDER, help=f"Embedder used when fitting. Default: {EMBEDDER}.")
    parser.add_argument(
        "--quantiles",
        type=lambda value: parse_csv(value, float),
        default=DEFAULT_QUANTILES,
        help="Comma-separated quantiles in (0, 1].",
    )
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--n-jobs", type=int, default=N_JOBS)
    return parser.parse_args()


def main() -> None:
    """Run the complete stratified-subsampling stability benchmark."""
    args = parse_args()
    if args.n_subsets < 2:
        raise ValueError("n_subsets must be at least 2")
    if args.n_neighbors < 1:
        raise ValueError("n_neighbors must be positive")
    if not 0 < args.subset_frac <= 1:
        raise ValueError("subset_frac must satisfy 0 < subset_frac <= 1")
    if any(not 0 < q <= 1 for q in args.quantiles):
        raise ValueError("all quantiles must satisfy 0 < q <= 1")

    adata_path = args.input_h5ad.expanduser().resolve()
    if not adata_path.is_file():
        raise FileNotFoundError(adata_path)
    outdir = args.output_dir.expanduser().resolve()
    subsets_dir = outdir / "subsets"
    subsets_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(adata_path)
    if args.stratify_by not in adata.obs:
        raise KeyError(f"{args.stratify_by!r} is not a column in adata.obs")
    if adata.obs[args.stratify_by].notna().sum() < 2:
        raise ValueError("at least two labelled cells are required")

    print(f"Fitting or loading one embedding for {adata.n_obs} cells", time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    embedding_tda = fit_embedding(
        adata, outdir, args.embedding_key, args.embedder.upper(), args.n_neighbors
    )
    embedding_path = outdir / "full_embedding.npy"
    np.save(embedding_path, embedding_tda.embedding)
    pd.DataFrame(
        {"source_cell_index": np.arange(adata.n_obs), "source_obs_name": adata.obs_names}
    ).to_csv(outdir / "full_embedding_cells.csv", index=False)

    diameter = embedding_diameter(embedding_tda.embedding)
    seed_sequence = np.random.SeedSequence(args.random_state)
    seeds = [int(seed.generate_state(1)[0]) for seed in seed_sequence.spawn(args.n_subsets)]
    jobs, metadata = [], []

    for subset_id, seed in enumerate(seeds):
        subset_dir = subsets_dir / f"subset_{subset_id}"
        subset_dir.mkdir(exist_ok=True)
        indices = sample_indices(adata, args.stratify_by, args.subset_frac, seed)
        cells_path = subset_dir / "cells.csv"
        pd.DataFrame(
            {
                "subset_id": subset_id,
                "subset_seed": seed,
                "subset_row": np.arange(len(indices)),
                "source_cell_index": indices,
                "source_obs_name": adata.obs_names[indices],
            }
        ).to_csv(cells_path, index=False)
        metadata.append(
            {
                "subset_id": subset_id,
                "subset_seed": seed,
                "cells_path": str(cells_path),
            }
        )
        jobs.append(
            delayed(fit_subset)(
                subset_id,
                str(embedding_path),
                indices,
                str(subset_dir / f"tda_subset_{subset_id}.joblib"),
                args.n_neighbors,
            )
        )

    save_json(
        outdir / "config.json",
        {
            "input_h5ad": str(adata_path),
            "stratify_by": args.stratify_by,
            "n_subsets": args.n_subsets,
            "subset_frac": args.subset_frac,
            "random_state": args.random_state,
            "embedding_key": args.embedding_key,
            "embedder": args.embedder.upper(),
            "layer": LAYER,
            "n_components": embedding_tda.n_components,
            "n_neighbors": args.n_neighbors,
            "maxdim": MAXDIM,
            "neighbor_backend": NEIGHBOR_BACKEND,
            "ph_backend": PH_BACKEND,
            "quantiles": args.quantiles,
            "embedding_diameter": diameter,
            "normalization": "full_embedding_diameter_only",
            "n_jobs": args.n_jobs,
        },
    )

    print("Computing subset persistence diagrams", time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    results = Parallel(
        n_jobs=bounded_jobs(args.n_jobs, args.n_subsets),
        backend=JOBLIB_BACKEND,
    )(jobs)
    results.sort(key=lambda result: result["subset_id"])
    metadata_by_id = {item["subset_id"]: item for item in metadata}
    for result in results:
        result.update(metadata_by_id[result["subset_id"]])

    summary_columns = (
        "subset_id",
        "label",
        "path",
        "cells_path",
        "subset_seed",
        "n_cells",
        "n_neighbors",
        "elapsed_seconds",
    )
    pd.DataFrame(
        [{column: result[column] for column in summary_columns} for result in results]
    ).to_csv(outdir / "subset_tda_jobs.csv", index=False)

    pairs = list(combinations(results, 2))
    ordinary = Parallel(n_jobs=args.n_jobs, backend=JOBLIB_BACKEND)(
        delayed(bottleneck_row)(a, b, dim, diameter)
        for a, b in pairs
        for dim in range(MAXDIM + 1)
    )
    ordinary_df = pd.DataFrame(ordinary)
    ordinary_df.to_csv(outdir / "pairwise_bottleneck.csv", index=False)
    ordinary_df.to_json(outdir / "pairwise_bottleneck.json", orient="records", indent=2)
    save_heatmaps(ordinary_df, [result["label"] for result in results], outdir)

    nested_rows = Parallel(n_jobs=args.n_jobs, backend=JOBLIB_BACKEND)(
        delayed(quantile_rows)(a, b, dim, args.quantiles, diameter)
        for a, b in pairs
        for dim in range(MAXDIM + 1)
    )
    pd.DataFrame(row for rows in nested_rows for row in rows).to_csv(
        outdir / "pairwise_quantile_bottleneck.csv", index=False
    )

    print(f"Done. Results written to {outdir}", time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)


if __name__ == "__main__":
    main()
