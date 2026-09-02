#!/usr/bin/env python3
"""Measure persistent-homology stability across kNN graph sizes.

The embedding is computed once. For every requested ``k``, XplainTDA then
recomputes persistent homology in that fixed embedding and saves the fitted TDA
object. H0 is additionally compared with the complete Euclidean-graph
reference, whose death times are obtained exactly by single-linkage clustering.

All reported distances are saved both raw and divided by the diameter of the
full embedding. No other normalization is applied.
"""

from __future__ import annotations

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
import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform

import defaults as d
from XplainTDA_Explainer import quantile_bottleneck_distance
from XplainTDA_PH import XplainTDA

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover - optional runtime optimization
    threadpool_limits = None


KS = (5, 15, 50, 75, 100, 250, 500, 750, 1000, 1250)
QUANTILES = (0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
EMBEDDING_KEY = None
LAYER = d.DEFAULT_LAYER
EMBEDDER = d.DEFAULT_EMBEDDER
N_COMPONENTS = d.DEFAULT_N_COMPONENTS
MAXDIM = d.DEFAULT_MAXDIM
NEIGHBOR_BACKEND = d.DEFAULT_NEIGHBOR_BACKEND
PH_BACKEND = d.DEFAULT_PH_BACKEND
DTYPE = d.DEFAULT_DTYPE
EMBEDDER_KWARGS = dict(d.DEFAULT_EMBEDDER_KWARGS or {})
JOBLIB_BACKEND = "loky"
N_JOBS_TDA = 1
N_JOBS_METRICS = min(2, d.DEFAULT_N_JOBS)
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



def finite_diagram(diagram: np.ndarray) -> np.ndarray:
    """Keep finite persistence points with strictly positive lifetime."""
    diagram = np.asarray(diagram, dtype=float).reshape(-1, 2)
    return diagram[
        np.isfinite(diagram).all(axis=1) & (diagram[:, 1] > diagram[:, 0])
    ]


def normalize(distance: float, diameter: float) -> float:
    """Divide by embedding diameter, returning NaN for a degenerate scale."""
    return float(distance / diameter) if np.isfinite(diameter) and diameter > 0 else np.nan


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
        n_neighbors=1,
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


def fit_knn_tda(k: int, embedding_path: str, output_path: str) -> dict:
    """Compute and save one kNN PH result in the shared embedding."""
    started = time.time()
    embedding = np.load(embedding_path, mmap_mode="r")
    with limit_threads():
        tda = XplainTDA(
            n_components=embedding.shape[1],
            n_neighbors=k,
            maxdim=MAXDIM,
            neighbor_backend=NEIGHBOR_BACKEND,
            ph_backend=PH_BACKEND,
            dtype=DTYPE,
        ).fit(embedding=np.asarray(embedding, dtype=DTYPE))
        path = tda.save(
            output_path,
            include_data=False,
            include_embedder=False,
            compress=COMPRESS,
        )
    return {
        "k": k,
        "label": f"k={k}",
        "path": str(path),
        "elapsed_seconds": time.time() - started,
        "dgms": tda.dgms,
    }


def complete_graph_h0(condensed_distances: np.ndarray, n_cells: int) -> list[np.ndarray]:
    """Return exact complete-graph H0 intervals from single linkage.

    The finite H0 death times of a complete Vietoris--Rips graph are the merge
    heights of single-linkage clustering. The final essential class is capped
    at the graph filtration endpoint, equal to the embedding diameter.
    """
    diameter = float(condensed_distances.max()) if len(condensed_distances) else 0.0
    deaths = linkage(condensed_distances, method="single")[:, 2] if n_cells > 1 else np.empty(0)
    return [np.column_stack((np.zeros(n_cells), np.r_[deaths, diameter]))]


def bottleneck_row(
    label_a: str,
    label_b: str,
    dim: int,
    diagrams: dict[str, list[np.ndarray]],
    diameter: float,
) -> dict:
    """Compute one ordinary bottleneck distance and diameter normalization."""
    distance = float(
        gd.bottleneck_distance(
            finite_diagram(diagrams[label_a][dim]),
            finite_diagram(diagrams[label_b][dim]),
        )
    )
    return {
        "label_a": label_a,
        "label_b": label_b,
        "homology_dim": dim,
        "distance": distance,
        "distance_by_diameter": normalize(distance, diameter),
    }


def quantile_rows(
    label_a: str,
    label_b: str,
    dim: int,
    quantiles: Iterable[float],
    diagrams: dict[str, list[np.ndarray]],
    diameter: float,
) -> list[dict]:
    """Compute requested quantile-bottleneck distances for one diagram pair."""
    reference, comparison = diagrams[label_a][dim], diagrams[label_b][dim]
    rows = []
    for quantile in quantiles:
        distance = quantile_bottleneck_distance(reference, comparison, q=quantile)
        rows.append(
            {
                "label_a": label_a,
                "label_b": label_b,
                "homology_dim": dim,
                "quantile": quantile,
                "distance": distance,
                "distance_by_diameter": normalize(distance, diameter),
            }
        )
    return rows


def save_heatmaps(results: pd.DataFrame, labels_by_dim: dict[int, list[str]], outdir: Path) -> None:
    """Save one symmetric raw bottleneck-distance matrix per homology dimension."""
    for dim, labels in labels_by_dim.items():
        matrix = pd.DataFrame(0.0, index=labels, columns=labels)
        for row in results[results["homology_dim"] == dim].itertuples(index=False):
            matrix.loc[row.label_a, row.label_b] = row.distance
            matrix.loc[row.label_b, row.label_a] = row.distance
        matrix.to_csv(outdir / f"bottleneck_heatmap_H{dim}.csv")



def main() -> None:
    """Run the complete kNN stability benchmark."""
    if len(sys.argv) not in (3, 4):
        script = Path(sys.argv[0]).name
        raise SystemExit(
            f"Usage: python {script} <adata_path> <outdir> [embedding_key]"
        )

    ADATA_PATH, OUTDIR = sys.argv[1:3]
    embedding_key = sys.argv[3] if len(sys.argv) == 4 else EMBEDDING_KEY

    outdir = Path(OUTDIR).expanduser().resolve()
    tda_dir = outdir / "tda_objects"
    outdir.mkdir(parents=True, exist_ok=True)
    tda_dir.mkdir(exist_ok=True)

    if any(k < 1 for k in KS):
        raise ValueError("all k values must be positive")
    if any(not 0 < q <= 1 for q in QUANTILES):
        raise ValueError("all quantiles must satisfy 0 < q <= 1")
    ks = tuple(sorted(set(KS)))

    adata_path = Path(ADATA_PATH).expanduser().resolve()
    print(f"ADATA_PATH: {adata_path}", flush=True)
    print(f"OUTDIR: {outdir}", flush=True)
    print(f"EMBEDDING_KEY: {embedding_key}", flush=True)

    if not adata_path.is_file():
        raise FileNotFoundError(adata_path)
    adata = ad.read_h5ad(adata_path)
    if adata.n_obs < 2:
        raise ValueError("at least two cells are required")
    invalid_ks = [k for k in ks if k >= adata.n_obs]
    if invalid_ks:
        raise ValueError(f"k must be smaller than n_cells={adata.n_obs}; invalid: {invalid_ks}")

    print(f"Fitting or loading one embedding for {adata.n_obs} cells", flush=True)
    embedding_tda = fit_embedding(adata, outdir, embedding_key, EMBEDDER.upper())
    embedding_path = outdir / "embedding.npy"
    np.save(embedding_path, embedding_tda.embedding)
    pd.DataFrame(
        {"source_cell_index": np.arange(adata.n_obs), "source_obs_name": adata.obs_names}
    ).to_csv(outdir / "embedding_cells.csv", index=False)

    config = {
        "input_h5ad": str(adata_path),
        "embedding_key": embedding_key,
        "embedder": EMBEDDER.upper(),
        "ks": ks,
        "quantiles": QUANTILES,
        "layer": LAYER,
        "n_components": embedding_tda.n_components,
        "maxdim": MAXDIM,
        "neighbor_backend": NEIGHBOR_BACKEND,
        "ph_backend": PH_BACKEND,
        "normalization": "full_embedding_diameter_only",
        "n_jobs_tda": N_JOBS_TDA,
        "n_jobs_metrics": N_JOBS_METRICS,
    }
    save_json(outdir / "config.json", config)

    print("Computing kNN persistence diagrams", flush=True)
    results = Parallel(n_jobs=bounded_jobs(N_JOBS_TDA, len(ks)), backend=JOBLIB_BACKEND)(
        delayed(fit_knn_tda)(k, str(embedding_path), str(tda_dir / f"tda_knn_k{k}.joblib"))
        for k in ks
    )
    results.sort(key=lambda result: result["k"])
    pd.DataFrame(
        [{key: result[key] for key in ("k", "label", "path", "elapsed_seconds")} for result in results]
    ).to_csv(outdir / "knn_tda_jobs.csv", index=False)

    print("Computing complete-graph H0 reference and embedding diameter", flush=True)
    condensed = pdist(embedding_tda.embedding, metric="euclidean")
    diameter = float(condensed.max()) if len(condensed) else 0.0
    distance_matrix_path = outdir / "full_distance_matrix.npy"
    np.save(distance_matrix_path, squareform(condensed).astype(DTYPE, copy=False))
    full_dgms = complete_graph_h0(condensed, adata.n_obs)
    config.update({
        "embedding_diameter": diameter,
        "full_h0_method": "single_linkage_complete_graph_equivalent",
    })
    save_json(outdir / "config.json", config)
    joblib.dump(
        {
            "mode": "complete_graph_h0",
            "distance_matrix_path": str(distance_matrix_path),
            "embedding_diameter": diameter,
            "dgms": full_dgms,
        },
        tda_dir / "tda_full_distance.joblib",
        compress=COMPRESS,
    )

    diagrams = {"full": full_dgms, **{result["label"]: result["dgms"] for result in results}}
    k_labels = [result["label"] for result in results]

    ordinary_tasks = [("full", label, 0) for label in k_labels]
    ordinary_tasks += [
        (label_a, label_b, dim)
        for label_a, label_b in combinations(k_labels, 2)
        for dim in range(MAXDIM + 1)
    ]
    ordinary = Parallel(n_jobs=N_JOBS_METRICS, backend=JOBLIB_BACKEND)(
        delayed(bottleneck_row)(a, b, dim, diagrams, diameter)
        for a, b, dim in ordinary_tasks
    )
    ordinary_df = pd.DataFrame(ordinary)
    ordinary_df.to_csv(outdir / "bottleneck_pairwise_long.csv", index=False)
    ordinary_df.to_json(outdir / "bottleneck_pairwise_long.json", orient="records", indent=2)
    save_heatmaps(
        ordinary_df,
        {0: ["full", *k_labels], **{dim: k_labels for dim in range(1, MAXDIM + 1)}},
        outdir,
    )

    reference_label = k_labels[-1]
    quantile_tasks = [("full", label, 0) for label in k_labels]
    quantile_tasks += [
        (reference_label, label, dim)
        for dim in range(1, MAXDIM + 1)
        for label in k_labels
    ]
    nested_rows = Parallel(n_jobs=N_JOBS_METRICS, backend=JOBLIB_BACKEND)(
        delayed(quantile_rows)(a, b, dim, QUANTILES, diagrams, diameter)
        for a, b, dim in quantile_tasks
    )
    quantile_df = pd.DataFrame(row for rows in nested_rows for row in rows)
    quantile_df.to_csv(outdir / "quantile_bottleneck_metrics.csv", index=False)

    print(f"Done. Results written to {outdir}", flush=True)


if __name__ == "__main__":
    main()
