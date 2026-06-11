"""
Subsampling stability of XplainTDA persistence diagrams with fixed embedding.

Design:
- Fit the embedder once on the full AnnData.
- Create N_SUBSETS independent stratified subsamples from the full AnnData.
- Each subset is sampled without replacement within each stratum.
- Cells are unique within each subset.
- Cells may overlap across different subsets.
- For each subset, take the corresponding rows of the fixed embedding.
- Compute kNN persistent homology with XplainTDA.fit(embedding=...).
- Compute all pairwise bottleneck and Wasserstein distances per homology dimension.
- Save the fixed embedding, one cell-membership CSV per subset, one TDA object per subset,
  one long-form distance CSV/JSON, and heatmap CSVs.

Usage:
    python Run_Stability_Subsampling.py input.h5ad output_dir
    python Run_Stability_Subsampling.py input.h5ad output_dir CellType 5 0.25
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from itertools import combinations
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import anndata as ad
import gudhi as gd
import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

try:
    from gudhi.wasserstein import wasserstein_distance
except Exception as e:
    wasserstein_distance = None
    WASSERSTEIN_IMPORT_ERROR = e
else:
    WASSERSTEIN_IMPORT_ERROR = None

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
    DEFAULT_WASSERSTEIN_INTERNAL_P,
    DEFAULT_WASSERSTEIN_ORDER,
)

from XplainTDA import XplainTDA


# ---------------------------------------------------------------------
# User-editable configuration
# ---------------------------------------------------------------------
N_SUBSETS = 5
SUBSET_FRAC = 0.25

LAYER = DEFAULT_LAYER
EMBEDDER = DEFAULT_EMBEDDER
N_COMPONENTS = DEFAULT_N_COMPONENTS
N_NEIGHBORS = DEFAULT_N_NEIGHBORS
MAXDIM = DEFAULT_MAXDIM
NEIGHBOR_BACKEND = DEFAULT_NEIGHBOR_BACKEND
PH_BACKEND = "ripser"
DTYPE = DEFAULT_DTYPE
EMBEDDER_KWARGS = DEFAULT_EMBEDDER_KWARGS or {}

WASSERSTEIN_ORDER = DEFAULT_WASSERSTEIN_ORDER
WASSERSTEIN_INTERNAL_P = DEFAULT_WASSERSTEIN_INTERNAL_P

N_JOBS_SUBSETS = DEFAULT_N_JOBS
JOBLIB_BACKEND_SUBSETS = DEFAULT_BACKEND

N_JOBS_DISTANCES = DEFAULT_N_JOBS
JOBLIB_BACKEND_DISTANCES = DEFAULT_BACKEND

INNER_THREAD_LIMIT = 1

SAVE_ADATA_IN_TDA_OBJECTS = False
SAVE_EMBEDDER_IN_TDA_OBJECTS = False
SAVE_FULL_EMBEDDER = True
JOBLIB_COMPRESS = 3
# ---------------------------------------------------------------------


@contextmanager
def pushd(path: Path):
    old = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


@contextmanager
def maybe_limit_threads(n_threads: int):
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


def fit_full_embedding(adata: ad.AnnData, outdir: Path):
    """
    Fit the embedder once on the full AnnData and return the embedding.

    This intentionally avoids XplainTDA.fit(), because fit() would also compute
    PH on the full dataset. Here we only need the full-data embedding.
    """
    workdir = outdir / "full_embedding_workdir"
    workdir.mkdir(parents=True, exist_ok=True)

    with pushd(workdir):
        with maybe_limit_threads(INNER_THREAD_LIMIT):
            model = XplainTDA(
                adata=adata,
                layer=LAYER,
                embedder=EMBEDDER,
                n_components=N_COMPONENTS,
                n_neighbors=N_NEIGHBORS,
                maxdim=MAXDIM,
                neighbor_backend=NEIGHBOR_BACKEND,
                dtype=DTYPE,
                embedder_kwargs=dict(EMBEDDER_KWARGS),
            )

            model.X = adata.X if LAYER == "X" else adata.layers[LAYER]
            model.n_cells, model.n_genes = model.X.shape

            if model.n_components is None:
                model.n_components = int(np.round(model.n_genes ** (1 / 3)))

            print(
                f"[{now()}] fitting {EMBEDDER} once on full data "
                f"({model.n_cells} cells, {model.n_genes} features, "
                f"n_components={model.n_components})",
                flush=True,
            )

            model._fit_embedder()
            embedding = np.asarray(model.embedder.transform(model.X), dtype=DTYPE)

            if SAVE_FULL_EMBEDDER:
                model.save_embedder(outdir / "full_embedder.joblib", compress=JOBLIB_COMPRESS)

    return embedding, int(model.n_components)


def subset_indices(
    adata: ad.AnnData,
    subset_frac: float,
    stratify_by: str = "CellType",
    seed: int | None = None,
) -> np.ndarray:
    """
    Draw indices for one independent stratified subset.

    Sampling is without replacement within each stratum, so every subset has
    unique cells. Different subsets may overlap because each subset gets an
    independent RNG seed.
    """
    if subset_frac <= 0:
        raise ValueError("subset_frac must be > 0.")
    if subset_frac > 1:
        raise ValueError("subset_frac must be <= 1 when cells must be unique within each subset.")
    if stratify_by not in adata.obs:
        raise KeyError(f"{stratify_by!r} not found in adata.obs.")

    rng = np.random.default_rng(seed)
    sampled = []
    groups = adata.obs[stratify_by].astype("category")

    for _, group_idx in adata.obs.groupby(groups, observed=True).indices.items():
        group_idx = np.asarray(group_idx)
        n_sample = min(max(1, int(round(subset_frac * len(group_idx)))), len(group_idx))
        sampled.append(rng.choice(group_idx, size=n_sample, replace=False))

    idx = np.concatenate(sampled)
    rng.shuffle(idx)
    return idx


def fit_subset_tda(
    subset_id: int,
    embedding_path: str,
    idx: np.ndarray,
    subset_dir: Path,
    n_components: int,
) -> dict:
    """
    Compute PH for one subset using rows of the fixed full-data embedding.
    """
    t0 = time.time()
    subset_dir = subset_dir.resolve()
    subset_dir.mkdir(parents=True, exist_ok=True)

    tda_path = subset_dir / f"tda_subset_{subset_id}.joblib"

    with maybe_limit_threads(INNER_THREAD_LIMIT):
        full_embedding = np.load(embedding_path, mmap_mode="r")
        sub_embedding = np.asarray(full_embedding[idx], dtype=DTYPE)
        n_neighbors = min(int(N_NEIGHBORS), sub_embedding.shape[0])

        print(
            f"[{now()}] subset_{subset_id}: computing PH on fixed embedding "
            f"({sub_embedding.shape[0]} cells)",
            flush=True,
        )

        model = XplainTDA(
            adata=None,
            layer=LAYER,
            embedder=EMBEDDER,
            n_components=n_components,
            n_neighbors=n_neighbors,
            maxdim=MAXDIM,
            neighbor_backend=NEIGHBOR_BACKEND,
            ph_backend=PH_BACKEND,
            dtype=DTYPE,
            embedder_kwargs=dict(EMBEDDER_KWARGS),
        ).fit(embedding=sub_embedding)

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
        "workdir": str(subset_dir),
        "n_cells": int(sub_embedding.shape[0]),
        "n_neighbors": int(model.n_neighbors),
        "n_components": int(n_components),
        "elapsed_seconds": time.time() - t0,
        "dgms": model.dgms,
    }


def finite_diagram(dgm: np.ndarray) -> np.ndarray:
    dgm = np.asarray(dgm, dtype=float)
    if dgm.size == 0:
        return dgm.reshape(0, 2)
    return dgm[np.isfinite(dgm).all(axis=1)]


def one_pair_distance(a: dict, b: dict, dim: int) -> list[dict]:
    da = finite_diagram(a["dgms"][dim])
    db = finite_diagram(b["dgms"][dim])

    rows = [{
        "label_a": a["label"],
        "label_b": b["label"],
        "homology_dim": dim,
        "metric": "bottleneck",
        "distance": float(gd.bottleneck_distance(da, db)),
    }]

    if wasserstein_distance is None:
        raise ImportError(
            f"Could not import gudhi.wasserstein.wasserstein_distance: "
            f"{WASSERSTEIN_IMPORT_ERROR}"
        )

    rows.append({
        "label_a": a["label"],
        "label_b": b["label"],
        "homology_dim": dim,
        "metric": "wasserstein",
        "distance": float(wasserstein_distance(
            da,
            db,
            order=WASSERSTEIN_ORDER,
            internal_p=WASSERSTEIN_INTERNAL_P,
            matching=False,
        )),
    })

    return rows


def save_heatmaps(pairwise_df: pd.DataFrame, labels: list[str], outdir: Path) -> None:
    for metric in sorted(pairwise_df["metric"].unique()):
        for dim in sorted(pairwise_df["homology_dim"].unique()):
            mat = pd.DataFrame(0.0, index=labels, columns=labels)
            sub = pairwise_df[
                (pairwise_df["metric"] == metric)
                & (pairwise_df["homology_dim"] == dim)
            ]

            for row in sub.itertuples(index=False):
                mat.loc[row.label_a, row.label_b] = row.distance
                mat.loc[row.label_b, row.label_a] = row.distance

            mat.to_csv(outdir / f"{metric}_heatmap_H{dim}.csv")


def main():
    if len(sys.argv) < 3:
        raise SystemExit(
            "Usage: python Run_Stability_Subsampling.py input.h5ad output_dir "
            "[stratify_by] [n_subsets] [subset_frac]"
        )

    print(f"[{now()}] Starting script", flush=True)

    adata_path = str(Path(sys.argv[1]).resolve())
    outdir = Path(sys.argv[2]).resolve()

    stratify_by = sys.argv[3] if len(sys.argv) > 3 else "CellType"
    n_subsets = int(sys.argv[4]) if len(sys.argv) > 4 else N_SUBSETS
    subset_frac = float(sys.argv[5]) if len(sys.argv) > 5 else SUBSET_FRAC

    outdir.mkdir(parents=True, exist_ok=True)
    subsets_dir = outdir / "subsets"
    subsets_dir.mkdir(parents=True, exist_ok=True)

    print("Script", sys.argv[0], flush=True)
    print("ADATA_PATH", adata_path, flush=True)
    print("OUTDIR", outdir, flush=True)
    print("stratify_by", stratify_by, flush=True)
    print("n_subsets", n_subsets, flush=True)
    print("subset_frac", subset_frac, flush=True)

    adata = ad.read_h5ad(adata_path)
    print(adata, flush=True)

    embedding, n_components = fit_full_embedding(adata, outdir)
    embedding_path = outdir / "full_embedding.npy"
    np.save(embedding_path, embedding)

    pd.DataFrame({
        "source_cell_index": np.arange(adata.n_obs),
        "source_obs_name": adata.obs_names.to_numpy(),
    }).to_csv(outdir / "full_embedding_cells.csv", index=False)

    seed_seq = np.random.SeedSequence()
    subset_seeds = [int(s.generate_state(1)[0]) for s in seed_seq.spawn(n_subsets)]

    subset_meta = []
    subset_jobs = []

    for i in range(n_subsets):
        subset_dir = subsets_dir / f"subset_{i}"
        subset_dir.mkdir(parents=True, exist_ok=True)

        idx = subset_indices(
            adata=adata,
            subset_frac=subset_frac,
            stratify_by=stratify_by,
            seed=subset_seeds[i],
        )

        cells_path = subset_dir / "cells.csv"
        pd.DataFrame({
            "subset_id": i,
            "subset_seed": subset_seeds[i],
            "subset_row": np.arange(len(idx)),
            "source_cell_index": idx,
            "source_obs_name": adata.obs_names[idx].to_numpy(),
        }).to_csv(cells_path, index=False)

        subset_meta.append({
            "subset_id": i,
            "label": f"subset_{i}",
            "cells_path": str(cells_path),
            "subset_seed": subset_seeds[i],
        })

        subset_jobs.append(
            delayed(fit_subset_tda)(
                subset_id=i,
                embedding_path=str(embedding_path),
                idx=idx,
                subset_dir=subset_dir,
                n_components=n_components,
            )
        )

    config = {
        "adata_path": adata_path,
        "n_subsets": n_subsets,
        "subset_frac": subset_frac,
        "sampling": (
            "fixed full-data embedding; independent stratified subsamples; "
            "unique cells within each subset; cells may overlap across subsets"
        ),
        "layer": LAYER,
        "embedder": EMBEDDER,
        "n_components": n_components,
        "n_neighbors": N_NEIGHBORS,
        "maxdim": MAXDIM,
        "neighbor_backend": NEIGHBOR_BACKEND,
        "dtype": str(DTYPE),
        "stratify_by": stratify_by,
        "embedder_kwargs": EMBEDDER_KWARGS,
        "wasserstein_order": WASSERSTEIN_ORDER,
        "wasserstein_internal_p": str(WASSERSTEIN_INTERNAL_P),
        "n_jobs_subsets": min(N_JOBS_SUBSETS, n_subsets),
        "n_jobs_distances": N_JOBS_DISTANCES,
        "joblib_backend_subsets": JOBLIB_BACKEND_SUBSETS,
        "joblib_backend_distances": JOBLIB_BACKEND_DISTANCES,
        "inner_thread_limit": INNER_THREAD_LIMIT,
        "save_adata_in_tda_objects": SAVE_ADATA_IN_TDA_OBJECTS,
        "save_embedder_in_tda_objects": SAVE_EMBEDDER_IN_TDA_OBJECTS,
    }
    dump_json(outdir / "config.json", config)

    print(f"[{now()}] Computing PH for {n_subsets} subsets from fixed embedding", flush=True)

    subset_results = Parallel(
        n_jobs=min(N_JOBS_SUBSETS, n_subsets),
        backend=JOBLIB_BACKEND_SUBSETS,
    )(subset_jobs)

    subset_results = sorted(subset_results, key=lambda r: r["subset_id"])
    meta_by_id = {m["subset_id"]: m for m in subset_meta}

    for r in subset_results:
        r["cells_path"] = meta_by_id[r["subset_id"]]["cells_path"]
        r["subset_seed"] = meta_by_id[r["subset_id"]]["subset_seed"]

    summary_cols = [
        "subset_id",
        "label",
        "path",
        "cells_path",
        "workdir",
        "subset_seed",
        "n_cells",
        "n_neighbors",
        "n_components",
        "elapsed_seconds",
    ]

    pd.DataFrame(
        [{c: r[c] for c in summary_cols} for r in subset_results]
    ).to_csv(outdir / "subset_tda_jobs.csv", index=False)

    print(f"[{now()}] Computing pairwise bottleneck and Wasserstein distances", flush=True)

    tasks = [
        delayed(one_pair_distance)(a, b, dim)
        for a, b in combinations(subset_results, 2)
        for dim in range(MAXDIM + 1)
    ]

    nested_rows = Parallel(
        n_jobs=N_JOBS_DISTANCES,
        backend=JOBLIB_BACKEND_DISTANCES,
    )(tasks)

    rows = [row for group in nested_rows for row in group]
    pairwise_df = pd.DataFrame(rows)

    pairwise_df.to_csv(outdir / "pairwise_distances_long.csv", index=False)
    pairwise_df.to_json(outdir / "pairwise_distances_long.json", orient="records", indent=2)

    save_heatmaps(pairwise_df, [r["label"] for r in subset_results], outdir)

    print(f"[{now()}] Done. Results written to: {outdir}", flush=True)


if __name__ == "__main__":
    main()
