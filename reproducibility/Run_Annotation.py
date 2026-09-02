#!/usr/bin/env python3
"""Compute connected-component annotations across informative filtration values.

Only filtration values that change the connected-component partition are kept.
NMI and AMI are then computed in parallel over those retained partitions.

Outputs, written under ``out_path``:
    1. ccids_filtrationvalue.parquet
       Cells x retained filtration values, saved transposed.
    2. combined_results.csv
       Barcode, Celltype, and clusterID at the highest-AMI filtration value.
"""
import warnings
warnings.filterwarnings('ignore')
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import pandas as pd
import scanpy as sc
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score
from tqdm import tqdm

from defaults import (
    DEFAULT_EMBEDDER,
    DEFAULT_N_JOBS,
    DEFAULT_N_NEIGHBORS,
    DEFAULT_NEIGHBOR_BACKEND,
)
from Utils import to_dense_float32
from XplainTDA_PH import XplainTDA, compute_neighbors, neighbors_to_edges


class UnionFind:
    """Disjoint-set structure with path compression and union by rank."""

    def __init__(self, n: int) -> None:
        self.parent = np.arange(n)
        self.rank = np.zeros(n, dtype=np.int32)
        self.n_components = n

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, x: int, y: int) -> bool:
        """Merge two components and return whether the partition changed."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        self.n_components -= 1
        return True

    def labels(self) -> np.ndarray:
        """Return component labels renumbered from 0 to n_components - 1."""
        roots = np.fromiter(
            (self.find(i) for i in range(len(self.parent))),
            dtype=np.int64,
            count=len(self.parent),
        )
        return np.unique(roots, return_inverse=True)[1].astype(np.int32, copy=False)


def compute_changed_partitions(
    edge_list: np.ndarray,
    n_cells: int,
    barcodes,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return connected-component labels only where the partition changes."""
    edges = np.asarray(edge_list)
    if edges.ndim != 2 or edges.shape[1] < 3:
        raise ValueError("edge_list must have columns: source, destination, weight")

    finite_positive = np.isfinite(edges[:, 2]) & (edges[:, 2] > 0)
    edges = edges[finite_positive]
    if len(edges) == 0:
        raise ValueError("No positive finite edge weights were found")

    order = np.argsort(edges[:, 2], kind="mergesort")
    edge_src = edges[order, 0].astype(np.int64, copy=False)
    edge_dst = edges[order, 1].astype(np.int64, copy=False)
    edge_w = edges[order, 2]
    filtration_values, starts = np.unique(edge_w, return_index=True)
    stops = np.r_[starts[1:], len(edge_w)]

    uf = UnionFind(n_cells)
    rows, kept_values, component_counts = [], [], []

    for value, start, stop in tqdm(
        zip(filtration_values, starts, stops),
        total=len(filtration_values),
    ):
        changed = False
        for i in range(start, stop):
            changed |= uf.union(int(edge_src[i]), int(edge_dst[i]))
        if changed:
            rows.append(uf.labels())
            kept_values.append(value)
            component_counts.append(uf.n_components)

    if not rows:
        raise ValueError("No filtration value changed the connected-component partition")

    df = pd.DataFrame(
        np.asarray(rows, dtype=np.int32),
        index=np.asarray(kept_values),
        columns=barcodes,
    )
    df.index.name = "filtration_value"
    return df, np.asarray(component_counts, dtype=np.int32)


def _score_partition(ground_truth: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    return (
        normalized_mutual_info_score(ground_truth, labels),
        adjusted_mutual_info_score(ground_truth, labels),
    )


def compute_nmi_per_filtration(
    df: pd.DataFrame,
    ground_truth,
    n_components: np.ndarray) -> pd.DataFrame:
    """Compute NMI and AMI independently over retained filtration values."""
    gt = np.asarray(ground_truth)
    labels_matrix = df.to_numpy(copy=False)
    if labels_matrix.shape[0] != len(n_components):
        raise ValueError("n_components must match the number of filtration values")

    scores = Parallel(n_jobs=DEFAULT_N_JOBS, backend="loky")(
            delayed(_score_partition)(gt, labels) for labels in labels_matrix)

    scores = np.asarray(scores, dtype=float)
    return pd.DataFrame(
        {
            "filtration_value": df.index.to_numpy(),
            "n_components": n_components,
            "nmi": scores[:, 0],
            "ami": scores[:, 1],
        }
    )


def get_annotation(
    adata_path: Path,
    celltype_col: str,
    out_path: Path,
    embedding: str = DEFAULT_EMBEDDER) -> None:
    """Compute changed partitions, score them, and save annotation outputs."""
    if not adata_path.exists():
        raise FileNotFoundError(f"adata_path does not exist: {adata_path}")

    adata = sc.read_h5ad(adata_path)
    if celltype_col not in adata.obs.columns:
        raise ValueError(
            f"Given annotation not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    obsm_key = f"X_{embedding}"
    if obsm_key in adata.obsm:
        embedding_arr = adata.obsm[obsm_key]
    else:
        print("No embedding found: computing embedding...", flush=True)
        tda_obj = XplainTDA(
            adata,
            embedder=embedding,
            n_neighbors=DEFAULT_N_NEIGHBORS,
        ).fit_embedding()
        embedding_arr = tda_obj.embedding

    embedding_arr = to_dense_float32(embedding_arr)
    neigh_ind, neigh_dist = compute_neighbors(
        embedding_arr,
        n_neighbors=DEFAULT_N_NEIGHBORS,
        backend=DEFAULT_NEIGHBOR_BACKEND,
    )
    edge_list = neighbors_to_edges(neigh_ind, neigh_dist)

    out_path.mkdir(parents=True, exist_ok=True)
    parquet_path = out_path / "ccids_filtrationvalue.parquet"
    merged_path = out_path / "combined_results.csv"
    scores_path = out_path / "nmi_per_filtration.csv"
    best_filtration_path = out_path / "best_filtration.csv"

    t0 = time.time()
    df, n_components = compute_changed_partitions(
        edge_list=edge_list,
        n_cells=adata.n_obs,
        barcodes=adata.obs_names,
    )
    compute_time = time.time() - t0

    tmp_path = parquet_path.with_suffix(".tmp.parquet")
    t1 = time.time()
    df.T.to_parquet(tmp_path)
    tmp_path.replace(parquet_path)
    print(
        f"Saved: {parquet_path} (shape={df.shape}, "
        f"compute={compute_time:.1f}s, save={time.time() - t1:.1f}s)",
        flush=True,
    )

    ground_truth = adata.obs.loc[df.columns, celltype_col].to_numpy()
    missing_mask = pd.isna(ground_truth)
    if missing_mask.any():
        print(
            f"Dropping {missing_mask.sum()} cells with missing "
            f"'{celltype_col}' annotation before scoring "
            f"({missing_mask.sum()}/{len(ground_truth)})",
            flush=True,
        )
        ground_truth = ground_truth[~missing_mask]
        score_df = df.loc[:, ~missing_mask]
        score_n_components = np.fromiter(
            (np.unique(labels).size for labels in score_df.to_numpy(copy=False)),
            dtype=np.int32,
            count=len(score_df),
        )
    else:
        score_df = df
        score_n_components = n_components

    t0 = time.time()
    scores_df = compute_nmi_per_filtration(
        score_df,
        ground_truth,
        n_components=score_n_components)
    print(
        f"Scored topology "
        f"({time.time() - t0:.1f}s)",
        flush=True,
    )

    scores_df.to_csv(scores_path, index=False)

    best_row = scores_df.loc[[scores_df["ami"].idxmax()]]
    best_row.to_csv(best_filtration_path, index=False)
    best_filtration = best_row.iloc[0]["filtration_value"]
    cluster_df = (
        df.loc[best_filtration]
        .rename("clusterID")
        .rename_axis("Barcode")
        .reset_index()
    )
    celltype_df = (
        adata.obs[[celltype_col]]
        .rename(columns={celltype_col: "Celltype"})
        .rename_axis("Barcode")
        .reset_index()
    )
    merged = cluster_df.merge(celltype_df, on="Barcode", how="left")[[
        "Barcode",
        "Celltype",
        "clusterID",
    ]]
    merged.to_csv(merged_path, index=False)

    print(f"best_filtration={best_filtration} -> {merged_path} ({len(merged)} rows)", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adata-path", required=True, type=Path)
    parser.add_argument("--celltype-col", required=True)
    parser.add_argument("--out-path", required=True, type=Path)
    parser.add_argument(
        "--embedding",
        default=DEFAULT_EMBEDDER,
        help=f"Embedding to use (default: {DEFAULT_EMBEDDER})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adata_path = args.adata_path.expanduser().resolve()
    out_path = args.out_path.expanduser().resolve()

    for name, value in (
        ("ADATA_PATH", adata_path),
        ("CELLTYPE_COL", args.celltype_col),
        ("OUT_PATH", out_path),
        ("EMBEDDING", args.embedding)):
        print(f"{name}: {value}", flush=True)

    get_annotation(
        adata_path=adata_path,
        celltype_col=args.celltype_col,
        out_path=out_path,
        embedding=args.embedding)
        
    print(f"Done. Results written under {out_path}", flush=True)


if __name__ == "__main__":
    main()
