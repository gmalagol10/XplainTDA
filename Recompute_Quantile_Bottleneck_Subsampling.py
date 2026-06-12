#!/usr/bin/env python3
"""Recompute quantile bottleneck subsampling stability metrics from saved XplainTDA PH objects.

This script does not recompute embeddings, subsets, kNN graphs, simplex trees, or PH.
It scans completed Run_Stability_Subsampling.py output folders for one dataset/omic,
loads saved subset TDA objects, computes quantile_bottleneck distances for
q = 0.5, 0.75, 1.0, applies the same three normalisations used for kNN stability,
and writes CSV files.

Usage:
    python Recompute_Quantile_Bottleneck_Subsampling.py <dataset> <omic>

Expected output layout, possibly nested under Stability/:
    Datasets/<dataset>/XplainTDA/<omic>/Stability/**/pairwise_distances_long.csv

A matched output folder must also contain:
    subset_tda_jobs.csv
    full_embedding.npy
    subsets/subset_*/tda_subset_*.joblib

Outputs:
    <OUTDIR>/quantile_bottleneck_subsampling_normalised.csv
    quantile_bottleneck_subsampling_normalised_all.csv
"""

from __future__ import annotations

import glob
import re
import sys
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from XplainTDA import quantile_bottleneck


QS = (0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
SCALES = ("S_diameter", "S_filtration_range", "S_longest_lifespan_dim")
DEFAULT_PER_OUTDIR = "quantile_bottleneck_subsampling_normalised.csv"
DEFAULT_COMBINED_OUT = "quantile_bottleneck_subsampling_normalised_all.csv"
DIAMETER_BLOCK_SIZE = 2048


def parse_path(f: str | Path) -> tuple[str, str]:
    """Extract dataset and omic from Datasets/<dataset>/XplainTDA/<omic>/..."""
    parts = Path(f).parts
    i = parts.index("Datasets")
    return parts[i + 1], parts[i + 3]


def subset_value(label: str) -> float:
    """Convert labels like 'subset_3' to 3 for sorting."""
    m = re.search(r"subset_(\d+)", str(label))
    return float(m.group(1)) if m else np.nan


def q_token(q: float) -> str:
    """Safe token for column names."""
    return "q" + f"{q:g}".replace(".", "_")


def safe_divide(x: float, s: float) -> float:
    """Avoid division by zero or missing denominators."""
    return float(x / s) if np.isfinite(s) and s > 0 else np.nan


def get_field(state, name: str, default=None):
    """Read from either a saved dict state or a loaded XplainTDA object."""
    return state.get(name, default) if isinstance(state, dict) else getattr(state, name, default)


def is_subsampling_marker(marker_csv: str | Path) -> bool:
    """Return True for completed Run_Stability_Subsampling.py output folders."""
    outdir = Path(marker_csv).parent
    return (
        (outdir / "subset_tda_jobs.csv").exists()
        and (outdir / "full_embedding.npy").exists()
        and (outdir / "subsets").exists()
    )


def load_subset_states(outdir: Path) -> dict[str, object]:
    """Load subset TDA states using subset_tda_jobs.csv, with a glob fallback."""
    states = {}
    jobs_path = outdir / "subset_tda_jobs.csv"

    if jobs_path.exists():
        jobs = pd.read_csv(jobs_path)
        for row in jobs.itertuples(index=False):
            label = str(getattr(row, "label", f"subset_{int(row.subset_id)}"))
            path = Path(str(row.path))
            if not path.exists():
                fallback = outdir / "subsets" / label / f"tda_{label}.joblib"
                path = fallback if fallback.exists() else path
            if path.exists():
                states[label] = joblib.load(path)

    if not states:
        for path in sorted((outdir / "subsets").glob("subset_*/tda_subset_*.joblib")):
            m = re.search(r"tda_subset_(\d+)\.joblib$", path.name)
            if m:
                states[f"subset_{int(m.group(1))}"] = joblib.load(path)

    return dict(sorted(states.items(), key=lambda kv: subset_value(kv[0])))


def finite_persistences(dgms, dim: int) -> np.ndarray:
    """Finite positive lifetimes death - birth for one homology dimension."""
    if dgms is None or int(dim) >= len(dgms):
        return np.array([], dtype=float)

    dgm = np.asarray(dgms[int(dim)], dtype=float)
    if dgm.size == 0:
        return np.array([], dtype=float)

    dgm = dgm[np.isfinite(dgm).all(axis=1)]
    pers = dgm[:, 1] - dgm[:, 0]
    return pers[pers > 0]


def filtration_range(state) -> float:
    """Effective filtration range of one saved subset PH object."""
    edge_list = get_field(state, "edge_list")
    if edge_list is not None and np.asarray(edge_list).size:
        return float(np.nanmax(np.asarray(edge_list)[:, 2]))

    dgms = get_field(state, "dgms")
    deaths = [np.asarray(dgm)[:, 1] for dgm in dgms or [] if np.asarray(dgm).size]
    deaths = np.concatenate(deaths) if deaths else np.array([], dtype=float)
    deaths = deaths[np.isfinite(deaths)]
    return float(np.nanmax(deaths)) if deaths.size else np.nan


def diameter_from_embedding(outdir: Path, block_size: int = DIAMETER_BLOCK_SIZE) -> float:
    """Exact Euclidean diameter of the saved full embedding, computed in blocks."""
    emb_path = outdir / "full_embedding.npy"
    if not emb_path.exists():
        raise FileNotFoundError(f"Missing {emb_path}")

    X = np.load(emb_path, mmap_mode="r")
    n = int(X.shape[0])
    if n < 2:
        return np.nan

    max_d2 = 0.0
    for i in range(0, n, block_size):
        A = np.asarray(X[i:i + block_size], dtype=float)
        A2 = np.sum(A * A, axis=1)[:, None]
        for j in range(i, n, block_size):
            B = np.asarray(X[j:j + block_size], dtype=float)
            d2 = A2 + np.sum(B * B, axis=1)[None, :] - 2.0 * (A @ B.T)
            np.maximum(d2, 0.0, out=d2)
            m = float(np.nanmax(d2)) if d2.size else np.nan
            if np.isfinite(m) and m > max_d2:
                max_d2 = m

    return float(np.sqrt(max_d2))


def longest_lifespan_by_dim(states: dict[str, object]) -> dict[int, float]:
    """Longest finite persistence across all subset diagrams, per dimension."""
    max_len = max((len(get_field(s, "dgms") or []) for s in states.values()), default=0)
    out = {}
    for dim in range(max_len):
        pieces = []
        for state in states.values():
            p = finite_persistences(get_field(state, "dgms"), dim)
            if p.size:
                pieces.append(p)
        out[int(dim)] = float(np.nanmax(np.concatenate(pieces))) if pieces else np.nan
    return out


def diagram(states: dict[str, object], label: str, dim: int) -> np.ndarray:
    dgms = get_field(states[label], "dgms")
    if dgms is None or dim >= len(dgms):
        raise IndexError(f"Missing H{dim} diagram for {label}")
    return np.asarray(dgms[dim], dtype=float)


def comparison_rows(states: dict[str, object]) -> list[dict]:
    """All subset-vs-subset pairs per available homology dimension."""
    labels = list(states)
    if len(labels) < 2:
        raise ValueError("Need at least two subset TDA objects")

    max_len = max(len(get_field(s, "dgms") or []) for s in states.values())
    rows = []
    for a, b in combinations(labels, 2):
        for dim in range(max_len):
            if dim < len(get_field(states[a], "dgms") or []) and dim < len(get_field(states[b], "dgms") or []):
                rows.append({"homology_dim": dim, "subset_a": a, "subset_b": b})
    return rows


def compute_one(row: dict, states: dict[str, object], scales: dict) -> dict:
    """Compute all q/S columns for one subset pair and homology dimension."""
    dim = int(row["homology_dim"])
    a, b = row["subset_a"], row["subset_b"]
    D, E = diagram(states, a, dim), diagram(states, b, dim)

    out = dict(row)
    pair_scales = {
        "S_diameter": scales["S_diameter"],
        "S_filtration_range": np.nanmax([
            scales["S_filtration_range_by_label"].get(a, np.nan),
            scales["S_filtration_range_by_label"].get(b, np.nan),
        ]),
        "S_longest_lifespan_dim": scales["S_longest_lifespan_by_dim"].get(dim, np.nan),
    }

    for q in QS:
        dist = quantile_bottleneck(D, E, q=q)
        for s_name in SCALES:
            out[f"bottleneck_{q_token(q)}_{s_name}"] = safe_divide(dist, pair_scales[s_name])

    return out


def process_outdir(marker_csv: str | Path, per_outdir_name: str = DEFAULT_PER_OUTDIR) -> pd.DataFrame:
    """Process one subsampling output directory and write its per-folder CSV."""
    marker_csv = Path(marker_csv)
    outdir = marker_csv.parent
    dataset, omic = parse_path(marker_csv)

    print(f"Processing {outdir}", flush=True)
    states = load_subset_states(outdir)
    if len(states) < 2:
        raise FileNotFoundError(f"Fewer than two saved subset states found under {outdir / 'subsets'}")

    scales = {
        "S_diameter": diameter_from_embedding(outdir),
        "S_filtration_range_by_label": {label: filtration_range(state) for label, state in states.items()},
        "S_longest_lifespan_by_dim": longest_lifespan_by_dim(states),
    }

    rows = comparison_rows(states)
    data = [compute_one(r, states, scales) for r in rows]
    df = pd.DataFrame(data)

    df.insert(0, "omic", omic)
    df.insert(0, "dataset", dataset)
    df.insert(2, "source_outdir", str(outdir))

    metric_cols = [f"bottleneck_{q_token(q)}_{s}" for q in QS for s in SCALES]
    cols = ["dataset", "omic", "source_outdir", "homology_dim", "subset_a", "subset_b"] + metric_cols
    df = df[cols]

    out_path = outdir / per_outdir_name
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}", flush=True)
    return df


def main():
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python Recompute_Quantile_Bottleneck_Subsampling.py <dataset> <omic>"
        )

    dataset = str(sys.argv[1])
    omic = str(sys.argv[2])

    pattern = f"Datasets/{dataset}/XplainTDA/{omic}/Stability/**/pairwise_distances_long.csv"
    files = sorted({f for f in glob.glob(pattern, recursive=True) if is_subsampling_marker(f)})

    if not files:
        raise FileNotFoundError(
            "No completed Run_Stability_Subsampling outputs found with pattern: "
            f"{pattern}"
        )

    dfs = [process_outdir(f, per_outdir_name=DEFAULT_PER_OUTDIR) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(DEFAULT_COMBINED_OUT, index=False)

    print(f"Wrote combined CSV: {DEFAULT_COMBINED_OUT}", flush=True)


if __name__ == "__main__":
    main()
