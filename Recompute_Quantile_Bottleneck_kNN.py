#!/usr/bin/env python3
"""Recompute quantile bottleneck stability metrics from saved XplainTDA PH objects.

This script does not recompute embeddings, kNN graphs, simplex trees, or PH.
It scans completed Run_Stability_kNN.py output folders, loads the saved
``tda_objects/*.joblib`` states, computes quantile_bottleneck distances for
q = 0.5, 0.75, 1.0, applies three normalisations, and writes CSV files.

Default scan pattern mirrors the aggregation code:
    Datasets/*/XplainTDA/*/Stability/kNN/bottleneck_pairwise_long.csv

Outputs:
    <OUTDIR>/quantile_bottleneck_normalised.csv
    quantile_bottleneck_normalised_all.csv  (or --combined-out path)
"""

from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from XplainTDA import quantile_bottleneck
from defaults import DEFAULT_N_JOBS

    
QS = (0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
SCALES = ("S_diameter", "S_filtration_range", "S_longest_lifespan_dim")
DEFAULT_PER_OUTDIR = "quantile_bottleneck_normalised.csv"
DEFAULT_COMBINED_OUT = "quantile_bottleneck_normalised_all.csv"


def parse_path(f: str | Path) -> tuple[str, str]:
    """Extract dataset and omic from Datasets/<dataset>/XplainTDA/<omic>/..."""
    parts = Path(f).parts
    i = parts.index("Datasets")
    return parts[i + 1], parts[i + 3]


def k_value(label: str) -> float:
    """Convert labels like 'k=50' to 50; use infinity for 'full'."""
    if label == "full":
        return np.inf
    m = re.search(r"k=(\d+)", str(label))
    return float(m.group(1)) if m else np.nan


def q_token(q: float) -> str:
    """Safe token for column names."""
    return ("q" + f"{q:g}".replace(".", "_"))


def safe_divide(x: float, s: float) -> float:
    """Avoid division by zero or missing denominators."""
    return float(x / s) if np.isfinite(s) and s > 0 else np.nan


def get_field(state, name: str, default=None):
    """Read from either a saved dict state or a loaded object."""
    if isinstance(state, dict):
        return state.get(name, default)
    return getattr(state, name, default)


def load_state(path: Path):
    return joblib.load(path)


def load_tda_states(outdir: Path) -> dict[str, object]:
    """Load saved PH states from <outdir>/tda_objects."""
    tda_dir = outdir / "tda_objects"
    states = {}

    full_path = tda_dir / "tda_full_distance.joblib"
    if full_path.exists():
        states["full"] = load_state(full_path)

    for path in sorted(tda_dir.glob("tda_knn_k*.joblib")):
        m = re.search(r"tda_knn_k(\d+)\.joblib$", path.name)
        if m:
            states[f"k={int(m.group(1))}"] = load_state(path)

    return states


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
    """Effective filtration range of one saved object."""
    edge_list = get_field(state, "edge_list")
    if edge_list is not None and np.asarray(edge_list).size:
        return float(np.nanmax(np.asarray(edge_list)[:, 2]))

    dgms = get_field(state, "dgms")
    deaths = [np.asarray(dgm)[:, 1] for dgm in dgms or [] if np.asarray(dgm).size]
    deaths = np.concatenate(deaths) if deaths else np.array([], dtype=float)
    deaths = deaths[np.isfinite(deaths)]
    return float(np.nanmax(deaths)) if deaths.size else np.nan


def diameter_scale(outdir: Path) -> float:
    """Diameter of the full embedded space from full_distance_matrix.npy."""
    D_path = outdir / "full_distance_matrix.npy"
    if not D_path.exists():
        raise FileNotFoundError(f"Missing {D_path}")

    D = np.load(D_path, mmap_mode="r")
    Dpos = D[np.isfinite(D) & (D > 0)]
    if Dpos.size == 0:
        return np.nan
    return float(np.nanmax(Dpos))


def longest_lifespan_by_dim(states: dict[str, object], dims=(0, 1, 2)) -> dict[int, float]:
    """Longest finite persistence across all saved diagrams, per dimension."""
    out = {}
    for dim in dims:
        pieces = []
        for state in states.values():
            p = finite_persistences(get_field(state, "dgms"), dim)
            if p.size:
                pieces.append(p)
        out[int(dim)] = float(np.nanmax(np.concatenate(pieces))) if pieces else np.nan
    return out


def available_knn_labels(states: dict[str, object]) -> list[str]:
    labels = [x for x in states if re.fullmatch(r"k=\d+", x)]
    return sorted(labels, key=k_value)


def diagram(states: dict[str, object], label: str, dim: int) -> np.ndarray:
    dgms = get_field(states[label], "dgms")
    if dgms is None or dim >= len(dgms):
        raise IndexError(f"Missing H{dim} diagram for {label}")
    return np.asarray(dgms[dim], dtype=float)


def comparison_rows(states: dict[str, object]) -> list[dict]:
    """Build requested comparisons: H0 full-vs-k; H1/H2 k_max-vs-k."""
    k_labels = available_knn_labels(states)
    if not k_labels:
        raise FileNotFoundError("No tda_knn_k*.joblib states found")

    rows = []
    if "full" in states:
        rows.extend({"homology_dim": 0, "k1": "full", "k2": k} for k in k_labels)
    else:
        print("Warning: full state missing; skipping H0 full-vs-k comparisons", flush=True)

    k_max = k_labels[-1]
    for dim in (1, 2):
        try:
            diagram(states, k_max, dim)
        except IndexError:
            print(f"Warning: {k_max} has no H{dim}; skipping H{dim}", flush=True)
            continue
        rows.extend({"homology_dim": dim, "k1": k_max, "k2": k} for k in k_labels)

    return rows


def compute_one(row: dict, states: dict[str, object], scales: dict) -> dict:
    """Compute all q/S columns for one pair and homology dimension."""
    dim, k1, k2 = int(row["homology_dim"]), row["k1"], row["k2"]
    D, E = diagram(states, k1, dim), diagram(states, k2, dim)

    out = dict(row)
    pair_scales = {
        "S_diameter": scales["S_diameter"],
        "S_filtration_range": np.nanmax([
            scales["S_filtration_range_by_label"].get(k1, np.nan),
            scales["S_filtration_range_by_label"].get(k2, np.nan),
        ]),
        "S_longest_lifespan_dim": scales["S_longest_lifespan_by_dim"].get(dim, np.nan),
    }

    for q in QS:
        dist = quantile_bottleneck(D, E, q=q)
        for s_name in SCALES:
            out[f"bottleneck_{q_token(q)}_{s_name}"] = safe_divide(dist, pair_scales[s_name])

    return out


def process_outdir(marker_csv: str | Path, n_jobs: int, per_outdir_name: str) -> pd.DataFrame:
    """Process one Stability/kNN output directory and write its per-folder CSV."""
    marker_csv = Path(marker_csv)
    outdir = marker_csv.parent
    dataset, omic = parse_path(marker_csv)

    print(f"Processing {outdir}", flush=True)
    states = load_tda_states(outdir)
    if not states:
        raise FileNotFoundError(f"No saved states found under {outdir / 'tda_objects'}")

    scales = {
        "S_diameter": diameter_scale(outdir),
        "S_filtration_range_by_label": {label: filtration_range(state) for label, state in states.items()},
        "S_longest_lifespan_by_dim": longest_lifespan_by_dim(states, dims=(0, 1, 2)),
    }

    rows = comparison_rows(states)
    data = Parallel(n_jobs=n_jobs)(delayed(compute_one)(r, states, scales) for r in rows)
    df = pd.DataFrame(data)

    df.insert(0, "omic", omic)
    df.insert(0, "dataset", dataset)
    df.insert(2, "source_outdir", str(outdir))

    metric_cols = [f"bottleneck_{q_token(q)}_{s}" for q in QS for s in SCALES]
    cols = ["dataset", "omic", "source_outdir", "homology_dim", "k1", "k2"] + metric_cols
    df = df[cols]

    out_path = outdir / per_outdir_name
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}", flush=True)
    return df


def main():
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python Recompute_Quantile_Bottleneck_kNN.py <dataset> <omic>"
        )

    dataset = str(sys.argv[1])
    omic = str(sys.argv[2])

    pattern = (
        f"Datasets/{dataset}/XplainTDA/{omic}/"
        "Stability/kNN/bottleneck_pairwise_long.csv"
    )

    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No completed Run_Stability_kNN outputs found with pattern: {pattern}"
        )

    dfs = [
        process_outdir(
            f,
            n_jobs=4,
            per_outdir_name=DEFAULT_PER_OUTDIR,
        )
        for f in files
    ]

    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(DEFAULT_COMBINED_OUT, index=False)

    print(f"Wrote combined CSV: {DEFAULT_COMBINED_OUT}", flush=True)


if __name__ == "__main__":
    main()
