#!/usr/bin/env python3
"""Compare stratified XplainTDA features with DA using the SEAGALL Fig. 5 metrics.

Runs, for a user-selected list of top-k thresholds:
- XplainTDA vs DA similarity
- expression/openness non-overlap
- coverage non-overlap
- signal-to-noise ratio (STN)
- XplainTDA specificity

XplainTDA metrics are computed independently for every homology dimension.
Stability and rank-importance analyses are intentionally excluded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


VERSION = "v4"


LABELS = {
    "PBMC": "Raw_CellType",
    "HumanBrain": "Raw_celltype",
    "BMMCMultiOme": "Coarse_CellType",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("."),
                   help="XplainTDA project root containing Datasets/ (default: current directory)")
    p.add_argument("--datasets", nargs="+", choices=list(LABELS), default=list(LABELS),
                   help="Datasets to analyze")
    p.add_argument("--omics", nargs="+", choices=["GEX", "Peak"], default=["GEX","Peak"],
                   help="Modalities to analyze")
    p.add_argument("-k", "--k", "--ks", dest="ks", nargs="+", type=int, default=[10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000],
                   help="Top-k thresholds used by every metric, e.g. --ks 10 25 50 100")
    p.add_argument("--layer", help='Override matrix layer; otherwise use each explanation config.json, then "X"')
    return p.parse_args()


def project_paths(root: Path, dataset: str, omic: str) -> tuple[Path, Path]:
    adata_path = root / "Datasets" / dataset / "CM" / f"{dataset}_{omic}_Def.h5ad"
    outdir = root / "Datasets" / dataset / "XplainTDA" / omic / "Explanations" / LABELS[dataset]
    return adata_path, outdir


def _matrix(adata: ad.AnnData, layer: str):
    if layer == "X":
        return adata.X
    if layer not in adata.layers:
        raise KeyError(f"Layer {layer!r} not found in adata.layers")
    return adata.layers[layer]


def _top(values: np.ndarray, names: pd.Index, k: int) -> list[str]:
    order = np.argsort(-np.asarray(values).ravel(), kind="stable")[:k]
    return names[order].astype(str).tolist()


def _stn(X) -> float:
    """numpy-style population mean/std without densifying sparse matrices."""
    n = X.shape[0] * X.shape[1]
    if n == 0:
        return np.nan
    if sparse.issparse(X):
        mean = float(X.sum()) / n
        mean2 = float(X.multiply(X).sum()) / n
        std = np.sqrt(max(mean2 - mean * mean, 0.0))
    else:
        a = np.asarray(X)
        mean, std = float(a.mean()), float(a.std())
    return np.nan if std == 0 else mean / std


def _xplain_top(expl: pd.DataFrame, groups: list[str], dims: list[int], k: int) -> dict[tuple[str, int], list[str]]:
    out = {}
    for group in groups:
        for dim in dims:
            x = expl[(expl["group"] == group) & (expl["homology_dim"] == dim)]
            x = x.sort_values(["rank", "gene_idx"], kind="stable")
            out[(group, dim)] = x["gene"].astype(str).head(k).tolist()
    return out


def _scanpy_da(adata: ad.AnnData, groupby: str, groups: list[str], layer: str) -> tuple[dict[str, list[str]], pd.DataFrame]:
    import scanpy as sc

    key = "_xplaintda_da"
    sc.tl.rank_genes_groups(
        adata,
        groupby=groupby,
        groups=groups,
        reference="rest",
        n_genes=adata.n_vars,
        method="wilcoxon",
        use_raw=False,
        layer=None if layer == "X" else layer,
        key_added=key,
    )

    rankings, tables = {}, []
    for group in groups:
        df = sc.get.rank_genes_groups_df(adata, group=group, key=key)
        df = df[(df["pvals_adj"] < 0.01) & (df["logfoldchanges"] > 0.5)].copy()
        df.insert(0, "group", group)
        df.insert(1, "rank", np.arange(1, len(df) + 1))
        df.rename(columns={"names": "feature"}, inplace=True)
        rankings[group] = df["feature"].astype(str).tolist()
        tables.append(df)
    return rankings, pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def _snapatac2_da(adata: ad.AnnData, groupby: str, groups: list[str], layer: str) -> tuple[dict[str, list[str]], pd.DataFrame]:
    import snapatac2 as snap

    if layer != "X":
        adata = adata.copy()
        adata.X = adata.layers[layer].copy()

    rankings, tables = {}, []
    labels = adata.obs[groupby].astype(str).to_numpy()
    for group in groups:
        g1 = np.flatnonzero(labels == group).tolist()
        g2 = np.flatnonzero(labels != group).tolist()
        df = snap.tl.diff_test(adata, cell_group1=g1, cell_group2=g2, direction="both").to_pandas()
        df = df[df["adjusted p-value"] < 0.05].copy()
        df.insert(0, "group", group)
        df.insert(1, "rank", np.arange(1, len(df) + 1))
        df.rename(columns={"feature name": "feature"}, inplace=True)
        rankings[group] = df["feature"].astype(str).tolist()
        tables.append(df)
    return rankings, pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def process_one(root: Path, dataset: str, modality: str, ks: list[int], layer_override: str | None) -> None:
    adata_path, outdir = project_paths(root, dataset, modality)
    expl_path = outdir / "results" / "explanations.csv"
    config_path = outdir / "config.json"
    groupby = LABELS[dataset]
    max_k = max(ks)

    if not adata_path.is_file():
        print(f"[SKIP] Missing AnnData: {adata_path}", flush=True)
        return
    if not expl_path.is_file():
        print(f"[SKIP] Missing explanations: {expl_path}", flush=True)
        return

    run_config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    layer = layer_override or run_config.get("layer", "X")

    print(f"\n[{dataset} | {modality}]")
    print(f"AnnData:      {adata_path}")
    print(f"Explanations: {expl_path}")
    print(f"Groupby:      {groupby}")
    print(f"k:            {ks}", flush=True)

    adata = ad.read_h5ad(adata_path)
    if groupby not in adata.obs:
        raise KeyError(f"{groupby!r} not found in {adata_path}.obs")
    if not adata.var_names.is_unique:
        raise ValueError(f"{adata_path}: adata.var_names must be unique")
    if max_k > adata.n_vars:
        old_max_k = max_k
        max_k = adata.n_vars
        print(f"{adata_path}: max(k)={old_max_k} exceeds n_vars={adata.n_vars}. New max_k is set to {max_k}")

    valid = adata.obs[groupby].notna().to_numpy()
    adata = adata[valid].copy()
    adata.obs[groupby] = pd.Categorical(adata.obs[groupby].astype(str))

    expl = pd.read_csv(expl_path)
    required = {"group", "homology_dim", "gene", "gene_idx", "rank"}
    missing = required - set(expl.columns)
    if missing:
        raise ValueError(f"Missing columns in {expl_path}: {sorted(missing)}")
    expl["group"] = expl["group"].astype(str)
    expl["homology_dim"] = expl["homology_dim"].astype(int)

    obs_groups = set(adata.obs[groupby].astype(str).unique())
    groups = [g for g in expl["group"].drop_duplicates() if g in obs_groups]
    dims = sorted(map(int, expl["homology_dim"].unique()))
    if len(groups) < 2:
        raise ValueError(f"{dataset}/{modality}: at least two shared groups are required")
    if not dims:
        raise ValueError(f"{dataset}/{modality}: no homology dimensions found")

    X = _matrix(adata, layer)
    var_names = adata.var_names.astype(str)
    var_set = set(var_names)

    xplain_top_max = _xplain_top(expl, groups, dims, max_k)
    short = [(g, d, len(fs)) for (g, d), fs in xplain_top_max.items() if len(fs) < max_k]
    if short:
        raise ValueError(f"Fewer than max(k) XplainTDA features, e.g. {short[:5]}")

    bad = sorted({f for fs in xplain_top_max.values() for f in fs if f not in var_set})
    if bad:
        raise ValueError(f"XplainTDA features missing from AnnData var_names, e.g. {bad[:5]}")

    da_top_all, da_table = (
        _scanpy_da(adata, groupby, groups, layer)
        if modality == "GEX"
        else _snapatac2_da(adata, groupby, groups, layer)
    )
    da_method = "Scanpy_DA" if modality == "GEX" else "SnapATAC2_DA"

    if "n_cells" in adata.var:
        coverage = pd.to_numeric(adata.var["n_cells"], errors="coerce").to_numpy()
        if np.isnan(coverage).any():
            raise ValueError("adata.var['n_cells'] contains non-numeric or missing values")
    else:
        coverage = X.getnnz(axis=0) if sparse.issparse(X) else np.count_nonzero(np.asarray(X), axis=0)
    top_coverage_max = _top(coverage, var_names, max_k)

    group_data = {}
    for group in groups:
        mask = adata.obs[groupby].astype(str).to_numpy() == group
        Xg = X[mask]
        expression = np.asarray(Xg.sum(axis=0)).ravel()
        group_data[group] = {"X": Xg, "top_expression": _top(expression, var_names, max_k)}

    metrics = []
    for k in ks:
        top_coverage = set(top_coverage_max[:k])

        for group in groups:
            Xg = group_data[group]["X"]
            top_expression = set(group_data[group]["top_expression"][:k])
            da_features_all = da_top_all.get(group, [])
            da_available = len(da_features_all) >= k

            if not da_available:
                print(
                    f"[SKIP DA] {dataset} | {modality} | {group}: "
                    f"k={k} exceeds {len(da_features_all)} available differential features"
                )

            for dim in dims:
                xfeatures = xplain_top_max[(group, dim)][:k]

                # XplainTDA metrics continue for every requested k.
                fs = set(xfeatures)
                e_inter = len(fs & top_expression)
                c_inter = len(fs & top_coverage)
                metrics.extend([
                    {
                        "dataset": dataset, "omic": modality, "metric": "expression_nonoverlap",
                        "method": "XplainTDA", "group": group, "homology_dim": dim,
                        "k": k, "value": e_inter / k, "overlap_count": e_inter,
                    },
                    {
                        "dataset": dataset, "omic": modality, "metric": "coverage_nonoverlap",
                        "method": "XplainTDA", "group": group, "homology_dim": dim,
                        "k": k, "value": c_inter / k, "overlap_count": c_inter,
                    },
                ])

                idx = var_names.get_indexer(xfeatures)
                if (idx < 0).any():
                    missing_features = [f for f, i in zip(xfeatures, idx) if i < 0]
                    raise ValueError(f"XplainTDA features missing from AnnData: {missing_features[:5]}")
                metrics.append({
                    "dataset": dataset, "omic": modality, "metric": "STN",
                    "method": "XplainTDA", "group": group, "homology_dim": dim,
                    "k": k, "value": _stn(Xg[:, idx]), "overlap_count": np.nan,
                })

                # Stop DA-related metrics once this group has fewer than k DA features.
                if not da_available:
                    continue

                # Here min(k, len(da_features_all)) == k by construction.
                m = min(k, len(da_features_all))
                dfeatures = da_features_all[:m]
                xfeatures_cmp = xplain_top_max[(group, dim)][:m]
                top_expression_cmp = set(group_data[group]["top_expression"][:m])
                top_coverage_cmp = set(top_coverage_max[:m])

                inter = len(set(xfeatures_cmp) & set(dfeatures))
                metrics.append({
                    "dataset": dataset, "omic": modality, "metric": "similarity",
                    "method": "XplainTDA_vs_DA", "group": group, "homology_dim": dim,
                    "k": k, "value": inter / m, "overlap_count": inter,
                })

                fs = set(dfeatures)
                e_inter = len(fs & top_expression_cmp)
                c_inter = len(fs & top_coverage_cmp)
                metrics.extend([
                    {
                        "dataset": dataset, "omic": modality, "metric": "expression_nonoverlap",
                        "method": da_method, "group": group, "homology_dim": dim,
                        "k": k, "value": e_inter / m, "overlap_count": e_inter,
                    },
                    {
                        "dataset": dataset, "omic": modality, "metric": "coverage_nonoverlap",
                        "method": da_method, "group": group, "homology_dim": dim,
                        "k": k, "value": c_inter / m, "overlap_count": c_inter,
                    },
                ])

                idx = var_names.get_indexer(dfeatures)
                if (idx < 0).any():
                    missing_features = [f for f, i in zip(dfeatures, idx) if i < 0]
                    raise ValueError(f"{da_method} features missing from AnnData: {missing_features[:5]}")
                metrics.append({
                    "dataset": dataset, "omic": modality, "metric": "STN",
                    "method": da_method, "group": group, "homology_dim": dim,
                    "k": k, "value": _stn(Xg[:, idx]), "overlap_count": np.nan,
                })

        for dim in dims:
            for group in groups:
                focal = set(xplain_top_max[(group, dim)][:k])
                intersections = [
                    len(focal & set(xplain_top_max[(other, dim)][:k]))
                    for other in groups if other != group
                ]
                mean_inter = float(np.mean(intersections))
                metrics.append({
                    "dataset": dataset, "omic": modality, "metric": "specificity",
                    "method": "XplainTDA", "group": group, "homology_dim": dim,
                    "k": k, "value": 1 - (mean_inter / k), "overlap_count": mean_inter,
                })

    top_rows = []
    for group in groups:
        for rank, feature in enumerate(da_top_all.get(group, [])[:max_k], 1):
            top_rows.append({
                "dataset": dataset, "omic": modality, "method": da_method,
                "group": group, "homology_dim": np.nan, "rank": rank, "feature": feature,
            })
        for dim in dims:
            for rank, feature in enumerate(xplain_top_max[(group, dim)], 1):
                top_rows.append({
                    "dataset": dataset, "omic": modality, "method": "XplainTDA",
                    "group": group, "homology_dim": dim, "rank": rank, "feature": feature,
                })

    result_dir = outdir / "results" / "xplaintda_vs_da"
    result_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(metrics).to_csv(result_dir / f"metrics.csv", index=False)
    pd.DataFrame(top_rows).to_csv(result_dir / f"top_features.csv", index=False)
    da_table.to_csv(result_dir / f"da_rankings.csv", index=False)
    (result_dir / f"config.json").write_text(json.dumps({
        "version": VERSION,
        "dataset": dataset,
        "omic": modality,
        "adata_path": str(adata_path),
        "explanations_dir": str(outdir),
        "explanations_path": str(expl_path),
        "groupby": groupby,
        "layer": layer,
        "ks": ks,
        "max_k": max_k,
        "groups": groups,
        "homology_dims": dims,
        "da_method": da_method,
    }, indent=2))

    print(f"Saved results to: {result_dir}", flush=True)


def main() -> None:
    args = parse_args()
    ks = list(dict.fromkeys(args.ks))
    if any(k < 1 for k in ks):
        raise ValueError("All k values must be >= 1")

    root = args.root.expanduser().resolve()
    for dataset in args.datasets:
        for omic in args.omics:
            process_one(root, dataset, omic, ks, args.layer)


if __name__ == "__main__":
    main()
