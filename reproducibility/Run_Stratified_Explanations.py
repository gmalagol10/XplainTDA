#!/usr/bin/env python3
"""Command-line entry point for resumable stratified XplainTDA explanations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

try:
    from . import defaults as d
    from .XplainTDA_PH import DEFAULT_PH_BACKEND, XplainTDA
    from .XplainTDA_Stratified import StratifiedExplainer
except ImportError:  # normal path when executed as a script
    import defaults as d
    from XplainTDA_PH import DEFAULT_PH_BACKEND, XplainTDA
    from XplainTDA_Stratified import StratifiedExplainer


RESULT_COLUMNS = {
    "group", "n_cells", "n_neighbors", "homology_dim",
    "gene_idx", "gene", "score", "quantile", "rank",
}


def _csv(value: str | None, cast=str):
    if value is None or value.strip().lower() in {"", "all", "none", "null"}:
        return None
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def _safe_name(value) -> str:
    return str(value).replace("/", "_").replace("\\", "_").replace(" ", "_")


def _read_complete(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    return df if len(df) and RESULT_COLUMNS.issubset(df.columns) else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute resumable group-specific XplainTDA explanations."
    )
    parser.add_argument("adata_path", type=Path)
    parser.add_argument("outdir", type=Path)
    parser.add_argument("groupby")
    parser.add_argument("--groups")
    parser.add_argument("--genes")
    parser.add_argument("--dims")
    parser.add_argument("--layer", default=d.DEFAULT_LAYER)
    parser.add_argument("--embedding-key", default="X_GRAE")
    parser.add_argument(
        "--embedder-path", "--grae-model-path", dest="embedder_path", type=Path
    )
    parser.add_argument("--embedder", default=d.DEFAULT_EMBEDDER)
    parser.add_argument("--n-components", type=int, default=d.DEFAULT_N_COMPONENTS)
    parser.add_argument("--n-neighbors", type=int, default=d.DEFAULT_N_NEIGHBORS)
    parser.add_argument("--maxdim", type=int, default=d.DEFAULT_MAXDIM)
    parser.add_argument("--min-cells", type=int, default=d.DEFAULT_MIN_CELLS)
    parser.add_argument("--neighbor-backend", default=d.DEFAULT_NEIGHBOR_BACKEND)
    parser.add_argument("--ph-backend", default=DEFAULT_PH_BACKEND)
    parser.add_argument(
        "--perturbation", choices=["zero", "permute"], default=d.DEFAULT_PERTURBATION
    )
    parser.add_argument("--n-jobs", type=int, default=d.DEFAULT_N_JOBS)
    parser.add_argument("--backend", default=d.DEFAULT_BACKEND)
    parser.add_argument("--random-state", type=int, default=d.DEFAULT_RANDOM_STATE)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore existing checkpoints and recompute all requested groups.",
    )
    return parser.parse_args()


def _reuse_tda(runner: StratifiedExplainer, adata, group) -> None:
    path = runner.outdir / "tda" / f"{_safe_name(group)}.joblib"
    if not path.is_file():
        return

    mask = np.asarray(adata.obs[runner.groupby] == group)
    adata_group = adata[mask].copy()
    try:
        tda = XplainTDA.load(
            path,
            adata=adata_group,
            fitted_embedder=runner.full_tda.embedder,
        )
        expected_k = min(runner.n_neighbors, int(mask.sum()) - 1)
        compatible = (
            tda.n_neighbors == expected_k
            and tda.maxdim == runner.maxdim
            and tda.neighbor_backend == runner.neighbor_backend
            and tda.ph_backend.lower() == runner.ph_backend.lower()
        )
        if not compatible:
            raise ValueError("saved TDA configuration differs from this run")
        runner.tdas[group] = tda
        if runner.verbose:
            print(f"Reusing saved TDA for {group}: {path}", flush=True)
    except Exception as exc:
        if runner.verbose:
            print(f"Cannot reuse saved TDA for {group} ({exc}); recomputing it.", flush=True)


def _merge_group_results(outdir: Path, groups) -> tuple[pd.DataFrame, list]:
    tables, missing = [], []
    for group in groups:
        path = outdir / "results" / "groups" / f"{_safe_name(group)}.csv"
        df = _read_complete(path)
        if df is None:
            missing.append(group)
        else:
            tables.append(df)

    combined = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    if tables:
        path = outdir / "results" / "explanations.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(path) + ".tmp")
        combined.to_csv(tmp, index=False)
        tmp.replace(path)
    return combined, missing


def main() -> None:
    args = parse_args()
    adata_path = args.adata_path.expanduser().resolve()
    outdir = args.outdir.expanduser().resolve()
    if not adata_path.is_file():
        raise FileNotFoundError(f"AnnData file not found: {adata_path}")

    adata = ad.read_h5ad(adata_path)
    if args.groupby not in adata.obs:
        raise KeyError(f"{args.groupby!r} is not a column in adata.obs")

    available = list(pd.Series(adata.obs[args.groupby]).dropna().unique())
    groups = _csv(args.groups) or available
    unknown = [g for g in groups if g not in set(available)]
    if unknown:
        raise ValueError(f"unknown groups for {args.groupby!r}: {unknown}")

    if args.restart:
        pending, completed = groups, []
    else:
        completed, pending = [], []
        for group in groups:
            path = outdir / "results" / "groups" / f"{_safe_name(group)}.csv"
            (completed if _read_complete(path) is not None else pending).append(group)

    if completed:
        print(f"Resume: skipping completed groups: {', '.join(map(str, completed))}", flush=True)
    if pending:
        print(f"Resume: groups still to run: {', '.join(map(str, pending))}", flush=True)

        runner = StratifiedExplainer(
            adata=adata,
            groupby=args.groupby,
            groups=pending,
            layer=args.layer,
            embedding_key=args.embedding_key,
            embedder_path=args.embedder_path,
            embedder=args.embedder,
            n_components=args.n_components,
            n_neighbors=args.n_neighbors,
            maxdim=args.maxdim,
            min_cells=args.min_cells,
            neighbor_backend=args.neighbor_backend,
            ph_backend=args.ph_backend,
            outdir=outdir,
            verbose=not args.quiet,
        )

        runner.fit_embedding()
        if not args.restart:
            for group in pending:
                _reuse_tda(runner, adata, group)

        runner.explain(
            genes=_csv(args.genes),
            dims=_csv(args.dims, int),
            perturbation=args.perturbation,
            n_jobs=args.n_jobs,
            backend=args.backend,
            random_state=args.random_state,
        )

        config_path = outdir / "config.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text())
            config["groups"] = groups
            config["resumable"] = True
            config_path.write_text(json.dumps(config, indent=2, default=str))
    else:
        print("Resume: all requested groups are already complete.", flush=True)

    combined, missing = _merge_group_results(outdir, groups)
    if missing:
        print(f"Still missing group results: {', '.join(map(str, missing))}", flush=True)
    print(f"Saved {len(combined)} combined explanation rows to {outdir}", flush=True)


if __name__ == "__main__":
    main()
