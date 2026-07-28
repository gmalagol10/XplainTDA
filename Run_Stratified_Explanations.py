#!/usr/bin/env python3
"""Command-line entry point for stratified XplainTDA explanations."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad

try:
    from . import defaults as d
    from .XplainTDA_PH import DEFAULT_PH_BACKEND
    from .XplainTDA_Stratified import StratifiedExplainer
except ImportError:  # pragma: no cover - normal path when executed as a script
    import defaults as d
    from XplainTDA_PH import DEFAULT_PH_BACKEND
    from XplainTDA_Stratified import StratifiedExplainer


def _csv(value: str | None, cast=str):
    """Parse a comma-separated CLI value; ``all`` and empty values mean None."""
    if value is None or value.strip().lower() in {"", "all", "none", "null"}:
        return None
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    """Define and parse the command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute group-specific persistent homology and q=0.95 gene "
            "perturbation explanations in one shared embedding space."
        )
    )
    parser.add_argument("adata_path", type=Path, help="Input .h5ad file")
    parser.add_argument("outdir", type=Path, help="Output directory")
    parser.add_argument("groupby", help="Column in adata.obs defining groups")
    parser.add_argument("--groups", help="Comma-separated groups; default: all")
    parser.add_argument("--genes", help="Comma-separated genes; default: all")
    parser.add_argument("--dims", help="Comma-separated homology dimensions; default: all")
    parser.add_argument("--layer", default=d.DEFAULT_LAYER)
    parser.add_argument("--embedding-key", default="X_GRAE")
    parser.add_argument(
        "--embedder-path",
        "--grae-model-path",
        dest="embedder_path",
        type=Path,
        help="Saved embedder paired with adata.obsm[embedding_key]",
    )
    parser.add_argument("--embedder", default=d.DEFAULT_EMBEDDER)
    parser.add_argument("--n-components", type=int, default=d.DEFAULT_N_COMPONENTS)
    parser.add_argument("--n-neighbors", type=int, default=d.DEFAULT_N_NEIGHBORS)
    parser.add_argument("--maxdim", type=int, default=d.DEFAULT_MAXDIM)
    parser.add_argument("--min-cells", type=int, default=d.DEFAULT_MIN_CELLS)
    parser.add_argument("--neighbor-backend", default=d.DEFAULT_NEIGHBOR_BACKEND)
    parser.add_argument("--ph-backend", default=DEFAULT_PH_BACKEND)
    parser.add_argument(
        "--perturbation",
        choices=["zero", "permute"],
        default=d.DEFAULT_PERTURBATION,
    )
    parser.add_argument("--n-jobs", type=int, default=d.DEFAULT_N_JOBS)
    parser.add_argument("--backend", default=d.DEFAULT_BACKEND)
    parser.add_argument("--random-state", type=int, default=d.DEFAULT_RANDOM_STATE)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Load AnnData, run the stratified workflow, and save all artifacts."""
    args = parse_args()
    adata_path = args.adata_path.expanduser().resolve()
    if not adata_path.is_file():
        raise FileNotFoundError(f"AnnData file not found: {adata_path}")

    runner = StratifiedExplainer(
        adata=ad.read_h5ad(adata_path),
        groupby=args.groupby,
        groups=_csv(args.groups),
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
        outdir=args.outdir,
        verbose=not args.quiet,
    )
    result = runner.explain(
        genes=_csv(args.genes),
        dims=_csv(args.dims, int),
        perturbation=args.perturbation,
        n_jobs=args.n_jobs,
        backend=args.backend,
        random_state=args.random_state,
    )
    print(f"Saved {len(result)} explanation rows to {runner.outdir}", flush=True)


if __name__ == "__main__":
    main()
