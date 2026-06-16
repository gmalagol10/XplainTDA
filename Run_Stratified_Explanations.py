#!/usr/bin/env python3
"""Run stratified local and global XplainTDA explanations.

Example
-------
python Run_Stratified_Explanations_GRAE_v2.py \
    data.h5ad results CellType \
    --embedding-key X_GRAE \
    --grae-model-path data_GRAE_embedder.joblib
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad

import defaults as d
from Stratified_XplainTDA import StratifiedXplainTDA


def _parse_csv(value, cast=str):
    if value in (None, "", "None", "none", "null", "all"):
        return None
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run stratified XplainTDA with one full-data GRAE model/embedding "
            "pair. If the pair is missing or incompatible, fit a new default "
            "embedder once and save both artifacts in the output directory."
        )
    )
    parser.add_argument("adata_path", help="Input .h5ad file.")
    parser.add_argument("outdir", help="Output directory.")
    parser.add_argument("groupby", help="adata.obs column defining strata.")
    parser.add_argument("--embedding-key", default="X_GRAE", help="Key containing the paired full-data embedding in adata.obsm.")
    parser.add_argument("--grae-model-path", default=None, help="Path to a model saved with GRAEEmbedder.save.")
    parser.add_argument("--groups", default=None, help="Comma-separated groups.")
    parser.add_argument("--genes", default=None, help="Comma-separated gene names.")
    parser.add_argument("--dims", default=None, help="Comma-separated homology dimensions; default is all.")
    parser.add_argument("--min-cells", type=int, default=d.DEFAULT_MIN_CELLS)
    parser.add_argument("--maxdim", type=int, default=d.DEFAULT_MAXDIM)
    parser.add_argument("--n-neighbors", type=int, default=d.DEFAULT_N_NEIGHBORS)
    parser.add_argument("--n-jobs", type=int, default=d.DEFAULT_N_JOBS)
    parser.add_argument("--ph-backend", default=d.DEFAULT_PH_BACKEND)
    parser.add_argument("--global-ph-backend", default=d.DEFAULT_GLOBAL_PH_BACKEND)
    parser.add_argument("--skip-local", action="store_true", help="Skip local explanations.")
    parser.add_argument("--skip-global", action="store_true", help="Skip global explanations.")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70, flush=True)
    print("RUN CONFIGURATION", flush=True)
    print("=" * 70, flush=True)

    for name, value in vars(args).items():
        print(f"{name:25s}: {value}", flush=True)
    print("=" * 70, flush=True)

    
    
    adata_path = Path(args.adata_path).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if not adata_path.is_file():
        raise FileNotFoundError(f"AnnData file not found: {adata_path}")
    if args.skip_local and args.skip_global:
        raise ValueError("Both local and global explanations were disabled.")

    print(f"Reading AnnData: {adata_path}", flush=True)
    adata = ad.read_h5ad(adata_path)

    groups = _parse_csv(args.groups, str)
    genes = _parse_csv(args.genes, str)
    dims = _parse_csv(args.dims, int)

    runner = StratifiedXplainTDA(
        adata=adata,
        groupby=args.groupby,
        groups=groups,
        min_cells=args.min_cells,
        n_neighbors=args.n_neighbors,
        maxdim=args.maxdim,
        ph_backend=args.ph_backend,
        embedding_key=args.embedding_key,
        grae_model_path=args.grae_model_path,
        outdir=outdir,
    )

    if not args.skip_local:
        print("Running local explanations", flush=True)
        runner.local_expl(
            genes=genes,
            dims=dims,
            n_jobs=args.n_jobs,
            backend=d.DEFAULT_BACKEND,
        )

    if not args.skip_global:
        print("Running global explanations", flush=True)
        runner.global_expl(
            genes=genes,
            dims=dims,
            n_jobs=args.n_jobs,
            backend=d.DEFAULT_BACKEND,
            ph_backend=args.global_ph_backend,
        )

    which = (
        "global"
        if args.skip_local
        else "local"
        if args.skip_global
        else "both"
    )
    paths = runner.save_results(outdir=outdir, which=which)
    for name, path in paths.items():
        print(f"Saved {name}: {path}", flush=True)


if __name__ == "__main__":
    main()
