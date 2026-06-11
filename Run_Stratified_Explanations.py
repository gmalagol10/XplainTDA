#!/usr/bin/env python3
"""Run stratified local and global XplainTDA explanations for an AnnData object.

Launch through Run_Exp.sh, e.g.:
    bash Run_Exp.sh Run_Stratified_Explanations data.h5ad results CellType
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad

from Stratified_XplainTDA import StratifiedXplainTDA


def _none_or_int(x: str | None) -> int | None:
    return None if x in (None, "", "None", "none", "null") else int(x)


def _parse_csv(x: str | None, cast=str):
    if x in (None, "", "None", "none", "null", "all"):
        return None
    return [cast(v.strip()) for v in x.split(",") if v.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run local then global stratified XplainTDA explanations.")
    p.add_argument("adata_path", help="Path to input .h5ad AnnData object.")
    p.add_argument("outdir", default="XplainTDA_results", help="Directory where result tables will be saved.")
    p.add_argument("groupby", help="adata.obs column used for stratification.")
    p.add_argument("--maxdim", type=int, default=2, help="Maximum homology dimension.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    adata_path, outdir = Path(args.adata_path), Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Reading AnnData: {adata_path}", flush=True)
    adata = ad.read_h5ad(adata_path)[:100, :50]

    runner = StratifiedXplainTDA(adata=adata, groupby=args.groupby, maxdim=args.maxdim)

    print("Running local explanations", flush=True)
    local_df = runner.local_expl()

    print("Running global explanations", flush=True)
    global_df = runner.global_expl()

    paths = runner.save_results(outdir, which="both")

if __name__ == "__main__":
    main()
