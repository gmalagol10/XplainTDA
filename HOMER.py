#!/usr/bin/env python3
"""Run HOMER on top-percent global XplainTDA Peak results.

Input files are discovered under:
  Datasets/<DATASET>/XplainTDA/Peak/Explanations/<LABEL>/global/global_results.csv

Output is written under the same global directory:
  Datasets/<DATASET>/XplainTDA/Peak/Explanations/<LABEL>/global/HOMER/top_<PCT>pct/<CELLTYPE>/

Fixed analysis choices:
  - omic folder: Peak
  - score: score_normalized if present, otherwise score
  - quantile_q: 0.95
  - homology aggregation: max score per peak across all H dimensions
  - genomes: PBMC/HumanBrain -> ../AnnotRef/hs/T2T/chm13v2.0.fa ; MouseBrain -> mm10
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shlex
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

Q = 0.95
OMIC = "Peak"
GENOME = {
    "PBMC": "../AnnotRef/hs/T2T/chm13v2.0.fa",
    "HumanBrain": "../AnnotRef/hs/T2T/chm13v2.0.fa",
    "MouseBrain": "mm10",
}
PEAK_COLS = ("gene", "peak", "feature", "feature_name", "var_name", "peak_id")


def csv_list(x):
    if x is None or str(x).lower() in {"", "all", "none", "null"}:
        return None
    return [v.strip() for v in str(x).split(",") if v.strip()]


def parse_args():
    p = argparse.ArgumentParser(description="HOMER motifs for top 1% or 5% global XplainTDA Peak results.")
    p.add_argument("--root", default="Datasets")
    p.add_argument("--top-pct", type=float, required=True, choices=[1, 5], help="Use the top 1% or 5% ranked peaks.")
    p.add_argument("--datasets", default=None, help="Optional comma-separated dataset filter.")
    p.add_argument("--labels", default=None, help="Optional comma-separated label filter, e.g. Raw_CellType.")
    p.add_argument("--groups", default=None, help="Optional comma-separated cell-type filter.")
    p.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    p.add_argument("--dry-run", action="store_true", help="Write files and commands but do not run HOMER.")
    p.add_argument("--overwrite", action="store_true", help="Rerun HOMER if homerResults.html already exists.")
    return p.parse_args()


def safe_name(x):
    return str(x).replace("/", "_").replace("\\", "_").replace(" ", "_")


def find_result_files(root, datasets=None, labels=None):
    rows = []
    pattern = f"*/XplainTDA/{OMIC}/Explanations/*/global/global_results.csv"
    for path in sorted(Path(root).glob(pattern)):
        ds = path.parts[path.parts.index("XplainTDA") - 1]
        label = path.parts[path.parts.index("Explanations") + 1]
        if datasets and ds not in datasets:
            continue
        if labels and label not in labels:
            continue
        rows.append((ds, label, path, path.parent))
    return rows


def peak_col(df):
    for col in PEAK_COLS:
        if col in df.columns:
            return col
    raise ValueError(f"No peak column found. Tried {PEAK_COLS}. Columns: {list(df.columns)}")


def score_col(df):
    if "score_normalized" in df.columns:
        return "score_normalized"
    if "score" in df.columns:
        return "score"
    raise ValueError(f"No score column found. Columns: {list(df.columns)}")


def rank_peaks(df, group):
    pc, sc = peak_col(df), score_col(df)
    x = df[df["group"].astype(str) == str(group)].copy()
    if "quantile_q" in x.columns:
        x = x[np.isclose(pd.to_numeric(x["quantile_q"], errors="coerce"), Q)]
    x[sc] = pd.to_numeric(x[sc], errors="coerce")
    x = x[np.isfinite(x[sc])]
    if x.empty:
        return pd.DataFrame(columns=["peak", "importance"])
    idx = x.groupby(pc)[sc].idxmax()
    return (x.loc[idx, [pc, sc]]
             .rename(columns={pc: "peak", sc: "importance"})
             .sort_values("importance", ascending=False)
             .reset_index(drop=True))


def peak_to_bed(peak):
    s = str(peak).replace(",", "")
    m = re.match(r"^(.+?)[:_](\d+)[-_](\d+)$", s)
    if not m:
        return None
    chrom, start, end = m.group(1), int(m.group(2)), int(m.group(3))
    return (chrom, start, end) if end > start else None


def write_bed(peaks, bed):
    rows, bad = [], []
    for peak in peaks:
        row = peak_to_bed(peak)
        rows.append(row) if row else bad.append(peak)
    pd.DataFrame(rows).to_csv(bed, sep="\t", header=False, index=False)
    if bad:
        pd.Series(bad, name="bad_peak_name").to_csv(bed.with_suffix(".bad_peaks.csv"), index=False)
    return len(rows)


def run_homer(bed, genome, outdir, threads, dry_run):
    cmd = ["findMotifsGenome.pl", str(bed), genome, str(outdir), "-p", str(threads)]
    (outdir / "command.txt").write_text(shlex.join(cmd) + "\n")
    print("CMD:", shlex.join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def main():
    a = parse_args()
    root = Path(a.root)
    datasets, labels, groups_filter = csv_list(a.datasets), csv_list(a.labels), csv_list(a.groups)
    tag = f"top_{int(a.top_pct)}pct"

    print("=" * 70, flush=True)
    print("Global XplainTDA Peak -> HOMER", flush=True)
    print(f"root={root} | q={Q} | {tag} | threads={a.threads} | dry_run={a.dry_run}", flush=True)
    print(f"genomes={GENOME}", flush=True)
    print("=" * 70, flush=True)

    files = find_result_files(root, datasets=datasets, labels=labels)
    if not files:
        raise FileNotFoundError(f"No files found under {root}/*/XplainTDA/{OMIC}/Explanations/*/global/global_results.csv")

    for ds, label, csv_path, global_dir in files:
        genome = GENOME.get(ds)
        if genome is None:
            print(f"[SKIP] {ds}: no hard-coded genome", flush=True)
            continue

        df = pd.read_csv(csv_path)
        if "group" not in df.columns:
            raise ValueError(f"Missing 'group' column in {csv_path}")
        groups = sorted(df["group"].dropna().astype(str).unique())
        if groups_filter:
            groups = [g for g in groups if g in groups_filter]

        print(f"\n{time.strftime('%a, %d %b %Y %H:%M:%S')} | {ds} | {label}", flush=True)
        print(f"results={csv_path}", flush=True)
        print(f"output_root={global_dir / 'HOMER' / tag}", flush=True)

        for group in groups:
            ranked = rank_peaks(df, group)
            if ranked.empty:
                print(f"[SKIP] {ds}/{group}: no ranked peaks", flush=True)
                continue

            n_top = max(1, math.ceil(len(ranked) * a.top_pct / 100.0))
            outdir = global_dir / "HOMER" / tag / safe_name(group)
            outdir.mkdir(parents=True, exist_ok=True)

            ranked.to_csv(outdir / "ranked_peaks.csv", index=False)
            top = ranked.head(n_top)
            top.to_csv(outdir / "top_peaks.csv", index=False)

            bed = outdir / "peaks.bed"
            n_bed = write_bed(top["peak"], bed)
            done = outdir / "homerResults.html"

            print(f"[{ds} | {group}] ranked={len(ranked)} | selected={n_top} | bed={n_bed} | out={outdir}", flush=True)
            if n_bed == 0:
                print(f"[SKIP] {ds}/{group}: no valid BED rows", flush=True)
                continue
            if done.exists() and not a.overwrite:
                print(f"[SKIP] HOMER already exists: {done} ; use --overwrite to rerun", flush=True)
                continue

            run_homer(bed, genome, outdir, a.threads, a.dry_run)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
