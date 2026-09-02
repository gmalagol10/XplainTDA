#!/usr/bin/env python3
"""Run HOMER motif enrichment on stratified XplainTDA Peak results.

Expected input files, one cell type per CSV:
  Datasets/<DATASET>/XplainTDA/Peak/Explanations/<LABEL>/results/groups/<CELLTYPE>.csv

Output root:
  Datasets/<DATASET>/XplainTDA/Peak/Explanations/<LABEL>/results/groups/Motif/

Default behavior:
  - process the Peak omic only
  - process every discovered group CSV
  - run each homology dimension separately when homology_dim is present
  - select top 150 and top 200 peaks per group/H
  - --top accepts fixed counts and percentages, e.g. 100,200,1%,5%
  - rank by rank ascending when available, otherwise by score descending
  - use the column gene as the peak identifier when present
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

OMIC = "Peak"
HUMAN_GENOME = "../AnnotRef/hs/T2T/chm13v2.0.fa"
GENOMES = {
    "PBMC": HUMAN_GENOME,
    "HumanBrain": HUMAN_GENOME,
    "BMMCMultiOme": "hg38"

}
PEAK_COLS = ("gene", "peak", "feature", "feature_name", "var_name", "peak_id")


def csv_list(x: str | None, cast=str):
    if x is None or str(x).strip().lower() in {"", "all", "none", "null"}:
        return None
    return [cast(v.strip()) for v in str(x).split(",") if v.strip()]


def parse_top_specs(x: str | None):
    """Parse --top entries: integers are top-N, values ending in % are percentages."""
    if x is None or str(x).strip().lower() in {"", "all", "none", "null"}:
        x = "150,200"

    specs = []
    for raw in str(x).split(","):
        item = raw.strip()
        if not item:
            continue

        if item.endswith("%"):
            pct = float(item[:-1])
            if not 0 < pct <= 100:
                raise ValueError(f"Invalid percentage in --top: {item}")
            specs.append(("pct", pct))
        else:
            n = int(item)
            if n <= 0:
                raise ValueError(f"Invalid top-N value in --top: {item}")
            specs.append(("n", n))

    if not specs:
        raise ValueError("No valid --top specifications were provided")
    return specs


def resolve_top_n(spec, n_ranked: int) -> int:
    kind, value = spec
    if n_ranked <= 0:
        return 0
    if kind == "n":
        return min(int(value), n_ranked)
    if kind == "pct":
        return max(1, min(n_ranked, math.ceil(n_ranked * float(value) / 100.0)))
    raise ValueError(f"Unknown top specification: {spec}")


def top_tag(spec) -> str:
    kind, value = spec
    if kind == "n":
        return f"top_{int(value)}"
    pct = int(value) if float(value).is_integer() else str(value).replace(".", "p")
    return f"top_{pct}pct"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run HOMER on top peaks from stratified XplainTDA Peak group CSVs."
    )
    p.add_argument("--root", default="Datasets")
    p.add_argument(
        "--top", "--top-n",
        dest="top",
        default="150,200",
        help="Comma-separated selections. Integers mean top-N; values ending in %% mean percentage, e.g. 100,200,1%%,5%%.",
    )
    p.add_argument("--datasets", default=None, help="Optional comma-separated dataset filter.")
    p.add_argument("--labels", default=None, help="Optional comma-separated label filter.")
    p.add_argument("--groups", default=None, help="Optional comma-separated group/cell-type filter.")
    p.add_argument("--dims", default=None, help="Optional comma-separated H dimensions, e.g. 0,1,2.")
    p.add_argument(
        "--aggregate-h",
        action="store_true",
        help="Collapse across H by keeping the best row per peak. Default: run HOMER separately per H.",
    )
    p.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    p.add_argument("--size", default="given", help="HOMER -size argument. Default: given.")
    p.add_argument("--extra-homer-args", default="", help="Extra arguments appended to findMotifsGenome.pl.")
    p.add_argument("--dry-run", action="store_true", help="Write files and commands but do not run HOMER.")
    p.add_argument("--overwrite", action="store_true", help="Rerun HOMER if homerResults.html already exists.")
    return p.parse_args()


def safe_name(x) -> str:
    return str(x).replace("/", "_").replace("\\", "_").replace(" ", "_")


def find_group_csvs(root: Path, datasets=None, labels=None, groups=None):
    rows = []
    pattern = f"*/XplainTDA/{OMIC}/Explanations/*/results/groups/*.csv"
    for path in sorted(root.glob(pattern)):
        if "Motif" in path.parts:
            continue
        parts = path.parts
        dataset = parts[parts.index("XplainTDA") - 1]
        label = parts[parts.index("Explanations") + 1]
        group = path.stem
        if datasets and dataset not in datasets:
            continue
        if labels and label not in labels:
            continue
        if groups and group not in groups:
            continue
        rows.append((dataset, label, group, path))
    return rows


def peak_col(df: pd.DataFrame) -> str:
    for col in PEAK_COLS:
        if col in df.columns:
            return col
    raise ValueError(f"No peak column found. Tried {PEAK_COLS}. Columns: {list(df.columns)}")


def score_col(df: pd.DataFrame) -> str:
    if "score_normalized" in df.columns:
        return "score_normalized"
    if "score" in df.columns:
        return "score"
    raise ValueError(f"No score column found. Columns: {list(df.columns)}")


def clean_result_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sc = score_col(df)
    df[sc] = pd.to_numeric(df[sc], errors="coerce")
    df = df[np.isfinite(df[sc])]
    if "quantile" in df.columns:
        df = df[np.isclose(pd.to_numeric(df["quantile"], errors="coerce"), 0.95)]
    if "quantile_q" in df.columns:
        df = df[np.isclose(pd.to_numeric(df["quantile_q"], errors="coerce"), 0.95)]
    return df


def rank_peaks(df: pd.DataFrame) -> pd.DataFrame:
    pc, sc = peak_col(df), score_col(df)
    x = clean_result_table(df)
    if x.empty:
        return pd.DataFrame(columns=["peak", "importance"])

    cols = [pc, sc]
    if "rank" in x.columns:
        x["rank"] = pd.to_numeric(x["rank"], errors="coerce")
        cols.append("rank")
        x = x.sort_values(["rank", sc], ascending=[True, False], kind="stable")
    else:
        x = x.sort_values(sc, ascending=False, kind="stable")

    if "homology_dim" in x.columns:
        cols.append("homology_dim")

    out = x[cols].drop_duplicates(pc, keep="first")
    return out.rename(columns={pc: "peak", sc: "importance"}).reset_index(drop=True)


def iter_ranked_tables(df: pd.DataFrame, dims=None, aggregate_h=False):
    if aggregate_h or "homology_dim" not in df.columns:
        yield "H_all", rank_peaks(df)
        return

    h_values = pd.to_numeric(df["homology_dim"], errors="coerce")
    hs = sorted(h_values.dropna().astype(int).unique())
    if dims is not None:
        hs = [h for h in hs if h in dims]

    for h in hs:
        ranked = rank_peaks(df[h_values == h])
        yield f"H{h}", ranked


def peak_to_bed(peak: str):
    s = str(peak).strip().replace(",", "")
    s = re.sub(r"\s+", "", s)
    m = re.match(r"^(.+?)[:_](\d+)[-_](\d+)$", s)
    if not m:
        return None
    chrom, start, end = m.group(1), int(m.group(2)), int(m.group(3))
    if not chrom.startswith("chr") and re.match(r"^(\d+|X|Y|M|MT)$", chrom):
        chrom = "chr" + chrom
    return (chrom, start, end) if end > start else None


def write_bed(peaks, bed: Path) -> int:
    rows, bad = [], []
    for peak in peaks:
        row = peak_to_bed(peak)
        rows.append(row) if row else bad.append(peak)

    pd.DataFrame(rows).to_csv(bed, sep="\t", header=False, index=False)
    if bad:
        pd.Series(bad, name="bad_peak_name").to_csv(bed.with_suffix(".bad_peaks.csv"), index=False)
    return len(rows)


def run_homer(bed: Path, genome: str, outdir: Path, threads: int, size: str, extra_args: str, dry_run: bool):
    cmd = ["findMotifsGenome.pl", str(bed), genome, str(outdir), "-size", str(size), "-p", str(threads)]
    if extra_args:
        cmd.extend(shlex.split(extra_args))

    (outdir / "command.txt").write_text(shlex.join(cmd) + "\n")
    print("CMD:", shlex.join(cmd), flush=True)

    if dry_run:
        return

    with (outdir / "homer.stdout.log").open("w") as out, (outdir / "homer.stderr.log").open("w") as err:
        subprocess.run(cmd, check=True, stdout=out, stderr=err)


def main() -> None:
    a = parse_args()
    root = Path(a.root)
    top_specs = parse_top_specs(a.top)
    datasets = csv_list(a.datasets)
    labels = csv_list(a.labels)
    groups = csv_list(a.groups)
    dims = csv_list(a.dims, int)

    print("=" * 80, flush=True)
    print("Stratified XplainTDA Peak group CSVs -> HOMER", flush=True)
    print(f"root={root} | top={','.join(top_tag(s) for s in top_specs)} | dims={dims or 'all'} | aggregate_h={a.aggregate_h}", flush=True)
    print(f"threads={a.threads} | size={a.size} | dry_run={a.dry_run}", flush=True)
    print(f"genomes={GENOMES}", flush=True)
    print("=" * 80, flush=True)

    files = find_group_csvs(root, datasets=datasets, labels=labels, groups=groups)
    if not files:
        raise FileNotFoundError(
            f"No group CSV files found under {root}/*/XplainTDA/{OMIC}/Explanations/*/results/groups/*.csv"
        )

    for dataset, label, group, csv_path in files:
        genome = GENOMES.get(dataset)
        if genome is None:
            print(f"[SKIP] {dataset}: no genome configured", flush=True)
            continue

        df = pd.read_csv(csv_path)
        motif_root = csv_path.parent / "Motif"

        print(f"\n{time.strftime('%a, %d %b %Y %H:%M:%S')} | {dataset} | {label} | {group}", flush=True)
        print(f"input={csv_path}", flush=True)
        print(f"output_root={motif_root}", flush=True)

        for h_tag, ranked in iter_ranked_tables(df, dims=dims, aggregate_h=a.aggregate_h):
            if ranked.empty:
                print(f"[SKIP] {dataset}/{group}/{h_tag}: no ranked peaks", flush=True)
                continue

            for spec in top_specs:
                n_top = resolve_top_n(spec, len(ranked))
                tag = top_tag(spec)
                outdir = motif_root / safe_name(group) / h_tag / tag
                outdir.mkdir(parents=True, exist_ok=True)

                ranked.to_csv(outdir / "ranked_peaks.csv", index=False)
                top = ranked.head(n_top)
                top.to_csv(outdir / "top_peaks.csv", index=False)

                bed = outdir / "peaks.bed"
                n_bed = write_bed(top["peak"], bed)
                done = outdir / "homerResults.html"

                print(
                    f"[{dataset} | {group} | {h_tag} | {tag}] "
                    f"ranked={len(ranked)} | selected={n_top} | bed={n_bed} | out={outdir}",
                    flush=True,
                )

                if n_bed == 0:
                    print(f"[SKIP] {dataset}/{group}/{h_tag}/{tag}: no valid BED rows", flush=True)
                    continue
                if done.exists() and not a.overwrite:
                    print(f"[SKIP] HOMER already exists: {done} ; use --overwrite to rerun", flush=True)
                    continue

                run_homer(bed, genome, outdir, a.threads, a.size, a.extra_homer_args, a.dry_run)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
