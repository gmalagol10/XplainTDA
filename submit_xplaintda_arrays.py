#!/usr/bin/env python3
"""Submit stratified XplainTDA jobs as four SLURM arrays.

Arrays submitted by this script:
    1. GEX local
    2. GEX global
    3. ATAC local, using the existing folder token "Peak"
    4. ATAC global, using the existing folder token "Peak"

Each array element processes one dataset/label pair.
The output folder layout preserves the previous base path and adds the mode:
    Datasets/<dataset>/XplainTDA/<GEX|Peak>/Explanations/<label>/<local|global>

The command intentionally passes only the arguments needed by the old
Run_Exp.sh convention, which forwards $2 ... $9. n_jobs is controlled through
XPLAINTDA_N_JOBS and SLURM_CPUS_PER_TASK.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


DATASETS = [
    ("PBMC", "Raw_CellType"),
    ("HumanBrain", "Raw_celltype"),
    ("MouseBrain", "CellType"),
]

# User-facing omic name -> folder/path token used by the existing directory layout.
OMICS = {
    "GEX": "GEX",
    "ATAC": "Peak",
}


def q(x) -> str:
    return shlex.quote(str(x))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Submit GEX/ATAC x local/global XplainTDA SLURM arrays.")
    p.add_argument("--base-dir", default="Datasets", help="Base directory containing dataset folders.")
    p.add_argument("--launcher", default="Run_Exp.sh", help="Launcher shell script.")
    p.add_argument("--script", default="Run_Stratified_Explanations", help="Python driver name without .py.")
    p.add_argument("--partition", default="slim18", help="SLURM partition.")
    p.add_argument("--time", default="7-00:00:00", help="SLURM time limit.")
    p.add_argument("--embedding-key", default="X_GRAE", help="AnnData .obsm key for the paired GRAE embedding.")
    p.add_argument("--mem-local", default="40G", help="Memory for local jobs.")
    p.add_argument("--mem-global", default="40G", help="Memory for global jobs.")
    p.add_argument("--cpus-local", type=int, default=30, help="CPUs/workers for local jobs.")
    p.add_argument("--cpus-global", type=int, default=36, help="CPUs/workers for global jobs.")
    p.add_argument("--array-max-running", type=int, default=None, help="Optional SLURM array throttle, e.g. 2 for --array=0-2%%2.")
    p.add_argument("--workdir", default=".xplaintda_slurm", help="Directory where generated sbatch scripts are written.")
    p.add_argument("--no-exclusive", action="store_true", help="Do not request exclusive nodes.")
    p.add_argument("--no-strict-input-check", action="store_true", help="Do not check that the h5ad contains the embedding key before running.")
    p.add_argument("--dry-run", action="store_true", help="Write sbatch scripts and print commands without submitting.")
    return p.parse_args()


def bash_array(values: list[str]) -> str:
    return " ".join(q(v) for v in values)


def make_sbatch_script(args: argparse.Namespace, omic_name: str, omic_folder: str, mode: str) -> str:
    labels = [label for _, label in DATASETS]
    datasets = [dataset for dataset, _ in DATASETS]

    is_local = mode == "local"
    cpus = args.cpus_local if is_local else args.cpus_global
    mem = args.mem_local if is_local else args.mem_global
    skip_flag = "--skip-global" if is_local else "--skip-local"
    mode_tag = "L" if is_local else "G"
    array_range = f"0-{len(DATASETS) - 1}"
    if args.array_max_running is not None:
        array_range += f"%{int(args.array_max_running)}"

    exclusive_line = "" if args.no_exclusive else "#SBATCH --exclusive\n"
    strict = "0" if args.no_strict_input_check else "1"

    job_name = f"xTDA{omic_name}{mode_tag}"
    bootstrap_log_dir = Path(args.base_dir) / "slurm_logs"

    return f"""#!/bin/bash
#SBATCH -p {args.partition}
#SBATCH -J {job_name}
#SBATCH --array={array_range}
#SBATCH --mem={mem}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --time={args.time}
{exclusive_line}#SBATCH --output={bootstrap_log_dir}/%x_%A_%a.bootstrap.out

set -euo pipefail

DATASETS=({bash_array(datasets)})
LABELS=({bash_array(labels)})

DATASET="${{DATASETS[$SLURM_ARRAY_TASK_ID]}}"
LABEL="${{LABELS[$SLURM_ARRAY_TASK_ID]}}"
OMIC_NAME={q(omic_name)}
OMIC_FOLDER={q(omic_folder)}
MODE={q(mode)}
SKIP_FLAG={q(skip_flag)}
BASE_DIR={q(args.base_dir)}
LAUNCHER={q(args.launcher)}
SCRIPT={q(args.script)}
EMBEDDING_KEY={q(args.embedding_key)}
STRICT_INPUT_CHECK={strict}

ADATA="${{BASE_DIR}}/${{DATASET}}/CM/${{DATASET}}_${{OMIC_FOLDER}}_Def.h5ad"
GRAE_MODEL="${{BASE_DIR}}/${{DATASET}}/CM/${{DATASET}}_${{OMIC_FOLDER}}_GRAE_embedder.joblib"
OUTDIR="${{BASE_DIR}}/${{DATASET}}/XplainTDA/${{OMIC_FOLDER}}/Explanations/${{LABEL}}/${{MODE}}"
LOG="${{BASE_DIR}}/${{DATASET}}/${{DATASET}}_${{OMIC_FOLDER}}_${{LABEL}}_${{MODE}}_Run_Stratified_Explanations_GRAE.out"

mkdir -p "$(dirname "${{LOG}}")" "${{OUTDIR}}" "${{BASE_DIR}}/slurm_logs"
exec > "${{LOG}}" 2>&1

echo "============================================================"
echo "XplainTDA SLURM array task"
echo "date              : $(date)"
echo "job id            : ${{SLURM_JOB_ID:-NA}}"
echo "array task id     : ${{SLURM_ARRAY_TASK_ID:-NA}}"
echo "dataset           : ${{DATASET}}"
echo "label             : ${{LABEL}}"
echo "omic              : ${{OMIC_NAME}} folder=${{OMIC_FOLDER}}"
echo "mode              : ${{MODE}}"
echo "cpus              : ${{SLURM_CPUS_PER_TASK}}"
echo "adata             : ${{ADATA}}"
echo "grae_model        : ${{GRAE_MODEL}}"
echo "outdir            : ${{OUTDIR}}"
echo "============================================================"

[[ -f "${{ADATA}}" ]] || {{ echo "ERROR: missing AnnData file: ${{ADATA}}"; exit 2; }}
[[ -f "${{GRAE_MODEL}}" ]] || {{ echo "ERROR: missing GRAE model: ${{GRAE_MODEL}}"; exit 2; }}
[[ -f "${{LAUNCHER}}" ]] || {{ echo "ERROR: missing launcher: ${{LAUNCHER}}"; exit 2; }}

if [[ "${{STRICT_INPUT_CHECK}}" == "1" ]]; then
    export ADATA EMBEDDING_KEY
    python - <<'INNERPY'
import os
try:
    import h5py
except Exception as exc:
    raise SystemExit(f"ERROR: strict h5ad check requires h5py: {{exc}}")

adata = os.environ["ADATA"]
key = os.environ["EMBEDDING_KEY"]
with h5py.File(adata, "r") as handle:
    ok = "obsm" in handle and key in handle["obsm"]
if not ok:
    raise SystemExit(f"ERROR: {{adata}} does not contain obsm/{{key}}. Refusing to submit a run that could refit GRAE.")
print(f"Confirmed h5ad contains obsm/{{key}}")
INNERPY
fi

# One worker per allocated CPU. defaults.py reads XPLAINTDA_N_JOBS first.
export XPLAINTDA_N_JOBS="${{SLURM_CPUS_PER_TASK}}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_NUM_THREADS=1
export TORCH_NUM_INTEROP_THREADS=1

# Keep this command within the old Run_Exp.sh $2 ... $9 forwarding limit.
bash "${{LAUNCHER}}" "${{SCRIPT}}" \
    "${{ADATA}}" \
    "${{OUTDIR}}" \
    "${{LABEL}}" \
    --embedding-key "${{EMBEDDING_KEY}}" \
    --grae-model-path "${{GRAE_MODEL}}" \
    "${{SKIP_FLAG}}"

echo "Completed ${{DATASET}} ${{OMIC_NAME}} ${{MODE}} at $(date)"
"""


def main() -> None:
    args = parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    for omic_name, omic_folder in OMICS.items():
        for mode in ("local", "global"):
            script_text = make_sbatch_script(args, omic_name, omic_folder, mode)
            script_path = workdir / f"submit_{omic_name}_{mode}.sbatch"
            script_path.write_text(script_text)
            cmd = ["sbatch", str(script_path)]
            print(" ".join(q(c) for c in cmd))
            if not args.dry_run:
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
