import os, sys, time
import numpy as np, pandas as pd
from joblib import load
from concurrent.futures import ProcessPoolExecutor

# =========================================================
# Arguments
# =========================================================

OUTDIR, GENOME, TOP_P = sys.argv[1], sys.argv[2], int(sys.argv[3])

print("=" * 60, flush=True)
print("TDA peak motif analysis", flush=True)
print(f"OUTDIR={OUTDIR} | GENOME={GENOME} | TOP_P={TOP_P}", flush=True)

# =========================================================
# Load TDA
# =========================================================

tda_file = [f"{OUTDIR}/{f}" for f in os.listdir(OUTDIR) if f.endswith("_H_tda.joblib")][0]

print(f"\nLoading TDA: {tda_file}", flush=True)

tda = load(tda_file)
peaks_all = np.array(tda.adata.var_names)

print(f"N cells={tda.n_cells} | N peaks={tda.n_genes}", flush=True)
print(f"Example peak={peaks_all[0]}", flush=True)

# =========================================================
# Worker
# =========================================================

def run_homer(args):

    H, feature_id, scores = args

    print(f"\n[H{H}] Feature {feature_id} | ranking peaks", flush=True)

    peaks = peaks_all[np.argsort(scores)[::-1][:TOP_P]]

    homer_dir = f"{OUTDIR}/HOMER/H_{H}/feature_{feature_id}"
    os.makedirs(homer_dir, exist_ok=True)

    bed = f"{homer_dir}/peaks.bed"

    pd.DataFrame([
        p.replace(":", "_").replace("-", "_").split("_")[:3]
        for p in peaks
    ]).to_csv(bed, sep="\t", header=False, index=False)

    print(f"[H{H}] Feature {feature_id} | {len(peaks)} peaks", flush=True)
    print(f"[H{H}] Feature {feature_id} | running HOMER", flush=True)

    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    cmd = f"findMotifsGenome.pl {bed} {GENOME} {homer_dir} -p {n_jobs}"
    os.system(cmd)

    print(f"[H{H}] Feature {feature_id} | completed", flush=True)

# =========================================================
# Main
# =========================================================

for H in [0,1,2]:

    print("\n" + "=" * 60, flush=True)
    print(f"Processing H{H}", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), flush=True)

    results_file = f"{OUTDIR}/" + [f for f in os.listdir(OUTDIR) if f"_H{H}_results.joblib" in f][0]
    results = load(results_file)

    print(f"H{H} contains {len(results)} topological features", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), flush=True)

    for fid, scores in results.items():

        print(f"\nRunning H{H} | feature {fid}", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), flush=True)

        run_homer((H, fid, scores))


print("\nPipeline completed.", flush=True)
