import os, time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from gudhi.wasserstein import wasserstein_distance
from gudhi.hera import bottleneck_distance
import scTDA

class Global_X_scTDA:

    def __init__(
        self,
        tda,
        n_pcs=30,
        n_neighbors=15,
        maxdim=2,
        layer="X",
        neighbor_backend="exact",
        wasserstein_order=2,
        internal_p=2,
        verbose=True
    ):

        self.tda = tda
        self.adata = tda.adata
        self.X     = tda.X
        self.w_order = wasserstein_order
        self.p       = internal_p
        self.verbose = verbose    
        self.base_dgms = self.cap_inf(
            self.tda.dgms,
            self.tda.edge_list)

    # =====================================================
    # Replace infinite deaths
    # =====================================================
    @staticmethod
    def cap_inf(dgms, edge_list):

        cap = edge_list[:,2].max()

        out = []

        for d in dgms:

            d = np.asarray(d).copy()

            if len(d):
                d[~np.isfinite(d[:,1]),1] = cap

            out.append(d)

        return out

    # =====================================================
    # Mask gene globally
    # =====================================================
    def mask_gene(self, g):

        ad = self.adata.copy()

        ad.X = self.X.copy()

        ad.X[:,g] = 0

        return ad

    # =====================================================
    # Process one gene
    # =====================================================
    def process_gene(self, g):

        t0   = time.time()
        gene = self.adata.var_names[g]

        if self.verbose:
            print(f"Start {gene}", flush=True)

        tda = scTDA.scTDA(
            adata=self.mask_gene(g),
            n_pcs=self.tda.n_pcs,
            n_neighbors=self.tda.n_neighbors,
            maxdim=self.tda.maxdim,
            neighbor_backend=self.tda.neighbor_backend)

        dgms = self.cap_inf(tda.dgms, tda.edge_list)

        W = {
            k: wasserstein_distance(
                self.base_dgms[k],
                dgms[k],
                order=self.w_order,
                internal_p=self.p
            )
            for k in range(self.tda.maxdim + 1)
        }

        B = {
            k: bottleneck_distance(
                self.base_dgms[k],
                dgms[k]
            )
            for k in range(self.tda.maxdim + 1)
        }

        out = {

            "gene"  : gene,
            "idx"   : g,

            "wasserstein" : W,
            "bottleneck"  : B,

            "runtime" : time.time() - t0,

            "dgms" : dgms,

            "pca_var" : (
                tda.pca.explained_variance_ratio_
            ),

            "n_edges" : len(tda.edge_list)
        }

        if self.verbose:
            print(
                f"Done {gene} | "
                f"{out['runtime']:.2f} sec",
                flush=True
            )

        return out

    # =====================================================
    # Run all genes
    # =====================================================
    def compute_global_scores(
        self,
        genes=None,
        n_jobs=None
    ):

        genes = (
            np.arange(self.adata.n_vars)
            if genes is None else np.asarray(genes)
        )

        n_jobs = n_jobs or int(
            os.environ.get(
                "SLURM_CPUS_PER_TASK",
                1
            )
        )

        out = Parallel(
            n_jobs=n_jobs,
            backend="loky"
        )(
            delayed(self.process_gene)(g)
            for g in genes
        )

        self.results = {
            r["gene"]: r
            for r in out
        }

        rows = []

        for r in out:

            row = {
                "gene"    : r["gene"],
                "runtime" : r["runtime"]
            }

            for k in range(self.tda.maxdim + 1):

                row[f"W_H{k}"] = r["wasserstein"][k]
                row[f"B_H{k}"] = r["bottleneck"][k]

            rows.append(row)

        self.summary = (
            pd.DataFrame(rows)
            .sort_values("W_H1", ascending=False)
        )

        return self.summary
