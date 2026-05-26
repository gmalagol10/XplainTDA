import os
import numpy as np
import pandas as pd
import time

from joblib import Parallel, delayed
from scipy.sparse import issparse
from scipy.stats import mannwhitneyu, wilcoxon


class X_scTDA:
    """
    Topology-aware perturbation analysis.

    This class perturbs genes directly in the
    PCA embedding space without recomputing PCA.
    """

    def __init__(
        self,
        adata,
        tda,
        marker_genes=None,
        target_dim=1,
        perturbation="zero",
        recompute_mode="exact",
        local_radius=2,
        n_random=20,
        top_frac=0.10,
        max_features=50,
        verbose=True
    ):

        self.adata = adata
        self.tda = tda
        self.top_frac = top_frac
        self.max_features = max_features
        self.X = adata.X

        self.target_dim = target_dim

        self.perturbation = perturbation

        self.recompute_mode = recompute_mode

        self.local_radius = local_radius

        self.n_random = n_random

        self.verbose = verbose

        # -------------------------------------------------
        # PCA loadings
        # -------------------------------------------------
        self.W = tda.pca.components_.T

        # -------------------------------------------------
        # Cache baseline persistence
        # -------------------------------------------------
        self.P0 = self.tda.max_persistence(
            self.tda.dgms[self.target_dim]
        )

        # -------------------------------------------------
        # Marker genes
        # -------------------------------------------------
        self.marker_genes = marker_genes or []

        self.marker_idx = (
            adata.var_names.get_indexer(
                self.marker_genes
            )
        )

        valid = self.marker_idx >= 0

        self.marker_idx = self.marker_idx[valid]

        self.marker_genes = list(
            np.array(self.marker_genes)[valid]
        )

    # =====================================================
    # Embedding perturbation
    # =====================================================
    def perturb_embedding(
        self,
        gene_idx,
        cell_idx
    ):
        """
        Perturb one gene directly in embedding space.
        """
    
        Zp = self.tda.embedding.copy()
    
        w = self.W[gene_idx].astype(np.float32)
    
        # ---------------------------------------------
        # Sparse-safe extraction
        # ---------------------------------------------
        vals = self.X[cell_idx, gene_idx]
    
        if issparse(vals):
    
            vals = vals.toarray().ravel()
    
        else:
    
            vals = np.asarray(vals).ravel()
    
        vals = vals.astype(np.float32)
    
        # ---------------------------------------------
        # Perturbation
        # ---------------------------------------------
        if self.perturbation == "zero":
    
            delta = -vals
    
        elif self.perturbation == "permute":
    
            perm = vals.copy()
    
            np.random.shuffle(perm)
    
            delta = perm - vals
    
        else:
    
            raise ValueError(
                "perturbation must be "
                "'zero' or 'permute'"
            )
    
        delta = delta.astype(np.float32)
    
        # ---------------------------------------------
        # Rank-1 embedding update
        # ---------------------------------------------
        Zp[cell_idx] += (
            delta[:, None]
            *
            w[None, :]
        )
    
        return Zp

    # =====================================================
    # Local recomputation
    # =====================================================
    def local_recompute_ph(
        self,
        Zp,
        cell_idx
    ):
        """
        Approximate PH recomputation.

        PH is recomputed only on a local
        neighborhood around perturbed cells.
        """

        affected = set(cell_idx)

        neigh = self.tda.neigh_ind

        frontier = set(cell_idx)

        for _ in range(self.local_radius):

            new_frontier = set()

            for c in frontier:

                new_frontier.update(
                    neigh[c]
                )

            affected.update(new_frontier)

            frontier = new_frontier

        affected = np.array(
            sorted(affected)
        )

        Z_local = Zp[affected]

        return self.tda.recompute_ph(
            Z_local
        )

    # =====================================================
    # Gene importance
    # =====================================================
    def gene_importance(
        self,
        gene_idx,
        cell_idx
    ):
        """
        Topological importance score
        for one gene.
        """

        Zp = self.perturb_embedding(
            gene_idx,
            cell_idx
        )

        # -------------------------------------------------
        # Exact or local PH recomputation
        # -------------------------------------------------
        if self.recompute_mode == "exact":

            dgms_p = self.tda.recompute_ph(
                Zp
            )

        elif self.recompute_mode == "local":

            dgms_p = self.local_recompute_ph(
                Zp,
                cell_idx
            )

        else:

            raise ValueError(
                "recompute_mode must be "
                "'exact' or 'local'"
            )

        P1 = self.tda.max_persistence(
            dgms_p[self.target_dim]
        )

        return (
            self.P0 - P1
        ) / (
            self.P0 + 1e-12
        )

    # =====================================================
    # Worker for one topological feature
    # =====================================================
    def process_feature(
        self,
        feature,
        genes
    ):
        """
        Compute scores for all genes
        for one topological feature.
        """

        feature_id = feature["feature_id"]

        cell_idx = feature["cells"]

        scores = np.zeros(
            len(genes),
            dtype=np.float32
        )

        for i, g in enumerate(genes):

            scores[i] = self.gene_importance(
                gene_idx=g,
                cell_idx=cell_idx
            )
        if self.verbose:
            print(f"Done feature {feature_id} | "f"{len(cell_idx)} cells", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), flush=True)
        return feature_id, scores

    # =====================================================
    # Batch gene screening
    # =====================================================
    def compute_feature_scores(
        self,
        genes=None,
        n_jobs=None
    ):
        """
        Compute topology importance scores.

        Parallelization strategy:
            - parallel over topological features
            - sequential over genes

        This is substantially more memory-safe
        than spawning one process per gene.
        """

        if genes is None:

            genes = np.arange(
                self.tda.n_genes
            )

        genes = np.asarray(genes)

        if n_jobs is None:
            n_jobs = min(int(os.environ.get("SLURM_CPUS_PER_TASK",1)) , len(self.tda.features[self.target_dim]))
            print(f"Launching {n_jobs} jobs", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), flush=True)

        # -------------------------------------------------
        # Select top persistent features
        # -------------------------------------------------
        features = self.tda.features[self.target_dim]

        # remove infinite persistence
        features = [

            f for f in features

            if np.isfinite(
                f["persistence"]
            )
        ]

        # sort by persistence
        features = sorted(
            features,
            key=lambda x: x["persistence"],
            reverse=True
        )

        # top fraction
        n_select = int(
            np.ceil(
                self.top_frac
                *
                len(features)
            )
        )

        # cap
        n_select = min(
            n_select,
            self.max_features
        )

        features = features[:n_select]
        print(f"I am examining {len(features)} topological features", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), flush=True)

        # -------------------------------------------------
        # Use threading backend:
        # avoids duplicating large topology objects
        # across processes.
        # -------------------------------------------------
        out = Parallel(
            n_jobs=n_jobs,
            backend="loky"
        )(
            delayed(self.process_feature)(
                feature,
                genes
            )
            for feature in features
        )

        self.results = {
            feature_id: scores
            for feature_id, scores in out
        }

        return self.results

    # =====================================================
    # Marker vs random comparison
    # =====================================================
    def marker_vs_random(self):
        """
        Compare marker gene importance
        against random genes.
        """

        marker_scores = np.concatenate([
            self.results[f][self.marker_idx]
            for f in self.results
        ])

        random_scores = []

        all_genes = np.arange(
            self.tda.n_genes
        )

        for _ in range(self.n_random):

            rand_idx = np.random.choice(
                all_genes,
                size=len(self.marker_idx),
                replace=False
            )

            for f in self.results:

                random_scores.extend(
                    self.results[f][rand_idx]
                )

        random_scores = np.array(
            random_scores
        )

        stat, pval = mannwhitneyu(
            marker_scores,
            random_scores,
            alternative="greater"
        )

        return {

            "statistic": stat,

            "pvalue": pval,

            "marker_scores": marker_scores,

            "random_scores": random_scores
        }

    # =====================================================
    # Rank enrichment
    # =====================================================
    def marker_rank_enrichment(self):
        """
        Test whether marker genes tend
        to have high topology importance.
        """

        rows = []

        for feature_id, scores in (
            self.results.items()
        ):

            percentiles = (

                scores.argsort()
                .argsort()

                / len(scores)
            )

            for g, gene in zip(
                self.marker_idx,
                self.marker_genes
            ):

                rows.append({

                    "feature_id":
                    feature_id,

                    "gene":
                    gene,

                    "percentile":
                    percentiles[g]
                })

        df = pd.DataFrame(rows)

        stat, pval = wilcoxon(
            df["percentile"] - 0.5,
            alternative="greater"
        )

        return {

            "statistic": stat,

            "pvalue": pval,

            "summary": df
        }
