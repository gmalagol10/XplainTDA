import os, time
import numpy as np
import pandas as pd

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
        matching_mode="max_persistence", #match_cells
        jaccard_threshold=0.3,
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
        self.matching_mode = matching_mode
        self.jaccard_threshold = jaccard_threshold

        # -------------------------------------------------
        # PCA loadings
        # -------------------------------------------------
        self.W = tda.pca.components_.T

        # -------------------------------------------------
        # Cache baseline persistence
        # -------------------------------------------------
        self.P0 = self.tda.max_persistence(self.tda.dgms[self.target_dim])

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
    # Jaccard similarity
    # =====================================================
    @staticmethod
    def jaccard(a, b):

        a = set(a)
        b = set(b)

        if len(a | b) == 0:
            return 0.0

        return len(a & b) / len(a | b)


    # =====================================================
    # Embedding perturbation
    # =====================================================
    def perturb_embedding(self, gene_idx, cell_idx):
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
    
            raise ValueError("perturbation must be 'zero' or 'permute'")
    
        delta = delta.astype(np.float32)

        # ---------------------------------------------
        # Rank-1 embedding update
        # ---------------------------------------------
        Zp[cell_idx] += delta[:, None]*w[None, :])
        
        return Zp

    # =====================================================
    # Match perturbed feature
    # =====================================================
    def match_feature(
        self,
        original_cells,
        perturbed_features
    ):
        """
        Match perturbed feature using
        Jaccard overlap between cells.
        """

        best_persistence = 0.0

        best_overlap = 0.0

        for feat in perturbed_features:

            overlap = self.jaccard(
                original_cells,
                feat["cells"]
            )

            if overlap > best_overlap:

                best_overlap = overlap

                best_persistence = (
                    feat["persistence"]
                )

        if best_overlap < self.jaccard_threshold:

            return 0.0

        return best_persistence


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
        feature
    ):
        """
        Topological importance score
        for one gene.
        """

        cell_idx = feature["cells"]

        P0 = feature["persistence"]

        Zp = self.perturb_embedding(
            gene_idx,
            cell_idx
        )

        # -------------------------------------------------
        # Exact or local PH recomputation
        # -------------------------------------------------
        if self.recompute_mode == "exact":

            dgms_p, features_p = (
                self.tda.recompute_ph(Zp)
            )

        elif self.recompute_mode == "local":

            dgms_p, features_p = (
                self.local_recompute_ph(
                    Zp,
                    cell_idx
                )
            )

        else:

            raise ValueError(
                "recompute_mode must be "
                "'exact' or 'local'"
            )

        # -------------------------------------------------
        # Persistence selection
        # -------------------------------------------------
        if self.matching_mode == "max_persistence":

            P1 = self.tda.max_persistence(
                dgms_p[self.target_dim]
            )

        elif self.matching_mode == "match_cells":

            P1 = self.match_feature(
                original_cells=cell_idx,
                perturbed_features=(
                    features_p[self.target_dim]
                )
            )

        else:

            raise ValueError(
                "matching_mode must be "
                "'max_persistence' or "
                "'jaccard'"
            )

        return (
            P0 - P1
        ) / (
            P0 + 1e-12
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

            scores[i] = self.gene_importance(gene_idx=g, feature=feature)
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
