import os, time, joblib

import numpy as np
import pandas as pd
from scipy.sparse import issparse
import defaults as d

VALID_PERTURBATIONS = {"zero", "permute"}
VALID_RECOMPUTE_MODES = {"exact", "local"}
VALID_MATCHING_MODES = {"max_persistence", "match_cells"}


# =====================================================
# Reusable perturbation utilities
# =====================================================
def gene_values(X, cell_idx, gene_idx, dtype=np.float32):
    """
    Extract the values of one gene on selected cells.

    Parameters
    ----------
    X : array-like or sparse matrix
        Expression matrix used by the fitted XplainTDA object.
    cell_idx : array-like
        Indices of cells to perturb.
    gene_idx : int
        Index of the gene to perturb.
    dtype : dtype, default=np.float32
        Output dtype.

    Returns
    -------
    np.ndarray
        Dense one-dimensional vector of gene values.
    """
    vals = X[cell_idx, gene_idx]

    # Sparse slices need explicit densification before raveling.
    if issparse(vals):
        vals = vals.toarray().ravel()
    else:
        vals = np.asarray(vals).ravel()

    return vals.astype(dtype, copy=False)


def perturb_values(vals, perturbation, rng=None):
    """
    Perturb a one-dimensional vector of gene values.

    Parameters
    ----------
    vals : np.ndarray
        Original gene values on selected cells.
    perturbation : {"zero", "permute"}
        Perturbation rule.
    rng : np.random.Generator, optional
        Random generator used for permutation.

    Returns
    -------
    np.ndarray
        Perturbed gene values.
    """
    if perturbation == "zero":
        return np.zeros_like(vals)

    if perturbation == "permute":
        vals_p = vals.copy()
        rng = np.random.default_rng() if rng is None else rng

        # Shuffle only within the selected cells.
        rng.shuffle(vals_p)

        return vals_p

    raise ValueError(
        f"perturbation must be one of {sorted(VALID_PERTURBATIONS)}"
    )


def assign_gene_values(X, cell_idx, gene_idx, vals):
    """
    Return a copy of X with one gene replaced on selected cells.

    This is used by nonlinear embedders, where the whole perturbed
    expression matrix must be passed through the trained embedder.
    """
    Xp = X.copy()

    # LIL is efficient and safe for sparse item assignment.
    if issparse(Xp):
        Xp = Xp.tolil(copy=False)
        Xp[cell_idx, gene_idx] = vals

        return Xp.tocsr()

    Xp[cell_idx, gene_idx] = vals

    return Xp


def linear_loadings(tda):
    """
    Return linear embedding loadings if the fitted embedder exposes them.

    Returns
    -------
    np.ndarray or None
        Matrix W = components_.T for PCA/TruncatedSVD-like embedders.
        Returns None for nonlinear embedders such as GRAE.
    """
    embedder = getattr(tda, "embedder", None)
    model = getattr(embedder, "model", None)

    if not hasattr(model, "components_"):
        return None

    return model.components_.T.astype(np.float32, copy=False)


def perturb_embedding(tda, X, gene_idx, cell_idx, perturbation="zero", rng=None):
    """
    Perturb one gene and return the resulting embedding.

    For linear embedders with ``components_`` such as PCA or TruncatedSVD,
    this uses the exact rank-1 update in embedding space:

        Zp = Z + delta_x_g W_g

    For nonlinear embedders such as GRAE, it perturbs X and calls the
    already-trained embedder through ``tda.embed(Xp)``. The embedder is not
    refitted.
    """
    vals = gene_values(X, cell_idx, gene_idx)
    vals_p = perturb_values(vals, perturbation, rng)
    delta = vals_p - vals

    W = linear_loadings(tda)

    if W is not None:
        Zp = tda.embedding.copy()

        # Exact linear propagation of the gene-level perturbation.
        Zp[cell_idx] += delta[:, None] * W[gene_idx][None, :]

        return Zp

    # Generic path: perturb X, then transform with the trained embedder.
    Xp = assign_gene_values(X, cell_idx, gene_idx, vals_p)

    return tda.embed(Xp)


def jaccard(a, b):
    """
    Compute Jaccard similarity between two collections of cell indices.
    """
    a = set(a)
    b = set(b)

    if len(a | b) == 0:
        return 0.0

    return len(a & b) / len(a | b)


# =====================================================
# Local topology explanation
# =====================================================
class Local_XplainTDA:
    """
    Local topology-aware gene perturbation analysis.

    The class scores genes with respect to localized persistent homology
    features. For each selected topological feature, genes are perturbed only
    on the cells assigned to that feature. Persistent homology is then
    recomputed either on the full perturbed embedding or on a local kNN
    neighborhood around the perturbed cells.

    Embedding perturbation is delegated to module-level utilities so that the
    same logic can also be reused by global explainers. Linear embedders use a
    fast exact rank-1 update. Nonlinear embedders reuse the already-trained
    embedder through ``tda.embed``.
    """

    def __init__(
        self,
        tda,
        target_dim=d.DEFAULT_TARGET_DIM,
        perturbation=d.DEFAULT_PERTURBATION,
        recompute_mode=d.DEFAULT_RECOMPUTE_MODE,
        local_radius=d.DEFAULT_LOCAL_RADIUS,
        top_frac=d.DEFAULT_TOP_FRAC,
        max_features=d.DEFAULT_MAX_FEATURES,
        matching_mode=d.DEFAULT_MATCHING_MODE,
        jaccard_threshold=d.DEFAULT_JACCARD_THRESHOLD,
        verbose=d.DEFAULT_VERBOSE,
        random_state=d.DEFAULT_RANDOM_STATE,
    ):
        """
        Initialize local topological gene perturbation analysis.

        Parameters
        ----------
        tda : XplainTDA
            Fitted topology object containing X, embedding, kNN graph,
            persistence diagrams, and localized features.
        target_dim : int, default=1
            Default homology dimension to explain.
        perturbation : {"zero", "permute"}, default="zero"
            Gene perturbation rule.
        recompute_mode : {"exact", "local"}, default="exact"
            Whether PH is recomputed on the full embedding or on a local
            neighborhood around perturbed cells.
        local_radius : int, default=2
            Number of kNN expansion steps used in local recomputation.
        top_frac : float, default=0.10
            Fraction of most persistent features to analyze.
        max_features : int, default=50
            Maximum number of features to analyze in each homology group.
        matching_mode : {"max_persistence", "match_cells"}, default="max_persistence"
            Rule used to select the perturbed persistence value.
        jaccard_threshold : float, default=0.3
            Minimum cell-overlap threshold for ``match_cells``.
        verbose : bool, default=True
            Whether to print progress messages.
        random_state : int, optional
            Seed for reproducible permutation perturbations.
        """
        self._validate_init(
            perturbation,
            recompute_mode,
            matching_mode
        )

        self.adata = tda.adata
        self.tda = tda
        self.X = tda.X
        self.target_dim = target_dim
        self.perturbation = perturbation
        self.recompute_mode = recompute_mode
        self.local_radius = local_radius
        self.top_frac = top_frac
        self.max_features = max_features
        self.matching_mode = matching_mode
        self.jaccard_threshold = jaccard_threshold
        self.verbose = verbose
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.results = None
        self._feature_cache = {}
        self._score_cache = {}

    # =====================================================
    # Validation and utilities
    # =====================================================
    @staticmethod
    def _validate_init(perturbation, recompute_mode, matching_mode):
        """
        Validate categorical initialization arguments.
        """
        if perturbation not in VALID_PERTURBATIONS:
            raise ValueError(
                f"perturbation must be one of {sorted(VALID_PERTURBATIONS)}"
            )

        if recompute_mode not in VALID_RECOMPUTE_MODES:
            raise ValueError(
                f"recompute_mode must be one of {sorted(VALID_RECOMPUTE_MODES)}"
            )

        if matching_mode not in VALID_MATCHING_MODES:
            raise ValueError(
                f"matching_mode must be one of {sorted(VALID_MATCHING_MODES)}"
            )

    def _available_dims(self):
        """Return all homology dimensions available in the fitted TDA object."""
        return sorted(self.tda.features.keys())


    def _dim(self, target_dim=None):
        """Resolve one homology dimension for single-dimension methods."""
        dim = self.target_dim if target_dim is None else target_dim
        if dim is None:
            raise ValueError(
                "target_dim must be specified for single-dimension methods. "
                "Use importance_all_homologies() to score all dimensions."
            )
        return int(dim)


    def _dims(self, dims=None):
        """Resolve one or more homology dimensions."""
        if dims is None:
            return self._available_dims()
        return [int(x) for x in np.atleast_1d(dims)]

    def _resolve_genes(self, genes=None):
        """
        Convert gene names or indices to aligned gene indices and names.

        Parameters
        ----------
        genes : None, int, str, or array-like
            If None, all genes are used. Strings are matched to
            ``adata.var_names``.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Gene indices and corresponding gene names.
        """
        if genes is None:
            gene_idx = np.arange(self.tda.n_genes, dtype=int)
            gene_names = np.asarray(self.adata.var_names)[gene_idx]

            return gene_idx, gene_names

        genes = [genes] if np.isscalar(genes) or isinstance(genes, str) else list(genes)

        if len(genes) == 0:
            return np.array([], dtype=int), np.array([], dtype=str)

        if isinstance(genes[0], str):
            gene_idx = self.adata.var_names.get_indexer(genes)
            missing = np.asarray(genes)[gene_idx < 0]

            if len(missing):
                raise ValueError(f"Unknown genes: {missing.tolist()}")

            gene_names = np.asarray(genes)

            return gene_idx.astype(int), gene_names

        gene_idx = np.asarray(genes, dtype=int)
        gene_names = np.asarray(self.adata.var_names)[gene_idx]

        return gene_idx, gene_names

    def _score_cache_key(self, target_dim, genes):
        """
        Build a cache key for deterministic score matrices.
        """
        return (
            int(target_dim),
            tuple(np.asarray(genes, dtype=int).tolist()),
            float(self.top_frac),
            int(self.max_features),
            self.perturbation,
            self.recompute_mode,
            int(self.local_radius),
            self.matching_mode,
            float(self.jaccard_threshold)
        )

    # =====================================================
    # Embedding perturbation
    # =====================================================
    def perturb_embedding(self, gene_idx, cell_idx):
        """
        Return the perturbed embedding for one gene and one cell set.

        This method is intentionally thin: the implementation lives outside
        the class so it can be reused by future global perturbation classes.
        """
        return perturb_embedding(
            self.tda,
            self.X,
            gene_idx,
            cell_idx,
            self.perturbation,
            self.rng
        )

    # =====================================================
    # Feature matching
    # =====================================================
    def match_feature(self, original_cells, perturbed_features):
        """
        Match a perturbed feature to the original one by cell overlap.

        Parameters
        ----------
        original_cells : array-like
            Global cell indices of the original localized feature.
        perturbed_features : list[dict]
            Features returned after PH recomputation.

        Returns
        -------
        float
            Persistence of the best-overlapping perturbed feature. Returns
            zero if the best overlap is below ``jaccard_threshold``.
        """
        best_persistence = 0.0
        best_overlap = 0.0

        for feat in perturbed_features:
            overlap = jaccard(original_cells, feat["cells"])

            # Keep the feature with maximal cell-set overlap.
            if overlap > best_overlap:
                best_overlap = overlap
                best_persistence = feat["persistence"]

        if best_overlap < self.jaccard_threshold:
            return 0.0

        return best_persistence

    # =====================================================
    # Local recomputation
    # =====================================================
    def local_recompute_ph(self, Zp, cell_idx):
        """
        Recompute PH on a local neighborhood around perturbed cells.

        The neighborhood is built by expanding through the original kNN graph.
        Since PH is recomputed on ``Zp[affected]``, feature cell IDs returned
        by ``tda.recompute_ph`` are local indices. They are mapped back to
        global cell IDs before returning, so ``match_cells`` remains valid.
        """
        affected = set(cell_idx)
        frontier = set(cell_idx)
        neigh = self.tda.neigh_ind

        for _ in range(self.local_radius):
            new_frontier = set()

            for c in frontier:
                new_frontier.update(neigh[c])

            # Do not revisit cells already included in the local subproblem.
            new_frontier -= affected

            if not new_frontier:
                break

            affected.update(new_frontier)
            frontier = new_frontier

        affected = np.array(sorted(affected))

        # PH is recomputed only on the affected local embedding.
        dgms, features = self.tda.recompute_ph(Zp[affected])

        # Convert local feature cell IDs back to global cell IDs.
        for feats in features.values():
            for feat in feats:
                feat["cells"] = affected[feat["cells"]]

        return dgms, features

    # =====================================================
    # Gene importance
    # =====================================================
    def gene_importance(self, gene_idx, feature, target_dim=None):
        """
        Compute topological importance of one gene for one feature.

        Importance is the relative persistence loss:

            score = (P0 - P1) / P0

        where P0 is the original feature persistence and P1 is the selected
        persistence after perturbing the gene on the feature-localized cells.
        """
        target_dim = self._dim(target_dim)
        cell_idx = feature["cells"]
        P0 = feature["persistence"]

        if len(cell_idx) == 0 or P0 <= 0:
            return 0.0

        Zp = self.perturb_embedding(gene_idx, cell_idx)

        # Recompute PH either globally or on the local affected neighborhood.
        if self.recompute_mode == "exact":
            dgms_p, features_p = self.tda.recompute_ph(Zp)
        else:
            dgms_p, features_p = self.local_recompute_ph(Zp, cell_idx)

        # Select the perturbed persistence according to the matching rule.
        if self.matching_mode == "max_persistence":
            P1 = self.tda.max_persistence(dgms_p[target_dim])
        else:
            P1 = self.match_feature(cell_idx, features_p[target_dim])

        return (P0 - P1) / (P0 + 1e-12)

    # =====================================================
    # Feature worker
    # =====================================================
    def process_feature(self, feature, genes, target_dim=None):
        """
        Compute all gene scores for one localized topological feature.

        This is the unit of parallel work used by ``compute_feature_scores``.
        Genes are processed sequentially within each feature to reduce memory
        pressure from repeatedly copying embeddings or expression matrices.
        """
        target_dim = self._dim(target_dim)
        feature_id = feature["feature_id"]
        cell_idx = feature["cells"]

        scores = np.zeros(len(genes), dtype=np.float32)

        for i, gene_idx in enumerate(genes):
            scores[i] = self.gene_importance(gene_idx, feature, target_dim)

        if self.verbose:
            now = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
            print(f"Done H{target_dim} feature {feature_id} | {len(cell_idx)} cells", now, flush=True)

        return feature_id, scores

    # =====================================================
    # Feature selection
    # =====================================================
    def selected_features(self, target_dim=None):
        """
        Select the finite, non-empty, most persistent features to analyze.

        Parameters
        ----------
        target_dim : int, optional
            Homology dimension. If None, uses ``self.target_dim``.

        Returns
        -------
        list[dict]
            Feature dictionaries from ``tda.features[target_dim]``.
        """
        target_dim = self._dim(target_dim)
        key = (int(target_dim), float(self.top_frac), int(self.max_features))

        if key in self._feature_cache:
            return self._feature_cache[key]

        # Infinite or empty features are not useful for finite perturbation scores.
        features = [
            f for f in self.tda.features[target_dim]
            if np.isfinite(f["persistence"]) and len(f["cells"]) > 0
        ]

        features = sorted(
            features,
            key=lambda f: f["persistence"],
            reverse=True
        )

        n_select = int(np.ceil(self.top_frac * len(features)))
        n_select = min(n_select, self.max_features)
        features = features[:n_select]

        self._feature_cache[key] = features

        return features

    # =====================================================
    # Low-level batch scoring
    # =====================================================
    def compute_feature_scores(
        self,
        genes=None,
        target_dim=None,
        n_jobs=None,
        backend="threading",
        cache=None
    ):
        """
        Compute local topology-importance scores for one homology dimension.

        This is the low-level engine. It returns a dictionary mapping each
        selected feature ID to a vector of scores aligned with ``genes``.
        The user-facing methods below convert this output to tidy DataFrames.

        Parameters
        ----------
        genes : array-like, optional
            Gene indices to score. If None, all genes are used.
        target_dim : int, optional
            Homology dimension to score. If None, uses ``self.target_dim``.
        n_jobs : int, optional
            Number of parallel jobs. If None, uses ``SLURM_CPUS_PER_TASK``
            capped by the number of selected features.
        backend : str, default="threading"
            Joblib backend. Threading avoids duplicating the large TDA object
            across processes. ``loky`` can be tested if memory allows it.
        cache : bool, optional
            Whether to reuse cached score matrices. If None, caching is used
            only for deterministic ``zero`` perturbations.

        Returns
        -------
        dict
            Mapping ``feature_id -> np.ndarray`` of gene scores.
        """
        target_dim = self._dim(target_dim)
        gene_idx, _ = self._resolve_genes(genes)
        use_cache = self.perturbation == "zero" if cache is None else cache
        key = self._score_cache_key(target_dim, gene_idx)

        if use_cache and key in self._score_cache:
            self.results = self._score_cache[key]

            return self.results

        features = self.selected_features(target_dim)

        if len(features) == 0 or len(gene_idx) == 0:
            self.results = {}

            return self.results
            
        if n_jobs is None:
            n_jobs = min(d.DEFAULT_N_JOBS, len(features))

        if self.verbose:
            now = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
            print(f"Launching {n_jobs} jobs | H{target_dim} | {len(features)} features", now, flush=True)

        # Parallelize over features; genes remain sequential inside each worker.
        out = joblib.Parallel(n_jobs=n_jobs, backend=backend)(
            joblib.delayed(self.process_feature)(feature, gene_idx, target_dim)
            for feature in features
        )

        self.results = dict(out)

        if use_cache:
            self._score_cache[key] = self.results

        return self.results

    # =====================================================
    # DataFrame formatting
    # =====================================================
    def scores_to_dataframe(self, scores, genes, gene_names, target_dim):
        """
        Convert a feature-score dictionary to a tidy long DataFrame.

        Each row corresponds to one H-feature-gene score. Rank is computed
        within each feature, so rank 1 is the highest-scoring gene for that
        feature.
        """
        rows = []
        features = {f["feature_id"]: f for f in self.selected_features(target_dim)}

        for feature_id, feature_scores in scores.items():
            order = np.argsort(-feature_scores)
            feature = features[feature_id]

            for rank, pos in enumerate(order, start=1):
                rows.append({
                    "homology_dim": target_dim,
                    "feature_id": feature_id,
                    "feature_persistence": feature["persistence"],
                    "rank_within_feature": rank,
                    "gene_idx": int(genes[pos]),
                    "gene": gene_names[pos],
                    "score": float(feature_scores[pos])
                })

        return pd.DataFrame(rows)

    # =====================================================
    # User-facing method 1
    # =====================================================
    def importance_for_genes(
        self,
        genes,
        target_dim=None,
        n_jobs=None,
        backend="threading",
        cache=None
    ):
        """
        Score one gene or a set of genes for top features in one H group.

        Parameters
        ----------
        genes : int, str, or array-like
            Gene index/name, or a list of gene indices/names.
        target_dim : int, optional
            Homology dimension. If None, uses ``self.target_dim``.
        n_jobs : int, optional
            Number of parallel jobs over selected features.
        backend : str, default="threading"
            Joblib backend.
        cache : bool, optional
            Whether to reuse cached scores.

        Returns
        -------
        pd.DataFrame
            Tidy table with importance of the requested genes for the selected
            top features in the chosen homology dimension.
        """
        target_dim = self._dim(target_dim)
        gene_idx, gene_names = self._resolve_genes(genes)

        scores = self.compute_feature_scores(
            genes=gene_idx,
            target_dim=target_dim,
            n_jobs=n_jobs,
            backend=backend,
            cache=cache
        )

        return self.scores_to_dataframe(
            scores,
            gene_idx,
            gene_names,
            target_dim
        )

    # =====================================================
    # User-facing method 2
    # =====================================================
    def importance_all_homologies(
        self,
        genes=None,
        dims=None,
        n_jobs=None,
        backend="threading",
        cache=None
    ):
        """
        Compute the complete H-feature-gene importance table.

        Parameters
        ----------
        genes : array-like, optional
            Gene indices or names. If None, all genes are scored.
        dims : array-like, optional
            Homology dimensions to score. If None, all dimensions available in
            ``tda.features`` are used.
        n_jobs : int, optional
            Number of parallel jobs over selected features within each H group.
        backend : str, default="threading"
            Joblib backend.
        cache : bool, optional
            Whether to reuse cached scores.

        Returns
        -------
        pd.DataFrame
            Tidy table containing H, selected feature, gene, and score.
        """
        dims = self._dims(dims)

        tables = [
            self.importance_by_homology(
                target_dim=dim,
                genes=genes,
                n_jobs=n_jobs,
                backend=backend,
                cache=cache
            )
            for dim in dims
        ]

        if len(tables) == 0:
            return pd.DataFrame()

        return pd.concat(tables, ignore_index=True)

    # =====================================================
    # User-facing method 3
    # =====================================================
    def importance_by_homology(
        self,
        target_dim,
        genes=None,
        n_jobs=None,
        backend="threading",
        cache=None
    ):
        """
        Compute the selected feature-gene importance table for one H group.

        This is the main user-facing method when a single homology dimension
        is of interest. Feature selection is always capped by ``top_frac`` and
        ``max_features``.

        Parameters
        ----------
        target_dim : int
            Homology dimension to score.
        genes : array-like, optional
            Gene indices or names. If None, all genes are scored.
        n_jobs : int, optional
            Number of parallel jobs over selected features.
        backend : str, default="threading"
            Joblib backend.
        cache : bool, optional
            Whether to reuse cached scores.

        Returns
        -------
        pd.DataFrame
            Tidy table with one row per selected feature-gene pair.
        """
        gene_idx, gene_names = self._resolve_genes(genes)

        scores = self.compute_feature_scores(
            genes=gene_idx,
            target_dim=target_dim,
            n_jobs=n_jobs,
            backend=backend,
            cache=cache
        )

        return self.scores_to_dataframe(
            scores,
            gene_idx,
            gene_names,
            target_dim
        )
        
        
    def save(self, path, save_tda=False, save_cache=True, compress=3):
        """
        Save the local explainer configuration and optional fitted state.

        Parameters
        ----------
        path : str
            Output path, usually ending in ``.joblib``.
        save_tda : bool, default=False
            Whether to include the fitted ``XplainTDA`` object. Keeping this
            False avoids duplicating a large object that may already be saved
            separately with ``XplainTDA.save``.
        save_cache : bool, default=True
            Whether to save computed score and feature caches.
        compress : int, default=3
            Joblib compression level. Use zero for faster uncompressed saving.
        """
        state = {
            "config": self._config(),
            "tda": self.tda if save_tda else None,
            "results": self.results if save_cache else None,
            "feature_cache": self._feature_cache if save_cache else {},
            "score_cache": self._score_cache if save_cache else {},
            "rng": self.rng}

        joblib.dump(state, path, compress=compress)

    @classmethod
    def load(cls, path, tda=None):
        """
        Load a saved local explainer.

        Parameters
        ----------
        path : str
            Path to a saved local explainer joblib file.
        tda : XplainTDA, optional
            Fitted topology object to attach if the explainer was saved with
            ``save_tda=False``.

        Returns
        -------
        obj : Local_XplainTDA
            Loaded local explainer.
        """
        state = joblib.load(path)

        saved_tda = state.get("tda", None)
        tda = tda if tda is not None else saved_tda

        if tda is None:
            raise ValueError(
                "No TDA object available. Pass `tda=...` or save with `save_tda=True`."
            )

        obj = cls(
            tda=tda,
            **state["config"]
        )

        obj.results = state.get("results", None)
        obj._feature_cache = state.get("feature_cache", {})
        obj._score_cache = state.get("score_cache", {})
        obj.rng = state.get("rng", obj.rng)

        return obj

    def _config(self):
        """
        Return constructor arguments needed to recreate this explainer.

        Returns
        -------
        config : dict
            Serializable configuration dictionary.
        """
        return {
            "target_dim": self.target_dim,
            "perturbation": self.perturbation,
            "recompute_mode": self.recompute_mode,
            "local_radius": self.local_radius,
            "top_frac": self.top_frac,
            "max_features": self.max_features,
            "matching_mode": self.matching_mode,
            "jaccard_threshold": self.jaccard_threshold,
            "verbose": self.verbose,
            "random_state": self.random_state}
