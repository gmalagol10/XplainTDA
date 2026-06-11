import os
import time
import joblib

import numpy as np
import pandas as pd

try:
    import gudhi as gd
except ImportError:
    gd = None

try:
    from gudhi.wasserstein import wasserstein_distance
except Exception:
    wasserstein_distance = None

from Local_XplainTDA import VALID_PERTURBATIONS, perturb_embedding

from XplainTDA import (
    VALID_PH_BACKENDS,
    compute_neighbors,
    compute_ph_from_neighbors,
    neighbors_to_edge_list,
    edge_list_to_sparse_distance,
    ripser_sparse_to_diagrams,
)

import defaults as d


VALID_DISTANCES = {"bottleneck", "wasserstein"}


class Global_XplainTDA:
    """
    Global topology-aware gene perturbation analysis.

    For each gene, this class perturbs the gene across all cells, recomputes
    the full persistence diagram of the perturbed embedding, and scores the
    gene by the distance between the reference and perturbed diagrams.

    The default PH backend is ``ripser``. In this mode, diagrams are computed
    from a sparse kNN distance matrix built from the embedding. This avoids
    feature localization and avoids constructing a full GUDHI SimplexTree for
    every perturbed gene.

    The fitted ``XplainTDA`` object is the only required input. The AnnData
    object is accessed internally as ``tda.adata``.
    """

    def __init__(
        self,
        tda,
        target_dim=d.DEFAULT_TARGET_DIM,
        perturbation=d.DEFAULT_PERTURBATION,
        distance=d.DEFAULT_DISTANCE,
        ph_backend=d.DEFAULT_GLOBAL_PH_BACKEND,
        bottleneck_epsilon=d.DEFAULT_BOTTLENECK_EPSILON,
        wasserstein_order=d.DEFAULT_WASSERSTEIN_ORDER,
        wasserstein_internal_p=d.DEFAULT_WASSERSTEIN_INTERNAL_P,
        verbose=d.DEFAULT_VERBOSE,
        random_state=d.DEFAULT_RANDOM_STATE,
    ):
        """
        Initialize the global explainer.

        Parameters
        ----------
        tda : XplainTDA
            Fitted XplainTDA object. Must contain ``adata``, ``X``,
            ``embedding``, ``n_cells``, ``n_genes``, ``n_neighbors``,
            ``neighbor_backend``, and ``maxdim``.
        target_dim : int, optional
            Default homology dimension for single-dimension methods. If None,
            methods that require one dimension must receive ``target_dim``.
        perturbation : {"zero", "permute"}, default=defaults.DEFAULT_PERTURBATION
            Rule used to perturb gene expression across all cells.
        distance : {"bottleneck", "wasserstein"}, default=defaults.DEFAULT_DISTANCE
            Persistence-diagram distance used as the importance score.
        ph_backend : {"ripser", "gudhi"}, default="ripser"
            Backend used to recompute reference and perturbed diagrams.
            ``"ripser"`` computes diagrams from a sparse kNN distance matrix.
            ``"gudhi"`` uses the same graph-to-complex path as XplainTDA.
        bottleneck_epsilon : float, default=defaults.DEFAULT_BOTTLENECK_EPSILON
            Approximation parameter passed to GUDHI bottleneck distance.
            Use 0.0 for exact bottleneck distance.
        wasserstein_order : float, default=defaults.DEFAULT_WASSERSTEIN_ORDER
            Wasserstein order for diagram distance.
        wasserstein_internal_p : float, default=defaults.DEFAULT_WASSERSTEIN_INTERNAL_P
            Internal ground-metric order for Wasserstein distance.
        verbose : bool, default=defaults.DEFAULT_VERBOSE
            Whether to print progress messages.
        random_state : int, optional
            Random seed for reproducible permutation perturbations.
        """
        self._validate_init(
            perturbation=perturbation,
            distance=distance,
            ph_backend=ph_backend,
            bottleneck_epsilon=bottleneck_epsilon,
        )

        self.tda = tda
        self.adata = tda.adata
        self.X = tda.X

        self.target_dim = target_dim
        self.perturbation = perturbation
        self.distance = distance
        self.ph_backend = ph_backend.lower()
        self.bottleneck_epsilon = bottleneck_epsilon
        self.wasserstein_order = wasserstein_order
        self.wasserstein_internal_p = wasserstein_internal_p
        self.verbose = verbose
        self.random_state = random_state

        self.cell_idx = np.arange(tda.n_cells, dtype=int)
        self.results = None
        self._score_cache = {}

        # Important: reference diagrams must use the same PH backend as the
        # perturbed diagrams. Do not compare tda.dgms from GUDHI to perturbed
        # diagrams from Ripser.
        self.reference_dgms = self.compute_diagrams(self.tda.embedding)

    # =====================================================
    # Validation and dimension/gene utilities
    # =====================================================
    @staticmethod
    def _validate_init(perturbation, distance, ph_backend, bottleneck_epsilon):
        """
        Validate categorical and numerical constructor arguments.

        Parameters
        ----------
        perturbation : str
            Perturbation mode.
        distance : str
            Diagram-distance mode.
        ph_backend : str
            Persistent-homology backend.
        bottleneck_epsilon : float or None
            Bottleneck approximation parameter.

        Returns
        -------
        None
            Raises ValueError on invalid input.
        """
        if perturbation not in VALID_PERTURBATIONS:
            raise ValueError(
                f"perturbation must be one of {sorted(VALID_PERTURBATIONS)}"
            )

        if distance not in VALID_DISTANCES:
            raise ValueError(
                f"distance must be one of {sorted(VALID_DISTANCES)}"
            )

        if ph_backend.lower() not in VALID_PH_BACKENDS:
            raise ValueError(
                f"ph_backend must be one of {sorted(VALID_PH_BACKENDS)}"
            )

        if bottleneck_epsilon is not None and bottleneck_epsilon < 0:
            raise ValueError("bottleneck_epsilon must be non-negative or None")

    def _available_dims(self):
        """
        Return homology dimensions available from the reference diagrams.

        Returns
        -------
        list[int]
            Available dimensions, usually ``[0, ..., tda.maxdim]``.
        """
        return list(range(len(self.reference_dgms)))

    def _dim(self, target_dim=None):
        """
        Resolve a single homology dimension.

        Parameters
        ----------
        target_dim : int, optional
            Explicit dimension. If None, uses ``self.target_dim``.

        Returns
        -------
        int
            Resolved homology dimension.
        """
        dim = self.target_dim if target_dim is None else target_dim

        if dim is None:
            raise ValueError(
                "target_dim must be specified for this method. "
                "Use importance_all_homologies() to score all dimensions."
            )

        dim = int(dim)

        if dim < 0 or dim >= len(self.reference_dgms):
            raise ValueError(
                f"target_dim={dim} is not available. "
                f"Available dimensions are {self._available_dims()}."
            )

        return dim

    def _dims(self, dims=None, target_dim=None):
        """
        Resolve one or more homology dimensions.

        Parameters
        ----------
        dims : array-like, optional
            Dimensions to score.
        target_dim : int, optional
            Single dimension convenience argument.

        Returns
        -------
        list[int]
            Resolved homology dimensions.
        """
        if dims is not None:
            out = [int(x) for x in np.atleast_1d(dims)]
        elif target_dim is not None or self.target_dim is not None:
            out = [self._dim(target_dim)]
        else:
            out = self._available_dims()

        available = set(self._available_dims())
        bad = [dim for dim in out if dim not in available]

        if bad:
            raise ValueError(
                f"Unavailable dimensions: {bad}. "
                f"Available dimensions are {self._available_dims()}."
            )

        return out

    def _resolve_genes(self, genes=None):
        """
        Resolve gene names or indices to aligned indices and names.

        Parameters
        ----------
        genes : None, int, str, or array-like
            Genes to score. If None, all genes are used. Strings are matched
            against ``tda.adata.var_names``.

        Returns
        -------
        gene_idx : np.ndarray
            Integer gene indices.
        gene_names : np.ndarray
            Gene names aligned with ``gene_idx``.
        """
        if genes is None:
            gene_idx = np.arange(self.tda.n_genes, dtype=int)
            return gene_idx, np.asarray(self.adata.var_names)[gene_idx]

        genes = [genes] if np.isscalar(genes) or isinstance(genes, str) else list(genes)

        if len(genes) == 0:
            return np.array([], dtype=int), np.array([], dtype=str)

        if isinstance(genes[0], str):
            gene_idx = self.adata.var_names.get_indexer(genes)
            missing = np.asarray(genes)[gene_idx < 0]

            if len(missing):
                raise ValueError(f"Unknown genes: {missing.tolist()}")

            return gene_idx.astype(int), np.asarray(genes)

        gene_idx = np.asarray(genes, dtype=int)

        if np.any(gene_idx < 0) or np.any(gene_idx >= self.tda.n_genes):
            raise IndexError("gene indices must be between 0 and tda.n_genes - 1")

        return gene_idx, np.asarray(self.adata.var_names)[gene_idx]

    def _rng_for_gene(self, gene_idx):
        """
        Create a reproducible random generator for one gene.

        Parameters
        ----------
        gene_idx : int
            Gene index.

        Returns
        -------
        np.random.Generator
            Random generator used for permutation perturbations.
        """
        if self.random_state is None:
            return np.random.default_rng()

        return np.random.default_rng(int(self.random_state) + int(gene_idx))

    def _score_cache_key(self, dims, genes):
        """
        Build a cache key for deterministic score tables.

        Parameters
        ----------
        dims : list[int]
            Homology dimensions.
        genes : array-like
            Gene indices.

        Returns
        -------
        tuple
            Hashable cache key.
        """
        return (
            tuple(map(int, dims)),
            tuple(np.asarray(genes, dtype=int).tolist()),
            self.perturbation,
            self.distance,
            self.ph_backend,
            None if self.bottleneck_epsilon is None else float(self.bottleneck_epsilon),
            float(self.wasserstein_order),
            float(self.wasserstein_internal_p),
        )

    # =====================================================
    # PH recomputation
    # =====================================================
    def compute_diagrams(self, embedding):
        """
        Compute persistence diagrams from an embedding using ``self.ph_backend``.

        Parameters
        ----------
        embedding : np.ndarray
            Cell embedding.

        Returns
        -------
        list[np.ndarray]
            Persistence diagrams for dimensions ``0, ..., tda.maxdim``.

        Notes
        -----
        With ``ph_backend="ripser"``, this method computes a kNN graph from
        the embedding, converts it to a symmetric sparse distance matrix, and
        calls Ripser on that sparse distance matrix.

        With ``ph_backend="gudhi"``, this method uses the same neighbor graph
        and graph-to-complex construction as XplainTDA, but disables feature
        localization because global explanations only require diagrams.
        """
        Z = np.asarray(embedding, dtype=self.tda.dtype)

        neigh_ind, neigh_dist = compute_neighbors(
            Z,
            n_neighbors=self.tda.n_neighbors,
            backend=self.tda.neighbor_backend,
        )

        if self.ph_backend == "ripser":
            edge_list = neighbors_to_edge_list(neigh_ind, neigh_dist)
            D = edge_list_to_sparse_distance(edge_list, n=Z.shape[0])

            return ripser_sparse_to_diagrams(
                D,
                maxdim=self.tda.maxdim,
            )

        dgms, _, _ = compute_ph_from_neighbors(
            neigh_ind,
            neigh_dist,
            maxdim=self.tda.maxdim,
            ph_backend="gudhi",
            localize=False,
        )

        return dgms

    # =====================================================
    # Diagram distances
    # =====================================================
    @staticmethod
    def _clean_diagram(dgm):
        """
        Convert a diagram to a finite float array.

        Parameters
        ----------
        dgm : array-like
            Persistence diagram.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_points, 2)`` containing only finite points.
        """
        dgm = np.asarray(dgm, dtype=float).reshape(-1, 2)

        if len(dgm) == 0:
            return dgm

        return dgm[np.isfinite(dgm).all(axis=1)]

    def diagram_distance(self, dgm0, dgm1):
        """
        Compute the configured distance between two diagrams.

        Parameters
        ----------
        dgm0 : array-like
            Reference persistence diagram.
        dgm1 : array-like
            Perturbed persistence diagram.

        Returns
        -------
        float
            Bottleneck or Wasserstein distance.
        """
        dgm0 = self._clean_diagram(dgm0)
        dgm1 = self._clean_diagram(dgm1)

        if len(dgm0) == 0 and len(dgm1) == 0:
            return 0.0

        if self.distance == "bottleneck":
            if gd is None:
                raise ImportError("gudhi is required for bottleneck distance.")

            if self.bottleneck_epsilon is None:
                return float(gd.bottleneck_distance(dgm0, dgm1))

            return float(
                gd.bottleneck_distance(
                    dgm0,
                    dgm1,
                    self.bottleneck_epsilon,
                )
            )

        if wasserstein_distance is None:
            raise ImportError(
                "gudhi.wasserstein.wasserstein_distance is required for "
                "Wasserstein distance."
            )

        return float(
            wasserstein_distance(
                dgm0,
                dgm1,
                order=self.wasserstein_order,
                internal_p=self.wasserstein_internal_p,
                matching=False,
            )
        )

    # =====================================================
    # Perturbation and scoring
    # =====================================================
    def perturb_embedding(self, gene_idx):
        """
        Perturb one gene across all cells and return the perturbed embedding.

        Parameters
        ----------
        gene_idx : int
            Gene index to perturb.

        Returns
        -------
        np.ndarray
            Perturbed embedding.

        Notes
        -----
        The actual perturbation logic is reused from ``Local_XplainTDA``.
        For linear embedders, that function applies the rank-1 embedding
        update. For nonlinear embedders, it perturbs ``tda.X`` and calls the
        fitted embedder through ``tda.embed``.
        """
        return perturb_embedding(
            self.tda,
            self.X,
            int(gene_idx),
            self.cell_idx,
            self.perturbation,
            self._rng_for_gene(gene_idx),
        )

    def _approximation_note(self):
        """
        Describe approximations introduced by the current configuration.

        Returns
        -------
        str
            Semicolon-separated approximation notes. Empty string means no
            explicit approximation was requested beyond floating-point
            numerical computation.
        """
        notes = []

        if self.ph_backend == "ripser":
            notes.append("Ripser backend on sparse kNN distance matrix")

        if getattr(self.tda, "neighbor_backend", "exact") == "pynndescent":
            notes.append("approximate kNN backend: pynndescent")

        if self.distance == "bottleneck" and self.bottleneck_epsilon not in (None, 0, 0.0):
            notes.append(f"bottleneck epsilon={self.bottleneck_epsilon}")

        return "; ".join(notes)

    def gene_importance(self, gene_idx, dims=None, target_dim=None):
        """
        Compute global importance of one gene.

        Parameters
        ----------
        gene_idx : int
            Gene index to perturb.
        dims : array-like, optional
            Homology dimensions to score.
        target_dim : int, optional
            Single homology dimension convenience argument.

        Returns
        -------
        dict[int, float]
            Mapping ``homology_dim -> diagram distance``.
        """
        dims = self._dims(dims=dims, target_dim=target_dim)

        Zp = self.perturb_embedding(gene_idx)
        dgms_p = self.compute_diagrams(Zp)

        return {
            dim: self.diagram_distance(self.reference_dgms[dim], dgms_p[dim])
            for dim in dims
        }

    def _process_gene(self, gene_idx, gene_name, dims):
        """
        Process one gene for joblib parallelization.

        Parameters
        ----------
        gene_idx : int
            Gene index.
        gene_name : str
            Gene name.
        dims : list[int]
            Homology dimensions to score.

        Returns
        -------
        list[dict]
            One output row per homology dimension.
        """
        scores = self.gene_importance(gene_idx, dims=dims)
        approx_note = self._approximation_note()

        rows = [
            {
                "homology_dim": int(dim),
                "gene_idx": int(gene_idx),
                "gene": gene_name,
                "ph_backend": self.ph_backend,
                "distance_metric": self.distance,
                "score": float(score),
                "approximation": bool(approx_note),
                "approximation_note": approx_note,
            }
            for dim, score in scores.items()
        ]

        if self.verbose:
            now = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
            print(
                f"Done gene {gene_name} ({gene_idx}) | dims={dims}",
                now,
                flush=True,
            )

        return rows

    # =====================================================
    # Batch scoring
    # =====================================================
    def compute_scores(
        self,
        genes=d.DEFAULT_GENES,
        dims=d.DEFAULT_DIMS,
        target_dim=d.DEFAULT_TARGET_DIM,
        n_jobs=d.DEFAULT_N_JOBS,
        backend=d.DEFAULT_BACKEND,
        cache=d.DEFAULT_CACHE,
    ):
        """
        Compute global topology-importance scores.

        Parameters
        ----------
        genes : None, int, str, or array-like
            Genes to score. If None, all genes are scored.
        dims : array-like, optional
            Homology dimensions to score. If None, uses ``target_dim`` or
            ``self.target_dim``. If both are None, all available dimensions
            are used.
        target_dim : int, optional
            Single homology dimension convenience argument.
        n_jobs : int, default=defaults.DEFAULT_N_JOBS
            Number of parallel jobs over genes. Use -1 for all available CPUs.
        backend : str, default=defaults.DEFAULT_BACKEND
            Joblib backend. ``"threading"`` avoids copying the fitted TDA
            object across processes.
        cache : bool, optional
            Whether to reuse cached results. If None, caching is enabled only
            for deterministic ``perturbation="zero"``.

        Returns
        -------
        pd.DataFrame
            Tidy table with one row per gene and homology dimension.
        """
        dims = self._dims(dims=dims, target_dim=target_dim)
        gene_idx, gene_names = self._resolve_genes(genes)

        use_cache = self.perturbation == "zero" if cache is None else cache
        key = self._score_cache_key(dims, gene_idx)

        if use_cache and key in self._score_cache:
            self.results = self._score_cache[key].copy()
            return self.results

        if len(gene_idx) == 0 or len(dims) == 0:
            self.results = pd.DataFrame()
            return self.results

        if n_jobs is None:
            n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

        if n_jobs != -1:
            n_jobs = max(1, min(int(n_jobs), len(gene_idx)))

        if self.verbose:
            now = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
            print(
                f"Launching {n_jobs} jobs | genes={len(gene_idx)} | "
                f"dims={dims} | ph_backend={self.ph_backend}",
                now,
                flush=True,
            )

        out = joblib.Parallel(n_jobs=n_jobs, backend=backend)(
            joblib.delayed(self._process_gene)(g, name, dims)
            for g, name in zip(gene_idx, gene_names)
        )

        rows = [row for gene_rows in out for row in gene_rows]
        df = pd.DataFrame(rows)

        if len(df):
            df["rank"] = (
                df.groupby("homology_dim")["score"]
                .rank(method="dense", ascending=False)
                .astype(int)
            )

            df = df.sort_values(
                ["homology_dim", "rank", "gene_idx"],
                kind="mergesort",
            ).reset_index(drop=True)

        self.results = df

        if use_cache:
            self._score_cache[key] = df.copy()

        return df

    # =====================================================
    # User-facing methods
    # =====================================================
    def importance_for_genes(
        self,
        genes,
        target_dim=None,
        n_jobs=d.DEFAULT_N_JOBS,
        backend=d.DEFAULT_BACKEND,
        cache=d.DEFAULT_CACHE,
    ):
        """
        Score one gene or a selected set of genes in one homology dimension.

        Parameters
        ----------
        genes : int, str, or array-like
            Gene index/name or collection of gene indices/names.
        target_dim : int, optional
            Homology dimension. If None, uses ``self.target_dim``.
        n_jobs : int, default=defaults.DEFAULT_N_JOBS
            Number of parallel jobs.
        backend : str, default=defaults.DEFAULT_BACKEND
            Joblib backend.
        cache : bool, optional
            Whether to reuse cached results.

        Returns
        -------
        pd.DataFrame
            Tidy global-importance table.
        """
        return self.compute_scores(
            genes=genes,
            target_dim=target_dim,
            n_jobs=n_jobs,
            backend=backend,
            cache=cache,
        )

    def importance_by_homology(
        self,
        target_dim,
        genes=None,
        n_jobs=d.DEFAULT_N_JOBS,
        backend=d.DEFAULT_BACKEND,
        cache=d.DEFAULT_CACHE,
    ):
        """
        Score selected genes, or all genes by default, for one homology group.

        Parameters
        ----------
        target_dim : int
            Homology dimension to score.
        genes : None, int, str, or array-like
            Genes to score. If None, all genes are scored.
        n_jobs : int, default=defaults.DEFAULT_N_JOBS
            Number of parallel jobs.
        backend : str, default=defaults.DEFAULT_BACKEND
            Joblib backend.
        cache : bool, optional
            Whether to reuse cached results.

        Returns
        -------
        pd.DataFrame
            Tidy global-importance table for one homology dimension.
        """
        return self.compute_scores(
            genes=genes,
            target_dim=target_dim,
            n_jobs=n_jobs,
            backend=backend,
            cache=cache,
        )

    def importance_all_homologies(
        self,
        genes=None,
        dims=None,
        n_jobs=d.DEFAULT_N_JOBS,
        backend=d.DEFAULT_BACKEND,
        cache=d.DEFAULT_CACHE,
    ):
        """
        Score selected genes, or all genes by default, for multiple dimensions.

        Parameters
        ----------
        genes : None, int, str, or array-like
            Genes to score. If None, all genes are scored.
        dims : array-like, optional
            Homology dimensions to score. If None, all available dimensions
            are used.
        n_jobs : int, default=defaults.DEFAULT_N_JOBS
            Number of parallel jobs.
        backend : str, default=defaults.DEFAULT_BACKEND
            Joblib backend.
        cache : bool, optional
            Whether to reuse cached results.

        Returns
        -------
        pd.DataFrame
            Tidy global-importance table across dimensions.
        """
        dims = self._available_dims() if dims is None else dims

        return self.compute_scores(
            genes=genes,
            dims=dims,
            n_jobs=n_jobs,
            backend=backend,
            cache=cache,
        )

    # =====================================================
    # Saving and loading
    # =====================================================
    def _state_dict(self, save_tda=True, save_results=True, save_cache=False):
        """
        Build a serializable state dictionary.

        Parameters
        ----------
        save_tda : bool, default=True
            Whether to include the fitted XplainTDA object.
        save_results : bool, default=True
            Whether to include ``self.results``.
        save_cache : bool, default=False
            Whether to include ``self._score_cache``.

        Returns
        -------
        dict
            Serializable state for joblib.
        """
        return {
            "config": {
                "target_dim": self.target_dim,
                "perturbation": self.perturbation,
                "distance": self.distance,
                "ph_backend": self.ph_backend,
                "bottleneck_epsilon": self.bottleneck_epsilon,
                "wasserstein_order": self.wasserstein_order,
                "wasserstein_internal_p": self.wasserstein_internal_p,
                "verbose": self.verbose,
                "random_state": self.random_state,
            },
            "tda": self.tda if save_tda else None,
            "reference_dgms": self.reference_dgms,
            "results": (
                self.results.copy()
                if save_results and self.results is not None
                else None
            ),
            "score_cache": self._score_cache.copy() if save_cache else {},
        }

    def save(
        self,
        path,
        save_tda=True,
        save_results=True,
        save_cache=False,
        compress=3,
    ):
        """
        Save the global explainer to disk.

        Parameters
        ----------
        path : str
            Output path, usually ending in ``.joblib``.
        save_tda : bool, default=True
            Whether to serialize the fitted XplainTDA object.
        save_results : bool, default=True
            Whether to save the most recent result table.
        save_cache : bool, default=False
            Whether to save cached score tables.
        compress : int, default=3
            Joblib compression level.

        Returns
        -------
        None
            Writes the serialized state to ``path``.
        """
        joblib.dump(
            self._state_dict(
                save_tda=save_tda,
                save_results=save_results,
                save_cache=save_cache,
            ),
            path,
            compress=compress,
        )

    @classmethod
    def load(cls, path, tda=None):
        """
        Load a saved global explainer.

        Parameters
        ----------
        path : str
            Path to a file created by ``Global_XplainTDA.save``.
        tda : XplainTDA, optional
            Fitted XplainTDA object. Required if the explainer was saved with
            ``save_tda=False``. If provided, it overrides any saved TDA object.

        Returns
        -------
        Global_XplainTDA
            Restored global explainer.
        """
        state = joblib.load(path)

        tda_loaded = state.get("tda", None)
        tda = tda if tda is not None else tda_loaded

        if tda is None:
            raise ValueError(
                "No XplainTDA object found. Pass a fitted object with "
                "`Global_XplainTDA.load(path, tda=tda)`."
            )

        obj = cls(tda=tda, **state["config"])
        obj.reference_dgms = state.get("reference_dgms", obj.reference_dgms)
        obj.results = state.get("results", None)
        obj._score_cache = state.get("score_cache", {})

        return obj
