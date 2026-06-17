"""Global gene explanations with optional reuse of a saved GRAE model."""

from pathlib import Path
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

from Embedder import GRAEEmbedder
from Local_XplainTDA import VALID_PERTURBATIONS, perturb_embedding
from XplainTDA import (
    VALID_PH_BACKENDS,
    compute_neighbors,
    compute_ph_from_neighbors,
    edge_list_to_sparse_distance,
    neighbors_to_edge_list,
    ripser_sparse_to_diagrams,
)
import defaults as d


VALID_DISTANCES = {"bottleneck", "wasserstein"}


class Global_XplainTDA:
    """Compute global topology-aware gene-importance scores.

    Each gene is perturbed across all cells. The perturbed count matrix is
    projected with the fitted embedder attached to ``tda`` and persistent
    homology is recomputed on the resulting embedding. Gene importance is the
    bottleneck or Wasserstein distance between the reference and perturbed
    persistence diagrams.

    A saved GRAE model can be supplied through ``grae_model_path``. The model
    is restored with :meth:`Embedder.GRAEEmbedder.load` and attached to
    ``tda.embedder``; it is never retrained.
    """

    def __init__(
        self,
        tda,
        grae_model_path=None,
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
        """Initialize the global explainer.

        Parameters
        ----------
        tda : XplainTDA
            Fitted topology object. The AnnData object and expression matrix
            are obtained from ``tda.adata`` and ``tda.X``.
        grae_model_path : str or pathlib.Path, optional
            Path to a model saved with ``GRAEEmbedder.save``. When supplied,
            the trained model is loaded and attached to ``tda`` for embedding
            perturbed count matrices.
        target_dim : int, optional
            Default homology dimension used by single-dimension methods.
        perturbation : {"zero", "permute"}
            Perturbation applied to each gene across all cells.
        distance : {"bottleneck", "wasserstein"}
            Persistence-diagram distance used as the importance score.
        ph_backend : {"ripser", "gudhi"}
            Backend used for both reference and perturbed diagrams. Ripser on
            the sparse kNN distance matrix is the default.
        bottleneck_epsilon : float or None
            Approximation parameter passed to GUDHI bottleneck distance.
        wasserstein_order : float
            Wasserstein order.
        wasserstein_internal_p : float
            Ground-metric order used by Wasserstein distance.
        verbose : bool
            Whether to print progress messages.
        random_state : int, optional
            Seed used for reproducible permutation perturbations.
        """
        self._validate_init(
            perturbation,
            distance,
            ph_backend,
            bottleneck_epsilon,
        )
        self._validate_tda(tda)

        self.tda = tda
        self.adata = tda.adata
        self.X = self._resolve_expression_matrix()
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
        self.grae_model_path = None
        self.results = None
        self._score_cache = {}

        if grae_model_path is not None:
            self.load_grae_embedder(grae_model_path)

        # Reference and perturbed diagrams must be computed with the same
        # backend and graph construction.
        self.reference_dgms = self.compute_diagrams(self.tda.embedding)

    # ------------------------------------------------------------------
    # Validation and input resolution
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_init(perturbation, distance, ph_backend, bottleneck_epsilon):
        """Validate constructor options.

        Raises
        ------
        ValueError
            If a categorical option or bottleneck epsilon is invalid.
        """
        if perturbation not in VALID_PERTURBATIONS:
            raise ValueError(
                f"perturbation must be one of {sorted(VALID_PERTURBATIONS)}"
            )
        if distance not in VALID_DISTANCES:
            raise ValueError(f"distance must be one of {sorted(VALID_DISTANCES)}")
        if ph_backend.lower() not in VALID_PH_BACKENDS:
            raise ValueError(
                f"ph_backend must be one of {sorted(VALID_PH_BACKENDS)}"
            )
        if bottleneck_epsilon is not None and bottleneck_epsilon < 0:
            raise ValueError("bottleneck_epsilon must be non-negative or None")

    @staticmethod
    def _validate_tda(tda):
        """Validate that ``tda`` contains the state required for explanation."""
        if tda is None:
            raise ValueError("A fitted `tda` object is required.")
        if getattr(tda, "adata", None) is None:
            raise ValueError("`tda.adata` is required for gene perturbations.")
        if getattr(tda, "embedding", None) is None:
            raise ValueError("`tda.embedding` is missing; fit XplainTDA first.")
        if getattr(tda, "n_cells", None) is None or getattr(tda, "n_genes", None) is None:
            raise ValueError("`tda.n_cells` and `tda.n_genes` must be defined.")

    def _resolve_expression_matrix(self):
        """Return ``tda.X``, restoring it from ``tda.adata`` when necessary."""
        if self.tda.X is None:
            layer = getattr(self.tda, "layer", "X")
            self.tda.X = (
                self.adata.X if layer == "X" else self.adata.layers[layer]
            )
        return self.tda.X

    def _available_dims(self):
        """Return homology dimensions present in the reference diagrams."""
        return list(range(len(self.reference_dgms)))

    def _dim(self, target_dim=None):
        """Resolve and validate one homology dimension."""
        dim = self.target_dim if target_dim is None else target_dim
        if dim is None:
            raise ValueError(
                "target_dim must be specified. Use "
                "importance_all_homologies() for all dimensions."
            )
        dim = int(dim)
        if dim not in self._available_dims():
            raise ValueError(
                f"target_dim={dim} is unavailable; available dimensions are "
                f"{self._available_dims()}."
            )
        return dim

    def _dims(self, dims=None, target_dim=None):
        """Resolve and validate one or more homology dimensions."""
        if dims is not None:
            out = [int(x) for x in np.atleast_1d(dims)]
        elif target_dim is not None or self.target_dim is not None:
            out = [self._dim(target_dim)]
        else:
            out = self._available_dims()

        invalid = sorted(set(out) - set(self._available_dims()))
        if invalid:
            raise ValueError(
                f"Unavailable dimensions: {invalid}; available dimensions are "
                f"{self._available_dims()}."
            )
        return out

    def _resolve_genes(self, genes=None):
        """Resolve gene names or positions to aligned indices and names.

        Parameters
        ----------
        genes : None, int, str, or array-like
            Genes to score. ``None`` selects every gene.

        Returns
        -------
        gene_idx : np.ndarray
            Integer gene positions.
        gene_names : np.ndarray
            Names aligned with ``gene_idx``.
        """
        if genes is None:
            idx = np.arange(self.tda.n_genes, dtype=int)
            return idx, np.asarray(self.adata.var_names)[idx]

        genes = [genes] if np.isscalar(genes) or isinstance(genes, str) else list(genes)
        if not genes:
            return np.array([], dtype=int), np.array([], dtype=str)

        if isinstance(genes[0], str):
            idx = self.adata.var_names.get_indexer(genes)
            missing = np.asarray(genes)[idx < 0]
            if len(missing):
                raise ValueError(f"Unknown genes: {missing.tolist()}")
            return idx.astype(int), np.asarray(genes)

        idx = np.asarray(genes, dtype=int)
        if np.any(idx < 0) or np.any(idx >= self.tda.n_genes):
            raise IndexError("gene indices must be between 0 and tda.n_genes - 1")
        return idx, np.asarray(self.adata.var_names)[idx]

    def _rng_for_gene(self, gene_idx):
        """Create a deterministic per-gene RNG when a seed is configured."""
        if self.random_state is None:
            return np.random.default_rng()
        return np.random.default_rng(int(self.random_state) + int(gene_idx))

    def _score_cache_key(self, dims, genes):
        """Build a cache key for one batch of scores."""
        return (
            tuple(map(int, dims)),
            tuple(np.asarray(genes, dtype=int).tolist()),
            self.perturbation,
            self.distance,
            self.ph_backend,
            self.grae_model_path,
            None if self.bottleneck_epsilon is None else float(self.bottleneck_epsilon),
            float(self.wasserstein_order),
            float(self.wasserstein_internal_p),
        )

    # ------------------------------------------------------------------
    # Saved GRAE model
    # ------------------------------------------------------------------
    @staticmethod
    def _saved_input_dim(embedder):
        """Best-effort extraction of the trained model input dimension."""
        shape = getattr(getattr(embedder, "model", None), "data_shape", None)
        if shape is None:
            return None
        try:
            values = np.atleast_1d(shape).astype(int)
        except (TypeError, ValueError):
            return None
        return int(values[-1]) if len(values) else None

    def load_grae_embedder(self, path):
        """Load a saved GRAE model and attach it to ``tda``.

        Parameters
        ----------
        path : str or pathlib.Path
            File created by ``GRAEEmbedder.save``.

        Returns
        -------
        Global_XplainTDA
            This explainer, with the restored model attached.

        Notes
        -----
        The model is restored from its saved weights and is not retrained.
        Existing results and caches are cleared because they may have been
        computed with a different embedder.
        """
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Saved GRAE embedder not found: {path}")

        embedder = GRAEEmbedder.load(path)
        model = getattr(embedder, "model", None)
        if model is None or getattr(model, "torch_module", None) is None:
            raise ValueError(f"{path} does not contain a fitted GRAE model.")

        latent_dim = int(self.tda.embedding.shape[1])
        if int(embedder.n_components) != latent_dim:
            raise ValueError(
                "Saved GRAE latent dimension does not match tda.embedding: "
                f"{embedder.n_components} != {latent_dim}."
            )

        input_dim = self._saved_input_dim(embedder)
        if input_dim is not None and input_dim != int(self.X.shape[1]):
            raise ValueError(
                "Saved GRAE input dimension does not match tda.X: "
                f"{input_dim} != {self.X.shape[1]}. Ensure identical features "
                "and feature order."
            )

        self.tda.embedder = embedder
        self.tda.embedder_name = "GRAE"
        self.tda.n_components = embedder.n_components
        self.grae_model_path = str(path)
        self.results = None
        self._score_cache.clear()
        return self

    # ------------------------------------------------------------------
    # Persistent homology
    # ------------------------------------------------------------------
    def compute_diagrams(self, embedding):
        """Compute diagrams from an embedding with the configured backend.

        Ripser receives a sparse symmetric distance matrix obtained from the
        embedding kNN graph. GUDHI receives the same kNN graph and builds the
        corresponding clique complex. Feature localization is skipped because
        global explanation only requires persistence diagrams.
        """
        Z = np.asarray(embedding, dtype=self.tda.dtype)
        neigh_ind, neigh_dist = compute_neighbors(
            Z,
            n_neighbors=self.tda.n_neighbors,
            backend=self.tda.neighbor_backend,
        )

        if self.ph_backend == "ripser":
            edges = neighbors_to_edge_list(neigh_ind, neigh_dist)
            distances = edge_list_to_sparse_distance(edges, n=Z.shape[0])
            return ripser_sparse_to_diagrams(
                distances,
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

    @staticmethod
    def _clean_diagram(dgm):
        """Return finite, positive-lifetime points from a diagram."""
        dgm = np.asarray(dgm, dtype=float).reshape(-1, 2)
        if not len(dgm):
            return dgm
        mask = np.isfinite(dgm).all(axis=1) & (dgm[:, 1] > dgm[:, 0])
        return dgm[mask]

    def diagram_distance(self, dgm0, dgm1):
        """Compute the configured bottleneck or Wasserstein distance."""
        dgm0, dgm1 = self._clean_diagram(dgm0), self._clean_diagram(dgm1)
        if len(dgm0) == 0 and len(dgm1) == 0:
            return 0.0

        if self.distance == "bottleneck":
            if gd is None:
                raise ImportError("gudhi is required for bottleneck distance")
            if self.bottleneck_epsilon is None:
                return float(gd.bottleneck_distance(dgm0, dgm1))
            return float(
                gd.bottleneck_distance(dgm0, dgm1, self.bottleneck_epsilon)
            )

        if wasserstein_distance is None:
            raise ImportError(
                "gudhi.wasserstein and its dependencies are required for "
                "Wasserstein distance"
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

    # ------------------------------------------------------------------
    # Perturbation and scoring
    # ------------------------------------------------------------------
    def perturb_embedding(self, gene_idx):
        """Perturb one gene across all cells and return its new embedding."""
        if getattr(self.tda, "embedder", None) is None:
            raise ValueError(
                "No fitted embedder is attached to tda. For a stored GRAE "
                "embedding, pass `grae_model_path` or call "
                "`load_grae_embedder(path)` before scoring genes."
            )
        return perturb_embedding(
            self.tda,
            self.X,
            int(gene_idx),
            self.cell_idx,
            self.perturbation,
            self._rng_for_gene(gene_idx),
        )

    def _approximation_note(self):
        """Describe approximations introduced by the current configuration."""
        notes = []
        if self.ph_backend == "ripser":
            notes.append("Ripser on sparse kNN distance matrix")
        if getattr(self.tda, "neighbor_backend", "exact") == "pynndescent":
            notes.append("approximate kNN backend: pynndescent")
        if self.distance == "bottleneck" and self.bottleneck_epsilon not in (None, 0, 0.0):
            notes.append(f"bottleneck epsilon={self.bottleneck_epsilon}")
        return "; ".join(notes)

    def gene_importance(self, gene_idx, dims=None, target_dim=None):
        """Compute diagram-distance scores for one perturbed gene."""
        dims = self._dims(dims=dims, target_dim=target_dim)
        perturbed_dgms = self.compute_diagrams(self.perturb_embedding(gene_idx))
        return {
            dim: self.diagram_distance(
                self.reference_dgms[dim],
                perturbed_dgms[dim],
            )
            for dim in dims
        }

    def _process_gene(self, gene_idx, gene_name, dims):
        """Joblib worker returning one result row per homology dimension."""
        scores = self.gene_importance(gene_idx, dims=dims)
        note = self._approximation_note()
        rows = [
            {
                "homology_dim": int(dim),
                "gene_idx": int(gene_idx),
                "gene": gene_name,
                "ph_backend": self.ph_backend,
                "distance_metric": self.distance,
                "score": float(score),
                "approximation": bool(note),
                "approximation_note": note,
            }
            for dim, score in scores.items()
        ]
        if self.verbose:
            print(
                f"Done gene {gene_name} ({gene_idx}) | dims={dims} | "
                f"{time.strftime('%a, %d %b %Y %H:%M:%S')}",
                flush=True,
            )
        return rows

    def compute_scores(
        self,
        genes=d.DEFAULT_GENES,
        dims=d.DEFAULT_DIMS,
        target_dim=d.DEFAULT_TARGET_DIM,
        n_jobs=d.DEFAULT_N_JOBS,
        backend=d.DEFAULT_BACKEND,
        cache=d.DEFAULT_CACHE,
    ):
        """Compute global scores for selected genes and homology dimensions.

        ``genes=None`` scores every gene. Parallel work is distributed across
        genes. For GPU-based GRAE inference, start with ``n_jobs=1`` to avoid
        concurrent use of the same model and excessive GPU memory pressure.
        """
        dims = self._dims(dims=dims, target_dim=target_dim)
        gene_idx, gene_names = self._resolve_genes(genes)
        use_cache = self.perturbation == "zero" if cache is None else bool(cache)
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
            print(
                f"Launching {n_jobs} jobs | genes={len(gene_idx)} | "
                f"dims={dims} | ph_backend={self.ph_backend}",
                flush=True,
            )

        output = joblib.Parallel(n_jobs=n_jobs, backend=backend)(
            joblib.delayed(self._process_gene)(idx, name, dims)
            for idx, name in zip(gene_idx, gene_names)
        )
        df = pd.DataFrame(row for rows in output for row in rows)

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

    # ------------------------------------------------------------------
    # Public convenience methods
    # ------------------------------------------------------------------
    def importance_for_genes(
        self,
        genes,
        target_dim=None,
        n_jobs=d.DEFAULT_N_JOBS,
        backend=d.DEFAULT_BACKEND,
        cache=d.DEFAULT_CACHE,
    ):
        """Score specified genes in one homology dimension."""
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
        """Score selected genes, or all genes, for one homology dimension."""
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
        """Score selected genes, or all genes, across requested dimensions."""
        dims = self._available_dims() if dims is None else dims
        return self.compute_scores(
            genes=genes,
            dims=dims,
            n_jobs=n_jobs,
            backend=backend,
            cache=cache,
        )

    # ------------------------------------------------------------------
    # Saving and loading
    # ------------------------------------------------------------------
    def _state_dict(self, save_tda=True, save_results=True, save_cache=False):
        """Build the serializable state used by :meth:`save`."""
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
            "grae_model_path": self.grae_model_path,
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
        """Save the explainer state with joblib."""
        joblib.dump(
            self._state_dict(save_tda, save_results, save_cache),
            path,
            compress=compress,
        )

    @classmethod
    def load(cls, path, tda=None, grae_model_path=None):
        """Load a saved explainer.

        ``grae_model_path`` overrides the path recorded in the saved state and
        should be supplied when the model file has moved.
        """
        state = joblib.load(path)
        tda = state.get("tda") if tda is None else tda
        if tda is None:
            raise ValueError(
                "No TDA object was saved. Pass it explicitly with `tda=...`."
            )

        explicit_override = grae_model_path is not None
        if grae_model_path is None and getattr(tda, "embedder", None) is None:
            saved_path = state.get("grae_model_path")
            if saved_path is not None and Path(saved_path).expanduser().is_file():
                grae_model_path = saved_path

        obj = cls(
            tda=tda,
            grae_model_path=grae_model_path,
            **state["config"],
        )
        obj.reference_dgms = state.get("reference_dgms", obj.reference_dgms)
        obj.results = None if explicit_override else state.get("results")
        obj._score_cache = {} if explicit_override else state.get("score_cache", {})
        return obj
