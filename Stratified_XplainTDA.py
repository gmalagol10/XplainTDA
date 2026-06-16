"""Stratified XplainTDA using one full-data embedding/model pair."""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import Embedder
import defaults as d
from Embedder import GRAEEmbedder
from Global_XplainTDA import Global_XplainTDA
from Local_XplainTDA import Local_XplainTDA
from XplainTDA import XplainTDA


class StratifiedXplainTDA:
    """Run group-specific PH explanations in one full-data embedding space.

    A GRAE embedding/model pair is treated atomically. The supplied model is
    reused only when ``grae_model_path`` exists and ``embedding_key`` is
    present in ``adata.obsm`` and both artifacts are compatible. Otherwise,
    one new embedder is fitted on all cells using the configured settings.

    The fallback embedding and model are saved separately in ``outdir``. The
    input ``.h5ad`` file is never rewritten; ``adata.obsm[embedding_key]`` is
    updated only in memory.
    """

    def __init__(
        self,
        adata,
        groupby,
        groups=d.DEFAULT_GROUPS,
        min_cells=d.DEFAULT_MIN_CELLS,
        layer=d.DEFAULT_LAYER,
        embedder=d.DEFAULT_EMBEDDER,
        n_components=d.DEFAULT_N_COMPONENTS,
        n_neighbors=d.DEFAULT_N_NEIGHBORS,
        maxdim=d.DEFAULT_MAXDIM,
        neighbor_backend=d.DEFAULT_NEIGHBOR_BACKEND,
        ph_backend=d.DEFAULT_PH_BACKEND,
        dtype=d.DEFAULT_DTYPE,
        embedder_kwargs=d.DEFAULT_EMBEDDER_KWARGS,
        embedding_key="X_GRAE",
        grae_model_path=None,
        verbose=d.DEFAULT_VERBOSE,
        outdir="StratifiedXplainTDA",
        save_intermediate=True,
        compress=3,
    ):
        if groupby not in adata.obs:
            raise ValueError(f"{groupby!r} is not a column in adata.obs")
        if embedding_key is None or not str(embedding_key):
            raise ValueError("`embedding_key` must be a non-empty string.")

        self.adata = adata
        self.groupby = groupby
        self.groups = (
            list(pd.Series(adata.obs[groupby]).dropna().unique())
            if groups is None
            else list(groups)
        )
        self.min_cells = int(min_cells)
        self.layer = layer
        self.embedder_name = str(embedder).upper()
        self.n_components = n_components
        self.n_neighbors = int(n_neighbors)
        self.maxdim = int(maxdim)
        self.neighbor_backend = neighbor_backend
        self.ph_backend = str(ph_backend).lower()
        self.dtype = dtype
        self.embedder_kwargs = dict(embedder_kwargs or {})
        self.embedding_key = str(embedding_key)
        self.requested_grae_model_path = (
            None
            if grae_model_path is None
            else str(Path(grae_model_path).expanduser().resolve())
        )
        self.verbose = bool(verbose)

        self.X = adata.X if layer == "X" else adata.layers[layer]
        self.n_cells, self.n_genes = self.X.shape
        self.n_components_ = (
            int(np.round(self.n_genes ** (1 / 3)))
            if n_components is None
            else int(n_components)
        )

        self.outdir = Path(outdir).expanduser().resolve()
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.save_intermediate = bool(save_intermediate)
        self.compress = int(compress)

        self.global_embedder = None
        self.global_embedding = None
        self.resolved_grae_model_path = None
        self.embedding_artifact_path = None
        self.embedding_source = None
        self.embedding_fallback_reason = None

        self.tdas = {}
        self.local_explainers = {}
        self.global_explainers = {}
        self.local_results = {}
        self.global_results = {}
        self.skipped_groups = {}

    # =====================================================
    # Full-data embedding/model pair
    # =====================================================
    @staticmethod
    def _saved_input_dim(embedder):
        """Best-effort extraction of a saved GRAE input dimension."""
        shape = getattr(getattr(embedder, "model", None), "data_shape", None)
        if shape is None:
            return None
        try:
            values = np.atleast_1d(shape).astype(int)
        except (TypeError, ValueError):
            return None
        return int(values[-1]) if len(values) else None

    def _validate_embedding(self, embedding):
        """Validate the full-data embedding and convert it to ``self.dtype``."""
        embedding = np.asarray(embedding, dtype=self.dtype)
        if embedding.ndim != 2:
            raise ValueError("Stored embedding must be two-dimensional.")
        if embedding.shape[0] != self.n_cells:
            raise ValueError(
                f"Stored embedding has {embedding.shape[0]} rows but adata has "
                f"{self.n_cells} cells."
            )
        if embedding.shape[1] < 1:
            raise ValueError("Stored embedding has no latent components.")
        if not np.isfinite(embedding).all():
            raise ValueError("Stored embedding contains NaN or infinite values.")
        if self.n_components is not None and embedding.shape[1] != self.n_components_:
            raise ValueError(
                "Stored embedding dimension does not match requested "
                f"n_components: {embedding.shape[1]} != {self.n_components_}."
            )
        return embedding

    def _load_requested_pair(self):
        """Load and validate the user-supplied GRAE model/embedding pair."""
        if self.requested_grae_model_path is None:
            raise ValueError("No saved GRAE model path was provided.")
        if self.embedding_key not in self.adata.obsm:
            raise KeyError(
                f"{self.embedding_key!r} is not present in adata.obsm."
            )

        model_path = Path(self.requested_grae_model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Saved GRAE model not found: {model_path}")

        embedding = self._validate_embedding(self.adata.obsm[self.embedding_key])
        embedder = GRAEEmbedder.load(model_path)
        model = getattr(embedder, "model", None)
        if model is None or getattr(model, "torch_module", None) is None:
            raise ValueError(f"{model_path} does not contain a fitted GRAE model.")
        if int(embedder.n_components) != int(embedding.shape[1]):
            raise ValueError(
                "Saved GRAE latent dimension does not match the stored "
                f"embedding: {embedder.n_components} != {embedding.shape[1]}."
            )

        input_dim = self._saved_input_dim(embedder)
        if input_dim is not None and input_dim != self.n_genes:
            raise ValueError(
                "Saved GRAE input dimension does not match adata: "
                f"{input_dim} != {self.n_genes}. Ensure identical features "
                "and feature order."
            )

        self.embedder_name = "GRAE"
        self.global_embedder = embedder
        self.global_embedding = embedding
        self.n_components_ = int(embedding.shape[1])
        self.embedding_source = "loaded_saved_pair"

    def _fit_fallback_pair(self, reason):
        """Fit one new default-configured embedder on all cells."""
        self.embedding_fallback_reason = str(reason)
        if self.verbose:
            print(
                "Saved GRAE model/embedding pair unavailable or incompatible; "
                f"fitting {self.embedder_name} on all cells. Reason: {reason}",
                flush=True,
            )

        try:
            cls = Embedder.Embedder.registry[self.embedder_name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown embedder {self.embedder_name!r}. Available: "
                f"{list(Embedder.Embedder.registry)}"
            ) from exc

        kwargs = dict(self.embedder_kwargs)
        if issubclass(cls, GRAEEmbedder):
            # Use the stratification variable for GRAE validation splitting
            # unless the caller explicitly supplied another label vector.
            kwargs.setdefault(
                "stratify_labels",
                self.adata.obs[self.groupby].to_numpy(),
            )

        self.global_embedder = cls(
            n_components=self.n_components_,
            dtype=self.dtype,
            **kwargs,
        )
        self.global_embedding = self._validate_embedding(
            self.global_embedder.fit_transform(self.X)
        )
        self.embedding_source = "fitted_full_data_fallback"

    def _save_embedding_pair(self):
        """Save the full embedding and fitted model as separate artifacts."""
        artifact_dir = self.outdir / "full_embedding"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        safe_key = self.embedding_key.replace("/", "_").replace(" ", "_")
        embedding_path = artifact_dir / f"{safe_key}.npy"
        model_path = artifact_dir / f"{self.embedder_name}_embedder.joblib"

        np.save(embedding_path, self.global_embedding)
        if hasattr(self.global_embedder, "save"):
            self.global_embedder.save(model_path, compress=self.compress)
        else:
            joblib.dump(self.global_embedder, model_path, compress=self.compress)

        self.embedding_artifact_path = str(embedding_path)
        if isinstance(self.global_embedder, GRAEEmbedder):
            self.resolved_grae_model_path = str(model_path.resolve())
        else:
            self.resolved_grae_model_path = None

        metadata = {
            "embedding_key": self.embedding_key,
            "embedding_path": self.embedding_artifact_path,
            "model_path": str(model_path.resolve()),
            "embedder": self.embedder_name,
            "embedding_source": self.embedding_source,
            "fallback_reason": self.embedding_fallback_reason,
            "shape": list(self.global_embedding.shape),
            "layer": self.layer,
            "groupby": self.groupby,
        }
        with (artifact_dir / "metadata.json").open("w") as handle:
            json.dump(metadata, handle, indent=2, default=str)

    def fit_embedding(self):
        """Load a valid pair or fit one new full-data embedding/model pair."""
        if self.global_embedding is not None:
            return self

        try:
            self._load_requested_pair()
            if self.verbose:
                print(
                    f"Using saved GRAE model and adata.obsm[{self.embedding_key!r}] "
                    f"with shape {self.global_embedding.shape}",
                    flush=True,
                )
        except Exception as exc:
            self._fit_fallback_pair(exc)

        # Keep the paired embedding available to group-specific AnnData views,
        # but do not write the input h5ad file.
        self.adata.obsm[self.embedding_key] = self.global_embedding
        self._save_embedding_pair()
        return self

    # =====================================================
    # Group-specific PH
    # =====================================================
    def _mask(self, group):
        return np.asarray(self.adata.obs[self.groupby] == group)

    def fit_group(self, group):
        """Compute group PH using rows of the one full-data embedding."""
        if group in self.tdas:
            return self.tdas[group]

        self.fit_embedding()
        mask = self._mask(group)
        n = int(mask.sum())
        if n < self.min_cells:
            self.skipped_groups[group] = (
                f"n_cells={n} < min_cells={self.min_cells}"
            )
            if self.verbose:
                print(f"Skipping {group}: {self.skipped_groups[group]}", flush=True)
            return None

        n_neighbors = min(self.n_neighbors, n)
        adata_g = self.adata[mask].copy()
        embedding_g = np.asarray(self.global_embedding[mask], dtype=self.dtype)
        adata_g.obsm[self.embedding_key] = embedding_g

        if self.verbose:
            print(
                f"Fitting PH for {group} | cells={n} | "
                f"n_neighbors={n_neighbors} | "
                f"{time.strftime('%a, %d %b %Y %H:%M:%S')}",
                flush=True,
            )

        tda = XplainTDA(
            adata=adata_g,
            layer=self.layer,
            embedder=self.embedder_name,
            n_components=self.n_components_,
            n_neighbors=n_neighbors,
            maxdim=self.maxdim,
            ph_backend=self.ph_backend,
            neighbor_backend=self.neighbor_backend,
            dtype=self.dtype,
            fitted_embedder=self.global_embedder,
            embedding_key=self.embedding_key,
        ).fit(embedding=embedding_g)

        if self.resolved_grae_model_path is not None:
            tda.grae_model_path = self.resolved_grae_model_path

        self.tdas[group] = tda
        self._save_group_tda(group, tda)
        return tda

    def fit(self):
        self.fit_embedding()
        for group in self.groups:
            self.fit_group(group)
        return self

    # =====================================================
    # Explanations
    # =====================================================
    @staticmethod
    def _add_group_columns(df, group, tda, method):
        if df is None or len(df) == 0:
            return pd.DataFrame()
        df = df.copy()
        df.insert(0, "group", group)
        df.insert(1, "n_cells", tda.n_cells)
        df.insert(2, "n_neighbors", tda.n_neighbors)
        df.insert(3, "method", method)
        return df

    @staticmethod
    def _concat(tables):
        tables = [table for table in tables if table is not None and len(table)]
        return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()

    def local_expl(
        self,
        genes=d.DEFAULT_GENES,
        dims=d.DEFAULT_DIMS,
        n_jobs=d.DEFAULT_N_JOBS,
        backend=d.DEFAULT_BACKEND,
        cache=d.DEFAULT_CACHE,
        target_dim=d.DEFAULT_TARGET_DIM,
        perturbation=d.DEFAULT_PERTURBATION,
        recompute_mode=d.DEFAULT_RECOMPUTE_MODE,
        local_radius=d.DEFAULT_LOCAL_RADIUS,
        top_frac=d.DEFAULT_TOP_FRAC,
        max_features=d.DEFAULT_MAX_FEATURES,
        matching_mode=d.DEFAULT_MATCHING_MODE,
        jaccard_threshold=d.DEFAULT_JACCARD_THRESHOLD,
        random_state=d.DEFAULT_RANDOM_STATE,
    ):
        tables = []
        for group in self.groups:
            print("=" * 60, flush=True)
            print(f"Start local XplainTDA for {group}", flush=True)
            print("=" * 60, flush=True)

            tda = self.fit_group(group)
            if tda is None:
                continue

            expl = Local_XplainTDA(
                tda=tda,
                grae_model_path=self.resolved_grae_model_path,
                embedding_key=(
                    self.embedding_key
                    if self.resolved_grae_model_path is not None
                    else None
                ),
                target_dim=target_dim,
                perturbation=perturbation,
                recompute_mode=recompute_mode,
                local_radius=local_radius,
                top_frac=top_frac,
                max_features=max_features,
                matching_mode=matching_mode,
                jaccard_threshold=jaccard_threshold,
                verbose=self.verbose,
                random_state=random_state,
            )
            df = expl.importance_all_homologies(
                genes=genes,
                dims=dims,
                n_jobs=n_jobs,
                backend=backend,
                cache=cache,
            )
            df = self._add_group_columns(df, group, tda, "local")
            self.local_explainers[group] = expl
            self.local_results[group] = df
            tables.append(df)
            self._save_explainer(group, expl, method="local")

        final = self._concat(tables)
        self._save_final_dataframe(final, method="local")
        return final

    def global_expl(
        self,
        genes=d.DEFAULT_GENES,
        dims=d.DEFAULT_DIMS,
        n_jobs=d.DEFAULT_N_JOBS,
        backend=d.DEFAULT_BACKEND,
        ph_backend=d.DEFAULT_GLOBAL_PH_BACKEND,
        cache=d.DEFAULT_CACHE,
        target_dim=d.DEFAULT_TARGET_DIM,
        perturbation=d.DEFAULT_PERTURBATION,
        distance=d.DEFAULT_DISTANCE,
        bottleneck_epsilon=d.DEFAULT_BOTTLENECK_EPSILON,
        wasserstein_order=d.DEFAULT_WASSERSTEIN_ORDER,
        wasserstein_internal_p=d.DEFAULT_WASSERSTEIN_INTERNAL_P,
        random_state=d.DEFAULT_RANDOM_STATE,
    ):
        tables = []
        for group in self.groups:
            print("=" * 60, flush=True)
            print(f"Start global XplainTDA for {group}", flush=True)
            print("=" * 60, flush=True)

            tda = self.fit_group(group)
            if tda is None:
                continue

            # The full-data model is already attached to tda. Avoid reloading
            # it for every group, but retain its path for saved-state recovery.
            expl = Global_XplainTDA(
                tda=tda,
                target_dim=target_dim,
                perturbation=perturbation,
                distance=distance,
                ph_backend=ph_backend,
                bottleneck_epsilon=bottleneck_epsilon,
                wasserstein_order=wasserstein_order,
                wasserstein_internal_p=wasserstein_internal_p,
                verbose=self.verbose,
                random_state=random_state,
            )
            expl.grae_model_path = self.resolved_grae_model_path

            df = expl.importance_all_homologies(
                genes=genes,
                dims=dims,
                n_jobs=n_jobs,
                backend=backend,
                cache=cache,
            )
            df = self._add_group_columns(df, group, tda, "global")
            self.global_explainers[group] = expl
            self.global_results[group] = df
            tables.append(df)
            self._save_explainer(group, expl, method="global")

        final = self._concat(tables)
        self._save_final_dataframe(final, method="global")
        return final

    # =====================================================
    # Saving
    # =====================================================
    @staticmethod
    def _safe_name(value):
        return str(value).replace("/", "_").replace("\\", "_").replace(" ", "_")

    def _ckpt_path(self, *parts):
        path = self.outdir.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _save_group_tda(self, group, tda):
        if not self.save_intermediate:
            return
        tda.save(
            self._ckpt_path("tdas", f"{self._safe_name(group)}.joblib"),
            save_embedder=False,
            save_adata=False,
            compress=self.compress,
        )

    def _save_explainer(self, group, explainer, method):
        if not self.save_intermediate:
            return
        explainer.save(
            self._ckpt_path(
                "explainers", method, f"{self._safe_name(group)}.joblib"
            ),
            save_tda=False,
            compress=self.compress,
        )

    def _save_final_dataframe(self, df, method, index=False):
        if not self.save_intermediate:
            return
        df.to_csv(
            self._ckpt_path("results", f"{method}_results.csv"),
            index=index,
        )

    def save_results(self, outdir=None, which="both", per_group=False, index=False):
        """Save accumulated result tables and return written paths."""
        outdir = self.outdir if outdir is None else Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        if which not in {"local", "global", "both"}:
            raise ValueError("which must be one of {'local', 'global', 'both'}")

        paths = {}
        methods = ["local", "global"] if which == "both" else [which]
        for method in methods:
            results = self.local_results if method == "local" else self.global_results
            combined = self._concat(results.values())
            if len(combined):
                path = outdir / f"{method}_results.csv"
                combined.to_csv(path, index=index)
                paths[f"{method}_all"] = path

            if per_group:
                for group, frame in results.items():
                    if frame is None or len(frame) == 0:
                        continue
                    path = outdir / f"{method}_{self._safe_name(group)}.csv"
                    frame.to_csv(path, index=index)
                    paths[f"{method}_{group}"] = path

        if self.skipped_groups:
            path = outdir / "skipped_groups.csv"
            pd.DataFrame(
                [
                    {"group": group, "reason": reason}
                    for group, reason in self.skipped_groups.items()
                ]
            ).to_csv(path, index=False)
            paths["skipped_groups"] = path

        return paths
