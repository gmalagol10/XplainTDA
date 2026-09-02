"""Stratified XplainTDA explanations in one shared embedding space."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd


from . import Embedder
from . import defaults as d
from .XplainTDA_Explainer import Explainer
from .XplainTDA_PH import DEFAULT_PH_BACKEND, XplainTDA

class StratifiedExplainer:
    """Run PH and gene explanations separately for groups in ``adata.obs``.

    A single embedder is fitted or loaded on all cells. Each group then uses
    the corresponding rows of that shared embedding, which keeps the latent
    coordinate system comparable across groups.

    If ``embedder_path`` is supplied, ``adata.obsm[embedding_key]`` must contain
    the paired embedding. Otherwise a new embedder is fitted on the full data.
    The code never silently refits after a requested model fails to load.
    """

    def __init__(
        self,
        adata: Any,
        groupby: str,
        *,
        groups: Iterable[Any] | None = None,
        layer: str = d.DEFAULT_LAYER,
        embedding_key: str = "X_GRAE",
        embedder_path: str | Path | None = None,
        embedder: str = d.DEFAULT_EMBEDDER,
        embedder_kwargs: dict[str, Any] | None = None,
        n_components: int | None = d.DEFAULT_N_COMPONENTS,
        n_neighbors: int = d.DEFAULT_N_NEIGHBORS,
        maxdim: int = d.DEFAULT_MAXDIM,
        min_cells: int = d.DEFAULT_MIN_CELLS,
        neighbor_backend: str = d.DEFAULT_NEIGHBOR_BACKEND,
        ph_backend: str = DEFAULT_PH_BACKEND,
        dtype: np.dtype = d.DEFAULT_DTYPE,
        outdir: str | Path = "xplaintda_results",
        verbose: bool = d.DEFAULT_VERBOSE,
    ) -> None:
        if groupby not in adata.obs:
            raise KeyError(f"{groupby!r} is not a column in adata.obs")
        if not embedding_key:
            raise ValueError("embedding_key must be a non-empty string")

        available = list(pd.Series(adata.obs[groupby]).dropna().unique())
        self.groups = available if groups is None else list(groups)
        unknown = [group for group in self.groups if group not in set(available)]
        if unknown:
            raise ValueError(f"unknown groups for {groupby!r}: {unknown}")

        self.adata = adata
        self.groupby = groupby
        self.layer = layer
        self.embedding_key = str(embedding_key)
        self.embedder_path = (
            None
            if embedder_path is None
            else Path(embedder_path).expanduser().resolve()
        )
        self.embedder_name = str(embedder).upper()
        self.embedder_kwargs = dict(embedder_kwargs or {})
        self.n_components = n_components
        self.n_neighbors = int(n_neighbors)
        self.maxdim = int(maxdim)
        self.min_cells = int(min_cells)
        self.neighbor_backend = neighbor_backend
        self.ph_backend = ph_backend
        self.dtype = dtype
        self.outdir = Path(outdir).expanduser().resolve()
        self.verbose = bool(verbose)

        if self.min_cells < 1:
            raise ValueError("min_cells must be positive")
        if layer != "X" and layer not in adata.layers:
            raise KeyError(f"layer {layer!r} is not present in adata.layers")

        self.X = adata.X if layer == "X" else adata.layers[layer]
        self.n_cells, self.n_genes = map(int, self.X.shape)
        self.full_tda: XplainTDA | None = None
        self.tdas: dict[Any, XplainTDA] = {}
        self.results_by_group: dict[Any, pd.DataFrame] = {}
        self.skipped_groups: dict[Any, str] = {}
        self.outdir.mkdir(parents=True, exist_ok=True)

    def _load_embedder(self) -> Any:
        """Load a saved embedder using its native loader when available."""
        if self.embedder_path is None or not self.embedder_path.is_file():
            raise FileNotFoundError(f"embedder not found: {self.embedder_path}")
        try:
            embedder_class = Embedder.Embedder.registry[self.embedder_name]
        except KeyError as exc:
            raise ValueError(
                f"unknown embedder {self.embedder_name!r}; available: "
                f"{sorted(Embedder.Embedder.registry)}"
            ) from exc
        return (
            embedder_class.load(self.embedder_path)
            if hasattr(embedder_class, "load")
            else joblib.load(self.embedder_path)
        )

    def _validate_loaded_pair(self, embedder: Any, embedding: Any) -> np.ndarray:
        """Check that a saved model and stored embedding match this AnnData."""
        embedding = np.asarray(embedding, dtype=self.dtype)
        if embedding.ndim != 2 or embedding.shape[0] != self.n_cells:
            raise ValueError(
                f"stored embedding must have {self.n_cells} rows; got {embedding.shape}"
            )
        if not np.isfinite(embedding).all():
            raise ValueError("stored embedding contains NaN or infinite values")
        if int(embedder.n_components) != embedding.shape[1]:
            raise ValueError(
                "saved embedder and stored embedding have different latent dimensions"
            )

        shape = getattr(getattr(embedder, "model", None), "data_shape", None)
        if shape is not None:
            try:
                input_dim = int(np.atleast_1d(shape)[-1])
            except (TypeError, ValueError):
                input_dim = None
            if input_dim is not None and input_dim != self.n_genes:
                raise ValueError(
                    "saved embedder was fitted with a different number of features"
                )
        return embedding

    def fit_embedding(self) -> "StratifiedExplainer":
        """Load a paired embedding/model or fit one embedder on all cells."""
        if self.full_tda is not None:
            return self

        if self.embedder_path is not None:
            if self.embedding_key not in self.adata.obsm:
                raise KeyError(
                    f"adata.obsm[{self.embedding_key!r}] is required with embedder_path"
                )
            fitted_embedder = self._load_embedder()
            embedding = self._validate_loaded_pair(
                fitted_embedder, self.adata.obsm[self.embedding_key]
            )
            self.full_tda = XplainTDA(
                adata=self.adata,
                layer=self.layer,
                embedder=self.embedder_name,
                n_components=embedding.shape[1],
                n_neighbors=self.n_neighbors,
                maxdim=self.maxdim,
                neighbor_backend=self.neighbor_backend,
                ph_backend=self.ph_backend,
                dtype=self.dtype,
                fitted_embedder=fitted_embedder,
            ).fit_embedding(embedding=embedding)
            source = "loaded"
        else:
            self.full_tda = XplainTDA(
                adata=self.adata,
                layer=self.layer,
                embedder=self.embedder_name,
                n_components=self.n_components,
                n_neighbors=self.n_neighbors,
                maxdim=self.maxdim,
                neighbor_backend=self.neighbor_backend,
                ph_backend=self.ph_backend,
                dtype=self.dtype,
                embedder_kwargs=self.embedder_kwargs,
            ).fit_embedding()
            self.adata.obsm[self.embedding_key] = self.full_tda.embedding
            source = "fitted"

        self._save_embedding_artifacts(source)
        return self

    def _save_embedding_artifacts(self, source: str) -> None:
        """Save the shared embedding, model, and run metadata once."""
        directory = self.outdir / "embedding"
        directory.mkdir(parents=True, exist_ok=True)
        embedding_name = self._safe_name(self.embedding_key)
        np.save(directory / f"{embedding_name}.npy", self.full_tda.embedding)
        model_path = self.full_tda.save_embedder(directory / "embedder.joblib")
        metadata = {
            "source": source,
            "embedding_key": self.embedding_key,
            "embedding_shape": list(self.full_tda.embedding.shape),
            "embedder": self.embedder_name,
            "model_path": str(model_path),
            "layer": self.layer,
            "groupby": self.groupby,
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

    def _group_mask(self, group: Any) -> np.ndarray:
        """Return the Boolean cell mask for one group."""
        return np.asarray(self.adata.obs[self.groupby] == group)

    def fit_group(self, group: Any) -> XplainTDA | None:
        """Compute reference PH for one group in the shared embedding space."""
        if group in self.tdas:
            return self.tdas[group]
        self.fit_embedding()
        mask = self._group_mask(group)
        n_cells = int(mask.sum())
        if n_cells < self.min_cells:
            self.skipped_groups[group] = f"n_cells={n_cells} < min_cells={self.min_cells}"
            if self.verbose:
                print(f"Skipping {group}: {self.skipped_groups[group]}", flush=True)
            return None

        adata_group = self.adata[mask].copy()
        embedding = np.asarray(self.full_tda.embedding[mask], dtype=self.dtype)
        tda = XplainTDA(
            adata=adata_group,
            layer=self.layer,
            embedder=self.embedder_name,
            n_components=embedding.shape[1],
            n_neighbors=min(self.n_neighbors, n_cells - 1),
            maxdim=self.maxdim,
            neighbor_backend=self.neighbor_backend,
            ph_backend=self.ph_backend,
            dtype=self.dtype,
            fitted_embedder=self.full_tda.embedder,
        ).fit(embedding=embedding)
        self.tdas[group] = tda
        tda.save(self._path("tda", f"{self._safe_name(group)}.joblib"), include_embedder=False)
        return tda

    def fit(self) -> "StratifiedExplainer":
        """Compute reference PH for every requested group."""
        self.fit_embedding()
        for group in self.groups:
            self.fit_group(group)
        return self

    def explain(
        self,
        genes: Iterable[str | int] | str | int | None = None,
        dims: Iterable[int] | int | None = None,
        *,
        perturbation: str = d.DEFAULT_PERTURBATION,
        n_jobs: int | None = None,
        backend: str = d.DEFAULT_BACKEND,
        random_state: int | None = d.DEFAULT_RANDOM_STATE,
    ) -> pd.DataFrame:
        """Explain all requested groups and save per-group and combined tables."""
        tables = []
        for group in self.groups:
            if self.verbose:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"Explaining {group} | {timestamp}", flush=True)
            tda = self.fit_group(group)
            if tda is None:
                continue
            result = Explainer(
                tda,
                perturbation=perturbation,
                random_state=random_state,
                verbose=self.verbose,
            ).explain(genes=genes, dims=dims, n_jobs=n_jobs, backend=backend)
            result.insert(0, "group", group)
            result.insert(1, "n_cells", tda.n_cells)
            result.insert(2, "n_neighbors", tda.n_neighbors)
            self.results_by_group[group] = result
            tables.append(result)
            result.to_csv(
                self._path("results", "groups", f"{self._safe_name(group)}.csv"),
                index=False,
            )

        combined = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
        combined.to_csv(self._path("results", "explanations.csv"), index=False)
        self._save_run_metadata(perturbation, genes, dims, n_jobs, backend, random_state)
        self._save_skipped_groups()
        return combined

    def _save_run_metadata(
        self,
        perturbation: str,
        genes: Any,
        dims: Any,
        n_jobs: int | None,
        backend: str,
        random_state: int | None,
    ) -> None:
        """Write the effective analysis configuration as JSON."""
        config = {
            "groupby": self.groupby,
            "groups": self.groups,
            "layer": self.layer,
            "embedding_key": self.embedding_key,
            "embedder": self.embedder_name,
            "n_components": self.full_tda.n_components,
            "n_neighbors": self.n_neighbors,
            "maxdim": self.maxdim,
            "min_cells": self.min_cells,
            "neighbor_backend": self.neighbor_backend,
            "ph_backend": self.ph_backend,
            "perturbation": perturbation,
            "quantile": Explainer.quantile,
            "genes": genes,
            "dims": dims,
            "n_jobs": n_jobs,
            "joblib_backend": backend,
            "random_state": random_state,
        }
        self._path("config.json").write_text(json.dumps(config, indent=2, default=str))

    def _save_skipped_groups(self) -> None:
        """Save skipped-group reasons when at least one group was omitted."""
        if self.skipped_groups:
            pd.DataFrame(
                [
                    {"group": group, "reason": reason}
                    for group, reason in self.skipped_groups.items()
                ]
            ).to_csv(self._path("results", "skipped_groups.csv"), index=False)

    @staticmethod
    def _safe_name(value: Any) -> str:
        """Convert an annotation value to a filesystem-safe name."""
        return str(value).replace("/", "_").replace("\\", "_").replace(" ", "_")

    def _path(self, *parts: str) -> Path:
        """Build an output path and create its parent directory."""
        path = self.outdir.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


# Backward-compatible class alias; new code should use StratifiedExplainer.
StratifiedXplainTDA = StratifiedExplainer
