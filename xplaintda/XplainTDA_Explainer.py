"""Gene-level explanations for fitted XplainTDA persistence diagrams.

For each gene, expression is perturbed across all cells, the fixed embedder is
reused, persistent homology is recomputed, and the impact is measured by the
95% quantile bottleneck distance. Work is parallelized over genes.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import issparse
from sklearn.metrics import pairwise_distances_chunked

from . import defaults as d

VALID_PERTURBATIONS = {"zero"}
BOTTLENECK_QUANTILE = 0.95


def _clean_diagram(diagram: Any) -> np.ndarray:
    """Keep finite persistence points with positive lifetime."""
    diagram = np.asarray(diagram, dtype=float).reshape(-1, 2)
    return diagram[
        np.isfinite(diagram).all(axis=1) & (diagram[:, 1] > diagram[:, 0])
    ]


def _linf_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise L-infinity distances between persistence points."""
    return np.maximum(
        np.abs(a[:, None, 0] - b[None, :, 0]),
        np.abs(a[:, None, 1] - b[None, :, 1]),
    )


def quantile_bottleneck_distance(
    reference: Any, perturbed: Any, q: float = BOTTLENECK_QUANTILE
) -> float:
    """Return the smallest matching threshold covering a fraction ``q`` of points.

    A point matched to another diagram point contributes two covered points;
    a point matched to the diagonal contributes one. At ``q=1`` this reduces
    to the usual bottleneck threshold for the augmented diagrams.
    """
    if not 0 < q <= 1:
        raise ValueError("q must satisfy 0 < q <= 1")

    reference, perturbed = _clean_diagram(reference), _clean_diagram(perturbed)
    n_reference, n_perturbed = len(reference), len(perturbed)
    n_points = n_reference + n_perturbed
    if n_points == 0:
        return 0.0

    reference_diagonal = (reference[:, 1] - reference[:, 0]) / 2
    perturbed_diagonal = (perturbed[:, 1] - perturbed[:, 0]) / 2
    cross = (
        _linf_distances(reference, perturbed)
        if n_reference and n_perturbed
        else np.empty((n_reference, n_perturbed))
    )
    thresholds = np.unique(
        np.r_[0.0, cross.ravel(), reference_diagonal, perturbed_diagonal]
    )
    required = int(np.ceil(q * n_points))

    def covered_points(threshold: float) -> int:
        size = n_reference + n_perturbed
        valid = np.zeros((size, size), dtype=bool)
        covered = np.zeros((size, size), dtype=int)

        valid[:n_reference, :n_perturbed] = True
        covered[:n_reference, :n_perturbed] = 2 * (cross <= threshold)

        rows = np.arange(n_reference)
        valid[rows, n_perturbed + rows] = True
        covered[rows, n_perturbed + rows] = reference_diagonal <= threshold

        rows = np.arange(n_perturbed)
        valid[n_reference + rows, rows] = True
        covered[n_reference + rows, rows] = perturbed_diagonal <= threshold

        # Dummy diagonal points complete the square assignment.
        valid[n_reference:, n_perturbed:] = True
        cost = np.where(valid, -covered, n_points + 1)
        row_ind, col_ind = linear_sum_assignment(cost)
        return int(covered[row_ind, col_ind].sum())

    low, high = 0, len(thresholds) - 1
    while low < high:
        middle = (low + high) // 2
        if covered_points(thresholds[middle]) >= required:
            high = middle
        else:
            low = middle + 1
    return float(thresholds[low])


def _gene_values(X: Any, gene_idx: int) -> np.ndarray:
    """Extract one gene as a dense one-dimensional float array."""
    values = X[:, gene_idx]
    values = values.toarray().ravel() if issparse(values) else np.asarray(values).ravel()
    return values.astype(np.float32, copy=False)


def _replace_gene(X: Any, gene_idx: int, values: np.ndarray) -> Any:
    """Copy an expression matrix and replace one complete gene column."""
    perturbed = X.copy()
    if issparse(perturbed):
        perturbed = perturbed.tolil(copy=False)
        perturbed[:, gene_idx] = values[:, None]
        return perturbed.tocsr()
    perturbed[:, gene_idx] = values
    return perturbed


def _linear_loadings(tda: Any) -> np.ndarray | None:
    """Return feature loadings for PCA/TruncatedSVD-like embedders."""
    model = getattr(getattr(tda, "embedder", None), "model", None)
    if not hasattr(model, "components_"):
        return None
    return np.asarray(model.components_.T, dtype=np.float32)


def perturb_gene_embedding(
    tda: Any,
    gene_idx: int,
    perturbation: str = "zero",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Perturb one gene across all cells and return the new embedding.

    Linear embedders use an exact rank-one update. Nonlinear embedders receive
    a copied expression matrix and are only transformed, never refitted.
    """
    if perturbation not in VALID_PERTURBATIONS:
        raise ValueError(f"perturbation must be one of {sorted(VALID_PERTURBATIONS)}")
    if tda.X is None:
        raise ValueError("tda.X is missing; load the TDA object with data attached")
    if tda.embedder is None:
        raise ValueError("a fitted embedder must be attached to the TDA object")

    original = _gene_values(tda.X, gene_idx)
    if perturbation == "zero":
        changed = np.zeros_like(original)
    else:
        changed = original.copy()
        (rng or np.random.default_rng()).shuffle(changed)

    loadings = _linear_loadings(tda)
    if loadings is not None:
        embedding = tda.embedding.copy()
        embedding += (changed - original)[:, None] * loadings[gene_idx]
        return embedding
    return tda.embed(_replace_gene(tda.X, gene_idx, changed))


def _available_cpus() -> int:
    """Resolve a scheduler-aware default worker count."""
    for name in ("XPLAINTDA_N_JOBS", "SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        if value := os.environ.get(name):
            return max(1, int(value))
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


class Explainer:
    """Measure gene importance by q=0.95 bottleneck loss.

    Parameters
    ----------
    tda
        Fitted :class:`XplainTDA_PH.XplainTDA` with expression data, a fitted
        embedder, an embedding, and reference persistence diagrams.
    perturbation
        ``"zero"`` sets a gene to zero; ``"permute"`` shuffles it across cells.
    random_state
        Base seed for reproducible per-gene permutations.
    verbose
        Print one progress line per completed gene.
    """

    quantile = BOTTLENECK_QUANTILE

    def __init__(
        self,
        tda: Any,
        perturbation: str = d.DEFAULT_PERTURBATION,
        random_state: int | None = d.DEFAULT_RANDOM_STATE,
        verbose: bool = d.DEFAULT_VERBOSE,
    ) -> None:
        if perturbation not in VALID_PERTURBATIONS:
            raise ValueError(f"perturbation must be one of {sorted(VALID_PERTURBATIONS)}")
        missing = [
            name
            for name in ("X", "embedding", "dgms", "embedder", "var_names")
            if getattr(tda, name, None) is None
        ]
        if missing:
            raise ValueError(f"fitted TDA object is missing: {', '.join(missing)}")

        self.tda = tda
        self.perturbation = perturbation
        self.random_state = random_state
        self.verbose = bool(verbose)
        self.results: pd.DataFrame | None = None

        self.diameter = max(float(np.max(block)) for block in pairwise_distances_chunked(self.tda.embedding, metric="euclidean"))
        if not np.isfinite(self.diameter) or self.diameter <= 0:
            raise ValueError("embedding diameter must be positive and finite")

    def _resolve_genes(
        self, genes: Iterable[str | int] | str | int | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resolve gene names or positions to aligned indices and names."""
        names = np.asarray(self.tda.var_names).astype(str)
        if genes is None:
            indices = np.arange(len(names), dtype=int)
            return indices, names
        genes = [genes] if isinstance(genes, (str, int, np.integer)) else list(genes)
        if not genes:
            return np.empty(0, dtype=int), np.empty(0, dtype=str)
        if isinstance(genes[0], str):
            lookup = pd.Index(names).get_indexer(genes)
            missing = np.asarray(genes, dtype=str)[lookup < 0]
            if len(missing):
                raise ValueError(f"unknown genes: {missing.tolist()}")
            return lookup.astype(int), np.asarray(genes, dtype=str)
        indices = np.asarray(genes, dtype=int)
        if np.any((indices < 0) | (indices >= len(names))):
            raise IndexError(f"gene indices must be between 0 and {len(names) - 1}")
        return indices, names[indices]

    def _resolve_dims(self, dims: Iterable[int] | int | None) -> list[int]:
        """Return validated homology dimensions, defaulting to all diagrams."""
        available = list(range(len(self.tda.dgms)))
        dims = available if dims is None else [int(x) for x in np.atleast_1d(dims)]
        invalid = sorted(set(dims) - set(available))
        if invalid:
            raise ValueError(
                f"unavailable homology dimensions: {invalid}; available: {available}"
            )
        return dims

    def _rng(self, gene_idx: int) -> np.random.Generator:
        """Create a deterministic per-gene generator when a seed is provided."""
        seed = None if self.random_state is None else int(self.random_state) + int(gene_idx)
        return np.random.default_rng(seed)

    def score_gene(
        self, gene_idx: int, dims: Iterable[int] | int | None = None
    ) -> list[tuple[int, float]]:
        """Perturb one gene and score the requested homology dimensions."""
        dims = self._resolve_dims(dims)
        embedding = perturb_gene_embedding(
            self.tda, gene_idx, self.perturbation, self._rng(gene_idx)
        )
        perturbed_diagrams = self.tda.recompute_ph(embedding)
        return [
            (
                dim,
                quantile_bottleneck_distance(
                    self.tda.dgms[dim], perturbed_diagrams[dim], self.quantile
                ) / self.diameter,
            )
            for dim in dims
        ]

    def _score_gene_rows(self, gene_idx: int, gene: str, dims: list[int]) -> list[dict[str, Any]]:
        """Joblib worker returning one tidy row per homology dimension."""
        rows = [
            {
                "homology_dim": dim,
                "gene_idx": int(gene_idx),
                "gene": gene,
                "score": score,
                "quantile": self.quantile,
            }
            for dim, score in self.score_gene(gene_idx, dims)
        ]
        if self.verbose:
            print(
                f"Explained {gene} ({gene_idx}) | "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')}",
                flush=True,
            )
        return rows

    def explain(
        self,
        genes: Iterable[str | int] | str | int | None = None,
        dims: Iterable[int] | int | None = None,
        *,
        n_jobs: int | None = None,
        backend: str = d.DEFAULT_BACKEND,
    ) -> pd.DataFrame:
        """Score genes in parallel and return a ranked tidy table.

        Parallelization is over genes. Use ``n_jobs=1`` when the embedder runs
        on a GPU or does not support concurrent inference.
        """
        gene_indices, gene_names = self._resolve_genes(genes)
        dims = self._resolve_dims(dims)
        if len(gene_indices) == 0 or len(dims) == 0:
            self.results = pd.DataFrame(
                columns=["homology_dim", "gene_idx", "gene", "score", "quantile", "rank"]
            )
            return self.results

        n_jobs = _available_cpus() if n_jobs is None else int(n_jobs)
        if n_jobs != -1:
            n_jobs = max(1, min(n_jobs, len(gene_indices)))
        nested = joblib.Parallel(n_jobs=n_jobs, backend=backend)(
            joblib.delayed(self._score_gene_rows)(idx, gene, dims)
            for idx, gene in zip(gene_indices, gene_names)
        )
        result = pd.DataFrame(row for rows in nested for row in rows)
        result["rank"] = (
            result.groupby("homology_dim")["score"]
            .rank(method="dense", ascending=False)
            .astype(int)
        )
        self.results = result.sort_values(
            ["homology_dim", "rank", "gene_idx"], kind="stable"
        ).reset_index(drop=True)
        return self.results

    def save_results(self, path: str | Path, index: bool = False) -> Path:
        """Save the latest explanation table as CSV."""
        if self.results is None:
            raise ValueError("no results are available; call explain() first")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.results.to_csv(path, index=index)
        return path


# Lightweight aliases for code written before the naming cleanup.
XplainTDAExplainer = Explainer
Global_XplainTDA = Explainer
