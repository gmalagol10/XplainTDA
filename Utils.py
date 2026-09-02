"""General utilities used by XplainTDA data-preparation notebooks.

Heavy single-cell dependencies are imported lazily so the module remains
importable in environments that only need the numerical helpers.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Sequence

import numpy as np
from scipy import sparse


COLORS = (
    (0.34550725069638827, 0.4203708006658883, 0.9696902293486781),
    (0.9893800026041992, 0.378955911742755, 0.21756841368122667),
    (0.3959642074605608, 0.24823947872676938, 0.4016676539297192),
    (0.9937826924994482, 0.4211527500079969, 0.8812994030921271),
    (0.4140058397372807, 0.9619317608252869, 0.3109026417629064),
    (0.2286247431221609, 0.6437632542888629, 0.4081322805120583),
    (0.25003260615661993, 0.938691496932296, 0.9192515923797947),
    (0.7646511697684856, 0.24254983894398235, 0.7085129830496552),
    (0.3017721221187747, 0.6522700618245787, 0.9844707721904342),
    (0.21782892631529166, 0.2854088109905996, 0.7174819557293214),
    (0.9991896546476063, 0.4844986464266022, 0.5344476773522967),
    (0.8919236560192338, 0.7949117963224906, 0.7730486745511909),
    (0.6428145648345002, 0.31108252586505475, 0.2041098261347507),
    (0.20269999368314223, 0.7748472379028853, 0.6824940025160084),
    (0.9819923318374866, 0.6965816490867496, 0.21263663000131983),
    (0.809020033101876, 0.23056728504993104, 0.9541856467035792),
    (0.9191255237720919, 0.20589563833687236, 0.4656154484972612),
    (0.4066816185888487, 0.7781338620666193, 0.22302116197384453),
    (0.9393586762460103, 0.9614694589148117, 0.22619331869433382),
    (0.29226458048518345, 0.42830335364029093, 0.6916335500700534),
)

colors_to_use_pastel=[(0.8246194 , 0.98468392, 0.50898921),
       (0.99265112, 0.65946808, 0.70062234),
       (0.72566147, 0.76371323, 0.99342785),
       (0.5719038 , 0.90206697, 0.57889311),
       (0.51246015, 0.97503846, 0.96976219),
       (0.51137255, 0.86574042, 0.77288782),
       (0.75701626, 0.53117515, 0.67286038),
       (0.58101604, 0.65251402, 0.79454428),
       (0.54298704, 0.75071957, 0.61247655),
       (0.99490121, 0.7553237 , 0.5055095 ),
       (0.99835163, 0.79840229, 0.73508254),
       (0.50846959, 0.50933663, 0.9990437 ),
       (0.99946801, 0.54018382, 0.53782693),
       (0.99881609, 0.7071878 , 0.89790902),
       (0.60137108, 0.67641061, 0.98955794),
       (0.56655168, 0.50371389, 0.50752152),
       (0.97285983, 0.50131345, 0.84696043),
       (0.50529725, 0.54441868, 0.78129649),
       (0.98157065, 0.95574324, 0.95781627)]

def _axis_max(X: Any, axis: int) -> np.ndarray:
    """Return dense axis-wise maxima for dense or sparse matrices."""
    maximum = X.max(axis=axis)
    return maximum.toarray().ravel() if sparse.issparse(maximum) else np.asarray(maximum).ravel()


def qc_filtering(adata: Any, omic: str = "ATAC") -> Any:
    """Apply the project's quantile-based QC and feature-selection workflow.

    Parameters
    ----------
    adata
        AnnData object containing raw GEX or ATAC values.
    omic
        ``"GEX"`` or ``"ATAC"``.

    Returns
    -------
    AnnData
        A filtered copy with library-size normalization and square-root
        transformation applied.
    """
    try:
        import episcanpy as epi
        import scanpy as sc
    except ImportError as exc:
        raise ImportError("qc_filtering requires scanpy and episcanpy") from exc

    omic = omic.upper()
    if omic not in {"GEX", "ATAC"}:
        raise ValueError("omic must be 'GEX' or 'ATAC'")

    adata = adata.copy()
    adata.var_names_make_unique()
    epi.pp.qc_stats(adata, verbose=False)

    if omic == "GEX":
        adata.var["MT"] = adata.var_names.str.startswith(("MT", "mt"))
        sc.pp.calculate_qc_metrics(
            adata, qc_vars=["MT"], percent_top=None, log1p=False, inplace=True
        )
        sc.pp.scrublet(adata)

    cell_limits = 10 ** np.quantile(adata.obs["log_n_features"], [0.05, 0.95])
    feature_limits = 10 ** np.quantile(adata.var["log_n_cells"], [0.05, 0.95])
    epi.pp.set_filter(
        adata,
        "n_features",
        min_threshold=cell_limits[0],
        max_threshold=cell_limits[1],
        verbose=False,
    )
    epi.pp.set_filter(
        adata,
        "n_cells",
        min_threshold=feature_limits[0],
        max_threshold=feature_limits[1],
        verbose=False,
    )
    adata = epi.pp.apply_filters(adata, verbose=False).copy()

    if omic == "ATAC" and {"nucleosome_signal", "tss_enrichment_score"} <= set(adata.obs):
        adata = adata[
            (adata.obs["nucleosome_signal"] < 2)
            & (adata.obs["tss_enrichment_score"] > 2)
        ].copy()
    elif omic == "GEX":
        max_mt = np.quantile(adata.obs["pct_counts_MT"].dropna(), 0.9)
        adata = adata[
            (adata.obs["pct_counts_MT"] <= max_mt)
            & (~adata.obs["predicted_doublet"].astype(bool))
        ].copy()

    sc.pp.highly_variable_genes(adata, flavor="seurat_v3")
    threshold = np.quantile(adata.var["variances_norm"], 0.9 if omic == "GEX" else 0.8)
    adata = adata[:, adata.var["variances_norm"] > threshold].copy()
    if adata.n_vars > 30_000:
        top = adata.var["variances_norm"].nlargest(30_000).index
        adata = adata[:, top].copy()

    sc.pp.normalize_total(adata)
    adata.X = adata.X.sqrt() if sparse.issparse(adata.X) else np.sqrt(adata.X)
    adata = adata[:, _axis_max(adata.X, 0) > 0].copy()
    return adata[_axis_max(adata.X, 1) > 0].copy()


def intersection(collections: Sequence[Iterable[Any]]) -> np.ndarray:
    """Return values present in every collection, preserving first-list order."""
    if len(collections) == 0:
        return np.array([])
    common = set(collections[0]).intersection(*map(set, collections[1:]))
    return np.asarray(list(dict.fromkeys(x for x in collections[0] if x in common)))


def flatten_unique(collections: Iterable[Iterable[Any]]) -> np.ndarray:
    """Flatten nested iterables and remove duplicates while preserving order."""
    return np.asarray(
        list(dict.fromkeys(item for collection in collections for item in collection))
    )


def most_common(values: Iterable[Any]) -> Any:
    """Return the most frequent value, breaking ties by first occurrence."""
    values = list(values)
    if not values:
        raise ValueError("values must not be empty")
    counts = Counter(values)
    return max(values, key=lambda value: counts[value])


def evaluate_polynomial(coefficients: Sequence[float], x: Any) -> np.ndarray:
    """Evaluate descending-order polynomial coefficients at ``x``."""
    if len(coefficients) == 0:
        raise ValueError("coefficients must not be empty")
    return np.polyval(np.asarray(coefficients), x)


def procrustes(
    X: Any,
    Y: Any,
    scaling: bool = True,
    reflection: str | bool = "best",
) -> tuple[float, np.ndarray, dict[str, np.ndarray | float]]:
    """Align ``Y`` to ``X`` with translation, rotation, and optional scaling.

    Returns the normalized residual, transformed coordinates, and a transform
    dictionary containing ``rotation``, ``scale``, and ``translation``.
    """
    X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must be 2D arrays with the same number of rows")
    if reflection not in {"best", True, False}:
        raise ValueError("reflection must be 'best', True, or False")

    n_points, x_dim = X.shape
    y_dim = Y.shape[1]
    mean_X, mean_Y = X.mean(axis=0), Y.mean(axis=0)
    centered_X, centered_Y = X - mean_X, Y - mean_Y
    norm_X, norm_Y = np.linalg.norm(centered_X), np.linalg.norm(centered_Y)
    if norm_X == 0 or norm_Y == 0:
        raise ValueError("X and Y must each contain more than one distinct point")

    normalized_X, normalized_Y = centered_X / norm_X, centered_Y / norm_Y
    if y_dim < x_dim:
        normalized_Y = np.c_[normalized_Y, np.zeros((n_points, x_dim - y_dim))]
    elif y_dim > x_dim:
        raise ValueError("Y cannot have more columns than X")

    U, singular_values, Vt = np.linalg.svd(normalized_X.T @ normalized_Y, full_matrices=False)
    V = Vt.T
    rotation = V @ U.T
    if reflection != "best" and (np.linalg.det(rotation) < 0) != reflection:
        V[:, -1] *= -1
        singular_values[-1] *= -1
        rotation = V @ U.T

    trace = singular_values.sum()
    if scaling:
        scale = trace * norm_X / norm_Y
        residual = 1 - trace**2
        transformed = norm_X * trace * normalized_Y @ rotation + mean_X
    else:
        scale = 1.0
        residual = 1 + norm_Y**2 / norm_X**2 - 2 * trace * norm_Y / norm_X
        transformed = norm_Y * normalized_Y @ rotation + mean_X

    rotation = rotation[:y_dim]
    translation = mean_X - scale * mean_Y @ rotation
    return float(residual), transformed, {
        "rotation": rotation,
        "scale": float(scale),
        "translation": translation,
    }


def to_dense_float32(X: Any) -> np.ndarray:
    """Convert dense or sparse input to a dense ``float32`` array."""
    return X.toarray().astype(np.float32) if sparse.issparse(X) else np.asarray(X, dtype=np.float32)


# Compatibility aliases for existing notebooks.
colors_to_use = COLORS
flat_list = flatten_unique
my_polynom = evaluate_polynomial
