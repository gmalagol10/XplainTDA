import numpy as np
import networkx as nx
import gudhi as gd
import joblib, time

from scipy import sparse
from scipy.optimize import linear_sum_assignment

import Embedder
import defaults as d

from sklearn.neighbors import NearestNeighbors

try:
    from pynndescent import NNDescent
    HAS_PYNNDESCENT = True
except ImportError:
    HAS_PYNNDESCENT = False

try:
    from ripser import ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False

VALID_PH_BACKENDS = {"gudhi", "ripser"}


# =========================================================
# kNN graph construction
# =========================================================
def compute_neighbors(X, n_neighbors=d.DEFAULT_N_NEIGHBORS, backend=d.DEFAULT_NEIGHBOR_BACKEND):
    """
    Compute k-nearest neighbors in an embedding.

    Parameters
    ----------
    X : np.ndarray, shape (n_cells, n_features)
        Cell embedding.
    n_neighbors : int, default=defaults.DEFAULT_N_NEIGHBORS
        Number of nearest neighbors. Clamped to ``n_cells``.
    backend : {"exact", "pynndescent"}, default=defaults.DEFAULT_NEIGHBOR_BACKEND
        Neighbor search backend.

    Returns
    -------
    neigh_ind : np.ndarray
        Neighbor indices.
    neigh_dist : np.ndarray
        Neighbor distances.
    """
    n_neighbors = min(int(n_neighbors), X.shape[0])

    if backend == "exact":
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", n_jobs=-1)
        neigh_dist, neigh_ind = nn.fit(X).kneighbors(X)

    elif backend == "pynndescent":
        if not HAS_PYNNDESCENT:
            raise ImportError("pynndescent is not installed.")
        neigh_ind, neigh_dist = NNDescent(X, n_neighbors=n_neighbors, metric="euclidean").neighbor_graph

    else:
        raise ValueError("backend must be 'exact' or 'pynndescent'")

    return neigh_ind, neigh_dist


# =========================================================
# Graph utilities shared by GUDHI and Ripser backends
# =========================================================
def neighbors_to_edge_list(neigh_ind, neigh_dist):
    """
    Convert directed neighbor arrays into an undirected weighted edge list.

    Duplicate directed edges are collapsed exactly as in ``build_simplex_tree``:
    the undirected edge ``{i, j}`` keeps the minimum observed distance.
    """
    edge_dict = {}
    for i, (nbrs, dists) in enumerate(zip(neigh_ind, neigh_dist)):
        for j, dist in zip(nbrs, dists):
            if i == j:
                continue
            edge = tuple(sorted((int(i), int(j))))
            edge_dict[edge] = min(float(dist), edge_dict.get(edge, np.inf))

    return (
        np.array([[i, j, dist] for (i, j), dist in edge_dict.items()], dtype=float)
        if edge_dict else np.empty((0, 3), dtype=float)
    )


def edge_list_to_sparse_distance(edge_list, n):
    """
    Convert an undirected edge list into a symmetric sparse distance matrix.

    Missing entries remain absent in the sparse matrix; Ripser interprets them
    as distances above the sparse threshold / infinity.
    """
    if edge_list.size == 0:
        return sparse.csr_matrix((n, n), dtype=float)

    ij = edge_list[:, :2].astype(int)
    vals = edge_list[:, 2].astype(float, copy=False)
    rows = np.concatenate([ij[:, 0], ij[:, 1]])
    cols = np.concatenate([ij[:, 1], ij[:, 0]])
    data = np.concatenate([vals, vals])
    return sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def ripser_sparse_to_diagrams(distance_matrix, maxdim=d.DEFAULT_MAXDIM, filtration_end=None):
    """Compute persistence diagrams from a sparse distance matrix using Ripser."""
    if not HAS_RIPSER:
        raise ImportError("ripser is not installed. Install ripser to use ph_backend='ripser'.")

    result = ripser(distance_matrix, maxdim=maxdim, distance_matrix=True)
    dgms = [np.asarray(dgm, dtype=float).copy() for dgm in result["dgms"]]

    if filtration_end is None:
        filtration_end = float(distance_matrix.data.max()) if distance_matrix.nnz else 0.0

    for dgm in dgms:
        if len(dgm):
            dgm[~np.isfinite(dgm[:, 1]), 1] = filtration_end

    return dgms


def compute_ph_from_neighbors(neigh_ind, neigh_dist, maxdim=d.DEFAULT_MAXDIM, ph_backend="gudhi", localize=True):
    """
    Compute PH from a kNN graph using either the GUDHI or Ripser backend.

    ``gudhi`` explicitly builds a SimplexTree. ``ripser`` computes diagrams
    from a sparse distance matrix and uses the same edge-list heuristic for
    approximate feature localization.
    """
    ph_backend = ph_backend.lower()
    if ph_backend not in VALID_PH_BACKENDS:
        raise ValueError(f"ph_backend must be one of {sorted(VALID_PH_BACKENDS)}")

    if ph_backend == "gudhi":
        st, edge_list = build_simplex_tree(neigh_ind, neigh_dist, maxdim=maxdim)
        dgms = simplex_tree_to_diagrams(st, maxdim=maxdim)
    else:
        edge_list = neighbors_to_edge_list(neigh_ind, neigh_dist)
        D = edge_list_to_sparse_distance(edge_list, n=neigh_ind.shape[0])
        dgms = ripser_sparse_to_diagrams(D, maxdim=maxdim)

    features = localize_features(dgms, edge_list, maxdim=maxdim) if localize else {dim: [] for dim in range(maxdim + 1)}
    return dgms, features, edge_list


# =========================================================
# Build weighted clique complex
# =========================================================
def build_simplex_tree(neigh_ind, neigh_dist, maxdim=d.DEFAULT_MAXDIM):
    """
    Build a weighted clique complex from a kNN graph.

    Parameters
    ----------
    neigh_ind : np.ndarray
        Neighbor indices.
    neigh_dist : np.ndarray
        Neighbor distances.
    maxdim : int, default=defaults.DEFAULT_MAXDIM
        Maximum homology dimension.

    Returns
    -------
    st : gudhi.SimplexTree
        Filtered simplex tree.
    edge_list : np.ndarray, shape (n_edges, 3)
        Edge table with columns ``source, target, filtration``.
    """
    n = neigh_ind.shape[0]
    st = gd.SimplexTree()

    for i in range(n):
        st.insert([i], filtration=0.0)

    edge_list = neighbors_to_edge_list(neigh_ind, neigh_dist)

    for i, j, dist in edge_list:
        st.insert([int(i), int(j)], filtration=float(dist))

    st.expansion(maxdim + 1)

    return st, edge_list


# =========================================================
# Persistence diagrams
# =========================================================
def simplex_tree_to_diagrams(st, maxdim=d.DEFAULT_MAXDIM, filtration_end=None):
    """
    Compute persistence diagrams and cap infinite deaths.

    Parameters
    ----------
    st : gudhi.SimplexTree
        Filtered simplex tree.
    maxdim : int, default=defaults.DEFAULT_MAXDIM
        Maximum homology dimension.
    filtration_end : float, optional
        Value used to replace infinite deaths. If None, uses the largest
        filtration value in the simplex tree.

    Returns
    -------
    dgms : list[np.ndarray]
        Persistence diagrams for dimensions ``0, ..., maxdim``.
    """
    st.compute_persistence()

    if filtration_end is None:
        filtration_end = max(f for _, f in st.get_filtration())

    dgms = []
    for dim in range(maxdim + 1):
        dgm = st.persistence_intervals_in_dimension(dim).copy()
        if len(dgm):
            dgm[~np.isfinite(dgm[:, 1]), 1] = filtration_end
        dgms.append(dgm)

    return dgms


# =========================================================
# Feature localization
# =========================================================
def localize_features(dgms, edge_list, maxdim=d.DEFAULT_MAXDIM, eps=1e-6):
    """
    Approximate localization of persistent homology features.

    For each interval, selects edges active between birth and death, builds
    a graph, and returns the largest connected component as a rough set of
    cells associated with the feature.

    Parameters
    ----------
    dgms : list[np.ndarray]
        Persistence diagrams by dimension.
    edge_list : np.ndarray, shape (n_edges, 3)
        Edge table with columns ``source, target, filtration``.
    maxdim : int, default=defaults.DEFAULT_MAXDIM
        Maximum homology dimension.
    eps : float, default=1e-6
        Numerical tolerance for filtration comparisons.

    Returns
    -------
    features : dict[int, list[dict]]
        Localized features grouped by homology dimension.
    """
    features = {dim: [] for dim in range(maxdim + 1)}

    for dim in range(maxdim + 1):
        for k, (birth, death) in enumerate(dgms[dim]):
            mask = (birth - eps <= edge_list[:, 2]) & (edge_list[:, 2] <= death + eps)

            G = nx.Graph()
            G.add_edges_from(edge_list[mask, :2].astype(int))

            cells = (
                np.array(sorted(max(nx.connected_components(G), key=len)))
                if G.number_of_nodes() else np.array([])
            )

            features[dim].append({
                "feature_id": k,
                "birth": float(birth),
                "death": float(death),
                "persistence": float(death - birth),
                "cells": cells,
            })

    return features


# =========================================================
# Main topology class
# =========================================================
class XplainTDA:
    """
    Persistent homology pipeline for single-cell data.

    The object stores configuration at initialization but does not compute
    anything until ``fit()`` is called.

    Pipeline
    --------
    AnnData matrix
        -> embedding
        -> kNN graph
        -> weighted clique complex
        -> persistent homology
        -> approximate feature localization
    """

    def __init__(
        self,
        adata=None,
        layer=d.DEFAULT_LAYER,
        embedder=d.DEFAULT_EMBEDDER,
        n_components=d.DEFAULT_N_COMPONENTS,
        n_neighbors=d.DEFAULT_N_NEIGHBORS,
        maxdim=d.DEFAULT_MAXDIM,
        neighbor_backend=d.DEFAULT_NEIGHBOR_BACKEND,
        ph_backend=d.DEFAULT_PH_BACKEND,
        dtype=d.DEFAULT_DTYPE,
        fitted_embedder=None,
        embedder_kwargs=None,
        embedding_key=None,
    ):
        """
        Initialize an unfitted XplainTDA object.

        Parameters
        ----------
        adata : AnnData, optional
            Input AnnData object. Can also be passed later to ``fit``.
        layer : str, default=defaults.DEFAULT_LAYER
            AnnData layer to use. Use ``"X"`` for ``adata.X``.
        embedder : str, default=defaults.DEFAULT_EMBEDDER
            Registered embedder name, e.g. ``"PCA"`` or ``"GRAE"``.
        n_components : int, optional
            Latent dimension. If None, set during ``fit`` to
            ``round(n_genes ** (1/3))``.
        n_neighbors : int, default=defaults.DEFAULT_N_NEIGHBORS
            Number of nearest neighbors.
        maxdim : int, default=defaults.DEFAULT_MAXDIM
            Maximum homology dimension.
        neighbor_backend : {"exact", "pynndescent"}, default=defaults.DEFAULT_NEIGHBOR_BACKEND
            kNN backend.
        dtype : np.dtype, default=defaults.DEFAULT_DTYPE
            Embedding dtype.
        fitted_embedder : Embedder, optional
            Already fitted embedder used for projection without refitting.
        embedder_kwargs : dict, optional
            Extra keyword arguments passed to the embedder constructor.
        """
        self.adata = adata
        self.layer = layer
        self.embedder_name = embedder.upper()
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.maxdim = maxdim
        self.neighbor_backend = neighbor_backend
        self.ph_backend = ph_backend.lower()
        if self.ph_backend not in VALID_PH_BACKENDS:
            raise ValueError(f"ph_backend must be one of {sorted(VALID_PH_BACKENDS)}")
        self.dtype = dtype
        self.embedder = fitted_embedder
        self.embedder_kwargs = embedder_kwargs or {}
        self.embedding_key = embedding_key

        self.X = None
        self.n_cells = None
        self.n_genes = None
        self.embedding = None
        self.neigh_ind = None
        self.neigh_dist = None
        self.edge_list = None
        self.dgms = None
        self.features = None

    # =====================================================
    # Fitting
    # =====================================================
    def fit(self, adata=None, embedding=None):
        """
        Run embedding, kNN construction, persistent homology, and localization.

        Parameters
        ----------
        adata : AnnData, optional
            AnnData object. If provided, replaces ``self.adata``.
        embedding : np.ndarray, optional
            Explicit precomputed embedding. This takes precedence over
            ``self.embedding_key``.

        Returns
        -------
        XplainTDA
            Fitted object.
        """
        if adata is not None:
            self.adata = adata

        if self.adata is None and embedding is None:
            raise ValueError("Provide `adata` or an explicit `embedding`.")

        if self.adata is not None:
            self.X = (
                self.adata.X
                if self.layer == "X"
                else self.adata.layers[self.layer]
            )

            self.n_cells, self.n_genes = self.X.shape

            if self.n_cells < 1:
                raise ValueError("adata must contain at least one cell.")

            if self.n_components is None:
                self.n_components = int(np.round(self.n_genes ** (1 / 3)))

        # Priority 1: explicit embedding passed to fit()
        if embedding is not None:
            self.embedding = self._validate_embedding(embedding)

        # Priority 2: embedding stored in adata.obsm
        elif self.embedding_key is not None:
            if self.adata is None:
                raise ValueError(
                    "`embedding_key` requires an AnnData object."
                )

            if self.embedding_key not in self.adata.obsm:
                raise KeyError(
                    f"Embedding {self.embedding_key!r} was not found in "
                    f"`adata.obsm`. Available keys: "
                    f"{list(self.adata.obsm.keys())}"
                )

            self.embedding = self._validate_embedding(
                self.adata.obsm[self.embedding_key]
            )

        # Priority 3: fit/use the configured embedder
        else:
            self._fit_embedder()
            self.embedding = self._validate_embedding(
                self.embedder.transform(self.X)
            )

        self.n_components = self.embedding.shape[1]
        self._fit_ph()

        return self


    def _validate_embedding(self, embedding):
        """
        Validate and standardize a precomputed cell embedding.

        Parameters
        ----------
        embedding : array-like
            Embedding with shape ``(n_cells, n_components)``.

        Returns
        -------
        np.ndarray
            Dense embedding converted to ``self.dtype``.

        Raises
        ------
        ValueError
            If the embedding is not two-dimensional, contains non-finite values,
            or has a number of rows inconsistent with ``adata``.
        """
        embedding = np.asarray(embedding, dtype=self.dtype)

        if embedding.ndim != 2:
            raise ValueError(
                "embedding must be a two-dimensional array with shape "
                "(n_cells, n_components)."
            )

        if self.n_cells is not None and embedding.shape[0] != self.n_cells:
            raise ValueError(
                f"embedding has {embedding.shape[0]} rows, but adata has "
                f"{self.n_cells} cells."
            )

        if embedding.shape[1] < 1:
            raise ValueError(
                "embedding must contain at least one component."
            )

        if not np.isfinite(embedding).all():
            raise ValueError(
                "embedding contains NaN or infinite values."
            )

        if self.n_cells is None:
            self.n_cells = embedding.shape[0]

        return embedding


    def _fit_embedder(self):
        """
        Fit the configured embedder unless a fitted embedder was provided.
        """
        if self.embedder is not None:
            return

        try:
            cls = Embedder.Embedder.registry[self.embedder_name]
        except KeyError:
            raise ValueError(
                f"Unknown embedder '{self.embedder_name}'. "
                f"Available: {list(Embedder.Embedder.registry)}"
            )

        self.embedder = cls(
            n_components=int(self.n_components),
            dtype=self.dtype,
            **self.embedder_kwargs
        )
        self.embedder.fit(self.X)

    def _fit_ph(self):
        """
        Compute kNN graph, clique complex, persistence diagrams, and features.
        """
        self.n_neighbors = min(int(self.n_neighbors), self.embedding.shape[0])

        self.neigh_ind, self.neigh_dist = compute_neighbors(
            self.embedding,
            n_neighbors=self.n_neighbors,
            backend=self.neighbor_backend,
        )

        self.dgms, self.features, self.edge_list = compute_ph_from_neighbors(
            self.neigh_ind,
            self.neigh_dist,
            maxdim=self.maxdim,
            ph_backend=self.ph_backend,
            localize=True,
        )

    # =====================================================
    # Embedding / recomputation
    # =====================================================
    def embed(self, X):
        """
        Project a matrix using the fitted embedder.

        Parameters
        ----------
        X : array-like
            Matrix to embed.

        Returns
        -------
        np.ndarray
            Embedded representation.
        """
        if self.embedder is None:
            raise ValueError("No fitted embedder available. Call `fit()` first or load one.")
        return self.embedder.transform(X)

    def recompute_ph_from_X(self, X):
        """
        Recompute PH after embedding a new matrix with the fitted embedder.

        Parameters
        ----------
        X : array-like
            Matrix to project and analyze.

        Returns
        -------
        dgms : list[np.ndarray]
            Persistence diagrams.
        features : dict
            Localized features.
        """
        return self.recompute_ph(self.embed(X))

    def recompute_ph(self, embedding):
        """
        Recompute PH directly from an embedding.

        Parameters
        ----------
        embedding : np.ndarray
            Cell embedding.

        Returns
        -------
        dgms : list[np.ndarray]
            Persistence diagrams.
        features : dict
            Localized features.
        """
        neigh_ind, neigh_dist = compute_neighbors(
            np.asarray(embedding, dtype=self.dtype),
            n_neighbors=self.n_neighbors,
            backend=self.neighbor_backend,
        )

        dgms, features, _ = compute_ph_from_neighbors(
            neigh_ind,
            neigh_dist,
            maxdim=self.maxdim,
            ph_backend=self.ph_backend,
            localize=True,
        )

        return dgms, features

    # =====================================================
    # Persistence summaries
    # =====================================================
    @staticmethod
    def max_persistence(dgm):
        """
        Return the largest finite persistence in one diagram.

        Parameters
        ----------
        dgm : np.ndarray, shape (n_features, 2)
            Persistence diagram.

        Returns
        -------
        float
            Maximum finite death-birth value.
        """
        if len(dgm) == 0:
            return 0.0

        finite = np.isfinite(dgm[:, 1])
        return 0.0 if finite.sum() == 0 else float(np.max(dgm[finite, 1] - dgm[finite, 0]))

    # =====================================================
    # Saving / loading
    # =====================================================
    def save(self, path, save_embedder=True, save_adata=False, compress=3):
        """
        Save fitted XplainTDA state to disk.

        Parameters
        ----------
        path : str
            Output path, usually ending in ``.joblib``.
        save_embedder : bool, default=True
            Whether to save the fitted embedder.
        save_adata : bool, default=False
            Whether to save AnnData inside the file.
        compress : int, default=3
            Joblib compression level. Use 0 for faster uncompressed saving.
        """
        state = {
            "config": {
                "layer": self.layer,
                "embedder": self.embedder_name,
                "n_components": self.n_components,
                "n_neighbors": self.n_neighbors,
                "maxdim": self.maxdim,
                "neighbor_backend": self.neighbor_backend,
                "ph_backend": self.ph_backend,
                "dtype": self.dtype,
                "embedder_kwargs": self.embedder_kwargs,
                "embedding_key": self.embedding_key,
            },
            "adata": self.adata if save_adata else None,
            "n_cells": self.n_cells,
            "n_genes": self.n_genes,
            "embedding": self.embedding,
            "neigh_ind": self.neigh_ind,
            "neigh_dist": self.neigh_dist,
            "edge_list": self.edge_list,
            "dgms": self.dgms,
            "features": self.features,
            "embedder": self.embedder if save_embedder else None}

        joblib.dump(state, path, compress=compress)

    @classmethod
    def load(cls, path):
        """
        Load a saved XplainTDA object without recomputing embedding or PH.

        Parameters
        ----------
        path : str
            Path to a saved ``.joblib`` file.

        Returns
        -------
        obj : XplainTDA
            Loaded object.
        """
        state = joblib.load(path)
        obj = cls(**state["config"], fitted_embedder=state["embedder"])

        obj.adata = state["adata"]
        obj.X = None if obj.adata is None else (obj.adata.X if obj.layer == "X" else obj.adata.layers[obj.layer])

        obj.n_cells = state["n_cells"]
        obj.n_genes = state["n_genes"]
        obj.embedding = state["embedding"]
        obj.neigh_ind = state["neigh_ind"]
        obj.neigh_dist = state["neigh_dist"]
        obj.edge_list = state["edge_list"]
        obj.dgms = state["dgms"]
        obj.features = state["features"]

        return obj

    def save_embedder(self, path, compress=3):
        """
        Save only the fitted embedder.

        For GRAEEmbedder, use its explicit save method. For other embedders,
        fall back to joblib.
        """
        if self.embedder is None:
            raise ValueError("No embedder available to save.")

        if hasattr(self.embedder, "save"):
            self.embedder.save(path, compress=compress)
        else:
            joblib.dump(self.embedder, path, compress=compress)

    def load_embedder(self, path, embedder_class=None):
        """
        Load a fitted embedder and attach it to this XplainTDA object.

        Parameters
        ----------
        path : str
            Path to saved embedder.
        embedder_class : class, optional
            Use ``GRAEEmbedder`` when loading a saved GRAE embedder.
        """
        if embedder_class is None:
            self.embedder = joblib.load(path)
        else:
            self.embedder = embedder_class.load(path)

        return self
        
        

def _prep(D):
    D = np.asarray(D, float).reshape(-1, 2)
    return D[np.isfinite(D).all(1) & (D[:, 1] > D[:, 0])]

def _linf(A, B):
    return np.maximum(abs(A[:, None, 0] - B[None, :, 0]),
                      abs(A[:, None, 1] - B[None, :, 1]))

def quantile_bottleneck(D, E, q=0.5):
    D, E = _prep(D), _prep(E)
    n, m, N = len(D), len(E), len(D) + len(E)
    if N == 0:
        return 0.0

    dD, dE = (D[:, 1] - D[:, 0]) / 2, (E[:, 1] - E[:, 0]) / 2
    DE = _linf(D, E) if n and m else np.empty((n, m))
    vals = np.unique(np.r_[0, DE.ravel(), dD, dE])
    need = int(np.ceil(q * N))
    BIG = 10**9

    def good_count(t):
        valid = np.zeros((n + m, n + m), bool)
        good = np.zeros((n + m, n + m), int)

        valid[:n, :m] = True
        good[:n, :m] = 2 * (DE <= t)

        valid[np.arange(n), m + np.arange(n)] = True
        good[np.arange(n), m + np.arange(n)] = dD <= t

        valid[n + np.arange(m), np.arange(m)] = True
        good[n + np.arange(m), np.arange(m)] = dE <= t

        valid[n:, m:] = True  # diagonal-to-diagonal fillers

        r, c = linear_sum_assignment(np.where(valid, -good, BIG))
        return good[r, c].sum()

    lo, hi = 0, len(vals) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if good_count(vals[mid]) >= need:
            hi = mid
        else:
            lo = mid + 1

    return float(vals[lo])
