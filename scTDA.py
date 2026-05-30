import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import gudhi as gd

from scipy.sparse import issparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import NearestNeighbors

try:
    from pynndescent import NNDescent
    HAS_PYNNDESCENT = True
except ImportError:
    HAS_PYNNDESCENT = False


# =========================================================
# kNN graph construction
# =========================================================
def compute_neighbors(
    X,
    n_neighbors=15,
    backend="exact"
):
    """
    Compute nearest neighbors.

    Parameters
    ----------
    X : np.ndarray
        Low-dimensional embedding.

    n_neighbors : int
        Number of neighbors.

    backend : str
        Neighbor search backend.

        "exact"
            sklearn exact neighbors

        "pynndescent"
            approximate neighbors

    Returns
    -------
    neigh_ind : np.ndarray
        Neighbor indices.

    neigh_dist : np.ndarray
        Neighbor distances.
    """

    if backend == "exact":

        nn = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric="euclidean"
        )

        nn.fit(X)

        neigh_dist, neigh_ind = nn.kneighbors(X)

        return neigh_ind, neigh_dist

    if backend == "pynndescent":

        if not HAS_PYNNDESCENT:

            raise ImportError(
                "pynndescent is not installed."
            )

        index = NNDescent(
            X,
            n_neighbors=n_neighbors,
            metric="euclidean"
        )

        neigh_ind, neigh_dist = index.neighbor_graph

        return neigh_ind, neigh_dist

    raise ValueError(
        "backend must be "
        "'exact' or 'pynndescent'"
    )


# =========================================================
# Build weighted clique complex
# =========================================================
def build_simplex_tree(
    neigh_ind,
    neigh_dist,
    maxdim=2
):
    """
    Build weighted clique complex.

    Parameters
    ----------
    neigh_ind : np.ndarray
        Neighbor indices.

    neigh_dist : np.ndarray
        Neighbor distances.

    maxdim : int
        Maximum homology dimension.

    Returns
    -------
    st : gudhi.SimplexTree

    edge_list : np.ndarray
        Array with:
        [source, target, weight]
    """

    n = neigh_ind.shape[0]

    st = gd.SimplexTree()

    # -----------------------------------------------------
    # Insert vertices
    # -----------------------------------------------------
    for i in range(n):

        st.insert([i], filtration=0.0)

    # -----------------------------------------------------
    # Construct unique weighted edges
    # -----------------------------------------------------
    edge_dict = {}

    for i in range(n):

        for j, d in zip(
            neigh_ind[i],
            neigh_dist[i]
        ):

            if i == j:
                continue

            a, b = sorted((i, j))

            key = (a, b)

            edge_dict[key] = min(
                d,
                edge_dict.get(key, np.inf)
            )

    # -----------------------------------------------------
    # Insert weighted edges
    # -----------------------------------------------------
    for (i, j), d in edge_dict.items():

        st.insert(
            [i, j],
            filtration=float(d)
        )

    # -----------------------------------------------------
    # Expand clique complex
    # -----------------------------------------------------
    st.expansion(maxdim + 1)

    edge_list = np.array([
        [i, j, d]
        for (i, j), d in edge_dict.items()
    ])

    return st, edge_list


# =========================================================
# Persistence diagrams
# =========================================================
def simplex_tree_to_diagrams(
    st,
    maxdim=2
):
    """
    Compute persistence diagrams.
    """

    st.compute_persistence()

    return [
        st.persistence_intervals_in_dimension(dim)
        for dim in range(maxdim + 1)
    ]


# =========================================================
# Feature localization
# =========================================================
def localize_features(
    dgms,
    edge_list,
    maxdim=2,
    eps=1e-6
):
    """
    Approximate localization of PH features.

    Each feature is associated with the
    largest connected component of edges
    active between birth and death.

    Returns
    -------
    features : dict
    """

    features = {}

    for dim in range(maxdim + 1):

        features[dim] = []

        for k, (birth, death) in enumerate(
            dgms[dim]
        ):

            mask = (
                (edge_list[:, 2] >= birth - eps)
                &
                (edge_list[:, 2] <= death + eps)
            )

            G = nx.Graph()

            G.add_edges_from(
                edge_list[mask][:, :2].astype(int)
            )

            cells = np.array([])

            if G.number_of_nodes() > 0:

                cells = np.array(sorted(
                    max(
                        nx.connected_components(G),
                        key=len
                    )
                ))

            features[dim].append({

                "feature_id": k,
                "birth": float(birth),
                "death": float(death),
                "persistence": float(
                    death - birth
                ),
                "cells": cells
            })

    return features


# =========================================================
# Main topology class
# =========================================================
class scTDA:
    """
    Persistent homology on single-cell data.

    Pipeline
    --------
    Count matrix
        -> PCA / TruncatedSVD
        -> kNN graph
        -> weighted clique complex
        -> persistent homology
    """

    def __init__(
        self,
        adata,
        layer="X",
        n_pcs=30,
        n_neighbors=15,
        maxdim=2,
        neighbor_backend="exact",
        dtype=np.float32
    ):

        self.adata = adata
        self.X = self.adata.X if layer == "X" else self.adata.layers[layer]
        self.n_cells, self.n_genes = self.X.shape

        self.n_pcs = n_pcs
        self.n_neighbors = n_neighbors
        self.maxdim = maxdim
        self.neighbor_backend = neighbor_backend

        # -------------------------------------------------
        # PCA embedding
        # -------------------------------------------------
        if issparse(self.X):

            self.pca = TruncatedSVD(
                n_components=n_pcs,
                random_state=0
            )

            self.embedding = self.pca.fit_transform(self.X)

        else:

            self.X = np.asarray(
                self.X,
                dtype=dtype
            )

            self.pca = PCA(
                n_components=n_pcs,
                svd_solver="randomized",
                random_state=0
            )

            self.embedding = self.pca.fit_transform(self.X)

        self.embedding = np.asarray(
            self.embedding,
            dtype=dtype
        )

        # -------------------------------------------------
        # kNN graph
        # -------------------------------------------------
        self.neigh_ind, self.neigh_dist = (
            compute_neighbors(
                self.embedding,
                n_neighbors=self.n_neighbors,
                backend=neighbor_backend
            )
        )

        # -------------------------------------------------
        # Weighted clique complex
        # -------------------------------------------------
        self.st, self.edge_list = (
            build_simplex_tree(
                self.neigh_ind,
                self.neigh_dist,
                maxdim=maxdim
            )
        )

        # -------------------------------------------------
        # Persistence diagrams
        # -------------------------------------------------
        self.dgms = simplex_tree_to_diagrams(
            self.st,
            maxdim=maxdim
        )

        # -------------------------------------------------
        # Feature localization
        # -------------------------------------------------
        self.features = localize_features(
            self.dgms,
            self.edge_list,
            maxdim=maxdim
        )
        del self.st
    # =====================================================
    # Persistence summary
    # =====================================================
    @staticmethod
    def max_persistence(dgm):
        """
        Largest finite persistence.
        """

        if len(dgm) == 0:
            return 0.0

        finite = np.isfinite(dgm[:, 1])

        if finite.sum() == 0:
            return 0.0

        return np.max(
            dgm[finite, 1]
            -
            dgm[finite, 0]
        )

    # =====================================================
    # Recompute PH from embedding
    # =====================================================
    def recompute_ph(
        self,
        embedding
    ):
        """
        Recompute persistent homology and
        localized features from an embedding.
        """

        neigh_ind, neigh_dist = (
            compute_neighbors(
                embedding,
                n_neighbors=self.n_neighbors,
                backend=self.neighbor_backend
            )
        )

        st, edge_list = build_simplex_tree(
            neigh_ind,
            neigh_dist,
            maxdim=self.maxdim
        )

        dgms = simplex_tree_to_diagrams(
            st,
            maxdim=self.maxdim
        )

        features = localize_features(
            dgms,
            edge_list,
            maxdim=self.maxdim
        )

        return dgms, features
    # =====================================================
    # Plot persistence
    # =====================================================

