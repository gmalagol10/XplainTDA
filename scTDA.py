import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import gudhi as gd

import Embedder
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import normalized_mutual_info_score

try:
    from pynndescent import NNDescent
    HAS_PYNNDESCENT = True
except ImportError:
    HAS_PYNNDESCENT = False


# =========================================================
# kNN graph construction
# =========================================================
def compute_neighbors(X, n_neighbors=15, backend="exact", metric="euclidean"):
    """
    Compute k-nearest neighbors for an embedding.

    Parameters
    ----------
    X : np.ndarray
        Input embedding of shape (n_cells, n_features).
    n_neighbors : int, default=15
        Number of nearest neighbors to compute.
    backend : {"exact", "pynndescent"}, default="exact"
        Neighbor search method:
        - "exact": exact search via sklearn.
        - "pynndescent": approximate search via pynndescent.
    metric: str
        "euclidean" or "cknn"
        if "cknn" the distances of the neighborhood graph are scaled using the last nearest neighbor.

    Returns
    -------
    neigh_ind : np.ndarray
        Indices of nearest neighbors.
    neigh_dist : np.ndarray
        Distances to nearest neighbors.
    """
    # Exact kNN search using sklearn
    if backend == "exact":
        neigh_dist, neigh_ind = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(X).kneighbors(X)

        if metric == "cknn": #Returns scaled distances
            rho = neigh_dist[:, -1]
            scale = np.sqrt( rho[:, None] * rho[neigh_ind] )

            neigh_dist = neigh_dist / scale

    # Approximate kNN search using pynndescent
    elif backend == "pynndescent":
        if not HAS_PYNNDESCENT:
            raise ImportError("pynndescent is not installed.")
        neigh_ind, neigh_dist = NNDescent(X, n_neighbors=n_neighbors, metric="euclidean").neighbor_graph

    else:
        raise ValueError("backend must be 'exact' or 'pynndescent'")

    return neigh_ind, neigh_dist


# =========================================================
# Build weighted clique complex
# =========================================================
def build_simplex_tree(neigh_ind, neigh_dist, maxdim=2):
    """
    Build a weighted clique complex from a kNN graph.

    Parameters
    ----------
    neigh_ind : np.ndarray
        Neighbor indices of shape (n_samples, k).
    neigh_dist : np.ndarray
        Corresponding neighbor distances.
    maxdim : int, default=2
        Maximum homology dimension to compute.

    Returns
    -------
    st : gudhi.SimplexTree
        Expanded weighted clique complex.
    edge_list : np.ndarray
        Array of shape (n_edges, 3) containing
        [source, target, weight].
    """
    n = neigh_ind.shape[0]
    st = gd.SimplexTree()

    # Insert all vertices
    for i in range(n):
        st.insert([i], filtration=0.0)

    # Build unique undirected edges using the minimum observed distance
    edge_dict = {}
    for i, (nbrs, dists) in enumerate(zip(neigh_ind, neigh_dist)):
        for j, d in zip(nbrs, dists):
            if i != j:
                edge = tuple(sorted((i, j)))
                edge_dict[edge] = min(d, edge_dict.get(edge, np.inf))

    # Insert weighted edges and expand to a clique complex
    for edge, d in edge_dict.items():
        st.insert(edge, filtration=float(d))

    st.expansion(maxdim + 1)

    edge_list = np.array([[i, j, d] for (i, j), d in edge_dict.items()])
    return st, edge_list


# =========================================================
# Persistence diagrams
# =========================================================
def simplex_tree_to_diagrams(st, maxdim=2):
    """
    Compute persistence diagrams from a simplex tree.

    Parameters
    ----------
    st : gudhi.SimplexTree
        Input simplicial complex.
    maxdim : int, default=2
        Maximum homology dimension to return.

    Returns
    -------
    list[np.ndarray]
        Persistence intervals for dimensions 0, ..., maxdim.
    """
    # Compute persistent homology and extract diagrams by dimension
    st.compute_persistence()
    return [st.persistence_intervals_in_dimension(dim) for dim in range(maxdim + 1)]

# =========================================================
# Feature localization
# =========================================================
def localize_features(dgms, edge_list, maxdim=2, eps=1e-6):
    """
    Approximate localization of persistent homology features.

    For each persistence interval, identify the largest connected
    component formed by edges whose filtration values lie between
    the feature's birth and death times.

    Parameters
    ----------
    dgms : list[np.ndarray]
        Persistence diagrams by homology dimension.
    edge_list : np.ndarray
        Edge array with columns [source, target, weight].
    maxdim : int, default=2
        Maximum homology dimension to process.
    eps : float, default=1e-6
        Numerical tolerance when comparing filtration values.

    Returns
    -------
    dict
        Mapping homology dimension to a list of localized features.
    """
    features = {}

    for dim in range(maxdim + 1):
        features[dim] = []

        for k, (birth, death) in enumerate(dgms[dim]):
            # Select edges active during the feature lifetime
            mask = (birth - eps <= edge_list[:, 2]) & (edge_list[:, 2] <= death + eps)

            G = nx.Graph()
            G.add_edges_from(edge_list[mask, :2].astype(int))

            # Largest connected component (if any)
            cells = np.array(sorted(max(nx.connected_components(G), key=len))) if G.number_of_nodes() else np.array([])
    
            features[dim].append({
                "feature_id": k,
                "birth": float(birth),
                "death": float(death),
                "persistence": float(death - birth),
                "cells": cells})

    return features


def get_filtration_values(neigh_dist, nbins=100):
    vals = np.asarray(neigh_dist).ravel()
    vals = vals[vals > 0]

    vals_sorted = np.sort(vals)
    bins = np.array_split(vals_sorted, nbins)

    reps = []
    for b in bins:
        if len(b) == 0:
            continue
        center = np.median(b)
        reps.append(b[np.argmin(np.abs(b - center))])

    return np.array(reps)


def cc_at_threshold(edge_list, n_cells, value):
    """
    Compute connected components at a given filtration threshold.

    Parameters
    ----------
    edge_list : np.ndarray
        Array of shape (n_edges, 3): [src, dst, weight]
    n_cells : int
        Number of cells (nodes)
    value : float
        Filtration cutoff (keep edges with weight <= value)

    Returns
    -------
    cc_type : np.ndarray
        Cluster label per cell
    nr_cc : int
        Number of connected components
    """

    # 1. filter edges by threshold
    mask = edge_list[:, 2] <= value
    filtered_edges = edge_list[mask, :2].astype(int)

    # 2. build graph
    G = nx.Graph()
    G.add_nodes_from(range(n_cells))
    G.add_edges_from(filtered_edges)

    # 3. connected components
    components = list(nx.connected_components(G))

    # 4. assign labels
    cc_type = np.empty(n_cells, dtype=int)

    for cc_id, comp in enumerate(components):
        for node in comp:
            cc_type[node] = cc_id

    nr_cc = len(components)

    return cc_type, nr_cc
# =========================================================
# Main topology class
# =========================================================
class scTDA:
    """
    Persistent homology on single-cell data.

    Pipeline
    --------
    Count matrix
        -> Embedding
        -> kNN graph
        -> weighted clique complex
        -> persistent homology
    """

    def __init__(
        self,
        adata,
        layer="X",
        embedder="PCA",
        n_components=None,
        n_neighbors=30,
        maxdim=2,
        neighbor_backend="exact",
        neighbor_metric="euclidean",
        dtype=np.float32,
    ):
        """
        Compute persistent homology on single-cell data.

        Parameters
        ----------
        adata : AnnData
            Input AnnData object.
        layer : str, default="X"
            Data matrix to use.
        embedder : str, default="PCA"
            Registered embedding method (e.g. "PCA", "GRAE").
        n_components : int, optional
            Latent dimensionality. If None, uses round(n_genes^(1/3)).
        n_neighbors : int, default=15
            Number of neighbors in the kNN graph.
        maxdim : int, default=2
            Maximum homology dimension.
        neighbor_backend : {"exact", "pynndescent"}, default="exact"
            kNN search backend.
        neighbor_metric : {"euclidean", "cknn"}, default="eclidean"
            "cknn" scales the neighbor graph according to local density
        dtype : np.dtype, default=np.float32
            Embedding dtype.
        """

        self.adata = adata
        self.X = adata.X if layer == "X" else adata.layers[layer]

        self.n_cells, self.n_genes = self.X.shape
        self.n_components = int(np.round(self.n_genes**(1/3))) if n_components is None else int(n_components)

        self.n_neighbors = n_neighbors
        self.maxdim = maxdim
        self.neighbor_backend = neighbor_backend
        self.neighbor_metric = neighbor_metric

        # Instantiate requested embedding model
        try:
            self.embedder = Embedder.Embedder.registry[embedder.upper()](
                n_components=self.n_components,
                dtype=dtype)
                
        except KeyError:
            raise ValueError(
                f"Unknown embedder '{embedder}'. "
                f"Available: {list(Embedder.Embedder.registry)}")

        # Learn embedding once and project cells
        self.embedding = self.embedder.fit_transform(self.X)

        # Construct weighted kNN graph
        self.neigh_ind, self.neigh_dist = compute_neighbors(
            self.embedding,
            n_neighbors=self.n_neighbors,
            backend=self.neighbor_backend,
            metric=self.neighbor_metric)

        # Build filtered clique complex
        self.st, self.edge_list = build_simplex_tree(
            self.neigh_ind,
            self.neigh_dist,
            maxdim=self.maxdim)

        # Compute persistent homology and feature localization
        self.dgms = simplex_tree_to_diagrams(self.st, maxdim=self.maxdim)
        self.features = localize_features(
            self.dgms,
            self.edge_list,
            maxdim=self.maxdim)

        del self.st
        
    # =====================================================
    # Embedding
    # =====================================================        
    def embed(self, X):
        return self.embedder.transform(X)
        
    def recompute_ph_from_X(self, X):
        embedding = self.embed(X)
        return self.recompute_ph(embedding)

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
                backend=self.neighbor_backend,
                metric=self.neighbor_metric
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
    
    def get_annotation(
            self,
    ):
        filtration_values = get_filtration_values(self.neigh_dist, nbins=100)
        celltype = np.array(self.adata.obs["CellType"])

        best_value = None
        best_nmi = -np.inf
        best_nr_cc = None
        best_cc_type = None

        for value in filtration_values:

            cc_type, nr_cc = cc_at_threshold(
                edge_list=self.edge_list,
                n_cells=self.n_cells,
                value=value
            )

            nmi = normalized_mutual_info_score(celltype, cc_type)

            if nmi > best_nmi:
                best_nmi = nmi
                best_value = value
                best_nr_cc = nr_cc
                best_cc_type = cc_type

        # ---- STORE RESULT IN ANNData ----
        self.adata.obs["cc_id"] = best_cc_type
        print(f"Found best cc annotation at filtration value {best_value} with nmi score {best_nmi} with {best_nr_cc} connected components")



    # =====================================================
    # Plot persistence
    # =====================================================

