from abc import ABC, abstractmethod

import numpy as np

from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD

from Models import GRAE
from base_dataset import BaseDataset

class Embedder(ABC):

    registry = {}

    @classmethod
    def register(cls, name):

        def decorator(embedder_cls):

            cls.registry[name.upper()] = embedder_cls

            return embedder_cls

        return decorator

    def __init__(
        self,
        n_components,
        dtype=np.float32
    ):

        self.n_components = n_components
        self.dtype = dtype

    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def transform(self, X):
        pass

    def fit_transform(self, X):

        self.fit(X)

        return self.transform(X)


@Embedder.register("PCA")
class PCAEmbedder(Embedder):

    def __init__(
        self,
        n_components,
        dtype=np.float32
    ):

        super().__init__(
            n_components=n_components,
            dtype=dtype
        )

        self.model = None

    def fit(self, X):

        if sparse.issparse(X):

            self.model = TruncatedSVD(
                n_components=self.n_components,
                random_state=0
            )

        else:

            self.model = PCA(
                n_components=self.n_components,
                svd_solver="randomized",
                random_state=0
            )

        self.model.fit(X)

    def transform(self, X):

        return np.asarray(
            self.model.transform(X),
            dtype=self.dtype
        )


@Embedder.register("GRAE")
class GRAEEmbedder(Embedder):

    def __init__(
        self,
        n_components,
        epochs=300,
        patience=30,
        dtype=np.float32
    ):

        super().__init__(n_components, dtype)

        self.epochs = epochs
        self.patience = patience
        self.model = None

    def _dataset(self, X):

        X = sparse.csr_matrix(X, dtype=np.float32).toarray()
        y = np.ones(X.shape[0])

        return BaseDataset(
            X,
            y=y,
            split="none",
            split_ratio=1,
            random_state=42,
            labels=y
        )

    def fit(self, X):

        ds = self._dataset(X)
        train_ds, val_ds, _ = ds.validation_split(ratio=0.15)

        self.model = GRAE(
            epochs=self.epochs,
            patience=self.patience,
            latent_dim=self.n_components,
            data_val=val_ds)

        self.model.fit(train_ds)

    def transform(self, X):

        return np.asarray(
            self.model.transform(self._dataset(X)),
            dtype=self.dtype)
