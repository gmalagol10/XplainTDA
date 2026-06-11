from abc import ABC, abstractmethod

import numpy as np

from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD

from Models import GRAE
from base_dataset import BaseDataset
import defaults as d

import joblib
import torch

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
        dtype=d.DEFAULT_DTYPE,
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
        dtype=d.DEFAULT_DTYPE
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
        epochs=d.DEFAULT_GRAE_EPOCHS,
        patience=d.DEFAULT_GRAE_PATIENCE,
        dtype=d.DEFAULT_DTYPE,
        **grae_kwargs,
    ):
        super().__init__(n_components, dtype)

        self.epochs = epochs
        self.patience = patience
        self.grae_kwargs = grae_kwargs
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
            data_val=val_ds,
            **self.grae_kwargs)

        self.model.fit(train_ds)

    def transform(self, X):

        return np.asarray(
            self.model.transform(self._dataset(X)),
            dtype=self.dtype)
            
    def _config(self):
        """Return constructor arguments needed to recreate this embedder."""
        return {
            "n_components": self.n_components,
            "epochs": self.epochs,
            "patience": self.patience,
            "dtype": self.dtype,
            "grae_kwargs": self.grae_kwargs,
        }

    def save(self, path, compress=3):
        """
        Save GRAEEmbedder configuration and fitted torch model state.

        Parameters
        ----------
        path : str
            Output path, usually ending in ``.joblib``.
        compress : int, default=3
            Joblib compression level.
        """
        state = {
            "config": self._config(),
            "model_state": None,
            "data_shape": None,
        }

        if self.model is not None and self.model.torch_module is not None:
            state["model_state"] = self.model.torch_module.state_dict()
            state["data_shape"] = self.model.data_shape

        joblib.dump(state, path, compress=compress)

    @classmethod
    def load(cls, path):
        """
        Load a saved GRAEEmbedder.

        Returns
        -------
        GRAEEmbedder
            Restored embedder with fitted torch module if model state was saved.
        """
        state = joblib.load(path)
        config = state["config"]

        obj = cls(
            n_components=config["n_components"],
            epochs=config["epochs"],
            patience=config["patience"],
            dtype=config["dtype"],
            **config.get("grae_kwargs", {}),
        )

        if state["model_state"] is not None:
            obj.model = GRAE(
                epochs=obj.epochs,
                patience=obj.patience,
                latent_dim=obj.n_components,
                data_val=None,
                **obj.grae_kwargs,
            )
            obj.model.init_torch_module(state["data_shape"])
            obj.model.torch_module.load_state_dict(state["model_state"])
            obj.model.fitted = True

        return obj
