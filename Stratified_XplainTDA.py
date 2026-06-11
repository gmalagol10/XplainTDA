import numpy as np
import pandas as pd

import Embedder
from XplainTDA import XplainTDA
from Local_XplainTDA import Local_XplainTDA
from Global_XplainTDA import Global_XplainTDA
import defaults as d

import time, joblib
from pathlib import Path

class StratifiedXplainTDA:
    """
    Run cell-group-specific PH and PH explanations in a global embedding space.

    The embedding is fitted once on all cells. For each group in
    ``adata.obs[groupby]``, PH is computed on the subset of globally embedded
    cells, and perturbation-based explanations are run only within that subset.
    Cell groups are processed sequentially; parallelism is delegated to the
    underlying local/global explainer classes.
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
        dtype=d.DEFAULT_DTYPE,
        embedder_kwargs=d.DEFAULT_EMBEDDER_KWARGS,
        verbose=d.DEFAULT_VERBOSE,
        outdir="StratifiedXplainTDA",
        save_intermediate=True,
        compress=3,
    ):
        """
        Parameters
        ----------
        adata : AnnData
            Input annotated data matrix.
        groupby : str
            Column in ``adata.obs`` defining the strata, e.g. cell type.
        groups : array-like, optional
            Subset of groups to analyze. If None, all non-null groups are used.
        min_cells : int, default=3
            Groups with fewer cells are skipped.
        layer, embedder, n_components, n_neighbors, maxdim, neighbor_backend, dtype
            Arguments forwarded to the PH pipeline.
        embedder_kwargs : dict, optional
            Extra keyword arguments for the embedder constructor, e.g. GRAE args.
        verbose : bool, default=True
            Whether to print progress messages.
        """
        if groupby not in adata.obs:
            raise ValueError(f"{groupby!r} is not a column in adata.obs")

        self.adata = adata
        self.groupby = groupby
        self.groups = list(pd.Series(adata.obs[groupby]).dropna().unique()) if groups is None else list(groups)
        self.min_cells = int(min_cells)
        self.layer = layer
        self.embedder_name = embedder
        self.n_components = n_components
        self.n_neighbors = int(n_neighbors)
        self.maxdim = int(maxdim)
        self.neighbor_backend = neighbor_backend
        self.dtype = dtype
        self.embedder_kwargs = embedder_kwargs or {}
        self.verbose = verbose
        self.ph_backend = ph_backend

        self.X = adata.X if layer == "X" else adata.layers[layer]
        self.n_cells, self.n_genes = self.X.shape
        self.n_components_ = int(np.round(self.n_genes ** (1 / 3))) if n_components is None else int(n_components)

        self.global_embedder = None
        self.global_embedding = None
        self.tdas = {}
        self.local_explainers = {}
        self.global_explainers = {}
        self.local_results = {}
        self.global_results = {}
        self.skipped_groups = {}
        
        self.outdir = Path(outdir) if outdir is not None else None
        self.save_intermediate = bool(save_intermediate)
        self.compress = compress

        if self.save_intermediate and self.outdir is None:
            raise ValueError("outdir must be provided when save_intermediate=True")

        if self.outdir is not None:
            self.outdir.mkdir(parents=True, exist_ok=True)
        

    # =====================================================
    # Embedding and group-specific PH
    # =====================================================
    def fit_embedding(self):
        """Fit the embedder once on all cells and store the global embedding."""
        if self.global_embedding is not None:
            return self

        try:
            cls = Embedder.Embedder.registry[self.embedder_name.upper()]
        except KeyError:
            raise ValueError(
                f"Unknown embedder '{self.embedder_name}'. "
                f"Available: {list(Embedder.Embedder.registry)}"
            )

        self.global_embedder = cls(
            n_components=self.n_components_,
            dtype=self.dtype,
            **self.embedder_kwargs,
        )
        self.global_embedding = self.global_embedder.fit_transform(self.X)

        # Checkpoint before starting stratification.
        self._save_global_embedding()

        return self

    def _mask(self, group):
        """Boolean mask selecting one group from adata.obs."""
        return np.asarray(self.adata.obs[self.groupby] == group)

    def _dims(self, tda, dims=None):
        """Resolve homology dimensions from the fitted TDA object."""
        if dims is None:
            return sorted(tda.features.keys())
        return [int(d) for d in np.atleast_1d(dims)]

    def fit_group(self, group):
        """Fit PH for one group using the precomputed global embedding rows."""
        if group in self.tdas:
            return self.tdas[group]

        self.fit_embedding()
        mask = self._mask(group)
        n = int(mask.sum())

        if n < self.min_cells:
            self.skipped_groups[group] = f"n_cells={n} < min_cells={self.min_cells}"
            if self.verbose:
                print(f"Skipping {group}: {self.skipped_groups[group]}", flush=True)
            return None

        # Use all available neighbors when the group is smaller than requested.
        n_neighbors = min(self.n_neighbors, n)
        adata_g = self.adata[mask].copy()
        embedding_g = self.global_embedding[mask]

        if self.verbose:
            print(f"Fitting PH for {group} | cells={n} | n_neighbors={n_neighbors}", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), flush=True)

        tda = XplainTDA(
            adata=adata_g,
            layer=self.layer,
            n_components=self.n_components_,
            n_neighbors=n_neighbors,
            maxdim=self.maxdim,
            ph_backend=self.ph_backend
            neighbor_backend=self.neighbor_backend,
            dtype=self.dtype,
            fitted_embedder=self.global_embedder)

        tda.fit(embedding=embedding_g)
        self.tdas[group] = tda
        self._save_group_tda(group, tda)
        return tda

    def fit(self):
        """Fit the global embedding and group-specific PH objects."""
        self.fit_embedding()
        for group in self.groups:
            self.fit_group(group)
        return self

    # =====================================================
    # Explanations
    # =====================================================
    @staticmethod
    def _add_group_columns(df, group, tda, method):
        """Attach group metadata to a result table."""
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
        """Concatenate non-empty tables without failing on all-empty results."""
        tables = [t for t in tables if t is not None and len(t)]
        return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()

    def local_expl(
        self,
        genes=d.DEFAULT_GENES,
        dims=d.DEFAULT_DIMS,
        n_jobs=d.DEFAULT_N_JOBS,
        backend=d.DEFAULT_BACKEND,
        cache=d.DEFAULT_CACHE,
        ph_backend=d.DEFAULT_PH_BACKEND,
        target_dim=d.DEFAULT_TARGET_DIM,
        perturbation=d.DEFAULT_PERTURBATION,
        recompute_mode=d.DEFAULT_RECOMPUTE_MODE,
        local_radius=d.DEFAULT_LOCAL_RADIUS,
        top_frac=d.DEFAULT_TOP_FRAC,
        max_features=d.DEFAULT_MAX_FEATURES,
        matching_mode=d.DEFAULT_MATCHING_MODE,
        jaccard_threshold=d.DEFAULT_JACCARD_THRESHOLD,
        random_state=d.DEFAULT_RANDOM_STATE
    ):
        tables = []

        for group in self.groups:
            print("=" * 60, flush=True)
            print(
                f"Start local XplainTDA for {group}",
                time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()),
                flush=True,
            )
            print("=" * 60, flush=True)

            tda = self.fit_group(group)
            if tda is None:
                continue

            expl = Local_XplainTDA(
                tda,
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

            # Checkpoint this label immediately after explanation.
            self._save_explainer(group, expl, method="local")

        final = self._concat(tables)

        # Save only after all labels are complete.
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
        random_state=d.DEFAULT_RANDOM_STATE
    ):
        tables = []

        for group in self.groups:
            print("=" * 60, flush=True)
            print(f"Start global XplainTDA for {group}", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), flush=True)
            print("=" * 60, flush=True)

            tda = self.fit_group(group)
            if tda is None:
                continue

            expl = Global_XplainTDA(
                tda,
                target_dim=target_dim,
                perturbation=perturbation,
                distance=distance,
                bottleneck_epsilon=bottleneck_epsilon,
                wasserstein_order=wasserstein_order,
                wasserstein_internal_p=wasserstein_internal_p,
                verbose=self.verbose,
                random_state=random_state)

            df = expl.importance_all_homologies(
                genes=genes,
                dims=dims,
                n_jobs=n_jobs,
                backend=backend,
                cache=cache)

            df = self._add_group_columns(df, group, tda, "global")
            self.global_explainers[group] = expl
            self.global_results[group] = df
            tables.append(df)

            # Checkpoint this label immediately after explanation.
            self._save_explainer(group, expl, method="global")

        final = self._concat(tables)

        # Save only after all labels are complete.
        self._save_final_dataframe(final, method="global")

        return final
        
        
    def save_results(
        self,
        outdir,
        which="both",
        per_group=False,
        index=False,
    ):
        """
        Save local/global explanation result tables to disk.

        Parameters
        ----------
        outdir : str or pathlib.Path
            Output directory.
        which : {"local", "global", "both"}, default="both"
            Which result tables to save..
        per_group : bool, default=False
            If True, also save one file per group and method.
        index : bool, default=False
            Whether to write DataFrame indices.

        Returns
        -------
        paths : dict
            Mapping result names to written file paths.
        """
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        if which not in {"local", "global", "both"}:
            raise ValueError("which must be one of {'local', 'global', 'both'}")

        def _write(df, name):
            path = outdir / f"{name}.csv"
            df.to_csv(path, index=index)
            return path

        paths = {}
        todo = ["local", "global"] if which == "both" else [which]

        for method in todo:
            results = self.local_results if method == "local" else self.global_results
            df = self._concat(results.values())

            if len(df):
                paths[f"{method}_all"] = _write(df, f"{method}_results")

            if per_group:
                for group, group_df in results.items():
                    if group_df is None or len(group_df) == 0:
                        continue
                    safe_group = str(group).replace("/", "_").replace(" ", "_")
                    paths[f"{method}_{group}"] = _write(
                        group_df,
                        f"{method}_{safe_group}",
                    )

        if self.skipped_groups:
            skipped = pd.DataFrame(
                [{"group": g, "reason": r} for g, r in self.skipped_groups.items()]
            )
            paths["skipped_groups"] = _write(skipped, "skipped_groups")

        return paths
        
    def _safe_name(self, x):
        """Return a filesystem-safe group name."""
        return str(x).replace("/", "_").replace("\\", "_").replace(" ", "_")


    def _ckpt_path(self, *parts):
        """Build a checkpoint path and create parent directories."""
        path = self.outdir.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


    def _save_global_embedding(self):
        """Save global embedding, fitted embedder, and embedding metadata."""
        if not self.save_intermediate:
            return

        joblib.dump(
            {
                "embedding": self.global_embedding,
                "embedder": self.global_embedder,
                "embedder_name": self.embedder_name,
                "n_components": self.n_components_,
                "dtype": self.dtype,
                "layer": self.layer,
                "embedder_kwargs": self.embedder_kwargs,
            },
            self._ckpt_path("global_embedding.joblib"),
            compress=self.compress,
        )


    def _save_group_tda(self, group, tda):
        """
        Save one fitted group-specific TDA without the count matrix.

        The global embedder is saved separately, so avoid duplicating it inside
        every group TDA file.
        """
        if not self.save_intermediate:
            return

        tda.save(
            self._ckpt_path("tdas", f"{self._safe_name(group)}.joblib"),
            save_embedder=False,
            save_adata=False,
            compress=self.compress,
        )


    def _save_explainer(self, group, explainer, method):
        """
        Save one fitted explainer after a group finishes.

        The group TDA is saved separately, so avoid duplicating it inside the
        explainer file.
        """
        if not self.save_intermediate:
            return

        explainer.save(
            self._ckpt_path("explainers", method, f"{self._safe_name(group)}.joblib"),
            save_tda=False,
            compress=self.compress,
        )


    def _save_final_dataframe(self, df, method, index=False):
        """Save the final concatenated result table."""
        if not self.save_intermediate:
            return

        path = self._ckpt_path("results", f"{method}_results.csv")
        df.to_csv(path, index=index)
