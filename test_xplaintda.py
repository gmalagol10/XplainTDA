import numpy as np
from anndata import AnnData

from xplaintda.XplainTDA_PH import XplainTDA
from xplaintda.XplainTDA_Explainer import Explainer

# Small fake count matrix: 20 cells × 8 genes
rng = np.random.default_rng(42)
X = rng.poisson(2, size=(20, 8)).astype(np.float32)

adata = AnnData(X)
adata.obs_names = [f"cell_{i}" for i in range(adata.n_obs)]
adata.var_names = [f"gene_{i}" for i in range(adata.n_vars)]

print("Count matrix:", adata.shape)

# 1. Embedding + persistent homology
tda = XplainTDA(
    adata,
    embedder="PCA",
    n_components=3,
    n_neighbors=4,
    maxdim=1,
    ph_backend="ripser",
).fit()

print("Embedding:", tda.embedding.shape)
print("Persistence diagrams:", [d.shape for d in tda.dgms])

# 2. Explain just one gene
result = Explainer(tda, verbose=False).explain(
    genes=["gene_0"],
    dims=[0, 1],
    n_jobs=1,
)

print(result)

assert tda.embedding.shape == (20, 3)
assert len(tda.dgms) == 2
assert len(result) == 2

print("\nXplainTDA smoke test PASSED")
