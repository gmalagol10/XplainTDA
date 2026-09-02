# XplainTDA

Explaining topological structure of single-cell multi-omics data through perturbation

isabel Giray, Gabriele Malagoli, Katharina T. Schmidt, Maria Colomé-Tatché



![alt text](https://github.com/gmalagol10/XplainTDA/blob/main/xplaintda/XplainTDA_Figure1.png?raw=true)


Data and notebook to reproduce all the results are available at https://github.com/gmalagol10/xplaintda/tree/main/reproducibility
## Installation

```bash

conda create -n my_env python=3.10.14

conda activate my_env

pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu

pip install git+https://github.com/gmalagol10/xplaintda
```

## Usage

```python
import scanpy as sc
from xplaintda.XplainTDA_Stratified import StratifiedExplainer

# Load a filtered single-cell count matrix
adata = sc.read_h5ad("adata_path")

# Fit a shared GRAE embedding and compute persistent homology
# separately for each cell type
xplainer = StratifiedExplainer(adata, groupby="you_label", maxdim=2, outdir="XplainTDA_results")

# Perturb each gene and quantify its topological importance
results = xplainer.explain(dims=[0, 1, 2], n_jobs=1)

# Show the most important genes for one cell type and homology dimension
scores = results[(results["group"] == "one_label") & (results["homology_dim"] == 1)].sort_values("score", ascending=False)

# Plot ranked gene importance
plt.scatter(range(len(scores)), scores["score"])
plt.xlabel("Gene rank")
plt.ylabel("Topological importance")
plt.show()

```

## Version History

* 0.1
    * Initial Release


## License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE.md file for details


