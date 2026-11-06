# Dynamic Spectral Clustering Experiments

This is the  repository for the paper *Faster Dynamic Graph Clustering
with Hierarchical Contraction*, accepted at NeurIPS 2026. It contains the
implementation and experiment drivers for the stochastic block model (SBM),
MNIST, and EMNIST experiments. The evaluated methods include the LS24 baseline,
a semi-contracted method, and our multi-level hierarchical contraction method.
## Installation

The experiments use [`graph-tool`](https://graph-tool.skewed.de/), so Conda is
the recommended installation method:

```bash
conda env create -f environment.yml
conda activate dynamic-spectral-clustering
```

## Running the experiments

Run commands from the repository root.

### SBM experiments

```bash
python sbm_exp1.py
python sbm_exp2.py
python sbm_exp3.py
```

- `sbm_exp1.py` evaluates sensitivity to update-batch size.
- `sbm_exp2.py` evaluates new-cluster and cluster-merge workloads.
- `sbm_exp3.py` evaluates the hierarchy depth under a changing SBM workload.

### MNIST and EMNIST experiments

```bash
python mnist_compare_with_ls24.py
python emnist_compare_with_ls24.py
```

The first run downloads the requested dataset and builds its k-nearest-neighbor
graph. These files are cached locally in `mnist_data/` or `emnist_data/` and are
not tracked by Git. Building the full graphs can take substantial time and
memory.

Experiment parameters are collected in each script's `main` block. Result CSVs
and figures are written to local `results*/` directories, which are also ignored
by Git.

## Repository layout

```text
.
├── environment.yml
├── sbm_exp1.py
├── sbm_exp2.py
├── sbm_exp3.py
├── mnist_compare_with_ls24.py
├── emnist_compare_with_ls24.py
├── src/
│   ├── contractedgraph.py
│   ├── hierarchy_contracted.py
│   ├── semicontracted.py
│   ├── sparsifier.py
│   └── ...
└── tests/
    └── test_sparsifier_weights.py
```

## Tests

After activating the environment, run:

```bash
python -m unittest discover -s tests -v
```

## Acknowledgements and license

The LS24 baseline builds on the code released with *Dynamic Spectral Clustering
with Provable Approximation Guarantee* (Laenen and Sun, ICML 2024). The retained
MIT license and attribution are in [`LICENSE`](LICENSE); further provenance is
recorded in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
