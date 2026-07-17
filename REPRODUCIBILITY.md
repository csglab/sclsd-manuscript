# Reproducibility record

This document records the data, software, hardware, random seeds, and model
configurations used by the manuscript workflows. It complements the notebooks
without changing their analysis logic or saved outputs.

## Repository revisions

The notebook configuration audit was performed against the
`sclsd-manuscript` notebook snapshot at commit
`55b47085b73227dc7ab2f4768fb530efd02cfc65`. The notebooks were inspected
read-only; no notebook was edited as part of this reproducibility update.

Two `sclsd` implementations are relevant:

| Implementation | Version or revision | Purpose |
|---|---|---|
| Original manuscript implementation | PyPI `sclsd==0.3.0`; Git commit `a45e46d7e38a37a05c73b9622791167fa43b389f` | Reproduces the original dense transition and PyTorch sampling path |
| Sparse revision | Branch `sclsd-sparse`; Git commit `8fc9285426b98e4e3d078ec50e849175b22d3fd4` | Keeps transition matrices sparse and performs random-walk sampling on CPU |

The Python source files in the published `sclsd` 0.3.0 wheel were verified to
match commit `a45e46d7e38a37a05c73b9622791167fa43b389f`. The wheel SHA-256 is
`8ddfc1183d47ce4fda01d89553591cdba07d114ed46678ff7208feb2f953608e`.

The current sparse revision is not the PyPI `0.3.0` artifact even though its
internal package version has not yet been incremented. It must therefore be
identified by its Git commit.

The sparse revision computes phylogeny masks and transition probabilities over
the nonzero edges of the cell-neighbor graph, stores transition matrices in CSR
form, and avoids copying a dense cell-by-cell transition matrix to GPU for the
main random-walk workflow. The transition probabilities agree with the legacy
dense calculation within floating-point tolerance. However, the sparse CPU
sampler and the original PyTorch GPU sampler use different random-number
streams and sampling implementations. The same seed therefore does not imply
identical walk indices, and downstream stochastic results may differ.

To use the original sampling implementation, install the published release:

```bash
python -m pip install "sclsd==0.3.0"
```

Alternatively, check out the corresponding source revision:

```bash
git checkout a45e46d7e38a37a05c73b9622791167fa43b389f
python -m pip install -e .
```

Using the same software version and seed restores the original sampling path,
but bit-for-bit training results can still depend on CUDA, hardware, and
nondeterministic GPU operations.

## Data release

The preprocessed inputs are distributed in
[Zenodo record 18331587](https://zenodo.org/records/18331587). The table below
records the files inspected locally with `scanpy.read`, their dimensions, and
SHA-256 checksums. These checksums allow an extracted dataset to be verified
independently of its directory name.

| File | Cells | Genes | SHA-256 |
|---|---:|---:|---|
| `Zenodo/BoneMarrow/preprocessed_adata.h5ad` | 5,292 | 2,000 | `97e9e740362ae7d38a145a0a5970734f20b61effce27d9ff4782132196135773` |
| `Zenodo/DentateGyrus/preprocessed_adata.h5ad` | 2,460 | 1,500 | `6a8d2cce85e6e746d877af091b0175ec6b0a36440e9e1cf3ef55541efb6876c5` |
| `Zenodo/Erythroid/preprocessed_adata.h5ad` | 9,815 | 5,000 | `493fd8123c73ab52bedce694ac694830fd7e7512dbc09145dfce7147546499a8` |
| `Zenodo/KP_tracer/preprocessed_adata.h5ad` | 33,179 | 1,999 | `86272debc454c59a54fe323c4a0534acea1f99ea5f170fe660b3e019dc1c7c91` |
| `Zenodo/Mouse_Cortex/adata_heldout.h5ad` | 10,398 | 32,285 | `baf94b089e3d1e0dcec204159331bd3414476788a9b19cd5ed8d08872be7e1cf` |
| `Zenodo/Mouse_Cortex/preprocessed_adata_train.h5ad` | 12,814 | 1,062 | `9b53b6c5fca8d7a01e7bacaa5200e25b5be149526797523bba701f6eb9a8c090` |
| `Zenodo/Pancreas/preprocessed_adata.h5ad` | 16,822 | 5,000 | `567f62221a596842292d01c4e40b5484dc6bbb205afdc165a4ea9381cbff9b3b` |
| `Zenodo/Tutorial/raw_adata.h5ad` | 5,780 | 14,319 | `12222efe4a9dd5916fa279860a2a4fdec383bd2e0db0249a1cd18c549c4a4c2c` |
| `Zenodo/Zebrafish/preprocessed_adata_train.h5ad` | 2,068 | 2,000 | `55384fe388cb66c3ac4de01146d526e1dffb1642e9f66a096e4acdcc6780b15d` |
| `Zenodo/Zebrafish/preprocessed_adata_valid.h5ad` | 2,434 | 2,000 | `8a56773536846863e7fa5e1f0c9a7a7afac3123fca953950991c41bcfb2caa46` |

The Vignette notebook currently loads the bone-marrow data through
`scvelo.datasets.bonemarrow()` rather than reading the local Zenodo file. The
Zenodo archive separately supplies `Tutorial/raw_adata.h5ad` with the same
recorded dimensions; the two objects were not asserted to be byte-identical.

## Software environment

The sparse revision and reproducibility checks used the following tested
environment:

| Software | Version |
|---|---:|
| Python | 3.10.12 |
| sclsd | 0.3.0 plus sparse commit `8fc9285426b98e4e3d078ec50e849175b22d3fd4` |
| PyTorch | 2.4.1 |
| Pyro | 1.9.1 |
| torchdiffeq | 0.2.5 |
| Scanpy | 1.11.5 |
| AnnData | 0.11.4 |
| CellRank | 2.0.7 |
| NumPy | 1.26.4 |
| SciPy | 1.15.3 |
| pandas | 2.3.3 |
| scikit-learn | 1.7.2 |

## Hardware environments

| Use | GPU | Driver | CUDA | CPU | Logical CPUs |
|---|---|---:|---:|---|---:|
| Original manuscript runs | NVIDIA GeForce RTX 3090 | 535.247.01 | 12.2 | Intel Xeon Gold 6346 @ 3.10 GHz | 64 |
| Additional compatibility testing | NVIDIA H100 80GB HBM3 | 560.35.05 | 12.6 | Intel Xeon Platinum 8468 | 192 |

The original manuscript models and notebooks were run on the RTX 3090 system.
The workflows were additionally tested on the H100 system. These statements
describe execution and compatibility; they are not substitutes for the
controlled runtime and peak-memory measurements described below.

## Training configurations and notebook order

All stochastic workflows use an effective random seed of `42`. Some notebooks
pass it explicitly to every call, while others combine `set_all_seeds(42)` with
the package defaults of `WalkConfig.random_state=42` and training seed `42`.
The table reports the effective values encoded in the audited notebook
snapshot. The training batch size is the `WalkConfig` batch size.

| Workflow | Training input | Training shape | Notebook order | Seed | Walks | Walk length | Batch size | Epochs |
|---|---|---:|---|---:|---:|---:|---:|---:|
| Bone Marrow | `BoneMarrow/preprocessed_adata.h5ad` | 5,292 × 2,000 | `BoneMarrow/train.ipynb` → `BoneMarrow/postprocessing.ipynb` | 42 | 4,096 | 10 | 256 | 200 |
| Cancer | `KP_tracer/preprocessed_adata.h5ad` | 33,179 × 1,999 | `Cancer/train.ipynb` | 42 | 8,192 | 8 | 256 | 100 |
| Dentate Gyrus | `DentateGyrus/preprocessed_adata.h5ad` | 2,460 × 1,500 | `Dentategyrus/train.ipynb` → `Dentategyrus/postprocessing.ipynb` | 42 | 2,048 | 12 | 256 | 250 |
| Erythroid | `Erythroid/preprocessed_adata.h5ad` | 9,815 × 5,000 | `Erythroid/train.ipynb` → `Erythroid/postprocessing.ipynb` | 42 | 4,096 | 16 | 256 | 150 |
| Mouse Cortex | `Mouse_Cortex/preprocessed_adata_train.h5ad` | 12,814 × 1,062 | `Mouse_cortex/Neuronal_train.ipynb` → `Mouse_cortex/Perturbation.ipynb` and `Mouse_cortex/Unseen.ipynb` | 42 | 8,192 | 30 | 256 | 100 |
| Pancreas | `Pancreas/preprocessed_adata.h5ad` | 16,822 × 5,000 | `Pancreas/train.ipynb` → `Pancreas/postprocessing.ipynb` | 42 | 4,096 | 50 | 256 | 100 |
| Unseen Pancreas | Pancreas input excluding `Fev+ Alpha` and `Fev+ Beta` | 15,050 × 5,000 | `Unseen_Pancreas/unseen_Pancreas.ipynb` | 42 | 4,096 | 50 | 256 | 100 |
| Vignette | `scvelo.datasets.bonemarrow()` | 5,780 × 14,319 before preprocessing | `Vignette/Tutorial_notebook.ipynb` | 42 | 1,024 | 12 | 256 | 150 |
| Zebrafish | `Zebrafish/preprocessed_adata_train.h5ad` | 2,068 × 2,000 | `Zebrafish/train.ipynb` → `Zebrafish/postprocessing.ipynb` | 42 | 2,048 | 40 | 256 | 250 |

Configuration values not listed below retain the defaults defined by
`LSDConfig` in the recorded software revision.

| Workflow | Non-default configuration |
|---|---|
| Bone Marrow | `optimizer.kl_schedule.af=2` |
| Cancer | `model.V_coeff=1e-4`; `x_encoder=[512,256,128]`; `potential=[16,16]`; `adam.lr=2e-3`; `wasserstein_schedule.max_W=1e-2`; `kl_schedule.af=2` |
| Dentate Gyrus | `adam.lr=2e-3` |
| Erythroid | `model.V_coeff=5e-3`; `kl_schedule.af=3` |
| Mouse Cortex | `potential=[16,16]`; `adam.lr=2e-3`; `adam.T_0=100` |
| Pancreas | `model.V_coeff=5e-3`; `kl_schedule.af=3`; `adam.lr=2e-3` |
| Unseen Pancreas | `model.V_coeff=0.005`; `kl_schedule.af=3`; `adam.lr=2e-3` |
| Vignette | `adam.lr=2e-3` |
| Zebrafish | `potential=[16,16]`; `adam.lr=2e-3`; `adam.T_0=60` |

## Runtime and memory reporting

End-to-end Bone Marrow and Cancer benchmarks were run on the H100 server with
32 CPU threads available to each process. Each run used a fresh process and the
audited notebook configuration. The scripts measured prior-transition,
random-walk, and training wall and CPU time internally, while an independent
monitor sampled process-tree RSS, GPU process memory, GPU utilization, and
power. Existing notebook progress-bar times were not used as benchmark data.

| Dataset | Implementation | Total wall | Peak CPU RSS | Peak GPU process memory | Peak PyTorch allocation |
|---|---|---:|---:|---:|---:|
| Bone Marrow | dense | 1,660.78 s | 3.05 GiB | 1.63 GiB | 0.79 GiB |
| Bone Marrow | sparse | 1,596.67 s | 2.89 GiB | 1.67 GiB | 0.78 GiB |
| Cancer | dense | 649.23 s | 33.25 GiB | 6.96 GiB | 6.14 GiB |
| Cancer | sparse | 698.30 s | 6.62 GiB | 1.39 GiB | 0.54 GiB |

For Cancer, the sparse implementation reduced peak CPU RSS by 80.08%, peak GPU
process memory by 80.04%, and peak PyTorch allocation by 91.20%. The total wall
time was 7.56% longer, illustrating that the revision targets memory safety and
does not guarantee faster model training. During Cancer walk generation, peak
PyTorch allocation fell from 6.14 GiB to approximately 10 MiB because the
complete cell-by-cell transition matrix was no longer transferred to the GPU.

Native dense and sparse runs use different random-walk samplers and therefore
do not consume identical trajectories from the same seed. A controlled
one-epoch validation froze the dense walk tensor and verified that both
implementations loaded the identical content hash for each dataset. The full
methodology, stage-level results, limitations, exact commits, and rerun commands
are in [`benchmarks/H100_BENCHMARK_REPORT.md`](benchmarks/H100_BENCHMARK_REPORT.md).

The package repository contains the
[comment 7 sparse-memory benchmark report](https://github.com/csglab/sclsd/blob/8fc9285426b98e4e3d078ec50e849175b22d3fd4/benchmarks/results/comment7_benchmark_report.md)
and its adjacent raw results. Those five-repetition measurements address
function-level transition construction and walk-generation scaling and
complement the end-to-end results above.
