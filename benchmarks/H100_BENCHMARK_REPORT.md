# H100 end-to-end training benchmark

This report records the end-to-end Bone Marrow and Cancer training benchmarks
run on July 17, 2026. The benchmark scripts reproduce the model construction,
prior-transition construction, random-walk preparation, and training settings
from the audited manuscript notebooks. They do not execute or modify the
notebooks. Plotting, postprocessing, checkpoint loading or saving, and model
artifact writes are excluded.

## Hardware and software

Only the server used for these measurements is reported here:

- GPU: NVIDIA H100 80GB HBM3
- NVIDIA driver: 560.35.05
- driver CUDA compatibility: 12.6
- PyTorch CUDA runtime: 12.1
- CPU: Intel Xeon Platinum 8468
- CPU threads available to each benchmark process: 32
- Python: 3.10.12
- PyTorch: 2.4.1
- `sclsd`: package version 0.3.0, with the exact source commits below

The legacy dense transition and GPU-walk implementation was run from commit
`7e1937986419b66d69d3f404add114d8dccba012` on branch `sclsd-dense`. The sparse
transition and CPU-walk implementation was run from commit
`8fc9285426b98e4e3d078ec50e849175b22d3fd4` on branch `sclsd-sparse`.

The scripts detect and record hardware at runtime, so Ali Poursina can repeat
the same matrix later on an RTX 3090. No RTX 3090 benchmark result is reported
until that run is performed.

## Notebook-faithful configurations

| Dataset | Cells x genes | Walks x length | Batch | Epochs | Seed |
|---|---:|---:|---:|---:|---:|
| Bone Marrow | 5,292 x 2,000 | 4,096 x 10 | 256 | 200 | 42 |
| Cancer | 33,179 x 1,999 | 8,192 x 8 | 256 | 100 | 42 |

The benchmark aborts unless the notebook SHA-256, input-data SHA-256, data
shape, effective model configuration, imported source path, Git commit, and
dense-versus-sparse sampler identity match the recorded manifest.

## Native end-to-end results

Each row is one complete fresh-process run. CPU time is the process CPU time
during training. CPU utilization is normalized to the 32 available threads.
Peak GPU process memory and GPU utilization are sampled independently with
NVIDIA's management interface; peak allocated memory is also recorded by
PyTorch. Times do not include interpreter startup or benchmark-monitor startup.

| Dataset | Implementation | Total wall | Training wall | Training CPU time | Peak CPU RSS | Peak GPU process memory | Peak PyTorch allocation | Mean training GPU utilization | Mean training CPU utilization |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Bone Marrow | dense | 1,660.78 s | 1,660.03 s | 2,035.96 s | 3.05 GiB | 1.63 GiB | 0.79 GiB | 36.26% | 3.83% |
| Bone Marrow | sparse | 1,596.67 s | 1,596.12 s | 1,995.92 s | 2.89 GiB | 1.67 GiB | 0.78 GiB | 36.66% | 3.91% |
| Cancer | dense | 649.23 s | 630.47 s | 1,009.99 s | 33.25 GiB | 6.96 GiB | 6.14 GiB | 34.63% | 5.00% |
| Cancer | sparse | 698.30 s | 696.90 s | 1,068.71 s | 6.62 GiB | 1.39 GiB | 0.54 GiB | 34.01% | 4.78% |

The sparse Cancer run reduced peak CPU RSS by 80.08%, peak GPU process memory
by 80.04%, and peak PyTorch allocation by 91.20% relative to the dense run.
Its total wall time was 7.56% longer. Bone Marrow used similar full-training GPU
memory in both implementations because model training, rather than its smaller
transition matrix, set the peak. Its sparse run used 5.10% less peak CPU RSS
and completed 3.86% faster. These are individual full runs, not timing medians,
and the native implementations do not consume identical random walks.

## Transition and walk stages

| Dataset | Implementation | Prior-transition wall | Walk wall | Peak PyTorch allocation during walk stage |
|---|---|---:|---:|---:|
| Bone Marrow | dense | 0.4857 s | 0.1688 s | 0.2760 GiB |
| Bone Marrow | sparse | 0.0059 s | 0.4417 s | 0.0100 GiB |
| Cancer | dense | 14.8853 s | 3.1517 s | 6.1373 GiB |
| Cancer | sparse | 0.0281 s | 0.6503 s | 0.0102 GiB |

For Cancer, the sparse path eliminated the dense transition tensor from the
GPU walk stage: peak PyTorch allocation during that stage fell from 6.14 GiB
to approximately 10 MiB, the model's existing baseline allocation. The sparse
walk stage used 0% sampled GPU utilization because it runs on CPU. This result
supports the specific claim that the revised main workflow no longer transfers
the complete cell-by-cell transition matrix to GPU. It does not establish that
all possible out-of-memory causes elsewhere in the pipeline have been removed.

## Controlled-walk validation

The same seed does not produce identical native trajectories with CUDA
`torch.multinomial` and the sparse CPU sampler. A separate controlled one-epoch
validation therefore generated the walk tensor once with the dense sampler and
loaded that exact tensor for both dense and sparse training. Both implementations
recorded the same content hash for each dataset:

| Dataset | Fixed-walk content hash | Dense status | Sparse status |
|---|---|---|---|
| Bone Marrow | `e5e075974ee31075ff693013fc4425e0d4387574e209cd901676e88c0f42abea` | passed | passed |
| Cancer | `c5739a7bdb17182a4952a733aeb1a45c60b81ff1dd732684f77973b8e250ed74` | passed | passed |

This validates the fixed-walk comparison mode and isolates model training from
the CPU-versus-GPU sampling difference. Native runs remain the practical
end-to-end measurements.

## Scaling interpretation

The dense path materializes cell-by-cell structures, so its transition memory
scales quadratically with cell count. The sparse revision operates over nonzero
neighbor-graph edges and scales with graph `nnz` for transition construction
and sampling. The 33,179-cell Cancer result demonstrates the practical effect:
the sparse run used about one fifth of the dense peak CPU and GPU process
memory. The two end-to-end workflows also differ in model architecture, walk
configuration, and epoch count, so they should not be used to fit a general
runtime-scaling exponent. The five-repetition function-level scaling benchmarks
for transition and walk construction are reported separately in the package
repository.

## Evidence and rerunning

The full native outputs are in
`benchmarks/results/h100/native-full-20260717T185454Z/`. The controlled smoke
outputs are in
`benchmarks/results/h100/controlled-smoke-20260717T201245Z/`. Each run contains
`result.json`, `monitor.json`, and `stdout.log`; each matrix contains generated
CSV and Markdown summaries.

Commands and measurement definitions are documented in `benchmarks/README.md`.
The scripts can be rerun on another NVIDIA server without source changes; the
result must be labeled with the hardware detected in that run.
