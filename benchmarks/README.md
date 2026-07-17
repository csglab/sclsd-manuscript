# End-to-end training benchmarks

These scripts measure the Bone Marrow and Cancer training configurations from
the manuscript notebooks without modifying or executing the notebooks. Plotting,
postprocessing, checkpoint writes, and model-artifact writes are intentionally
excluded so that the reported training measurement reflects computation rather
than visualization or filesystem performance.

Every run verifies the source notebook hash, dataset hash and dimensions,
effective configuration, imported `sclsd` source tree, and Git commit before
training. Results are written to a new directory and never overwrite notebook
artifacts.

## Implementations

- Dense: `../sclsd-dense`, branch `sclsd-dense`
- Sparse: `../sclsd`, branch `sclsd-sparse`

Both use the same `sclsd-env` interpreter and dependencies. The monitor sets
`PYTHONPATH` explicitly and the runner aborts if Python imports from the wrong
worktree.

## Current H100 server

From the `sclsd-manuscript` directory, run a one-epoch safety check first:

```bash
../sclsd-env/bin/python benchmarks/run_matrix.py \
  --phase smoke \
  --mode native \
  --datasets bone_marrow cancer \
  --cpu-threads 32 \
  --output-root benchmark-results
```

Then run the full notebook configurations:

```bash
../sclsd-env/bin/python benchmarks/run_matrix.py \
  --phase full \
  --mode native \
  --datasets bone_marrow cancer \
  --cpu-threads 32 \
  --output-root benchmark-results
```

The monitor refuses to start when the selected GPU already has a compute
process. Use `--allow-busy-gpu` only when shared-GPU measurements are explicitly
intended and will be reported as such.

## Controlled training comparison

Native dense and sparse runs use different samplers and therefore produce
different walks from the same seed. A controlled run first freezes walks from
the dense implementation and then trains both implementations with that exact
walk tensor:

```bash
../sclsd-env/bin/python benchmarks/run_matrix.py \
  --phase full \
  --mode controlled \
  --datasets bone_marrow cancer \
  --cpu-threads 32 \
  --output-root benchmark-results
```

Native runs are the primary practical benchmark. Controlled runs isolate model
training from random-walk sampling differences.

## Measurements

`run_training.py` records stage-level wall time, CPU time, PyTorch peak allocated
GPU memory, and PyTorch peak reserved GPU memory. `monitor_run.py` independently
samples the complete process tree and NVIDIA device to record peak CPU RSS,
CPU utilization, process GPU memory, device GPU memory, GPU utilization, and
power. It also preserves stdout and exceptions, including an OOM and the stage
where it occurred.

Each result directory contains:

- `result.json`: provenance, configuration, stage metrics, and run status;
- `monitor.json`: sampled CPU/GPU measurements and their summary;
- `stdout.log`: complete benchmark output;
- `summary.csv` and `summary.md` at the matrix root.

The same scripts can later be run on the RTX 3090. Hardware is detected and
recorded at runtime; results must only be reported for machines on which the
commands were actually executed.

The completed H100 measurements and their interpretation are recorded in
[`H100_BENCHMARK_REPORT.md`](H100_BENCHMARK_REPORT.md). Raw outputs are under
[`results/h100/`](results/h100/).
