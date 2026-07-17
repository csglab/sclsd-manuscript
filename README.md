# A Latent Space Thermodynamic Model of Cell Differentiation

This repository contains the dataset-specific notebooks used for the analyses
and figures in the LSD manuscript. This README is the maintained guide to the
repository and its reproducibility materials.

For the sclsd package, see [csglab/sclsd](https://github.com/csglab/sclsd).
The manuscript is available as a
[bioRxiv preprint](https://www.biorxiv.org/content/10.64898/2026.03.04.709512v1).

## Datasets

| Folder | Dataset |
|--------|---------|
| BoneMarrow | Human hematopoiesis |
| Cancer | Lung adenocarcinoma progression |
| Dentategyrus | Dentate gyrus neurogenesis |
| Erythroid | Mouse erythroid gastrulation |
| Mouse_cortex | Mouse cortical development |
| Pancreas | Pancreatic endocrinogenesis |
| Zebrafish | Zebrafish axial mesoderm development |
| Vignette | Human hematopoiesis |
| Unseen_Pancreas | Pancreatic endocrinogenesis with held-out populations |

## Setup

1. Create and activate a Python 3.10 environment:

   ```bash
   conda create -n sclsd python=3.10 -y
   conda activate sclsd
   ```

2. Install the package and notebook dependencies:

   ```bash
   python -m pip install --upgrade pip
   pip install torch==2.4.1
   pip install sclsd
   pip install ipykernel ipywidgets gseapy scvelo
   ```

3. [Download the preprocessed datasets from Zenodo](https://zenodo.org/records/18331587).
   Extract the dataset directories under this repository's `Zenodo/` directory
   so that paths such as `Zenodo/BoneMarrow/preprocessed_adata.h5ad` exist.

## Running the notebooks

Notebooks are organized by dataset under `Notebooks/`. For analyses that have
separate training and postprocessing notebooks, run the training notebook first
to generate the model artifacts consumed during postprocessing. The notebooks
use paths relative to their own directories, such as `../../Zenodo/`; run each
notebook with its working directory set to the directory containing that
notebook.

The notebooks use stochastic model training. The exact input files and
checksums, dataset dimensions, package revisions, tested software environment,
random seeds, model configurations, notebook order, and hardware are recorded
in [REPRODUCIBILITY.md](REPRODUCIBILITY.md). Controlled end-to-end runtime and
peak-memory measurements are tracked separately from this static
reproducibility record.

## Vignette: end-to-end example from raw data

The `Vignette/` directory provides an end-to-end example of training an LSD
model from a raw single-cell dataset, including preprocessing, model fitting,
and post-training analysis. It illustrates the workflow underlying the
preprocessed AnnData objects distributed through Zenodo and complements the
dataset-focused notebooks.

## Citation

If you use `sclsd` or the accompanying analyses, please cite:

> Poursina, A., Hajhashemi, S., Mikaeili Namini, A., Saberi, A., Emad, A., &
> Najafabadi, H. S. (2026). A Latent Space Thermodynamic Model of Cell
> Differentiation. *bioRxiv*, 2026.03.04.709512.
> https://doi.org/10.64898/2026.03.04.709512

[View version 1 on bioRxiv](https://www.biorxiv.org/content/10.64898/2026.03.04.709512v1)

### BibTeX

```bibtex
@article{poursina2026latent,
  title     = {A Latent Space Thermodynamic Model of Cell Differentiation},
  author    = {Poursina, Ali and Hajhashemi, Shayan and
               {Mikaeili Namini}, Arsham and Saberi, Ali and
               Emad, Amin and Najafabadi, Hamed S.},
  journal   = {bioRxiv},
  pages     = {2026.03.04.709512},
  year      = {2026},
  publisher = {Cold Spring Harbor Laboratory},
  doi       = {10.64898/2026.03.04.709512},
  url       = {https://www.biorxiv.org/content/10.64898/2026.03.04.709512v1}
}
```

## Contact

For questions about the software or reproducibility materials, contact Ali
Poursina at [ali.poursina@mail.mcgill.ca](mailto:ali.poursina@mail.mcgill.ca).
