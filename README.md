# sclsd-manuscript

Notebooks for reproducing analyses and figures from the LSD manuscript.

For the sclsd package, see [csglab/sclsd](https://github.com/csglab/sclsd).

## Datasets

| Folder | Dataset | Reference |
|--------|---------|-----------|
| BoneMarrow | Human hematopoiesis | Setty et al., Nat Biotechnol 2019 |
| Cancer | Lung adenocarcinoma progression | Yang et al., Cell 2022 |
| Dentategyrus | Dentate gyrus neurogenesis | Hochgerner et al., Nat Neurosci 2018 |
| Erythroid | Mouse erythroid gastrulation | Pijuan-Sala et al., Nature 2019 |
| Mouse_cortex | Mouse cortical development | Zheng et al., Cell 2024 |
| Pancreas | Pancreatic endocrinogenesis | Klein et al., Nature 2025 |
| Zebrafish | Zebrafish axial mesoderm development | Farrell et al., Science 2018 |
| Vignette | Human hematopoiesis | Setty et al., Nat Biotechnol 2019 |
| Unseen_Pancreas | Human hematopoiesis | Setty et al., Nat Biotechnol 2019 |

## Setup

1. Create and activate a virtual environment:
   ```bash
   conda create -n sclsd python=3.10 -y
   conda activate sclsd
   ```

2. Install sclsd:
   ```bash
   pip install torch==2.4.1
   pip install sclsd
   ```

3. Install dependencies to run notebooks:
   ```bash
   pip install ipykernel ipywidgets
   ```

4. Install gseapy for gene set enrichment analysis:
   ```bash
   pip install gseapy
   ```

5. Download preprocessed data from Zenodo: [https://zenodo.org/records/18331587]

## Vignette: end-to-end example from raw data
The `Vignette/` directory provides an end-to-end example of training an LSD model starting from a raw single-cell dataset, including preprocessing, model fitting and post training analysis. This vignette is intended to illustrate the full workflow underlying the preprocessed AnnData objects distributed here and complements the dataset-focused notebooks in the reproducibility repository.



## Citation

> Poursina A, Hajhashemi S, Mikaeili Namini A, Saberi A, Emad A, Najafabadi HS. A Latent Space Thermodynamic Model of Cell Differentiation. 2026.