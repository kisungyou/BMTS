# Bayesian mixture transport signatures

This repo is a notebook-first public companion to *Bayesian Mixture
Transport Signatures for Uncertainty-Aware Observation Matching*. It contains
the reusable BMTS routines, executable versions of the two simulations and two
real-data experiments, a compact public HSPC derivative, and archived
paper-scale summaries.

The release is intentionally smaller than the research workspace. Manuscript
sources, build products, duplicate figure formats, preparation caches, and
internal logs are not included. Each checked-in notebook is executed, so its
tables and figures render directly on GitHub.

## Quick start

Use Python 3.10 or newer:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[notebooks]"
jupyter lab
```

Open any file under `notebooks/`. Every notebook locates this folder whether
Jupyter starts here or inside `notebooks/`.

## Notebook map

| Notebook | Manuscript study | Main visual checks |
|---|---|---|
| `simulation1_nearby_and_distant_components.ipynb` | Simulation 1: component geometry | fitted geometry, PSM/BMTS heatmaps, pair distributions, uncertainty graph, and misspecification check |
| `simulation2_residual_aware_matching.ipynb` | Simulation 2: residual awareness | broad-component fit, paired heatmaps, within/between distributions, and distance graph |
| `realdata1_iris_relations.ipynb` | Real data 1: Fisher iris | label-free PCA, pairwise heatmaps, posterior relation graph, and uncertainty summaries |
| `realdata2_hematopoietic_boundaries.ipynb` | Real data 2: Nestorowa HSPCs | RNA-PC maps, cross-modality associations, phenotype-status comparison, and sensitivity analysis |

The filenames follow the studies directly rather than using numerical prefixes
or generic names such as “quickstart.”

## Fresh and paper-scale calculations

Each notebook has a `PAPER_SCALE` switch near the beginning.

- The default fresh run uses deterministic seeds and shorter chains suitable
  for an interactive kernel. All displayed BMTS quantities and figures are
  recomputed locally.
- Setting `PAPER_SCALE=True` restores the chain lengths used for the manuscript.
  The compact `data/paper_scale_summary.json` is always available for comparison
  without rerunning the longer analysis.

The four checked-in notebooks run fully offline. The two simulations generate
their own data, Fisher iris is distributed with scikit-learn, and the Nestorowa
experiment reads the included pickle-free derivative.

The release was executed end to end with Python 3.11.15, NumPy 2.3.5, SciPy
1.17.0, Matplotlib 3.10.8, scikit-learn 1.8.0, pandas 2.2.3, nbclient 0.11.0,
and nbformat 5.10.4.

## License

The BMTS software is distributed under the MIT License; see `LICENSE`. The included Nestorowa derivative retains
its separately documented CC0 source status.

## Composition

- `bmts.py` contains the public samplers, finite-metric transport routines,
  residual-aware extension, posterior summaries, data helpers, and the shared
  manuscript-aligned Matplotlib `rcParams`.
- `notebooks/` contains the four executable studies and keeps experiment-specific
  construction visible.
- `data/` contains source metadata, the compact Nestorowa derivative, and the
  archived paper-scale summary.
- `tests/test_bmts.py` contains deterministic numerical and portability checks.
- `results/` is reserved for user-generated outputs; the public notebooks do
  not write into it.
- `LICENSE` and `CITATION.cff` record the MIT terms and sole-author citation
  metadata.

## Numerical scope

For a metric ground cost on a small component dictionary, `bmts.py` uses an
exact finite Kantorovich--Rubinstein dual-vertex method. Vertex enumeration is
combinatorial in the number of mixture components; the released examples use
at most four components for BMTS calculations. For larger dictionaries,
nonmetric costs, or pair-specific residual costs, use a scalable transport
solver or the included primal linear-programming routine.

The mixture samplers keep component covariances fixed so the notebooks remain
transparent and fast. They are study implementations, not a general-purpose
mixture-model library.

## Verification

From this folder, run:

```sh
python tests/test_bmts.py
python bmts.py
```

Licensing and sole-author citation metadata are included in `LICENSE` and
`CITATION.cff`. The remaining optional publication steps are recorded in
`RELEASE_CHECKLIST.md`.
# BMTS
