# analysis

The analyses of the feature matrix and of the labour-market variables. All
of them read the tables built in `data_prep/` and write to `output/`.

## Order and dependencies

```
python analysis/clustering.py                # master_clean -> clustering/
python analysis/resampling_stability.py      # master_clean -> resampled_spectra.npz, resampling_stability/,
                                             #                 assets/subspace_stability.png
python analysis/pca_rotated.py               # master_clean -> occupation_coordinates.csv, axis_loadings.csv
python analysis/validate_axes.py [onet dir]  # axes, loadings, O*NET text files -> validate_axes/
python analysis/content_model_halves.py      # master_clean, axes -> content_model_halves/
python analysis/check_curvature.py           # master_clean, resampled_spectra.npz -> check_curvature/
python analysis/residual_structure.py [onet dir]  # master_clean, resampled_spectra.npz, O*NET text files
                                                  #   -> residual_structure/
python analysis/labour_market_regression.py  # master, labour, master_clean, labour_clean, axes -> orthogonality.xlsx
python analysis/acs_young_share.py           # IPUMS extract or its aggregate, axes, labour_clean -> acs_young_share.csv
```

"axes" and "loadings" are `occupation_coordinates.csv` and
`axis_loadings.csv`. Paths are in `output/` unless they start with
`assets/`.

- `clustering.py` needs nothing but `master_clean.csv`. It regresses its
  k = 2 labels on the axes only if `occupation_coordinates.csv` already
  exists; otherwise it says so and skips that table.
- `pca_rotated.py` does not need `resampling_stability.py` to run: the
  groups of components it rotates are fixed by hand in the script
  (`STABLE_COMPONENTS`, `STABLE_SUBSPACE`), copied from
  `resampling_stability/dimensions.csv`. It warns if they differ from what
  `resampling_stability.py` last found. Rerun `resampling_stability.py`
  whenever `master_clean.csv` changes.
- `check_curvature.py` and `residual_structure.py` read `k_max` (and
  `check_curvature.py` the split-halves) from `resampled_spectra.npz`.
- `validate_axes.py` and `residual_structure.py` read the O\*NET text files,
  from `data_raw/onet/db_31_0_text` or the folder given as the argument.
  Without them, `validate_axes.py` writes only its corner table and
  `residual_structure.py` stops.
- `validate_axes.py` also reads `output/skillscape/skill_clusters.csv` from
  `supplementary/skillscape.py`, if it is there.
- `acs_young_share.py` builds its occupation-year aggregate from the IPUMS
  extract the first time and reuses it afterwards.

## What each script answers

| script | question |
|---|---|
| `clustering.py` | Do occupations fall into k-means clusters? How do the k = 2 partition and the SOC major groups sit in the space? |
| `resampling_stability.py` | Which components, blocks of components and rotated axes reproduce across resamples of the occupations? Defines `k_max` by parallel analysis. |
| `pca_rotated.py` | The rotated axes R1-R6: one PCA, varimax within each fixed group of components, signs set by anchor columns. |
| `validate_axes.py` | Which occupations lie deepest in the corners of R1-R3? Do the axes agree with O\*NET descriptors that take no part in forming them (categories, interests, job zones, software, work styles)? |
| `content_model_halves.py` | Do the job half and the worker half of the Content Model, which share no column, find the same axes on their own? |
| `check_curvature.py` | Are R1-R3 related non-linearly? Are R4-R6 curvature of R1-R3? Is each of PC7 to PC `k_max` a non-linear function of PC1-6? |
| `residual_structure.py` | How much of the variance beyond PC6 is rating error, curvature of PC1-6, local signal and remainder? Which occupations make up the heaviest tail? |
| `labour_market_regression.py` | How much of each labour-market variable can the feature columns predict? |
| `acs_young_share.py` | Where in the space did the share of young workers change between 2022 and 2024? |

`check_curvature.py` tests curvature component by component and at the
boundaries between groups; the share of all variance beyond PC6 that is
curvature is in `residual_structure.py`.

## Supplementary

In `analysis/supplementary/`, run as needed:

| script | question |
|---|---|
| `variants_experiment.py` | Does the resampling experiment find the same dimensions on other input matrices: Level instead of Importance, both, without Knowledge, the Work Context categories alone, reweighted blocks, all scales together, rank-normal scores? |
| `joint_rotation.py` | What do PC1-6 look like rotated together by one varimax? |
| `skillscape.py` | Does the skill-skill similarity of Alabdulkareem et al. (2018) split skills into two clusters, and does a plain correlation similarity do the same? |

## Shared code

Defined once and imported elsewhere:

- `pca_rotated.py`: the paths (`MASTER`, `OUTDIR`, `AXES`, `LOADINGS`), the
  feature columns (`FEATURE_PREFIXES`), `SEED`, and `load_and_prep`,
  `varimax`, `rotate_group`, `rotate_groups`, `group_scores`. Its last
  section holds helpers used only by other scripts: `standardize`, `eigen`,
  `congruence`, `subspace_cosines`, `match_axes`, `block_of`.
- `resampling_stability.py`: `run_experiment` (the whole resampling
  experiment on any z-scored matrix, used by `content_model_halves.py` and
  `variants_experiment.py`), `found_groups`, `load_spectra`, `STABLE_THRESH`.
- `check_curvature.py`: `block_r2` and `oof_predictions`, used by
  `residual_structure.py`.

## Conventions

- Results go to `output/`, one CSV per table; figures as PNG. The console
  reports run status only.
- Scripts select columns by prefix (`skills_im__`, `workctx_cx__`, ...);
  there is no shared column list beyond `FEATURE_PREFIXES`.
- Seeds are fixed (`SEED = 0`), so every run gives the same results.
- Axis signs are set by anchor columns in `pca_rotated.py`, so they stay the
  same when the data are rebuilt.
- An OEWS wage of exactly 239,200 means "at least this much" (the top code).