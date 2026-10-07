# Occupational structure and exposure to automation

Occupation-level analysis of the occupational feature space: what separates
jobs from one another, and which of those differences are about skill and which
are about institutional position.
Built on O*NET 31.0 (910 detailed occupations) joined to wages, union coverage,
occupational prestige, separation rates and self-employment.

**[Paper draft, with interactive figures](https://alicebobcharlie.github.io/AI_and_employment/)** (source: `paper_draft.qmd`)

Everything reads and writes through `output/`. Run from the project root, the
folder containing `data_raw/`.

## Run order

```
python build_master.py          #  1. assemble the raw table          -> master.csv
python clean_master.py          #  2. drop, fill and derive           -> master_clean.csv
python check_data.py            #  3. data quality                    (Sections 4.2-4.3)

python cluster_onet.py          #  4. are there clusters at all?       (Section 5.2)
python pca_rotated.py           #  5. two tiers, six rotated axes     (Sections 5.3 and 5.5, Figure 5.1)
python validate_tier1.py        #  6. do the tier-1 names hold up?     (Section 5.3; needs 4 and 5)
python validate_tier2.py        #  7. R4-R6 without tier-1 curvature   (Section 5.5; needs 5)
python tier2_residual_stability.py  # 8. does that residual reproduce? (Section 5.5)
python content_model_halves.py  #  9. job and worker halves           (Section 5.4; needs 5)
python check_curvature.py       # 10. curvature beyond the six axes   (Section 5.5)
python rating_error.py          # 11. rating error beyond the axes    (Section 5.5)
python check_orthogonality.py   # 12. what the features can predict   (Section 5.6)
python plot_axes.py             # 13. data for the figures             (needs 5)
python corner_table.py          # 14. Table 6.1                        (needs 5)
python acs_young_share.py       # 15. data for Figure 6.2              (needs 5 and the ACS extract)
```

Steps 1 and 2 run in that order; every analysis afterwards reads
`output/master_clean.csv`, and `check_data.py`, `check_orthogonality.py` and
`supplementary/input_variants.py` also read `output/master.csv`.
`check_data.py`, `validate_tier1.py`, `validate_tier2.py` and `rating_error.py`
also read the O*NET text files in `data_raw/onet/db_31_0_text` (the two
validation scripts skip their expert-category tables without them).
`validate_tier1.py` needs the outputs of both `cluster_onet.py` and
`pca_rotated.py`.

The scripts in `supplementary/` are run as needed rather than as part of the
sequence; see below.

## Two tables

| table | what it is |
|---|---|
| `output/master.csv` | The raw table: every O*NET-SOC code in O*NET 31.0, rated or not; every scale of the five feature blocks and of education, training and experience, as published; the labour-market variables joined on the six-digit SOC code. Nothing dropped, filled or derived. |
| `output/master_clean.csv` | Analysis-ready: the 910 occupations rated in all five feature blocks, the 216 feature columns, the 37 education, training and experience columns, and the labour-market variables, every gap filled and the derived columns added. All of this lives in `clean_master.py` and nowhere else. |
| standardisation | Deliberately **not** stored. Each analysis z-scores its own matrix, because `check_orthogonality.py` must fit its scaler on training folds only; pre-standardising would leak test-fold information into the cross-validated R². |

## Building the data

**`build_master.py`** → `master.csv`

Reads O*NET 31.0 from `data_raw/onet/db_31_0_text` and the external sources
from `data_raw/`, and joins them. It only reads and joins. Two source codes are
read rather than left blank: an OEWS wage marked `#` (at or above $239,200 a
year) is read as 239200, and a blank or dash in BLS table 1.2 (negligible
self-employment) is read as 0.

**`clean_master.py`** → `master_clean.csv`

Keeps the occupations rated in all five feature blocks, keeps the Importance
scale for four blocks and the Context scale for Work Context, merges the two
education scales (RL and RQ), drops the highest category of each education,
training and experience scale, and fills the remaining gaps: self-employment at
0 for occupations absent from the BLS table, wages and everything else at the
column median. Adds the three wage ratios, `ext_employment_log`,
`ext_wage_level_log` and `ext_wage_disp_p90p10`. The Level scale stays in the
raw table only.

## Results

**`cluster_onet.py`** → `cluster_occupations.png`,
`cluster_skill_similarity.png`, `cluster_assignments.csv`, `cluster_skills.csv`,
`cluster_kurtosis.xlsx`

Whether the polarisation reported in earlier work is a property of skills or of
the way the skill matrix is built. Part A asks whether occupations themselves
fall into groups under continuous or binarised ratings — they do not, which is
what licenses describing them with continuous axes. Part B goes after the
published claim directly, which is about skills rather than occupations:
skill-skill similarity is rebuilt both by the published route (binarise at
RCA > 1, then take the minimum conditional co-occurrence probability) and
straightforwardly by correlating the published ratings across occupations, and
the two are compared on how bimodal the resulting edge weights are and how
separable the skill clusters are. The published network is also thresholded
before its communities are read off, so the script reports how much of the data
that step removes. Part C asks whether the components after the first three
describe a small, separate group of occupations (the "local feature" reading of
their instability). k-means is poor at finding a small group next to a large
one, so it uses kurtosis instead: in each subspace (PC1-3, PC4-6, PC7-14,
PC4-14, PC1-14) a projection pursuit finds the most and least kurtotic
directions, compared with the same search on Gaussian data of the same size
and dimension, with an outlier check, the occupation counts along the tail and
the occupations in it. Nothing here uses the rotated axes; the comparisons that
do are in `validate_tier1.py`.

**`pca_rotated.py`** → `rotated_axes.csv`, `rotated_loadings.csv`,
`oblique_pattern.csv`, `pca_stability.xlsx`, `scree.png`,
`assets/subspace_stability.png` (Figure 5.1)

The main analysis, on the 216 feature columns only, in two tiers. Tier 1:
eigenvalues and parallel analysis (reported, not used to choose), then
bootstrap and split-half congruence of each unrotated component, which finds
the components that reproduce one by one (the first three). Tier 2: the
eigenvalue gaps against their sampling error (North's ratio), which shows
components 4-6 to be nearly degenerate; split-half principal angles between
the leading-k subspaces, k = 1 to 14 (Figure 5.1); then each candidate block
after tier 1 judged as a whole, by the root mean square of the principal-angle
cosines between the two halves' block subspaces (p05 >= 0.90 over 100
split-halves). The data picks components 4-6; the rotated axes of each
candidate, and every block after tier 2 up to component 14, are reported as
well. Each tier is varimax-rotated on its own, so R1-R3 are the same axes
whatever tier 2 turns out to be, and each axis is signed by a marker variable.
The labour-market variables, which took no part in forming the axes, are
correlated against all six, and a promax rotation of tier 1 checks whether its
axes are orthogonal by preference of the data or only by construction.

**`validate_tier1.py`** → terminal

Whether R1-R3 are what they are called. The names came from reading loadings,
which is a judgement, and every substantive claim rests on them. Checks, each
using something the analysis did not see: O*NET's own expert grouping of
elements (this needs the O*NET text files in `data_raw/`); whether the
off-diagonal cells of the R1 × R2 plane are populated, with the median wage in
each; the occupations at each extreme of the three axes; whether adding the
labour-market variables to the decomposition changes the axes; the facts the
paper states about the poles of R1 and R2; and the parts of Section 5.2 that
need the axes: how the two-cluster split of occupations and the two skill
clusters line up with them, and the SOC major groups in the space of the
leading components.

**`validate_tier2.py`** → `validate_tier2.xlsx`

What R4-R6 measure. First the part of each axis that R1-R3 can predict
non-linearly (quadratic, cubic and nearest-neighbour fits, out of fold) is
removed; the residual is then read through its correlations with the 216
columns, its share in the job and worker halves and in each block, the
occupations at its ends and in its middle, and its correlations with the
labour-market variables, required education and the Work Context items on
error and automation (and the Eloundou et al. exposure measures, if
`data_raw/eloundou_occ_level.csv` is present). Last, whether the labour-market
variables change the tier-2 axes.

**`tier2_residual_stability.py`** → `tier2_residual_stability.xlsx`

Whether tier 2 still reproduces once the curvature of tier 1 is taken out.
Each half of a split is decomposed on its own and has its own curvature removed
(cubic least squares, or nearest neighbours out of fold). Two versions: the
residual of the 4-6 scores, whose directions in column space are what
`validate_tier2.py` interprets, compared between halves as a subspace and axis
by axis; and the data matrix with PC1-3 and their curvature removed from every
column, decomposed again, to see whether it still leads with a stable
three-dimensional block in the space of the original components 4-6, and
whether the boundary after it holds. The original 4-6 block is scored on the
same splits as the baseline.

**`content_model_halves.py`** → `content_model_halves.xlsx`

Whether the job half of the Content Model (Work Activities, Work Context) and
the worker half (Skills, Abilities, Knowledge), which share no column, arrive
at the same axes on their own: each half's reproducing leading subspace, how
well it recovers each of R1-R6, canonical correlations between the halves, and
each half's own rotated axes against the other's.

**`check_curvature.py`** → `curvature_check.xlsx`

Whether components 7 to 14, and the second tier itself, are new directions or
curvature of the components before them: non-linear functions of the leading
components predicting the later ones, a permutation null, refits inside each
fold, normal scores against the floor of the rating scales, and synthetic data
with a known floor as a positive control.

**`rating_error.py`** → `rating_error.xlsx`

How much of each item, and of each principal component, is sampling error in
O*NET's ratings, from the published standard errors; an error-only spectrum as
a measured null for parallel analysis; and whether the items the six axes
explain poorly are simply the unreliable ones.

**`check_orthogonality.py`** → `orthogonality.xlsx`

How much of each labour-market variable the description of the work can
predict, by ridge regression with cross-validated R², from all 216 feature
columns and from each feature block alone (Section 5.6); the education,
training and experience block column by column; and the descriptive figures
Section 5.6 quotes. Values that `clean_master.py` filled in are left out, so every R² is
computed on published values only. Unlike anything read off a PCA, the answer
does not depend on how many feature columns happen to be in the matrix.

**`plot_axes.py`** → `plot_data.csv`

The table the interactive figures are drawn from: each occupation's position on
the three axes, with union coverage and employment. It needs `rotated_axes.csv`,
so run it after `pca_rotated.py`. The figures themselves are drawn in the
browser by plotly.js. `rotated_axes.csv` and `plot_data.csv` are both committed,
so the axes can be downloaded without running anything.

**`corner_table.py`** → terminal

Table 6.1: the ten occupations furthest into the corner positive on all three
axes and the corner negative on all three. Depth into a corner is the smallest
of the three coordinates taken with the corner's sign, so an occupation that is
extreme on one axis and near zero on another does not rank as deep. Prints a
Markdown table to paste into `paper_draft.qmd`. Needs `rotated_axes.csv`.

**`acs_young_share.py`** → `acs_young_share.csv`

Figure 6.2: the change in the share of workers aged 22 to 25 within each
occupation between 2022 and 2024, from the American Community Survey, matched to
O\*NET codes and projected onto the axes. Builds its occupation-year aggregate
from an IPUMS extract in `data_raw/` the first time, then reuses it. The
microdata is not redistributable and stays out of the repository; the
occupation-level output is committed so the web paper can draw the figure.

## Supplementary analyses

In `supplementary/`. Run from the project root, e.g.
`python supplementary/pca_ete.py`. None of them is reported in the main text
except where noted.

**`supplementary/input_variants.py`** → `input_variants.xlsx`

The same analysis on different inputs: Level instead of Importance and the two
together (Appendix A), without Knowledge, Work Context alone, and each half of
the Content Model (Appendix B). For each: parallel analysis, components
reproducing one by one, nested-subspace stability, the paper's tiers rotated
and tested, and agreement with the six axes. Its parallel-analysis counts are
the table in Section 5.5. Run after `pca_rotated.py`.

**`supplementary/block_weights.py`** → `block_weights.xlsx`

The same analysis with the blocks reweighted (Appendix B): every block given
the same total variance, and every block's leading eigenvalue set to 1 as in
multiple factor analysis. Run after `pca_rotated.py`.

**`supplementary/pca_ete.py`** → `ete_axes.csv`

Whether the education, training and experience block has axes of its own. Same
procedure — stability, then rotation — followed by correlating the resulting
axes with the rotated axes of `pca_rotated.py`, to see how far the preparation
a job demands is separable from what the work requires. Run after
`pca_rotated.py`.

## Data-quality checks

**`check_data.py`** → `data_quality.xlsx`

Every fact Sections 4.2 and 4.3 state about the quality of the data. Which
O*NET-SOC codes are rated, and what the unrated ones are; Recommend Suppress
and Not Relevant flags per block and scale; the share of ratings with a
standard error; how closely Importance and Level agree, element by element and
as whole matrices; what the two Work Context Category items would add; the
missing cells of the final matrix (Table 4.1); and the missingness and
distributions of the labour-market variables, which established that their
gaps are structural rather than scattered and which variables need a log. The
flags and standard errors are not in the master table, so this script reads
them from the O*NET text files.

A further one-off check established that BLS occupational openings is a size
proxy (r = 0.89 with employment), that the "typical education needed" threshold
is redundant with the ETE distributions (R² = 0.82), and that "work experience
needed" is mostly missing. All three were excluded on that basis.

## The web paper

The paper is written in `paper_draft.qmd` and published as a web page with
[Quarto](https://quarto.org).

| file | role |
|---|---|
| `paper_draft.qmd` | the paper: title and abstract in the header, then the text |
| `_quarto.yml` | page layout, fonts, and the scripts the page loads |
| `theme.scss` | typography and colours |
| `figures.js` | draws the interactive figures in the reader's browser with plotly.js, from `output/plot_data.csv` and `output/acs_young_share.csv` |

No Python runs when the page is built. The figures read `output/plot_data.csv`
and `output/acs_young_share.csv` directly, so after re-running the analysis, run
`plot_axes.py` and `acs_young_share.py` and commit the new CSVs for the page to
pick them up.

To preview locally, install Quarto and run `quarto preview` from the project
root. Opening `_site/index.html` straight from disk will not show the figures,
because browsers block a local file from loading the CSV.

Every push to `main` that touches one of the files above, or the CSV, triggers
`.github/workflows/publish.yml`, which renders the page and deploys it to GitHub
Pages. The rendered site is never committed. This requires Settings → Pages →
Source to be set to *GitHub Actions*.

## Conventions
 
- The six rotated axes are signed by marker variables. For tier 1, positive
  means physically intensive, judgement-intensive and person-facing
  respectively. For tier 2 the markers are medicine and dentistry (R4), the
  importance of being exact or accurate (R5) and sales and marketing (R6);
  their signs carry no claim about substitutability. The
  first is a level of physical capability demanded, not a contrast between
  embodied and symbolic work: its negative pole is low physical demand, which in
  this data covers both verbal work and light service work. Whether the positive
  pole is also the pole less amenable to substitution is a conjecture and is
  treated as one.
- The sign anchors, `varimax`, `congruence`, the tier rotation and the
  subspace helpers are defined once, in `pca_rotated.py`; the other scripts
  import them rather than carrying their own copies.
- Seeds are fixed (`SEED = 0`) and axis signs are anchored to marker variables,
  so results reproduce across runs and stay stable if the dataset is rebuilt.
- Wage top-coding: OEWS publishes no annual wage at or above $239,200 and marks
  it `#`. `build_master.py` reads the mark as 239200, so a value of exactly
  239200 always means "at least this much".
- Each analysis script names the columns it uses by their block prefix, e.g.
  `skills_im__` or `workctx_cx__`; there is no shared column module.
 