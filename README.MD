# Occupational structure and exposure to automation

Occupation-level analysis of the occupational feature space: what separates
jobs from one another, and which of those differences are about skill and which
are about institutional position.
Built on O*NET 31.0 (910 detailed occupations) joined to wages, union coverage,
occupational prestige, separation rates and self-employment.

**[Paper draft, with interactive figures](https://alicebobcharlie.github.io/AI_and_employment/)** · **[PDF of the latest release](https://github.com/AliceBobCharlie/AI_and_employment/releases/latest/download/paper.pdf)** (source: `paper_draft.qmd`)

Everything reads and writes through `output/`. Run from the project root, the
folder containing `data_raw/`.

## Run order

```
python build_master.py          #  1. assemble the raw table          -> master.csv
python clean_master.py          #  2. drop, fill and derive           -> master_clean.csv
python check_data.py            #  3. data quality                    (Sections 4.2-4.3)

python cluster_onet.py          #  4. are there clusters at all?       (Section 5.2)
python pca_rotated.py           #  5. two tiers, six rotated axes     (Section 5.3, Figure 5.1)
python validate_axes.py         #  6. do the six names hold up?       (Sections 5.2, 5.4; needs 4 and 5)
python validate_labour_market.py #  7. the axes and the labour market  (Sections 5.6, 5.7; needs 5)
python content_model_halves.py  #  8. job and worker halves           (Section 5.4; needs 5)
python check_curvature.py       #  9. non-linear relations among the axes (Sections 5.3, 5.5)
python rating_error.py          # 10. rating error beyond the axes    (Section 5.5)
python check_locality.py        # 11. local structure beyond the axes (Section 5.5)
python check_orthogonality.py   # 12. what the features can predict   (Section 5.6)
python corner_table.py          # 13. Table 6.1                        (needs 5)
python acs_young_share.py       # 14. data for Figure 6.2              (needs 5 and the ACS extract)
```

Steps 1 and 2 run in that order; every analysis afterwards reads
`output/master_clean.csv`, and `check_data.py`, `check_orthogonality.py` and
`supplementary/input_variants.py` also read `output/master.csv`.
`check_data.py`, `validate_axes.py` and `rating_error.py` also read the O*NET
text files in `data_raw/onet/db_31_0_text` (`validate_axes.py` skips the checks
whose files are missing). `validate_axes.py` needs the outputs of both
`cluster_onet.py` and `pca_rotated.py`.

The scripts in `supplementary/` are run as needed rather than as part of the
sequence; see below.

## Two tables

| table | what it is |
|---|---|
| `output/master.csv` | The raw table: every O*NET-SOC code in O*NET 31.0, rated or not; every scale of the five feature blocks and of education, training and experience, as published; the labour-market variables joined on the six-digit SOC code. Nothing dropped, filled or derived. |
| `output/master_clean.csv` | Analysis-ready: the 910 occupations rated in all five feature blocks, the 220 feature columns, the 37 education, training and experience columns, and the labour-market variables, every gap filled and the derived columns added. All of this lives in `clean_master.py` and nowhere else. |
| | `master.csv` is committed; `master_clean.csv` is not, since `clean_master.py` rebuilds it from `master.csv` in seconds. |
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
scale for four blocks and, for Work Context, the Context scale (CX) and the
category percentages (CTP) of the two working-time items, Duration of Typical
Work Week and Work Schedules, with the standard category of each (40 hours,
regular) dropped. Merges the two
education scales (RL and RQ), drops the highest category of each education,
training and experience scale, and fills the remaining gaps: self-employment at
0 for occupations absent from the BLS table, wages and everything else at the
column median, as are the CTP cells of the nine occupations without them.
Adds the three wage ratios, `ext_employment_log`, `ext_wage_level_log` and
`ext_wage_disp_p90p10`. The Level scale and the Work Context category
percentages of the CX items (CXP) stay in the raw table only.

## Results

**`cluster_onet.py`** → `cluster_occupations.png`,
`cluster_skill_similarity.png`, `cluster_assignments.csv`, `cluster_skills.csv`

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
that step removes. Nothing here uses the rotated axes; the comparisons that
do are in `validate_axes.py`.

**`pca_rotated.py`** → `rotated_axes.csv`, `rotated_loadings.csv`,
`oblique_pattern.csv`, `pca_stability.xlsx`, `scree.png`,
`assets/subspace_stability.png` (Figure 5.1)

The main analysis, on the 220 feature columns only, in two tiers. Every
resampling experiment (bootstrap, split-halves, permutations) is repeated
`N_REPEAT` = 200 times. Tier 1:
eigenvalues and parallel analysis (reported, not used to choose), then
bootstrap and split-half congruence of each unrotated component, which finds
the components that reproduce one by one (the first three). Tier 2: the
eigenvalue gaps against their sampling error (North's ratio), which shows
components 4-6 to be nearly degenerate; split-half principal angles between
the leading-k subspaces, k = 1 to 14, against the same statistic on
column-permuted data (Figure 5.1); then each candidate block
after tier 1 judged as a whole, by the root mean square of the principal-angle
cosines between the two halves' block subspaces (p05 >= 0.90 over 200
split-halves). The data picks components 4-6; the rotated axes of each
candidate, the smaller blocks inside the candidates (component 4 alone,
components 5-6, ...) and every block of two or three components after tier 2
up to component 14 are reported as well, and the same blocks are judged again
by their medians to show that the boundary after tier 2 does not depend on
how strictly stability is defined. The congruence of each rotated tier-2 axis
with the unrotated components shows how far rotation moves them. Each tier is varimax-rotated on its own, so R1-R3 are the same axes
whatever tier 2 turns out to be, and each axis is signed by a marker variable.
The labour-market variables, which took no part in forming the axes, are
correlated against all six, and a promax rotation of tier 1 checks whether its
axes are orthogonal by preference of the data or only by construction.

**`validate_axes.py`** → `validate_axes.xlsx`

Whether R1-R6 measure what their names say, against descriptions of the same
occupations that took no part in forming the axes, all from the O*NET text
files: O*NET's expert grouping of the elements (mean loading per category and
the categories that form each pole), the RIASEC interest types with Prediger's
people-things and data-ideas dimensions, the 41 basic interest areas, the job
zones, the software each occupation uses and the work styles (Impact scale).
Also the parts of Section 5.2 that need the axes: how the two-cluster split of
occupations and the two skill clusters line up with them, and the SOC major
groups in the space of the leading components. Each check runs if its input is
present. Takes the O*NET folder as an optional argument; prints one line per
check, and every table with `-v`.

**`validate_labour_market.py`** → `validate_labour_market.xlsx`

The axes and the labour-market variables, which take no part in forming them:
whether adding them to the decomposition changes any of R1-R6; the median wage
in each quadrant of the R1 × R2 plane; and the correlation of each axis with the
labour-market variables, required education, the Work Context items on
decisions and error, and the Eloundou et al. exposure measures (if
`data_raw/eloundou_occ_level.csv` is present). Prints one line per check, and
every table with `-v`.

**`content_model_halves.py`** → `content_model_halves.xlsx`

Whether the job half of the Content Model (Work Activities, Work Context) and
the worker half (Skills, Abilities, Knowledge), which share no column, arrive
at the same axes on their own: each half's reproducing leading subspace, how
well it recovers each of R1-R6, canonical correlations between the halves, and
each half's own rotated axes against the other's.

**`check_curvature.py`** → `curvature_check.xlsx`

Whether the axes are related non-linearly. Part 0, within tier 1: whether
R1, R2 and R3 depend on one another beyond the linear, under varimax and under
promax (conditional means out of fold with permutation nulls, distance
correlation, changes in spread, the shape of the joint distribution), and
whether the pattern "physical work needs less judgement" belongs to physical
effort or to physical skill. Then the same question at both tier boundaries:
how much of what comes after is curvature of the components before it, and is
what is left still structure? Part 1, the boundary at 3: how much of R4-R6 R1-R3 predict non-linearly (out of
fold, with permutation nulls), which squares and products do it, and whether
tier 2 still reproduces once that is removed, each half of a split decomposed
and cleaned on its own (the residual of the 4-6 scores, as a subspace and axis
by axis; and the data matrix with PC1-3 and
their curvature removed, decomposed again, to see whether it still leads with a
stable three-dimensional block in the space of the original 4-6 and whether the
boundary after it holds); and, on the full sample, the square or product of
R1-R3 most correlated with each of R4-R6, and R4-R6 rotated again after the
removal, with their congruence with the originals and their share of the
variance before and after. Part 2, the boundary at 6: how much of PC7-14 PC1-6
predict non-linearly, with refits inside each fold and synthetic data with a
known floor as a positive control, the spectrum after that curvature is
removed, and which bends carry it. Part 3: normal scores against the floor of
the rating scales.

**`rating_error.py`** → `rating_error.xlsx`

How much of each item, and of each principal component, is sampling error in
O*NET's ratings, from the published standard errors; an error-only spectrum as
a measured null for parallel analysis; where components 7-14 load by block
(candidate method factors: errors shared within one questionnaire); and whether
the items the six axes explain poorly are simply the unreliable ones.

**`check_locality.py`** → `check_locality.xlsx`

Whether what lies beyond the six axes is local: features that separate a few
occupations sharply rather than ordering all of them. k-means is poor at
finding a small group next to a large one, so kurtosis is used instead. Part A
finds the most and least kurtotic direction in each subspace (PC1-3, PC4-6,
PC7-14, PC4-14, PC1-14) by projection pursuit, against the same search on
Gaussian data, with an outlier check, the occupation counts along the tail and
the occupations in it. Part B goes through PC7-14 one direction at a time,
curvature of the six removed, to find how many of its directions are
heavy-tailed and how much of its variance they carry, and repeats this on
rank-normal scores so that skewed single scales cannot produce the tails.
Part C asks whether some structure is specific to one region of the space, by
comparing regional and global directions on held-out occupations.

**`check_orthogonality.py`** → `orthogonality.xlsx`

How much of each labour-market variable the description of the work can
predict, by ridge regression with cross-validated R², from all 220 feature
columns and from each feature block alone (Section 5.6); the education,
training and experience block column by column; and the descriptive figures
Section 5.6 quotes. Values that `clean_master.py` filled in are left out, so every R² is
computed on published values only. Unlike anything read off a PCA, the answer
does not depend on how many feature columns happen to be in the matrix.

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

Whether the education, training and experience block has axes of its own. Its
first two components are rotated and correlated with the rotated axes of
`pca_rotated.py`, to see how far the preparation a job demands is separable
from what the work requires. Descriptive only: the block is too small for a
resampling test to mean much. Run after `pca_rotated.py`.

## Data-quality checks

**`check_data.py`** → `data_quality.xlsx`

Every fact Sections 4.2 and 4.3 state about the quality of the data. Which
O*NET-SOC codes are rated, and what the unrated ones are; Recommend Suppress
and Not Relevant flags per block and scale; the share of ratings with a
standard error; how closely Importance and Level agree, element by element and
as whole matrices; what the Work Context category percentages (CXP) hold
beyond their mean (CX); the missing cells of the final matrix as published
(Table 4.1); and the missingness and
distributions of the labour-market variables, which established that their
gaps are structural rather than scattered and which variables need a log. The
flags and standard errors are not in the master table, so this script reads
them from the O*NET text files.

A further one-off check established that BLS occupational openings is a size
proxy (r = 0.89 with employment), that the "typical education needed" threshold
is redundant with the ETE distributions (R² = 0.82), and that "work experience
needed" is mostly missing. All three were excluded on that basis.

## The web paper and the PDF

The paper is written in `paper_draft.qmd` and rendered with
[Quarto](https://quarto.org) in two ways. The web page, with interactive
figures, is rebuilt on every push and always shows the current text and data.
The PDF is built only for a release: each [release](https://github.com/AliceBobCharlie/AI_and_employment/releases) is one version of
the paper, with its PDF attached, which does not change afterwards. The two may
therefore differ: the web page is the working version, the PDF the released
one.

| file | role |
|---|---|
| `paper_draft.qmd` | the paper: title and abstract in the header, then the text |
| `_quarto.yml` | the web page: layout, fonts and the scripts it loads |
| `theme.scss` | typography and colours of the web page |
| `figures.js` | draws the interactive figures in the reader's browser with plotly.js, from `output/rotated_axes.csv`, `output/rotated_loadings.csv` and `output/acs_young_share.csv` |
| `_quarto-pdf.yml` | the PDF: page size, margins, type size, and the two steps below |
| `pdf/print_figures.ts` | before the PDF is rendered: opens each interactive figure in headless Chrome, drawn by `figures.js` as the web page first shows it, and prints it to `output/figures/<id>.pdf` |
| `pdf/figures.lua` | while the PDF is rendered: puts each figure of the paper in it, with the caption text that applies to print |
| `pdf/typst-template.typ` | the PDF's layout, that of the NeurIPS preprint style common on arXiv: US Letter, a 5.5 × 9 in text block, 10 pt New Computer Modern, booktabs tables |

The figures are HTML blocks in `paper_draft.qmd`, one source for both formats.
In a caption, text inside `<span class="web-only">` (how to rotate, hover,
choose axes) is left out of the PDF, and text inside
`<span class="print-only">` (which view the image shows) appears only in the
PDF. The figures of the PDF are drawn by the same `figures.js` as those of the
web page, so there is nothing to keep in step.

No Python runs when either is built. The web page reads the committed CSVs, so
after re-running the analysis, commit the new `rotated_axes.csv`,
`rotated_loadings.csv` and `acs_young_share.csv`.

To build locally, install Quarto and run, from the project root:

```
quarto render                                          # the web page -> _site/
quarto render paper_draft.qmd --profile pdf --to typst # the PDF -> _site/paper.pdf
```

The PDF also needs Chrome or Chromium; set `CHROME` to its path if it is not
in the usual place. `pdf/print_figures.ts` runs on the Deno that ships with
Quarto. To preview the web page, run `quarto preview`. Opening
`_site/index.html` straight from disk will not show the figures, because
browsers block a local file from loading the CSV.

Every push to `main` triggers `.github/workflows/publish.yml`, which renders
the web page and deploys it to GitHub Pages; only the latest deployment is
kept, and the rendered site is never committed. This requires Settings → Pages
→ Source to be set to *GitHub Actions*. Publishing a release triggers
`.github/workflows/release.yml`, which renders the PDF from the release's tag
and attaches it to the release as `paper.pdf`. To release a version:

```
gh release create v1.0 --title "v1.0" --notes "What changed."
```

The web page links the PDF of the latest release from its table of contents.

## Conventions

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