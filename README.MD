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
python build_master.py          # 1. assemble the raw table          -> master.csv
python clean_master.py          # 2. drop, fill and derive           -> master_clean.csv

python cluster_onet.py          # 3. are there clusters at all?       (Section 5.2)
python pca_rotated.py           # 4. dimensions and rotated axes      (Sections 5.3-5.4)
python validate_axes.py         # 5. do the axis names hold up?       (needs 3 and 4)
python check_orthogonality.py   # 6. what the features can predict   (Section 5.5)
python plot_axes.py             # 7. data for the figures             (needs 4)
python corner_table.py          # 8. Table 6.1                        (needs 4)
python acs_young_share.py       # 9. data for Figure 6.2              (needs 4 and the ACS extract)
```

Steps 1 and 2 run in that order; every analysis afterwards reads
`output/master_clean.csv`, and `check_orthogonality.py` also reads
`output/master.csv`. `validate_axes.py` needs the outputs of both
`cluster_onet.py` and `pca_rotated.py`.

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
that step removes. Nothing here uses the axes; the comparisons that do are in
`validate_axes.py`.

**`pca_rotated.py`** → `rotated_axes.csv`, `rotated_loadings.csv`,
`oblique_pattern.csv`, `rotated_scree.png`

The main analysis, on the 216 feature columns only. Eigenvalues, parallel
analysis, and bootstrap and split-half congruence, which decide how many
components to keep; varimax rotation makes those components interpretable; a
sign convention anchored on marker variables fixes each axis, and keeps the
signs stable if the data is rebuilt. Then the labour-market variables, which
took no part in forming the axes, are correlated against them, and a promax
rotation checks whether the axes are orthogonal by preference of the data or
only by construction.

**`validate_axes.py`** → terminal

Whether the three axes are what they are called. The names came from reading
loadings, which is a judgement, and every substantive claim rests on them.
Checks, each using something the analysis did not see: O*NET's own expert
grouping of elements (this needs the O*NET text files in `data_raw/`); whether
the off-diagonal cells of the R1 × R2 plane are populated, with the median wage
in each; the occupations at each extreme; whether adding the labour-market
variables to the decomposition changes the axes; the facts the paper states
about the poles of R1 and R2; and the parts of Section 5.2 that need the axes:
how the two-cluster split of occupations and the two skill clusters line up
with them, and the SOC major groups in the space of the leading components.

**`check_orthogonality.py`** → `orthogonality.xlsx`

How much of each labour-market variable the description of the work can
predict, by ridge regression with cross-validated R², from all 216 feature
columns and from each feature block alone (Table 5.5); the education, training
and experience block column by column; and the descriptive figures Section 5.5
quotes. Values that `clean_master.py` filled in are left out, so every R² is
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
`python supplementary/pca_ete.py`.

**`supplementary/pca_ete.py`** → `ete_axes.csv`

Whether the education, training and experience block has axes of its own. Same
procedure — stability, then rotation — followed by correlating the resulting
axes with the rotated axes of `pca_rotated.py`, to see how far the preparation
a job demands is separable from what the work requires. Run after
`pca_rotated.py`.

## Data-quality checks

**`check_external.py`** → `qc_external.xlsx` — missingness of the labour-market
variables as an occupation × variable matrix, plus their distributions, on the
raw values of the 910 occupations kept. Established that gaps in the external
variables are structural rather than scattered — whole rows absent from a
source, or wages above the OEWS cap — and which variables are skewed enough to
need a log.

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
 
- The three rotated axes are signed by marker variables so that positive means
  physically intensive, judgement-intensive and person-facing respectively. The
  first is a level of physical capability demanded, not a contrast between
  embodied and symbolic work: its negative pole is low physical demand, which in
  this data covers both verbal work and light service work. Whether the positive
  pole is also the pole less amenable to substitution is a conjecture and is
  treated as one.
- `varimax` and `congruence` are defined once, in `pca_rotated.py`; the other
  scripts import them rather than carrying their own copies.
- Seeds are fixed (`SEED = 0`) and axis signs are anchored to marker variables,
  so results reproduce across runs and stay stable if the dataset is rebuilt.
- Wage top-coding: OEWS publishes no annual wage at or above $239,200 and marks
  it `#`. `build_master.py` reads the mark as 239200, so a value of exactly
  239200 always means "at least this much".
- Each analysis script names the columns it uses by their block prefix, e.g.
  `skills_im__` or `workctx_cx__`; there is no shared column module.
 