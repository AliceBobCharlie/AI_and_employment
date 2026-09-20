# Occupational structure and exposure to automation

Occupation-level analysis of the occupational feature space: what separates
jobs from one another, and which of those differences are about skill and which
are about institutional position.
Built on O*NET 30.2 (894 detailed occupations) joined to wages, union coverage,
occupational prestige, separation rates and self-employment.

**[Paper draft](paper_draft.md)** · **[Interactive figures](https://alicebobcharlie.github.io/AI_and_employment/)**

Everything reads and writes through `output/`. Run from the project root, the
folder containing `data_raw/`.

## Run order

```
python build_master.py          # 1. assemble the raw dataset -> master_wide.xlsx
python clean_master.py          # 2. impute + transform       -> master_clean.xlsx

python cluster_onet.py          # 3. are there clusters at all?
python pca_rotated.py           # 4. main result: three rotated axes
python pca_skills.py            # 5. skill-only axes (needed by pca_ete.py)
python pca_ete.py               # 6. structure of the education/training block
python check_orthogonality.py   # 7. which external variables skills can predict
python validate_axes.py         # 8. do the axis names hold up?
python plot_axes.py             # 9. figures
```

Steps 1 and 2 run in that order; every analysis afterwards reads
`output/master_clean.xlsx`. `pca_ete.py` needs `output/skills_axes.csv`, so
`pca_skills.py` comes first. Nothing else has dependencies.

`scan_quality.py`, `check_external.py`, `mfa_analysis.py` and `pca_mixed.py`
are run as needed rather than as part of the sequence; see below.

## Three layers of data

| layer | what it is |
|---|---|
| `output/master_wide.xlsx` | The raw published values, gaps and all. Nothing imputed or filtered beyond dropping occupations that are >90% empty. This is what the quality checks inspect. |
| `output/master_clean.xlsx` | Analysis-ready: every gap filled, wage top-coding handled, derived columns added. All imputation lives in `clean_master.py` and nowhere else. |
| standardization | Deliberately **not** stored. Each analysis z-scores its own matrix, because `check_orthogonality.py` must fit its scaler on training folds only — pre-standardizing would leak test-fold information into the cross-validated R². |

## Building the data

**`build_master.py`** → `master_wide.xlsx`

Assembles the O*NET matrix and joins the external sources in one pass. The
O*NET side is pruned on evidence, not taste: Skills, Abilities, Knowledge and
Work Activities enter at Importance only (Importance and Level correlate at
r ≈ 0.94 per element and Level is recoverable from Importance at R² ≈ 0.90, so
keeping both doubles the column count without adding information); Work Context
keeps the CX scale; the education/training/experience distributions keep every
category except each scale's highest, which is implied by the rest. Work Styles
and Interests are dropped as entirely model-generated, Work Values as stale and
largely redundant, Job Zones as a coarse restatement of the main skill axis
(r = 0.85 with the first component).

**`clean_master.py`** → `master_clean.xlsx`

The single place imputation happens. Wage gaps are filled according to what
caused them: a high percentile missing while p10 is present means OEWS
top-coded it, so the value goes to the $239,200 cap — filling at the median
would place a surgeon below their own 10th percentile. An occupation missing
every wage percentile is absent from OEWS entirely and has no anchor, so it
takes the column median. Wage ratios are then recomputed from the filled levels
so they stay consistent with them. Adds `ext_employment_log`,
`ext_wage_level_log` and `ext_wage_disp_p90p10`. Remaining gaps take the column
median.

## Results

**`cluster_onet.py`** → `cluster_occupations.png`,
`cluster_skill_similarity.png`, `cluster_assignments.csv`

Whether the polarization reported in earlier work is a property of skills or of
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
that step removes.

**`pca_rotated.py`** → `rotated_axes.csv`, `rotated_loadings.csv`, `rotated_scree.png`

The main analysis. Wages enter as two columns, a log level and a p90/p10
dispersion, so that nine collinear wage measures cannot capture an axis by
sheer count. Union coverage, self-employment and separation rates take part.
Bootstrap and split-half congruence decide how many components to keep;
varimax rotation makes those components interpretable; a sign convention
anchored on marker variables fixes each axis so that **+ always means harder to
automate**, and keeps the signs stable if the data is rebuilt.

**`check_orthogonality.py`** → terminal

How much of each external variable O*NET can predict, by ridge regression with
honest cross-validated R², reported for all features, for the leading
components, per block, and incrementally. This is the test that licenses
statements about which external variables are separate dimensions rather than
restatements of skill. Unlike anything read off a PCA, the answer does not
depend on how many skill columns happen to be in the matrix — which is why it,
not the loadings, carries the argument.

**`pca_skills.py`** → `skills_axes.csv`, `skills_scree.png`

Axes from the skill columns alone, with external variables projected on
afterwards as supplementary variables taking no part in forming them. Where
`check_orthogonality.py` gives the magnitude — how much of a variable skills
account for in total — this gives the direction: which axis a variable points
along, and therefore what it means substantively. Complementary tests, not
duplicate ones; a variable can correlate weakly with every single axis and still
be well predicted by a combination of them.

**`validate_axes.py`** → terminal

Whether the three axes are what they are called. The names came from reading
loadings, which is a judgement, and every substantive claim rests on them — the
data would be unchanged if a name were wrong, the argument would not survive.
Three checks, each using something the analysis did not see: O*NET's own expert
grouping of elements into cognitive, psychomotor, physical and sensory abilities
and into information input, mental processes, work output and interaction;
whether the off-diagonal cells of the R1 × R2 plane are populated, since the
claim that judgement and physical intensity are separate requires occupations
that are physically demanding and judgement-intensive, and physically light and
procedure-following, to exist; and the occupations at each extreme. The quadrant
medians also show directly whether pay follows judgement or the manual/mental
divide.

**`pca_ete.py`** → `ete_axes.csv`

Whether the education, training and experience block has axes of its own.
Same procedure — stability, then rotation — followed by correlating the
resulting axes against the skill axes to see how far the preparation a job
demands is separable from the ability it demands.

**`plot_axes.py`** → `fig_3d_space.html`, `fig_panels.html`

Interactive figures: the three axes in 3-D, drawn as free coordinate axes and
coloured by union coverage, and the six pairwise 2-D sections on one page, any
of which enlarges on click. Marker size is log employment; hovering names the
occupation.
Locating other measures in the space
A coordinate system earns its keep if measures built independently of it can be placed inside it. These two scripts do that, and they are also where the negative results live.
project_measures.py → measure_projection.csv
Projects published substitution measures onto the three axes: Eloundou et al. (2024), under both their human and GPT-4 annotations; the OECD AI Capability Gap Index, both its total and its nine capability domains, with the sign reversed so that higher always means more substitutable; and, where the files are present, the Anthropic Economic Index. The domain-level projection is the sharper test of the OECD measure, since summing nine correlated domains is what cancels one of the axes. The AEI results carry a caveat the output makes explicit: the median occupation matches a single task, so its automation ratio is reported only where at least ten tasks match.
acs_young_share.py → acs_young_share.csv
Builds occupation-level young-worker shares from an IPUMS ACS extract — employed, wage and salary, 35 hours or more, non-military, aged 22 to 25 — and projects the 2022–2024 change onto the axes. The build step is skipped when the aggregate already exists. The result is a null: about 60 percent of the cross-occupational variance in the change is real rather than sampling noise, but correlations with all three axes are below 0.2 and change sign between specifications. It is retained because the null is reported in the paper.

## Supporting analyses

**`mfa_analysis.py`** → `mfa_row_coords.csv`

Multiple factor analysis over three weighted groups — skills, education and
training, economic-institutional — so that group size cannot decide influence.
An alternative to compressing the wage columns: instead of reducing a group, it
rescales each group by its first singular value. Prestige enters as a
supplementary variable only, being collinear with skills.

**`pca_mixed.py`** → `mixed_scree.png`

Equal-weight PCA over skills and all external variables together, kept as the
baseline the other two designs correct. It shows what happens when a collinear
block of nine wage columns and a set of low-variance institutional variables are
given equal footing by default, and is the reason the main analysis compresses
wages and the MFA weights groups. Retained as documentation of a design choice;
it produces no result the paper relies on.

## Data-quality checks

Each was run once. Their findings are what justify the variables that were
dropped, so they belong in the methods section even though they are not part of
the analysis sequence.

**`scan_quality.py`** → `qc_onet_quality.xlsx` — per-block scan of O*NET's own
reliability flags and data sources. Established that Work Styles and Interests
are entirely model-generated, and that Knowledge's "not relevant" flags sit
almost entirely on the Level rows, so dropping Level removes that problem too.

**`check_external.py`** → `qc_external.xlsx` — missingness as an occupation ×
variable matrix, plus distributions. Established that gaps in the external
variables are structural rather than scattered — whole rows absent from a
source, or wages suppressed above the top-code — which is what the two-way fill
rule in `clean_master.py` responds to, and which variables are skewed enough to
need a log.

A further one-off check established that BLS occupational openings is a size
proxy (r = 0.89 with employment), that the "typical education needed" threshold
is redundant with the ETE distributions (R² = 0.82), and that "work experience
needed" is 85% missing. All three were excluded on that basis.

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
- Wage top-coding: OEWS suppresses annual wages above $239,200. Where a high
  percentile is missing but a low one is present the value is filled at that
  cap, because the true value is known to lie above it.
 