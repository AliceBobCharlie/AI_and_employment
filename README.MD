# AI and employment

This repository holds the data preparation, the analyses and the publishing
set-up behind a working paper on the dimensions of occupational space, built
from O\*NET descriptors of US occupations and joined to labour-market data.

**[The paper, with interactive figures](https://alicebobcharlie.github.io/AI_and_employment/)** ·
**[PDF of the latest release](https://github.com/AliceBobCharlie/AI_and_employment/releases/latest/download/paper.pdf)**

## Layout

```
data/raw/          raw downloads (not committed; see "Data" below)
data/prepared/     the tables data_prep/ builds, which every analysis reads (not committed)
data_prep/         build, clean and check the prepared tables
analysis/          the analyses; analysis/supplementary/ holds side experiments
output/            results, one folder per script
paper_draft.qmd    the paper
figures.qmd        a web page with the paper's figures at full size
publish/           the web page and PDF build (Quarto); publish/assets/ holds hand-made images
.github/workflows/ automatic publishing of the web page and the release PDF
```

Each of `data_prep/`, `analysis/` and `publish/` has a README describing what
its files do and how they depend on each other. Each script's docstring
describes how it works.

## Data

Nothing in `data/` is committed. The raw data are all publicly available;
place the files at these paths:

| source | path in `data/raw/` | used by |
|---|---|---|
| [O\*NET 31.0 database](https://www.onetcenter.org/database.html), text format | `onet/db_31_0_text/` | data_prep, analysis |
| [OEWS national, May 2024](https://www.bls.gov/oes/) | `national_M2024_dl.xlsx` | data_prep |
| [BLS Employment Projections](https://www.bls.gov/emp/tables.htm), tables 1.2 and 1.10 | `bls_projections.xlsx` | data_prep |
| BLS National Employment Matrix, CPS-to-SOC crosswalk | `nem-occcode-cps-crosswalk.xlsx` | data_prep |
| [Union coverage by occupation, 2024](https://www.unionstats.com/) (Hirsch, Macpherson and Even) | `occ_2024.xlsx` | data_prep |
| Occupational prestige ratings (Smith and Son, GSS 2012) | `OccupationalPrestigeRatings.csv` | data_prep |
| [IPUMS USA](https://usa.ipums.org/) ACS extract (optional; not redistributable) | `usa_00001.csv` | analysis/acs_young_share.py |
| Eloundou et al. (2024) occupation-level exposure (optional) | `eloundou_occ_level.csv` | analysis/labour_market_regression.py |

`data_prep/Readme.md` says which columns are read from each.

## Reproducing

Python 3.11 or later, with the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

Run every script from the repository root, in three stages:

1. **Data preparation**: `data_prep/Readme.md`. Builds the tables in
   `data/prepared/` that the analyses read, and checks them.
2. **Analyses**: `analysis/Readme.md`. Each script writes its results to its
   own folder in `output/`, as CSV tables and PNG figures; the console
   reports run status only.
3. **Paper**: `publish/Readme.md`. Renders the web page and the PDF.

Seeds are fixed, so every run gives the same results.

## Committed outputs

Four files in `output/` are committed, because the paper draws on them and
no Python runs when it is built:

- `output/pca_rotated/occupation_coordinates.csv` and
  `output/pca_rotated/axis_loadings.csv`
- `output/resampling_stability/subspace_stability.png`
- `output/acs_young_share/acs_young_share.csv`

After re-running the scripts that write them, commit them again. Everything
else in `output/` is rebuilt by the scripts.

## Licence

The code is released under the MIT licence; the text and figures of the
paper under CC BY 4.0. Data from O\*NET are used under CC BY 4.0
(USDOL/ETA); the other sources keep their own terms.
