# AI and employment

This repository holds the data preparation, the analyses and the publishing
set-up behind a working paper on the dimensions of occupational space, built
from O\*NET descriptors of US occupations and joined to labour-market data.

**[The paper, with interactive figures](https://alicebobcharlie.github.io/AI_and_employment/)** ·
**[PDF of the latest release](https://github.com/AliceBobCharlie/AI_and_employment/releases/latest/download/paper.pdf)**

## Layout

```
data_raw/          raw downloads (not committed; see "Data" below)
data_prep/         build, clean and check the tables every analysis reads
analysis/          the analyses; analysis/supplementary/ holds side experiments
output/            every table, figure and intermediate file the scripts write
assets/            images used by the paper
paper_draft.qmd    the paper
figures.qmd        a web page with the paper's figures at full size
publish/           the web page and PDF build (Quarto)
.github/workflows/ automatic publishing of the web page and the release PDF
```

Each of `data_prep/`, `analysis/` and `publish/` has a README describing what
its files do and how they depend on each other. Each script's docstring
describes how it works.

## Data

Nothing in `data_raw/` is committed. All of it is publicly available; place
the files at these paths:

| source | path in `data_raw/` | used by |
|---|---|---|
| [O\*NET 31.0 database](https://www.onetcenter.org/database.html), text format | `onet/db_31_0_text/` | data_prep, analysis |
| [OEWS national, May 2024](https://www.bls.gov/oes/) | `national_M2024_dl.xlsx` | data_prep |
| [BLS Employment Projections](https://www.bls.gov/emp/tables.htm), tables 1.2 and 1.10 | `bls_projections.xlsx` | data_prep |
| BLS National Employment Matrix, CPS-to-SOC crosswalk | `nem-occcode-cps-crosswalk.xlsx` | data_prep |
| [Union coverage by occupation, 2024](https://www.unionstats.com/) (Hirsch, Macpherson and Even) | `occ_2024.xlsx` | data_prep |
| Occupational prestige ratings (Smith and Son, GSS 2012) | `OccupationalPrestigeRatings.csv` | data_prep |
| [IPUMS USA](https://usa.ipums.org/) ACS extract (optional; not redistributable) | `usa_00001.csv` | analysis/acs_young_share.py |
| Eloundou et al. (2024) occupation-level exposure (optional) | `eloundou_occ_level.csv` | analysis/labour_market_regression.py |

`data_prep/README.md` says which columns are read from each.

## Reproducing

Python 3.11 or later, with the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

Run every script from the repository root, in three stages:

1. **Data preparation**: `data_prep/README.md`. Builds and checks the four
   tables in `output/` that the analyses read.
2. **Analyses**: `analysis/README.md`. Writes its results to `output/` as CSV
   tables and PNG figures; the console reports run status only.
3. **Paper**: `publish/README.md`. Renders the web page and the PDF.

Seeds are fixed, so every run gives the same results.

## Committed outputs

Three files in `output/` are committed, because the web page and the PDF
draw their figures from them and no Python runs when the paper is built:
`occupation_coordinates.csv`, `axis_loadings.csv` and `acs_young_share.csv`.
After re-running the scripts that write them, commit them again. Everything
else in `output/` is rebuilt by the scripts.

## Licence

The code is released under the MIT licence; the text and figures of the
paper under CC BY 4.0. Data from O\*NET are used under CC BY 4.0
(USDOL/ETA); the other sources keep their own terms.