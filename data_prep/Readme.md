# data_prep

These scripts turn the raw downloads in `data/raw/` into the four tables in
`data/prepared/` that every analysis reads, and check their quality. They
make no analytic choices beyond which rows and columns to keep and how to
fill gaps.

## Order

```
python data_prep/build_master.py   # data/raw/onet/            -> master.csv
python data_prep/clean_master.py   # master.csv                -> master_clean.csv
python data_prep/check_data.py     # master, master_clean, O*NET text files -> output/check_data/
python data_prep/build_lmv.py      # O*NET + external sources  -> labour.csv
python data_prep/clean_lmv.py      # labour.csv, master_clean  -> labour_clean.csv
python data_prep/check_lmv.py      # labour, labour_clean, master_clean
                                   #   -> output/check_lmv/labour_data_quality.xlsx
```

The four tables are written to and read from `data/prepared/`; the checks
write to `output/`.

`clean_lmv.py` takes its rows from `master_clean.csv`, so it runs after
`clean_master.py`. `build_lmv.py` imports the O\*NET reading functions of
`build_master.py`.

## The four tables

| table | rows | contents |
|---|---|---|
| `master.csv` | every O\*NET-SOC code in Occupation Data | every scale of the five feature blocks, as published |
| `master_clean.csv` | the occupations rated in all five feature blocks | every scale of `master.csv` except one redundant category of each percentage item; gaps filled |
| `labour.csv` | the rows of `master.csv` | the education, training and experience distributions and the external variables, as published |
| `labour_clean.csv` | the rows of `master_clean.csv` | the same, merged, filled and with derived columns added |

None is committed. No table stores standardised values; each analysis
z-scores its own matrix.

Column names follow `<block>_<scale>__<element>`, with `__c<category>` for
category scales, e.g. `abilities_im__Oral Comprehension` or
`workctx_cxp__Public Speaking__c3`. The analyses select columns by these
prefixes.

## What each script does

- **build_master.py** reads the five O\*NET feature blocks (Abilities,
  Skills, Knowledge, Work Activities, Work Context) and joins them. It
  drops, fills and derives nothing.
- **clean_master.py** keeps the occupations rated in all five blocks, drops
  the redundant category of each percentage item, and fills the remaining
  gaps with column medians.
- **check_data.py** checks the feature data: which codes are rated, the
  Recommend Suppress and Not Relevant flags, the coverage of standard errors,
  the rating error of each item, Importance against Level, how far the Work
  Context category shares reduce to their mean and end categories, and the
  missing cells of the feature matrix. It reads the O\*NET text files for the
  flags and standard errors, which the tables do not carry.
- **build_lmv.py** reads O\*NET's education, training and experience
  distributions and the external sources, and joins them on the six-digit
  SOC code. It drops, fills and derives nothing, except for reading source
  codes: an OEWS wage marked `#` is read as the cap, 239,200, and a blank or
  dash in BLS table 1.2 as 0.
- **clean_lmv.py** keeps the rows of `master_clean.csv`, merges the two
  education scales, drops the redundant category of each education, training
  and experience scale, fills gaps, and adds the wage ratios and log columns.
- **check_lmv.py** checks the labour-market data on raw values (missing
  values, distributions, skew), and how far the education, training and
  experience distributions and the wage quantiles reduce to a few variables.

## Source columns

| source | read by | columns |
|---|---|---|
| O\*NET: Abilities, Essential Skills, Transferable Skills, Knowledge, Work Activities, Work Context | build_master, check_data | every scale and category |
| O\*NET: Education, Training and Experience | build_lmv | every category of RL, RQ, RW, PT, OJ |
| OEWS May 2024 | build_lmv | employment; mean, median, 10th, 25th, 75th and 90th percentile of annual pay |
| BLS Employment Projections | build_lmv | table 1.10: exit and transfer rates; table 1.2: self-employed share |
| National Employment Matrix crosswalk | build_lmv | CPS occupation code to SOC |
| Union coverage | build_lmv | coverage rate by CPS occupation |
| Prestige ratings | build_lmv | prestige score, averaged within a six-digit SOC code |
