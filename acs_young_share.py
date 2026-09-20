#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
acs_young_share.py -- build occupation-level young-worker shares from ACS
microdata, then ask where in the three-axis space that share moved.

Brynjolfsson, Chandar and Chen (2025) find that the employment effects of
generative AI appear first in reduced hiring of young workers rather than in
layoffs of incumbents. A falling share of young workers within an occupation is
therefore a candidate signal of where AI is displacing entry-level work.

The script does two jobs. Given an IPUMS ACS extract it aggregates the microdata
to occupation-years; given the aggregate it projects the 2022-2024 change onto
the axes. The build step is skipped when the aggregate already exists, since the
IPUMS extract is large and cannot be redistributed.

It does NOT establish that any change is caused by AI. An annual occupational
share cannot separate AI from the business cycle, interest-rate-sensitive
hiring, cohort size, immigration or post-pandemic reallocation, and it carries
none of the firm-level controls the payroll-microdata evidence relies on. What
it can answer is the descriptive question: which region of the space did
young-worker employment contract in?

Sample restrictions, applied in the build and inherited by everything after:
employed, working for wages (so the self-employed are excluded, since the
displacement mechanism differs), usually 35 hours or more, non-military, and
aged 22 to 25 for the young group. The band follows Brynjolfsson et al. rather
than the age at which entry occurs in every occupation: a wider band would cover
late entry into medicine and law, but those occupations are protected by
licensing and would contribute little movement, while the wider band dilutes the
entry-level signal everywhere else.

Crosswalk. ACS occupation codes are IPUMS OCCSOC: six characters without a
hyphen, and some are aggregates carrying X as a wildcard (1110XX covers every
SOC beginning 11-10). Those cannot identify a single O*NET occupation, so their
value is broadcast to every code they cover and the match is flagged. Broadcast
rows share one value across several occupations, which manufactures correlation
structure, so results are reported on exact matches and on all matches.

Inputs:  usa_00001.csv (IPUMS extract, optional if the aggregate exists),
         output/rotated_axes.csv, output/master_clean.xlsx
Outputs: young_share.csv, output/acs_young_share.csv
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

IPUMS_CSV = Path("data_raw/usa_00001.csv")       # IPUMS ACS extract, not redistributable
ACS = Path("data_raw/young_share.csv")           # the aggregate this script builds
AXES = Path("output/rotated_axes.csv")
MASTER = Path("output/master_clean.xlsx")
OUT = Path("output/acs_young_share.csv")

Y0, Y1 = 2022, 2024
YOUNG_MIN, YOUNG_MAX = 22, 25           # age band counted as "young",
                                        # matching Brynjolfsson et al. (2025)
MIN_CELL_N = 100                        # occupation-years below this are flagged
MIN_TOTAL_N = 500                       # minimum TOTAL sample in BOTH years.
                                        # The filter is on the total rather than
                                        # on the young count because the standard
                                        # error of a share depends on the total,
                                        # and because filtering on the young count
                                        # would be selecting on the numerator of
                                        # the outcome: it excludes occupations
                                        # whose young share is already low.

AXIS_NAMES = {"R1": "physical intensity", "R2": "judgement",
              "R3": "person-facing"}

COLUMNS = ["YEAR", "PERWT", "AGE", "EMPSTAT", "CLASSWKR", "UHRSWORK", "OCCSOC"]

# EMPSTAT 1 employed; CLASSWKR 2 wage and salary; UHRSWORK >= 35 full time.
# OCCSOC beginning 55 is military; '0' and '' are not applicable.
FILTERS = """
    EMPSTAT = 1
    AND CLASSWKR = 2
    AND UHRSWORK >= 35
    AND OCCSOC IS NOT NULL
    AND OCCSOC NOT IN ('0', '')
    AND OCCSOC NOT LIKE '55%'
"""


def build_from_ipums():
    """Aggregate the IPUMS extract to occupation-years. Run only when the
    aggregate is absent; needs duckdb and the extract itself."""
    import duckdb
    con = duckdb.connect()
    parquet = "acs.parquet"
    con.execute(f"""
        COPY (SELECT {", ".join(COLUMNS)}
              FROM read_csv('{IPUMS_CSV}', auto_detect = true,
                            types = {{'OCCSOC': 'VARCHAR'}}))
        TO '{parquet}' (FORMAT PARQUET)""")
    df = con.execute(f"""
        SELECT YEAR, OCCSOC,
               SUM(PERWT) FILTER (WHERE AGE BETWEEN {YOUNG_MIN} AND {YOUNG_MAX}) AS young_wt,
               SUM(PERWT)                                                        AS total_wt,
               COUNT(*)   FILTER (WHERE AGE BETWEEN {YOUNG_MIN} AND {YOUNG_MAX}) AS young_n,
               COUNT(*)                                                          AS total_n
        FROM '{parquet}'
        WHERE {FILTERS}
        GROUP BY YEAR, OCCSOC
        ORDER BY YEAR, OCCSOC""").df()

    df["young_wt"] = df["young_wt"].fillna(0)
    df["young_share"] = df["young_wt"] / df["total_wt"]
    # the unweighted share shows how much of any change comes from revisions to
    # the survey weights rather than from the population
    df["young_share_unwt"] = df["young_n"] / df["total_n"]
    df["small_cell"] = df["total_n"] < MIN_CELL_N
    df.to_csv(ACS, index=False)
    print(f"[build] {len(df)} occupation-years, "
          f"{df.OCCSOC.nunique()} occupations, "
          f"{int(df.small_cell.sum())} small cells")
    return df


def acs_to_soc_pattern(code):
    """IPUMS OCCSOC -> a regex matching the 6-digit SOC codes it covers.
    '111021' -> '^11-1021$'      (exact)
    '1110XX' -> '^11-10..$'      (aggregate: X is a wildcard)
    """
    c = str(code).strip().upper()
    if len(c) != 6:
        return None, None
    body = c[:2] + "-" + c[2:]
    exact = "X" not in c
    return "^" + body.replace("X", ".") + "$", exact


def main():
    # ---------------- build, or reuse ----------------
    if not ACS.exists():
        if not IPUMS_CSV.exists():
            print(f"neither {ACS} nor {IPUMS_CSV} found -- nothing to do")
            return
        build_from_ipums()
    else:
        print(f"[build] {ACS} exists; skipping the microdata step")

    # ---------------- ACS side ----------------
    d = pd.read_csv(ACS, dtype={"OCCSOC": str})
    d = d[d.YEAR.isin([Y0, Y1])]

    piv = d.pivot_table(index="OCCSOC", columns="YEAR",
                        values=["young_share", "young_n", "total_n", "total_wt"],
                        aggfunc="first")
    ok = (piv[("total_n", Y0)] >= MIN_TOTAL_N) & (piv[("total_n", Y1)] >= MIN_TOTAL_N)
    print(f"ACS codes with both years: {len(piv)}; "
          f"passing total_n>={MIN_TOTAL_N} in both: {int(ok.sum())}")

    acs = pd.DataFrame({
        "delta": piv[("young_share", Y1)] - piv[("young_share", Y0)],
        "share_2022": piv[("young_share", Y0)],
        "share_2024": piv[("young_share", Y1)],
        "acs_weight": piv[("total_wt", Y1)],
    })[ok]
    # how much of the spread in delta is real rather than sampling noise
    s0, s1 = piv[("young_share", Y0)][ok], piv[("young_share", Y1)][ok]
    n0, n1 = piv[("total_n", Y0)][ok], piv[("total_n", Y1)][ok]
    se = np.sqrt(s0 * (1 - s0) / n0 + s1 * (1 - s1) / n1)
    snr = max(0.0, (acs.delta.var() - (se ** 2).mean()) / acs.delta.var())
    print(f"delta: mean {acs.delta.mean():+.4f}, sd {acs.delta.std():.4f}, "
          f"range {acs.delta.min():+.3f} to {acs.delta.max():+.3f}")
    print(f"       expected sd from sampling alone {np.sqrt((se**2).mean()):.4f}; "
          f"share of variance that is real {snr:.2f}")

    # ---------------- O*NET side ----------------
    ax = pd.read_csv(AXES, index_col=0)
    axis_cols = [c for c in ax.columns if re.fullmatch(r"R\d+", c)]
    ax["soc6"] = [re.match(r"(\d{2}-\d{4})", str(i)).group(1) for i in ax.index]

    # ---------------- match ----------------
    rows = []
    for code, r in acs.iterrows():
        pat, exact = acs_to_soc_pattern(code)
        if pat is None:
            continue
        hit = ax[ax.soc6.str.match(pat)]
        for onet_soc, a in hit.iterrows():
            rows.append({"onet_soc": onet_soc, "title": a["title"],
                         "acs_code": code, "exact_match": exact,
                         "delta": r.delta, "share_2022": r.share_2022,
                         "share_2024": r.share_2024,
                         **{c: a[c] for c in axis_cols}})
    M = pd.DataFrame(rows)
    if M.empty:
        print("no matches -- check the code formats"); return

    print(f"\nmatched {M.onet_soc.nunique()} of {len(ax)} O*NET occupations")
    print(f"  exact (1 ACS code -> 1 SOC) : {int(M.exact_match.sum())} rows")
    print(f"  broadcast (aggregate codes)  : {int((~M.exact_match).sum())} rows, "
          f"from {M[~M.exact_match].acs_code.nunique()} ACS aggregates")

    # employment weights, if available
    if MASTER.exists():
        m = pd.read_excel(MASTER).set_index("onet_soc")
        M["employment"] = m["ext_employment"].reindex(M.onet_soc).values

    # ---------------- projection ----------------
    for label, sub in [("exact matches only", M[M.exact_match]),
                       ("all matches", M)]:
        print("\n" + "=" * 62)
        print(f"CHANGE IN YOUNG-WORKER SHARE {Y0}->{Y1} vs THE AXES "
              f"({label}, n={len(sub)})")
        print("=" * 62)
        for c in axis_cols:
            r = sub[c].corr(sub["delta"])
            line = f"  corr(delta, {c} {AXIS_NAMES.get(c,''):<18}) = {r:+.3f}"
            if "employment" in sub.columns and sub["employment"].notna().any():
                w = sub["employment"].fillna(0).values
                x, y = sub[c].values, sub["delta"].values
                keep = ~np.isnan(x) & ~np.isnan(y) & (w > 0)
                if keep.sum() > 10:
                    xw = np.average(x[keep], weights=w[keep])
                    yw = np.average(y[keep], weights=w[keep])
                    cov = np.average((x[keep]-xw)*(y[keep]-yw), weights=w[keep])
                    rw = cov / np.sqrt(np.average((x[keep]-xw)**2, weights=w[keep])
                                       * np.average((y[keep]-yw)**2, weights=w[keep]))
                    line += f"   employment-weighted {rw:+.3f}"
            print(line)

    # ---------------- terciles of each axis ----------------
    sub = M[M.exact_match]
    print("\n" + "=" * 62)
    print("MEAN CHANGE BY AXIS TERCILE (exact matches)")
    print("=" * 62)
    for c in axis_cols:
        q = pd.qcut(sub[c], 3, labels=["low", "mid", "high"])
        g = sub.groupby(q, observed=True)["delta"].agg(["mean", "size"])
        cells = "   ".join(f"{k}: {v['mean']:+.4f} (n={int(v['size'])})"
                           for k, v in g.iterrows())
        print(f"  {c} {AXIS_NAMES.get(c,''):<18} {cells}")

    # ---------------- who moved most ----------------
    print("\n" + "=" * 62)
    print("LARGEST FALLS in young-worker share (exact matches)")
    print("=" * 62)
    for _, r in sub.nsmallest(12, "delta").iterrows():
        print(f"  {r['delta']:+.3f}  "
              f"R1{r.get('R1', float('nan')):+.1f} R2{r.get('R2', float('nan')):+.1f} "
              f"R3{r.get('R3', float('nan')):+.1f}  {r['title'][:46]}")
    print("\nLARGEST RISES")
    for _, r in sub.nlargest(12, "delta").iterrows():
        print(f"  {r['delta']:+.3f}  "
              f"R1{r.get('R1', float('nan')):+.1f} R2{r.get('R2', float('nan')):+.1f} "
              f"R3{r.get('R3', float('nan')):+.1f}  {r['title'][:46]}")

    M.to_csv(OUT, index=False)
    print(f"\n[ok] wrote {OUT}")
    print("\nReminder: a negative delta means the young-worker share fell. Any")
    print("association with an axis is a location, not a cause -- the ACS cannot")
    print("separate AI from the cycle, rates, cohort size or reallocation.")


if __name__ == "__main__":
    main()