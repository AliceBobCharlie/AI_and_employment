#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_master.py -- one script, one output: assemble the lean O*NET skill matrix
AND the social-structural variables into a single master table keyed by
O*NET-SOC, then drop near-empty occupations.

O*NET side (all observed, single-scale, redundancy-pruned):
  Skills / Abilities / Knowledge / Work Activities -> IM only
  Work Context                                     -> CX only
  Education/Training/Experience (ETE)              -> RL/RW/PT/OJ percent
      distributions, with the LAST (highest) category of each scale DROPPED to
      remove the compositional collinearity (categories sum to 100, so one is
      redundant; the dropped column is implied by the rest).
Dropped blocks: LV (redundant with IM), Work Values, Work Styles, Interests,
  Job Zones, Work Context CT.

External side (prefix ext_):
  OEWS wages/employment, CPS union coverage (via crosswalk), OPR prestige,
  BLS separation rates (labor-force exit, occupational transfer) + % self-
  employed.

No row filtering / no imputation on the O*NET values; only the >90%-empty
occupation drop at the end. Self-employment blanks -> 0 (negligible).
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

RAW = Path("data_raw")
ONET_DIR = RAW / "onet" / "O*NET_30_2_excel"
OUT = Path("output/master_wide.xlsx")
OUT.parent.mkdir(exist_ok=True)

CROSSWALK  = RAW / "nem-occcode-cps-crosswalk.xlsx"
OEWS_FILE  = RAW / "national_M2024_dl.xlsx"
UNION_FILE = RAW / "occ_2024.xlsx"
PRESTIGE_FILE = RAW / "OccupationalPrestigeRatings.csv"
BLS_FILE   = RAW / "bls_projections.xlsx"
OCCUPATION_FILE = "Occupation Data.xlsx"

DROP_THRESHOLD = 0.90

POINT_BLOCKS = [
    ("Skills", "IM", "skills"), ("Abilities", "IM", "abilities"),
    ("Knowledge", "IM", "knowledge"), ("Work Activities", "IM", "workact"),
    ("Work Context", "CX", "workctx"),
]
ETE_SCALES = ["RL", "RW", "PT", "OJ"]


def soc6(code):
    if pd.isna(code):
        return None
    m = re.match(r"(\d{2}-\d{4})", str(code))
    return m.group(1) if m else None


# ------------------------- O*NET wide ------------------------------------- #
def onet_wide():
    base = pd.read_excel(ONET_DIR / OCCUPATION_FILE).rename(columns={
        "O*NET-SOC Code": "onet_soc", "Title": "title"})
    wide = base[["onet_soc", "title"]].set_index("onet_soc")

    for stem, scale, prefix in POINT_BLOCKS:
        df = pd.read_excel(ONET_DIR / f"{stem}.xlsx").rename(columns={
            "O*NET-SOC Code": "soc", "Element Name": "element", "Data Value": "val"})
        df["val"] = pd.to_numeric(df["val"], errors="coerce")
        sub = df[df["Scale ID"] == scale]
        w = sub.pivot_table(index="soc", columns="element", values="val", aggfunc="mean")
        w.columns = [f"{prefix}_{scale.lower()}__{c}" for c in w.columns]
        wide = wide.join(w, how="left")
        print(f"[onet] {stem:24s} {scale}: {w.shape[1]} cols")

    # ETE percent distributions, dropping each scale's highest category
    ete = pd.read_excel(ONET_DIR / "Education, Training, and Experience.xlsx").rename(
        columns={"O*NET-SOC Code": "soc", "Element Name": "element",
                 "Data Value": "val", "Category": "cat"})
    ete["val"] = pd.to_numeric(ete["val"], errors="coerce")
    ete["cat"] = pd.to_numeric(ete["cat"], errors="coerce").astype("Int64")
    for scale in ETE_SCALES:
        sub = ete[ete["Scale ID"] == scale].copy()
        max_cat = int(sub["cat"].max())          # highest category -> drop as reference
        sub = sub[sub["cat"] != max_cat]
        sub["colkey"] = sub["element"].astype(str) + "__c" + sub["cat"].astype(str)
        w = sub.pivot_table(index="soc", columns="colkey", values="val", aggfunc="mean")
        w.columns = [f"ete_{scale.lower()}__{c}" for c in w.columns]
        wide = wide.join(w, how="left")
        print(f"[onet] ETE {scale}: {w.shape[1]} cols (dropped category c{max_cat})")

    wide = wide.reset_index()
    wide["soc6"] = wide["onet_soc"].map(soc6)
    print(f"[onet] wide: {wide.shape[0]} occ x "
          f"{len([c for c in wide.columns if '__' in c])} features")
    return wide


# ------------------------- external sources ------------------------------- #
def load_oews():
    df = pd.read_excel(OEWS_FILE, dtype=str)
    df = df[df["O_GROUP"].str.lower() == "detailed"].copy()
    df["soc6"] = df["OCC_CODE"].map(soc6)
    for c in ["TOT_EMP", "A_MEAN", "A_MEDIAN", "A_PCT10", "A_PCT25", "A_PCT75", "A_PCT90"]:
        df[c] = pd.to_numeric(df[c].replace({"*": None, "**": None, "#": None, "~": None}),
                              errors="coerce")
    df["ext_wage_p90p10"] = df["A_PCT90"] / df["A_PCT10"]
    df["ext_wage_p90p50"] = df["A_PCT90"] / df["A_MEDIAN"]
    df["ext_wage_p50p10"] = df["A_MEDIAN"] / df["A_PCT10"]
    keep = df[["soc6", "TOT_EMP", "A_MEAN", "A_MEDIAN", "A_PCT10", "A_PCT25",
               "A_PCT75", "A_PCT90", "ext_wage_p90p10", "ext_wage_p90p50",
               "ext_wage_p50p10"]].rename(columns={
        "TOT_EMP": "ext_employment", "A_MEAN": "ext_wage_mean",
        "A_MEDIAN": "ext_wage_median", "A_PCT10": "ext_wage_p10",
        "A_PCT25": "ext_wage_p25", "A_PCT75": "ext_wage_p75", "A_PCT90": "ext_wage_p90"})
    return keep.drop_duplicates("soc6")


def load_crosswalk():
    cw = pd.read_excel(CROSSWALK, dtype=str, skiprows=4).rename(columns={
        "National Employment Matrix code": "soc", "CPS code": "cps"})
    cw["soc6"] = cw["soc"].map(soc6)
    cw = cw[["soc6", "cps"]].dropna()
    cw["cps"] = cw["cps"].str.strip()
    return cw.drop_duplicates("soc6")


def load_union():
    raw = pd.read_excel(UNION_FILE, dtype=str, skiprows=2)
    raw.columns = [str(c).strip() for c in raw.columns]
    rename = {}
    for c in raw.columns:
        cl = c.lower()
        if cl in ("coc", "code") or re.match(r"^coc", cl):
            rename[c] = "cps"
        elif ("cov" in cl and "%" in c) or "density" in cl or "percent" in cl:
            rename[c] = "ext_union_cov_pct"
    raw = raw.rename(columns=rename)
    keep = [c for c in ["cps", "ext_union_cov_pct"] if c in raw.columns]
    u = raw[keep].copy()
    u["cps"] = u["cps"].astype(str).str.strip()
    if "ext_union_cov_pct" in u:
        u["ext_union_cov_pct"] = pd.to_numeric(u["ext_union_cov_pct"], errors="coerce")
    return u.drop_duplicates("cps")


def load_prestige():
    p = pd.read_csv(PRESTIGE_FILE, dtype=str)
    code_col = next((c for c in p.columns if "onet" in c.lower() and "code" in c.lower()), None)
    rate_col = next((c for c in p.columns if "opr" in c.lower() and "rating" in c.lower()), None)
    p = p[[code_col, rate_col]].rename(columns={code_col: "c", rate_col: "ext_prestige"})
    p["ext_prestige"] = pd.to_numeric(p["ext_prestige"], errors="coerce")
    p["soc6"] = p["c"].map(soc6)
    return p.dropna(subset=["soc6"]).groupby("soc6", as_index=False)["ext_prestige"].mean()


def load_bls():
    def read_li(sheet):
        d = pd.read_excel(BLS_FILE, sheet_name=sheet, skiprows=1)
        d.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in d.columns]
        code = next(c for c in d.columns if "code" in c.lower())
        typ = next(c for c in d.columns if "occupation" in c.lower() and "type" in c.lower())
        d = d[d[typ].astype(str).str.strip() == "Line item"].copy()
        d["soc6"] = d[code].map(soc6)
        return d

    def find(d, *kw):
        return next((c for c in d.columns if all(k in c.lower() for k in kw)), None)

    t110 = read_li("Table 1.10")
    S = pd.DataFrame({"soc6": t110["soc6"]})
    S["ext_sep_exit_rate"] = pd.to_numeric(t110[find(t110, "exit", "rate")], errors="coerce")
    S["ext_sep_transfer_rate"] = pd.to_numeric(t110[find(t110, "transfer", "rate")], errors="coerce")
    S = S.drop_duplicates("soc6")

    t12 = read_li("Table 1.2")
    T = pd.DataFrame({"soc6": t12["soc6"]})
    T["ext_self_employed_pct"] = pd.to_numeric(
        t12[find(t12, "self employed")].replace("—", 0), errors="coerce").fillna(0)
    T = T.drop_duplicates("soc6")
    return S.merge(T, on="soc6", how="outer")


# ------------------------- assemble --------------------------------------- #
def main():
    onet = onet_wide()
    n0 = len(onet)
    M = onet.merge(load_oews(), on="soc6", how="left")
    M = M.merge(load_crosswalk(), on="soc6", how="left")
    M = M.merge(load_union(), on="cps", how="left")
    M = M.merge(load_prestige(), on="soc6", how="left")
    M = M.merge(load_bls(), on="soc6", how="left")
    M = M.drop(columns=["cps"], errors="ignore")
    assert len(M) == n0, f"row count changed {n0}->{len(M)}"

    if "ext_self_employed_pct" in M.columns:
        M["ext_self_employed_pct"] = M["ext_self_employed_pct"].fillna(0)

    onet_feats = [c for c in M.columns if "__" in c]
    ext_feats = [c for c in M.columns if c.startswith("ext_")]
    all_feats = onet_feats + ext_feats

    keep = M[M[all_feats].isna().mean(axis=1) <= DROP_THRESHOLD].copy()
    print(f"\n[merge] {len(onet_feats)} skill + {len(ext_feats)} external")
    print(f"[drop ] {n0 - len(keep)} occ >90% empty | [keep] {len(keep)}")
    keep.to_excel(OUT, index=False)
    print(f"[ok] wrote {OUT} ({keep.shape[0]} x {keep.shape[1]})")

    print("\n--- external coverage ---")
    for c in ext_feats:
        print(f"  {c:22s} {keep[c].notna().mean()*100:5.1f}%")
    print("\n--- ETE blocks (should be n-1 cols each) ---")
    for s in ETE_SCALES:
        cols = [c for c in onet_feats if c.startswith(f"ete_{s.lower()}__")]
        print(f"  ete_{s.lower()}: {len(cols)} cols")


if __name__ == "__main__":
    main()