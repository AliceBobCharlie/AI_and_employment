#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rating_error_share.py -- how much of the variance in the paper's matrix, and
how much of each principal component, is sampling error in O*NET's ratings?

Every O*NET rating is the mean over a sample of incumbents, occupation experts
or analysts, and O*NET publishes its standard error. For the 216 columns of
the paper's matrix (Importance for Abilities, Skills, Knowledge and Work
Activities, Context for Work Context) and the 910 occupations of
output/master_clean.csv:

  item error share   e_j = mean over occupations of SE^2, divided by the
                     variance of the item across occupations. Because the
                     analysis z-scores every column, e_j is also the error
                     variance of the standardised column. The reliability of
                     the item is 1 - e_j.
                     SE is published only for incumbent and analyst ratings;
                     occupation-expert and provisional analyst ratings carry
                     none. e_j is therefore computed on the occupations with a
                     SE and applied to the item as a whole, which assumes the
                     ratings without a SE are no more and no less precise. The
                     share of occupations with a SE is reported for every item.

  error along each component
                     With error covariance C (216 x 216, standardised units),
                     the error variance along the unit eigenvector v_k of the
                     correlation matrix is v_k' C v_k, to be set against the
                     eigenvalue lambda_k. Summed over all 216 components it is
                     trace(C) = sum of e_j, however C is arranged.
                       independent errors   C = diag(e_j)
                       within-group errors  the same respondents rate every
                                            item of a questionnaire, so a
                                            response style shared within an
                                            occupation's sample correlates the
                                            errors of items in one group. The
                                            correlation rho cannot be estimated
                                            from published data, so it is set
                                            to RHOS as a sensitivity range.
                                            Groups: Abilities with Skills (the
                                            same analysts rate both), Knowledge,
                                            Work Activities, Work Context (each
                                            a separate incumbent questionnaire).
                                            Cov_jl = rho * mean_i(SE_ij SE_il) /
                                            (sd_j sd_l) for j, l in one group.

  error-only spectrum
                     N_SIM matrices of pure rating error are simulated, each
                     cell drawn with the published SE of that occupation and
                     item (the item's root-mean-square SE where none is
                     published), standardised by the item's sd, with the
                     within-group correlation rho. The 95th percentile of their
                     k-th eigenvalue is what rating error alone produces; an
                     observed eigenvalue above it is more than rating error.
                     This is parallel analysis with a measured null instead of
                     a permutation null.

  reduced correlation matrix
                     Eigenvalues of R - diag(e_j): the correlation matrix with
                     the error variance taken off the diagonal, as in principal
                     axis factoring. What is left in the tail is item-specific
                     content plus any weak shared structure.

Reads data_raw/onet/db_31_0_text and output/master_clean.csv. Files are read
one at a time, only the columns needed.

Run from the project root:  python supplementary/rating_error_share.py
Writes output/rating_error_share.xlsx and prints every sheet.
"""

import numpy as np
import pandas as pd
from pathlib import Path

ONET_DIR = Path("data_raw/onet/db_31_0_text")
CLEAN = Path("output/master_clean.csv")
OUT = Path("output/rating_error_share.xlsx")

SEED = 0
N_SIM = 50
RHOS = [0.0, 0.1, 0.3]
N_COMPONENTS_SHOWN = 30

# Column prefix -> (files, analysis scale, error group)
BLOCKS = {
    "abilities": (["Abilities"], "IM", "analysts: abilities and skills"),
    "skills":    (["Essential Skills", "Transferable Skills"], "IM", "analysts: abilities and skills"),
    "knowledge": (["Knowledge"], "IM", "knowledge questionnaire"),
    "workact":   (["Work Activities"], "IM", "work activities questionnaire"),
    "workctx":   (["Work Context"], "CX", "work context questionnaire"),
}
COLUMNS = {
    "O*NET-SOC Code": "soc", "Element Name": "element", "Scale ID": "scale",
    "Category": "category", "Data Value": "value", "N": "n",
    "Standard Error": "se", "Recommend Suppress": "suppress",
    "Domain Source": "source",
}
TIERS = {"components 1-3": range(0, 3), "components 4-6": range(3, 6),
         "components 7-14": range(6, 14), "components 15-216": range(14, 216)}


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def read_block(prefix, socs):
    stems, scale, _ = BLOCKS[prefix]
    parts = []
    for stem in stems:
        path = ONET_DIR / f"{stem}.txt"
        print(f"  reading {path}")
        parts.append(pd.read_csv(path, sep="\t", dtype=str, encoding="utf-8",
                                 usecols=lambda c: c in COLUMNS).rename(columns=COLUMNS))
    df = pd.concat(parts, ignore_index=True)
    if "category" not in df.columns:
        df["category"] = np.nan
    df = df[(df["scale"] == scale) & df["category"].isna() & df["soc"].isin(socs)].copy()
    for col in ("value", "n", "se"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["column"] = prefix + "_" + scale.lower() + "__" + df["element"]
    df["group"] = BLOCKS[prefix][2]
    return df


def load(clean):
    socs = set(clean["onet_soc"])
    rows = pd.concat([read_block(p, socs) for p in BLOCKS], ignore_index=True)
    dup = rows.duplicated(["soc", "column"])
    rows = rows[~dup]
    cols = [c for c in clean.columns if c.startswith(tuple(f"{p}_{s.lower()}__"
                                                           for p, (_, s, _) in BLOCKS.items()))]
    value = rows.pivot(index="soc", columns="column", values="value")
    se = rows.pivot(index="soc", columns="column", values="se")
    n = rows.pivot(index="soc", columns="column", values="n")
    source = rows.pivot(index="soc", columns="column", values="source")
    order = clean["onet_soc"]
    missing_cols = [c for c in cols if c not in value.columns]
    value, se, n, source = (t.reindex(index=order, columns=cols) for t in (value, se, n, source))

    published = clean.set_index("onet_soc")[cols]
    diff = (value - published).abs().to_numpy()
    checks = pd.DataFrame([
        {"check": "columns in master_clean (feature columns)", "value": len(cols)},
        {"check": "of these not found in the text files", "value": len(missing_cols)},
        {"check": "occupations", "value": len(order)},
        {"check": "duplicate (occupation, item) rows dropped", "value": int(dup.sum())},
        {"check": "cells with no value in the text files", "value": int(value.isna().sum().sum())},
        {"check": "max |text value - master_clean value|", "value": float(np.nanmax(diff))},
        {"check": "cells with a published SE", "value": int(se.notna().sum().sum())},
        {"check": "share of cells with a published SE", "value": float(se.notna().mean().mean())},
    ])
    if missing_cols:
        print("  columns not found:", missing_cols[:10])
    groups = np.array([BLOCKS[c.split("_")[0]][2] for c in cols])
    return cols, groups, value.to_numpy(float), se.to_numpy(float), n.to_numpy(float), source, checks


# --------------------------------------------------------------------------- #
# Error measures
# --------------------------------------------------------------------------- #
def item_table(cols, groups, X, SE, N, source):
    var = X.var(axis=0, ddof=1)
    has = ~np.isnan(SE)
    mean_se2 = np.nanmean(SE ** 2, axis=0)
    rows = []
    for j, c in enumerate(cols):
        src = source.iloc[:, j]
        row = {"item": c, "block": c.split("__")[0], "group": groups[j],
               "variance across occupations": var[j],
               "occupations with SE": int(has[:, j].sum()),
               "share with SE": has[:, j].mean(),
               "mean SE^2": mean_se2[j],
               "median N (with SE)": np.nanmedian(N[:, j]),
               "error share e_j": mean_se2[j] / var[j],
               "reliability 1 - e_j": 1 - mean_se2[j] / var[j]}
        for s in ("Analyst", "Incumbent"):
            mask = (src == s).to_numpy() & has[:, j]
            row[f"error share, {s} ratings only"] = (
                np.mean(SE[mask, j] ** 2) / var[j] if mask.any() else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def standardised_sigma(X, SE):
    """SE of every cell in standardised units, the item's RMS SE where none."""
    sd = X.std(axis=0, ddof=1)
    rms = np.sqrt(np.nanmean(SE ** 2, axis=0))
    filled = np.where(np.isnan(SE), rms, SE)
    return filled / sd


def error_covariance(S, groups, rho):
    """C in standardised units: diag = mean_i S_ij^2; within-group off-diagonal
    = rho * mean_i S_ij S_il."""
    M = (S.T @ S) / len(S)
    same = groups[:, None] == groups[None, :]
    C = np.where(same, rho * M, 0.0)
    np.fill_diagonal(C, np.diag(M))
    return C


def component_table(ev, V, covs):
    rows = []
    for k in range(N_COMPONENTS_SHOWN):
        row = {"component": k + 1, "eigenvalue": ev[k]}
        for rho, C in covs.items():
            e = V[:, k] @ C @ V[:, k]
            row[f"error variance, rho={rho}"] = e
            row[f"error share, rho={rho}"] = e / ev[k]
        rows.append(row)
    for name, idx in TIERS.items():
        idx = list(idx)
        row = {"component": name, "eigenvalue": ev[idx].sum()}
        for rho, C in covs.items():
            e = np.einsum("jk,jl,lk->", V[:, idx], C, V[:, idx])
            row[f"error variance, rho={rho}"] = e
            row[f"error share, rho={rho}"] = e / ev[idx].sum()
        rows.append(row)
    total = {"component": "all 216", "eigenvalue": ev.sum()}
    for rho, C in covs.items():
        total[f"error variance, rho={rho}"] = np.trace(C)
        total[f"error share, rho={rho}"] = np.trace(C) / ev.sum()
    rows.append(total)
    return pd.DataFrame(rows)


def error_spectrum(S, groups, rng, ev):
    """95th percentile of the k-th eigenvalue of simulated pure-error matrices."""
    n, p = S.shape
    names = list(dict.fromkeys(groups))
    gidx = np.array([names.index(g) for g in groups])
    out = {"component": np.arange(1, N_COMPONENTS_SHOWN + 1), "observed eigenvalue": ev[:N_COMPONENTS_SHOWN]}
    counts = {}
    for rho in RHOS:
        sims = np.empty((N_SIM, N_COMPONENTS_SHOWN))
        for s in range(N_SIM):
            shared = rng.standard_normal((n, len(names)))[:, gidx]
            unique = rng.standard_normal((n, p))
            E = S * (np.sqrt(rho) * shared + np.sqrt(1 - rho) * unique)
            E = E - E.mean(axis=0)
            sims[s] = (np.linalg.svd(E, compute_uv=False) ** 2 / (n - 1))[:N_COMPONENTS_SHOWN]
        p95 = np.percentile(sims, 95, axis=0)
        out[f"error-only p95, rho={rho}"] = p95
        above = ev[:N_COMPONENTS_SHOWN] > p95
        counts[rho] = int(np.argmin(above)) if not above.all() else N_COMPONENTS_SHOWN
    table = pd.DataFrame(out)
    summary = pd.DataFrame([{"rho": rho, "observed components above the error-only p95": c}
                            for rho, c in counts.items()])
    return table, summary


# --------------------------------------------------------------------------- #
def main():
    rng = np.random.default_rng(SEED)
    clean = pd.read_csv(CLEAN)
    print(f"[read] {CLEAN}: {len(clean)} occupations")
    cols, groups, X, SE, N, source, checks = load(clean)
    if np.isnan(X).any():
        raise ValueError("some cells have no value in the text files; see the checks sheet")

    items = item_table(cols, groups, X, SE, N, source)
    blocks = items.groupby("block").agg(
        items=("item", "size"),
        share_with_SE=("share with SE", "mean"),
        error_share_mean=("error share e_j", "mean"),
        error_share_median=("error share e_j", "median"),
        error_share_min=("error share e_j", "min"),
        error_share_max=("error share e_j", "max"),
        reliability_mean=("reliability 1 - e_j", "mean"),
    ).reset_index()
    blocks.loc[len(blocks)] = ["all", len(items), items["share with SE"].mean(),
                               items["error share e_j"].mean(), items["error share e_j"].median(),
                               items["error share e_j"].min(), items["error share e_j"].max(),
                               items["reliability 1 - e_j"].mean()]

    R = np.corrcoef(X.T)
    ev, V = np.linalg.eigh(R)
    ev, V = ev[::-1], V[:, ::-1]
    S = standardised_sigma(X, SE)
    covs = {rho: error_covariance(S, groups, rho) for rho in RHOS}
    components = component_table(ev, V, covs)
    spectrum, spectrum_summary = error_spectrum(S, groups, rng, ev)

    e = items["error share e_j"].to_numpy()
    ev_reduced = np.linalg.eigvalsh(R - np.diag(e))[::-1]
    reduced = pd.DataFrame({"component": np.arange(1, N_COMPONENTS_SHOWN + 1),
                            "eigenvalue of R": ev[:N_COMPONENTS_SHOWN],
                            "eigenvalue of R - diag(e)": ev_reduced[:N_COMPONENTS_SHOWN]})
    tail = pd.DataFrame([{
        "variance beyond component 14 (sum of eigenvalues)": ev[14:].sum(),
        "share of total": ev[14:].sum() / ev.sum(),
        "total rating error (sum of e_j)": e.sum(),
        "rating error as share of total": e.sum() / ev.sum(),
        "error along components 15-216, independent errors": components.loc[
            components["component"] == "components 15-216", "error variance, rho=0.0"].iloc[0],
        "reduced matrix: variance beyond 14": ev_reduced[14:].clip(min=0).sum(),
        "reduced matrix: negative eigenvalues (sum)": ev_reduced[ev_reduced < 0].sum(),
    }])

    sheets = {"checks": checks, "blocks": blocks, "tail_summary": tail,
              "components": components, "error_spectrum": spectrum,
              "error_spectrum_summary": spectrum_summary, "reduced_matrix": reduced,
              "items": items.sort_values("error share e_j", ascending=False)}
    OUT.parent.mkdir(exist_ok=True)
    with pd.ExcelWriter(OUT) as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30,
                           "display.max_rows", 300, "display.float_format", "{:.4f}".format):
        for name, df in sheets.items():
            print(f"\n=== {name} ({len(df)} rows) ===")
            print(df.to_string(index=False))
    print(f"\n[ok] wrote {OUT}")


if __name__ == "__main__":
    main()