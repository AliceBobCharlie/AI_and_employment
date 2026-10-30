#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rating_error.py -- estimate how much of the variance of the IM and CX columns
of the feature matrix, and of each of its principal components, is sampling
error of the O*NET ratings.

Input:  output/master_clean.csv, output/resampled_spectra.npz (for k_max), and
        the O*NET text files in ONET_DIR (one per block in BLOCKS)
Output: one CSV per table in output/rating_error/

The columns are the IM (Abilities, Skills, Knowledge, Work Activities) and CX
(Work Context) columns of master_clean.csv. The CTP columns are left out: the
text files give them no standard error.

1. load() reads each block with read_block(), keeping the analysis scale, the
   rows without a category and the occupations of master_clean.csv, and
   pivots value, standard error (SE), N and domain source to occupation x
   column tables. checks.csv records the column and cell counts and the
   largest difference from the values in master_clean.csv.
2. item_table() gives for each column: its variance across occupations, the
   share of occupations with a SE, the mean SE^2, the error share e_j = mean
   SE^2 / variance and the reliability 1 - e_j, also on Analyst and on
   Incumbent ratings alone. e_j is computed on the occupations with a SE.
   blocks.csv summarises it by block.
3. standardised_sigma() gives every cell's SE divided by the column's sd, with
   the column's RMS SE where none is published. error_covariance() builds the
   error covariance C for each rho in RHOS: the mean squared SE on the
   diagonal, rho times the mean product of SEs between columns of the same
   error group in BLOCKS, and 0 elsewhere.
4. component_table() gives, for each of the first N_COMPONENTS_SHOWN
   eigenvectors v of the correlation matrix and for the ranges in
   component_ranges(), the error variance v' C v and its share of the
   eigenvalue.
5. error_spectrum() simulates N_SIM matrices of error alone (each cell normal
   with its standardised SE, with correlation rho within an error group) and
   gives the p95 of each eigenvalue and the number of observed eigenvalues
   above it before the first one below.
6. The eigenvalues of R - diag(e_j) (reduced_matrix.csv) and the variance
   beyond component k_max before and after the reduction (tail_summary.csv).
7. block_concentration() gives, for PC7 to PC k_max, the share of the squared
   loadings in each block against the block's share of columns.
8. items_vs_axes() leaves each column out, computes the first N_AXES principal
   components of the others, and gives the five-fold out-of-fold ridge R2 of
   the column on those scores and on all other columns (cv_r2()), each
   divided by the column's reliability; summarised by reliability band and by
   block.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from resampling_stability import load_spectra   # noqa: E402

ONET_DIR = Path("data_raw/onet/db_31_0_text")
CLEAN = Path("output/master_clean.csv")
OUT = Path("output/rating_error")

SEED = 0
N_SIM = 200
RHOS = [0.0, 0.1, 0.3]
N_COMPONENTS_SHOWN = 30
N_AXES = 6

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


def component_ranges(k_max, p):
    """Label -> component indices: PC1-3, PC4-6, PC7 to PC k_max and the rest."""
    return {"components 1-3": range(0, 3), "components 4-6": range(3, 6),
            f"components 7-{k_max}": range(6, k_max), f"components {k_max + 1}-{p}": range(k_max, p)}


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def read_block(prefix, socs):
    stems, scale, _ = BLOCKS[prefix]
    parts = []
    for stem in stems:
        path = ONET_DIR / f"{stem}.txt"
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


def component_table(ev, V, covs, ranges):
    rows = []
    for k in range(N_COMPONENTS_SHOWN):
        row = {"component": k + 1, "eigenvalue": ev[k]}
        for rho, C in covs.items():
            e = V[:, k] @ C @ V[:, k]
            row[f"error variance, rho={rho}"] = e
            row[f"error share, rho={rho}"] = e / ev[k]
        rows.append(row)
    for name, idx in ranges.items():
        idx = list(idx)
        row = {"component": name, "eigenvalue": ev[idx].sum()}
        for rho, C in covs.items():
            e = np.einsum("jk,jl,lk->", V[:, idx], C, V[:, idx])
            row[f"error variance, rho={rho}"] = e
            row[f"error share, rho={rho}"] = e / ev[idx].sum()
        rows.append(row)
    total = {"component": f"all {len(ev)}", "eigenvalue": ev.sum()}
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


def cv_r2(X, y):
    """Five-fold out-of-fold R2 of a ridge regression, scaler and penalty fitted in-fold."""
    pred = np.empty_like(y)
    for tr, te in KFold(5, shuffle=True, random_state=SEED).split(X):
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 5, 30)))
        pred[te] = model.fit(X[tr], y[tr]).predict(X[te])
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def items_vs_axes(cols, X, items):
    """Leave-one-column-out R2 from the six leading components and from all
    other columns, set against each item's reliability."""
    Z = (X - X.mean(axis=0)) / X.std(axis=0)
    other = f"R2 other {len(cols) - 1}"
    rows = []
    for j, col in enumerate(cols):
        rest = np.delete(Z, j, axis=1)
        rest = rest - rest.mean(axis=0)
        _, _, vt = np.linalg.svd(rest, full_matrices=False)
        rows.append({"item": col,
                     "R2 six axes": cv_r2(rest @ vt[:N_AXES].T, Z[:, j]),
                     other: cv_r2(rest, Z[:, j])})
    d = pd.DataFrame(rows).merge(items[["item", "block", "reliability 1 - e_j"]], on="item")
    d["six axes / reliability"] = d["R2 six axes"] / d["reliability 1 - e_j"]
    d[f"{other} / reliability"] = d[other] / d["reliability 1 - e_j"]

    rel = d["reliability 1 - e_j"]
    summary = pd.DataFrame([
        {"measure": "Spearman(reliability, R2 six axes)", "value": spearmanr(rel, d["R2 six axes"])[0]},
        {"measure": "Pearson(reliability, R2 six axes)", "value": np.corrcoef(rel, d["R2 six axes"])[0, 1]},
        {"measure": f"Spearman(reliability, {other})", "value": spearmanr(rel, d[other])[0]},
    ])
    bands = pd.cut(rel, [0, .7, .8, .9, 1], labels=["<0.7", "0.7-0.8", "0.8-0.9", ">=0.9"])
    by_band = (d.groupby(bands, observed=True)[["R2 six axes", other, "six axes / reliability"]]
                .median().assign(items=d.groupby(bands, observed=True).size()).reset_index()
                .rename(columns={"reliability 1 - e_j": "reliability band"}))
    by_block = d.groupby("block")[["reliability 1 - e_j", "R2 six axes", "six axes / reliability",
                                   other, f"{other} / reliability"]].median().reset_index()
    return d.sort_values("R2 six axes"), summary, by_band, by_block


# --------------------------------------------------------------------------- #
def block_concentration(V, cols, k_max):
    """For PC7 to PC k_max: the share of the squared loadings in each block,
    and the largest block share over that block's share of columns."""
    blocks = np.array([c.split("__")[0] for c in cols])
    names = list(dict.fromkeys(blocks))
    rows = [{"component": "share of columns", **{b: (blocks == b).mean() for b in names}}]
    for t in range(6, k_max):
        sq = V[:, t] ** 2
        row = {"component": f"PC{t + 1}", **{b: sq[blocks == b].sum() for b in names}}
        row["largest block share / its column share"] = max(
            row[b] / (blocks == b).mean() for b in names)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(SEED)
    try:
        clean = pd.read_csv(CLEAN)
    except OSError as err:
        sys.exit(f"[fail] read {CLEAN}: {err}")
    if not ONET_DIR.exists():
        sys.exit(f"[fail] {ONET_DIR} not found")
    k_max = int(load_spectra()["k_max"])
    print(f"[ok] read {CLEAN}: {len(clean)} occupations; k_max {k_max}")
    cols, groups, X, SE, N, source, checks = load(clean)
    if np.isnan(X).any():
        OUT.mkdir(parents=True, exist_ok=True)
        checks.to_csv(OUT / "checks.csv", index=False)
        sys.exit(f"[fail] some cells have no value in the text files; see {OUT / 'checks.csv'}")
    print(f"[ok] {len(cols)} columns with values for every occupation")

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
    p = len(cols)
    ranges = component_ranges(k_max, p)
    S = standardised_sigma(X, SE)
    covs = {rho: error_covariance(S, groups, rho) for rho in RHOS}
    components = component_table(ev, V, covs, ranges)
    spectrum, spectrum_summary = error_spectrum(S, groups, rng, ev)
    print("[ok] error along components and error-only spectrum")

    e = items["error share e_j"].to_numpy()
    ev_reduced = np.linalg.eigvalsh(R - np.diag(e))[::-1]
    reduced = pd.DataFrame({"component": np.arange(1, N_COMPONENTS_SHOWN + 1),
                            "eigenvalue of R": ev[:N_COMPONENTS_SHOWN],
                            "eigenvalue of R - diag(e)": ev_reduced[:N_COMPONENTS_SHOWN]})
    rest = list(ranges)[-1]
    tail = pd.DataFrame([{
        f"variance beyond component {k_max} (sum of eigenvalues)": ev[k_max:].sum(),
        "share of total": ev[k_max:].sum() / ev.sum(),
        "total rating error (sum of e_j)": e.sum(),
        "rating error as share of total": e.sum() / ev.sum(),
        f"error along {rest}, independent errors": components.loc[
            components["component"] == rest, "error variance, rho=0.0"].iloc[0],
        f"reduced matrix: variance beyond {k_max}": ev_reduced[k_max:].clip(min=0).sum(),
        "reduced matrix: negative eigenvalues (sum)": ev_reduced[ev_reduced < 0].sum(),
    }])
    concentration = block_concentration(V, cols, k_max)

    vs_axes, vs_summary, vs_band, vs_block = items_vs_axes(cols, X, items)
    print(f"[ok] {p} leave-one-column-out fits")

    tables = {"checks": checks, "blocks": blocks, "tail_summary": tail,
              "components": components, "error_spectrum": spectrum,
              "error_spectrum_summary": spectrum_summary, "reduced_matrix": reduced,
              "block_concentration": concentration,
              "items": items.sort_values("error share e_j", ascending=False),
              "axes_vs_reliability": vs_summary, "axes_by_reliability_band": vs_band,
              "axes_by_block": vs_block, "items_vs_axes": vs_axes}
    OUT.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(OUT / f"{name}.csv", index=False)
    print(f"[ok] wrote {len(tables)} tables to {OUT}/")


if __name__ == "__main__":
    main()