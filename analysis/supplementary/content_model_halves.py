#!/usr/bin/env python3
"""
content_model_halves.py -- find and rotate the stable dimensions of the job
half and of the worker half of the feature matrix separately, describe their
axes, and compare them with each other and with R1-R6.

Input:  data/prepared/master_clean.csv, output/pca_rotated/occupation_coordinates.csv
Output: output/content_model_halves/<half>/          for each half in HALVES:
          eigenvalues.csv ... dimensions.csv, resampled_spectra.npz,
          subspace_stability.png                      as written by run_experiment()
          axes.csv          each axis: group, % of variance, matched R axis and
                            r, r with R1-R6, strongest columns and occupations
                            at each end
          axes.html         one interactive figure per axis, chosen from a menu:
                            loadings by block (above) and occupation scores
                            by SOC major group (below)
        output/content_model_halves/recover_axes.csv  R2 of R1-R6 from each half's axes
        output/content_model_halves/canonical.csv     canonical correlations, job axes
                                                      against worker axes
        output/content_model_halves/job_x_worker.csv  r between every job axis and
                                                      every worker axis

HALVES gives each half's column prefixes. For each half, in order:

1. The half's columns of master_clean.csv are z-scored. run_experiment() from
   resampling_stability.py finds k_max, the stable components and the stable
   subspace (found_groups()), and tests their rotated axes.
2. axes() rotates each group on the full sample with rotate_group() from
   pca_rotated.py and gives the loadings and unit-variance scores. match()
   pairs the axes with R1-R6 by the largest total |r| (linear_sum_assignment)
   and flips each matched axis so its r is positive; an unmatched axis is
   flipped so the sum of its cubed loadings is positive. The axes are named
   with the half's initial (J1, J2, ... and W1, W2, ...).
3. describe() lists, for each axis, the TOP columns with the largest and the
   smallest loadings and the TOP occupations at each end.
4. plot_half() writes axes.html with plotly, one figure per axis shown
   through a menu: each column's loading, by block (block_of() from
   pca_rotated.py), with jitter; and each occupation's score, by SOC major
   group (SOC_GROUPS), the groups ordered by their median, which a bar marks.
   Hovering shows the column or occupation. plotly.js is embedded in the file.

After both halves: r2() gives the R2 of each of R1-R6 from each half's axes;
canonical() the canonical correlations between the job and the worker axes;
and job_x_worker.csv the correlation of every job axis with every worker axis.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pca_rotated import AXES, MASTER, OUTDIR, SEED, block_of, rotate_group, standardize   # noqa: E402
from resampling_stability import found_groups, run_experiment                          # noqa: E402

OUT = OUTDIR / "content_model_halves"
HALVES = {"job": ("workact_im__", "workctx_cx__", "workctx_ctp__"),
          "worker": ("skills_im__", "abilities_im__", "knowledge_im__")}
N_EIGEN = 60            # eigenvalues tested by the parallel analysis (at most the columns)
TOP = 8                 # columns and occupations listed per end
BLOCK_NAMES = {"skills_im": "Skills", "abilities_im": "Abilities", "knowledge_im": "Knowledge",
               "workact_im": "Work Activities", "workctx_cx": "Work Context"}
SOC_GROUPS = {
    "11": "Management", "13": "Business and financial operations",
    "15": "Computer and mathematical", "17": "Architecture and engineering",
    "19": "Life, physical and social science", "21": "Community and social service",
    "23": "Legal", "25": "Educational instruction and library",
    "27": "Arts, design, entertainment, sports and media",
    "29": "Healthcare practitioners and technical", "31": "Healthcare support",
    "33": "Protective service", "35": "Food preparation and serving",
    "37": "Building and grounds cleaning and maintenance",
    "39": "Personal care and service", "41": "Sales and related",
    "43": "Office and administrative support", "45": "Farming, fishing and forestry",
    "47": "Construction and extraction", "49": "Installation, maintenance and repair",
    "51": "Production", "53": "Transportation and material moving",
}


def short(col):
    return col.split("__", 1)[1]


def axes(Xz, ev, V, groups):
    """Loadings and unit-variance scores of every group, each rotated by rotate_group()."""
    T = standardize(Xz) @ V / np.sqrt(ev[: V.shape[1]])
    Ls, Ss = [], []
    for idx in groups:
        L, R = rotate_group(V, ev, idx)
        Ls.append(L)
        Ss.append(T[:, idx] @ R)
    return np.hstack(Ls), np.hstack(Ss)


def match(L, S, paper):
    """L and S flipped as in step 2, the matched R axis of each axis (or ""),
    and the r of every axis with R1-R6."""
    C = np.corrcoef(S.T, paper.to_numpy().T)[: S.shape[1], S.shape[1]:]
    rows, cols = linear_sum_assignment(-np.abs(C))
    matched = [""] * S.shape[1]
    sign = np.where((L ** 3).sum(axis=0) < 0, -1.0, 1.0)
    for i, j in zip(rows, cols):
        matched[i] = paper.columns[j]
        sign[i] = np.sign(C[i, j]) or 1.0
    return L * sign, S * sign, matched, C * sign[:, None]


def describe(names, groups, L, S, C, matched, feats, titles, paper, total):
    group_of = [g for g, idx in zip(("stable components", "stable subspace"), groups) for _ in idx]
    rows = []
    for j, name in enumerate(names):
        lo, so = np.argsort(L[:, j]), np.argsort(S[:, j])
        rows.append({"axis": name, "group": group_of[j],
                     "% of variance": 100 * (L[:, j] ** 2).sum() / total,
                     "matched R axis": matched[j],
                     **{f"r with {c}": C[j, i] for i, c in enumerate(paper.columns)},
                     "+ columns": "; ".join(f"{short(feats[i])} {L[i, j]:+.2f}" for i in lo[::-1][:TOP]),
                     "- columns": "; ".join(f"{short(feats[i])} {L[i, j]:+.2f}" for i in lo[:TOP]),
                     "+ occupations": "; ".join(titles[i] for i in so[::-1][:TOP]),
                     "- occupations": "; ".join(titles[i] for i in so[:TOP])})
    return pd.DataFrame(rows)


def axis_traces(loadings, feats, scores, codes, titles, rng):
    """The traces of one axis, and the SOC groups in the order of their median."""
    traces = []
    blocks = np.array([BLOCK_NAMES[block_of(c)] for c in feats])
    for i, b in enumerate([b for b in BLOCK_NAMES.values() if b in blocks]):
        k = np.flatnonzero(blocks == b)
        traces.append((1, go.Scatter(
            x=i + rng.uniform(-0.3, 0.3, len(k)), y=loadings[k], mode="markers", name=b,
            text=[short(feats[j]) for j in k], marker={"size": 6, "opacity": 0.75},
            hovertemplate="%{text}<br>loading %{y:.2f}<extra>" + b + "</extra>")))
    groups = np.array([SOC_GROUPS.get(c[:2], c[:2]) for c in codes])
    medians = pd.Series(scores).groupby(groups).median().sort_values()
    position = {g: i for i, g in enumerate(medians.index)}
    x = np.array([position[g] for g in groups]) + rng.uniform(-0.3, 0.3, len(groups))
    traces.append((2, go.Scatter(
        x=x, y=scores, mode="markers", name="occupations", showlegend=False,
        text=[f"{t}<br>{g}" for t, g in zip(titles, groups)],
        marker={"size": 5, "opacity": 0.6, "color": "#4C72B0"},
        hovertemplate="%{text}<br>score %{y:.2f}<extra></extra>")))
    traces.append((2, go.Scatter(
        x=list(range(len(medians))), y=medians.to_numpy(), mode="markers", name="median",
        showlegend=False, text=list(medians.index),
        marker={"symbol": "line-ew-open", "size": 22, "line": {"width": 3}, "color": "black"},
        hovertemplate="%{text}<br>median %{y:.2f}<extra></extra>")))
    return traces, list(medians.index)


def plot_half(path, names, L, S, feats, codes, titles):
    fig = make_subplots(rows=2, cols=1, row_heights=[0.4, 0.6], vertical_spacing=0.08)
    rng = np.random.default_rng(SEED)
    per_axis, orders = [], []
    for j, name in enumerate(names):
        traces, order = axis_traces(L[:, j], feats, S[:, j], codes, titles, rng)
        for row, t in traces:
            t.visible = j == 0
            fig.add_trace(t, row=row, col=1)
        per_axis.append(len(traces))
        orders.append(order)
    blocks = [b for b in BLOCK_NAMES.values() if b in {BLOCK_NAMES[block_of(c)] for c in feats}]
    buttons, start = [], 0
    for j, name in enumerate(names):
        visible = [False] * sum(per_axis)
        visible[start:start + per_axis[j]] = [True] * per_axis[j]
        start += per_axis[j]
        buttons.append({"label": name, "method": "update", "args": [
            {"visible": visible},
            {"yaxis.title.text": f"loading on {name}", "yaxis2.title.text": f"score on {name}",
             "xaxis2.tickvals": list(range(len(orders[j]))), "xaxis2.ticktext": orders[j]}]})
    fig.update_layout(updatemenus=[{"buttons": buttons, "x": 0, "xanchor": "left", "y": 1.08, "yanchor": "bottom"}],
                      height=900, template="plotly_white", hovermode="closest",
                      legend={"orientation": "h", "y": 1.02, "x": 1, "xanchor": "right"})
    fig.update_xaxes(tickvals=list(range(len(blocks))), ticktext=blocks, row=1, col=1)
    fig.update_xaxes(tickvals=list(range(len(orders[0]))), ticktext=orders[0], tickangle=-60, row=2, col=1)
    fig.update_yaxes(title_text=f"loading on {names[0]}", zeroline=True, row=1, col=1)
    fig.update_yaxes(title_text=f"score on {names[0]}", zeroline=True, row=2, col=1)
    fig.write_html(path, include_plotlyjs=True)


def r2(y, B):
    """In-sample R2 of y on the columns of B, with an intercept."""
    Q, _ = np.linalg.qr(np.column_stack([np.ones(len(y)), B]))
    yc = y - y.mean()
    return float(((Q.T @ yc) ** 2).sum() / (yc ** 2).sum())


def canonical(A, B):
    """Canonical correlations between the columns of A and of B."""
    Qa, _ = np.linalg.qr(A - A.mean(axis=0))
    Qb, _ = np.linalg.qr(B - B.mean(axis=0))
    return np.linalg.svd(Qa.T @ Qb, compute_uv=False)


def main():
    try:
        clean = pd.read_csv(MASTER, index_col="onet_soc")
        paper = pd.read_csv(AXES, index_col=0).reindex(clean.index)
    except OSError as err:
        sys.exit(f"[fail] {err}")
    paper = paper[[c for c in paper.columns if c.startswith("R")]]
    titles = clean["title"].tolist()
    codes = list(clean.index)
    print(f"[ok] read {MASTER}: {len(clean)} occupations; {AXES}: {', '.join(paper.columns)}")

    found, recover = {}, []
    for half, prefixes in HALVES.items():
        out = OUT / half
        feats = [c for c in clean.columns if c.startswith(prefixes)]
        Xz = standardize(clean[feats].to_numpy(float))
        tag = half[0].upper()

        def label(idx, tag=tag):
            return [f"{tag}{i + 1}" for i in idx]

        res = run_experiment(Xz, feats, out, out / "resampled_spectra.npz", out / "subspace_stability.png",
                             n_eigen=min(N_EIGEN, len(feats)), names=label)
        groups = found_groups(res["n_comp"], res["sub_end"])
        s = res["spectra"]
        L, S = axes(Xz, s["full_ev"], s["full_V"], groups)
        L, S, matched, C = match(L, S, paper)
        names = [n for idx in groups for n in label(idx)]
        describe(names, groups, L, S, C, matched, feats, titles, paper, s["full_ev"].sum()).to_csv(
            out / "axes.csv", index=False)
        plot_half(out / "axes.html", names, L, S, feats, codes, titles)
        found[half] = (names, S)
        recover.append({"half": half, "axes": len(names),
                        **{c: r2(paper[c].to_numpy(), S) for c in paper.columns}})
        print(f"[ok] {half} half: {len(feats)} columns, axes {', '.join(names)}; tables and figures in {out}/")

    (jn, J), (wn, W) = found["job"], found["worker"]
    cc = canonical(J, W)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(recover).to_csv(OUT / "recover_axes.csv", index=False)
    pd.DataFrame({"pair": range(1, len(cc) + 1), "canonical r": cc}).to_csv(OUT / "canonical.csv", index=False)
    pd.DataFrame(np.corrcoef(J.T, W.T)[: J.shape[1], J.shape[1]:], index=jn, columns=wn
                 ).reset_index(names="job axis").to_csv(OUT / "job_x_worker.csv", index=False)
    print(f"[ok] wrote recover_axes.csv, canonical.csv and job_x_worker.csv to {OUT}/")


if __name__ == "__main__":
    main()