#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_axes.py -- do the six axes measure what their names say?
(Sections 5.2 and 5.4)

The names of R1-R6 were read from the loadings and from the occupations at
each end. This script sets each name against descriptions of the same
occupations that took no part in forming the axes, all from the O*NET 31.0
text files, and sets the clusters of Section 5.2 against the axes.

  1 categories   O*NET's own grouping of the Content Model elements (the
                 hierarchy encoded in each Element ID): the mean loading of
                 each category on each axis, and the categories of the 15
                 strongest items at each pole. R4-R6 contrast items within a
                 category more than they contrast categories, so the pole
                 composition is the more telling table for them.
  2 interests    The six RIASEC interest types (Career Interest Types.txt,
                 scale OI; "Interests.txt" before release 30.3) with Prediger's
                 (1982) two summary dimensions,
                     people - things = 2S + A + E - 2R - I - C
                     data - ideas    = 1.73 (E + C - I - A)
                 and the 41 basic interest areas (Specific Interest
                 Areas.txt, scale OI): the areas most associated with each end
                 of each axis.
  3 job zones    O*NET's levels of preparation (Job Zones.txt).
  4 software     The software each occupation uses (Software Skills.txt;
                 "Technology Skills.txt" before release 30.3): the number of
                 software categories used, and the categories most associated
                 with each end of each axis. Release 31.0 has no tools table.
  5 work styles  The 21 work styles (Work Styles.txt) on the Impact scale (WI,
                 -3 to +3); the Distinctiveness Rank (DR) is not used.
  6 clusters     The k = 2 partition of occupations from cluster_onet.py
                 against the axes, the two skill clusters by O*NET category
                 and by R1 loading, and the SOC major groups in the space of
                 the three leading components (Section 5.2).

What each name predicts:
  R1 physical - symbolic       Realistic +, fewer software categories
  R2 complex - simple          Job Zone +, Investigative +
  R3 people - artefacts        Social +, engineering and IT interests -
  R4 hands-on - field          health-care interests +, transport and
                               outdoor interests -
  R5 standardised - open-ended Conventional +, Artistic -, data - ideas +
  R6 commercial - individual   Enterprising +, business software +

Each section runs only if its input is found and says so otherwise. The
terminal shows one line per section; every table is in the workbook, and
-v prints them as well.

Run after pca_rotated.py and cluster_onet.py:
    python validate_axes.py [O*NET text folder] [-v]
Reads output/occupation_coordinates.csv, output/axis_loadings.csv,
output/master_clean.csv, output/cluster_assignments.csv,
output/cluster_skills.csv and the O*NET text files (default
data_raw/onet/db_31_0_text). Writes output/validate_axes.xlsx.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from pca_rotated import AXES, LOADINGS, MASTER, FEATURE_PREFIXES

OUT = Path("output/validate_axes.xlsx")
CLUSTERS = Path("output/cluster_assignments.csv")
SKILL_CLUSTERS = Path("output/cluster_skills.csv")

AX = ["R1", "R2", "R3", "R4", "R5", "R6"]
NAMES = {"R1": "physical - symbolic", "R2": "complex - simple",
         "R3": "people - artefacts", "R4": "hands-on - field",
         "R5": "standardised - open-ended", "R6": "commercial - individual"}
CODE = "O*NET-SOC Code"
RIASEC = ["Realistic", "Investigative", "Artistic", "Social", "Enterprising", "Conventional"]
POLE_ITEMS = 15          # items per pole in the category composition
MIN_OCC = 40             # a software category must be used by this many occupations
TOP = 6                  # interest areas or software categories listed per end

# The O*NET files each feature block comes from (for Element IDs).
BLOCK_FILES = {"skills_im": ["Essential Skills", "Transferable Skills"],
               "abilities_im": ["Abilities"], "knowledge_im": ["Knowledge"],
               "workact_im": ["Work Activities"], "workctx_cx": ["Work Context"]}

# Fallback labels for the O*NET hierarchy, used if Content Model Reference.txt
# is not present: the release 31.0 names. These are the expert groupings, not ours.
FALLBACK = {
    "1.A.1": "Cognitive abilities", "1.A.2": "Psychomotor abilities",
    "1.A.3": "Physical abilities", "1.A.4": "Sensory abilities",
    "2.A.1": "Essential skills: foundational areas",
    "2.A.2": "Essential skills: ways of working",
    "2.B.1": "Social skills", "2.B.2": "Complex problem solving skills",
    "2.B.3": "Technical skills", "2.B.4": "Systems skills",
    "2.B.5": "Resource management skills",
    "2.C.1": "Business and management", "2.C.2": "Manufacturing and production",
    "2.C.3": "Engineering and technology", "2.C.4": "Mathematics and science",
    "2.C.5": "Health services", "2.C.6": "Education and training",
    "2.C.7": "Arts and humanities", "2.C.8": "Law and public safety",
    "2.C.9": "Communications", "2.C.10": "Transportation",
    "4.A.1": "Information input", "4.A.2": "Mental processes",
    "4.A.3": "Work output", "4.A.4": "Interacting with others",
    "4.C.1": "Interpersonal relationships", "4.C.2": "Physical work conditions",
    "4.C.3": "Structural job characteristics",
}

ONET_DIR = Path("data_raw/onet/db_31_0_text")   # replaced by the argument, if given


# --------------------------------------------------------------------------- #
# Reading and reporting
# --------------------------------------------------------------------------- #
def find(*stems):
    """The first O*NET file that exists among the given names, or None."""
    for stem in stems:
        path = ONET_DIR / f"{stem}.txt"
        if path.exists():
            return path
    return None


def read(path, columns=None):
    return pd.read_csv(path, sep="\t", dtype=str, encoding="utf-8",
                       usecols=(lambda c: c in columns) if columns else None)


def ratings(path, scale):
    """Occupation x element table of one scale of a rated O*NET file."""
    d = read(path)
    d = d[d["Scale ID"] == scale]
    return d.pivot_table(index=CODE, columns="Element Name", values="Data Value",
                         aggfunc=lambda v: float(v.iloc[0]))


def corr_table(axes, X):
    """Pearson correlation of each axis with each column of X, on the
    occupations present in both. Axes as rows."""
    common = axes.index.intersection(X.index)
    A, B = axes.loc[common, AX], X.loc[common].astype(float)
    return pd.DataFrame({c: [A[a].corr(B[c]) for a in AX] for c in B.columns}, index=AX)


def ends(r, label):
    """For each axis, the columns of a correlation table most associated with
    each end."""
    rows = []
    for a in AX:
        s = r.loc[a].sort_values()
        rows.append({"axis": a,
                     f"+ end ({label})": "; ".join(f"{k} {v:+.2f}" for k, v in s[::-1][:TOP].items()),
                     f"- end ({label})": "; ".join(f"{k} {v:+.2f}" for k, v in s[:TOP].items())})
    return pd.DataFrame(rows)


def strongest(r):
    """'R1 Realistic +0.81, R2 ...': the column most correlated with each axis."""
    return ", ".join(f"{a} {r.loc[a].abs().idxmax()} {r.loc[a, r.loc[a].abs().idxmax()]:+.2f}"
                     for a in AX)


class Report:
    """One line per section on the terminal; every table to the workbook,
    and to the terminal as well with -v."""

    def __init__(self, verbose):
        self.verbose, self.sheets = verbose, {}

    def line(self, section, text):
        print(f"[{section}] {text}")

    def table(self, name, df, index=False):
        self.sheets[name] = df.reset_index() if index else df
        if self.verbose:
            with pd.option_context("display.width", 250, "display.max_colwidth", 400,
                                   "display.max_rows", 500):
                print(f"\n--- {name}\n{df.round(3).to_string(index=index)}\n")

    def write(self):
        OUT.parent.mkdir(exist_ok=True)
        with pd.ExcelWriter(OUT) as writer:
            for name, df in self.sheets.items():
                df.to_excel(writer, sheet_name=name[:31], index=False)
        print(f"[ok] wrote {OUT} ({len(self.sheets)} sheets)")


# --------------------------------------------------------------------------- #
# 1. O*NET's own categories
# --------------------------------------------------------------------------- #
def element_categories():
    """Map (block prefix, element name) to its O*NET Content Model category."""
    ref = {}
    cmr = ONET_DIR / "Content Model Reference.txt"
    if cmr.exists():
        c = read(cmr, ["Element ID", "Element Name"])
        ref = dict(zip(c["Element ID"], c["Element Name"]))
    name2cat = {}
    for prefix, stems in BLOCK_FILES.items():
        for stem in stems:
            f = ONET_DIR / f"{stem}.txt"
            if not f.exists():
                continue
            d = read(f, ["Element ID", "Element Name"]).drop_duplicates()
            for eid, nm in zip(d["Element ID"], d["Element Name"]):
                cat_id = ".".join(eid.split(".")[:3])
                name2cat[(prefix, nm)] = ref.get(cat_id) or FALLBACK.get(cat_id) or cat_id
    return name2cat


def categories(rep, L, name2cat):
    cat = pd.Series({c: name2cat.get(tuple(c.split("__", 1))) for c in L.index}).dropna()
    if cat.empty:
        rep.line("1 categories", f"no Element IDs found in {ONET_DIR}; skipped")
        return
    Lc = L.loc[cat.index]
    means = Lc.groupby(cat).mean()
    means.insert(0, "n", cat.value_counts())
    rows = []
    for a in AX:
        order = Lc[a].sort_values()
        for pole, items in (("+", order.index[::-1][:POLE_ITEMS]), ("-", order.index[:POLE_ITEMS])):
            counts = cat[items].value_counts()
            rows.append({"axis": a, "pole": pole,
                         "categories of the strongest items":
                             "; ".join(f"{k} {v}" for k, v in counts.items())})
    rep.table("category_means", means.rename_axis("category"), index=True)
    rep.table("category_poles", pd.DataFrame(rows))
    top = {a: means[a].idxmax() for a in AX}
    rep.line("1 categories", f"{len(cat)} items in {len(means)} categories; highest mean loading: "
             + ", ".join(f"{a} {top[a]} {means.at[top[a], a]:+.2f}" for a in AX))


# --------------------------------------------------------------------------- #
# 2. Interests
# --------------------------------------------------------------------------- #
def interests(rep, axes):
    path = find("Career Interest Types", "Interests")
    if path is None:
        rep.line("2 interests", "Career Interest Types.txt not found; RIASEC skipped")
    else:
        X = ratings(path, "OI")
        X = X[[c for c in RIASEC if c in X.columns]]
        if len(X.columns) == len(RIASEC):
            X["people - things"] = (2 * X["Social"] + X["Artistic"] + X["Enterprising"]
                                    - 2 * X["Realistic"] - X["Investigative"] - X["Conventional"])
            X["data - ideas"] = 1.73 * (X["Enterprising"] + X["Conventional"]
                                        - X["Investigative"] - X["Artistic"])
        r = corr_table(axes, X)
        rep.table("riasec", r.rename_axis("axis"), index=True)
        rep.line("2 interests", "RIASEC, strongest type per axis: " + strongest(r[[c for c in RIASEC if c in r]]))

    path = find("Specific Interest Areas")
    if path is None:
        rep.line("2 interests", "Specific Interest Areas.txt not found; basic interests skipped")
    else:
        X = ratings(path, "OI")
        r = corr_table(axes, X)
        rep.table("interest_areas", r.T.rename_axis("interest area"), index=True)
        rep.table("interest_area_ends", ends(r, "interest areas"))
        rep.line("2 interests", f"{X.shape[1]} basic interest areas; strongest per axis: " + strongest(r))


# --------------------------------------------------------------------------- #
# 3. Job zones
# --------------------------------------------------------------------------- #
def job_zones(rep, axes):
    path = find("Job Zones")
    if path is None:
        rep.line("3 job zones", "Job Zones.txt not found; skipped")
        return
    z = pd.to_numeric(read(path).set_index(CODE)["Job Zone"], errors="coerce").rename("job zone")
    r = corr_table(axes, z.to_frame())
    common = axes.index.intersection(z.index)
    m = axes.loc[common, AX].groupby(z.loc[common]).median()
    m.insert(0, "n", z.loc[common].value_counts())
    rep.table("job_zone_corr", r.rename_axis("axis"), index=True)
    rep.table("job_zone_medians", m, index=True)
    rep.line("3 job zones", "correlation: " + ", ".join(f"{a} {r.at[a, 'job zone']:+.2f}" for a in AX))


# --------------------------------------------------------------------------- #
# 4. Software
# --------------------------------------------------------------------------- #
def software(rep, axes):
    path = find("Software Skills", "Technology Skills")
    if path is None:
        rep.line("4 software", "Software Skills.txt not found; skipped")
        return
    d = read(path)
    category = "Element Name" if "Element Name" in d.columns else "Commodity Title"
    d = d[[CODE, category]].dropna().drop_duplicates()
    count = d.groupby(CODE).size().rename("software categories used").reindex(axes.index).fillna(0)
    r = corr_table(axes, count.to_frame())
    use = pd.crosstab(d[CODE], d[category]).clip(upper=1)
    use = use.loc[:, use.sum() >= MIN_OCC].reindex(axes.index).fillna(0)
    ru = pd.DataFrame({c: [axes[a].corr(use[c]) for a in AX] for c in use.columns}, index=AX)
    rep.table("software_count", r.rename_axis("axis"), index=True)
    rep.table("software_ends", ends(ru, "software"))
    rep.line("4 software", f"{d[category].nunique()} categories; number used, correlation: "
             + ", ".join(f"{a} {r.at[a, 'software categories used']:+.2f}" for a in AX))


# --------------------------------------------------------------------------- #
# 5. Work styles
# --------------------------------------------------------------------------- #
def work_styles(rep, axes):
    path = find("Work Styles")
    if path is None:
        rep.line("5 work styles", "Work Styles.txt not found; skipped")
        return
    scales = read(path, ["Scale ID"])["Scale ID"].unique()
    scale = next((s for s in ("WI", "IM") if s in scales), None)
    if scale is None:
        rep.line("5 work styles", f"neither WI nor IM among the scales ({', '.join(scales)}); skipped")
        return
    X = ratings(path, scale)
    r = corr_table(axes, X)
    rep.table("work_styles", r.T.rename_axis("style"), index=True)
    rep.line("5 work styles", f"scale {scale}, {X.shape[1]} styles; strongest per axis: " + strongest(r))


# --------------------------------------------------------------------------- #
# 6. Clusters against the axes (Section 5.2)
# --------------------------------------------------------------------------- #
def mean_distances(X, groups):
    """Mean Euclidean distance between occupations in the same group and in
    different groups."""
    from scipy.spatial.distance import pdist, squareform
    D = squareform(pdist(X))
    same = groups[:, None] == groups[None, :]
    iu = np.triu_indices(len(X), k=1)
    return D[iu][same[iu]].mean(), D[iu][~same[iu]].mean()


def clusters(rep, L, S, m, name2cat):
    if CLUSTERS.exists():
        k2 = pd.read_csv(CLUSTERS, index_col=0)["cluster_k2"].reindex(S.index)
        targets = {"R1 - R2": S["R1"] - S["R2"], **{a: S[a] for a in AX}}
        part = pd.DataFrame({"compared with": list(targets),
                             "|r| with the k = 2 partition":
                                 [abs(np.corrcoef(k2, v)[0, 1]) for v in targets.values()]})
        sizes = k2.value_counts().sort_index()
        rep.table("k2_partition", part)
        rep.line("6 clusters", "k = 2 partition (sizes " + ", ".join(str(n) for n in sizes) + "), |r|: "
                 + ", ".join(f"{t} {v:.2f}" for t, v in part.iloc[:4].to_numpy()))
    else:
        rep.line("6 clusters", f"{CLUSTERS} not found; run cluster_onet.py first")

    if SKILL_CLUSTERS.exists():
        sk = pd.read_csv(SKILL_CLUSTERS)
        sk["category"] = [name2cat.get(tuple(c.split("__", 1)), "?") for c in sk["column"]]
        tab = pd.crosstab(sk["category"], sk["cluster"])
        rep.table("skill_clusters_by_category", tab.loc[tab.sum(axis=1).sort_values(ascending=False).index],
                  index=True)
        sk["R1 loading"] = L["R1"].reindex(sk["column"]).to_numpy()
        r1 = sk.groupby("cluster")["R1 loading"].agg(["size", "mean"])
        rep.table("skill_clusters_R1", r1, index=True)

    Z = StandardScaler().fit_transform(m[[c for c in m.columns if c.startswith(FEATURE_PREFIXES)]]
                                       .to_numpy(float))
    pca = PCA(random_state=0).fit(Z)
    T3 = pca.transform(Z)[:, :3]
    groups = np.array([s[:2] for s in m.index])
    n_groups = len(set(groups))
    within, between = mean_distances(T3, groups)
    means = pd.DataFrame(Z).groupby(groups).transform("mean").to_numpy()
    share = pca.explained_variance_ratio_
    soc = pd.DataFrame([{
        "SOC major groups": n_groups, "mean distance within": within,
        "mean distance between": between, "within / between": within / between,
        "silhouette, 3 components": silhouette_score(T3, groups),
        "silhouette, all features": silhouette_score(Z, groups),
        "feature variance reconstructed, 3 components": share[:3].sum(),
        "feature variance reconstructed, major groups": (means ** 2).sum() / (Z ** 2).sum(),
        f"feature variance reconstructed, {n_groups - 1} components": share[:n_groups - 1].sum()}])
    rep.table("soc_major_groups", soc)


# --------------------------------------------------------------------------- #
def main():
    global ONET_DIR
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("onet_dir", nargs="?", default=str(ONET_DIR), help="O*NET text folder")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every table")
    args = ap.parse_args()
    ONET_DIR = Path(args.onet_dir)
    if not ONET_DIR.exists():
        print(f"[warn] {ONET_DIR} not found: only the cluster checks will run")

    axes = pd.read_csv(AXES, index_col=0)
    S, axes = axes, axes[AX]
    L = pd.read_csv(LOADINGS, index_col=0)[AX]
    m = pd.read_csv(MASTER, index_col="onet_soc").reindex(S.index)
    print(f"[data] {len(axes)} occupations, {len(L)} feature columns; O*NET files from {ONET_DIR}")

    rep = Report(args.verbose)
    name2cat = element_categories()
    categories(rep, L, name2cat)
    interests(rep, axes)
    job_zones(rep, axes)
    software(rep, axes)
    work_styles(rep, axes)
    clusters(rep, L, S, m, name2cat)
    rep.write()


if __name__ == "__main__":
    main()