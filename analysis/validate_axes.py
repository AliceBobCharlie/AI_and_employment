#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_axes.py -- list the occupations deepest in two corners of the R1-R3
space, and correlate the six rotated axes with O*NET descriptors that are
not in the feature matrix and with O*NET's own categories of the feature
columns.

Input:  output/pca_rotated/occupation_coordinates.csv, output/pca_rotated/axis_loadings.csv,
        output/skillscape/skill_clusters.csv (optional), and the O*NET text
        files in the folder given as the first argument (default ONET_DIR)
Output: one CSV per table in output/validate_axes/

main() reads the R1-R6 scores (AXES) and loadings (LOADINGS) and runs the
sections in order. Section 0 needs only AXES. If the O*NET folder does not
exist, sections 1 to 5 are skipped with a message; a section whose O*NET
file is missing is skipped likewise. Report.table() writes each table; Report.done() prints one status
line per section.

0. corners() takes the occupations positive on all of CORNER_AXES and those
   negative on all of them. An occupation's depth in a corner is the smallest
   of its CORNER_AXES scores taken with the corner's sign. corners.csv lists
   the TOP_CORNER deepest occupations of each corner, with their scores, their
   depth and the number of occupations in the corner.
1. element_categories() maps each feature column to its Content Model
   category through the Element IDs of the files in BLOCK_FILES, named from
   Content Model Reference.txt or FALLBACK. categories() gives the mean
   loading of each category on each axis and the categories of the
   POLE_ITEMS strongest items at each end of each axis. skill_clusters()
   crosses the two skill-cluster labels of skill_clusters.csv with the
   categories and gives the mean R1 loading of each cluster.
2. interests() correlates the axes with the six RIASEC scores (scale OI of
   Career Interest Types.txt or Interests.txt), with Prediger's
   people - things = 2S + A + E - 2R - I - C and
   data - ideas = 1.73 (E + C - I - A), and with the basic interest areas
   (Specific Interest Areas.txt); ends() lists the TOP areas at each end of
   each axis.
3. job_zones() correlates the axes with the Job Zone and gives the median
   score of each axis by zone.
4. software() correlates the axes with the number of software categories an
   occupation uses (Software Skills.txt or Technology Skills.txt) and, through
   ends(), lists the categories used by at least MIN_OCC occupations most
   associated with each end.
5. work_styles() correlates the axes with the work styles on scale WI (or IM
   where WI is absent) of Work Styles.txt.

corr_table() gives the Pearson correlation of each axis with each column, on
the occupations present in both.

Run from the project root:  python analysis/validate_axes.py [O*NET folder]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from pca_rotated import AXES, LOADINGS, OUTDIR

OUT = OUTDIR / "validate_axes"
SKILL_CLUSTERS = OUTDIR / "skillscape" / "skill_clusters.csv"

AX = ["R1", "R2", "R3", "R4", "R5", "R6"]
CODE = "O*NET-SOC Code"
RIASEC = ["Realistic", "Investigative", "Artistic", "Social", "Enterprising", "Conventional"]
POLE_ITEMS = 15          # items per pole in the category composition
CORNER_AXES = ["R1", "R2", "R3"]
TOP_CORNER = 10          # occupations listed per corner
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

ONET_DIR = Path("data/raw/onet/db_31_0_text")   # replaced by the argument, if given


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


class Report:
    """Writes each table to OUT as a CSV and prints one status line per section."""

    def __init__(self):
        self.written = []

    def table(self, name, df, index=False):
        OUT.mkdir(parents=True, exist_ok=True)
        (df.reset_index() if index else df).to_csv(OUT / f"{name}.csv", index=False)
        self.written.append(name)

    def skip(self, section, reason):
        print(f"[skip] {section}: {reason}")

    def done(self, section):
        if self.written:
            print(f"[ok] {section}: " + ", ".join(f"{n}.csv" for n in self.written))
        self.written = []


# --------------------------------------------------------------------------- #
# 0. Corners
# --------------------------------------------------------------------------- #
def corners(rep, scores):
    rows = []
    for label, sign in (("positive on all", 1), ("negative on all", -1)):
        signed = scores[CORNER_AXES] * sign
        inside = scores[(signed > 0).all(axis=1)].copy()
        inside["depth"] = signed.loc[inside.index].min(axis=1)
        inside = inside.sort_values("depth", ascending=False)
        rows += [{"corner": f"{label} of {', '.join(CORNER_AXES)}", "occupations in corner": len(inside),
                  "rank": rank, "occupation": r["title"], **{a: r[a] for a in CORNER_AXES}, "depth": r["depth"]}
                 for rank, (_, r) in enumerate(inside.head(TOP_CORNER).iterrows(), 1)]
    rep.table("corners", pd.DataFrame(rows))


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
        rep.skip("1 categories", f"no Element IDs found in {ONET_DIR}")
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


# --------------------------------------------------------------------------- #
# 2. Interests
# --------------------------------------------------------------------------- #
def interests(rep, axes):
    path = find("Career Interest Types", "Interests")
    if path is None:
        rep.skip("2 interests", "Career Interest Types.txt not found")
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

    path = find("Specific Interest Areas")
    if path is None:
        rep.skip("2 interests", "Specific Interest Areas.txt not found")
    else:
        X = ratings(path, "OI")
        r = corr_table(axes, X)
        rep.table("interest_areas", r.T.rename_axis("interest area"), index=True)
        rep.table("interest_area_ends", ends(r, "interest areas"))


# --------------------------------------------------------------------------- #
# 3. Job zones
# --------------------------------------------------------------------------- #
def job_zones(rep, axes):
    path = find("Job Zones")
    if path is None:
        rep.skip("3 job zones", "Job Zones.txt not found")
        return
    z = pd.to_numeric(read(path).set_index(CODE)["Job Zone"], errors="coerce").rename("job zone")
    r = corr_table(axes, z.to_frame())
    common = axes.index.intersection(z.index)
    m = axes.loc[common, AX].groupby(z.loc[common]).median()
    m.insert(0, "n", z.loc[common].value_counts())
    rep.table("job_zone_corr", r.rename_axis("axis"), index=True)
    rep.table("job_zone_medians", m, index=True)


# --------------------------------------------------------------------------- #
# 4. Software
# --------------------------------------------------------------------------- #
def software(rep, axes):
    path = find("Software Skills", "Technology Skills")
    if path is None:
        rep.skip("4 software", "Software Skills.txt not found")
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


# --------------------------------------------------------------------------- #
# 5. Work styles
# --------------------------------------------------------------------------- #
def work_styles(rep, axes):
    path = find("Work Styles")
    if path is None:
        rep.skip("5 work styles", "Work Styles.txt not found")
        return
    scales = read(path, ["Scale ID"])["Scale ID"].unique()
    scale = next((s for s in ("WI", "IM") if s in scales), None)
    if scale is None:
        rep.skip("5 work styles", f"neither WI nor IM among the scales ({', '.join(scales)})")
        return
    X = ratings(path, scale)
    r = corr_table(axes, X)
    rep.table("work_styles", r.T.rename_axis("style"), index=True)


# --------------------------------------------------------------------------- #
# 1b. Skill clusters against the categories
# --------------------------------------------------------------------------- #
def skill_clusters(rep, L, name2cat):
    if not SKILL_CLUSTERS.exists():
        rep.skip("1 categories", f"{SKILL_CLUSTERS} not found; skill clusters skipped")
        return
    sk = pd.read_csv(SKILL_CLUSTERS)
    sk["category"] = [name2cat.get(tuple(c.split("__", 1)), "?") for c in sk["column"]]
    sk["R1 loading"] = L["R1"].reindex(sk["column"]).to_numpy()
    for label in [c for c in sk.columns if c.startswith("cluster")]:
        name = "skill_clusters_" + label.split(", ")[-1].split(" ")[0].lower()
        tab = pd.crosstab(sk["category"], sk[label])
        rep.table(f"{name}_by_category", tab.loc[tab.sum(axis=1).sort_values(ascending=False).index],
                  index=True)
        rep.table(f"{name}_R1", sk.groupby(label)["R1 loading"].agg(["size", "mean"]), index=True)


# --------------------------------------------------------------------------- #
def main():
    global ONET_DIR
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("onet_dir", nargs="?", default=str(ONET_DIR), help="O*NET text folder")
    ONET_DIR = Path(ap.parse_args().onet_dir)
    try:
        scores = pd.read_csv(AXES, index_col=0)
        L = pd.read_csv(LOADINGS, index_col=0)[AX]
    except OSError as err:
        sys.exit(f"[fail] {err}")
    axes = scores[AX]
    print(f"[ok] read {len(axes)} occupations, {len(L)} feature columns")

    rep = Report()
    corners(rep, scores)
    rep.done("0 corners")
    if not ONET_DIR.exists():
        print(f"[skip] 1-5: {ONET_DIR} not found")
        return
    print(f"[ok] O*NET files from {ONET_DIR}")
    name2cat = element_categories()
    categories(rep, L, name2cat)
    if name2cat:
        skill_clusters(rep, L, name2cat)
    rep.done("1 categories")
    for section, run in (("2 interests", interests), ("3 job zones", job_zones),
                         ("4 software", software), ("5 work styles", work_styles)):
        run(rep, axes)
        rep.done(section)


if __name__ == "__main__":
    main()