#!/usr/bin/env python3
"""
resampling_stability.py -- resample the occupations, decompose every resample,
and tabulate how stable the components, blocks of components and rotated axes
are.

Input:  output/master_clean.csv, read by load_and_prep() from pca_rotated.py
        (the 220 Importance, CX and CTP columns, z-scored)
Output: output/resampled_spectra.npz    bootstrap and split-half decompositions
        output/resampling_stability/    seven CSV tables
        assets/subspace_stability.png   Figure 5.1

run_experiment() runs steps 1 to 4 on a z-scored matrix; main() calls it on
this one. Every resample and every half is z-scored on its own and decomposed
by decompose(): eigen(), or, if column weights w are passed, eigen() after
each column is multiplied by its weight.

1. parallel_analysis() decomposes N (200) copies of the data with every column
   permuted. k_max is the number of leading eigenvalues above the 95th
   percentile of the permuted ones; the later steps use PC1 to PC k_max only.
   The eigenvalues go to eigenvalues.csv.

2. draw_resamples() decomposes N bootstrap resamples, N split-halves, and N
   split-halves of column-permuted data (the null), keeping each one's
   eigenvalues and first k_max eigenvectors. The bootstrap and split-half
   decompositions, with the rows drawn, are saved to resampled_spectra.npz.
   The null ones are discarded after step 3.

3. pairs() pairs the two halves of each split, and each bootstrap resample
   with the full sample. The pairs are compared by principal-angle cosines and
   congruence:
     component_table()     components.csv: each component. The leading run
                           with p05 >= STABLE_THRESH in both split and
                           bootstrap is the stable components (at least two).
     nested_table()        nested_subspaces.csv: PC1 to PCk for k = 1 to
                           k_max, with the null and the eigenvalue gaps.
     block_table()         blocks.csv: candidate blocks of 2 to MAX_BLOCK
                           components starting after the stable components.
                           The largest with split RMS cosine p05 >=
                           STABLE_THRESH is the stable subspace. Also every
                           smaller block inside the stable subspace (inside the
                           largest candidate if none is found), and, if a
                           stable subspace is found, every block of 2 to
                           MAX_BLOCK components after it, up to k_max.
     looser_table()        looser_criteria.csv: medians for the components
                           and blocks after the stable subspace.
     rotated_axes_table()  rotated_axes.csv: the stable components and the
                           stable subspace, each rotated by rotate_group() in
                           every resample and paired by match_axes().

4. dimensions_table() writes dimensions.csv: the stable components and stable
   subspace found, against STABLE_COMPONENTS and STABLE_SUBSPACE in
   pca_rotated.py. main() warns if they differ. plot_subspaces() draws
   Figure 5.1 from nested_subspaces.csv.

SEED seeds three generators: one for the permuted copies, one for the
bootstrap resamples, and one shared by the split-halves and the null.

load_spectra() reads resampled_spectra.npz back as a dict of arrays.
"""

import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from pca_rotated import (CSV_FLOAT, OUTDIR, SEED, STABLE_COMPONENTS, STABLE_SUBSPACE,
                         axis_names, congruence, eigen, load_and_prep, match_axes,
                         rotate_group, standardize, subspace_cosines)

SPECTRA = OUTDIR / "resampled_spectra.npz"
TABLES = OUTDIR / "resampling_stability"
FIGURE = Path("assets/subspace_stability.png")

N = 200                 # resamples of each kind
STABLE_THRESH = 0.90    # p05 a component, block or axis needs to count as stable
N_EIGEN_ROWS = 30       # eigenvalues tabulated and tested by parallel analysis
MAX_BLOCK = 4           # largest block of components tested


# --------------------------------------------------------------------------- #
# Resamples
# --------------------------------------------------------------------------- #
def parallel_analysis(Xz):
    """Eigenvalues of N column-permuted copies of Xz, each centred."""
    n = len(Xz)
    rng = np.random.default_rng(SEED)
    out = []
    for _ in range(N):
        Xp = rng.permuted(Xz, axis=0)
        Xp = Xp - Xp.mean(axis=0)
        out.append(np.linalg.svd(Xp, compute_uv=False) ** 2 / (n - 1))
    return np.stack(out)


def decompose(X, w=None):
    """eigen() of X; with column weights w, X is z-scored and multiplied by w first."""
    if w is None:
        return eigen(X)
    Z = standardize(X) * w
    _, s, vt = np.linalg.svd(Z, full_matrices=False)
    return s ** 2 / (len(Z) - 1), vt.T


def leading(X, k_max, w=None):
    ev, V = decompose(X, w)
    return ev, V[:, :k_max]


def halves(X, perm, k_max, w=None):
    n = len(X)
    return leading(X[perm[: n // 2]], k_max, w), leading(X[perm[n // 2:]], k_max, w)


def draw_resamples(Xz, k_max, w=None):
    """Bootstrap and split decompositions, as saved to SPECTRA, and the null
    split decompositions as a list of pairs."""
    n = len(Xz)
    rng = np.random.default_rng(SEED)
    boot_idx = np.stack([rng.integers(0, n, n) for _ in range(N)])
    boot = [leading(Xz[i], k_max, w) for i in boot_idx]

    rng = np.random.default_rng(SEED)
    split_perm = np.stack([rng.permutation(n) for _ in range(N)])
    split = [halves(Xz, p, k_max, w) for p in split_perm]
    null = []
    for _ in range(N):
        Xp = np.column_stack([rng.permutation(c) for c in Xz.T])
        null.append(halves(Xp, rng.permutation(n), k_max, w))

    full_ev, full_V = leading(Xz, k_max, w)
    s = {"full_ev": full_ev, "full_V": full_V,
         "bootstrap_index": boot_idx,
         "bootstrap_ev": np.stack([b[0] for b in boot]),
         "bootstrap_V": np.stack([b[1] for b in boot]),
         "split_perm": split_perm,
         "split_ev": np.stack([[h[0] for h in pair] for pair in split]),
         "split_V": np.stack([[h[1] for h in pair] for pair in split])}
    return s, null


def load_spectra():
    """The arrays saved to SPECTRA by main(); exits if the file is missing."""
    if not SPECTRA.exists():
        sys.exit(f"[fail] {SPECTRA} not found; run resampling_stability.py first")
    with np.load(SPECTRA) as z:
        return {k: z[k] for k in z.files}


def pairs(s, kind):
    """((ev_a, V_a), (ev_b, V_b)) per resample: the two halves of each split,
    or the full sample and each bootstrap resample."""
    if kind == "bootstrap":
        full = (s["full_ev"], s["full_V"])
        return [(full, (e, V)) for e, V in zip(s["bootstrap_ev"], s["bootstrap_V"])]
    ev, V = s["split_ev"], s["split_V"]
    return [((ev[i, 0], V[i, 0]), (ev[i, 1], V[i, 1])) for i in range(len(ev))]


# --------------------------------------------------------------------------- #
# Agreement between two decompositions
# --------------------------------------------------------------------------- #
def pair_stats(Va, Vb, k_max):
    """Smallest and mean squared principal-angle cosine of the leading-k
    subspaces, k = 1..k_max, and each of Va's components' best-match
    congruence among Vb's."""
    min_cos, msq_cos = np.empty(k_max), np.empty(k_max)
    for k in range(1, k_max + 1):
        c = subspace_cosines(Va[:, :k], Vb[:, :k])
        min_cos[k - 1], msq_cos[k - 1] = c.min(), (c ** 2).mean()
    single = np.abs(Va[:, :k_max].T @ Vb[:, :k_max]).max(axis=1)
    return min_cos, msq_cos, single


def stats_over(sample_pairs, k_max):
    """pair_stats() over all pairs: (min_cos, msq_cos, single), each (resamples, k_max)."""
    out = [pair_stats(a[1], b[1], k_max) for a, b in sample_pairs]
    return tuple(np.stack([o[i] for o in out]) for i in range(3))


def block_cosines(sample_pairs, block):
    a, z = block
    return np.stack([subspace_cosines(A[1][:, a:z], B[1][:, a:z]) for A, B in sample_pairs])


def rotation_congruence(sample_pairs, ref, idx):
    """Per pair: rotate_group() on idx in both decompositions, match_axes() to
    ref, then the congruence of each axis between the two. A decomposition
    that recurs across pairs (the full sample) is rotated once."""
    cache = {}

    def rotated(e, V):
        if id(V) not in cache:
            cache[id(V)] = match_axes(ref, rotate_group(V, e, idx)[0])
        return cache[id(V)]

    out = []
    for (ea, Va), (eb, Vb) in sample_pairs:
        A, B = rotated(ea, Va), rotated(eb, Vb)
        out.append([congruence(A[:, j], B[:, j]) for j in range(ref.shape[1])])
    return np.array(out)


def p05(x):
    return np.percentile(x, 5, axis=0)


def median(x):
    return np.median(x, axis=0)


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def eigenvalue_table(full_ev, permuted_ev, n_rows=N_EIGEN_ROWS):
    q95 = np.percentile(permuted_ev[:, :n_rows], 95, axis=0)
    share = full_ev / full_ev.sum()
    return pd.DataFrame({"component": range(1, n_rows + 1),
                         "eigenvalue": full_ev[:n_rows],
                         "% variance": 100 * share[:n_rows],
                         "cumulative %": 100 * np.cumsum(share)[:n_rows],
                         "null p95": q95,
                         "above null": full_ev[:n_rows] > q95})


def component_table(split, boot, k_max):
    t = pd.DataFrame({"component": [f"PC{j + 1}" for j in range(k_max)],
                      "bootstrap p05": p05(boot[2]),
                      "bootstrap median": median(boot[2]),
                      "split-half p05": p05(split[2]),
                      "split-half median": median(split[2])})
    t["passes"] = (t["bootstrap p05"] >= STABLE_THRESH) & (t["split-half p05"] >= STABLE_THRESH)
    return t


def nested_table(s, split, boot, null, null_ev, k_max):
    n = s["bootstrap_index"].shape[1]
    ev = s["full_ev"]
    half_ev = s["split_ev"].reshape(-1, s["split_ev"].shape[-1])
    half_error = np.sqrt(2 / (n / 2))
    rows = []
    for j in range(k_max):
        rows.append({
            "k": j + 1,
            "eigenvalue k (full sample)": ev[j],
            "gap k to k+1 / half-sample error": (ev[j] - ev[j + 1]) / (ev[j] * half_error),
            "min cos p05": np.percentile(split[0][:, j], 5),
            "min cos median": np.median(split[0][:, j]),
            "mean sq cos p05": np.percentile(split[1][:, j], 5),
            "mean sq cos median": np.median(split[1][:, j]),
            "null min cos median": np.median(null[0][:, j]),
            "null min cos p95": np.percentile(null[0][:, j], 95),
            "null single congruence p95": np.percentile(null[2][:, j], 95),
            "null mean sq cos median": np.median(null[1][:, j]),
            "null mean sq cos p95": np.percentile(null[1][:, j], 95),
            "single component k: congruence p05": np.percentile(split[2][:, j], 5),
            "single component k: congruence median": np.median(split[2][:, j]),
            "half-sample eigenvalue k (mean)": half_ev[:, j].mean(),
            "half-sample null eigenvalue k (p95)": np.percentile(null_ev[:, j], 95),
            "bootstrap min cos p05": np.percentile(boot[0][:, j], 5),
            "bootstrap min cos median": np.median(boot[0][:, j]),
            "bootstrap mean sq cos p05": np.percentile(boot[1][:, j], 5),
            "bootstrap mean sq cos median": np.median(boot[1][:, j]),
        })
    t = pd.DataFrame(rows)
    t["half-sample eigenvalue above noise floor"] = (
        t["half-sample eigenvalue k (mean)"] > t["half-sample null eigenvalue k (p95)"])
    return t


def block_row(block, group, split_pairs, boot_pairs):
    a, z = block
    row = {"group": group, "components": f"{a + 1}-{z}" if z - a > 1 else f"{z}", "size": z - a}
    for prefix, sample_pairs in (("", split_pairs), ("bootstrap ", boot_pairs)):
        cos = block_cosines(sample_pairs, block)
        rms = np.sqrt((cos ** 2).mean(axis=1))
        row.update({f"{prefix}RMS cosine p05": np.percentile(rms, 5),
                    f"{prefix}RMS cosine median": np.median(rms),
                    f"{prefix}smallest cosine p05": np.percentile(cos.min(axis=1), 5),
                    f"{prefix}smallest cosine median": np.median(cos.min(axis=1))})
    row["RMS cosine p05 >= 0.90"] = row["RMS cosine p05"] >= STABLE_THRESH
    return row


def candidate_rotation(s, split_pairs, boot_pairs, idx):
    """Split and bootstrap congruence of a candidate's rotated axes, in variance order."""
    ref = rotate_group(s["full_V"], s["full_ev"], idx)[0]
    row = {}
    for prefix, sample_pairs in (("", split_pairs), ("bootstrap ", boot_pairs)):
        cong = rotation_congruence(sample_pairs, ref, idx)
        row.update({f"{prefix}rotated axis {j + 1} p05": v for j, v in enumerate(p05(cong))})
        row.update({f"{prefix}rotated axis {j + 1} median": v for j, v in enumerate(median(cong))})
        if not prefix:
            row["all rotated axes p05 >= 0.90"] = bool((p05(cong) >= STABLE_THRESH).all())
    return row


def block_table(s, split_pairs, boot_pairs, n_comp, k_max):
    """Rows for every block tested; returns the table and sub_end, the last
    component of the stable subspace (n_comp if there is none)."""
    last = min(n_comp + MAX_BLOCK, k_max)
    rows, sub_end = [block_row((0, n_comp), "stable components", split_pairs, boot_pairs)], n_comp
    for e in range(n_comp + 2, last + 1):
        row = block_row((n_comp, e), "stable subspace candidate", split_pairs, boot_pairs)
        row.update(candidate_rotation(s, split_pairs, boot_pairs, list(range(n_comp, e))))
        if row["RMS cosine p05 >= 0.90"]:
            sub_end = e
        rows.append(row)

    inner_end = sub_end if sub_end > n_comp else last
    inner = [(a, z) for a in range(n_comp, inner_end) for z in range(a + 1, inner_end + 1)
             if (a, z) != (n_comp, inner_end)]
    group = "inside stable subspace" if sub_end > n_comp else "inside candidates"
    rows += [block_row(b, group, split_pairs, boot_pairs) for b in inner]

    if sub_end > n_comp:
        later = [(a, z) for a in range(sub_end, k_max)
                 for z in range(a + 2, min(a + MAX_BLOCK, k_max) + 1)]
        rows += [block_row(b, "after stable subspace", split_pairs, boot_pairs) for b in later]
    return pd.DataFrame(rows), sub_end


def looser_table(nested, blocks, n_comp, sub_end):
    rows = []
    singles = nested.loc[nested["k"] > sub_end]
    best = singles.loc[singles["single component k: congruence median"].idxmax()]
    rows.append({"quantity": "single component after the stable subspace: highest median congruence",
                 "components": str(int(best["k"])),
                 "value": best["single component k: congruence median"]})
    after = blocks.loc[blocks["group"] == "after stable subspace"]
    if len(after):
        for col in ("smallest cosine median", "RMS cosine median"):
            best = after.loc[after[col].idxmax()]
            rows.append({"quantity": f"block after the stable subspace: highest {col}",
                         "components": best["components"], "value": best[col]})
    sub = blocks.loc[(blocks["group"] == "stable subspace candidate")
                     & (blocks["components"] == f"{n_comp + 1}-{sub_end}")]
    for col in ("smallest cosine median", "RMS cosine median"):
        if len(sub):
            rows.append({"quantity": f"stable subspace: {col}", "components": f"{n_comp + 1}-{sub_end}",
                         "value": sub.iloc[0][col]})
    return pd.DataFrame(rows)


def rotated_axes_table(split_pairs, boot_pairs, groups, rotated, names=axis_names):
    """rotated holds (L, R) of each group of groups on the full sample;
    names(idx) gives the axis names of a group."""
    rows = []
    labels = ["stable components", "stable subspace"]
    for label, idx, (L, _) in zip(labels, groups, rotated):
        split = rotation_congruence(split_pairs, L, idx)
        boot = rotation_congruence(boot_pairs, L, idx)
        for j, name in enumerate(names(idx)):
            rows.append({"axis": name, "group": label,
                         "variance share": (L[:, j] ** 2).sum() / L.shape[0],
                         "split-half p05": p05(split)[j], "split-half median": median(split)[j],
                         "bootstrap p05": p05(boot)[j], "bootstrap median": median(boot)[j]})
    return pd.DataFrame(rows)


def found_groups(n_comp, sub_end):
    """The stable components and, if there is one, the stable subspace, as index lists."""
    return [list(range(n_comp))] + ([list(range(n_comp, sub_end))] if sub_end > n_comp else [])


def dimensions_table(n_comp, sub_end):
    def label(idx):
        return f"PC{idx[0] + 1}-{idx[-1] + 1}" if idx else "none"

    found = {"stable components": list(range(n_comp)),
             "stable subspace": list(range(n_comp, sub_end))}
    fixed = {"stable components": STABLE_COMPONENTS, "stable subspace": STABLE_SUBSPACE}
    return pd.DataFrame([{"dimensions": d, "found": label(found[d]),
                          "pca_rotated.py": label(fixed[d]), "match": found[d] == fixed[d]}
                         for d in found])


# --------------------------------------------------------------------------- #
# Figure 5.1
# --------------------------------------------------------------------------- #
def plot_subspaces(t, n_comp, sub_end, k_max, figure=FIGURE):
    """Split-half p05 of the smallest cosine of the leading-k subspaces, with the
    null p95 dashed and the ends of the stable components and subspace marked."""
    k = t["k"].to_numpy()
    sub = t["min cos p05"].to_numpy()
    sub_null = t["null min cos p95"].to_numpy()
    ink, secondary, muted, grid, surface = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
    blue = "#2a78d6"

    fig, ax = plt.subplots(figsize=(8.0, 5.0), facecolor=surface)
    ax.set_facecolor(surface)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(muted)
    ax.tick_params(colors=secondary, labelsize=9)
    ax.grid(axis="y", color=grid, linewidth=0.8)
    ax.set_axisbelow(True)

    edges = [0, n_comp] + ([sub_end] if sub_end > n_comp else [])
    for x in edges[1:]:
        ax.axvline(x + 0.5, color=muted, linewidth=0.8, linestyle=(0, (2, 3)))
    for label, (a, b) in zip(["stable one by one", "stable as a subspace"],
                             zip(edges[:-1], edges[1:])):
        ax.text((a + b + 1) / 2, 1.035, label, color=secondary, fontsize=8, ha="center")
    ax.axhline(STABLE_THRESH, color=muted, linewidth=1, linestyle=":")
    ax.text(k_max + 0.4, STABLE_THRESH + 0.012, f"{STABLE_THRESH:.2f}", color=secondary,
            fontsize=8, ha="right", va="bottom")

    ax.plot(k, sub, color=blue, linewidth=2, marker="o", markersize=7,
            markeredgecolor=surface, markeredgewidth=1.5,
            label="leading-k subspace, smallest cosine (5th percentile)")
    ax.plot(k, sub_null, color=blue, linewidth=1.2, linestyle=(0, (4, 3)), alpha=0.7,
            label="same, column-permuted data (95th percentile)")
    if sub_end > n_comp:
        ax.annotate(f"{sub[sub_end - 1]:.2f}", (sub_end, sub[sub_end - 1]), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=8, color=ink)
        ax.annotate(f"{sub[sub_end - 2]:.2f}", (sub_end - 1, sub[sub_end - 2]), xytext=(9, -3),
                    textcoords="offset points", ha="left", va="center", fontsize=8, color=ink)

    ax.set_xlim(0.5, k_max + 0.5)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(range(1, k_max + 1))
    ax.set_xlabel("k", color=secondary, fontsize=9)
    ax.set_ylabel(f"5th percentile over {N} split-halves", color=secondary, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=ink, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.tight_layout()
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=200, facecolor=surface)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def run_experiment(Xz, feats, tables_dir, spectra, figure, w=None,
                   n_eigen=N_EIGEN_ROWS, names=axis_names):
    """Steps 1-4 and the figure on the z-scored matrix Xz, with optional column
    weights w; names(idx) names the rotated axes. Writes the tables to
    tables_dir; returns k_max, n_comp, sub_end, the saved decompositions and
    the tables."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    full_ev, _ = decompose(Xz, w)
    permuted = parallel_analysis(Xz if w is None else Xz * w)
    eig = eigenvalue_table(full_ev, permuted, n_eigen)
    k_max = int(eig["above null"].cumprod().sum())
    if k_max == n_eigen:
        print(f"[warn] all {n_eigen} tested eigenvalues exceed the null; raise n_eigen")
    print(f"[ok] parallel analysis: k_max = {k_max}")

    s, null_pairs = draw_resamples(Xz, k_max, w)
    np.savez(spectra, features=np.array(feats), seed=SEED, k_max=k_max, **s)
    print(f"[ok] decomposed {N} bootstrap, {N} split-half and {N} null resamples; wrote {spectra}")

    split_pairs, boot_pairs = pairs(s, "split"), pairs(s, "bootstrap")
    split, boot = stats_over(split_pairs, k_max), stats_over(boot_pairs, k_max)
    null = stats_over(null_pairs, k_max)
    null_ev = np.stack([h[0] for pair in null_pairs for h in pair])
    del null_pairs

    comp = component_table(split, boot, k_max)
    n_comp = max(int(comp["passes"].cumprod().sum()), 2)
    nested = nested_table(s, split, boot, null, null_ev, k_max)
    blocks, sub_end = block_table(s, split_pairs, boot_pairs, n_comp, k_max)
    looser = looser_table(nested, blocks, n_comp, sub_end)
    groups = found_groups(n_comp, sub_end)
    rotated = [rotate_group(s["full_V"], s["full_ev"], idx) for idx in groups]
    rot = rotated_axes_table(split_pairs, boot_pairs, groups, rotated, names)
    dims = dimensions_table(n_comp, sub_end)

    tables = {"eigenvalues": eig, "components": comp, "nested_subspaces": nested,
              "blocks": blocks, "looser_criteria": looser, "rotated_axes": rot,
              "dimensions": dims}
    for name, table in tables.items():
        table.to_csv(tables_dir / f"{name}.csv", index=False, float_format=CSV_FLOAT)
    plot_subspaces(nested, n_comp, sub_end, k_max, figure)
    return {"k_max": k_max, "n_comp": n_comp, "sub_end": sub_end, "spectra": s, "tables": tables}


def main():
    Xz, feats, _ = load_and_prep()
    out = run_experiment(Xz, feats, TABLES, SPECTRA, FIGURE)
    dims = out["tables"]["dimensions"]
    found = ", ".join(f"{r.dimensions} {r.found}" for r in dims.itertuples())
    if dims["match"].all():
        print(f"[ok] {found}, as in pca_rotated.py")
    else:
        print(f"[warn] {found}; pca_rotated.py differs (see dimensions.csv)")
    print(f"[ok] wrote {len(out['tables'])} tables to {TABLES}/ and {FIGURE}")


if __name__ == "__main__":
    main()