#!/usr/bin/env python3
"""
resampling_stability.py -- resampling experiment on the dimensions of the
feature matrix.

Reads Importance, CX and CTP from output/master_clean.csv and z-scores its 220 columns. Draws
N_SPLIT_SUB (200) resamples of each of four kinds:
  permuted   every column permuted on its own
  bootstrap  as many occupations as the data, drawn with replacement
  split      the occupations split into two random halves
  null       every column permuted, then split into two halves
Decomposes each resample, and each half, by SVD after z-scoring it on its own
(the permuted ones are only centred). Saves every eigenvalue and the first
N_VEC (20) eigenvectors of each, with the rows each resample drew, to
output/resampled_spectra.npz.

From these it computes, and writes to output/resampling_stability/:
  eigenvalues.csv       the full sample's first 30 eigenvalues and % variance,
                        with the 95th percentile of the permuted eigenvalues
  components.csv        for PC1 to PC{K_MAX}, the congruence of each component
                        with its best match: half with half, and bootstrap
                        resample with full sample (p05 and median). The leading
                        components with p05 >= STABLE_THRESH on both are the
                        stable components (at least two).
  nested_subspaces.csv  for k = 1 to K_MAX, the principal-angle cosines between
                        the leading-k subspaces of the two halves, of each
                        bootstrap resample and the full sample, and of the two
                        null halves; each eigenvalue's gap to the next over its
                        half-sample error; the halves' eigenvalues
  blocks.csv            the same cosines (root mean square and smallest) for
                        blocks of components: the stable components; candidate
                        blocks of two or more components after them, ending at
                        or before component N_PC, with the congruence of their
                        varimax axes; the smaller blocks inside the chosen
                        candidate; and every block of two to MAX_LATER_BLOCK
                        components after it, up to K_MAX. The largest
                        candidate whose split-half RMS cosine has
                        p05 >= STABLE_THRESH is the stable subspace.
  looser_criteria.csv   medians instead of 5th percentiles, for the single
                        components and blocks after the stable subspace and for
                        the subspace itself
  rotated_axes.csv      for R1-R6, rotated as in pca_rotated.py: each axis's
                        share of the variance, and its congruence between the
                        two halves and between each bootstrap resample and the
                        full sample, every resample rotated from scratch and
                        its axes paired with the full sample's
  dimensions.csv        the stable components and stable subspace found here,
                        against STABLE_COMPONENTS and STABLE_SUBSPACE in
                        pca_rotated.py; a mismatch is also printed as a warning
Then draws Figure 5.1 from nested_subspaces.csv to assets/subspace_stability.png.

The permuted and the bootstrap resamples each come from a generator seeded
with SEED; the split and null resamples share a third.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from pca_rotated import (CSV_FLOAT, K_MAX, N_SPLIT_SUB, OUTDIR, ROTATION_GROUPS, SEED,
                         STABLE_COMPONENTS, STABLE_SUBSPACE, STABLE_THRESH, axis_names,
                         congruence, eigen, load_and_prep, match_axes, rotate_group,
                         rotate_groups, subspace_cosines)

SPECTRA = OUTDIR / "resampled_spectra.npz"
TABLES = OUTDIR / "resampling_stability"
FIGURE = Path("assets/subspace_stability.png")   # Figure 5.1; assets/ is published

N = N_SPLIT_SUB         # samples of each kind
N_VEC = K_MAX + 5       # eigenvectors kept per sample
N_EIGEN_ROWS = 30       # rows of the eigenvalue table
N_PC = 8                # last component a candidate for the stable subspace may end at
MAX_LATER_BLOCK = 3     # largest block after the stable subspace that is tabulated


# --------------------------------------------------------------------------- #
# Samples
# --------------------------------------------------------------------------- #
def leading(X):
    """Eigenvalues and the leading N_VEC eigenvectors of X, z-scored on its own."""
    ev, V = eigen(X)
    return ev, V[:, :N_VEC]


def halves(X, perm):
    n = len(X)
    return leading(X[perm[: n // 2]]), leading(X[perm[n // 2:]])


def draw_samples(Xz):
    """Decompose every sample. Returns a dict of arrays, as saved to SPECTRA."""
    n = len(Xz)

    rng = np.random.default_rng(SEED)
    permuted_ev = []
    for _ in range(N):
        Xp = rng.permuted(Xz, axis=0)
        Xp = Xp - Xp.mean(axis=0)
        permuted_ev.append(np.linalg.svd(Xp, compute_uv=False) ** 2 / (n - 1))

    rng = np.random.default_rng(SEED)
    boot_idx = np.stack([rng.integers(0, n, n) for _ in range(N)])
    boot = [leading(Xz[i]) for i in boot_idx]

    rng = np.random.default_rng(SEED)
    split_perm = np.stack([rng.permutation(n) for _ in range(N)])
    split = [halves(Xz, p) for p in split_perm]
    null_perm, null = [], []
    for _ in range(N):
        Xp = np.column_stack([rng.permutation(c) for c in Xz.T])
        p = rng.permutation(n)
        null_perm.append(p)
        null.append(halves(Xp, p))

    def stack(pairs, i):
        return np.stack([[half[i] for half in pair] for pair in pairs])

    full_ev, full_V = leading(Xz)
    return {
        "full_ev": full_ev, "full_V": full_V,
        "permuted_ev": np.stack(permuted_ev),
        "bootstrap_index": boot_idx,
        "bootstrap_ev": np.stack([b[0] for b in boot]),
        "bootstrap_V": np.stack([b[1] for b in boot]),
        "split_perm": split_perm,
        "split_ev": stack(split, 0), "split_V": stack(split, 1),
        "null_perm": np.stack(null_perm),
        "null_ev": stack(null, 0), "null_V": stack(null, 1),
    }


def pairs(s, kind):
    """The decompositions compared, as a list of ((ev_a, V_a), (ev_b, V_b)):
    the two halves of each split, or the full sample and each bootstrap sample."""
    if kind == "bootstrap":
        full = (s["full_ev"], s["full_V"])
        return [(full, (e, V)) for e, V in zip(s["bootstrap_ev"], s["bootstrap_V"])]
    ev, V = s[f"{kind}_ev"], s[f"{kind}_V"]
    return [((ev[i, 0], V[i, 0]), (ev[i, 1], V[i, 1])) for i in range(len(ev))]


# --------------------------------------------------------------------------- #
# Agreement between two decompositions
# --------------------------------------------------------------------------- #
def pair_stats(Va, Vb):
    """Smallest and mean squared principal-angle cosine of the leading-k
    subspaces for k = 1..K_MAX, and for each of Va's first K_MAX components
    the congruence of its best match among Vb's."""
    min_cos, msq_cos = np.empty(K_MAX), np.empty(K_MAX)
    for k in range(1, K_MAX + 1):
        c = subspace_cosines(Va[:, :k], Vb[:, :k])
        min_cos[k - 1], msq_cos[k - 1] = c.min(), (c ** 2).mean()
    single = np.abs(Va[:, :K_MAX].T @ Vb[:, :K_MAX]).max(axis=1)
    return min_cos, msq_cos, single


def stats_over(sample_pairs):
    """pair_stats over all pairs: (min_cos, msq_cos, single), each (samples, K_MAX)."""
    out = [pair_stats(a[1], b[1]) for a, b in sample_pairs]
    return tuple(np.stack([o[i] for o in out]) for i in range(3))


def block_cosines(sample_pairs, block):
    a, z = block
    return np.stack([subspace_cosines(A[1][:, a:z], B[1][:, a:z]) for A, B in sample_pairs])


def rotation_congruence(sample_pairs, ref, idx):
    """Per pair, the congruence of each axis of ref between the two
    decompositions, each rotated within idx from scratch and paired to ref."""
    out = []
    for (ea, Va), (eb, Vb) in sample_pairs:
        A = match_axes(ref, rotate_group(Va, ea, idx)[0])
        B = match_axes(ref, rotate_group(Vb, eb, idx)[0])
        out.append([congruence(A[:, j], B[:, j]) for j in range(ref.shape[1])])
    return np.array(out)


def p05(x):
    return np.percentile(x, 5, axis=0)


def median(x):
    return np.median(x, axis=0)


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def eigenvalue_table(s):
    ev = s["full_ev"]
    q95 = np.percentile(s["permuted_ev"][:, :N_EIGEN_ROWS], 95, axis=0)
    share = ev / ev.sum()
    return pd.DataFrame({"component": range(1, N_EIGEN_ROWS + 1),
                         "eigenvalue": ev[:N_EIGEN_ROWS],
                         "% variance": 100 * share[:N_EIGEN_ROWS],
                         "cumulative %": 100 * np.cumsum(share)[:N_EIGEN_ROWS],
                         "null p95": q95,
                         "above null": ev[:N_EIGEN_ROWS] > q95})


def component_table(split, boot):
    t = pd.DataFrame({"component": [f"PC{j + 1}" for j in range(K_MAX)],
                      "bootstrap p05": p05(boot[2]),
                      "bootstrap median": median(boot[2]),
                      "split-half p05": p05(split[2]),
                      "split-half median": median(split[2])})
    t["passes"] = (t["bootstrap p05"] >= STABLE_THRESH) & (t["split-half p05"] >= STABLE_THRESH)
    return t


def nested_table(s, split, boot, null):
    n = len(s["bootstrap_index"][0])
    ev = s["full_ev"]
    half_ev = s["split_ev"].reshape(-1, s["split_ev"].shape[-1])
    null_ev = s["null_ev"].reshape(-1, s["null_ev"].shape[-1])
    half_error = np.sqrt(2 / (n / 2))
    rows = []
    for j in range(K_MAX):
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
    """Stability of the rotated axes of a candidate subspace, in variance order."""
    ref = rotate_group(s["full_V"], s["full_ev"], idx)[0]
    row = {}
    for prefix, sample_pairs in (("", split_pairs), ("bootstrap ", boot_pairs)):
        cong = rotation_congruence(sample_pairs, ref, idx)
        row.update({f"{prefix}rotated axis {j + 1} p05": v for j, v in enumerate(p05(cong))})
        row.update({f"{prefix}rotated axis {j + 1} median": v for j, v in enumerate(median(cong))})
        if not prefix:
            row["all rotated axes p05 >= 0.90"] = bool((p05(cong) >= STABLE_THRESH).all())
    return row


def block_table(s, split_pairs, boot_pairs, n_comp):
    """Blocks of components: the stable components as a block, the candidates
    for the stable subspace (with the stability of their rotated axes), the
    blocks inside it and the blocks after it. The candidates start after the
    n_comp stable components. Returns the table and sub_end, the number of
    components up to the end of the stable subspace (n_comp if there is none)."""
    rows, sub_end = [block_row((0, n_comp), "stable components", split_pairs, boot_pairs)], n_comp
    for e in range(n_comp + 2, N_PC + 1):
        row = block_row((n_comp, e), "stable subspace candidate", split_pairs, boot_pairs)
        row.update(candidate_rotation(s, split_pairs, boot_pairs, list(range(n_comp, e))))
        if row["RMS cosine p05 >= 0.90"]:
            sub_end = e
        rows.append(row)

    inner_end = max(sub_end, n_comp + 3)
    inner = [(a, z) for a in range(n_comp, N_PC) for z in range(a + 1, min(a + 3, N_PC) + 1)
             if z <= inner_end and (a, z) != (n_comp, inner_end)]
    rows += [block_row(b, "inside stable subspace", split_pairs, boot_pairs) for b in inner]

    later = [(a, z) for a in range(sub_end, K_MAX)
             for z in range(a + 2, min(a + MAX_LATER_BLOCK, K_MAX) + 1)]
    rows += [block_row(b, "after stable subspace", split_pairs, boot_pairs) for b in later]
    return pd.DataFrame(rows), sub_end


def looser_table(nested, blocks, n_comp, sub_end):
    """Medians instead of 5th percentiles for what lies after the stable
    subspace, with the stable subspace for comparison."""
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


def rotated_axes_table(s, split_pairs, boot_pairs, feats):
    """Stability of the rotated axes of pca_rotated.py, in its order and signs."""
    rows = []
    labels = ["stable components", "stable subspace"]
    groups = rotate_groups(s["full_V"], s["full_ev"], feats, ROTATION_GROUPS)
    for label, idx, (L, _) in zip(labels, ROTATION_GROUPS, groups):
        split = rotation_congruence(split_pairs, L, idx)
        boot = rotation_congruence(boot_pairs, L, idx)
        for j, name in enumerate(axis_names(idx)):
            rows.append({"axis": name, "group": label,
                         "variance share": (L[:, j] ** 2).sum() / L.shape[0],
                         "split-half p05": p05(split)[j], "split-half median": median(split)[j],
                         "bootstrap p05": p05(boot)[j], "bootstrap median": median(boot)[j]})
    return pd.DataFrame(rows)


def dimensions_table(n_comp, sub_end):
    """The stable components and stable subspace found here, against pca_rotated.py."""
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
def plot_subspaces(t, n_comp, sub_end):
    """For each k, the 5th percentile over split-halves of the smallest
    principal-angle cosine between the two halves' leading-k subspaces, with
    the 95th percentile of the same statistic on column-permuted data
    (dashed). The ends of the stable components (n_comp) and of the stable
    subspace (sub_end) are marked."""
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
    ax.text(K_MAX + 0.4, STABLE_THRESH + 0.012, f"{STABLE_THRESH:.2f}", color=secondary,
            fontsize=8, ha="right", va="bottom")

    ax.plot(k, sub, color=blue, linewidth=2, marker="o", markersize=7,
            markeredgecolor=surface, markeredgewidth=1.5,
            label="leading-k subspace, smallest cosine (5th percentile)")
    ax.plot(k, sub_null, color=blue, linewidth=1.2, linestyle=(0, (4, 3)), alpha=0.7,
            label="same, column-permuted data (95th percentile)")
    if sub_end > n_comp:   # label the end of the stable subspace and the point before it
        ax.annotate(f"{sub[sub_end - 1]:.2f}", (sub_end, sub[sub_end - 1]), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=8, color=ink)
        ax.annotate(f"{sub[sub_end - 2]:.2f}", (sub_end - 1, sub[sub_end - 2]), xytext=(9, -3),
                    textcoords="offset points", ha="left", va="center", fontsize=8, color=ink)

    ax.set_xlim(0.5, K_MAX + 0.5)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(range(1, K_MAX + 1))
    ax.set_xlabel("k", color=secondary, fontsize=9)
    ax.set_ylabel(f"5th percentile over {N} split-halves", color=secondary, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=ink, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.tight_layout()
    FIGURE.parent.mkdir(exist_ok=True)
    fig.savefig(FIGURE, dpi=200, facecolor=surface)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    Xz, feats, _ = load_and_prep()

    s = draw_samples(Xz)
    np.savez(SPECTRA, features=np.array(feats), seed=SEED, **s)
    print(f"[ok] decomposed {N} permuted, {N} bootstrap, {N} split-half and {N} null samples; "
          f"wrote {SPECTRA}")

    eig = eigenvalue_table(s)
    n_above = int(eig["above null"].cumprod().sum())
    if n_above != K_MAX:
        print(f"[warn] {n_above} components exceed the parallel-analysis null "
              f"but K_MAX = {K_MAX}; set K_MAX in pca_rotated.py to match")

    split_pairs, boot_pairs = pairs(s, "split"), pairs(s, "bootstrap")
    split, boot, null = stats_over(split_pairs), stats_over(boot_pairs), stats_over(pairs(s, "null"))

    comp = component_table(split, boot)
    n_comp = max(int(comp["passes"].cumprod().sum()), 2)   # rotate at least two components
    nested = nested_table(s, split, boot, null)
    blocks, sub_end = block_table(s, split_pairs, boot_pairs, n_comp)
    looser = looser_table(nested, blocks, n_comp, sub_end)
    rot = rotated_axes_table(s, split_pairs, boot_pairs, feats)
    dims = dimensions_table(n_comp, sub_end)

    tables = {"eigenvalues": eig, "components": comp, "nested_subspaces": nested,
              "blocks": blocks, "looser_criteria": looser, "rotated_axes": rot,
              "dimensions": dims}
    for name, table in tables.items():
        table.to_csv(TABLES / f"{name}.csv", index=False, float_format=CSV_FLOAT)
    plot_subspaces(nested, n_comp, sub_end)

    found = ", ".join(f"{r.dimensions} {r.found}" for r in dims.itertuples())
    if dims["match"].all():
        print(f"[ok] {found}, as in pca_rotated.py")
    else:
        print(f"[warn] {found}; pca_rotated.py differs (see dimensions.csv)")
    print(f"[ok] wrote {len(tables)} tables to {TABLES}/ and {FIGURE}")


if __name__ == "__main__":
    main()