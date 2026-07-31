#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_axes.py -- interactive scatter plots of the occupation space.

Dimensions (signs fixed by the anchors in pca_rotated.py, so '+' always means
harder to automate):

  embodiment    abstract/symbolic (-)   <->  physical/hands-on (+)
  cognitive     routine execution (-)   <->  complex reasoning (+)
  interpersonal technical/systems (-)   <->  care/people (+)
  union         institutional protection

Outputs two standalone pages in master_out/figures/:
  fig_3d_space.html  the three skill axes as free-floating coordinate axes drawn
                   through the origin (no plot box or background grid, which
                   otherwise occludes points while rotating), colored by union.
  fig_panels.html    all six pairwise 2-D sections on ONE page in a grid, with
                   semantic axis names; click any panel to expand it full-screen,
                   click again (or press Esc) to return.

Every point is one occupation; hover gives its title. Marker size encodes
employment on a LOG scale (raw employment is right-skewed ~69x median, so linear
sizing lets a few giant occupations swallow the chart). SIZE_BY_EMPLOYMENT=False
gives uniform dots.

Reads master_out/rotated_axes.csv and master_wide.xlsx.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from itertools import combinations

AXES_CSV = Path("output/rotated_axes.csv")
MASTER = Path("output/master_clean.xlsx")
FIGDIR = Path("output"); FIGDIR.mkdir(exist_ok=True)

SIZE_BY_EMPLOYMENT = True
SIZE_MAX = 16

# semantic names for the rotated axes (R1/R2/R3 in rotated_axes.csv)
DIMS = {
    "R1": ("embodiment", "abstract  \u2190  embodiment  \u2192  physical"),
    "R2": ("cognitive load", "routine  \u2190  cognitive load  \u2192  complex"),
    "R3": ("interpersonal", "technical  \u2190  interpersonal  \u2192  care"),
    "union": ("union coverage", "union coverage (institutional protection)"),
}


# label map handed to plotly express so the semantic names replace R1/R2/R3
# everywhere: axis titles, colorbar, legend AND the hover tooltip.
LBL = {k: v[0] for k, v in DIMS.items()}
LBL["employment"] = "employment"


def load():
    ax = pd.read_csv(AXES_CSV, index_col=0)
    m = pd.read_excel(MASTER).set_index("onet_soc")

    df = pd.DataFrame(index=ax.index)
    df["title"] = ax["title"]
    for c in ["R1", "R2", "R3"]:
        if c in ax.columns:
            df[c] = ax[c]
    df["union"] = pd.to_numeric(m["ext_union_cov_pct"], errors="coerce").reindex(df.index)
    emp = pd.to_numeric(m["ext_employment"], errors="coerce").reindex(df.index)
    df["employment"] = emp
    df["size"] = np.log1p(emp.fillna(emp.median()))
    df = df.dropna(subset=[c for c in DIMS if c in df.columns])
    print(f"{len(df)} occupations plotted")
    return df


HOVER = {"employment": ":,.0f", "union": ":.2f", "size": False}


def axis_lines(df, cols):
    """Three lines through the origin acting as free-floating coordinate axes."""
    traces = []
    span = {c: (df[c].min() * 1.05, df[c].max() * 1.05) for c in cols}
    for i, c in enumerate(cols):
        pts = {k: [0, 0] for k in cols}
        pts[c] = list(span[c])
        traces.append(go.Scatter3d(
            x=pts[cols[0]], y=pts[cols[1]], z=pts[cols[2]],
            mode="lines", line=dict(color="rgba(90,90,90,0.85)", width=4),
            hoverinfo="skip", showlegend=False))
        # label at the positive end
        lab = {k: [0] for k in cols}
        lab[c] = [span[c][1]]
        traces.append(go.Scatter3d(
            x=lab[cols[0]], y=lab[cols[1]], z=lab[cols[2]],
            mode="text", text=["+" + DIMS[c][0]],
            textfont=dict(size=12, color="rgb(60,60,60)"),
            hoverinfo="skip", showlegend=False))
    return traces


def build_3d(df, size_arg):
    cols = ["R1", "R2", "R3"]
    fig = px.scatter_3d(
        df, x="R1", y="R2", z="R3", color="union",
        hover_name="title", hover_data=HOVER,
        color_continuous_scale="Viridis", opacity=0.78,
        labels=LBL, **size_arg)
    for t in axis_lines(df, cols):
        fig.add_trace(t)

    blank = dict(showbackground=False, showgrid=False, zeroline=False,
                 showticklabels=False, title="", visible=False)
    fig.update_layout(
        title="Occupation space \u2014 three skill axes, colored by union coverage",
        scene=dict(xaxis=blank, yaxis=blank, zaxis=blank,
                   aspectmode="cube",
                   camera=dict(eye=dict(x=1.5, y=1.5, z=1.1))),
        margin=dict(l=0, r=0, t=48, b=0), height=760)
    return fig


def build_panels(df, size_arg):
    dims = [d for d in DIMS if d in df.columns]
    figs = []
    for x, y in combinations(dims, 2):
        color = "union" if "union" not in (x, y) else "R2"
        fig = px.scatter(
            df, x=x, y=y, color=color,
            hover_name="title", hover_data=HOVER,
            color_continuous_scale="Viridis", opacity=0.75,
            labels=LBL, **size_arg)
        fig.add_hline(y=df[y].median(), line_width=1, line_dash="dot",
                      line_color="rgba(120,120,120,.6)")
        fig.add_vline(x=df[x].median(), line_width=1, line_dash="dot",
                      line_color="rgba(120,120,120,.6)")
        # both axes are rotated skill axes -> same standardized units, so lock
        # equal scaling; otherwise the cloud stretches to whatever aspect ratio
        # the container happens to have. (Not applied when union is an axis:
        # its 0-0.9 range against +-4 would collapse to a sliver.)
        if "union" not in (x, y):
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_layout(
            xaxis_title=DIMS[x][1], yaxis_title=DIMS[y][1],
            margin=dict(l=52, r=10, t=34, b=48),
            coloraxis_colorbar=dict(title=DIMS[color][0], thickness=10),
            title=dict(text=f"{DIMS[x][0]} \u00d7 {DIMS[y][0]}", font=dict(size=14)))
        figs.append((f"{DIMS[x][0]} \u00d7 {DIMS[y][0]}", fig))
    return figs


PAGE = """<!doctype html>
<meta charset="utf-8">
<title>Occupation space \u2014 pairwise sections</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  body {{ font-family: -apple-system, Helvetica, Arial, sans-serif;
         margin: 24px; color: #222; background: #fff; }}
  h1 {{ font-size: 18px; font-weight: 600; margin: 0 0 4px; }}
  p.note {{ font-size: 13px; color: #666; margin: 0 0 20px; }}
  .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }}
  .cell {{ border: 1px solid #e3e3e3; border-radius: 8px; padding: 4px;
          cursor: zoom-in; background: #fff; height: 360px;
          box-sizing: border-box; }}
  .cell .plotly-graph-div {{ width: 100% !important; height: 100% !important; }}
  .cell:hover {{ border-color: #999; }}
  .cell.zoom {{ position: fixed; top: 3vh; height: 94vh;
               width: min(94vw, 130vh); left: 50%;
               transform: translateX(-50%); z-index: 99; cursor: zoom-out;
               box-shadow: 0 8px 40px rgba(0,0,0,.25); }}
  .backdrop {{ display: none; position: fixed; inset: 0; z-index: 98;
              background: rgba(255,255,255,.75); }}
  .backdrop.on {{ display: block; }}
  @media (max-width: 1100px) {{ .grid {{ grid-template-columns: repeat(2, 1fr); }} }}
</style>
<h1>Occupation space \u2014 six pairwise sections</h1>
<p class="note">Each point is an occupation; hover for its title, marker size is
log employment. On every axis, <b>+ means harder to automate</b>.
Click a panel to enlarge, click again or press Esc to close.</p>
<div class="backdrop" id="bd"></div>
<div class="grid">{cells}</div>
<script>
const bd = document.getElementById('bd');
function close_all() {{
  document.querySelectorAll('.cell.zoom').forEach(c => {{
    c.classList.remove('zoom');
    Plotly.Plots.resize(c.querySelector('.plotly-graph-div'));
  }});
  bd.classList.remove('on');
}}
document.querySelectorAll('.cell').forEach(cell => {{
  cell.addEventListener('click', ev => {{
    if (ev.target.closest('.modebar')) return;   // don't hijack plotly toolbar
    const zoomed = cell.classList.contains('zoom');
    close_all();
    if (!zoomed) {{ cell.classList.add('zoom'); bd.classList.add('on'); }}
    Plotly.Plots.resize(cell.querySelector('.plotly-graph-div'));
  }});
}});
bd.addEventListener('click', close_all);
document.addEventListener('keydown', e => {{ if (e.key === 'Escape') close_all(); }});
</script>
"""


def main():
    df = load()
    size_arg = dict(size="size", size_max=SIZE_MAX) if SIZE_BY_EMPLOYMENT else {}

    fig3d = build_3d(df, size_arg)
    fig3d.write_html(FIGDIR / "fig_3d_space.html")
    print("wrote fig_3d_space.html")

    cells = []
    for i, (name, fig) in enumerate(build_panels(df, size_arg)):
        fig.update_layout(autosize=True)
        div = fig.to_html(full_html=False, include_plotlyjs=False,
                          default_height="100%", div_id=f"p{i}",
                          config={"displaylogo": False})
        cells.append(f'<div class="cell">{div}</div>')
    (FIGDIR / "fig_panels.html").write_text(PAGE.format(cells="\n".join(cells)),
                                        encoding="utf-8")
    print("wrote fig_panels.html")
    print(f"\n[ok] open {FIGDIR}/panels.html and {FIGDIR}/3d_space.html")


if __name__ == "__main__":
    main()