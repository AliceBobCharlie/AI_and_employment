// figures.js -- draws the paper's interactive figures with plotly.js.
//
// Data: output/plot_data.csv, written by plot_axes.py. One row per occupation:
// onet_soc, title, R1, R2, R3, employment, union.
//
// Figure 6.2 reads output/acs_young_share.csv, written by acs_young_share.py:
// one row per matched O*NET occupation, with the change in the young-worker
// share (delta) and the occupation's axis scores.
//
// Axis signs are fixed in pca_rotated.py so that + always means harder to
// automate. Marker size is log employment, scaled the way plotly express does
// it (area proportional to value, largest marker SIZE_MAX pixels across).

(function () {
  "use strict";

  const DATA_URL = "output/plot_data.csv";
  const YOUNG_URL = "output/acs_young_share.csv";
  const SIZE_MAX = 16;
  const FONT = '"Source Sans 3", "Helvetica Neue", Arial, sans-serif';

  const DIMS = {
    R1: { name: "Physical intensity", low: "symbolic", high: "embodied" },
    R2: { name: "Judgement", low: "procedural", high: "complex" },
    R3: { name: "Person-facing work", low: "technical", high: "caring" },
    union: { name: "Union coverage" },
  };

  // Every plane through the space. When union coverage is not an axis it is
  // the colour; when it is, the points are drawn in one colour, since colouring
  // by one of the two plotted axes would repeat what the position already shows.
  const SECTIONS = [
    ["R1", "R2"], ["R1", "R3"], ["R2", "R3"],
    ["R1", "union"], ["R2", "union"], ["R3", "union"],
  ];

  // scrollZoom is off so that the wheel scrolls the page. Plotly turns it on
  // for 3-D scenes by default, which traps the page when the reader scrolls
  // over the figure. Zooming the 3-D view moves to Ctrl/Cmd + wheel below.
  const CONFIG = {
    responsive: true,
    scrollZoom: false,
    displaylogo: false,
    showSendToCloud: false,
    modeBarButtonsToRemove: ["select2d", "lasso2d"],
  };

  const ZOOM_STEP = 0.0008;       // camera distance change per wheel pixel
  const ZOOM_PER_EVENT = [0.8, 1.25];   // biggest jump one wheel event may make
  const ZOOM_LIMITS = [0.5, 4];         // how close to and far from the origin

  // ---------------------------------------------------------------- data

  // Minimal RFC 4180 parser: quoted fields may contain commas, newlines and
  // doubled quotes. Occupation titles do contain commas.
  function parseCsv(text) {
    const rows = [];
    let row = [], field = "", quoted = false;
    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      if (quoted) {
        if (c === '"' && text[i + 1] === '"') { field += '"'; i++; }
        else if (c === '"') { quoted = false; }
        else { field += c; }
      } else if (c === '"') {
        quoted = true;
      } else if (c === ",") {
        row.push(field); field = "";
      } else if (c === "\n" || c === "\r") {
        if (c === "\r" && text[i + 1] === "\n") i++;
        row.push(field); rows.push(row); row = []; field = "";
      } else {
        field += c;
      }
    }
    if (field !== "" || row.length) { row.push(field); rows.push(row); }

    const header = rows.shift();
    return rows
      .filter((r) => r.length === header.length)
      .map((r) => Object.fromEntries(header.map((h, j) => [h, r[j]])));
  }

  function toColumns(records) {
    const col = (k) => records.map((r) => Number(r[k]));
    const employment = col("employment");
    return {
      title: records.map((r) => r.title),
      R1: col("R1"), R2: col("R2"), R3: col("R3"),
      union: col("union"),
      employment: employment,
      size: employment.map((e) => Math.log1p(e)),
    };
  }

  function median(values) {
    const s = [...values].sort((a, b) => a - b);
    const m = Math.floor(s.length / 2);
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  }

  // ---------------------------------------------------------------- shared pieces

  function markerSize(d) {
    return {
      size: d.size,
      sizemode: "area",
      sizeref: (2 * Math.max(...d.size)) / (SIZE_MAX * SIZE_MAX),
      sizemin: 1,
    };
  }

  function unionColour(d, colourbarExtra) {
    return {
      color: d.union,
      colorscale: "Viridis",
      colorbar: Object.assign({
        title: { text: "Union<br>coverage", side: "top" },
        tickformat: ".0%",
        thickness: 9,
        outlinewidth: 0,
        len: 0.55,
        lenmode: "fraction",
        x: 1,
        xanchor: "right",
        y: 0.5,
        yanchor: "middle",
        tickfont: { size: 12 },
      }, colourbarExtra || {}),
    };
  }

  function hoverTemplate(dims) {
    const lines = dims.map((k, i) => {
      const v = "customdata[" + i + "]";
      return k === "union"
        ? DIMS[k].name + ": %{" + v + ":.0%}"
        : DIMS[k].name + ": %{" + v + ":+.2f}";
    });
    return "<b>%{text}</b><br>" + lines.join("<br>") +
      "<br>Employment: %{customdata[" + dims.length + "]:,.0f}<extra></extra>";
  }

  function customData(d, dims) {
    return d.title.map((_, i) => dims.map((k) => d[k][i]).concat(d.employment[i]));
  }

  function axisTitle(k) {
    const dim = DIMS[k];
    return dim.low ? "← " + dim.low + "     " + dim.name + "     " + dim.high + " →" : dim.name;
  }

  function baseLayout() {
    return {
      font: { family: FONT, size: 14, color: "#000000" },
      paper_bgcolor: "#ffffff",
      hoverlabel: { font: { family: FONT, size: 13 }, bgcolor: "#ffffff", bordercolor: "#000000" },
      margin: { l: 64, r: 16, t: 16, b: 56 },
    };
  }

  // ---------------------------------------------------------------- figure 6.1

  function drawSpace(el, d) {
    const dims = ["R1", "R2", "R3"];
    const points = {
      type: "scatter3d",
      mode: "markers",
      x: d.R1, y: d.R2, z: d.R3,
      text: d.title,
      customdata: customData(d, dims.concat("union")),
      hovertemplate: hoverTemplate(dims.concat("union")),
      marker: Object.assign(markerSize(d), unionColour(d), { opacity: 0.8, line: { width: 0 } }),
      showlegend: false,
    };

    // Three lines through the origin stand in for the axes; a plot box and
    // grid would hide points while the figure is rotated.
    const traces = [points];
    dims.forEach((k) => {
      const lo = Math.min(...d[k]) * 1.05, hi = Math.max(...d[k]) * 1.05;
      const line = { x: [0, 0], y: [0, 0], z: [0, 0] };
      const end = { x: [0], y: [0], z: [0] };
      const key = { R1: "x", R2: "y", R3: "z" }[k];
      line[key] = [lo, hi];
      // The label sits beyond the end of its axis line: placed at the line's
      // end it falls inside the point cloud from most camera angles.
      end[key] = [hi * 1.22];
      traces.push(Object.assign({
        type: "scatter3d", mode: "lines",
        line: { color: "#555555", width: 4 },
        hoverinfo: "skip", showlegend: false,
      }, line));
      traces.push(Object.assign({
        type: "scatter3d", mode: "text",
        text: ["+ " + DIMS[k].name.toLowerCase()],
        textfont: { family: FONT, size: 13, color: "#000000" },
        hoverinfo: "skip", showlegend: false,
      }, end));
    });

    const hidden = { visible: false };
    const layout = Object.assign(baseLayout(), {
      margin: { l: 0, r: 0, t: 0, b: 0 },
      scene: {
        xaxis: hidden, yaxis: hidden, zaxis: hidden,
        aspectmode: "cube",
        camera: { eye: { x: 0.86, y: 0.86, z: 0.62 } },
      },
    });
    return Plotly.newPlot(el, traces, layout, CONFIG).then(() => {
      enableModifierZoom(el);
      return el;
    });
  }

  // Plotly cancels the wheel event over a 3-D scene whether or not scrollZoom
  // is on, which traps the page when a reader scrolls onto the figure. This
  // listener runs in the capture phase, before Plotly's own: a plain wheel is
  // stopped there, so Plotly never sees it and the page scrolls normally, and
  // Ctrl/Cmd + wheel is handled here instead, moving the camera towards or
  // away from the point it looks at.
  function enableModifierZoom(el) {
    el.addEventListener("wheel", (ev) => {
      if (!ev.ctrlKey && !ev.metaKey) {
        ev.stopPropagation();
        return;
      }
      ev.stopPropagation();
      ev.preventDefault();

      const camera = el.layout && el.layout.scene && el.layout.scene.camera;
      const eye = camera && camera.eye;
      if (!eye) return;

      const step = clamp(1 + ev.deltaY * ZOOM_STEP, ZOOM_PER_EVENT[0], ZOOM_PER_EVENT[1]);
      const distance = Math.hypot(eye.x, eye.y, eye.z);
      const wanted = clamp(distance * step, ZOOM_LIMITS[0], ZOOM_LIMITS[1]);
      const factor = wanted / distance;
      if (factor === 1) return;

      Plotly.relayout(el, {
        "scene.camera.eye": { x: eye.x * factor, y: eye.y * factor, z: eye.z * factor },
      });
    }, { capture: true, passive: false });
  }

  function clamp(v, lo, hi) {
    return Math.min(Math.max(v, lo), hi);
  }

  // ---------------------------------------------------------------- figure 6.3

  function sectionFigure(d, x, y) {
    const unionIsAxis = x === "union" || y === "union";
    const dims = unionIsAxis ? [x, y] : [x, y, "union"];
    const marker = Object.assign(markerSize(d), { opacity: 0.75, line: { width: 0 } },
      unionIsAxis ? { color: "#333333" } : unionColour(d, { len: 0.8 }));

    const trace = {
      type: "scatter", mode: "markers",
      x: d[x], y: d[y],
      text: d.title,
      customdata: customData(d, dims),
      hovertemplate: hoverTemplate(dims),
      marker: marker,
      showlegend: false,
    };

    const rule = { line: { color: "#8c8c8c", width: 1, dash: "dot" }, layer: "below" };
    const mx = median(d[x]), my = median(d[y]);
    const axis = (k) => ({
      title: { text: axisTitle(k) },
      zeroline: false,
      showgrid: false,
      linecolor: "#000000",
      ticks: "outside",
      tickformat: k === "union" ? ".0%" : "",
    });

    const layout = Object.assign(baseLayout(), {
      plot_bgcolor: "#ffffff",
      xaxis: axis(x),
      yaxis: Object.assign(axis(y),
        // Two rotated axes share standardised units, so they are drawn at the
        // same scale. Union coverage (0 to 0.9) against an axis of about ±4
        // would collapse to a sliver, so that pair is left free.
        unionIsAxis ? {} : { scaleanchor: "x", scaleratio: 1 }),
      shapes: [
        Object.assign({ type: "line", xref: "x", yref: "paper", x0: mx, x1: mx, y0: 0, y1: 1 }, rule),
        Object.assign({ type: "line", xref: "paper", yref: "y", x0: 0, x1: 1, y0: my, y1: my }, rule),
      ],
    });
    return { data: [trace], layout: layout };
  }

  function drawSections(el, select, d) {
    SECTIONS.forEach(([x, y], i) => {
      const opt = document.createElement("option");
      opt.value = String(i);
      opt.textContent = DIMS[x].name + " × " + DIMS[y].name.toLowerCase();
      select.appendChild(opt);
    });
    const draw = () => {
      const [x, y] = SECTIONS[Number(select.value)];
      const fig = sectionFigure(d, x, y);
      return Plotly.react(el, fig.data, fig.layout, CONFIG);
    };
    select.addEventListener("change", draw);
    return draw();
  }

  // ---------------------------------------------------------------- figure 6.2

  // Change in the share of workers aged 22 to 25, 2022 to 2024, against each
  // axis. Only exact matches are drawn: an aggregate ACS code broadcasts one
  // value to several occupations, which would put identical points in a row.
  // Each panel reports the correlation unweighted and employment-weighted,
  // computed the same way as in acs_young_share.py.

  function youngColumns(records) {
    const exact = records.filter((r) => r.exact_match === "True");
    const col = (k) => exact.map((r) => Number(r[k]));
    const employment = col("employment");
    return {
      title: exact.map((r) => r.title),
      R1: col("R1"), R2: col("R2"), R3: col("R3"),
      delta: col("delta").map((v) => v * 100),     // percentage points
      employment: employment,
      size: employment.map((e) => Math.log1p(e)),
    };
  }

  function correlation(x, y, w) {
    // Weighted Pearson correlation; with w all ones it is the ordinary one.
    // Rows with a weight of zero (no employment figure) are left out.
    const keep = x.map((_, i) => i).filter((i) => w[i] > 0);
    const total = keep.reduce((s, i) => s + w[i], 0);
    const mean = (v) => keep.reduce((s, i) => s + w[i] * v[i], 0) / total;
    const mx = mean(x), my = mean(y);
    let sxy = 0, sxx = 0, syy = 0;
    keep.forEach((i) => {
      sxy += w[i] * (x[i] - mx) * (y[i] - my);
      sxx += w[i] * (x[i] - mx) ** 2;
      syy += w[i] * (y[i] - my) ** 2;
    });
    return sxy / Math.sqrt(sxx * syy);
  }

  function formatSigned(v) {
    return (v < 0 ? "−" : "+") + Math.abs(v).toFixed(2);
  }

  function drawYoung(el, d) {
    const dims = ["R1", "R2", "R3"];
    // Side by side in the text column; stacked on a narrow screen, where three
    // panels across would each be too thin to read.
    const stacked = el.clientWidth < 600;
    if (stacked) el.classList.add("plot-young-stacked");
    const ones = d.delta.map(() => 1);

    const traces = [];
    const annotations = [];
    const layout = Object.assign(baseLayout(), {
      plot_bgcolor: "#ffffff",
      grid: stacked
        ? { rows: 3, columns: 1, pattern: "independent", ygap: 0.45 }
        : { rows: 1, columns: 3, pattern: "independent", xgap: 0.08 },
      margin: { l: 64, r: 16, t: 36, b: 56 },
    });

    dims.forEach((k, i) => {
      const n = i === 0 ? "" : String(i + 1);
      traces.push({
        type: "scatter", mode: "markers",
        x: d[k], y: d.delta,
        xaxis: "x" + n, yaxis: "y" + n,
        text: d.title,
        customdata: d.employment,
        hovertemplate: "<b>%{text}</b><br>" + DIMS[k].name + ": %{x:+.2f}" +
          "<br>Change in young share: %{y:+.1f} pp" +
          "<br>Employment: %{customdata:,.0f}<extra></extra>",
        marker: Object.assign(markerSize(d), { color: "#333333", opacity: 0.6, line: { width: 0 } }),
        showlegend: false,
      });

      layout["xaxis" + n] = {
        title: { text: DIMS[k].name },
        zeroline: false, showgrid: false,
        linecolor: "#000000", ticks: "outside",
      };
      layout["yaxis" + n] = {
        title: { text: i === 0 || stacked ? "Change in young share (pp)" : "" },
        zeroline: true, zerolinecolor: "#8c8c8c", zerolinewidth: 1,
        showgrid: false,
        linecolor: "#000000", ticks: "outside",
      };
      // Every panel shares one vertical scale so they can be compared.
      if (i > 0) layout["yaxis" + n].matches = "y";

      const r = correlation(d[k], d.delta, ones);
      const rw = correlation(d[k], d.delta, d.employment);
      annotations.push({
        xref: "x" + n + " domain", yref: "y" + n + " domain",
        x: 0, y: 1.02, xanchor: "left", yanchor: "bottom",
        showarrow: false, align: "left",
        text: "r = " + formatSigned(r) + "   weighted " + formatSigned(rw),
        font: { family: FONT, size: 13, color: "#000000" },
      });
    });
    layout.annotations = annotations;

    return Plotly.newPlot(el, traces, layout, CONFIG);
  }

  // ---------------------------------------------------------------- start

  async function loadCsv(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(url + " returned " + res.status);
    return parseCsv(await res.text());
  }

  const LOAD_HELP = "If the page was opened as a local file, serve it over HTTP instead, for example with quarto preview.";

  function showError(elements, message) {
    elements.forEach((el) => {
      const p = document.createElement("p");
      p.className = "plot-error";
      p.textContent = message;
      el.replaceChildren(p);
    });
  }

  async function drawAxesFigures(space, section, select) {
    const targets = [space, section].filter(Boolean);
    if (!targets.length) return;
    let d;
    try {
      d = toColumns(await loadCsv(DATA_URL));
    } catch (err) {
      showError(targets, "The figure data could not be loaded (" + err.message + "). " + LOAD_HELP);
      return;
    }
    if (space) drawSpace(space, d);
    if (section && select) drawSections(section, select, d);
  }

  async function drawYoungFigure(young) {
    if (!young) return;
    let d;
    try {
      d = youngColumns(await loadCsv(YOUNG_URL));
    } catch (err) {
      showError([young], "The figure data could not be loaded (" + err.message + "). " + LOAD_HELP);
      return;
    }
    drawYoung(young, d);
  }

  function main() {
    const space = document.getElementById("plot-space");
    const section = document.getElementById("plot-section");
    const select = document.getElementById("section-select");
    const young = document.getElementById("plot-young");
    const targets = [space, section, young].filter(Boolean);
    if (!targets.length) return;

    if (typeof Plotly === "undefined") {
      showError(targets, "The figure library could not be loaded. Check the connection and reload the page.");
      return;
    }

    // Separate data files, loaded independently, so a missing file leaves
    // only its own figure empty.
    drawAxesFigures(space, section, select);
    drawYoungFigure(young);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", main);
  } else {
    main();
  }
})();
