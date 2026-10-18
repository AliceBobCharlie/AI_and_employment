// figures.js -- draws the paper's interactive figures with plotly.js.
//
// Data: output/plot_data.csv, written by plot_axes.py. One row per occupation:
// onet_soc, title, soc_group, R1 to R6, union, employment.
//
// The axis figures of Section 5.4 also read output/axis_loadings.csv, written
// by plot_axes.py: one row per feature variable, with its block of the Content
// Model and its loading on each rotated axis.
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
  const LOADINGS_URL = "output/axis_loadings.csv";
  const YOUNG_URL = "output/acs_young_share.csv";
  const SIZE_MAX = 16;
  const FONT = '"Source Sans 3", "Helvetica Neue", Arial, sans-serif';

  const DIMS = {
    R1: { name: "Physical against symbolic work", low: "symbolic", high: "physical" },
    R2: { name: "Complex against simple work", low: "simple", high: "complex" },
    R3: { name: "People against artefact work", low: "artefacts", high: "people" },
    R4: { name: "Hands-on against field work" },
    R5: { name: "Standardised against open-ended work" },
    R6: { name: "Commercial against individual work" },
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
      group: records.map((r) => r.soc_group),
      R1: col("R1"), R2: col("R2"), R3: col("R3"),
      R4: col("R4"), R5: col("R5"), R6: col("R6"),
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

  // ---------------------------------------------------------------- axis figures

  // What one rotated axis is made of (Section 5.4), in two panels that share
  // nothing but the figure. Above: every feature variable's loading on the
  // axis, one row per block of the Content Model, so no variable is left out
  // of the picture. Below: every occupation's score, one row per SOC major
  // group, ordered by the group's median, which a bar marks. Points in a row
  // are spread vertically by a fixed offset, not at random, so the figure is
  // the same on every load.

  const BLOCK_ORDER = ["Skills", "Abilities", "Knowledge", "Work Activities", "Work Context"];
  const AXIS_BLUE = "#2a78d6";
  const AXIS_GREY = "#8c8c8c";
  const STRONG = 0.5;          // loadings counted as strong in the panel heading
  const LINE_MAX = 28;         // characters per line of a group's name: every name fits in two lines

  function spread(i, width) {
    // Golden-ratio sequence: evenly spread, never the same twice in a row.
    return (((i * 0.6180339887) % 1) - 0.5) * width;
  }

  function wrap(text, n) {
    // Break between words so that no line is longer than n characters, unless
    // a single word is.
    const lines = [];
    text.split(" ").forEach((word) => {
      const last = lines[lines.length - 1];
      if (last !== undefined && (last + " " + word).length <= n) lines[lines.length - 1] = last + " " + word;
      else lines.push(word);
    });
    return lines.join("<br>");
  }

  function loadingColumns(records) {
    return {
      variable: records.map((r) => r.variable),
      block: records.map((r) => r.block),
      value: (k) => records.map((r) => Number(r[k])),
    };
  }

  function axisFigure(d, L, k) {
    const name = DIMS[k] ? DIMS[k].name : k;
    const traces = [];

    // ---- top panel: loadings
    const load = L.value(k);
    const rowOf = BLOCK_ORDER.map((_, i) => BLOCK_ORDER.length - 1 - i);   // first block on top
    traces.push({
      type: "scatter", mode: "markers",
      x: load,
      y: L.block.map((b, i) => rowOf[BLOCK_ORDER.indexOf(b)] + spread(i, 0.6)),
      xaxis: "x", yaxis: "y",
      text: L.variable,
      customdata: L.block,
      hovertemplate: "<b>%{text}</b><br>%{customdata}<br>Loading: %{x:+.2f}<extra></extra>",
      marker: { color: AXIS_BLUE, size: 7, opacity: 0.65, line: { width: 0 } },
      showlegend: false,
    });
    const strongPos = load.filter((v) => v >= STRONG).length;
    const strongNeg = load.filter((v) => v <= -STRONG).length;

    // ---- bottom panel: occupations by SOC major group
    const scores = d[k];
    const groups = [...new Set(d.group)];
    const members = Object.fromEntries(groups.map((g) => [g, []]));
    d.group.forEach((g, i) => members[g].push(i));
    const med = Object.fromEntries(groups.map((g) => [g, median(members[g].map((i) => scores[i]))]));
    groups.sort((a, b) => med[a] - med[b]);          // highest median at the top
    const row = Object.fromEntries(groups.map((g, j) => [g, j]));

    traces.push({
      type: "scatter", mode: "markers",
      x: scores,
      y: d.group.map((g, i) => row[g] + spread(i, 0.5)),
      xaxis: "x2", yaxis: "y2",
      text: d.title,
      customdata: d.group,
      hovertemplate: "<b>%{text}</b><br>%{customdata}<br>" + name + ": %{x:+.2f}<extra></extra>",
      marker: { color: AXIS_GREY, size: 6, opacity: 0.55, line: { width: 0 } },
      showlegend: false,
    });
    traces.push({
      type: "scatter", mode: "markers",
      x: groups.map((g) => med[g]),
      y: groups.map((g) => row[g]),
      xaxis: "x2", yaxis: "y2",
      text: groups,
      hovertemplate: "<b>%{text}</b><br>Group median: %{x:+.2f}<extra></extra>",
      marker: { symbol: "line-ns", size: 20, line: { width: 3, color: "#000000" } },
      showlegend: false,
    });
    const xaxis = (title, range) => ({
      title: { text: title, standoff: 6 },
      range: range,
      zeroline: true, zerolinecolor: "#8c8c8c", zerolinewidth: 1,
      showgrid: true, gridcolor: "#ececec",
      linecolor: "#000000", ticks: "outside",
    });
    const yaxis = (domain, labels) => ({
      domain: domain,
      tickvals: labels.map((_, i) => i),
      ticktext: labels,
      range: [-0.6, labels.length - 0.4],
      showgrid: false, zeroline: false, fixedrange: true,
      ticks: "", automargin: true,
      tickfont: { size: 12.5 },
    });

    const span = Math.max(...scores.map(Math.abs)) * 1.05;
    const layout = Object.assign(baseLayout(), {
      plot_bgcolor: "#ffffff",
      margin: { l: 16, r: 16, t: 28, b: 48 },
      xaxis: Object.assign(xaxis("Loading on " + name.toLowerCase(), [-1, 1]), { anchor: "y" }),
      yaxis: yaxis([0.8, 1], rowOf.map((_, r) => BLOCK_ORDER[BLOCK_ORDER.length - 1 - r])),
      xaxis2: Object.assign(xaxis(name + " (standard deviations)", [-span, span]), { anchor: "y2" }),
      yaxis2: yaxis([0, 0.71], groups.map((g) => wrap(g, LINE_MAX))),
      annotations: [
        {
          xref: "paper", yref: "paper", x: 0, y: 1.0, yanchor: "bottom", xanchor: "left",
          showarrow: false, font: { family: FONT, size: 13, color: "#52514e" },
          text: "All " + load.length + " variables: " + strongPos + " load at +" +
            STRONG.toFixed(1) + " or above, " + strongNeg + " at −" + STRONG.toFixed(1) + " or below",
        },
        {
          xref: "paper", yref: "paper", x: 0, y: 0.725, yanchor: "bottom", xanchor: "left",
          showarrow: false, font: { family: FONT, size: 13, color: "#52514e" },
          text: "All " + scores.length + " occupations, by SOC major group",
        },
      ],
    });
    return { data: traces, layout: layout };
  }

  async function drawAxisFigures(targets, d) {
    if (!targets.length) return;
    let L;
    try {
      L = loadingColumns(await loadCsv(LOADINGS_URL));
    } catch (err) {
      showError(targets, "The figure data could not be loaded (" + err.message + "). " + LOAD_HELP);
      return;
    }
    targets.forEach((el) => {
      const fig = axisFigure(d, L, el.dataset.axis);
      Plotly.newPlot(el, fig.data, fig.layout, CONFIG);
    });
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

  async function drawAxesFigures(space, section, select, axisPlots) {
    const targets = [space, section].filter(Boolean).concat(axisPlots);
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
    drawAxisFigures(axisPlots, d);
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
    const axisPlots = [...document.querySelectorAll(".plot-axis[data-axis]")];
    const targets = [space, section, young].filter(Boolean).concat(axisPlots);
    if (!targets.length) return;

    if (typeof Plotly === "undefined") {
      showError(targets, "The figure library could not be loaded. Check the connection and reload the page.");
      return;
    }

    // Separate data files, loaded independently, so a missing file leaves
    // only its own figure empty.
    drawAxesFigures(space, section, select, axisPlots);
    drawYoungFigure(young);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", main);
  } else {
    main();
  }
})();