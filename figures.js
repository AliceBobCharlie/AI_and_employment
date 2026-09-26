// figures.js -- draws the paper's interactive figures with plotly.js.
//
// Data: output/plot_data.csv, written by plot_axes.py. One row per occupation:
// onet_soc, title, R1, R2, R3, employment, union.
//
// Axis signs are fixed in pca_rotated.py so that + always means harder to
// automate. Marker size is log employment, scaled the way plotly express does
// it (area proportional to value, largest marker SIZE_MAX pixels across).

(function () {
  "use strict";

  const DATA_URL = "output/plot_data.csv";
  const SIZE_MAX = 16;
  const FONT = '"Source Sans 3", "Helvetica Neue", Arial, sans-serif';

  const DIMS = {
    R1: { name: "Physical intensity", low: "symbolic", high: "physical" },
    R2: { name: "Judgement", low: "procedural", high: "judgement" },
    R3: { name: "Person-facing work", low: "technical", high: "person-facing" },
    union: { name: "Union coverage" },
  };

  // Every plane through the space. When union coverage is not an axis it is
  // the colour; when it is, the points are drawn in one colour, since colouring
  // by one of the two plotted axes would repeat what the position already shows.
  const SECTIONS = [
    ["R1", "R2"], ["R1", "R3"], ["R2", "R3"],
    ["R1", "union"], ["R2", "union"], ["R3", "union"],
  ];

  const CONFIG = {
    responsive: true,
    displaylogo: false,
    showSendToCloud: false,
    modeBarButtonsToRemove: ["select2d", "lasso2d"],
  };

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
        thickness: 10,
        outlinewidth: 0,
        len: 0.6,
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
      end[key] = [hi];
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
        camera: { eye: { x: 1.5, y: 1.5, z: 1.1 } },
      },
    });
    return Plotly.newPlot(el, traces, layout, CONFIG);
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

  // ---------------------------------------------------------------- start

  function showError(elements, message) {
    elements.forEach((el) => {
      const p = document.createElement("p");
      p.className = "plot-error";
      p.textContent = message;
      el.replaceChildren(p);
    });
  }

  async function main() {
    const space = document.getElementById("plot-space");
    const section = document.getElementById("plot-section");
    const select = document.getElementById("section-select");
    const targets = [space, section].filter(Boolean);
    if (!targets.length) return;

    if (typeof Plotly === "undefined") {
      showError(targets, "The figure library could not be loaded. Check the connection and reload the page.");
      return;
    }

    let d;
    try {
      const res = await fetch(DATA_URL);
      if (!res.ok) throw new Error(DATA_URL + " returned " + res.status);
      d = toColumns(parseCsv(await res.text()));
    } catch (err) {
      showError(targets, "The figure data could not be loaded (" + err.message + "). " +
        "If the page was opened as a local file, serve it over HTTP instead, for example with quarto preview.");
      return;
    }

    if (space) drawSpace(space, d);
    if (section && select) drawSections(section, select, d);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", main);
  } else {
    main();
  }
})();
