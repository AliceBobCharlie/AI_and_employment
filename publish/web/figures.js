// figures.js -- draws the paper's interactive figures with plotly.js.
//
// Data: output/pca_rotated/occupation_coordinates.csv, written by pca_rotated.py. One row per
// occupation: onet_soc, title, R1 to R6. The SOC major group is read off the
// first two digits of the code (SOC_GROUPS below).
//
// The axis figures of Section 5.4 also read output/pca_rotated/axis_loadings.csv,
// written by pca_rotated.py: one row per feature variable, named by its column
// in the feature matrix (block prefix, then the O*NET element), with its
// loading on each rotated axis. BLOCKS and CTP_NAMES below turn the column into
// a block of the Content Model and a readable name.
//
// The PDF has no interactive figures; publish/pdf/print_figures.ts prints each figure,
// as drawn here in its default view, to a static PDF for it.
//
// Figure 6.2 reads output/acs_young_share/acs_young_share.csv, written by acs_young_share.py:
// one row per matched O*NET occupation, with the change in the young-worker
// share (delta) and the occupation's axis scores. Its marker size is log
// employment, scaled the way plotly express does it (area proportional to
// value, largest marker SIZE_MAX pixels across).
//
// Axis signs are fixed in pca_rotated.py by marker variables. Figure 5.8
// names each axis by its two poles, the + pole first.

(function () {
  "use strict";

  const DATA_URL = "output/pca_rotated/occupation_coordinates.csv";
  const LOADINGS_URL = "output/pca_rotated/axis_loadings.csv";
  const YOUNG_URL = "output/acs_young_share/acs_young_share.csv";
  const SIZE_MAX = 16;
  const FONT = '"Source Sans 3", "Helvetica Neue", Arial, sans-serif';

  // name: used by the axis figures of Section 5.4 and by Figure 6.2.
  // label, high, low: used by Figure 5.8 only.
  const DIMS = {
    R1: { name: "Physical against symbolic work", label: "Physical ↔ symbolic", high: "physical", low: "symbolic" },
    R2: { name: "Complex against simple work", label: "Complex ↔ simple", high: "complex", low: "simple" },
    R3: { name: "People against artefact work", label: "People ↔ artefact", high: "people", low: "artefact" },
    R4: { name: "Hands-on against field work", label: "Hands-on ↔ field", high: "hands-on", low: "field" },
    R5: { name: "Standardised against open-ended work", label: "Standardised ↔ open-ended", high: "standardised", low: "open-ended" },
    R6: { name: "Commercial against individual work", label: "Commercial ↔ individual", high: "commercial", low: "individual" },
  };

  // SOC 2018 major groups, by the first two digits of the O*NET-SOC code.
  const SOC_GROUPS = {
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
  };

  // Column prefix of the feature matrix -> block of the Content Model.
  const BLOCKS = {
    "skills_im__": "Skills", "abilities_im__": "Abilities",
    "knowledge_im__": "Knowledge", "workact_im__": "Work Activities",
    "workctx_cx__": "Work Context", "workctx_ctp__": "Work Context",
  };

  // The working-time items enter as category shares (see clean_master.py);
  // these are the names the figures show for them.
  const CTP_NAMES = {
    "Duration of Typical Work Week__c1": "Work week under 40 hours (share)",
    "Duration of Typical Work Week__c3": "Work week over 40 hours (share)",
    "Work Schedules__c2": "Irregular work schedule (share)",
    "Work Schedules__c3": "Seasonal work schedule (share)",
  };

  // The occupation space (Figure 5.8). The reader ticks one to three axes to
  // plot, taken in the order of AXES, and at most one axis to colour by.
  const AXES = ["R1", "R2", "R3", "R4", "R5", "R6"];
  const DEFAULT_AXES = ["R1", "R2", "R3"];
  const DEFAULT_COLOUR = "R4";
  const MAX_AXES = 3;

  const POINT_COLOUR = "#2a78d6";
  // Diverging: blue for +, red for -, a grey midpoint at zero that still shows
  // on the white page.
  const DIVERGING = [
    [0, "#a32b2a"], [0.25, "#e3807e"], [0.5, "#c4c3bf"], [0.75, "#6da7ec"], [1, "#104281"],
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
    const group = records.map((r) => {
      const g = SOC_GROUPS[r.onet_soc.slice(0, 2)];
      if (!g) throw new Error("no SOC major group for " + r.onet_soc);
      return g;
    });
    return {
      title: records.map((r) => r.title),
      group: group,
      R1: col("R1"), R2: col("R2"), R3: col("R3"),
      R4: col("R4"), R5: col("R5"), R6: col("R6"),
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

  // Axis title for the plane and the line of Figure 5.8: the axis between
  // its two poles, each on the side it lies.
  function axisTitle(k) {
    const dim = DIMS[k];
    return "← " + dim.low + "        " + k + "        " + dim.high + " →";
  }

  function baseLayout() {
    return {
      font: { family: FONT, size: 14, color: "#000000" },
      paper_bgcolor: "#ffffff",
      hoverlabel: { font: { family: FONT, size: 13 }, bgcolor: "#ffffff", bordercolor: "#000000" },
      margin: { l: 64, r: 16, t: 16, b: 56 },
    };
  }

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

  function maxAbs(values) {
    return Math.max(...values.map(Math.abs));
  }

  // ---------------------------------------------------------------- figure 5.8

  // The points of the occupation space. Hover shows the occupation, its SOC
  // major group, its score on each plotted axis and on the colour axis.
  function pointsTrace(d, dims, colour) {
    const shown = colour && !dims.includes(colour) ? dims.concat(colour) : dims;
    const lines = shown.map((k, i) =>
      k + " " + DIMS[k].label.toLowerCase() + ": %{customdata[" + (i + 1) + "]:+.2f}");
    return {
      text: d.title,
      customdata: d.title.map((_, i) => [d.group[i]].concat(shown.map((k) => d[k][i]))),
      hovertemplate: "<b>%{text}</b><br>%{customdata[0]}<br>" + lines.join("<br>") + "<extra></extra>",
      showlegend: false,
    };
  }

  // Where the colour bar goes. "inside": at the right edge of a 3-D scene,
  // which is narrowed to leave it room. "outside": right of a 2-D plot, in its
  // margin. "below" and "above": lying flat, for a narrow screen.
  const COLOURBAR_PLACE = {
    inside: { len: 0.55, x: 1, xanchor: "right", y: 0.5, yanchor: "middle" },
    outside: { len: 0.7, x: 1.03, xanchor: "left", y: 0.5, yanchor: "middle" },
    below: { orientation: "h", len: 0.8, x: 0.5, xanchor: "center", y: 0, yanchor: "bottom" },
    above: { orientation: "h", len: 0.8, x: 0.5, xanchor: "center", y: 1.04, yanchor: "bottom" },
  };

  // Marker colour: one colour, or diverging about zero on the colour axis,
  // with the same range on both sides so that zero is always the grey midpoint.
  function markerColour(d, colour, place) {
    if (!colour) return { color: POINT_COLOUR };
    const c = maxAbs(d[colour]);
    const flat = place === "below" || place === "above";
    const title = flat
      ? "Colour: " + colour + " " + DIMS[colour].label.toLowerCase()
      : "Colour: " + colour + "<br>" + wrap(DIMS[colour].label, 18);
    return {
      color: d[colour],
      colorscale: DIVERGING,
      cmin: -c, cmax: c, cmid: 0,
      colorbar: Object.assign({
        title: { text: title, side: "top" },
        tickvals: [-c * 0.9, 0, c * 0.9],
        ticktext: ["− " + DIMS[colour].low, "0", "+ " + DIMS[colour].high],
        thickness: 9,
        outlinewidth: 0,
        lenmode: "fraction",
        tickfont: { size: 12 },
      }, COLOURBAR_PLACE[place]),
    };
  }

  // Three axes. They share one range, centred on the origin, so the origin is
  // the centre of the scene and the point the camera looks at: the axis lines
  // then cross in the middle of the view and the vertical one stays upright.
  // All six axes are in standard deviations, so one range also keeps their
  // scales equal.
  function spaceFigure3d(d, dims, colour, narrow) {
    const [x, y, z] = dims;
    const extent = Math.max(maxAbs(d[x]), maxAbs(d[y]), maxAbs(d[z])) * 1.05;
    const labelAt = extent * 1.25;
    const range = [-labelAt * 1.05, labelAt * 1.05];

    const points = Object.assign(pointsTrace(d, dims, colour), {
      type: "scatter3d",
      mode: "markers",
      x: d[x], y: d[y], z: d[z],
      marker: Object.assign({ size: 3.5, opacity: 0.85, line: { width: 0 } },
        markerColour(d, colour, narrow ? "below" : "inside")),
    });

    // Three lines through the origin stand in for the axes; a plot box and
    // grid would hide points while the figure is rotated. Each end is labelled
    // with its pole, beyond the end of the line: placed at the line's end the
    // label falls inside the point cloud from most camera angles.
    const traces = [points];
    [x, y, z].forEach((k, j) => {
      const key = ["x", "y", "z"][j];
      const line = { x: [0, 0], y: [0, 0], z: [0, 0] };
      const ends = { x: [0, 0], y: [0, 0], z: [0, 0] };
      line[key] = [-extent, extent];
      ends[key] = [-labelAt, labelAt];
      traces.push(Object.assign({
        type: "scatter3d", mode: "lines",
        line: { color: "#555555", width: 4 },
        hoverinfo: "skip", showlegend: false,
      }, line));
      traces.push(Object.assign({
        type: "scatter3d", mode: "text",
        text: ["− " + DIMS[k].low + " (" + k + ")", "+ " + DIMS[k].high + " (" + k + ")"],
        textfont: { family: FONT, size: 13, color: "#000000" },
        hoverinfo: "skip", showlegend: false,
      }, ends));
    });

    const axis = { visible: false, range: range };
    const layout = Object.assign(baseLayout(), {
      margin: { l: 0, r: 0, t: 0, b: 0 },
      scene: {
        xaxis: axis, yaxis: axis, zaxis: axis,
        aspectmode: "cube",
        // Leaves room for the colour bar, so no axis label runs under it.
        domain: !colour ? { x: [0, 1], y: [0, 1] }
          : narrow ? { x: [0, 1], y: [0.2, 1] } : { x: [0, 0.84], y: [0, 1] },
        // On a narrow screen the camera stands further back, so the labels at
        // the ends of the horizontal axes stay inside the frame.
        camera: {
          center: { x: 0, y: 0, z: 0 },
          up: { x: 0, y: 0, z: 1 },
          eye: narrow ? { x: 1.25, y: 1.25, z: 0.78 } : { x: 0.9, y: 0.9, z: 0.56 },
        },
        // Keeps the reader's rotation when the axes shown change.
        uirevision: "space",
      },
    });
    return { data: traces, layout: layout };
  }

  function axis2d(k) {
    return {
      title: { text: axisTitle(k) },
      zeroline: false,
      showgrid: false,
      linecolor: "#000000",
      ticks: "outside",
    };
  }

  function medianRule(axisRef, value) {
    const rule = { type: "line", line: { color: "#8c8c8c", width: 1, dash: "dot" }, layer: "below" };
    return axisRef === "x"
      ? Object.assign({ xref: "x", yref: "paper", x0: value, x1: value, y0: 0, y1: 1 }, rule)
      : Object.assign({ xref: "paper", yref: "y", x0: 0, x1: 1, y0: value, y1: value }, rule);
  }

  // Layout shared by the plane and the line, with room in the margin for the
  // colour bar: at the right, or above the plot on a narrow screen.
  function layout2d(colour, narrow, extra) {
    const base = baseLayout();
    if (colour) {
      if (narrow) base.margin = Object.assign({}, base.margin, { t: 76 });
      else base.margin = Object.assign({}, base.margin, { r: 150 });
    }
    return Object.assign(base, { plot_bgcolor: "#ffffff" }, extra);
  }

  // Two axes: a plane, both axes at the same scale.
  function spaceFigure2d(d, dims, colour, narrow) {
    const [x, y] = dims;
    const trace = Object.assign(pointsTrace(d, dims, colour), {
      type: "scatter", mode: "markers",
      x: d[x], y: d[y],
      marker: Object.assign({ size: 6, opacity: colour ? 0.8 : 0.6, line: { width: 0 } },
        markerColour(d, colour, narrow ? "above" : "outside")),
    });
    const layout = layout2d(colour, narrow, {
      xaxis: axis2d(x),
      yaxis: Object.assign(axis2d(y), { scaleanchor: "x", scaleratio: 1 }),
      shapes: [medianRule("x", median(d[x])), medianRule("y", median(d[y]))],
    });
    return { data: [trace], layout: layout };
  }

  // One axis: a line. The vertical offset only separates the points and
  // carries no value.
  function spaceFigure1d(d, dims, colour, narrow) {
    const [x] = dims;
    const trace = Object.assign(pointsTrace(d, dims, colour), {
      type: "scatter", mode: "markers",
      x: d[x], y: d[x].map((_, i) => spread(i, 1)),
      marker: Object.assign({ size: 6, opacity: colour ? 0.8 : 0.6, line: { width: 0 } },
        markerColour(d, colour, narrow ? "above" : "outside")),
    });
    const layout = layout2d(colour, narrow, {
      xaxis: axis2d(x),
      // A band a third of the height, so it does not read as a second axis.
      yaxis: { visible: false, range: [-1.5, 1.5], fixedrange: true },
      shapes: [medianRule("x", median(d[x]))],
    });
    return { data: [trace], layout: layout };
  }

  function spaceFigure(d, dims, colour, narrow) {
    if (dims.length === 3) return spaceFigure3d(d, dims, colour, narrow);
    if (dims.length === 2) return spaceFigure2d(d, dims, colour, narrow);
    return spaceFigure1d(d, dims, colour, narrow);
  }

  function addBoxes(fieldset, text, ticked) {
    return AXES.map((k) => {
      const label = document.createElement("label");
      const box = document.createElement("input");
      box.type = "checkbox";
      box.value = k;
      box.checked = ticked.includes(k);
      label.append(box, " " + text(k));
      fieldset.appendChild(label);
      return box;
    });
  }

  // The axes to plot: one checkbox per axis. At most MAX_AXES can be ticked,
  // and the last one cannot be unticked, so the figure always has something
  // to draw.
  function axisPicker(fieldset, onChange) {
    const boxes = addBoxes(fieldset, (k) => k + " " + DIMS[k].label.toLowerCase(), DEFAULT_AXES);
    const update = () => {
      const ticked = boxes.filter((b) => b.checked).length;
      boxes.forEach((b) => {
        b.disabled = b.checked ? ticked === 1 : ticked >= MAX_AXES;
      });
    };
    boxes.forEach((b) => b.addEventListener("change", () => { update(); onChange(); }));
    update();
    return () => boxes.filter((b) => b.checked).map((b) => b.value);
  }

  // The axis to colour by: at most one. Ticking a box unticks the one ticked
  // before; unticking it leaves the points in one colour.
  function colourPicker(fieldset, onChange) {
    const boxes = addBoxes(fieldset, (k) => k, [DEFAULT_COLOUR]);
    boxes.forEach((box) => box.addEventListener("change", () => {
      if (box.checked) boxes.forEach((b) => { if (b !== box) b.checked = false; });
      onChange();
    }));
    return () => {
      const ticked = boxes.find((b) => b.checked);
      return ticked ? ticked.value : null;
    };
  }

  function drawSpace(el, axesBox, colourBox, d) {
    let axes = () => DEFAULT_AXES;
    let colour = () => DEFAULT_COLOUR;
    const draw = () => {
      const fig = spaceFigure(d, axes(), colour(), el.clientWidth < 600);
      return Plotly.react(el, fig.data, fig.layout, CONFIG);
    };
    if (axesBox) axes = axisPicker(axesBox, draw);
    if (colourBox) colour = colourPicker(colourBox, draw);
    enableModifierZoom(el);
    return draw();
  }

  // Plotly cancels the wheel event over a 3-D scene whether or not scrollZoom
  // is on, which traps the page when a reader scrolls onto the figure. This
  // listener runs in the capture phase, before Plotly's own: a plain wheel is
  // stopped there, so Plotly never sees it and the page scrolls normally, and
  // Ctrl/Cmd + wheel is handled here instead, moving the camera towards or
  // away from the point it looks at. When the figure is a plane or a line
  // there is no scene, and the wheel is left alone.
  function enableModifierZoom(el) {
    el.addEventListener("wheel", (ev) => {
      const camera = el.layout && el.layout.scene && el.layout.scene.camera;
      const eye = camera && camera.eye;
      if (!eye) return;

      ev.stopPropagation();
      if (!ev.ctrlKey && !ev.metaKey) return;
      ev.preventDefault();

      // The camera as the reader left it, which may differ from the layout's.
      const current = (el._fullLayout && el._fullLayout.scene &&
        el._fullLayout.scene.camera && el._fullLayout.scene.camera.eye) || eye;
      const step = clamp(1 + ev.deltaY * ZOOM_STEP, ZOOM_PER_EVENT[0], ZOOM_PER_EVENT[1]);
      const distance = Math.hypot(current.x, current.y, current.z);
      const wanted = clamp(distance * step, ZOOM_LIMITS[0], ZOOM_LIMITS[1]);
      const factor = wanted / distance;
      if (factor === 1) return;

      Plotly.relayout(el, {
        "scene.camera.eye": { x: current.x * factor, y: current.y * factor, z: current.z * factor },
      });
    }, { capture: true, passive: false });
  }

  function clamp(v, lo, hi) {
    return Math.min(Math.max(v, lo), hi);
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

  // The first column of axis_loadings.csv, unnamed, holds the feature
  // column, e.g. "skills_im__Active Learning".
  function loadingColumns(records) {
    const key = Object.keys(records[0])[0];
    const columns = records.map((r) => r[key]);
    return {
      variable: columns.map((c) => {
        const name = c.split("__").slice(1).join("__");
        return CTP_NAMES[name] || name;
      }),
      block: columns.map((c) => {
        const prefix = Object.keys(BLOCKS).find((p) => c.startsWith(p));
        if (!prefix) throw new Error("no block of the Content Model for " + c);
        return BLOCKS[prefix];
      }),
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

  async function drawAxesFigures(space, axesBox, colourBox, axisPlots) {
    const targets = [space].filter(Boolean).concat(axisPlots);
    if (!targets.length) return;
    let d;
    try {
      d = toColumns(await loadCsv(DATA_URL));
    } catch (err) {
      showError(targets, "The figure data could not be loaded (" + err.message + "). " + LOAD_HELP);
      return;
    }
    if (space) drawSpace(space, axesBox, colourBox, d);
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
    const axesBox = document.getElementById("space-picker");
    const colourBox = document.getElementById("colour-picker");
    const young = document.getElementById("plot-young");
    const axisPlots = [...document.querySelectorAll(".plot-axis[data-axis]")];
    const targets = [space, young].filter(Boolean).concat(axisPlots);
    if (!targets.length) return;

    if (typeof Plotly === "undefined") {
      showError(targets, "The figure library could not be loaded. Check the connection and reload the page.");
      return;
    }

    // Separate data files, loaded independently, so a missing file leaves
    // only its own figure empty.
    drawAxesFigures(space, axesBox, colourBox, axisPlots);
    drawYoungFigure(young);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", main);
  } else {
    main();
  }
})();