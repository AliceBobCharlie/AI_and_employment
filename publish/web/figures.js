// figures.js -- draws the interactive figures of the web page with plotly.js.
//
// Inputs:
//   output/pca_rotated/occupation_coordinates.csv  one row per occupation:
//       onet_soc, title, R1 to R6. occupationColumns reads it and takes the
//       SOC major group from the first two digits of onet_soc (SOC_GROUPS).
//   output/pca_rotated/axis_loadings.csv  one row per feature column, with its
//       loading on each axis. loadingColumns maps each column to a block
//       (BLOCKS) and a display name (CTP_NAMES).
//   output/acs_young_share/acs_young_share.csv  one row per occupation, with
//       the change in the young-worker share (delta), employment and R1 to R3.
//       youngColumns reads it.
//
// Outputs, drawn into elements of the page:
//   #plot-space, with the checkboxes in #space-picker and #colour-picker:
//       drawSpace plots the occupations on one, two or three axes.
//   .plot-axis[data-axis]: drawAxisFigures plots the loadings and the
//       occupation scores of the axis named in data-axis.
//   #plot-young: drawYoung plots delta against R1, R2 and R3.
//
// main finds these elements and loads each CSV separately. A CSV that fails to
// load puts an error message in its own figures only.

(function () {
  "use strict";

  const COORDINATES_URL = "output/pca_rotated/occupation_coordinates.csv";
  const LOADINGS_URL = "output/pca_rotated/axis_loadings.csv";
  const YOUNG_URL = "output/acs_young_share/acs_young_share.csv";
  const SIZE_MAX = 16;
  const FONT = '"Source Sans 3", "Helvetica Neue", Arial, sans-serif';

  // Display names of the axes. name: axis figures and #plot-young.
  // label, high, low: #plot-space.
  const AXIS_NAMES = {
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

  // Display names of the Work Context category-share columns.
  const CTP_NAMES = {
    "Duration of Typical Work Week__c1": "Work week under 40 hours (share)",
    "Duration of Typical Work Week__c3": "Work week over 40 hours (share)",
    "Work Schedules__c2": "Irregular work schedule (share)",
    "Work Schedules__c3": "Seasonal work schedule (share)",
  };

  // #plot-space: the reader ticks one to MAX_AXES axes to plot, taken in the
  // order of AXES, and at most one axis to colour by.
  const AXES = ["R1", "R2", "R3", "R4", "R5", "R6"];
  const DEFAULT_AXES = ["R1", "R2", "R3"];
  const DEFAULT_COLOUR = "R4";
  const MAX_AXES = 3;

  const POINT_COLOUR = "#2a78d6";
  // Diverging colour scale: red for -, grey at zero, blue for +.
  const DIVERGING = [
    [0, "#a32b2a"], [0.25, "#e3807e"], [0.5, "#c4c3bf"], [0.75, "#6da7ec"], [1, "#104281"],
  ];

  // scrollZoom is off, so the mouse wheel scrolls the page. enableModifierZoom
  // zooms the 3-D view on Ctrl/Cmd + wheel.
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

  // Parses CSV text into one object per row, keyed by the header. Quoted
  // fields may contain commas, newlines and doubled quotes. Rows with the
  // wrong number of fields are dropped.
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

  // Turns the rows of occupation_coordinates.csv into columns: title, group,
  // R1 to R6. The other functions call this object d.
  function occupationColumns(records) {
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

  // Marker size by area, from d.size; the largest marker is SIZE_MAX pixels.
  function markerSize(d) {
    return {
      size: d.size,
      sizemode: "area",
      sizeref: (2 * Math.max(...d.size)) / (SIZE_MAX * SIZE_MAX),
      sizemin: 1,
    };
  }

  // Axis title for the 2-D and 1-D views of #plot-space: the axis key between
  // its two poles, low pole on the left.
  function axisTitle(k) {
    const dim = AXIS_NAMES[k];
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
    // Offset of point i within a band of the given width, from a golden-ratio
    // sequence. The same i always gives the same offset.
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

  // ---------------------------------------------------------------- #plot-space

  // The occupation points. Hover shows the title, the SOC major group, and the
  // score on each plotted axis and on the colour axis.
  function pointsTrace(d, shownAxes, colour) {
    const shown = colour && !shownAxes.includes(colour) ? shownAxes.concat(colour) : shownAxes;
    const lines = shown.map((k, i) =>
      k + " " + AXIS_NAMES[k].label.toLowerCase() + ": %{customdata[" + (i + 1) + "]:+.2f}");
    return {
      text: d.title,
      customdata: d.title.map((_, i) => [d.group[i]].concat(shown.map((k) => d[k][i]))),
      hovertemplate: "<b>%{text}</b><br>%{customdata[0]}<br>" + lines.join("<br>") + "<extra></extra>",
      showlegend: false,
    };
  }

  // Colour bar positions. "inside": right edge of the 3-D scene. "outside":
  // right margin of a 2-D plot. "below" and "above": horizontal, for a
  // narrow screen.
  const COLOURBAR_PLACE = {
    inside: { len: 0.55, x: 1, xanchor: "right", y: 0.5, yanchor: "middle" },
    outside: { len: 0.7, x: 1.03, xanchor: "left", y: 0.5, yanchor: "middle" },
    below: { orientation: "h", len: 0.8, x: 0.5, xanchor: "center", y: 0, yanchor: "bottom" },
    above: { orientation: "h", len: 0.8, x: 0.5, xanchor: "center", y: 1.04, yanchor: "bottom" },
  };

  // Marker colour: POINT_COLOUR if no colour axis is ticked; otherwise the
  // DIVERGING scale on the colour axis, symmetric about zero.
  function markerColour(d, colour, place) {
    if (!colour) return { color: POINT_COLOUR };
    const c = maxAbs(d[colour]);
    const flat = place === "below" || place === "above";
    const title = flat
      ? "Colour: " + colour + " " + AXIS_NAMES[colour].label.toLowerCase()
      : "Colour: " + colour + "<br>" + wrap(AXIS_NAMES[colour].label, 18);
    return {
      color: d[colour],
      colorscale: DIVERGING,
      cmin: -c, cmax: c, cmid: 0,
      colorbar: Object.assign({
        title: { text: title, side: "top" },
        tickvals: [-c * 0.9, 0, c * 0.9],
        ticktext: ["− " + AXIS_NAMES[colour].low, "0", "+ " + AXIS_NAMES[colour].high],
        thickness: 9,
        outlinewidth: 0,
        lenmode: "fraction",
        tickfont: { size: 12 },
      }, COLOURBAR_PLACE[place]),
    };
  }

  // Three axes: a 3-D scene. All three axes share one range centred on the
  // origin, and the camera looks at the origin.
  function spaceFigure3d(d, shownAxes, colour, narrow) {
    const [x, y, z] = shownAxes;
    const extent = Math.max(maxAbs(d[x]), maxAbs(d[y]), maxAbs(d[z])) * 1.05;
    const labelAt = extent * 1.25;
    const range = [-labelAt * 1.05, labelAt * 1.05];

    const points = Object.assign(pointsTrace(d, shownAxes, colour), {
      type: "scatter3d",
      mode: "markers",
      x: d[x], y: d[y], z: d[z],
      marker: Object.assign({ size: 3.5, opacity: 0.85, line: { width: 0 } },
        markerColour(d, colour, narrow ? "below" : "inside")),
    });

    // Each axis is drawn as a line through the origin, from -extent to
    // +extent. Its pole labels sit further out, at -labelAt and +labelAt.
    // The scene's own axes, box and grid are hidden.
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
        text: ["− " + AXIS_NAMES[k].low + " (" + k + ")", "+ " + AXIS_NAMES[k].high + " (" + k + ")"],
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
        // Narrows the scene when a colour bar is shown.
        domain: !colour ? { x: [0, 1], y: [0, 1] }
          : narrow ? { x: [0, 1], y: [0.2, 1] } : { x: [0, 0.84], y: [0, 1] },
        // The camera stands further back on a narrow screen.
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

  // Layout for the 2-D and 1-D views. With a colour axis, the margin is
  // widened on the right, or at the top on a narrow screen.
  function layout2d(colour, narrow, extra) {
    const base = baseLayout();
    if (colour) {
      if (narrow) base.margin = Object.assign({}, base.margin, { t: 76 });
      else base.margin = Object.assign({}, base.margin, { r: 150 });
    }
    return Object.assign(base, { plot_bgcolor: "#ffffff" }, extra);
  }

  // Two axes: a scatter plot with equal scales and dotted median lines.
  function spaceFigure2d(d, shownAxes, colour, narrow) {
    const [x, y] = shownAxes;
    const trace = Object.assign(pointsTrace(d, shownAxes, colour), {
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

  // One axis: points along x, spread vertically by spread(), with a dotted
  // median line.
  function spaceFigure1d(d, shownAxes, colour, narrow) {
    const [x] = shownAxes;
    const trace = Object.assign(pointsTrace(d, shownAxes, colour), {
      type: "scatter", mode: "markers",
      x: d[x], y: d[x].map((_, i) => spread(i, 1)),
      marker: Object.assign({ size: 6, opacity: colour ? 0.8 : 0.6, line: { width: 0 } },
        markerColour(d, colour, narrow ? "above" : "outside")),
    });
    const layout = layout2d(colour, narrow, {
      xaxis: axis2d(x),
      // The points fill the middle third of the height.
      yaxis: { visible: false, range: [-1.5, 1.5], fixedrange: true },
      shapes: [medianRule("x", median(d[x]))],
    });
    return { data: [trace], layout: layout };
  }

  function spaceFigure(d, shownAxes, colour, narrow) {
    if (shownAxes.length === 3) return spaceFigure3d(d, shownAxes, colour, narrow);
    if (shownAxes.length === 2) return spaceFigure2d(d, shownAxes, colour, narrow);
    return spaceFigure1d(d, shownAxes, colour, narrow);
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

  // Checkboxes in #space-picker, one per axis. At most MAX_AXES can be
  // ticked, and the last ticked box cannot be unticked. Returns a function
  // that gives the ticked axes.
  function axisPicker(fieldset, onChange) {
    const boxes = addBoxes(fieldset, (k) => k + " " + AXIS_NAMES[k].label.toLowerCase(), DEFAULT_AXES);
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

  // Checkboxes in #colour-picker, one per axis. Ticking a box unticks the
  // others. Returns a function that gives the ticked axis, or null.
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

  // Draws #plot-space and redraws it with Plotly.react when a checkbox changes.
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

  // A wheel listener on #plot-space, in the capture phase, active only while
  // a 3-D scene is shown. A plain wheel event is stopped before it reaches
  // plotly, so the page scrolls. Ctrl/Cmd + wheel moves the camera towards or
  // away from the origin, by at most ZOOM_PER_EVENT per event and within
  // ZOOM_LIMITS.
  function enableModifierZoom(el) {
    el.addEventListener("wheel", (ev) => {
      const camera = el.layout && el.layout.scene && el.layout.scene.camera;
      const eye = camera && camera.eye;
      if (!eye) return;

      ev.stopPropagation();
      if (!ev.ctrlKey && !ev.metaKey) return;
      ev.preventDefault();

      // The current camera, after any rotation by the reader.
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

  // ---------------------------------------------------------------- #plot-young

  // Turns the rows of acs_young_share.csv with exact_match True into columns:
  // title, R1 to R3, delta in percentage points, employment, and size
  // (log employment).
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
    // Weighted Pearson correlation of x and y with weights w. Rows with a
    // weight of zero are left out.
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

  // Three panels, delta against R1, R2 and R3, with a shared y range. Each
  // panel is headed by the unweighted and the employment-weighted correlation.
  // The panels sit side by side, or stacked when the element is narrower than
  // 600 px.
  function drawYoung(el, d) {
    const panelAxes = ["R1", "R2", "R3"];
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

    panelAxes.forEach((k, i) => {
      const n = i === 0 ? "" : String(i + 1);
      traces.push({
        type: "scatter", mode: "markers",
        x: d[k], y: d.delta,
        xaxis: "x" + n, yaxis: "y" + n,
        text: d.title,
        customdata: d.employment,
        hovertemplate: "<b>%{text}</b><br>" + AXIS_NAMES[k].name + ": %{x:+.2f}" +
          "<br>Change in young share: %{y:+.1f} pp" +
          "<br>Employment: %{customdata:,.0f}<extra></extra>",
        marker: Object.assign(markerSize(d), { color: "#333333", opacity: 0.6, line: { width: 0 } }),
        showlegend: false,
      });

      layout["xaxis" + n] = {
        title: { text: AXIS_NAMES[k].name },
        zeroline: false, showgrid: false,
        linecolor: "#000000", ticks: "outside",
      };
      layout["yaxis" + n] = {
        title: { text: i === 0 || stacked ? "Change in young share (pp)" : "" },
        zeroline: true, zerolinecolor: "#8c8c8c", zerolinewidth: 1,
        showgrid: false,
        linecolor: "#000000", ticks: "outside",
      };
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

  // ---------------------------------------------------------------- .plot-axis

  // axisFigure builds the figure for one axis in two panels. Top: the loading
  // of every feature column, one row per block in BLOCK_ORDER. Bottom: the
  // score of every occupation, one row per SOC major group, groups ordered by
  // their median, which a bar marks. Points in a row are offset by spread().

  const BLOCK_ORDER = ["Skills", "Abilities", "Knowledge", "Work Activities", "Work Context"];
  const AXIS_BLUE = "#2a78d6";
  const AXIS_GREY = "#8c8c8c";
  const STRONG = 0.5;          // loadings counted as strong in the panel heading
  const LINE_MAX = 28;         // characters per line of a group's name: every name fits in two lines

  // Turns the rows of axis_loadings.csv into: variable (display name), block,
  // and value(k), the loadings on axis k. The first, unnamed column holds the
  // feature column name, e.g. "skills_im__Active Learning".
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

  function axisFigure(d, loadings, k) {
    const name = AXIS_NAMES[k] ? AXIS_NAMES[k].name : k;
    const traces = [];

    // ---- top panel: loadings
    const load = loadings.value(k);
    const rowOf = BLOCK_ORDER.map((_, i) => BLOCK_ORDER.length - 1 - i);   // first block on top
    traces.push({
      type: "scatter", mode: "markers",
      x: load,
      y: loadings.block.map((b, i) => rowOf[BLOCK_ORDER.indexOf(b)] + spread(i, 0.6)),
      xaxis: "x", yaxis: "y",
      text: loadings.variable,
      customdata: loadings.block,
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
    const groupMedian = Object.fromEntries(groups.map((g) => [g, median(members[g].map((i) => scores[i]))]));
    groups.sort((a, b) => groupMedian[a] - groupMedian[b]);          // highest median at the top
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
      x: groups.map((g) => groupMedian[g]),
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

  // Loads axis_loadings.csv and draws axisFigure into each .plot-axis element.
  async function drawAxisFigures(targets, d) {
    if (!targets.length) return;
    let loadings;
    try {
      loadings = loadingColumns(await loadCsv(LOADINGS_URL));
    } catch (err) {
      showError(targets, "The figure data could not be loaded (" + err.message + "). " + LOAD_HELP);
      return;
    }
    targets.forEach((el) => {
      const fig = axisFigure(d, loadings, el.dataset.axis);
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

  // Loads occupation_coordinates.csv, then draws #plot-space and the
  // .plot-axis figures.
  async function drawCoordinateFigures(space, axesBox, colourBox, axisPlots) {
    const targets = [space].filter(Boolean).concat(axisPlots);
    if (!targets.length) return;
    let d;
    try {
      d = occupationColumns(await loadCsv(COORDINATES_URL));
    } catch (err) {
      showError(targets, "The figure data could not be loaded (" + err.message + "). " + LOAD_HELP);
      return;
    }
    if (space) drawSpace(space, axesBox, colourBox, d);
    drawAxisFigures(axisPlots, d);
  }

  // Loads acs_young_share.csv and draws #plot-young.
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

    drawCoordinateFigures(space, axesBox, colourBox, axisPlots);
    drawYoungFigure(young);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", main);
  } else {
    main();
  }
})();