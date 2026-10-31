// print_figures.ts -- exports each interactive figure of the paper as a static
// image, drawn by publish/web/figures.js in headless Chrome.
//
// Inputs:
//   paper_draft.qmd   every <div class="plot ..." id="..."> is one figure
//                     (figureDivs). Its class picks its size from SIZES.
//   _quarto.yml       the plotly.js <script> tag (plotlyTag).
//   PLOTLY_JS         optional environment variable: a local plotly.min.js,
//                     used in place of the tag.
//   CHROME            optional environment variable: the Chrome or Chromium
//                     binary. Otherwise findChrome looks in the usual places.
//
// Output, in output/figures/ (emptied first):
//   <id>.svg   a 2-D figure. inPaperFont sets every font-family to FIGURE_FONT.
//   <id>.png   a 3-D figure, at IMAGE_SCALE times its size.
//
// Steps: figurePage writes one HTML page per figure. serve serves these pages
// and the project folder on a local port. launch starts headless Chrome, and
// Browser talks to it over the DevTools Protocol. exportFigure opens each page,
// waits for the figure (DRAWN), exports it with Plotly.toImage (EXPORT), and
// main writes the file.
//
// Run from the project root:  quarto run publish/pdf/print_figures.ts

const PAPER = "paper_draft.qmd";
const QUARTO_YML = "_quarto.yml";
const OUT_DIR = "output/figures";
const FIGURE_FONT = "New Computer Modern";

// Width and height in CSS pixels for each figure class.
const SIZES: Record<string, [number, number]> = {
  "plot-axis": [700, 1050],
  "plot-space": [700, 560],
  "plot-young": [860, 340],
};
const IMAGE_SCALE = 3;          // PNG pixels per CSS pixel
const TIMEOUT_MS = 60_000;

// ---------------------------------------------------------------- inputs

interface Figure { id: string; kind: string; tag: string; }

function figureDivs(paper: string): Figure[] {
  const out: Figure[] = [];
  for (const m of paper.matchAll(/<div class="plot ([^"]*)"[^>]*>/g)) {
    const tag = m[0];
    const id = tag.match(/\bid="([^"]+)"/)?.[1];
    const kind = m[1].split(/\s+/).find((c) => c in SIZES);
    if (!id || !kind) throw new Error(`a figure div ${PAPER} cannot place: ${tag}`);
    out.push({ id, kind, tag });
  }
  if (!out.length) throw new Error(`no figure divs found in ${PAPER}`);
  return out;
}

function plotlyTag(yml: string): string {
  const tag = yml.match(/<script src="https:\/\/cdn\.jsdelivr\.net\/npm\/plotly[^>]*>\s*<\/script>/)?.[0];
  if (!tag) throw new Error(`no plotly.js script tag found in ${QUARTO_YML}`);
  return tag;
}

// One HTML page that holds the figure div, plotly.js and figures.js.
function figurePage(fig: Figure, plotly: string): string {
  const [w, h] = SIZES[fig.kind];
  return `<!doctype html>
<html lang="en-GB">
<head>
<meta charset="utf-8">
<style>
  html, body { margin: 0; padding: 0; background: #ffffff; overflow: hidden; }
  #${fig.id} { width: ${w}px; height: ${h}px; }
</style>
</head>
<body>
${fig.tag}</div>
${plotly}
<script src="publish/web/figures.js"></script>
</body>
</html>
`;
}

// ---------------------------------------------------------------- local server

// Serves the generated pages at /__figure__.<id>.html, a local plotly.js at
// /__plotly__.js, and any other path from the project folder. Pages sit at the
// server root, so the relative CSV paths in figures.js resolve.

const TYPES: Record<string, string> = {
  ".js": "application/javascript", ".csv": "text/csv; charset=utf-8",
  ".html": "text/html; charset=utf-8", ".css": "text/css", ".png": "image/png",
};

function serve(pages: Map<string, string>, plotlyFile: string | undefined) {
  const root = Deno.cwd();
  return Deno.serve({ hostname: "127.0.0.1", port: 0, onListen() {} }, async (req) => {
    const path = decodeURIComponent(new URL(req.url).pathname);
    if (pages.has(path)) {
      return new Response(pages.get(path), { headers: { "content-type": TYPES[".html"] } });
    }
    if (path === "/__plotly__.js" && plotlyFile) {
      return new Response(await Deno.readFile(plotlyFile), { headers: { "content-type": TYPES[".js"] } });
    }
    // Paths containing ".." are refused.
    const file = root + path;
    if (path.includes("..")) return new Response("forbidden", { status: 403 });
    try {
      const ext = path.slice(path.lastIndexOf("."));
      return new Response(await Deno.readFile(file),
                          { headers: { "content-type": TYPES[ext] ?? "application/octet-stream" } });
    } catch {
      return new Response("not found", { status: 404 });
    }
  });
}

// ---------------------------------------------------------------- Chrome

function findChrome(): string {
  const env = Deno.env.get("CHROME");
  if (env) return env;
  const candidates = {
    linux: ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser"],
    darwin: ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
             "/Applications/Chromium.app/Contents/MacOS/Chromium"],
    windows: ["C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
              "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe"],
  }[Deno.build.os as "linux" | "darwin" | "windows"] ?? [];
  const path = (Deno.env.get("PATH") ?? "").split(Deno.build.os === "windows" ? ";" : ":");
  for (const c of candidates) {
    const full = c.includes("/") || c.includes("\\") ? [c] : path.map((d) => `${d}/${c}`);
    for (const f of full) {
      try { if (Deno.statSync(f).isFile) return f; } catch { /* not here */ }
    }
  }
  throw new Error("Chrome or Chromium not found. Install it, or set CHROME to its path.");
}

// A minimal client for the Chrome DevTools Protocol, over its WebSocket.
class Browser {
  private ws: WebSocket;
  private next = 1;
  private waiting = new Map<number, { resolve: (v: any) => void; reject: (e: Error) => void }>();

  private constructor(ws: WebSocket) {
    this.ws = ws;
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      const w = msg.id && this.waiting.get(msg.id);
      if (!w) return;
      this.waiting.delete(msg.id);
      if (msg.error) w.reject(new Error(msg.error.message));
      else w.resolve(msg.result);
    };
  }

  static async connect(url: string): Promise<Browser> {
    const ws = new WebSocket(url);
    await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });
    return new Browser(ws);
  }

  send(method: string, params: object = {}, sessionId?: string): Promise<any> {
    const id = this.next++;
    this.ws.send(JSON.stringify({ id, method, params, sessionId }));
    return new Promise((resolve, reject) => this.waiting.set(id, { resolve, reject }));
  }

  close() { this.ws.close(); }
}

async function launch(chrome: string) {
  const profile = await Deno.makeTempDir({ prefix: "print-figures-" });
  const args = [
    "--headless=new", "--remote-debugging-port=0", `--user-data-dir=${profile}`,
    "--no-first-run", "--no-default-browser-check", "--hide-scrollbars",
    // Software WebGL, for the 3-D figure on a machine without a GPU.
    "--enable-unsafe-swiftshader",
  ];
  if (Deno.build.os === "linux" && Deno.uid() === 0) args.push("--no-sandbox");
  const proc = new Deno.Command(chrome, { args, stdout: "null", stderr: "piped" }).spawn();

  // Chrome announces its DevTools address on stderr.
  const reader = proc.stderr.pipeThrough(new TextDecoderStream()).getReader();
  let text = "";
  const deadline = Date.now() + TIMEOUT_MS;
  while (Date.now() < deadline) {
    const { value, done } = await reader.read();
    if (done) break;
    text += value;
    const m = text.match(/DevTools listening on (ws:\/\/\S+)/);
    if (m) {
      reader.releaseLock();
      proc.stderr.cancel().catch(() => {});
      return { proc, profile, url: m[1] };
    }
  }
  proc.kill();
  throw new Error("Chrome did not start:\n" + text);
}

// ---------------------------------------------------------------- export

// Page script: resolves when the figure holds a .main-svg, rejects on a
// .plot-error or after TIMEOUT_MS.
const DRAWN = (id: string) => `new Promise((resolve, reject) => {
  const t0 = Date.now();
  (function check() {
    const el = document.getElementById(${JSON.stringify(id)});
    const failed = el && el.querySelector(".plot-error");
    if (failed) return reject(new Error(failed.textContent));
    if (el && el.querySelector(".main-svg")) return resolve(true);
    if (Date.now() - t0 > ${TIMEOUT_MS}) {
      const loaded = [...document.scripts].map((s) => s.src).filter(Boolean).join(", ");
      return reject(new Error("timed out; plotly.js " +
        (typeof Plotly === "undefined" ? "not loaded" : "loaded") + "; scripts: " + loaded +
        "; figure holds: " + (el ? el.innerHTML.slice(0, 200) : "nothing")));
    }
    setTimeout(check, 100);
  })();
})`;

// Page script: Plotly.toImage of the figure, PNG if it has a 3-D scene,
// otherwise SVG. Returns the extension and a data URL.
const EXPORT = (id: string, w: number, h: number) => `(async () => {
  const el = document.getElementById(${JSON.stringify(id)});
  const scene = !!(el._fullLayout && el._fullLayout.scene);
  const url = await Plotly.toImage(el, scene
    ? { format: "png", width: ${w}, height: ${h}, scale: ${IMAGE_SCALE} }
    : { format: "svg", width: ${w}, height: ${h} });
  return { ext: scene ? "png" : "svg", url };
})()`;

async function evaluate(b: Browser, session: string, expression: string) {
  const r = await b.send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true }, session);
  if (r.exceptionDetails) {
    throw new Error(r.exceptionDetails.exception?.description ?? r.exceptionDetails.text);
  }
  return r.result.value;
}

// Replaces every font-family in the SVG, in style and attribute form, with
// FIGURE_FONT.
function inPaperFont(svg: string): string {
  return svg
    .replace(/font-family:[^;"]*/g, `font-family: '${FIGURE_FONT}'`)
    .replace(/font-family="[^"]*"/g, `font-family="${FIGURE_FONT}"`);
}

async function exportFigure(b: Browser, origin: string, fig: Figure) {
  const [w, h] = SIZES[fig.kind];
  const { targetId } = await b.send("Target.createTarget", { url: "about:blank" });
  const { sessionId } = await b.send("Target.attachToTarget", { targetId, flatten: true });
  try {
    await b.send("Emulation.setDeviceMetricsOverride",
                 { width: w, height: h, deviceScaleFactor: 1, mobile: false }, sessionId);
    await b.send("Page.navigate", { url: `${origin}/__figure__.${fig.id}.html` }, sessionId);
    await evaluate(b, sessionId, DRAWN(fig.id));
    await new Promise((r) => setTimeout(r, 500));     // waits for the first WebGL frame
    const { ext, url } = await evaluate(b, sessionId, EXPORT(fig.id, w, h));
    const data = url.slice(url.indexOf(",") + 1);
    const bytes = ext === "svg"
      ? new TextEncoder().encode(inPaperFont(decodeURIComponent(data)))
      : Uint8Array.from(atob(data), (c) => c.charCodeAt(0));
    return { ext, bytes };
  } finally {
    await b.send("Target.closeTarget", { targetId });
  }
}

// ---------------------------------------------------------------- main

async function main() {
  const figs = figureDivs(await Deno.readTextFile(PAPER));
  const plotlyFile = Deno.env.get("PLOTLY_JS");
  const plotly = plotlyFile ? `<script src="/__plotly__.js"></script>`
                            : plotlyTag(await Deno.readTextFile(QUARTO_YML));
  const pages = new Map(figs.map((f) => [`/__figure__.${f.id}.html`, figurePage(f, plotly)]));

  const server = serve(pages, plotlyFile);
  const origin = `http://127.0.0.1:${server.addr.port}`;
  const chrome = await launch(findChrome());
  const browser = await Browser.connect(chrome.url);
  try {
    await Deno.remove(OUT_DIR, { recursive: true }).catch(() => {});
    await Deno.mkdir(OUT_DIR, { recursive: true });
    for (const fig of figs) {
      const { ext, bytes } = await exportFigure(browser, origin, fig);
      const path = `${OUT_DIR}/${fig.id}.${ext}`;
      await Deno.writeFile(path, bytes);
      console.log(`[ok] wrote ${path}`);
    }
  } finally {
    browser.close();
    chrome.proc.kill();
    await chrome.proc.status;
    await Deno.remove(chrome.profile, { recursive: true }).catch(() => {});
    await server.shutdown();
  }
}

await main();
