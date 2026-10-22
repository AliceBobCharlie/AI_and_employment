// publish/pdf/print_figures.ts -- the PDF's figures, printed from the web page's own.
//
// The PDF cannot show interactive figures. This script draws each one with
// figures.js, exactly as the web page draws it when it opens, in headless
// Chrome, and prints it to output/figures/<id>.pdf, where publish/pdf/figures.lua
// finds it. Plotly draws 2-D figures as SVG, so they print as vector graphics;
// the 3-D figure is WebGL and is printed as an image at three times screen
// resolution.
//
// Quarto runs it before rendering the PDF (pre-render in _quarto-pdf.yml)
// with the Deno that ships inside Quarto, so it needs no Python or Node, only
// Chrome or Chromium. It looks for Chrome in the usual places; set CHROME to
// its path if it is somewhere else. GitHub's Ubuntu runners have Chrome.
//
// It reads the figures from paper_draft.qmd (every <div class="plot ..." id>),
// the plotly.js and font tags from _quarto.yml, so the PDF uses what the web
// page uses, and serves the project folder on a local port so that figures.js
// can load its CSV files. Offline, set PLOTLY_JS to a local plotly.min.js.
//
// By hand, from the project root:  quarto run publish/pdf/print_figures.ts

const PAPER = "paper_draft.qmd";
const QUARTO_YML = "_quarto.yml";
const OUT_DIR = "output/figures";

// The pages sit at the root of the server, like the web page, so that the
// relative paths in figures.js (figures.js itself, output/*.csv) resolve.

// Each kind of figure is laid out at a size of its own in CSS pixels, then
// printed 5.5 in wide, the width of the PDF's text block, so its type prints
// at about 14 px / 700 px of 5.5 in, 8 pt. The axis figures keep the web
// page's height (theme.scss); the three panels of the young-share figure need
// the web page's width, or their axis titles run together.
const TEXT_WIDTH_IN = 5.5;
const SIZES: Record<string, [number, number]> = {
  "plot-axis": [700, 1050],
  "plot-space": [700, 560],
  "plot-young": [860, 340],
};
const IMAGE_SCALE = 3;          // resolution of the 3-D figure, times screen
const TIMEOUT_MS = 60_000;

// ---------------------------------------------------------------- inputs

interface Figure { id: string; kind: string; tag: string; }

function figures(paper: string): Figure[] {
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

function tagsFromQuartoYml(yml: string) {
  const fonts = yml.match(/<link rel="stylesheet" href="https:\/\/fonts\.googleapis\.com[^>]*>/)?.[0] ?? "";
  const plotly = yml.match(/<script src="https:\/\/cdn\.jsdelivr\.net\/npm\/plotly[^>]*>\s*<\/script>/)?.[0];
  if (!plotly) throw new Error(`no plotly.js script tag found in ${QUARTO_YML}`);
  return { fonts, plotly };
}

function page(fig: Figure, fonts: string, plotly: string): string {
  const [w, h] = SIZES[fig.kind];
  return `<!doctype html>
<html lang="en-GB">
<head>
<meta charset="utf-8">
${fonts}
<style>
  html, body { margin: 0; padding: 0; background: #ffffff; overflow: hidden; }
  #${fig.id} { width: ${w}px; height: ${h}px; }
  .modebar-container { display: none !important; }
</style>
</head>
<body>
${fig.tag}</div>
${plotly}
<script>
  // plotly lays out text with the font it finds when it draws, so figures.js
  // is loaded only once the fonts are, or after 5 s without them (offline),
  // when figures.js falls back to the fonts it names after Source Sans 3.
  const fonts = Promise.all(['400 14px "Source Sans 3"', '600 14px "Source Sans 3"']
      .map((f) => document.fonts.load(f)));
  Promise.race([fonts, new Promise((r) => setTimeout(r, 5000))])
    .then(() => {
      const s = document.createElement("script");
      s.src = "publish/web/figures.js";
      document.body.appendChild(s);
    });
</script>
</body>
</html>
`;
}

// ---------------------------------------------------------------- local server

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
    // Files of the project folder only.
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
    // The 3-D figure needs WebGL, which has no GPU on a build machine.
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

// ---------------------------------------------------------------- printing

// Resolves once figures.js has drawn the figure, or rejects with its error.
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

// A WebGL scene does not print; replace it with an image of itself.
const SCENE_TO_IMAGE = (id: string, w: number, h: number) => `(async () => {
  const el = document.getElementById(${JSON.stringify(id)});
  if (!el._fullLayout || !el._fullLayout.scene) return false;
  const url = await Plotly.toImage(el, { format: "png", width: ${w}, height: ${h}, scale: ${IMAGE_SCALE} });
  const img = new Image(${w}, ${h});
  img.src = url;
  await img.decode();
  el.replaceChildren(img);
  return true;
})()`;

async function evaluate(b: Browser, session: string, expression: string) {
  const r = await b.send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true }, session);
  if (r.exceptionDetails) {
    throw new Error(r.exceptionDetails.exception?.description ?? r.exceptionDetails.text);
  }
  return r.result.value;
}

async function printFigure(b: Browser, origin: string, fig: Figure): Promise<Uint8Array> {
  const [w, h] = SIZES[fig.kind];
  const { targetId } = await b.send("Target.createTarget", { url: "about:blank" });
  const { sessionId } = await b.send("Target.attachToTarget", { targetId, flatten: true });
  try {
    await b.send("Emulation.setDeviceMetricsOverride",
                 { width: w, height: h, deviceScaleFactor: 1, mobile: false }, sessionId);
    await b.send("Page.navigate", { url: `${origin}/__figure__.${fig.id}.html` }, sessionId);
    await evaluate(b, sessionId, DRAWN(fig.id));
    await new Promise((r) => setTimeout(r, 500));     // the 3-D scene's first frame
    await evaluate(b, sessionId, SCENE_TO_IMAGE(fig.id, w, h));

    const scale = (TEXT_WIDTH_IN * 96) / w;
    const { data } = await b.send("Page.printToPDF", {
      paperWidth: TEXT_WIDTH_IN,
      paperHeight: (h * scale) / 96 + 0.01,
      marginTop: 0, marginBottom: 0, marginLeft: 0, marginRight: 0,
      scale, printBackground: true, pageRanges: "1",
    }, sessionId);
    return Uint8Array.from(atob(data), (c) => c.charCodeAt(0));
  } finally {
    await b.send("Target.closeTarget", { targetId });
  }
}

// ---------------------------------------------------------------- main

async function main() {
  const figs = figures(await Deno.readTextFile(PAPER));
  const plotlyFile = Deno.env.get("PLOTLY_JS");
  const tags = tagsFromQuartoYml(await Deno.readTextFile(QUARTO_YML));
  const plotly = plotlyFile ? `<script src="/__plotly__.js"></script>` : tags.plotly;
  const pages = new Map(figs.map((f) => [`/__figure__.${f.id}.html`, page(f, tags.fonts, plotly)]));

  const server = serve(pages, plotlyFile);
  const origin = `http://127.0.0.1:${server.addr.port}`;
  const chrome = await launch(findChrome());
  const browser = await Browser.connect(chrome.url);
  try {
    await Deno.mkdir(OUT_DIR, { recursive: true });
    for (const entry of Deno.readDirSync(OUT_DIR)) {
      if (entry.name.endsWith(".pdf") && !figs.some((f) => `${f.id}.pdf` === entry.name)) {
        await Deno.remove(`${OUT_DIR}/${entry.name}`);
      }
    }
    for (const fig of figs) {
      const path = `${OUT_DIR}/${fig.id}.pdf`;
      await Deno.writeFile(path, await printFigure(browser, origin, fig));
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
