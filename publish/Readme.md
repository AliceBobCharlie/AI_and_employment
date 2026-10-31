# publish

The paper is written in `paper_draft.qmd` and rendered with
[Quarto](https://quarto.org) in two forms. The web page, with interactive
figures, is rebuilt on every push and shows the current text and data. The
PDF is built only for a [release](https://github.com/AliceBobCharlie/AI_and_employment/releases):
each release is one fixed version of the paper with its PDF attached. The
web page is the working version; the PDF the released one.

No Python runs when either is built. Both read the committed CSVs in
`output/`, so commit them again after re-running the scripts that write
them.

## Files

| file | role |
|---|---|
| `paper_draft.qmd` | the paper: title and abstract in the header, then the text |
| `figures.qmd` | a web page with the figures at full size |
| `_quarto.yml` | the web page: which files are rendered, layout, fonts, the scripts and CSVs it loads |
| `publish/web/theme.scss` | typography and colours of the web page |
| `publish/web/figures.js` | draws the interactive figures in the browser with plotly.js, from `output/occupation_coordinates.csv`, `output/axis_loadings.csv` and `output/acs_young_share.csv` |
| `_quarto-pdf.yml` | the PDF: page size, margins, type size, and the steps below |
| `publish/pdf/print_figures.ts` | before rendering: opens each interactive figure in headless Chrome, drawn by `figures.js`, and exports it to `output/figures/` as SVG (PNG for the 3-D figure) |
| `publish/pdf/figures.lua` | during rendering: puts each figure into the PDF with its print caption |
| `publish/pdf/move_pdf.ts` | after rendering: moves the PDF from `_site/` to `_site/_pdf/` |
| `publish/pdf/typst-template.typ` | the PDF layout: US Letter, a 5.5 × 9 in text block, 10 pt New Computer Modern, booktabs tables |

The figures are HTML blocks in `paper_draft.qmd`, one source for both forms.
In a caption, text inside `<span class="web-only">` is left out of the PDF
and text inside `<span class="print-only">` appears only in the PDF.

## Building locally

Install Quarto, then from the repository root:

```
quarto render                                          # web page -> _site/
quarto render paper_draft.qmd --profile pdf --to typst # PDF -> _site/_pdf/paper.pdf
```

The PDF also needs Chrome or Chromium; set `CHROME` to its path if it is not
in the usual place. `print_figures.ts` runs on the Deno that ships with
Quarto. Use `quarto preview` to view the web page: opened straight from
disk, `_site/index.html` shows no figures, because browsers block a local
file from loading the CSVs.

## Automatic publishing

- `.github/workflows/publish.yml`: every push to `main` renders the web page
  and deploys it to GitHub Pages. The rendered site is never committed.
  Requires Settings → Pages → Source set to *GitHub Actions*.
- `.github/workflows/release.yml`: publishing a release renders the PDF from
  the release's tag and attaches it as `paper.pdf`. To release a version:

  ```
  gh release create v1.0 --title "v1.0" --notes "What changed."
  ```

The web page links the PDF of the latest release from its table of contents.