-- publish/pdf/figures.lua -- the PDF's figures.
--
-- Runs for the PDF only (the profile in _quarto-pdf.yml).
--
-- In paper_draft.qmd every figure is a raw HTML block, which the PDF would
-- otherwise drop. Each block becomes a figure here:
--
--   <img src="...">          the same image file;
--   <div class="plot" id=X>  output/figures/X.svg (or X.png for the 3-D
--                            figure), the interactive figure as figures.js
--                            draws it by default, which
--                            publish/pdf/print_figures.ts exports before the
--                            render. The SVG's text is set by Typst in the
--                            paper's own font, so the PDF embeds no other.
--
-- The caption is the block's <figcaption>, less what only makes sense on the
-- web: the "Open full screen" link and any <span class="web-only">. Text in a
-- <span class="print-only">, hidden on the web page, is kept. The figure
-- number is written in the caption, so Typst's own numbering is off. Each
-- figure floats to the top or bottom of a page, as in a printed paper, so a
-- tall one does not leave a page half empty.
--
-- Tables. Pandoc gives a table with long lines fixed column widths that add
-- up to the full text width, which sets it flush with both margins. Here every
-- column is sized to its content instead, so a table is as wide as it needs
-- and centred, as in a LaTeX paper; one too wide for the page still wraps.
--
-- Also sets the notes below tables (.table-note) in the table type size.

local FIGURE_DIR = "output/figures/"

local function exists(path)
  local f = io.open(path, "rb")
  if f then f:close() end
  return f ~= nil
end

local function image_path(html)
  local src = html:match('<img[^>]-src="([^"]+)"')
  if src then return src end

  local id = html:match('<div[^>]-class="plot[^"]*"[^>]-id="([^"]+)"')
  if not id then return nil end
  for _, ext in ipairs({ ".svg", ".png" }) do
    local path = FIGURE_DIR .. id .. ext
    if exists(path) then return path end
  end
  error("publish/pdf/figures.lua: " .. FIGURE_DIR .. id .. ".svg is missing. Render the PDF " ..
        "with --profile pdf, which exports the figures first (publish/pdf/print_figures.ts).")
end

local function caption_typst(html)
  local cap = html:match("<figcaption>(.-)</figcaption>") or ""
  cap = cap:gsub('<a class="plot%-fullscreen".-</a>', "")
  cap = cap:gsub('<span class="web%-only">.-</span>', "")
  local blocks = pandoc.read(cap, "html").blocks
  local inlines = pandoc.utils.blocks_to_inlines(blocks)
  local text = pandoc.write(pandoc.Pandoc({ pandoc.Plain(inlines) }), "typst")
  return (text:gsub("%s+$", ""))
end

function RawBlock(el)
  if el.format ~= "html" or not el.text:match("<figure") then
    return nil
  end
  local path = image_path(el.text)
  if not path then
    error("publish/pdf/figures.lua: a figure block has neither an <img> nor a plot <div>:\n" .. el.text)
  end
  return pandoc.RawBlock("typst", string.format(
    '#figure(image("%s", width: 100%%), caption: [%s], numbering: none, placement: auto)',
    path, caption_typst(el.text)))
end

function Table(tbl)
  for i, spec in ipairs(tbl.colspecs) do
    tbl.colspecs[i] = { spec[1], nil }
  end
  return tbl
end

function Div(el)
  if not el.classes:includes("table-note") then
    return nil
  end
  local out = { pandoc.RawBlock("typst", "#block(width: 100%)[#set text(size: 9pt)") }
  for _, b in ipairs(el.content) do table.insert(out, b) end
  table.insert(out, pandoc.RawBlock("typst", "]"))
  return out
end
