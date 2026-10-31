-- figures.lua -- Pandoc filter that turns the paper's HTML figures and tables
-- into Typst for the PDF.
--
-- RawBlock: an HTML block holding <figure> becomes a Typst #figure.
--   image_path picks the image: the src of an <img>, or for
--   <div class="plot" id=X> the file FIGURE_DIR/X.svg or X.png. It raises an
--   error if neither exists.
--   caption_typst converts the <figcaption> to Typst. It removes the
--   <a class="plot-fullscreen"> link and every <span class="web-only">, and
--   keeps <span class="print-only">.
--   The figure is unnumbered and placed with placement: auto.
-- Table: drops the column widths, so each column is sized to its content.
-- Div: a .table-note div is wrapped in a 9 pt text block.

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
        "with --profile pdf, or run publish/pdf/print_figures.ts first.")
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
