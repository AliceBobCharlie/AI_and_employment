// pdf/typst-template.typ -- the layout of the PDF.
//
// Replaces Quarto's default Typst template (template-partials in _quarto.yml).
// The layout is that of the NeurIPS preprint style, the most common layout of
// computer-science papers on arXiv:
//
//   - US Letter, a text block 5.5 in wide and 9 in long, 1.5 in from the left
//     edge and 1 in from the top (set in _quarto.yml);
//   - 10 pt type on 11 pt leading, in New Computer Modern, the face of LaTeX,
//     which ships with Typst; paragraphs unindented and 5.5 pt apart;
//   - the title 17 pt bold, centred between a 4 pt and a 1 pt rule; the
//     abstract under a centred heading, indented 0.5 in on both sides;
//   - section headings 12 pt bold, lower ones 10 pt bold;
//   - tables centred, numbered and titled above, with three horizontal rules
//     (booktabs) and no vertical ones; figure captions below the figure.
//
// The parameters are those Quarto's typst-show.typ passes; most keep Quarto's
// meaning, and the ones this layout fixes are ignored.

#let article(
  title: none,
  subtitle: none,
  authors: none,
  keywords: (),
  date: none,
  abstract-title: none,
  abstract: none,
  thanks: none,
  cols: 1,
  lang: "en",
  region: "US",
  font: none,
  fontsize: 10pt,
  title-size: 17pt,
  subtitle-size: 12pt,
  heading-family: none,
  heading-weight: "bold",
  heading-style: "normal",
  heading-color: black,
  heading-line-height: none,
  mathfont: none,
  codefont: none,
  linestretch: 1,
  sectionnumbering: none,
  linkcolor: none,
  citecolor: none,
  filecolor: none,
  toc: false,
  toc_title: none,
  toc_depth: none,
  toc_indent: 1.5em,
  doc,
) = {
  set document(title: title, keywords: keywords)
  set document(
    author: authors.map(author => content-to-string(author.name)).join(", ", last: " & "),
  ) if authors != none and authors != ()

  // ---- type
  // The text box runs from 0.7 em above the baseline to 0.3 em below it, so a
  // line is 1 em tall and the leading is what is added between lines:
  // 10 pt + 1 pt = 11 pt from baseline to baseline. Between paragraphs the
  // gap is a line and 5.5 pt: 10 pt + 6.5 pt.
  set text(lang: lang, region: region, size: fontsize,
           top-edge: 0.7em, bottom-edge: -0.3em)
  // New Computer Modern lacks a few symbols (′, ↔); its maths face, which
  // also ships with Typst, has them.
  set text(font: font + ("New Computer Modern Math",)) if font != none
  set par(justify: true, leading: 1pt * linestretch, spacing: 6.5pt)
  show math.equation: set text(font: mathfont) if mathfont != none
  show raw: set text(font: codefont) if codefont != none
  show raw: set text(size: 0.9em)

  // ---- headings
  set heading(numbering: sectionnumbering)
  show heading: set text(weight: "bold")
  show heading: set par(justify: false, leading: 2pt)
  show heading.where(level: 1): set text(size: 12pt)
  show heading.where(level: 1): set block(above: 18pt, below: 9pt)
  show heading.where(level: 2): set text(size: 10pt)
  show heading.where(level: 2): set block(above: 14pt, below: 7pt)
  show heading.where(level: 3): set text(size: 10pt)
  show heading.where(level: 3): set block(above: 11pt, below: 6pt)

  // ---- tables: booktabs. The rule under the header is Pandoc's table.hline;
  // the rules above and below are drawn round the table.
  // Rows about a line apart; the first column, which names the rows, flush
  // left, the others as the source aligns them (centred by default).
  set table(stroke: none, inset: (x: 5pt, y: 2.5pt))
  set table.hline(stroke: 0.5pt)
  show table.cell.where(x: 0): set align(left)
  show table: set text(size: 9pt)
  show table: set par(justify: false, leading: 1.5pt)
  show table: it => box(stroke: (top: 0.8pt, bottom: 0.8pt), it)

  // ---- figures and captions. A caption that fits on one line is centred; a
  // longer one is set full width, as LaTeX does.
  set figure(gap: 8pt)
  set place(clearance: 16pt)
  show figure.caption: set text(size: 9pt)
  show figure.caption: set par(leading: 1.5pt)
  show figure.caption: it => layout(size => {
    if measure(it).width <= size.width { align(center, it) } else { align(left, it) }
  })

  // ---- title block
  if title != none {
    line(length: 100%, stroke: 4pt)
    v(0.25in, weak: true)
    align(center, block({
      set par(justify: false, leading: 5pt)
      text(size: title-size, weight: "bold")[#title]
      if thanks != none {
        footnote(thanks, numbering: "*")
        counter(footnote).update(n => n - 1)
      }
      if subtitle != none {
        parbreak()
        text(size: subtitle-size)[#subtitle]
      }
    }))
    v(0.25in, weak: true)
    line(length: 100%, stroke: 1pt)
  }

  if authors != none and authors != () {
    v(0.2in)
    let ncols = calc.min(authors.len(), 3)
    grid(
      columns: (1fr,) * ncols,
      row-gutter: 1.5em,
      ..authors.map(author => align(center)[
        #text(weight: "bold")[#author.name] \
        #author.affiliation \
        #author.email
      ])
    )
  }

  if date != none {
    v(0.1in)
    align(center)[#date]
  }

  if abstract != none {
    v(22pt)
    align(center, text(size: 12pt, weight: "bold")[#abstract-title])
    v(6pt)
    pad(x: 0.5in, abstract)
    v(12pt)
  }

  if toc {
    outline(title: toc_title, depth: toc_depth, indent: toc_indent)
  }

  doc
}
