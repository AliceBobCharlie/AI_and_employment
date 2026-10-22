// publish/pdf/move_pdf.ts -- puts the PDF in _site/_pdf/.
//
// Quarto writes every output to the project's output folder, _site/, and has
// no option for a subfolder of it. Run after the PDF is rendered (post-render
// in _quarto-pdf.yml), this moves _site/paper.pdf to _site/_pdf/paper.pdf, so
// the PDF stays apart from the web page.

const FROM = "_site/paper.pdf";
const TO_DIR = "_site/_pdf";

await Deno.mkdir(TO_DIR, { recursive: true });
await Deno.rename(FROM, `${TO_DIR}/paper.pdf`);
console.log(`[ok] moved ${FROM} to ${TO_DIR}/paper.pdf`);
