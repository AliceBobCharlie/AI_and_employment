// move_pdf.ts -- moves the rendered PDF from _site/paper.pdf to
// _site/_pdf/paper.pdf, creating _site/_pdf/ if needed.

const FROM = "_site/paper.pdf";
const TO_DIR = "_site/_pdf";

await Deno.mkdir(TO_DIR, { recursive: true });
await Deno.rename(FROM, `${TO_DIR}/paper.pdf`);
console.log(`[ok] moved ${FROM} to ${TO_DIR}/paper.pdf`);
