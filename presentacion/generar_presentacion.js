// Presentación Taller RL — tema "Modern Minimalist" (theme-factory). node build.js <repo>
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require("C:/Users/User/.claude/skills/synced/e0e53bc7-d63e-4ef0-ba3c-109bd8731fb5_cbc81a8d-ec2b-40c9-b0ad-2cdeb69ffb97/pptx/scripts/apply_theme.js");

const REPO = process.argv[2];
const FIG = (f) => path.join(REPO, "informe", "figuras", f);
const OUT = process.argv[3] || path.join(REPO, "presentacion", "presentacion_taller_rl.pptx");

// Modern Minimalist: charcoal / slate / light gray / white. Color solo en datos: Q-L azul, SARSA naranja (= figuras).
const HEX = { charcoal: "36454F", slate: "708090", light: "D3D3D3", white: "FFFFFF", tint: "F2F3F4", ql: "2A78D6", sa: "EE6A33" };
const THEME = {
  name: "Modern Minimalist",
  headFontFace: "Arial",   // theme-factory pide DejaVu Sans; Arial es su equivalente seguro en Office
  bodyFontFace: "Arial",
  colors: {
    dk1: HEX.charcoal, lt1: HEX.white, dk2: HEX.slate, lt2: HEX.tint,
    accent1: HEX.ql, accent2: HEX.sa, accent3: HEX.slate, accent4: HEX.light, accent5: HEX.charcoal, accent6: "A9B4BC",
    hlink: HEX.ql, folHlink: HEX.slate,
  },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.title = "Taller RL: agente trader con Q-Learning y SARSA";
pres.author = "Cristian Camilo, Daniel Gallego, Marco Corpa";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const W = 13.333, M = 0.6, CW = W - 2 * M;

pres.defineSlideMaster({
  title: "PORTADA",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 1.7, w: 7.4, h: 2.2, fontSize: 40, bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENIDO",
  background: { color: C.background1 },
  margin: [0.4, M, 0.5, M],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.35, w: CW, h: 0.85, fontSize: 32, bold: true, color: C.text1, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: "Taller RL · Q-Learning y SARSA · 29/10/2026", options: { x: M, y: 7.0, w: 8, h: 0.3, fontSize: 10, color: C.text2, margin: 0 } } },
  ],
  slideNumber: { x: W - M - 0.6, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: C.text2, align: "right" },
});

const T = (slide, text, opts) => slide.addText(text, { isTextBox: true, margin: 0, ...opts });

// ---------- 1. Portada ----------
pres.addSection({ title: "Portada" });
let s = pres.addSlide({ masterName: "PORTADA", sectionTitle: "Portada" });
s.addText("Agente trader con\nQ-Learning y SARSA", { placeholder: "title" });
T(s, "Aprendizaje por refuerzo tabular en 4 mercados sintéticos y en BTC real", { x: M, y: 4.1, w: 7.4, h: 0.8, fontSize: 18, color: C.accent4, objectName: "subtitulo" });
T(s, "Cristian Camilo  ·  Daniel Gallego  ·  Marco Corpa", { x: M, y: 6.1, w: 7.4, h: 0.4, fontSize: 16, bold: true, color: C.background1, objectName: "autores" });
T(s, "Taller RL  ·  29 de octubre de 2026", { x: M, y: 6.5, w: 7.4, h: 0.35, fontSize: 12, color: C.accent4, objectName: "fecha" });
[["4", "mercados sintéticos"], ["2", "algoritmos: Q-Learning y SARSA"], ["1 140", "entrenamientos reproducibles"]].forEach(([n, l], i) => {
  const y = 1.25 + i * 1.75;
  T(s, n, { x: 8.9, y, w: 3.8, h: 0.95, fontSize: 54, bold: true, color: C.background1, objectName: `cifra_${i}` });
  T(s, l, { x: 8.9, y: y + 0.95, w: 3.8, h: 0.4, fontSize: 14, color: C.accent4, objectName: `cifra_label_${i}` });
});
s.addNotes("Habla: todos (presentación rápida). Ver chuleta.");

// ---------- 2. Qué aprende (Marco) ----------
pres.addSection({ title: "Qué aprende" });
s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Qué aprende" });
s.addText("Aprende la estrategia que cada mercado permite", { placeholder: "title" });
const imgW = CW, imgH = imgW * 1069 / 2723;
s.addImage({ path: FIG("fig_senal_politica.png"), x: M, y: 1.35, w: imgW, h: imgH, objectName: "fig_senal_politica",
  altText: "Arriba: señal real por bin de tendencia en los 4 mercados. Abajo: política aprendida por Q-Learning y SARSA." });
const chips = ["Momentum", "No operar", "Comprar caídas, vender subidas", "Salir en los extremos"];
const colW = (CW - 0.3 * 3) / 4;
chips.forEach((t, i) => {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M + i * (colW + 0.3), y: 6.3, w: colW, h: 0.5, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.1, objectName: `chip_${i}` });
  T(s, t, { x: M + i * (colW + 0.3), y: 6.3, w: colW, h: 0.5, fontSize: 14, bold: true, color: C.text1, align: "center", valign: "middle", objectName: `chip_txt_${i}` });
});
s.addNotes("Habla: Marco. Estado = (bin de tendencia por cuantiles, posición): 20 estados × 3 acciones. Arriba la señal real, abajo la política aprendida: coinciden.");

// ---------- 3. Evolución (Dani) ----------
pres.addSection({ title: "Evolución" });
s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Evolución" });
s.addText("La clave fue α, no entrenar más", { placeholder: "title" });
s.addChart(pres.charts.BAR, [{ name: "Q-Learning", labels: ["α = 0,05", "α = 0,01", "α = 0,005", "α = 0,002"], values: [0, 11.6, 17.2, 21.6] }], {
  x: M, y: 1.4, w: 6.6, h: 5.3, barDir: "col", objectName: "grafico_alpha",
  chartColors: [HEX.light, HEX.slate, HEX.slate, HEX.ql], barGapWidthPct: 45,
  showTitle: true, title: "Reversión a la media · log-retorno en test (%)", titleFontSize: 14, titleColor: HEX.charcoal, titleFontFace: "+mn-lt",
  showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 16, dataLabelFontBold: true, dataLabelColor: HEX.charcoal, dataLabelFontFace: "+mn-lt",
  dataLabelFormatCode: "+0.0;-0.0;0",
  catAxisLabelColor: HEX.charcoal, catAxisLabelFontSize: 14, catAxisLabelFontFace: "+mn-lt", catAxisLineShow: false,
  valAxisHidden: true, valAxisMaxVal: 25, valAxisMinVal: 0, valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false,
});
const pasos = [
  ["1", "Por defecto (α = 0,05)", "20 de 20 agentes no operan nunca"],
  ["2", "¿Más episodios? N = 10 000", "Siguen sin operar: no es la duración"],
  ["3", "Bajar α, mismo N", "+23,7 pp en reversión"],
];
const cx = 7.75, cw = W - M - cx, ch = 1.35;
pasos.forEach(([n, h, d], i) => {
  const y = 1.5 + i * (ch + 0.3);
  const last = i === pasos.length - 1;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: cx, y, w: cw, h: ch, fill: { color: last ? C.text1 : C.background2 }, line: { color: last ? C.text1 : C.background2 }, rectRadius: 0.12, objectName: `paso_${n}` });
  s.addShape(pres.shapes.OVAL, { x: cx + 0.3, y: y + (ch - 0.6) / 2, w: 0.6, h: 0.6, fill: { color: last ? C.accent1 : C.text1 }, line: { color: last ? C.accent1 : C.text1 }, objectName: `paso_circulo_${n}` });
  T(s, n, { x: cx + 0.3, y: y + (ch - 0.6) / 2, w: 0.6, h: 0.6, fontSize: 20, bold: true, color: C.background1, align: "center", valign: "middle", objectName: `paso_num_${n}` });
  T(s, h, { x: cx + 1.15, y: y + 0.25, w: cw - 1.35, h: 0.45, fontSize: 18, bold: true, color: last ? C.background1 : C.text1, objectName: `paso_titulo_${n}` });
  T(s, d, { x: cx + 1.15, y: y + 0.75, w: cw - 1.35, h: 0.45, fontSize: 15, color: last ? C.accent4 : C.text2, objectName: `paso_desc_${n}` });
});
T(s, "Ruido en Q > ventaja por paso → deja de probar «comprar»", { x: cx, y: 6.35, w: cw, h: 0.35, fontSize: 12, italic: true, color: C.text2, objectName: "mecanismo" });
s.addNotes("Habla: Dani. Qué probamos, qué falló, qué cambiamos y qué aprendimos. Ver chuleta.");

// ---------- 4. Sobre el precio (Cristian) ----------
pres.addSection({ title: "Sobre el precio" });
s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Sobre el precio" });
s.addText("Así opera sobre el precio", { placeholder: "title" });
const pW = CW, pH = pW * 1304 / 2977;
s.addImage({ path: path.join(REPO, "presentacion", "fig_precio_senales.png"), x: (W - pW) / 2, y: 1.3, w: pW, h: pH, objectName: "fig_precio_senales",
  altText: "Precio con las compras y ventas de Q-Learning en una serie de test de los mercados cíclico, reversión y regímenes, y en BTC real durante 2025." });
s.addNotes("Habla: Cristian. Triángulo azul = compra, rojo = venta, sombreado = en posición. Q-Learning, semilla mediana, serie de test.");

// ---------- 5. Resultados (Cristian) ----------
pres.addSection({ title: "Resultados" });
s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Resultados" });
s.addText("Opera donde hay señal; se queda fuera donde no", { placeholder: "title" });
const tiles = [
  ["Tendencia + ciclo", "+610", "Momentum · Buy & Hold +338", true],
  ["Reversión a la media", "+21,6", "Buy & Hold −1,1", true],
  ["Browniano", "≈ 0", "Sin señal → no opera (correcto)", false],
  ["Regímenes", "≈ 0", "Evita pérdidas · Buy & Hold −21,9", false],
];
const tw = (CW - 0.3 * 3) / 4, th = 2.65;
tiles.forEach(([mk, num, sub, sig], i) => {
  const x = M + i * (tw + 0.3), y = 1.4;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: tw, h: th, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.12, objectName: `tile_${i}` });
  T(s, mk, { x: x + 0.25, y: y + 0.2, w: tw - 0.5, h: 0.35, fontSize: 14, bold: true, color: C.text2, objectName: `tile_mercado_${i}` });
  T(s, num, { x: x + 0.25, y: y + 0.6, w: tw - 0.5, h: 0.9, fontSize: 44, bold: true, color: sig ? C.accent1 : C.text1, objectName: `tile_num_${i}` });
  T(s, sub, { x: x + 0.25, y: y + 1.5, w: tw - 0.5, h: 0.5, fontSize: 12, color: C.text2, valign: "top", objectName: `tile_sub_${i}` });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + 0.25, y: y + th - 0.6, w: tw - 0.5, h: 0.4, fill: { color: sig ? C.text1 : C.background1 }, line: { color: C.text1, width: 1 }, rectRadius: 0.2, objectName: `tile_chip_${i}` });
  T(s, sig ? "Supera al efectivo" : "Igual que efectivo", { x: x + 0.25, y: y + th - 0.6, w: tw - 0.5, h: 0.4, fontSize: 12, bold: true, color: sig ? C.background1 : C.text1, align: "center", valign: "middle", objectName: `tile_chip_txt_${i}` });
});
T(s, "Log-retorno medio en test (pp) frente a no operar · Q-Learning · IC95 bootstrap sobre 100 series", { x: M, y: 4.15, w: CW, h: 0.3, fontSize: 11, color: C.text2, objectName: "nota_tiles" });

const bw = (CW - 0.3) / 2, by = 4.7, bh = 1.95;
// Q-L ≈ SARSA
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: by, w: bw, h: bh, fill: { color: C.background1 }, line: { color: C.accent4, width: 1 }, rectRadius: 0.12, objectName: "caja_qs" });
s.addShape(pres.shapes.OVAL, { x: M + 0.35, y: by + 0.42, w: 0.45, h: 0.45, fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: "punto_ql" });
T(s, "≈", { x: M + 0.85, y: by + 0.32, w: 0.45, h: 0.6, fontSize: 30, bold: true, color: C.text1, align: "center", valign: "middle", objectName: "aprox" });
s.addShape(pres.shapes.OVAL, { x: M + 1.35, y: by + 0.42, w: 0.45, h: 0.45, fill: { color: C.accent2 }, line: { color: C.accent2 }, objectName: "punto_sarsa" });
T(s, "Q-Learning ≈ SARSA", { x: M + 2.1, y: by + 0.35, w: bw - 2.35, h: 0.55, fontSize: 20, bold: true, color: C.text1, valign: "middle", objectName: "qs_titulo" });
T(s, "Ninguna diferencia significativa en los 4 mercados.\nBatir a Buy & Hold no prueba nada: en regímenes, no operar ya lo bate.", { x: M + 0.35, y: by + 1.0, w: bw - 0.7, h: 0.8, fontSize: 13, color: C.text2, valign: "top", objectName: "qs_desc" });
// BTC
const bx = M + bw + 0.3;
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: bx, y: by, w: bw, h: bh, fill: { color: C.text1 }, line: { color: C.text1 }, rectRadius: 0.12, objectName: "caja_btc" });
T(s, "Datos reales · BTC 2023–2026", { x: bx + 0.35, y: by + 0.25, w: bw - 0.7, h: 0.4, fontSize: 14, bold: true, color: C.accent4, objectName: "btc_titulo" });
[["+82 %", "Q-Learning"], ["+381 %", "Buy & Hold"], ["−11 %", "en 2022 (B&H −44 %)"]].forEach(([n, l], i) => {
  const x = bx + 0.35 + i * ((bw - 0.7) / 3);
  T(s, n, { x, y: by + 0.7, w: (bw - 0.7) / 3, h: 0.6, fontSize: 28, bold: true, color: i === 0 ? C.background1 : C.accent4, objectName: `btc_num_${i}` });
  T(s, l, { x, y: by + 1.3, w: (bw - 0.7) / 3 - 0.1, h: 0.45, fontSize: 12, color: C.accent4, valign: "top", objectName: `btc_label_${i}` });
});
s.addNotes("Habla: Cristian. Resultados frente al efectivo, Q-L ≈ SARSA, BTC y conclusión. Ver chuleta.");

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("OK", OUT);
})();
