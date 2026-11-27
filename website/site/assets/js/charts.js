// Hand-drawn SVG chart in the paper's visual language: thin marks, recessive
// axes, direct labels, a hover layer on every mark.
import { s, esc, showTip, hideTip, cssVar, ORIGIN, tokens, signedInt, inkOn, hexToRgb } from './util.js?v=e38b2b7f7d';

// Drawing width: narrower on phones so text stays legible when scaled down.
const cw = (container) => ((container.clientWidth || 720) < 560 ? 480 : 720);

function linear([d0, d1], [r0, r1]) {
  const f = (v) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);
  f.domain = [d0, d1];
  return f;
}
function niceTicks(lo, hi, count = 5) {
  const span = hi - lo;
  const step = 10 ** Math.floor(Math.log10(span / count));
  const err = (count / span) * step;
  const mult = err <= 0.15 ? 10 : err <= 0.35 ? 5 : err <= 0.75 ? 2 : 1;
  const inc = step * mult;
  const out = [];
  for (let v = Math.ceil(lo / inc) * inc; v <= hi + inc * 1e-9; v += inc) out.push(+v.toFixed(10));
  return out;
}

function frame(container, height, label, W) {
  const svg = s('svg', { viewBox: `0 0 ${W} ${height}`, role: 'img', 'aria-label': label });
  container.replaceChildren(svg);
  return svg;
}

function yAxis(svg, y, ticks, left, right, format) {
  const g = s('g', { class: 'grid' });
  for (const t of ticks) {
    g.append(s('line', { x1: left, x2: right, y1: y(t), y2: y(t) }));
    g.append(s('text', { x: left - 8, y: y(t) + 4, 'text-anchor': 'end' }, format(t)));
  }
  svg.append(g);
}

// ── Compression origin: stacked bars, the paper's figure ──────────────────
// entries: [{ label, total, parts: {key: value}, color? }]; values in % points.
export function originBars(container, entries, { unit = '%', height = 380, compact = false } = {}) {
  const W = cw(container);
  const H = height, L = 52, R = W - 12, T = 24, B = H - (compact ? 40 : 46);
  let maxV = 0, minV = 0;
  for (const e of entries) {
    const pos = ORIGIN.reduce((a, o) => a + Math.max(0, e.parts[o.key] || 0), 0);
    const neg = ORIGIN.reduce((a, o) => a + Math.min(0, e.parts[o.key] || 0), 0);
    maxV = Math.max(maxV, pos, e.total ?? 0); minV = Math.min(minV, neg);
  }
  const top = unit === '%' ? Math.min(100, Math.ceil((maxV + 6) / 10) * 10) : maxV * 1.12;
  const bottom = unit === '%' ? Math.floor((minV - 2) / 5) * 5 : minV * 1.12;
  const y = linear([bottom, top], [B, T]);
  const svg = frame(container, H, 'Compression origin', W);
  const fmt = unit === '%' ? (t) => `${t}%` : (t) => tokens(t);
  yAxis(svg, y, niceTicks(bottom, top, 5), L, R, fmt);
  svg.append(s('line', { x1: L, x2: R, y1: y(0), y2: y(0), stroke: cssVar('--ink-2'), 'stroke-width': 1 }));
  const band = (R - L) / entries.length;
  const bw = Math.min(64, band * 0.62);
  entries.forEach((e, i) => {
    const cx = L + band * (i + 0.5);
    let up = 0, down = 0;
    for (const o of ORIGIN) {
      const v = e.parts[o.key] || 0;
      if (!v) continue;
      const y0 = v > 0 ? y(up) : y(down), y1 = v > 0 ? y(up + v) : y(down + v);
      if (v > 0) up += v; else down += v;
      const color = cssVar(o.color);
      const hgt = Math.abs(y1 - y0);
      const rect = s('rect', { x: cx - bw / 2, y: Math.min(y0, y1) + 1, width: bw, height: Math.max(0, hgt - 2), fill: color, rx: 1.5 });
      rect.addEventListener('mousemove', (ev) => showTip(ev, `<b>${esc(e.label)}</b><div class="row"><span><i class="swatch" style="background:${color}"></i>${o.label}</span><span>${unit === '%' ? `${v.toFixed(1)} pts` : signedInt(v) + ' tokens'}</span></div>`));
      rect.addEventListener('mouseleave', hideTip);
      svg.append(rect);
      if (hgt > 16 && bw > 30) {
        svg.append(s('text', { class: 'onbar', x: cx, y: (y0 + y1) / 2 + 4, 'text-anchor': 'middle', style: `fill:${inkOn(`rgb(${hexToRgb(color)})`)}`, 'pointer-events': 'none' },
          unit === '%' ? v.toFixed(1) : tokens(v)));
      }
    }
    svg.append(s('text', { class: 'value', x: cx, y: y(Math.max(up, e.total ?? 0)) - 7, 'text-anchor': 'middle' },
      unit === '%' ? (e.total ?? up + down).toFixed(1) : tokens(e.total ?? up + down)));
    const label = s('text', { class: 'label', x: cx, y: B + 18, 'text-anchor': 'middle' }, e.label);
    svg.append(label);
    if (e.sub) svg.append(s('text', { x: cx, y: B + 32, 'text-anchor': 'middle' }, e.sub));
  });
}
