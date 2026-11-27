// Layered dependency-graph viewer on canvas.
//
// Form: protected results in the top lane, every other declaration in a
// dependency layer below the declarations that use it, so edges read
// downward (LeanLean's Sugiyama-style preprocessing layout, computed offline).
//
// CPU budget: nothing simulates and nothing animates. A frame is drawn only
// when the view, selection or data changes. Edges are pre-bundled offline and
// batched into three Path2D objects by bundle weight; nodes are batched into
// one Path2D per colour, rebuilt only when the zoom level changes.
//
// A node's area is its size in Lean tokens; its shape is its kind: theorems
// are circles, definitions squares, anything else (notation, macros, axioms)
// triangles.
import { cssVar } from './util.js?v=e38b2b7f7d';

export const STATUS = ['unchanged', 'rewritten', 'structural', 'deleted', 'dead', 'comments'];

// 0 circle, 1 square, 2 triangle.
const SHAPE = { theorem: 0, def: 1, inductive: 1, opaque: 1, instance: 1 };
export const shapeOf = (kind) => SHAPE[kind] ?? 2;

// Half the width of each shape, per unit of radius.
const HALF_WIDTH = [1, 0.886, 1.347];

// Area proportional to Lean tokens for all but the largest declarations,
// which saturate smoothly towards `cap` (larger stays larger).
export const radius = (tokens, cap) => Math.max(1.6, cap * Math.tanh(0.32 * Math.sqrt(tokens) / cap));

// A shape of the same area as a circle of radius r, centred on (x, y).
function shape(path, s, x, y, r) {
  if (s === 0) { path.moveTo(x + r, y); path.arc(x, y, r, 0, Math.PI * 2); return; }
  if (s === 1) { const a = r * 0.886; path.rect(x - a, y - a, 2 * a, 2 * a); return; }
  const R = r * 1.555;
  path.moveTo(x, y - R); path.lineTo(x + R * 0.866, y + R / 2); path.lineTo(x - R * 0.866, y + R / 2); path.closePath();
}

// Mix two CSS colours (weight t of a) into an opaque one.
let colorCtx = null;
function rgb(color) {
  colorCtx ||= document.createElement('canvas').getContext('2d');
  colorCtx.fillStyle = '#000'; colorCtx.fillStyle = color;
  const c = colorCtx.fillStyle;
  return c[0] === '#' ? [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16)) : c.match(/[\d.]+/g).slice(0, 3).map(Number);
}
function mix(a, b, t) {
  const [p, q] = [rgb(a), rgb(b)];
  return `rgb(${p.map((v, i) => Math.round(v * t + q[i] * (1 - t))).join(',')})`;
}

function curve(path, x1, y1, x2, y2) {
  path.moveTo(x1, y1);
  const dx = x2 - x1, dy = y2 - y1;
  if (Math.abs(dy) < 16 && Math.abs(dx) > 18) {
    const bend = (dx >= 0 ? -1 : 1) * Math.min(46, 12 + Math.abs(dx) * 0.08);
    path.bezierCurveTo(x1 + dx * 0.28, y1 + bend, x1 + dx * 0.72, y2 + bend, x2, y2);
  } else {
    const mid = (y1 + y2) / 2;
    path.bezierCurveTo(x1, mid, x2, mid, x2, y2);
  }
}
function curveAt(x1, y1, x2, y2, t) {
  const dx = x2 - x1, dy = y2 - y1;
  let c1, c2;
  if (Math.abs(dy) < 16 && Math.abs(dx) > 18) {
    const bend = (dx >= 0 ? -1 : 1) * Math.min(46, 12 + Math.abs(dx) * 0.08);
    c1 = [x1 + dx * 0.28, y1 + bend]; c2 = [x1 + dx * 0.72, y2 + bend];
  } else { const mid = (y1 + y2) / 2; c1 = [x1, mid]; c2 = [x2, mid]; }
  const u = 1 - t;
  const p = [u * u * u * x1 + 3 * u * u * t * c1[0] + 3 * u * t * t * c2[0] + t * t * t * x2,
    u * u * u * y1 + 3 * u * u * t * c1[1] + 3 * u * t * t * c2[1] + t * t * t * y2];
  const d = [3 * u * u * (c1[0] - x1) + 6 * u * t * (c2[0] - c1[0]) + 3 * t * t * (x2 - c2[0]),
    3 * u * u * (c1[1] - y1) + 6 * u * t * (c2[1] - c1[1]) + 3 * t * t * (y2 - c2[1])];
  return [p, Math.atan2(d[1], d[0])];
}

export class GraphView {
  constructor(wrap, { onSelect, onHover, onView } = {}) {
    this.wrap = wrap;
    this.base = document.createElement('canvas');
    this.base.className = 'base';
    this.over = document.createElement('canvas');
    this.over.style.pointerEvents = 'none';
    wrap.append(this.base, this.over);
    this.onSelect = onSelect || (() => {});
    this.onHover = onHover || (() => {});
    this.onView = onView || (() => {});
    this.view = { k: 1, x: 0, y: 0 };
    this.phase = 'all'; // 'before' | 'after' | 'all'
    this.showEdges = true;
    // Room the fitted graph leaves for controls floating over the canvas.
    this.pad = { top: 52, bottom: 32 };
    this.selected = -1;
    this.hover = -1;
    this.frame = 0;
    this.pointers = new Map();
    this._bind();
    this.resizeObserver = new ResizeObserver(() => { this.resize(); if (!this.fitted && this.N) { this.fit(); this.fitted = true; } });
    this.resizeObserver.observe(wrap);
    window.__leanleanGraph = this; // handy for profiling from the console
  }

  destroy() {
    this.resizeObserver.disconnect();
    cancelAnimationFrame(this.frame);
    clearTimeout(this.idleTimer);
    this.base.remove(); this.over.remove();
  }

  setData(graph, pair) {
    const nodes = graph.nodes;
    const n = nodes.name.length;
    const added = pair ? pair.added : [];
    const N = n + added.length;
    Object.assign(this, { n, N, graph, pair });
    // Added declarations open lanes in the layers below their users; in the
    // After and Complete views the baseline lanes below them move down.
    const shift = pair?.shift || [];
    const shiftAt = (yy) => { let d = 0; for (const [t, dy] of shift) if (yy > t) d = dy; return d; };
    // No shape grows wider than its slot in a lane, so neighbours never overlap.
    this.cap = HALF_WIDTH.map((f) => ((graph.spacing ?? 14) / 2 - 1) / f);
    const x = new Float32Array(N), y0 = new Float32Array(N), y1 = new Float32Array(N);
    const rBefore = new Float32Array(N), rAfter = new Float32Array(N), shapes = new Uint8Array(N);
    for (let i = 0; i < n; i++) {
      x[i] = nodes.x[i]; y0[i] = nodes.y[i]; y1[i] = nodes.y[i] + shiftAt(nodes.y[i]);
      shapes[i] = shapeOf(graph.kinds[nodes.kind[i]]);
      rBefore[i] = radius(nodes.tokens[i], this.cap[shapes[i]]); rAfter[i] = pair ? radius(pair.after[i], this.cap[shapes[i]]) : rBefore[i];
    }
    added.forEach((a, k) => {
      x[n + k] = a.x; y0[n + k] = y1[n + k] = a.y; shapes[n + k] = shapeOf(a.kind); rAfter[n + k] = radius(a.tokens, this.cap[shapes[n + k]]);
    });
    Object.assign(this, { x, y0, y1, rBefore, rAfter, shapes });
    this.protected = new Set(nodes.protected);

    const pairs = [...graph.edges, ...(pair ? pair.added_edges : [])];
    const m = pairs.length / 2;
    const outStart = new Uint32Array(N + 1), inStart = new Uint32Array(N + 1);
    for (let e = 0; e < m; e++) { outStart[pairs[2 * e] + 1]++; inStart[pairs[2 * e + 1] + 1]++; }
    for (let i = 0; i < N; i++) { outStart[i + 1] += outStart[i]; inStart[i + 1] += inStart[i]; }
    const outAdj = new Uint32Array(m), inAdj = new Uint32Array(m);
    const oc = outStart.slice(), ic = inStart.slice();
    for (let e = 0; e < m; e++) { const u = pairs[2 * e], d = pairs[2 * e + 1]; outAdj[oc[u]++] = d; inAdj[ic[d]++] = u; }
    Object.assign(this, { outStart, outAdj, inStart, inAdj, edgeCount: m });

    const [w, h] = graph.world;
    const lanes = graph.lanes.map((l) => ({ y: l[0], label: l[1], status: l[2], count: l[3] }));
    const addedLanes = (pair?.added_lanes || []).map((l) => ({ y: l[0], label: l[1], status: l[2], count: l[3] }));
    this.lanes = lanes;
    // Baseline and added lanes together, in the shifted layout.
    this.mergedLanes = [...lanes.map((l) => ({ ...l, y: l.y + shiftAt(l.y) })), ...addedLanes].sort((a, b) => a.y - b.y);
    // Lanes of declarations removed by preprocessing exist only before it.
    const kept = lanes.filter((l) => l.status !== 'removed');
    const grown = shift.length ? shift[shift.length - 1][1] : 0;
    const keptEnd = kept.length < lanes.length ? kept[kept.length - 1].y + 44 : h + grown;
    const bottom = addedLanes.filter((l) => l.status === 'added');
    const addedEnd = bottom.length ? bottom[bottom.length - 1].y + 24 : null;
    this.worlds = { before: [w, h], after: [w, addedEnd ?? keptEnd], all: [w, addedEnd ?? h + grown] };

    if (this.selected >= N) this.selected = -1;
    this.layoutPhase();
    this.rebuild();
    if (!this.fitted && this.w) { this.fit(); this.fitted = true; }
  }

  // Positions, sizes and the hit grid of the current phase.
  layoutPhase() {
    const before = this.phase === 'before';
    this.y = before ? this.y0 : this.y1;
    this.r = new Float32Array(this.N);
    for (let i = 0; i < this.N; i++) {
      const c = this.pair ? this.category(i) : 'unchanged';
      this.r[i] = before || c === 'deleted' || c === 'dead' ? this.rBefore[i]
        : this.phase === 'after' || c === 'added' ? this.rAfter[i] : Math.max(this.rBefore[i], this.rAfter[i]);
    }
    // The exported world includes space for lane labels. On phones, fit the
    // visible declarations themselves so that spare space cannot offset them.
    let left = Infinity, right = -Infinity, top = Infinity, bottom = -Infinity;
    for (let i = 0; i < this.N; i++) {
      if (!this.visible(i)) continue;
      const rx = this.r[i] * HALF_WIDTH[this.shapes[i]];
      const ry = this.r[i] * (this.shapes[i] === 2 ? 1.555 : 1);
      left = Math.min(left, this.x[i] - rx); right = Math.max(right, this.x[i] + rx);
      top = Math.min(top, this.y[i] - ry); bottom = Math.max(bottom, this.y[i] + ry);
    }
    this.nodeBounds = Number.isFinite(left) ? [left, top, right - left, bottom - top] : null;
    this.cell = 28;
    this.grid = new Map();
    for (let i = 0; i < this.N; i++) {
      const key = `${Math.floor(this.x[i] / this.cell)},${Math.floor(this.y[i] / this.cell)}`;
      if (!this.grid.has(key)) this.grid.set(key, []);
      this.grid.get(key).push(i);
    }
  }

  // In the Complete view a modified declaration is two-toned: pale at its
  // size before, solid at its size after, the larger drawn behind the smaller.
  twoTone(i) {
    return this.phase === 'all' && this.pair && Math.abs(this.rBefore[i] - this.rAfter[i]) >= 0.25 && this.category(i) === 'modified';
  }

  // Fill one node (overlay drawing), two-toned where it applies.
  fillNode(ctx, i, k) {
    const draw = (color, rr) => { const p = new Path2D(); shape(p, this.shapes[i], this.x[i], this.y[i], this.shown(i, k, rr)); ctx.fillStyle = color; ctx.fill(p); };
    if (!this.twoTone(i)) { draw(this.nodeColor[i], this.r[i]); return; }
    const shrank = this.rBefore[i] > this.rAfter[i];
    draw(shrank ? this.colors.pale : this.nodeColor[i], this.r[i]);
    draw(shrank ? this.nodeColor[i] : this.colors.pale, Math.min(this.rBefore[i], this.rAfter[i]));
  }

  // On-screen radius: at least 1.3px when zoomed out, but never past the shape's cap.
  shown(i, k, r = this.r[i]) { return Math.max(r, Math.min(1.3 / k, this.cap[this.shapes[i]])); }

  outline(i, rr) { const p = new Path2D(); shape(p, this.shapes[i], this.x[i], this.y[i], rr); return p; }

  visible(i) {
    if (i >= this.n) return this.phase !== 'before';
    if (this.phase === 'after') return !this.pair || this.pair.status[i] < 3;
    return true;
  }

  // Edge bundles (and added edges in the After view) as three weight classes.
  rebuildEdges() {
    const { x, y } = this;
    const classes = [new Path2D(), new Path2D(), new Path2D()];
    const b = this.graph.bundles;
    for (let k = 0; k < b.length; k += 3) {
      const u = b[k], d = b[k + 1], c = b[k + 2];
      if (!this.visible(u) || !this.visible(d)) continue;
      curve(classes[c >= 8 ? 2 : c >= 2 ? 1 : 0], x[u], y[u], x[d], y[d]);
    }
    if (this.pair && this.phase !== 'before') {
      const a = this.pair.added_edges;
      for (let k = 0; k < a.length; k += 2) curve(classes[0], x[a[k]], y[a[k]], x[a[k + 1]], y[a[k + 1]]);
    }
    this.edgePaths = classes;
  }

  // A node's state: unchanged, modified, comments (only comments or layout
  // changed), deleted, dead (deleted and unused), or added.
  category(i) {
    if (!this.pair) return 'unchanged';
    if (i >= this.n) return 'added';
    const status = this.pair.status[i];
    if (status === 5) return 'comments';
    if (status === 4) return 'dead';
    if (status === 3) return 'deleted';
    if (status === 0) return 'unchanged';
    return 'modified';
  }

  colorOf(i) { return this.colors[this.category(i)]; }

  rebuild() {
    this.colors = {
      unchanged: cssVar('--g-flat'), modified: cssVar('--g-modified'), comments: cssVar('--g-comments'), deleted: cssVar('--g-deleted'),
      dead: cssVar('--g-dead'), added: cssVar('--g-added'),
      ink: cssVar('--ink'), edge: cssVar('--muted'),
      surface: cssVar('--surface'), accent: cssVar('--accent'), users: cssVar('--g-users'), muted: cssVar('--muted'),
      sunk: cssVar('--sunk'),
    };
    this.colors.pale = mix(this.colors.modified, this.colors.surface, 0.38);
    this.nodeColor = Array.from({ length: this.N }, (_, i) => this.colorOf(i));
    this.rebuildEdges();
    this.nodeK = null;
    this.request();
  }

  // Node paths depend on zoom (radii keep a minimum on-screen size).
  nodePaths() {
    const k = this.view.k;
    if (this.nodeK === k) return this.cachedNodes;
    // Outer shapes keyed by size class and colour; inner (two-tone) shapes by colour.
    const outer = new Map(), inner = new Map(), ring = new Path2D();
    const put = (map, key, i, rr) => {
      if (!map.has(key)) map.set(key, new Path2D());
      shape(map.get(key), this.shapes[i], this.x[i], this.y[i], rr);
    };
    for (let i = 0; i < this.N; i++) {
      if (!this.visible(i)) continue;
      const rr = this.shown(i, k);
      const color = this.nodeColor[i];
      const size = this.r[i] > 6 ? 0 : this.r[i] > 3.5 ? 1 : 2;
      if (this.twoTone(i)) {
        const shrank = this.rBefore[i] > this.rAfter[i];
        put(outer, `${size}|${shrank ? this.colors.pale : color}`, i, rr);
        put(inner, shrank ? color : this.colors.pale, i, this.shown(i, k, Math.min(this.rBefore[i], this.rAfter[i])));
      } else put(outer, `${size}|${color}`, i, rr);
      if (this.protected.has(i)) shape(ring, this.shapes[i], this.x[i], this.y[i], rr + Math.max(1.6, 2.2 / k));
    }
    // Large nodes first so small ones stay visible; within a size, unchanged first so changes sit on top.
    const rank = (key) => { const [size, c] = key.split('|'); return Number(size) * 2 + (c === this.colors.unchanged ? 0 : 1); };
    const order = [...outer.keys()].sort((a, b) => rank(a) - rank(b));
    this.cachedNodes = { fills: [...order.map((key) => [key.split('|')[1], outer.get(key)]), ...inner], ring };
    this.nodeK = k;
    return this.cachedNodes;
  }

  setPhase(phase) {
    this.phase = phase;
    if (this.selected >= 0 && !this.visible(this.selected)) this.selected = -1;
    this.layoutPhase();
    this.rebuild();
    this.fit();
    this.onSelect(this.selected);
  }
  setEdges(on) { this.showEdges = on; this.request(); }
  world() { return this.worlds[this.phase]; }
  phaseLanes() {
    if (this.phase === 'before') return this.lanes;
    return this.phase === 'after' ? this.mergedLanes.filter((l) => l.status !== 'removed') : this.mergedLanes;
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const { clientWidth: w, clientHeight: h } = this.wrap;
    if (!w || !h) return;
    for (const c of [this.base, this.over]) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
    Object.assign(this, { dpr, w, h });
    this.request();
  }

  fitBounds() { return this.fitToNodes && this.nodeBounds ? this.nodeBounds : [0, 0, ...this.world()]; }
  fitK() {
    const [, , ww, wh] = this.fitBounds();
    return Math.min((this.w - 24) / ww, (this.h - this.pad.top - this.pad.bottom) / wh);
  }
  fit() {
    if (!this.w || !this.graph) return;
    const [left, top, ww, wh] = this.fitBounds();
    const k = this.fitK();
    this.view = { k, x: (this.w - ww * k) / 2 - left * k,
      y: this.pad.top + Math.max(0, (this.h - this.pad.top - this.pad.bottom - wh * k) / 2) - top * k };
    this.request();
  }
  focus(i) {
    if (i < 0) return;
    const k = Math.max(this.view.k, Math.min(1.6, this.fitK() * 6), 0.9);
    this.view = { k, x: this.w / 2 - this.x[i] * k, y: this.h / 2 - this.y[i] * k };
    this.request();
  }
  zoomBy(factor, cx = this.w / 2, cy = this.h / 2) {
    const k = Math.max(this.fitK() * 0.5, Math.min(this.view.k * factor, 14));
    const f = k / this.view.k;
    this.view = { k, x: cx - (cx - this.view.x) * f, y: cy - (cy - this.view.y) * f };
    this.poke();
  }

  // Mark a gesture in progress; the full frame follows once input settles.
  poke() {
    this.interacting = true;
    clearTimeout(this.idleTimer);
    this.idleTimer = setTimeout(() => { this.interacting = false; this.request(); }, 150);
    this.request();
  }

  select(i, { focus = false } = {}) {
    this.selected = i;
    if (focus && i >= 0) this.focus(i);
    this.request();
    this.onSelect(i);
  }

  neighbours(i) {
    const deps = [], users = [];
    for (let p = this.outStart[i]; p < this.outStart[i + 1]; p++) if (this.visible(this.outAdj[p])) deps.push(this.outAdj[p]);
    for (let p = this.inStart[i]; p < this.inStart[i + 1]; p++) if (this.visible(this.inAdj[p])) users.push(this.inAdj[p]);
    return { deps, users };
  }

  label(i) { return i >= this.n ? this.pair.added[i - this.n].name : this.graph.nodes.name[i]; }

  pick(px, py) {
    const k = this.view.k;
    const wx = (px - this.view.x) / k, wy = (py - this.view.y) / k;
    const slack = 5 / k;
    const cx = Math.floor(wx / this.cell), cy = Math.floor(wy / this.cell);
    const reach = Math.max(1, Math.ceil(slack / this.cell));
    let best = -1, bestD = Infinity;
    for (let gx = cx - reach; gx <= cx + reach; gx++) for (let gy = cy - reach; gy <= cy + reach; gy++) {
      const bucket = this.grid.get(`${gx},${gy}`);
      if (!bucket) continue;
      for (const i of bucket) {
        if (!this.visible(i)) continue;
        const d = Math.hypot(this.x[i] - wx, this.y[i] - wy) - this.shown(i, k);
        if (d < slack && d < bestD) { best = i; bestD = d; }
      }
    }
    return best;
  }

  _bind() {
    const c = this.base;
    let down = null, moved = false;
    c.addEventListener('pointerdown', (e) => {
      c.setPointerCapture(e.pointerId);
      this.pointers.set(e.pointerId, [e.offsetX, e.offsetY]);
      down = { x: e.offsetX, y: e.offsetY, vx: this.view.x, vy: this.view.y };
      moved = false;
    });
    c.addEventListener('pointermove', (e) => {
      if (this.pointers.has(e.pointerId) && this.pointers.size === 2) {
        // Two fingers: the midpoint pans, the spread zooms around it.
        const [a, b] = [...this.pointers.values()];
        const before = Math.hypot(a[0] - b[0], a[1] - b[1]);
        this.pointers.set(e.pointerId, [e.offsetX, e.offsetY]);
        const [a2, b2] = [...this.pointers.values()];
        // Two fingers zoom about their midpoint and pan as the midpoint moves.
        const mx = (a2[0] + b2[0]) / 2, my = (a2[1] + b2[1]) / 2;
        this.view = { ...this.view, x: this.view.x + mx - (a[0] + b[0]) / 2, y: this.view.y + my - (a[1] + b[1]) / 2 };
        if (before > 0) this.zoomBy(Math.hypot(a2[0] - b2[0], a2[1] - b2[1]) / before, mx, my);
        else this.poke();
        moved = true;
        return;
      }
      if (down && this.pointers.has(e.pointerId)) {
        const dx = e.offsetX - down.x, dy = e.offsetY - down.y;
        if (Math.abs(dx) + Math.abs(dy) > 3) moved = true;
        if (moved) { this.wrap.classList.add('dragging'); this.view = { ...this.view, x: down.vx + dx, y: down.vy + dy }; this.poke(); }
        return;
      }
      const hit = this.pick(e.offsetX, e.offsetY);
      if (hit !== this.hover) { this.hover = hit; this.drawOverlay(); }
      this.onHover(hit, e);
      c.style.cursor = hit >= 0 ? 'pointer' : '';
    });
    const end = (e) => {
      if (!this.pointers.has(e.pointerId)) return;
      this.pointers.delete(e.pointerId);
      this.wrap.classList.remove('dragging');
      if (down && !moved && this.pointers.size === 0) this.select(this.pick(e.offsetX, e.offsetY));
      if (this.pointers.size === 0) down = null;
      // Lifting one finger of a pinch: the other keeps dragging from where it is.
      else if (this.pointers.size === 1) { const [p] = this.pointers.values(); down = { x: p[0], y: p[1], vx: this.view.x, vy: this.view.y }; }
    };
    c.addEventListener('pointerup', end);
    c.addEventListener('pointercancel', end);
    c.addEventListener('pointerleave', () => { if (this.hover !== -1) { this.hover = -1; this.drawOverlay(); } this.onHover(-1); });
    c.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.zoomBy(Math.exp(-e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0018)), e.offsetX, e.offsetY);
    }, { passive: false });
    c.addEventListener('dblclick', (e) => { const i = this.pick(e.offsetX, e.offsetY); if (i >= 0) this.focus(i); else this.zoomBy(1.8, e.offsetX, e.offsetY); });
  }

  request() {
    if (this.frame) return;
    this.frame = requestAnimationFrame(() => { this.frame = 0; this.draw(); });
  }

  drawLanes(ctx) {
    const { k, x, y } = this.view;
    const [ww] = this.world();
    const lanes = this.phaseLanes();
    // Background bands only for removed or added sections.
    let i = 0;
    while (i < lanes.length) {
      let j = i;
      while (j + 1 < lanes.length && lanes[j + 1].status === lanes[i].status) j++;
      const top = (lanes[i].y - 17) * k + y, bottom = (lanes[j].y + 17) * k + y;
      if (!['kept', 'inserted', 'protected'].includes(lanes[i].status)) {
        ctx.fillStyle = this.colors.sunk;
        ctx.fillRect(x, top, ww * k, bottom - top);
      }
      i = j + 1;
    }
  }

  drawLaneLabels(ctx) {
    const { k, x, y } = this.view;
    const lanes = this.phaseLanes();
    ctx.font = '500 11px "IBM Plex Sans", system-ui, sans-serif';
    ctx.textBaseline = 'bottom';
    const lx = Math.max(x + 8, 10);
    // A label names its layer and counts the declarations across its wrapped lanes.
    const totals = new Map();
    let current = null;
    for (const lane of lanes) {
      if (lane.status === 'inserted') continue; // added declarations in a layer's lanes count as added, not in the layer
      if (lane.label) current = lane;
      totals.set(current, (totals.get(current) || 0) + lane.count);
    }
    for (const lane of lanes) {
      // Only the removed and added sections are named; protected results and dependency layers go unlabelled.
      if (!lane.label || lane.status === 'protected' || lane.status === 'kept') continue;
      const ly = lane.y * k + y - 6;
      if (ly < 16 || ly > this.h - 4) continue;
      const text = `${lane.label} · ${totals.get(lane)}`;
      const tw = ctx.measureText(text).width;
      ctx.globalAlpha = 0.88; ctx.fillStyle = this.colors.surface;
      ctx.fillRect(lx - 4, ly - 13, tw + 8, 15);
      ctx.globalAlpha = 1; ctx.fillStyle = this.colors.muted; ctx.fillText(text, lx, ly);
    }
  }

  draw() {
    if (!this.w || !this.graph) return;
    const ctx = this.base.getContext('2d');
    const { k, x, y } = this.view;
    const dpr = this.dpr;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = this.colors.surface;
    ctx.fillRect(0, 0, this.base.width, this.base.height);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.drawLanes(ctx);
    ctx.setTransform(k * dpr, 0, 0, k * dpr, x * dpr, y * dpr);
    const dim = this.selected >= 0;
    // Edges stay at hairline width (the rasteriser's fast path); bundle
    // weight is carried by opacity. Heavy graphs skip edges mid-gesture.
    if (this.showEdges && !(this.interacting && this.graph.bundles.length > 12000)) {
      ctx.strokeStyle = this.colors.edge;
      ctx.lineWidth = 0.9 / k;
      [0.13, 0.24, 0.42].forEach((alpha, c) => {
        ctx.globalAlpha = dim ? alpha * 0.4 : alpha;
        ctx.stroke(this.edgePaths[c]);
      });
    }
    const paths = this.nodePaths();
    ctx.globalAlpha = dim ? 0.3 : 1;
    for (const [color, path] of paths.fills) { ctx.fillStyle = color; ctx.fill(path); }
    ctx.globalAlpha = dim ? 0.4 : 1;
    ctx.strokeStyle = this.colors.ink;
    ctx.lineWidth = 1.3 / k;
    ctx.stroke(paths.ring);
    ctx.globalAlpha = 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (!this.interacting) this.drawLabels(ctx);
    this.drawLaneLabels(ctx);
    this.drawOverlay();
    this.onView(k);
  }

  drawLabels(ctx) {
    const { k, x, y } = this.view;
    if (k < 0.75 && this.selected < 0) {
      // Zoomed out: only protected results get names.
      if (k < 0.35) return;
    }
    ctx.font = '500 10.5px "IBM Plex Sans", system-ui, sans-serif';
    ctx.textBaseline = 'middle';
    const candidates = [];
    for (let i = 0; i < this.N; i++) {
      if (!this.visible(i)) continue;
      const isProtected = this.protected.has(i);
      if (k < 0.75 && !isProtected) continue;
      const sx = this.x[i] * k + x, sy = this.y[i] * k + y;
      if (sx < -40 || sy < 16 || sx > this.w + 40 || sy > this.h + 10) continue;
      candidates.push([(isProtected ? 1e6 : 0) + this.r[i], i, sx, sy]);
    }
    candidates.sort((a, b) => b[0] - a[0]);
    const taken = [];
    let drawn = 0;
    for (const [, i, sx, sy] of candidates) {
      if (drawn > 120) break;
      const full = this.label(i);
      const text = full.split('.').pop() || full;
      const w = ctx.measureText(text).width;
      // Labels sit above-right of the node, angled space permitting.
      const box = [sx + 4, sy - 16, w + 4, 12];
      if (taken.some((t) => box[0] < t[0] + t[2] && box[0] + box[2] > t[0] && box[1] < t[1] + t[3] && box[1] + box[3] > t[1])) continue;
      taken.push(box);
      ctx.lineWidth = 3; ctx.strokeStyle = this.colors.surface; ctx.strokeText(text, box[0], sy - 10);
      ctx.fillStyle = this.selected >= 0 ? this.colors.muted : this.colors.ink;
      ctx.fillText(text, box[0], sy - 10);
      drawn++;
    }
  }

  arrow(ctx, x1, y1, x2, y2, t, size) {
    const [[px, py], angle] = curveAt(x1, y1, x2, y2, t);
    ctx.beginPath();
    ctx.moveTo(px, py);
    ctx.lineTo(px - size * Math.cos(angle - 0.5), py - size * Math.sin(angle - 0.5));
    ctx.lineTo(px - size * Math.cos(angle + 0.5), py - size * Math.sin(angle + 0.5));
    ctx.closePath();
    ctx.fill();
  }

  drawOverlay() {
    const ctx = this.over.getContext('2d');
    const { k, x, y } = this.view;
    const dpr = this.dpr;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.over.width, this.over.height);
    if (!this.N) return;
    ctx.setTransform(k * dpr, 0, 0, k * dpr, x * dpr, y * dpr);
    const s = this.selected;
    if (s >= 0 && this.visible(s)) {
      const { deps, users } = this.neighbours(s);
      const size = 5.5 / Math.max(k, 0.5);
      for (const [list, color, outgoing] of [[users, this.colors.users, false], [deps, this.colors.accent, true]]) {
        const path = new Path2D();
        for (const o of list) outgoing ? curve(path, this.x[s], this.y[s], this.x[o], this.y[o]) : curve(path, this.x[o], this.y[o], this.x[s], this.y[s]);
        ctx.strokeStyle = color; ctx.lineWidth = 1.6 / k; ctx.globalAlpha = 0.9; ctx.stroke(path);
        ctx.fillStyle = color;
        for (const o of list.slice(0, 400)) outgoing ? this.arrow(ctx, this.x[s], this.y[s], this.x[o], this.y[o], 0.82, size) : this.arrow(ctx, this.x[o], this.y[o], this.x[s], this.y[s], 0.82, size);
        ctx.globalAlpha = 1;
        for (const o of list) {
          this.fillNode(ctx, o, k);
          ctx.lineWidth = 1.5 / k; ctx.strokeStyle = color; ctx.stroke(this.outline(o, this.shown(o, k)));
        }
      }
      this.fillNode(ctx, s, k);
      ctx.lineWidth = 2.2 / k; ctx.strokeStyle = this.colors.ink; ctx.stroke(this.outline(s, this.shown(s, k) + 4 / k));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.font = '500 11px "IBM Plex Sans", system-ui, sans-serif';
      ctx.textBaseline = 'middle';
      const taken = [];
      for (const i of [s, ...deps.slice(0, 60), ...users.slice(0, 60)]) {
        const sx = this.x[i] * k + x + 6, sy = this.y[i] * k + y - 10;
        const text = this.label(i).split('.').pop() || this.label(i);
        const w = ctx.measureText(text).width;
        if (i !== s && taken.some((t) => sx < t[0] + t[2] && sx + w > t[0] && Math.abs(sy - t[1]) < 12)) continue;
        taken.push([sx, sy, w]);
        ctx.lineWidth = 3; ctx.strokeStyle = this.colors.surface; ctx.strokeText(text, sx, sy);
        ctx.fillStyle = this.colors.ink; ctx.fillText(text, sx, sy);
      }
      ctx.setTransform(k * dpr, 0, 0, k * dpr, x * dpr, y * dpr);
    }
    const hv = this.hover;
    if (hv >= 0 && hv !== s) {
      ctx.lineWidth = 2 / k; ctx.strokeStyle = this.colors.ink; ctx.stroke(this.outline(hv, this.shown(hv, k) + 3 / k));
    }
  }
}
