import { h, s, logo, logoMarkup, modelColor, modelLabel, EFFORTS, highestReasoningModels, cssVar, pct, usd, hours, tokens, ramp, inkOn, esc, showTip, hideTip, AUTHORS, PAPER_TITLE, PAPER_URL, CODE_URL, DATASET_URL, CITATION, CONTACT_EMAIL, ORIGIN } from './util.js?v=e38b2b7f7d';

function sectionHead(eyebrow, title, text) {
  return h('div', { class: 'section-head' },
    h('div', {}, eyebrow ? h('span', { class: 'eyebrow' }, eyebrow) : null, h('h2', {}, title), text ? h('p', {}, text) : null));
}

function boardTable(bench) {
  let models = bench.models;
  // Largest first, by Lean tokens after preprocessing; the size bands follow in that order.
  const repos = [...bench.repositories].sort((a, c) => c.tokens - a.tokens);
  const bands = bench.bands.filter((b) => repos.some((r) => r.band === b.band))
    .sort((a, b) => repos.findIndex((r) => r.band === a.band) - repos.findIndex((r) => r.band === b.band));
  let sortKey = 'score', dir = -1;
  const getters = { score: (m) => m.score, cost: (m) => m.cost, hours: (m) => m.hours };
  const tbody = h('tbody');
  const firsts = new Set(bands.map((b) => repos.find((r) => r.band === b.band).id));

  const cell = (m, repo) => {
    const r = bench.results[repo.id]?.[m.key];
    // Runs without graphs (failed submissions, Leanstral) open on their trace.
    const view = r?.explorer ? 'graph' : 'trace';
    const go = () => { location.href = `viewer.html#/${view}/${repo.id}/${m.short}`; };
    // A failed submission scores 0; where its size was measured, the cell still shows what it tried to remove.
    const attempted = r && r.status !== 'passed' && r.post > 0 ? (1 - r.post / repo.tokens) * 100 : null;
    const tip = (e) => showTip(e, `<b>${repos.indexOf(repo) + 1} · ${esc(repo.name)}</b> · ${esc(modelLabel(m))}${r?.status === 'pending'
      ? '<div>Not finished yet · not counted in the score</div>' : r && r.status === 'passed'
      ? `<div class="row"><span>Score</span><span>${pct(r.score)}</span></div><div class="row"><span>Tokens</span><span>${tokens(repo.tokens)} → ${tokens(r.post)}</span></div><div class="row"><span>Cost</span><span>${usd(r.cost)}</span></div><div class="row"><span>Time</span><span>${hours(r.seconds / 3600)}</span></div>`
      : `<div>${r ? 'Failed verification' : 'No run'} · scored 0</div>${attempted != null
        ? `<div class="row"><span>Attempted</span><span>${pct(attempted)}</span></div><div class="row"><span>Tokens</span><span>${tokens(repo.tokens)} → ${tokens(r.post)}</span></div>` : ''}`}`);
    const cls = firsts.has(repo.id) ? 'rc first' : 'rc';
    const tag = { 'data-repo': repo.id, 'data-model': m.short };
    const figure = (v) => (v >= 99.95 ? '100' : v.toFixed(v < 10 ? 1 : 0));
    // A run without a verdict yet (a model still being evaluated) is left out of its score, not failed.
    if (r?.status === 'pending') return h('td', { class: cls, ...tag }, h('span', { class: 'pending', role: 'img', onmousemove: tip, onmouseleave: hideTip,
      'aria-label': `${modelLabel(m)} on ${repo.name}: not finished yet, not counted in the score` }, '…'));
    if (!r || r.status !== 'passed') return h('td', { class: cls, ...tag }, h('button', { type: 'button', class: 'fail', onclick: go, onmousemove: tip, onmouseleave: hideTip,
      'aria-label': `View outputs for ${modelLabel(m)} on ${repo.name}: failed${attempted != null ? `, attempted ${pct(attempted)}` : ''}` },
    attempted != null ? [h('span', { class: 'tried' }, figure(Math.max(0, attempted))), h('i', { 'aria-hidden': 'true' }, '×')] : '×'));
    const bg = ramp(Math.max(0, r.score) / 100);
    return h('td', { class: cls, ...tag }, h('button', { type: 'button', style: { background: bg, color: inkOn(bg) }, onclick: go, onmousemove: tip, onmouseleave: hideTip,
      'aria-label': `View outputs for ${modelLabel(m)} on ${repo.name}: ${pct(r.score)}` }, figure(r.score)));
  };

  const draw = () => {
    const rows = [...models].sort((a, b) => dir * (getters[sortKey](a) - getters[sortKey](b)));
    tbody.replaceChildren(...rows.map((m) => h('tr', { style: { '--model': modelColor(m) } },
      h('td', { class: 'fx fx-rank' }, m.rank),
      h('td', { class: 'fx fx-model' }, h('a', { class: 'model-cell model-link', href: `#/model/${m.short}`, title: `${modelLabel(m)}: harness, settings, tools and prompts` },
        logo(m, 'xs'), h('div', { class: 'model-name' }, h('b', {}, m.label),
          m.effort ? h('span', { class: 'reasoning-badge', title: `Reasoning effort: ${m.effort}`, 'aria-label': `Reasoning effort: ${m.effort}` }, m.effort) : null))),
      h('td', { class: 'fx fx-score' }, h('div', { class: 'scorebar' },
        h('div', { class: 'track' }, h('div', { class: 'fill', style: { width: `${Math.max(0, Math.min(100, m.score))}%` } })), h('b', {}, pct(m.score)))),
      h('td', { class: 'fx fx-cost num' }, usd(m.cost)),
      h('td', { class: 'fx fx-time num' }, hours(m.hours)),
      ...repos.map((repo) => cell(m, repo)))));
  };
  const sortable = (key, label) => h('button', { type: 'button', onclick: () => {
    if (sortKey === key) dir = -dir; else { sortKey = key; dir = key === 'score' ? -1 : 1; }
    draw();
  } }, label);

  const bandRow = h('tr', { class: 'bands' },
    h('th', { class: 'fx fx-rank' }), h('th', { class: 'fx fx-model' }), h('th', { class: 'fx fx-score' }), h('th', { class: 'fx fx-cost' }), h('th', { class: 'fx fx-time' }),
    ...bands.map((b) => h('th', { colspan: repos.filter((r) => r.band === b.band).length, class: 'band-head' }, h('span', {}, b.scale), h('i', {}, `${b.band} tokens`))));
  const headRow = h('tr', { class: 'cols' },
    h('th', { class: 'fx fx-rank' }, '#'), h('th', { class: 'fx fx-model l' }, 'Model'),
    h('th', { class: 'fx fx-score' }, sortable('score', 'Score')), h('th', { class: 'fx fx-cost' }, sortable('cost', 'Cost')), h('th', { class: 'fx fx-time' }, sortable('hours', 'Time')),
    ...repos.map((repo, i) => h('th', { class: firsts.has(repo.id) ? 'rh first' : 'rh', 'data-repo': repo.id }, h('a', { href: `viewer.html#/graph/${repo.id}`, 'aria-label': repo.name,
      onmousemove: (e) => showTip(e, `<b>${i + 1} · ${esc(repo.name)}</b><div>${tokens(repo.tokens)} tokens</div>`), onmouseleave: hideTip }, i + 1))));
  draw();
  const scroller = h('div', { class: 'lb-scroll', tabindex: 0, 'aria-label': 'Leaderboard, scrolls sideways through all 64 repositories' },
    h('table', { class: 'lb' }, h('thead', {}, bandRow, headRow), tbody));
  // On a phone the table shows the summary columns until the Results tabs switch
  // it to the per-repository scores (.per-repo); from 760px it always shows both (see site.css).
  const wrap = h('div', { class: 'lb-wrap' });
  const hint = h('div', { class: 'lb-hint' },
    h('span', { class: 'lb-swipe' }, h('strong', {}, 'Click a score to explore its outputs.'), ' ', `Scroll for all ${repos.length} repositories →`),
    h('span', { class: 'heat-legend' }, '0%', h('span', { class: 'ramp' }, [0, 0.25, 0.5, 0.75, 1].map((t) => h('i', { style: { background: ramp(t) } }))), '100% removed'));
  scroller.addEventListener('scroll', () => scroller.classList.toggle('scrolled', scroller.scrollLeft > 4), { passive: true });
  wrap.append(hint, scroller);
  wrap.setModels = (next) => { models = next; draw(); };
  wrap.getState = () => ({ sortKey, dir, scrollLeft: scroller.scrollLeft });
  wrap.setState = (state) => {
    if (!state) return;
    if (Object.hasOwn(getters, state.sortKey)) sortKey = state.sortKey;
    if (state.dir === 1 || state.dir === -1) dir = state.dir;
    draw();
    scroller.scrollLeft = Number.isFinite(state.scrollLeft) ? state.scrollLeft : 0;
  };
  return wrap;
}

// Cost against score, one logo per agent on a log cost axis. Solid lines
// connect each model family in reasoning-effort order.
function frontierChart(models) {
  const host = h('div', { class: 'chart frontier' });
  const pts = models.filter((m) => m.cost > 0);
  const groups = new Map();
  for (const model of pts) {
    const key = `${model.provider}/${model.label}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(model);
  }
  const families = [...groups.values()].filter((family) => family.length > 1)
    .map((family) => family.sort((a, b) => EFFORTS.indexOf(a.effort) - EFFORTS.indexOf(b.effort)));
  const measure = document.createElement('canvas').getContext('2d');
  const tip = (m) => (e) => showTip(e, `<span>${esc(modelLabel(m))}</span>`
    + `<div class="row"><span>Score</span><span>${pct(m.score)}</span></div>`
    + `<div class="row"><span>Cost per repository</span><span>${usd(m.cost)}</span></div>`
    + `<div class="row"><span>Time per repository</span><span>${hours(m.hours)}</span></div>`
    + `<div class="row"><span>Verified</span><span>${m.passed}/${m.expected}</span></div>`
    + (m.scored < m.expected ? `<div class="note">Scored on ${m.scored} of ${m.expected} repositories; the rest are not finished yet.</div>` : ''));
  let drawn = 0;
  const draw = () => {
    // Hidden (the other tab on a phone) or unchanged: nothing to do.
    if (!host.clientWidth || host.clientWidth === drawn) return;
    drawn = host.clientWidth;
    // The link-preview card (pipeline/build_og.py) sets data-zoom, data-height and
    // data-mark: it draws at 1/zoom of its width, at a set height and with smaller
    // logos, and the SVG scales up.
    const zoom = +host.dataset.zoom || 1;
    const W = Math.round(drawn / zoom);
    const narrow = W < 560;
    // A wide chart (a desktop) sets its labels a size up; see .chart.frontier .large in site.css.
    const large = W >= 760, fs = large ? 16 : 12;
    const H = +host.dataset.height || (narrow ? (pts.length > 10 ? 620 : 300) : W < 900 ? 360 : 400);
    const L = narrow ? 36 : 44, R = W - (narrow ? 10 : 18), T = 30, B = H - 42;
    const S = +host.dataset.mark || (narrow ? 16 : 18), icon = Math.round(S * 0.58);
    const costs = pts.map((m) => m.cost);
    const lo = Math.log(Math.min(...costs) / 1.7), hi = Math.log(Math.max(...costs) * 1.7);
    const x = (c) => L + ((Math.log(c) - lo) / (hi - lo)) * (R - L);
    // The full 0–100% scale, with a little room below 0% so a logo near zero
    // clears the cost labels.
    const top = 100, floor = -3;
    const y = (v) => B - ((v - floor) / (top - floor)) * (B - T);

    const svg = s('svg', { class: large ? 'large' : null, viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img',
      'aria-label': `Score against cost per repository. ${pts.map((m) => `${modelLabel(m)}: ${pct(m.score)} at ${usd(m.cost)}`).join('; ')}.` });
    const grid = s('g', { class: 'grid' });
    for (let t = 0; t <= top; t += 10) {
      grid.append(s('line', { x1: L, x2: R, y1: y(t), y2: y(t) }));
      grid.append(s('text', { x: L - 8, y: y(t) + 4, 'text-anchor': 'end' }, `${t}%`));
    }
    // 1-2-5 steps per decade; a phone drops the 2s when the axis gets crowded.
    const ticks = [];
    for (let d = Math.floor(lo / Math.LN10); d <= Math.ceil(hi / Math.LN10); d++) {
      for (const k of [1, 2, 5]) { const v = k * 10 ** d; if (Math.log(v) >= lo && Math.log(v) <= hi) ticks.push(v); }
    }
    const shown = narrow && ticks.length > 7 ? ticks.filter((v) => !String(v).startsWith('2')) : ticks;
    for (const v of shown) {
      grid.append(s('line', { x1: x(v), x2: x(v), y1: T, y2: B }));
      grid.append(s('text', { x: x(v), y: B + 18, 'text-anchor': 'middle' }, `$${v}`));
    }
    svg.append(grid,
      s('text', { class: 'axis-title', x: L - (narrow ? 30 : 38), y: T - 14 }, 'Score (%)'),
      s('text', { class: 'axis-title cost-axis-title', x: R, y: H - 6, 'text-anchor': 'end' }, 'Cost per repository (USD)'),
      ...families.map((family) => s('polyline', { class: 'effort-line', stroke: modelColor(family[0]),
        'aria-label': `${family[0].label}: ${family.map((m) => m.effort).join(' → ')}`,
        points: family.map((m) => `${x(m.cost)},${y(m.score)}`).join(' ') })));

    // Direct labels: beside each logo where there is room (right, left, below,
    // above, then below or above flush with either edge), placed from the top
    // score down so leaders get first pick, backtracking when a later label
    // would find no free spot.
    const badgeFs = large ? 11 : 9, badgeHeight = badgeFs + 6, badgeGap = 5;
    const labelSizes = new Map(pts.map((m) => {
      measure.font = `400 ${fs}px ${cssVar('--font-ui')}`;
      const nameWidth = measure.measureText(m.label).width;
      measure.font = `400 ${badgeFs}px ${cssVar('--font-ui')}`;
      const badgeWidth = m.effort ? measure.measureText(m.effort).width + 12 : 0;
      return [m, { nameWidth, badgeWidth, width: nameWidth + (badgeWidth ? badgeGap + badgeWidth : 0) }];
    }));
    const boxes = pts.map((m) => ({ x: x(m.cost) - S / 2 - 2, y: y(m.score) - S / 2 - 2, w: S + 4, h: S + 4 }));
    const hits = (a, b) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
    // Sample the effort lines so labels keep clear of them too.
    const line = families.flatMap((family) => family.slice(1).flatMap((m, i) => {
      const [x0, y0, x1, y1] = [x(family[i].cost), y(family[i].score), x(m.cost), y(m.score)];
      const n = Math.ceil(Math.hypot(x1 - x0, y1 - y0) / 4);
      return Array.from({ length: n + 1 }, (_, k) => ({ x: x0 + ((x1 - x0) * k) / n - 3, y: y0 + ((y1 - y0) * k) / n - 3, w: 6, h: 6 }));
    }));
    // The page padding can hold the last few pixels of a mobile label. Keeping
    // it close to its marker is clearer than moving it several score bands away.
    const labelRight = narrow ? Math.min(W + 12, window.innerWidth - host.getBoundingClientRect().left - 4) : W - 2;
    const inside = (a) => a.x >= (narrow ? 4 : L - 4) && a.x + a.w <= labelRight && a.y >= T - 6 && a.y + a.h <= B + 4;
    // Reserve the crowded cluster before the other mobile labels, so the
    // highest-scoring point cannot push its neighbours across the chart.
    const mobilePriority = ['gpt-6.1-sol-medium', 'gpt-6.1-sol-max', 'opus-5.5-medium', 'gemini-4-argon'];
    const priority = (m) => narrow && mobilePriority.includes(m.key) ? mobilePriority.indexOf(m.key) : mobilePriority.length;
    const order = [...pts].sort((a, b) => priority(a) - priority(b) || b.score - a.score);
    const options = order.map((m) => {
      const cx = x(m.cost), cy = y(m.score), r = S / 2 + 6, e = S / 2;
      const w = labelSizes.get(m).width + (narrow ? 2 : 4), lh = fs * 1.5, mid = cy + fs / 3;
      const above = { ty: cy - r - 3 }, below = { ty: cy + r + fs + 1 };
      // Prefer consistent placements for the reasoning variants, with alternate
      // positions available when another model crowds that part of the chart.
      const solAbove = ['gpt-6.1-sol-medium', 'gpt-6.1-sol-high', 'gpt-6.1-sol', 'gpt-6.1-sol-max'].includes(m.key);
      const aligned = narrow
        ? { x: cx - w / 2, anchor: 'middle', tx: cx }
        : { x: cx - e, anchor: 'start', tx: cx - e };
      const preferred = m.key === 'gemini-4-argon'
        ? narrow
          ? { x: cx - e + 4, anchor: 'start', tx: cx - e + 4, ty: below.ty }
          : { ...aligned, ty: below.ty }
        : m.key === 'opus-5.5-medium'
        ? { ...aligned, ty: above.ty - lh * (narrow ? 2 : 1), leader: true }
        : solAbove
        ? { x: cx + e - w, anchor: 'end', tx: cx + e, ty: above.ty }
        : !narrow && m.key === 'opus-5.5-low' ? { ...aligned, ty: below.ty } : null;
      return [
        ...(preferred ? [preferred] : []),
        { x: cx + r, y: cy - lh / 2, anchor: 'start', tx: cx + r, ty: mid },
        { x: cx - r - w, y: cy - lh / 2, anchor: 'end', tx: cx - r, ty: mid },
        { ...below, x: cx - w / 2, anchor: 'middle', tx: cx },
        { ...above, x: cx - w / 2, anchor: 'middle', tx: cx },
        ...[below, above].flatMap((v) => [
          { ...v, x: cx - e, anchor: 'start', tx: cx - e },
          { ...v, x: cx + e - w, anchor: 'end', tx: cx + e },
        ]),
        // A crowded frontier can move a label farther from its mark; a leader
        // keeps the association clear on a phone.
        ...[1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8].flatMap((k) => [
          { x: cx + r, anchor: 'start', tx: cx + r, ty: mid + k * lh, leader: true },
          { x: cx - r - w, anchor: 'end', tx: cx - r, ty: mid + k * lh, leader: true },
          { x: cx - w / 2, anchor: 'middle', tx: cx, ty: mid + k * lh, leader: true },
        ]),
      ].map((o) => ({ ...o, y: o.ty - fs * 1.1, w, h: lh, model: m.key }))
        .filter((o) => m.key === 'gemini-4-argon' ? o.y >= cy + S / 2 + 2
          : (!narrow && solAbove) || ['gpt-6.1-sol-medium', 'gpt-6.1-sol-max', 'opus-5.5-medium'].includes(m.key)
            ? o.y + o.h <= cy - S / 2 - 2 : true)
        .filter((o) => !narrow || m.key !== 'gpt-6.1-sol-medium' || !o.leader);
    });
    const free = (o, placed) => {
      if (narrow && o.model === 'opus-5.5-medium') {
        const sol = placed.find((b) => b.model === 'gpt-6.1-sol-max');
        if (sol && o.y + o.h + 6 > sol.y) return false;
      }
      return inside(o) && ![...boxes, ...placed, ...line].some((b) => hits(o, b));
    };
    // A step budget keeps a crowded chart from searching every combination.
    let budget = 20000;
    const place = (i, placed) => {
      if (i === order.length) return placed;
      for (const o of options[i]) {
        if (--budget < 0) return null;
        if (free(o, placed)) { const done = place(i + 1, [...placed, o]); if (done) return done; }
      }
      return null;
    };
    // No arrangement keeps every label clear: each takes its first free spot, if any.
    const picks = place(0, []) || options.reduce((placed, opts) =>
      [...placed, opts.find((o) => free(o, placed)) || opts.find(inside) || opts[0]], []);
    order.forEach((m, i) => {
      const pick = picks[i];
      if (pick.leader) svg.append(s('line', { class: 'label-leader', 'data-model': m.key, x1: x(m.cost), y1: y(m.score),
        x2: Math.max(pick.x, Math.min(x(m.cost), pick.x + pick.w)),
        y2: Math.max(pick.y, Math.min(y(m.score), pick.y + pick.h)) }));
      const { nameWidth, badgeWidth, width } = labelSizes.get(m);
      const left = pick.tx - (pick.anchor === 'end' ? width : pick.anchor === 'middle' ? width / 2 : 0);
      const label = s('g', { class: 'frontier-model-label', 'data-model': m.key, transform: `translate(${left},${pick.ty})`,
        onmousemove: tip(m), onmouseleave: hideTip }, s('text', { class: 'label' }, m.label));
      if (m.effort) {
        const badgeX = nameWidth + badgeGap, centerY = -fs / 3;
        label.append(s('g', { class: 'frontier-reasoning-badge' },
          s('title', {}, `Reasoning effort: ${m.effort}`),
          s('rect', { x: badgeX, y: centerY - badgeHeight / 2, width: badgeWidth, height: badgeHeight, rx: badgeHeight / 2 }),
          s('text', { x: badgeX + badgeWidth / 2, y: centerY + badgeFs * 0.35, 'text-anchor': 'middle' }, m.effort)));
      }
      svg.append(label);
    });

    for (const m of pts) {
      const g = s('g', { class: 'pt', transform: `translate(${x(m.cost) - S / 2},${y(m.score) - S / 2})`, onmousemove: tip(m), onmouseleave: hideTip });
      g.append(s('rect', { x: 0.75, y: 0.75, width: S - 1.5, height: S - 1.5, rx: S * 0.28, style: `stroke:${modelColor(m)}` }));
      const mark = s('g');
      mark.innerHTML = logoMarkup(m);
      const inner = mark.firstElementChild;
      if (inner) {
        for (const [k, v] of Object.entries({ x: (S - icon) / 2, y: (S - icon) / 2, width: icon, height: icon })) inner.setAttribute(k, v);
        g.append(inner);
      } else g.append(s('text', { class: 'label', x: S / 2, y: S / 2 + 4, 'text-anchor': 'middle' }, m.label[0]));
      svg.append(g);
    }
    host.replaceChildren(svg);
  };
  new ResizeObserver(draw).observe(host);
  // Labels are measured in the web font; redraw once it has loaded.
  document.fonts?.ready.then(() => { drawn = 0; draw(); });
  return host;
}

// Preprocessing, evaluation and postprocessing (paper §3 and §4.2), each under
// its panel from the paper's overview figure (built by pipeline/build_overview.sh).
function methodSection(bench) {
  // One image per section of the panel; the card and its rules are drawn here.
  const step = (title, panel, sections, ...body) => h('article', { class: 'step' },
    h('div', { class: 'step-panel' }, sections.map((alt, i) => h('div', {}, h('img', { src: `assets/overview-${panel}-${i + 1}.svg`, alt })))),
    h('h3', {}, title), ...body);
  return h('section', { class: 'section' },
    // A short summary of the paper's abstract.
    sectionHead(null, 'Method', 'Can coding agents make a verified Lean library smaller? '
      + `LeanLean gives an agent 12 hours on each of ${bench.repositories.length} real-world Lean repositories to compress the Lean codebase as much as it can while preserving the protected theorems.`),
    h('div', { class: 'steps' },
      step('Preprocessing', 'preprocessing', ['A dependency graph: protected theorems and required roots at the top, the lemmas and definitions they use below, and unused declarations faded out.'],
        h('p', {}, 'We take ', `${bench.repositories.length}`, ' repositories from the ',
          h('a', { href: 'https://palomar-registry.org/', target: '_blank', rel: 'noopener' }, 'Palomar Registry'), ', chosen to cover diverse subjects and sizes. ',
          'Each states its main theorems in ', h('code', {}, 'Challenge.lean'), ' and proves them in ', h('code', {}, 'Solution.lean'), '.'),
        h('p', {}, 'We build a declaration-level dependency graph from the ', h('code', {}, '.olean'), ' and ', h('code', {}, '.ilean'), ' artifacts plus a text search, and make ', h('code', {}, 'grind'), '’s implicit lemma use explicit.'),
        h('p', {}, 'Everything outside the dependency closure of the protected theorems and other required roots (instances, notation, ', h('code', {}, '@[simp]'), ' lemmas) is stripped: a median 10.7% of tokens. Deleting dead code alone therefore scores little.')),
      step('Evaluation', 'task', ['A coding agent with a 12-hour budget.', 'Its tools: lake build, lean_verify and proof_length.py.', '8 CPUs, 64 GiB RAM, offline Docker.'],
        h('p', {}, 'Each agent runs in its own harness (Claude Code for Opus 5, Codex for Sol 5.6, …) with skills, MCP and subagents disabled. ',
          'Its goal is to minimize ', h('i', {}, 'T'), ', the number of Lean tokens in every ', h('code', {}, '.lean'), ' file except ', h('code', {}, 'Challenge.lean'), '.'),
        h('p', {}, 'Lean tokens are the size metric of ',
          h('a', { href: 'https://arxiv.org/abs/2605.20244', target: '_blank', rel: 'noopener' }, 'Lean Refactor'), ' (Lu et al., 2026). ',
          'A token is a whole name or number, such as ', h('code', {}, 'Nat.succ_le_of_lt'), ', or a single symbol; a few operators such as ', h('code', {}, ':='), ' and ', h('code', {}, '<;>'), ' count as one. ',
          'Comments, imports and whitespace are not counted.'),),
      step('Postprocessing', 'evaluation', ['The submitted edits, as lines added and removed per file.', 'Lean Comparator checks that the dependency graphs before and after prove the same results.', 'The token change, here −32%.'],
        h('p', {}, 'We restore ', h('code', {}, 'Challenge.lean'), ', rebuild from scratch and run Lean Comparator: the protected results must be proven exactly, using only the permitted axioms.'),
        h('p', { class: 'formula' }, h('i', {}, 'S'), ' = 1 − ', h('i', {}, 'T'), h('sub', {}, 'after'), ' / ', h('i', {}, 'T'), h('sub', {}, 'before')),
        h('p', {}, 'A failed check scores 0; the score is the mean over all ', `${bench.repositories.length}`, ' repositories.'),
        h('p', {}, 'We classify compression into 6 categories:'),
        h('ul', { class: 'step-keys' }, ORIGIN.map((o) => h('li', {}, h('i', { style: { background: `var(${o.color})` } }), o.label))),
        h('p', {}, 'Deleted declarations dominate: agents restructure proofs until existing lemmas are no longer needed. Click a leaderboard cell to see the breakdown for one run.'))));
}

// A hero link: the brand's own coloured logo and a label, inline in the text.
const heroLink = (href, icon, label) => h('a', { class: 'hero-link', href, target: '_blank', rel: 'noopener' },
  h('img', { src: `assets/logos/${icon}.svg`, alt: '', width: 16, height: 16 }), label);

export function renderLeaderboard(view, bench) {
  const models = bench.models;

  const hero = h('section', { class: 'hero' },
    h('h1', {}, PAPER_TITLE),
    h('p', { class: 'authors' }, AUTHORS.map((author, i) =>
      [i ? ', ' : '', h('a', { href: author.url }, author.name)])),
    h('div', { class: 'logos' },
      h('a', { href: 'https://www.sri.inf.ethz.ch', target: '_blank', rel: 'noopener' }, h('img', { src: 'assets/sri-logo.svg', alt: 'SRI Lab', width: 136, height: 24 })),
      h('a', { href: 'https://ethz.ch', target: '_blank', rel: 'noopener' }, h('img', { src: 'assets/eth-logo.svg', alt: 'ETH Zurich', width: 144, height: 24 }))),
    h('div', { class: 'hero-links' },
      heroLink(PAPER_URL, 'paper', 'Paper'),
      heroLink(CODE_URL, 'github', 'Code'),
      heroLink(DATASET_URL, 'huggingface', 'Benchmark')),
    h('p', { class: 'hero-intro' },
      'As AI generates ever larger formal proofs, the next challenge is to make them concise and reusable. ',
      'In LeanLean, models are evaluated on shortening large proofs.'));

  // From 760px both panels show, the frontier first; on a phone they are tabs,
  // opening on the frontier, with the leaderboard's summary and per-repository
  // columns as separate tabs (see site.css).
  const board = boardTable(bench);
  const chart = frontierChart(models);
  // Explain the score below the chart, and below the table on phones.
  const scoreNote = 'Score is mean Lean token compression across repositories. A failed build or Comparator check scores zero.';
  const boardHeader = h('div', { class: 'board-header' }, h('h2', {}, 'Leaderboard'));
  const panels = {
    board: h('div', { class: 'result-panel board-panel', id: 'results-board', role: 'tabpanel' },
      boardHeader, board, h('p', { class: 'tab-note' }, scoreNote)),
    frontier: h('div', { class: 'result-panel frontier-panel', id: 'results-frontier', role: 'tabpanel' },
      h('figure', { class: 'frontier-figure' }, chart,
        h('figcaption', {}, h('span', {}, scoreNote), h('span', { class: 'frontier-cost-label' }, 'Cost per repository (USD)')))),
  };
  const body = h('div', { class: 'results-body', 'data-tab': 'frontier' }, panels.frontier, panels.board);
  const tabs = [['frontier', 'Frontier', panels.frontier], ['summary', 'Leaderboard', panels.board], ['repos', 'Per repository', panels.board]]
    .map(([key, label, panel]) => h('button', {
      type: 'button', role: 'tab', 'data-tab': key, 'aria-controls': panel.id, 'aria-selected': String(key === 'frontier'), onclick: (e) => {
        body.dataset.tab = key;
        board.classList.toggle('per-repo', key === 'repos');
        for (const t of tabs) t.setAttribute('aria-selected', String(t === e.currentTarget));
      } }, label));
  const updateFilter = () => {
    const visible = highestOnly.checked ? highestReasoningModels(models) : models;
    const ranks = new Map([...visible].sort((a, b) => b.score - a.score).map((m, i) => [m.key, i + 1]));
    const ranked = visible.map((m) => ({ ...m, rank: ranks.get(m.key) }));
    board.setModels(ranked);
  };
  const highestOnly = h('input', { type: 'checkbox', role: 'switch', checked: true, onchange: updateFilter });
  updateFilter();
  const filter = h('label', { class: 'results-filter' }, highestOnly, h('span', {}, 'Only highest reasoning'));
  const results = h('section', { class: 'section results', 'aria-label': 'Results' },
    h('div', { class: 'results-tabs', role: 'tablist', 'aria-label': 'Results' }, tabs),
    body);
  boardHeader.append(filter);

  view.replaceChildren(h('div', { class: 'page' }, hero, results));
}

// Method and citation follow the result previews on the landing page.
export function renderMethod(bench) {
  return h('div', { class: 'page method-page' }, methodSection(bench));
}

// The landing page's closing section: citation and contact.
export function renderCitation() {
  return h('div', { class: 'page cite-page' }, h('section', { class: 'section cite' },
    sectionHead(null, CITATION ? 'Citation' : 'Contact', null), CITATION ? citationBlock() : null,
    h('p', { class: 'contact' }, 'Questions? Email ', h('a', { href: `mailto:${CONTACT_EMAIL}` }, CONTACT_EMAIL), '.')));
}

// CITATION is defined with the paper metadata in util.js.
function citationBlock() {
  const pre = h('pre', { class: 'bibtex' }, h('code', {}, CITATION));
  const status = h('span', { class: 'muted', 'aria-live': 'polite' });
  const copy = h('button', { type: 'button', class: 'btn', onclick: async () => {
    try { await navigator.clipboard.writeText(CITATION); status.textContent = 'Copied'; }
    catch {
      const range = document.createRange(); range.selectNodeContents(pre);
      const selection = getSelection(); selection.removeAllRanges(); selection.addRange(range);
      status.textContent = 'Selected; press Ctrl+C to copy';
    }
    setTimeout(() => { status.textContent = ''; }, 2500);
  } }, 'Copy BibTeX');
  return h('div', { class: 'bibtex-wrap' }, pre, h('div', { class: 'bibtex-actions' }, copy, status));
}
