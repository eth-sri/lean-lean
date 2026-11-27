// Trace viewer: one agent run as a list of steps under a Lean-token timeline.
// Every harness's native trace is normalised offline (pipeline/export_traces.py)
// into the same steps: say, think, user, compact and tool. Tool calls keep their
// native names; their category picks how the input is drawn, and the paper's action
// class (Build, Verify, Measure, Read, Search, Edit, Git, Lake, Sleep, Other) says what they did.
import { h, s, load, loadChunk, loadPack, int, tokens, usd, signedInt, esc, showTip, hideTip } from './util.js?v=e38b2b7f7d';
import { codeBlock, patchBlock } from './code.js?v=e38b2b7f7d';

// The paper's action classes (scripts/analysis/plot_action_classes.py), as in its action figure.
const ACTIONS = ['Build', 'Verify', 'Measure', 'Read', 'Search', 'Edit', 'Git', 'Lake', 'Sleep', 'Other'];
const ACTION_TIP = {
  Build: 'lake build', Verify: 'lean_verify', Measure: 'proof_length.py', Read: 'Reading files',
  Search: 'Searching files', Edit: 'Changing a file of the repository; scratch files, logs and /tmp do not count', Git: 'Git commands',
  Lake: 'Lake commands other than a build', Sleep: 'Waiting with sleep',
  Other: 'Every other command, and the tools the paper does not count: polling a running command, todo lists, asking the user',
};
// A tool call's actions: one, or for a code-mode script one per command or patch it ran, as the paper counts them.
const actionsOf = (st) => Array.isArray(st.ac) ? st.ac : [st.ac || 'Other'];
const KIND = { say: 'Agent', think: 'Thinking', user: 'Prompt', compact: 'Context compacted' };
const FILTERS = [['all', 'All'], ['talk', 'Messages']];
// A run of this many polls in a row folds into one row.
const FOLD_WAITS = 3;

export function clock(seconds, approx = false) {
  const t = Math.max(0, Math.round(seconds || 0));
  const hh = Math.floor(t / 3600), mm = Math.floor((t % 3600) / 60), ss = t % 60;
  const text = hh ? `${hh}:${String(mm).padStart(2, '0')}:${String(ss).padStart(2, '0')}` : `${mm}:${String(ss).padStart(2, '0')}`;
  return approx ? `≈${text}` : text;
}

export class TraceView {
  constructor(host) {
    this.host = host;
    this.filter = 'all';
    this.action = null;
    this.query = '';
    this.summary = h('div', { class: 'trace-summary' });
    this.chart = h('div', { class: 'trace-chart' });
    this.searchInput = h('input', { class: 'search', type: 'search', placeholder: 'Search the trace', 'aria-label': 'Search the trace', autocomplete: 'off' });
    this.count = h('span', { class: 'trace-count muted num' });
    this.filterSeg = h('div', { class: 'seg', role: 'group', 'aria-label': 'Steps shown' }, FILTERS.map(([key, label]) =>
      h('button', { type: 'button', 'data-k': key, 'aria-pressed': String(key === this.filter), onclick: () => this.setFilter(key) }, label)));
    this.expandAll = h('button', { type: 'button', class: 'btn icon-btn', onclick: () => this.toggleAll() }, 'Expand all');
    this.toolbar = h('div', { class: 'trace-toolbar' }, this.filterSeg, this.searchInput, this.count, this.expandAll);
    this.list = h('div', { class: 'trace-list', role: 'list' });
    this.actionBar = h('div', { class: 'trace-actions', role: 'group', 'aria-label': 'Tool calls by action' });
    this.host.append(this.summary, this.chart, this.toolbar, this.actionBar, this.list);
    let timer = 0;
    this.searchInput.addEventListener('input', () => {
      clearTimeout(timer);
      timer = setTimeout(() => this.search(this.searchInput.value.trim().toLowerCase()), 160);
    });
    this.list.addEventListener('scroll', () => this.onScroll(), { passive: true });
    this.resize = new ResizeObserver(() => { if (this.data) this.renderChart(); });
    this.resize.observe(this.chart);
  }

  destroy() { this.resize.disconnect(); }

  // context: { repo, model, result, label } from the repository page.
  async show(id, short, context) {
    const key = `${id}/${short}`;
    if (this.key === key) return;
    this.key = key;
    this.summary.replaceChildren(h('span', { class: 'muted' }, 'Loading the trace…'));
    this.list.replaceChildren();
    this.chart.replaceChildren();
    let data;
    try {
      data = await load(`data/traces/${id}/${short}.json.gz`);
    } catch (error) {
      if (this.key !== key) return;
      this.data = null;
      this.summary.replaceChildren(h('div', { class: 'note' }, `No trace for this run (${error.message}).`));
      return;
    }
    if (this.key !== key) return;
    this.data = data;
    this.context = context;
    this.packPath = `data/traces/${id}/${short}.bin`;
    this.chunks = new Map();
    this.allLoaded = null;
    this.bodiesSearched = false;
    this.query = this.searchInput.value.trim().toLowerCase();
    this.expanded = new Set();
    this.allOpen = false;
    this.action = null;
    this.expandAll.textContent = 'Expand all';
    this.prepare();
    this.renderSummary();
    this.renderChart();
    this.renderList();
    this.list.scrollTop = 0;
  }

  // Token level after each step, and the text a search matches.
  prepare() {
    const d = this.data;
    let level = d.baseline ?? this.context.repo.tokens;
    for (const step of d.steps) {
      if (step.d?.tok != null) {
        step.delta = step.d.tok - level;
        level = step.d.tok;
      }
    }
    this.final = d.final ?? this.context.result?.post ?? level;
  }

  // Share of the baseline's Lean tokens removed at a token count, as on the timeline.
  compression(count) {
    const base = this.data.baseline ?? this.context.repo.tokens;
    return `${((1 - count / base) * 100).toFixed(1)}%`;
  }

  // ── bodies: inputs, outputs and diffs live in the run's pack, one chunk per Range request ──
  body(p) {
    if (p == null) return Promise.resolve();
    if (!this.chunks.has(p)) {
      const steps = this.data.steps;
      const loading = loadChunk(this.packPath, this.data.pack[p]).then((chunk) => {
        for (const [i, b] of Object.entries(chunk)) {
          const st = steps[i];
          if (b.i != null) st.i = b.i;
          if (b.o != null) st.o = b.o;
          if (b.diff != null && st.d) st.d.diff = b.diff;
          if (b.x != null) st.x = b.x;
          st.hay = undefined;
        }
        loading.done = true;
      }).catch((error) => { this.chunks.delete(p); throw error; });
      this.chunks.set(p, loading);
    }
    return this.chunks.get(p);
  }

  // Every body at once (one request for the whole pack), for search and Expand all.
  loadAll() {
    if (!this.allLoaded) {
      const data = this.data;
      this.allLoaded = loadPack(this.packPath).then(() => Promise.all(data.pack.map((_, p) => this.body(p))));
    }
    return this.allLoaded;
  }

  // Titles and messages are searched at once; inputs and outputs once the pack is in.
  search(query) {
    this.query = query;
    this.renderList();
    if (!query || this.bodiesSearched) return;
    const key = this.key;
    this.count.textContent += ' · searching inputs and outputs…';
    this.loadAll().then(() => {
      if (this.key !== key) return;
      this.bodiesSearched = true;
      if (this.query) this.renderList();
    }).catch(() => { this.count.textContent = this.count.textContent.replace(' · searching inputs and outputs…', ''); });
  }

  haystack(step) {
    if (step.hay === undefined) {
      step.hay = [step.n, step.ac, step.ti, step.x, step.i, step.o, step.d?.f?.join(' '), step.a].filter(Boolean).join('\n').toLowerCase();
    }
    return step.hay;
  }

  // Show only one action's calls, or all steps (null).
  // Messages have no action, so picking one leaves the Messages filter, and back.
  setAction(action) {
    this.action = action;
    if (this.action && this.filter === 'talk') this.filter = 'all';
    this.renderList();
  }

  setFilter(key) {
    this.filter = key;
    if (key === 'talk') this.action = null;
    this.renderList();
  }

  pressed() {
    for (const b of this.filterSeg.children) b.setAttribute('aria-pressed', String(b.dataset.k === this.filter));
    for (const b of this.actionBar.children) b.setAttribute('aria-pressed', String((b.dataset.a || null) === this.action));
  }

  async toggleAll() {
    if (!this.allOpen) {
      const key = this.key;
      this.expandAll.textContent = 'Loading…';
      try { await this.loadAll(); } catch { this.expandAll.textContent = 'Expand all'; return; }
      if (this.key !== key) return;
    }
    this.allOpen = !this.allOpen;
    this.expandAll.textContent = this.allOpen ? 'Collapse all' : 'Expand all';
    this.expanded = new Set();
    this.renderList();
  }

  // ── summary ────────────────────────────────────────────────────────────
  renderSummary() {
    const d = this.data;
    const tools = d.steps.filter((st) => st.k === 'tool');
    const byAction = new Map();
    for (const st of tools) for (const a of actionsOf(st)) byAction.set(a, (byAction.get(a) || 0) + 1);
    // All steps, then the actions, most used first and Other, the rest, last.
    const top = ACTIONS.filter((a) => byAction.has(a)).map((a) => [a, byAction.get(a)])
      .sort((a, b) => (a[0] === 'Other') - (b[0] === 'Other') || b[1] - a[1]);
    const chip = (action, label, n, title) => h('button', { type: 'button', class: 'chip act', 'data-a': action || '',
      'aria-pressed': String((action || null) === this.action), title, onclick: () => this.setAction(action) }, label, h('b', { class: 'num' }, int(n)));
    this.actionBar.replaceChildren(chip(null, 'All', d.steps.length, 'Every step'),
      ...top.map(([action, n]) => chip(action, action, n, `${ACTION_TIP[action]}: ${int(n)} action${n === 1 ? '' : 's'}`)));
    const errors = tools.filter((st) => st.e).length;
    const changes = tools.filter((st) => st.d).length;
    const thoughts = d.steps.filter((st) => st.k === 'think').length;
    const fact = (label, value) => h('div', { class: 'stat' }, h('span', {}, label), h('b', {}, value));
    this.summary.replaceChildren(
      h('div', { class: 'stats' },
        fact('Harness', d.harness),
        fact('Steps', int(d.steps.length)),
        fact('Tool calls', int(tools.length)),
        fact('Source changes', int(changes)),
        thoughts ? fact(d.steps.some((st) => st.k === 'think' && st.s) ? 'Reasoning summaries' : 'Reasoning blocks', int(thoughts)) : null,
        fact('Failed calls', int(errors)),
        fact('Wall time', clock(d.duration))),
      d.note || d.estimated ? h('p', { class: 'trace-note muted' }, d.note) : null);
  }

  // ── timeline ───────────────────────────────────────────────────────────
  // Compression over wall time, on the same scale as the score: the share of the
  // baseline's Lean tokens removed, 0 to 100%. A tick under it for every step that
  // changed the sources, and a cursor at the step at the top of the list. Click to jump.
  renderChart() {
    const d = this.data;
    const width = Math.max(240, this.chart.clientWidth - 16 || 600);
    const narrow = width < 560;
    const height = narrow ? 128 : 150, pad = { l: 40, r: narrow ? 40 : 52, t: 10, b: 30 };
    const duration = Math.max(d.duration || 0, d.steps.at(-1)?.t || 0, 1);
    const base = d.baseline ?? this.context.repo.tokens;
    const pct = (v) => (1 - v / base) * 100;
    const pts = d.progress.filter((p) => p[1] != null);
    const lowest = Math.min(0, ...pts.map((p) => pct(p[1])), ...(d.builds || []).map((b) => b[2]));
    // A run that grew the repository extends the scale below 0, in steps of 25 points, at most to −100%.
    const lo = Math.max(-100, Math.floor(lowest / 25) * 25), hi = 100;
    const x = (t) => pad.l + (Math.min(t, duration) / duration) * (width - pad.l - pad.r);
    const y = (c) => pad.t + (1 - (Math.max(lo, Math.min(hi, c)) - lo) / (hi - lo)) * (height - pad.t - pad.b);
    this.x = x;
    const color = this.context.color;
    const finalPct = this.final != null ? pct(this.final) : null;

    const svg = s('svg', { width, height, viewBox: `0 0 ${width} ${height}`, class: 'trace-svg', role: 'img',
      'aria-label': `Compression over the run${finalPct != null ? `, ending at ${finalPct.toFixed(1)}%` : ''}` });
    const stepT = niceStep(duration, Math.max(3, Math.floor(width / 110)));
    for (let t = 0; t <= duration + 1; t += stepT) {
      svg.append(s('line', { x1: x(t), x2: x(t), y1: pad.t, y2: height - pad.b, class: 'grid' }),
        s('text', { x: x(t), y: height - pad.b + 13, class: 'tick', 'text-anchor': t === 0 ? 'start' : 'middle' }, clock(t)));
    }
    for (let c = lo; c <= hi; c += lo < -50 ? 50 : 25) {
      svg.append(s('line', { x1: pad.l, x2: width - pad.r, y1: y(c), y2: y(c), class: c === 0 ? 'zero' : 'grid' }),
        s('text', { x: pad.l - 6, y: y(c) + 3, class: 'tick', 'text-anchor': 'end' }, `${c}%`));
    }
    // Verified: the compression of the last checkpoint that built and passed verification, as
    // in the paper's budget curves; 0% until the first one. A verified submission ends on its score.
    const builds = d.builds || [];
    const verified = builds.length ? [[0, 0], ...builds.filter((b) => b[1]).map((b) => [b[0], b[2]])] : null;
    if (verified && this.context.result?.status === 'passed' && this.context.result.score != null) verified.push([duration, this.context.result.score]);
    const stepPath = (points) => {
      let path = `M${x(points[0][0])},${y(points[0][1])}`, prev = points[0][1];
      for (const [t, c] of points.slice(1)) { path += `H${x(t)}`; if (c !== prev) path += `V${y(c)}`; prev = c; }
      return [`${path}H${x(duration)}`, prev];
    };
    const size = pts.length ? [[0, pct(pts[0][1])], ...pts.slice(1).map(([t, v]) => [t, pct(v)])] : null;
    const main = verified || size;
    if (main) {
      const [line, last] = stepPath(main);
      if (verified && size) svg.append(s('path', { d: stepPath(size)[0], class: 'size', style: `stroke:${color}` }));
      svg.append(s('path', { d: `${line}V${y(0)}H${x(0)}Z`, class: 'area', style: `fill:${color}` }),
        s('path', { d: line, class: 'level', style: `stroke:${color}` }));
      // Every replayed build: filled where it built, hollow where it failed.
      for (const [t, ok, c] of builds) {
        svg.append(s('circle', { cx: x(t), cy: y(c), r: ok ? 2.4 : 2.6, class: ok ? 'build ok' : 'build fail', style: ok ? `fill:${color}` : null }));
      }
      svg.append(s('circle', { cx: x(duration), cy: y(last), r: 3.5, style: `fill:${color}` }),
        s('text', { x: x(duration) + 7, y: y(last) + 4, class: 'end-label' }, `${last.toFixed(1)}%`));
    } else {
      svg.append(s('text', { x: (pad.l + width - pad.r) / 2, y: (pad.t + height - pad.b) / 2, class: 'tick', 'text-anchor': 'middle' },
        'Lean tokens were not measured during this run'));
    }
    for (const st of d.steps) {
      if (st.k === 'compact') svg.append(s('line', { x1: x(st.t), x2: x(st.t), y1: pad.t, y2: height - pad.b, class: 'compact' }));
    }
    let ticks = '';
    for (const st of d.steps) if (st.d) ticks += `M${x(st.t).toFixed(1)},${height - pad.b + 18}v6`;
    svg.append(s('path', { d: ticks, class: 'changes' }));
    this.cursor = s('line', { x1: pad.l, x2: pad.l, y1: pad.t - 4, y2: height - pad.b + 24, class: 'cursor' });
    const hover = s('line', { x1: -10, x2: -10, y1: pad.t, y2: height - pad.b, class: 'hover' });
    svg.append(this.cursor, hover);
    const timeAt = (e) => {
      const box = svg.getBoundingClientRect();
      return Math.max(0, Math.min(duration, ((e.clientX - box.left - pad.l) / (width - pad.l - pad.r)) * duration));
    };
    svg.addEventListener('pointermove', (e) => {
      if (e.pointerType === 'touch') return;
      const t = timeAt(e);
      hover.setAttribute('x1', x(t)); hover.setAttribute('x2', x(t));
      let level = base, cost = 0, held = null;
      for (const p of d.progress) { if (p[0] > t) break; if (p[1] != null) level = p[1]; cost = p[2] ?? cost; }
      if (verified) for (const [vt, c] of verified) { if (vt > t) break; held = c; }
      showTip(e, `<b>${clock(t, d.estimated)}</b>${held != null ? `<div class="row"><span>Verified</span><span>${held.toFixed(1)}%</span></div>` : ''}
        ${pts.length ? `<div class="row"><span>${verified ? 'Repository size' : 'Compression'}</span><span>${pct(level).toFixed(1)}%</span></div>
        <div class="row"><span>Lean tokens</span><span>${int(level)}</span></div>` : ''}
        ${cost ? `<div class="row"><span>Cost so far</span><span>${usd(cost)}</span></div>` : ''}<div class="note">Click to jump to this point</div>`);
    });
    svg.addEventListener('pointerleave', () => { hideTip(); hover.setAttribute('x1', -10); hover.setAttribute('x2', -10); });
    svg.addEventListener('click', (e) => { hideTip(); this.jumpTo(timeAt(e)); });
    // The paper's cost for the run; the playback's own estimate where the paper has none.
    const cost = this.context.result?.cost ?? d.cost;
    const legend = h('div', { class: 'trace-legend muted' },
      verified ? [
        h('span', { title: 'Compression of the last checkpoint that built and passed verification, as in the paper\'s budget curves' },
          h('i', { class: 'line-key', style: { background: color } }), 'Verified'),
        h('span', { title: 'Compression of the sources at every checkpoint, built or not' }, h('i', { class: 'line-key size', style: { background: color } }), 'Repository size'),
        h('span', {}, h('i', { class: 'dot-key', style: { background: color } }), 'Built'),
        h('span', {}, h('i', { class: 'dot-key fail' }), 'Build failed'),
      ] : h('span', {}, h('i', { class: 'swatch', style: { background: color } }), 'Compression'),
      h('span', {}, h('i', { class: 'tick-key' }), 'Source change'),
      d.steps.some((st) => st.k === 'compact') ? h('span', {}, h('i', { class: 'compact-key' }), 'Context compacted') : null,
      h('span', { class: 'num' }, `${tokens(base)} → ${tokens(this.final)} Lean tokens`, cost != null ? ` · ${usd(cost)}` : ''));
    this.chart.replaceChildren(svg, legend);
    this.onScroll();
  }

  // ── steps ──────────────────────────────────────────────────────────────
  visibleSteps() {
    const q = this.query;
    return this.data.steps.filter((st) => {
      if (this.action && (st.k !== 'tool' || !actionsOf(st).includes(this.action))) return false;
      if (this.filter === 'talk' && st.k === 'tool') return false;
      return !q || this.haystack(st).includes(q);
    });
  }

  renderList() {
    if (!this.data) return;
    this.pressed();
    const steps = this.visibleSteps();
    const fold = this.filter === 'all' && !this.query && !this.action;
    const frag = document.createDocumentFragment();
    this.rows = [];
    for (let k = 0; k < steps.length; k++) {
      const st = steps[k];
      let end = k;
      if (fold && st.k === 'tool' && st.c === 'wait') while (end + 1 < steps.length && steps[end + 1].k === 'tool' && steps[end + 1].c === 'wait') end++;
      if (end - k + 1 >= FOLD_WAITS) {
        const group = steps.slice(k, end + 1);
        const row = this.groupRow(group);
        frag.append(row);
        this.rows.push([row, st.t]);
        k = end;
        continue;
      }
      const row = this.stepRow(st);
      frag.append(row);
      this.rows.push([row, st.t]);
    }
    if (!steps.length) frag.append(h('div', { class: 'note' }, 'No step matches.'));
    this.list.replaceChildren(frag);
    const total = this.data.steps.length;
    this.count.textContent = steps.length === total ? `${int(total)} steps` : `${int(steps.length)} of ${int(total)} steps`;
    this.onScroll();
  }

  stepRow(st) {
    const approx = this.data.estimated;
    const time = h('span', { class: 'tr-t num', title: approx ? 'Estimated: this harness logs no clock' : null }, clock(st.t, approx));
    if (st.k !== 'tool') {
      const label = st.k === 'think' && st.s ? 'Reasoning summary' : KIND[st.k];
      // Compaction summaries are long and repeat the task: folded until asked for.
      if (st.k === 'compact') return this.foldRow(st, time, h('span', { class: 'tr-title' }, label),
        () => [h('div', { class: 'tr-text' }, st.x || '')]);
      const long = st.x.length > 600 || st.x.split('\n').length > 8;
      const body = h('div', { class: `tr-text${long ? ' clamp' : ''}` }, st.x);
      const open = this.allOpen || this.expanded.has(st);
      if (open) body.classList.remove('clamp');
      const more = long ? h('button', { type: 'button', class: 'tr-more', onclick: () => {
        const opened = body.classList.toggle('clamp');
        more.textContent = opened ? 'Show all' : 'Show less';
        if (!opened) this.expanded.add(st); else this.expanded.delete(st);
      } }, open ? 'Show less' : 'Show all') : null;
      return h('div', { class: `tr-step ${st.k}`, role: 'listitem' },
        h('div', { class: 'tr-line' }, time, h('span', { class: 'tr-kind' }, label), st.a ? h('span', { class: 'chip' }, st.a) : null),
        body, more);
    }
    const badges = [];
    if (st.d) {
      if (st.delta) badges.push(h('span', { class: `tr-delta ${st.delta < 0 ? 'down' : 'up'}`, title: `After this step: ${int(st.d.tok)} Lean tokens, ${this.compression(st.d.tok)} compressed` }, `${signedInt(st.delta)} tok`));
      badges.push(h('span', { class: 'tr-lines num', title: `${st.d.f.length} file${st.d.f.length === 1 ? '' : 's'} changed` },
        h('b', { class: 'add' }, `+${int(st.d.a)}`), ' ', h('b', { class: 'del' }, `−${int(st.d.r)}`)));
    }
    if (st.d?.b != null) badges.push(h('span', { class: `chip ${st.d.b ? 'good' : 'bad'}`, title: st.d.b ? 'This checkpoint built and passed verification in the replay' : 'This checkpoint failed to build or verify in the replay' }, st.d.b ? 'builds' : 'build failed'));
    if (st.e) badges.push(h('span', { class: 'chip bad' }, 'error'));
    const extra = st.c === 'run' ? st.ln ?? (st.i ? st.i.split('\n').length - 1 : 0) : 0;
    const title = [h('code', { class: 'tr-name' }, st.n), h('span', { class: 'tr-title', title: st.ti }, shortPaths(st.ti || ''),
      extra > 0 ? h('span', { class: 'muted' }, `  +${int(extra)} line${extra === 1 ? '' : 's'}`) : null), st.a ? h('span', { class: 'chip' }, st.a) : null];
    // On a phone the row shows only the kind of action; what it was opens with it.
    const what = () => h('div', { class: 'tr-what' }, h('code', { class: 'tr-name' }, st.n), h('span', {}, shortPaths(st.ti || '')),
      st.a ? h('span', { class: 'chip' }, st.a) : null);
    return this.foldRow(st, time, title, () => [what(), ...this.toolBody(st)], badges);
  }

  // A row that opens to show its body, built on first open.
  foldRow(st, time, title, body, badges = []) {
    const cls = st.k === 'tool' ? `tr-step tool c-${st.c}${st.e ? ' err' : ''}${st.d ? ' chg' : ''}` : `tr-step ${st.k}`;
    const panel = h('div', { class: 'tr-body', hidden: true });
    const head = h('button', { type: 'button', class: 'tr-head', 'aria-expanded': 'false' },
      time, h('span', { class: 'tr-cat' }, st.k === 'tool' ? [...new Set(actionsOf(st))].map((a) => h('span', {}, a)) : ''), title, h('span', { class: 'tr-badges' }, badges));
    const row = h('div', { class: cls, role: 'listitem' }, head, panel);
    // The body may still be in the pack: fetch its chunk, then draw it.
    const open = (on) => {
      panel.hidden = !on;
      head.setAttribute('aria-expanded', String(on));
      if (on) this.expanded.add(st); else this.expanded.delete(st);
      if (!on || panel.dataset.drawn) return;
      panel.dataset.drawn = '1';
      const draw = () => panel.replaceChildren(...body());
      const pending = this.chunks.get(st.p);
      if (st.p == null || (pending && pending.done)) { draw(); return; }
      panel.replaceChildren(h('div', { class: 'muted tr-empty' }, 'Loading…'));
      this.body(st.p).then(draw)
        .catch(() => { delete panel.dataset.drawn; panel.replaceChildren(h('div', { class: 'note' }, 'Could not load this step. Tap again to retry.')); });
    };
    head.addEventListener('click', () => open(panel.hidden));
    if (this.allOpen || this.expanded.has(st)) open(true);
    return row;
  }

  groupRow(group) {
    const first = group[0], last = group.at(-1);
    const panel = h('div', { class: 'tr-group-body', hidden: true });
    const names = [...new Set(group.map((st) => st.n))].join(', ');
    const head = h('button', { type: 'button', class: 'tr-head', 'aria-expanded': 'false' },
      h('span', { class: 'tr-t num' }, clock(first.t, this.data.estimated)), h('span', { class: 'tr-cat' }, 'Other'),
      h('code', { class: 'tr-name' }, names),
      h('span', { class: 'tr-title' }, `${group.length} polls while a command ran, until ${clock(last.t, this.data.estimated)}`),
      h('span', { class: 'tr-badges' }, h('span', { class: 'tr-polls muted num' }, `×${group.length}`)));
    head.addEventListener('click', () => {
      if (!panel.childElementCount) panel.append(...group.map((st) => this.stepRow(st)));
      panel.hidden = !panel.hidden;
      head.setAttribute('aria-expanded', String(!panel.hidden));
    });
    return h('div', { class: 'tr-step tool c-wait group', role: 'listitem' }, head, panel);
  }

  toolBody(st) {
    const parts = [];
    const section = (label, node) => h('div', { class: 'tr-sec' }, h('span', { class: 'eyebrow' }, label), node);
    if (st.i) {
      const label = st.c === 'run' ? (st.il === 'js' ? 'Script' : 'Command') : st.il === 'diff' ? 'Edit' : st.c === 'write' ? 'Content' : 'Input';
      // An edit's own patch repeats the source change below: keep it folded.
      if (st.d && (st.c === 'edit' || st.c === 'write')) {
        const folded = h('details', { class: 'tr-sec tr-fold' }, h('summary', { class: 'eyebrow' }, `${label} as the agent wrote it`));
        folded.addEventListener('toggle', () => { if (folded.open && folded.childElementCount === 1) folded.append(textBlock(st.i, st.il)); });
        parts.push(folded);
      } else parts.push(section(label, textBlock(st.i, st.il)));
    }
    if (st.o && !/^\s*(\{\s*\}|null)?\s*$/.test(st.o)) parts.push(section(st.e ? 'Output (error)' : 'Output', textBlock(st.o.replace(/^\n+/, ''), 'out', st.e)));
    if (!st.i && !st.o && !st.d) parts.push(h('div', { class: 'muted tr-empty' }, 'No input or output recorded.'));
    if (st.d) {
      const files = st.d.f.map((f) => h('code', {}, f));
      parts.push(section('Source change', h('div', { class: 'tr-change' },
        h('div', { class: 'tr-files' }, files, st.d.tok != null ? h('span', { class: 'muted num' }, `${int(st.d.tok)} Lean tokens after · ${this.compression(st.d.tok)} compressed`) : null),
        patchBlock(st.d.diff))));
    }
    return parts;
  }

  // ── scroll position ↔ timeline ─────────────────────────────────────────
  onScroll() {
    if (this.pending || !this.cursor || !this.rows?.length) return;
    this.pending = requestAnimationFrame(() => {
      this.pending = 0;
      const top = this.list.scrollTop;
      let lo = 0, hi = this.rows.length - 1;
      while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (this.rows[mid][0].offsetTop <= top + 4) lo = mid; else hi = mid - 1; }
      const t = this.rows[lo][1];
      this.cursor.setAttribute('x1', this.x(t)); this.cursor.setAttribute('x2', this.x(t));
    });
  }

  jumpTo(t) {
    if (!this.rows?.length) return;
    let k = this.rows.findIndex(([, rt]) => rt >= t);
    if (k < 0) k = this.rows.length - 1;
    const row = this.rows[k][0];
    this.list.scrollTop = row.offsetTop - 4;
    row.classList.remove('flash');
    void row.offsetWidth;
    row.classList.add('flash');
  }
}

// Harness scratch directories make titles unreadable: /tmp/leancompression-antigravity-home/.gemini/…/brain/<id>/x → ~scratch/x.
function shortPaths(text) {
  return text.replace(/\/tmp\/leancompression-[\w-]+-home\/(?:\.[\w-]+\/)*(?:[\w-]+\/)*?brain\/[0-9a-f-]{36}\//g, '~scratch/')
    .replace(/\/testbed\//g, '');
}

function niceStep(duration, count = 6) {
  const target = duration / count;
  const steps = [60, 120, 300, 600, 900, 1800, 3600, 7200, 10800, 14400, 21600, 43200];
  return steps.find((v) => v >= target) || 86400;
}

// An input or output: diffs and Lean as code, everything else as plain text.
function textBlock(text, lang, error = false) {
  if (lang === 'diff') return patchBlock(text);
  if (lang === 'lean') return codeBlock(text.split('\n'), 1, { wrap: true });
  const pre = h('pre', { class: `tr-pre${error ? ' err' : ''}${lang === 'out' ? ' out' : ''}` });
  // The exporter marks clipped middles with a line "… N more lines …".
  const html = esc(text).replace(/^… (\d+) more lines …$/gm, '<span class="omit">… $1 more lines …</span>');
  pre.innerHTML = html;
  return pre;
}
