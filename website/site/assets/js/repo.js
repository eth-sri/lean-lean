import { h, load, loadChunk, logo, modelColor, modelLabel, highestReasoningModels, pct, usd, hours, tokens, int, signedInt, signedPct, seg, esc, showTip, hideTip, cssVar, ramp, ORIGIN, ASSETS } from './util.js?v=e38b2b7f7d';
import { GraphView, STATUS, shapeOf, radius } from './graph.js?v=e38b2b7f7d';
import { codeBlock, diffBlock, splitBlock, hunksBlock, wholeFileBlock } from './code.js?v=e38b2b7f7d';
import { TraceView } from './trace.js?v=e38b2b7f7d';

const CATEGORY_TEXT = {
  unchanged: 'Unchanged', modified: 'Modified', comments: 'Comments or layout only', deleted: 'Deleted', dead: 'Dead code, removed', added: 'Added',
};
const CATEGORY_VAR = {
  unchanged: '--g-flat', modified: '--g-modified', comments: '--g-comments', deleted: '--g-deleted', dead: '--g-dead', added: '--g-added',
};
const STATUS_TEXT = {
  unchanged: ['Unchanged', ''], rewritten: ['Proof rewritten', 'good'], structural: ['Rewritten, dependencies changed', 'good'],
  comments: ['Comments or layout only', ''],
  deleted: ['Deleted', 'bad'], dead: ['Dead code, deleted', 'bad'], added: ['Added by the agent', 'warn'],
};
// Preprocessing sits beside the agents: the raw repository before it, the benchmark baseline after it.
const PREP = { short: 'prep', label: 'Preprocessing' };
const KIND_LABEL = { theorem: 'theorem', def: 'def', inductive: 'inductive', ctor: 'constructor', rec: 'recursor', opaque: 'opaque', axiom: 'axiom',
  instance: 'instance', syntax: 'notation / macro', other: 'declaration' };
// Node shapes of the graph (graph.js shapeOf), as legend rows.
const SHAPE_TEXT = [['', 'Theorem'], ['sq', 'Definition'], ['tri', 'Notation, macro or other']];

const CLOSE_ICON = '<svg viewBox="0 0 20 20" width="18" height="18" aria-hidden="true"><path d="M5 5l10 10M15 5 5 15" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>';
const CHEVRON_ICON = '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="m4 6 4 4 4-4" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>';

let current = null;
let repoHighestOnly = true; // live page state for in-place updates

// The repository picker: a button that opens a filterable list, grouped by scale.
// go(id) is called with the chosen repository; close() hides the list.
function repoPicker(bench, repo, go) {
  const button = h('button', { type: 'button', class: 'pick-btn', id: 'repo-picker', 'aria-haspopup': 'listbox', 'aria-expanded': 'false', 'aria-label': `Repository: ${repo.name}. Switch repository` },
    h('span', { class: 'pick-name' }, repo.name), h('span', { class: 'pick-size' }, tokens(repo.tokens)), h('span', { class: 'pick-chev', html: CHEVRON_ICON }));
  const filter = h('input', { type: 'search', class: 'pick-filter', placeholder: 'Find a repository', 'aria-label': 'Find a repository', autocomplete: 'off', spellcheck: 'false',
    role: 'combobox', 'aria-controls': 'pick-list', 'aria-expanded': 'true' });
  const options = [];
  const groups = bench.bands.map((b) => h('li', { class: 'pick-group', role: 'presentation' },
    h('div', { class: 'pick-group-head', 'aria-hidden': 'true' }, h('b', {}, b.scale), h('span', {}, `${b.band} tokens`)),
    h('ul', { role: 'group', 'aria-label': b.scale },
      bench.repositories.filter((x) => x.band === b.band).sort((a, c) => a.name.localeCompare(c.name)).map((x) => {
        const li = h('li', { role: 'option', id: `pick-${x.id}`, title: x.name, 'aria-selected': String(x.id === repo.id),
          onclick: () => choose(x.id), onmousemove: () => { if (active !== li) setActive(li); } },
        h('span', { class: 'pick-name' }, x.name), h('span', { class: 'pick-size' }, tokens(x.tokens)));
        li.dataset.id = x.id; li.dataset.key = x.name.toLowerCase();
        options.push(li);
        return li;
      }))));
  const list = h('ul', { class: 'pick-list', id: 'pick-list', role: 'listbox', 'aria-label': 'Repositories' }, groups);
  const empty = h('p', { class: 'pick-empty', hidden: true }, 'No repository matches.');
  const menu = h('div', { class: 'pick-menu', hidden: true }, filter, list, empty);
  const node = h('div', { class: 'pick' }, button, menu);

  let active = null;
  const visible = () => options.filter((li) => !li.hidden);
  // Scroll the list (never the page) so that li shows below its sticky group heading.
  const reveal = (li) => {
    const head = li.closest('.pick-group').firstChild.offsetHeight;
    if (li.offsetTop - head < list.scrollTop) list.scrollTop = li.offsetTop - head - 4;
    else if (li.offsetTop + li.offsetHeight > list.scrollTop + list.clientHeight) list.scrollTop = li.offsetTop + li.offsetHeight - list.clientHeight + 4;
  };
  const setActive = (li) => {
    active?.classList.remove('active');
    active = li;
    li?.classList.add('active');
    if (li) filter.setAttribute('aria-activedescendant', li.id); else filter.removeAttribute('aria-activedescendant');
  };
  const apply = () => {
    const q = filter.value.trim().toLowerCase();
    for (const li of options) li.hidden = Boolean(q) && !li.dataset.key.includes(q);
    for (const g of groups) g.hidden = !g.querySelector('[role=option]:not([hidden])');
    const shown = visible();
    empty.hidden = shown.length > 0;
    setActive(shown[0] || null);
    list.scrollTop = 0;
  };
  const outside = (e) => { if (!node.contains(e.target)) close(); };
  const open = () => {
    menu.hidden = false; node.classList.add('open'); button.setAttribute('aria-expanded', 'true');
    filter.value = ''; apply();
    // Open at the current repository, and keep the list inside the screen.
    const here = options.find((li) => li.dataset.id === repo.id);
    setActive(here);
    list.scrollTop = here.offsetTop - list.clientHeight / 2;
    menu.classList.remove('flip');
    if (menu.getBoundingClientRect().right > document.documentElement.clientWidth - 16) menu.classList.add('flip');
    // A phone's keyboard would cover the list: there the field is focused only when tapped.
    if (matchMedia('(pointer: fine)').matches) filter.focus();
    document.addEventListener('pointerdown', outside, true);
  };
  const close = (refocus = false) => {
    if (menu.hidden) return;
    menu.hidden = true; node.classList.remove('open'); button.setAttribute('aria-expanded', 'false');
    document.removeEventListener('pointerdown', outside, true);
    if (refocus) button.focus();
  };
  const choose = (id) => { close(); if (id !== repo.id) go(id); };
  button.addEventListener('click', () => (menu.hidden ? open() : close()));
  filter.addEventListener('input', apply);
  node.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !menu.hidden) { e.stopPropagation(); close(true); }
    else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      if (menu.hidden) { open(); return; }
      const shown = visible();
      if (!shown.length) return;
      const i = shown.indexOf(active), step = e.key === 'ArrowDown' ? 1 : -1;
      setActive(shown[i < 0 ? 0 : (i + step + shown.length) % shown.length]);
      reveal(active);
    } else if (e.key === 'Enter' && !menu.hidden && active) { e.preventDefault(); choose(active.dataset.id); }
    else if (e.key === 'Tab') close();
  });
  return { node, close };
}

// A compact model menu shares the desktop pills' content and availability.
function modelPicker(tabs, choose) {
  const content = h('span', { class: 'model-pick-value' });
  const button = h('button', { type: 'button', class: 'pick-btn model-pick-btn',
    'aria-haspopup': 'listbox', 'aria-expanded': 'false', 'aria-controls': 'model-options' },
    content, h('span', { class: 'pick-chev', html: CHEVRON_ICON }));
  const list = h('div', { id: 'model-options', role: 'listbox', 'aria-label': 'Models', class: 'model-options' });
  const menu = h('div', { class: 'pick-menu model-pick-menu', hidden: true },
    h('div', { class: 'model-menu-head' }, h('span', {}, 'Choose a model'), h('span', {}, 'Score')), list);
  const node = h('div', { class: 'pick model-picker' }, button, menu);
  let selected = null;
  const close = (refocus = false) => {
    menu.hidden = true; node.classList.remove('open'); button.setAttribute('aria-expanded', 'false');
    document.removeEventListener('pointerdown', outside, true);
    if (refocus) button.focus();
  };
  const outside = (event) => { if (!node.contains(event.target)) close(); };
  const available = () => [...list.children].filter((option) => !option.disabled);
  const open = () => {
    menu.hidden = false; node.classList.add('open'); button.setAttribute('aria-expanded', 'true');
    document.addEventListener('pointerdown', outside, true);
    const option = available().find((option) => option.dataset.short === selected) || available()[0];
    option?.focus({ preventScroll: true });
    if (option) list.scrollTop = option.offsetTop - list.offsetTop - list.clientHeight / 2;
  };
  const copy = (pill) => [...pill.children].map((child) => child.cloneNode(true));
  const refresh = (short) => {
    selected = short;
    const pills = [...tabs.querySelectorAll('button')].filter((pill) => !pill.hidden);
    const active = pills.find((pill) => pill.dataset.short === short);
    button.style.setProperty('--model', active?.style.getPropertyValue('--model') || 'var(--ink-2)');
    button.setAttribute('aria-label', `Model: ${active?.querySelector('.t b').textContent || 'Choose a model'}`);
    content.replaceChildren(...(active ? copy(active) : ['Choose a model']));
    list.replaceChildren(...pills.map((pill) => h('button', {
      type: 'button', role: 'option', tabindex: '-1', class: 'model-option',
      'data-short': pill.dataset.short, 'aria-selected': String(pill === active),
      disabled: pill.disabled, title: pill.title, style: { '--model': pill.style.getPropertyValue('--model') || 'var(--ink-2)' },
      onclick: () => { close(true); if (pill.dataset.short !== selected) choose(pill.dataset.short); },
    }, ...copy(pill), h('span', { class: 'model-check', 'aria-hidden': 'true' }, pill === active ? '✓' : ''))));
  };
  button.addEventListener('click', () => menu.hidden ? open() : close());
  node.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && !menu.hidden) { event.preventDefault(); event.stopPropagation(); close(true); }
    else if (['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) {
      event.preventDefault();
      if (menu.hidden) { open(); return; }
      const options = available();
      const index = options.indexOf(document.activeElement);
      const next = event.key === 'Home' ? 0 : event.key === 'End' ? options.length - 1
        : (index + (event.key === 'ArrowDown' ? 1 : -1) + options.length) % options.length;
      options[next]?.focus({ preventScroll: true });
      options[next]?.scrollIntoView({ block: 'nearest' });
    } else if (event.key === 'Tab') close();
  });
  node.addEventListener('focusout', (event) => { if (!node.contains(event.relatedTarget)) close(); });
  return { node, refresh, close };
}

// Agents with before/after graphs and diffs: verified submissions whose graphs were extracted.
const explorable = (bench, results, m) => bench.models.includes(m) && results[m.key]?.status === 'passed' && Boolean(results[m.key].explorer);

// kind: which explorer sits under the summary, 'repo' (files and diff), 'graph' (dependency graph) or 'trace' (the agent's run)
// target: the declaration (Graph view) or file (Diff view) to open, from the address.
// `home` is the address an empty hash stands for: while it is what is open, the address stays empty.
export async function renderRepository(view, bench, id, shortArg, kind, inPlace, target = null, home = null) {
  const repo = bench.repositories.find((r) => r.id === id);
  if (!repo) { view.replaceChildren(h('div', { class: 'error' }, `Unknown repository ${id}.`)); return; }
  const results = bench.results[id] || {};
  const traces = await load('data/traces/index.json').catch(() => ({}));
  const agents = [...bench.models, ...(bench.extra_models || [])];
  // A linked run without dependency graphs opens its own trace.
  const requested = agents.find((m) => m.short === shortArg);
  if (kind !== 'trace' && requested && !explorable(bench, results, requested) && traces[id]?.[requested.short]) kind = 'trace';
  // Graph and Diff need a verified submission; Trace any run, failed ones too.
  const available = kind === 'trace' ? agents.filter((m) => traces[id]?.[m.short]) : agents.filter((m) => explorable(bench, results, m));
  const byShort = Object.fromEntries(agents.map((m) => [m.short, m]));
  const model = shortArg === PREP.short && kind !== 'trace' ? PREP : byShort[shortArg] && available.includes(byShort[shortArg]) ? byShort[shortArg]
    : [...available].sort((a, b) => (results[b.key]?.score ?? -1) - (results[a.key]?.score ?? -1))[0] || null;

  if (inPlace && current && current.id === id && document.body.contains(current.root)) {
    current.home = home;
    await current.update(model, kind, target);
    return;
  }
  if (current) current.destroy();
  view.replaceChildren(h('div', { class: 'loading' }, `Loading ${repo.name}…`));
  const graph = await load(`data/repos/${id}/graph.json.gz`);
  load(`data/repos/${id}/source.json.gz`).catch(() => {});  // the source index (≈1–11 kB), so the first declaration opens fast
  current = new RepoPage(view, bench, repo, graph, kind, traces[id] || {});
  current.home = home;
  // Direct links open their requested view immediately on narrow screens.
  current.pendingFull = kind;
  await current.update(model, kind, target);
  const page = current;
  return () => { page.destroy(); if (current === page) current = null; };
}

class RepoPage {
  constructor(view, bench, repo, graph, kind, traces) {
    this.bench = bench; this.repo = repo; this.graph = graph; this.id = repo.id;
    this.traces = traces;
    this.graphs = { base: graph };
    this.results = bench.results[repo.id] || {};
    this.selected = -1;
    const byScore = [...bench.models].sort((a, b) => (this.results[b.key]?.score ?? -1) - (this.results[a.key]?.score ?? -1));
    this.byScore = byScore;

    // Header
    const r = repo;
    this.picker = repoPicker(bench, r, (id) => { location.hash = `#/${this.kind}/${id}/${this.model?.short || '-'}`; });
    const head = h('div', { class: 'repo-head compact' },
      h('div', { style: { display: 'grid', gap: '4px', minWidth: 0 } },
        h('span', { class: 'eyebrow' }, 'Repository'),
        h('div', { class: 'repo-title' }, h('h2', {}, r.name),
          r.url ? iconLink(`${r.url}/tree/${r.commit}`, 'github', `${r.name} on GitHub, at the benchmarked commit`) : null,
          r.id.startsWith('palomar__') ? iconLink(`https://palomar-registry.org/entry?id=${encodeURIComponent(r.id.replace(/^palomar__/, 'PALOMAR-'))}`, 'palomar', 'Its entry in the Palomar Registry') : null),
        h('div', { class: 'meta' },
          h('span', {}, `${r.scale} · ${tokens(r.tokens)} Lean tokens`), h('span', {}, `${int(r.core)} protected result${r.core === 1 ? '' : 's'}`),
          r.family ? h('span', {}, `${r.family} (${r.arxiv})`) : null, h('code', {}, r.toolchain.replace('leanprover/lean4:', 'Lean ')),
          h('code', {}, r.commit.slice(0, 10)))),
      // The explorer page provides its own navigation back to the leaderboard.
      bench.solo ? null : h('div', { class: 'repo-head-tools' }, this.picker.node));

    const prepSaved = r.raw_tokens ? (1 - r.tokens / r.raw_tokens) * 100 : 0;
    const prepTab = h('button', {
      type: 'button', class: 'model-tab prep', 'data-short': PREP.short, 'aria-pressed': 'false',
      title: 'The repository before and after LeanLean\'s preprocessing',
      onclick: () => { location.hash = `#/${this.kind}/${this.id}/${PREP.short}`; },
    }, h('span', { class: 'prep-mark', 'aria-hidden': 'true', style: { '--removed': `${prepSaved * 3.6}deg` } }), h('span', { class: 't' }, h('b', {}, PREP.label),
      h('span', {}, signedPct(-prepSaved, 1))));
    const tab = (m) => {
      const res = this.results[m.key];
      return h('button', {
        type: 'button', class: 'model-tab', 'data-short': m.short, style: { '--model': modelColor(m) }, 'aria-pressed': 'false',
        onclick: () => { location.hash = `#/${this.kind}/${this.id}/${m.short}`; },
      }, logo(m, 'xs'), h('span', { class: 't' }, h('b', {}, modelLabel(m)),
        res?.status === 'passed' ? h('span', {}, pct(res.score)) : res?.status === 'pending' ? h('span', { class: 'muted' }, 'not finished')
          : h('span', { class: 'fail' }, res ? 'failed' : 'no run')));
    };
    // Models run only on the Mini subset follow the main models, where they ran.
    const extras = (bench.extra_models || []).filter((m) => this.results[m.key]);
    this.agents = [...byScore, ...extras];
    this.highestModels = new Set(highestReasoningModels(this.agents));
    this.highestOnly = h('input', { type: 'checkbox', role: 'switch', checked: repoHighestOnly, onchange: () => {
      repoHighestOnly = this.highestOnly.checked;
      this.refreshTabs();
      if (!repoHighestOnly || !this.model || this.model === PREP || this.highestModels.has(this.model)) return;
      const available = this.agents.filter((m) => this.highestModels.has(m) &&
        (this.kind === 'trace' ? this.traces[m.short] : explorable(this.bench, this.results, m)));
      const next = available.find((m) => m.label === this.model.label && m.provider === this.model.provider) || available[0];
      location.hash = `#/${this.kind}/${this.id}/${next?.short || PREP.short}`;
    } });
    const filter = h('label', { class: 'results-filter' }, this.highestOnly, h('span', {}, 'Show only highest reasoning levels'));
    this.tabs = h('div', { class: 'model-tabs', role: 'group', 'aria-label': 'Agent' }, prepTab, h('i', { class: 'tab-sep', 'aria-hidden': 'true' }),
      byScore.map(tab), extras.length ? [h('i', { class: 'tab-sep', 'aria-hidden': 'true' }), extras.map(tab)] : null);

    this.modelPicker = modelPicker(this.tabs, (short) => {
      location.hash = `#/${this.kind}/${this.id}/${short}`;
    });

    // Outcome of the selected agent: one strip of numbers and its compression origin.
    this.outcome = h('div', { class: 'outcome-strip' });

    // Explorer: Diff and Graph are two views of the same repository and agent; only this part switches.
    const viewLink = (target, label) => h('a', { href: `#/${target}/${r.id}`, 'data-view': target }, label);
    this.viewTabs = h('nav', { class: 'view-tabs', 'aria-label': 'Explorer view' }, viewLink('graph', 'Graph'), viewLink('repo', 'Diff'), viewLink('trace', 'Trace'));
    this.files = h('div', { class: 'files' });
    this.graphHost = h('div', { class: 'graph-host' });

    // Below 1100px Graph and Diff are not part of the page: two buttons open them
    // full screen, under a bar that closes them again (site.css).
    this.narrow = matchMedia('(max-width: 1099.98px)');
    this.diffTitle = h('div', { class: 'full-title' });
    this.diffToggle = h('button', { type: 'button', class: 'files-toggle', 'aria-expanded': 'false', 'aria-label': 'Show the files',
      onclick: () => this.setDrawer(!this.diffHost.classList.contains('files-open')) }, svgIcon('files'), h('span', {}, 'Files'));
    this.diffHost = h('div', { class: 'diff-host' },
      h('div', { class: 'full-bar' }, this.diffToggle, this.diffTitle, this.closeButton('Close the diff')),
      this.files, h('div', { class: 'files-scrim', onclick: () => this.setDrawer(false) }));
    this.traceTitle = h('div', { class: 'full-title' });
    this.traceBody = h('div', { class: 'trace-body' });
    this.traceHost = h('div', { class: 'trace-host' },
      h('div', { class: 'full-bar' }, this.traceTitle, this.closeButton('Close the trace')), this.traceBody);
    this.graphLaunchSub = h('span', {});
    this.diffLaunchSub = h('span', {});
    this.traceLaunchSub = h('span', {});
    const launchButton = (target, label, sub) => h('button', { type: 'button', class: 'view-launch-btn', onclick: () => this.launchView(target) },
      svgIcon(target === 'repo' ? 'diff' : target), h('span', { class: 'view-launch-text' }, h('b', {}, label), sub));
    this.launch = h('div', { class: 'view-launch' },
      launchButton('graph', 'Graph', this.graphLaunchSub), launchButton('repo', 'Diff', this.diffLaunchSub), launchButton('trace', 'Trace', this.traceLaunchSub));

    const inner = h('div', { style: { maxWidth: '96rem', margin: '0 auto', width: '100%', display: 'grid', gap: '14px' } },
      head, filter, this.tabs, this.modelPicker.node, this.outcome, this.viewTabs, this.launch, this.diffHost, this.graphHost, this.traceHost);
    this.root = h('div', { class: 'page wide' }, inner);
    view.replaceChildren(this.root);

    this.onKey = (e) => {
      if (e.key !== 'Escape' || !this.full || this.dialog?.open || document.activeElement === this.searchInput) return;
      if (this.full === 'graph') {
        if (this.selected >= 0) this.gv.select(-1);
        else if (this.listExpanded) { this.listExpanded = false; this.renderInspector(); }
        else this.closeFull();
      } else if (this.diffHost.classList.contains('files-open')) this.setDrawer(false);
      else this.closeFull();
    };
    // The phone's back button closes a full-screen view rather than leaving the page.
    this.onPop = () => { if (this.full && !history.state?.leanleanFull) this.closeFull(true); };
    this.onNarrow = () => { if (!this.narrow.matches) this.closeFull(); };
    document.addEventListener('keydown', this.onKey);
    window.addEventListener('popstate', this.onPop);
    this.narrow.addEventListener('change', this.onNarrow);
    this.setKind(kind);
  }

  closeButton(label) {
    return h('button', { type: 'button', class: 'full-close', 'aria-label': label, onclick: () => this.closeFull(), html: CLOSE_ICON });
  }

  // Repository, agent and score, for the bar over a full-screen view; counts for the buttons that open them.
  renderTitles() {
    const res = this.model && !this.isPrep ? this.results[this.model.key] : null;
    const title = () => [h('b', {}, this.repo.name),
      h('span', {}, this.isPrep ? 'Preprocessing' : this.model ? `${modelLabel(this.model)}${res ? ` · ${pct(res.score)}` : ''}` : 'Baseline')];
    this.diffTitle.replaceChildren(...title());
    this.graphTitle?.replaceChildren(...title());
    this.traceTitle.replaceChildren(...title());
    const run = this.model && this.traces[this.model.short];
    this.traceLaunchSub.textContent = run ? `${int(run.steps)} steps` : 'No trace';
    this.graphLaunchSub.textContent = `${int(this.graph.counts.nodes)} declarations`;
    const changed = this.pair ? this.pair.files.filter((f) => f.state !== 'unchanged').length : 0;
    this.diffLaunchSub.textContent = this.pair ? `${int(changed)} of ${int(this.pair.files.filter((f) => !f.harness).length)} files changed` : 'No submission';
  }

  // A button on a narrow screen: switch to that view, then open it full screen.
  launchView(kind) {
    if (kind === this.kind) { this.openFull(); return; }
    this.pendingFull = kind;
    location.replace(`#/${kind}/${this.id}/${this.model?.short || '-'}`);
  }

  // Which agents a view can show: Graph and Diff need a verified submission with graphs, Trace any recorded run.
  refreshTabs() {
    const agents = [...this.bench.models, ...(this.bench.extra_models || [])];
    for (const b of this.tabs.querySelectorAll('button')) {
      if (b.dataset.short === PREP.short) {
        b.disabled = this.kind === 'trace';
        b.title = this.kind === 'trace' ? 'Preprocessing is not an agent run' : 'The repository before and after LeanLean\'s preprocessing';
        continue;
      }
      const m = agents.find((x) => x.short === b.dataset.short);
      b.hidden = this.highestOnly.checked && !this.highestModels.has(m);
      const res = this.results[m.key];
      if (this.kind === 'trace') {
        const run = this.traces[m.short];
        b.disabled = !run;
        b.title = run ? `${modelLabel(m)}: ${int(run.steps)} steps, ${int(run.tools)} tool calls` : 'No trace was recorded for this run';
      } else {
        b.disabled = !explorable(this.bench, this.results, m);
        b.title = !res ? 'No run' : res.status === 'pending' ? 'This run is not finished yet' : res.status !== 'passed' ? 'Submission failed verification; scored 0. Its trace is under Trace.'
          : b.disabled ? 'Dependency graphs were not extracted for this agent. Its trace is under Trace.' : `Show ${modelLabel(m)}`;
      }
    }
    this.modelPicker.refresh(this.model?.short);
  }

  setDrawer(open) {
    this.diffHost.classList.toggle('files-open', open);
    this.diffToggle.setAttribute('aria-expanded', String(open));
  }

  setKind(kind) {
    this.kind = kind;
    const isGraph = kind === 'graph';
    for (const a of this.viewTabs.children) {
      if (a.dataset.view === kind) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
    }
    this.diffHost.hidden = kind !== 'repo';
    this.graphHost.hidden = !isGraph;
    this.traceHost.hidden = kind !== 'trace';
    // The trace has its own compression-over-time chart.
    this.root?.classList.toggle('on-trace', kind === 'trace');
    if (isGraph && !this.gv) this.buildGraph();
    if (kind === 'trace' && !this.tv) this.tv = new TraceView(this.traceBody);
  }

  buildGraph() {
    this.phaseHost = h('span', { style: { display: 'contents' } });
    this.edgeToggle = h('button', { type: 'button', class: 'chip edge-toggle', id: 'edges-toggle', 'aria-pressed': 'true', onclick: (e) => {
      const on = e.currentTarget.getAttribute('aria-pressed') !== 'true';
      e.currentTarget.setAttribute('aria-pressed', String(on));
      this.gv.setEdges(on);
    } }, h('i', { 'aria-hidden': 'true' }), 'Edges');
    const bar = h('div', { class: 'explorer-bar' },
      h('div', { style: { display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap' } }, h('span', { class: 'muted', style: { fontSize: '0.8rem' } }, 'Graph'), this.phaseHost),
      h('div', { style: { display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap' } },
        this.edgeToggle));

    this.canvasWrap = h('div', { class: 'canvas-wrap' });
    this.legend = h('div', { class: 'graph-legend' });
    this.sizeLegend = h('div', { class: 'graph-legend size-legend',
      title: 'A node\'s area is the size of the declaration (statement and proof) in Lean tokens, drawn at the current zoom. '
        + 'The largest grow more slowly, so no node is wider than its place in the lane.' });
    this.searchInput = h('input', { class: 'search', type: 'search', id: 'decl-search', placeholder: 'Find a declaration', 'aria-label': 'Find a declaration', autocomplete: 'off' });
    this.searchList = h('ul', { role: 'listbox', hidden: true });
    this.stats = h('span', {});
    this.canvasWrap.append(
      h('div', { class: 'canvas-overlay' }, h('div', { class: 'legend-stack' }, this.legend, this.sizeLegend), h('div', { class: 'graph-search' }, this.searchInput, this.searchList)),
      h('div', { class: 'canvas-foot' }, h('span', {}, this.stats, h('span', { class: 'touch-hint' }, 'Two fingers to pan and zoom; tap a circle to open it')), h('div', { class: 'zoom' },
        h('button', { type: 'button', 'aria-label': 'Zoom in', onclick: () => this.gv.zoomBy(1.5) }, '+'),
        h('button', { type: 'button', 'aria-label': 'Zoom out', onclick: () => this.gv.zoomBy(1 / 1.5) }, '−'),
        h('button', { type: 'button', 'aria-label': 'Fit graph', title: 'Fit', onclick: () => this.gv.fit() }, '⤢'))));
    this.inspector = h('aside', { class: 'inspector', 'aria-label': 'Declaration details' });
    this.explorer = h('div', { class: 'explorer' }, this.canvasWrap, this.inspector);

    this.graphTitle = h('div', { class: 'full-title' });
    this.graphHost.append(h('div', { class: 'full-bar' }, this.graphTitle, this.closeButton('Close the graph')), bar, this.explorer);

    this.gv = new GraphView(this.canvasWrap, {
      onSelect: (i) => {
        if (i >= 0) this.openFull();
        if (i < 0 && this.selected < 0) this.listExpanded = false;
        this.selected = i; this.renderInspector();
        if (this.isSheet()) hideTip();
        this.revealSelected();
        this.syncLink();
      },
      onHover: (i, e) => this.hoverTip(i, e),
      onView: (k) => this.renderSizeLegend(k),
    });
    this.bindSearch();
    // Tapping the peeking sheet slides the lists up.
    this.inspector.addEventListener('click', () => {
      if (this.inspector.classList.contains('peek')) { this.listExpanded = true; this.renderInspector(); }
    });
  }

  openFull() {
    if (this.full || !this.narrow.matches) return;
    this.full = this.kind;
    document.documentElement.classList.add('full-open');
    history.pushState({ leanleanFull: true }, '');
    if (this.kind === 'trace') {
      this.traceHost.classList.add('full');
      this.tv?.renderChart();
      return;
    }
    if (this.kind !== 'graph') {
      this.diffHost.classList.add('full');
      this.setDrawer(true);
      return;
    }
    this.graphHost.classList.add('full');
    // The canvas had no size while hidden: measure it now, before a selection pans to a node.
    this.gv.resize();
    // Fit between the key at the top and the controls and peeking sheet at the bottom.
    const overlayBottom = this.graphHost.querySelector('.canvas-overlay').getBoundingClientRect().bottom;
    const canvasTop = this.gv.wrap.getBoundingClientRect().top;
    this.gv.pad = { top: Math.max(92, overlayBottom - canvasTop + 12), bottom: 172 };
    this.gv.fitToNodes = true;
    this.gv.fit(); this.gv.fitted = true;
    this.listExpanded = false;
    this.renderInspector();
  }

  closeFull(fromHistory = false) {
    if (!this.full) return;
    const was = this.full;
    this.full = null;
    document.documentElement.classList.remove('full-open');
    this.diffHost.classList.remove('full');
    if (was === 'trace') { this.traceHost.classList.remove('full'); this.tv?.renderChart(); }
    this.setDrawer(false);
    if (was === 'graph') {
      this.graphHost.classList.remove('full');
      this.gv.pad = { top: 52, bottom: 32 };
      this.gv.fitToNodes = false;
      hideTip();
      this.listExpanded = false;
      if (this.selected >= 0) this.gv.select(-1); else this.renderInspector();
    }
    if (!fromHistory && history.state?.leanleanFull) history.back();
  }

  // Select a declaration from a list, opening the graph first where it is full screen.
  pick(i) { this.openFull(); this.gv.select(i, { focus: true }); }

  // Below 1100px the inspector of a selected declaration is a bottom sheet over the graph (site.css).
  isSheet() { return getComputedStyle(this.inspector).position === 'fixed'; }

  // Keep the selected node in the part of the graph the sheet leaves uncovered.
  revealSelected() {
    const i = this.selected;
    if (i < 0 || !this.isSheet()) return;
    requestAnimationFrame(() => {
      if (this.selected !== i || !this.gv.w) return;
      const rect = this.canvasWrap.getBoundingClientRect();
      const sheetTop = this.inspector.getBoundingClientRect().top;
      // Below the floating search, above the sheet.
      const search = this.searchInput.getBoundingClientRect();
      const top = Math.max(rect.top, 0, search.bottom) - rect.top, bottom = Math.min(rect.bottom, sheetTop) - rect.top;
      const { k, x, y } = this.gv.view;
      const nx = x + this.gv.x[i] * k, ny = y + this.gv.y[i] * k, pad = 28;
      if (nx > pad && nx < this.gv.w - pad && ny > top + pad && ny < bottom - pad) return;
      this.gv.view = { k, x: this.gv.w / 2 - this.gv.x[i] * k, y: (top + bottom) / 2 - this.gv.y[i] * k };
      this.gv.request();
    });
  }

  get isPrep() { return this.model === PREP; }

  // Complete (for an agent), Before and After for the graph's source.
  renderPhase() {
    const options = this.isPrep ? [['before', 'Before'], ['after', 'After']] : [['all', 'Complete'], ['before', 'Before'], ['after', 'After']];
    // Each source opens on its fullest picture: Complete for an agent, Before for preprocessing.
    if (this.phasePrep !== this.isPrep) { this.gv.phase = this.isPrep ? 'before' : 'all'; this.phasePrep = this.isPrep; }
    this.phaseHost.replaceChildren(seg(options, this.gv.phase, (v) => { this.gv.setPhase(v); this.renderLegend(); }, 'Graph state'));
  }

  catText(category) { return this.isPrep && category === 'deleted' ? 'Removed by preprocessing' : CATEGORY_TEXT[category]; }

  // One baseline file's text: its chunk of source.bin, by an HTTP Range request (the index is source.json).
  async baseText(path) {
    this.baseTexts ||= new Map();
    if (!this.baseTexts.has(path)) {
      this.baseTexts.set(path, load(`data/repos/${this.id}/source.json.gz`).then((index) =>
        (index[path] ? loadChunk(`data/repos/${this.id}/source.bin`, index[path]) : undefined)));
    }
    return this.baseTexts.get(path);
  }

  // Source text before (raw for preprocessing, else the baseline) and after, per path.
  async sources() {
    const raw = this.isPrep ? await load(`data/repos/${this.id}/prep.source.json.gz`) : null;
    return { before: async (path) => raw?.[path] ?? this.baseText(path), after: async (path) => this.pair?.text[path] ?? this.baseText(path) };
  }

  destroy() {
    this.gv?.destroy(); this.tv?.destroy(); this.dialog?.remove(); this.picker.close(); this.modelPicker.close(); hideTip();
    document.documentElement.classList.remove('full-open');
    document.removeEventListener('keydown', this.onKey);
    window.removeEventListener('popstate', this.onPop);
    this.narrow.removeEventListener('change', this.onNarrow);
  }

  // The address of what is open: repository, agent, view, and the selected
  // declaration (Graph) or file (Diff).
  link() {
    const target = this.kind === 'graph' ? (this.selected >= 0 && this.gv ? this.gv.label(this.selected) : null) : this.filePath;
    return `#/${this.kind}/${this.id}/${this.model?.short || '-'}${target ? `/${encodeURIComponent(target)}` : ''}`;
  }

  // Keep the address in step without a history entry per click; never while a different page is routed,
  // and not while the page shows the run it opens on, so the bare address stays bare.
  syncLink() {
    if (!this.routed || !document.body.contains(this.root)) return;
    const link = this.link();
    if (!location.hash && link === this.home) return;
    if (location.hash !== link) history.replaceState(null, '', link);
  }

  // Open the declaration or file named in the address, if it exists here.
  openTarget(target) {
    if (this.kind === 'graph') {
      let found = -1;
      for (let i = 0; i < this.gv.N && found < 0; i++) if (this.gv.label(i) === target) found = i;
      if (found >= 0 && found !== this.selected) this.gv.select(found, { focus: true });
    }
  }

  // Called on every route into this repository: a new agent or a new view.
  async update(model, kind = this.kind, target = null) {
    this.routed = false;
    if (kind === 'repo' && target) this.filePath = target;
    const refresh = model === this.model && kind === this.kind && this.pair !== undefined;
    if (kind !== this.kind) { this.closeFull(); this.setKind(kind); }
    // A direct link to a lower effort must reveal the selected run.
    if (model && model !== PREP && !this.highestModels.has(model)) {
      this.highestOnly.checked = false;
      repoHighestOnly = false;
    }
    this.refreshTabs();
    this.modelPicker.refresh(model?.short);
    for (const b of this.tabs.querySelectorAll('button')) b.setAttribute('aria-pressed', String(model && b.dataset.short === model.short));
    for (const a of this.viewTabs.children) a.href = `#/${a.dataset.view}/${this.id}/${model?.short || '-'}`;
    if (model !== this.model || this.pair === undefined) {
      this.model = model;
      const key = model === PREP ? 'prep' : 'base';
      const hasPair = model === PREP || (model && explorable(this.bench, this.results, model));
      const [pair, graph] = await Promise.all([hasPair ? load(`data/repos/${this.id}/${model.short}.json.gz`) : null,
        this.graphs[key] || load(`data/repos/${this.id}/prep.graph.json.gz`)]);
      this.graphs[key] = graph;
      if (graph !== this.graph) { this.graph = graph; this.selected = -1; }
      this.pair = pair;
      this.diff = null;
      this.renderOutcome();
    }
    if (refresh) {
      for (const b of this.tabs.querySelectorAll('button')) {
        const m = this.bench.models.find((x) => x.short === b.dataset.short);
        if (m) b.style.setProperty('--model', modelColor(m));
      }
      this.renderOutcome();
    }
    if (this.kind === 'trace') {
      if (this.model && this.traces[this.model.short]) {
        const res = this.results[this.model.key];
        await this.tv.show(this.id, this.model.short, { repo: this.repo, result: res, color: modelColor(this.model) });
      } else {
        this.tv.key = null;
        this.traceBody.querySelector('.trace-summary')?.replaceChildren(h('div', { class: 'note' }, 'No agent run to show.'));
      }
    } else if (this.kind === 'graph') {
      if (this.gvPair !== this.pair) {
        const newGraph = this.gv.graph !== this.graph;
        this.gvPair = this.pair;
        if (newGraph) this.gv.selected = -1;
        this.renderPhase();
        this.gv.setData(this.graph, this.pair);
        if (newGraph) this.gv.fit();
        if (this.selected >= this.gv.N) this.selected = -1;
        const extra = this.isPrep ? ` · ${int(this.pair.outcome.removed)} removed` : this.pair?.added.length ? ` · ${int(this.pair.added.length)} added` : '';
        this.stats.textContent = `${int(this.graph.counts.nodes)} declarations · ${int(this.graph.counts.edges)} dependencies${extra} · area = Lean tokens`;
      } else if (refresh) this.gv.rebuild();
      this.renderLegend();
      this.gv.resize();
      this.renderInspector();
    } else if (this.filesPair !== this.pair || refresh || (target && this.fileShown !== target)) {
      this.filesPair = this.pair;
      await this.renderFiles();
    }
    if (target && this.kind === 'graph') this.openTarget(target);
    this.routed = true;
    this.syncLink();
    this.renderTitles();
    if (this.pendingFull === this.kind) { this.pendingFull = null; this.openFull(); }
  }

  // ── summary ──────────────────────────────────────────────────────────
  renderOutcome() {
    const m = this.model;
    const stat = (label, value, color) => h('div', { class: 'stat' }, h('span', {}, label), h('b', { style: color ? { color } : {} }, value));
    if (this.isPrep) {
      const o = this.pair.outcome;
      const changed = this.pair.files.filter((f) => f.state !== 'unchanged').length;
      this.outcome.replaceChildren(
        h('div', { class: 'stats' },
          stat('Before', `${tokens(o.raw_tokens)} tokens`), stat('After', `${tokens(o.baseline_tokens)} tokens`),
          stat('Removed', o.raw_tokens ? signedPct(-(1 - o.baseline_tokens / o.raw_tokens) * 100, 1) : '—', cssVar('--g-deleted')),
          stat('Declarations', `${int(o.declarations)} → ${int(o.declarations - o.removed)}`), stat('Files changed', `${int(changed)} of ${int(this.pair.files.length)}`)),
        h('p', { class: 'muted', style: { fontSize: '0.8rem', margin: 0, flex: '1', minWidth: 'min(100%, 22rem)' } },
          'Before the agents see a repository, LeanLean strips every declaration its protected results do not depend on. The result is the baseline every agent starts from and is scored against.'));
      return;
    }
    const res = m ? this.results[m.key] : null;
    if (!m || !res) {
      this.outcome.replaceChildren(h('p', { class: 'muted' }, 'No agent produced a verified submission for this repository.'));
      return;
    }
    if (res.status === 'pending') {
      this.outcome.replaceChildren(h('p', { class: 'muted' }, `${modelLabel(m)}'s run on this repository is not finished yet.`));
      return;
    }
    if (res.status !== 'passed') {
      this.outcome.replaceChildren(
        h('div', { class: 'stats' },
          stat('Before', `${tokens(this.repo.tokens)} tokens`), stat('Result', 'Failed verification', cssVar('--bad')),
          stat('Score', '0%'), stat('Cost', usd(res.cost)), stat('Duration', hours(res.seconds / 3600))),
        h('p', { class: 'muted', style: { fontSize: '0.8rem', margin: 0, flex: '1', minWidth: 'min(100%, 22rem)' } },
          `${modelLabel(m)}'s submission did not pass verification, so it scores 0. The trace shows how the run went.`));
      return;
    }
    this.outcome.replaceChildren(
      h('div', { class: 'stats' },
        stat('Before', `${tokens(this.repo.tokens)} tokens`), stat('After', `${tokens(res.post)} tokens`),
        stat('Score', pct(res.score), modelColor(m)), heartbeatStat(this.repo.heartbeats, res),
        stat('Cost', usd(res.cost)), stat('Duration', hours(res.seconds / 3600))),
      res.origin ? originStrip(res.origin, this.repo.tokens, modelLabel(m)) : null);
  }

  renderLegend() {
    const phase = this.gv.phase;
    const shown = { before: ['unchanged', 'modified', 'comments', 'deleted', 'dead'], after: ['unchanged', 'modified', 'comments', 'added'],
      all: ['unchanged', 'modified', 'comments', 'deleted', 'dead', 'added'] }[phase];
    const present = new Set();
    if (this.pair) for (let i = 0; i < this.gv.N; i++) present.add(this.gv.category(i));
    const rows = (this.pair ? shown.filter((k) => present.has(k)) : ['unchanged']).map((key) => h('span', { class: 'row' },
      h('i', { class: 'dot', style: { background: cssVar(CATEGORY_VAR[key]) } }), this.pair ? this.catText(key) : 'Baseline graph'));
    if (phase === 'all' && present.has('modified')) {
      // Area is Lean tokens, pale before and solid after, the larger behind the smaller.
      rows.push(h('span', { class: 'row', title: 'Area is size in Lean tokens: pale before, solid after' },
        h('i', { class: 'dot two' }), 'Shrank', h('i', { class: 'dot two grew' }), 'Grew'));
    }
    rows.push(h('span', { class: 'row' }, h('i', { class: 'dot ring' }), 'Protected result'));
    const shapes = new Set();
    for (const k of this.graph.nodes.kind) shapes.add(shapeOf(this.graph.kinds[k]));
    if (this.pair && phase !== 'before') for (const a of this.pair.added) shapes.add(shapeOf(a.kind));
    for (const [s, [cls, text]] of SHAPE_TEXT.entries()) {
      if (shapes.has(s)) rows.push(h('span', { class: 'row' }, h('i', { class: `dot shape ${cls}` }), text));
    }
    this.legend.replaceChildren(...rows);
  }

  // Reference circles at the size nodes of 10, 100 and 1,000 tokens have on screen now.
  renderSizeLegend(k) {
    const cap = this.gv.cap?.[0];
    if (!cap || Math.abs(k - (this.sizeK ?? 0)) < 0.005 * k) return;
    this.sizeK = k;
    this.sizeLegend.replaceChildren(h('span', {}, 'Size'), ...sizeMarks(cap, k), h('span', {}, 'tokens'));
  }

  // ── node info helpers ────────────────────────────────────────────────
  info(i) {
    const n = this.graph.nodes;
    if (i >= this.gv.n) {
      const a = this.pair.added[i - this.gv.n];
      return { name: a.name, kind: a.kind, status: 'added', category: this.gv.category(i), before: 0, after: a.tokens, file: null, afterRange: [a.file, a.start, a.end], added: a };
    }
    const status = this.pair ? STATUS[this.pair.status[i]] : 'unchanged';
    return {
      name: n.name[i], kind: this.graph.kinds[n.kind[i]], status, category: this.gv.category(i), protected: this.gv.protected.has(i),
      before: n.tokens[i], after: this.pair ? this.pair.after[i] : n.tokens[i],
      file: this.graph.files[n.file[i]], start: n.start[i], end: n.end[i],
      afterRange: this.pair ? this.pair.range[i] : null, ledger: this.pair?.ledger[i],
      aliases: this.graph.aliases[i] || [],
    };
  }

  hoverTip(i, e) {
    // A finger has no hover; on a tap the sheet shows the same facts.
    if (i < 0 || !e || e.pointerType === 'touch') { hideTip(); return; }
    const d = this.info(i);
    const change = d.before ? (1 - d.after / d.before) * 100 : null;
    showTip(e, `<b style="font-family:var(--font-mono);font-weight:500">${esc(d.name)}</b>
      <div class="row"><span>${esc(KIND_LABEL[d.kind] || d.kind)}</span><span><i class="swatch" style="background:${cssVar(CATEGORY_VAR[d.category])}"></i>${esc(this.catText(d.category))}</span></div>
      <div class="row"><span>Tokens</span><span>${int(d.before)} → ${int(d.after)}${change != null && d.status !== 'unchanged' && d.status !== 'comments' ? ` (${signedPct(-change, 0)})` : ''}</span></div>
      ${this.savedRows(d)}
      ${d.protected ? '<div>Protected result</div>' : ''}`);
  }

  // Tokens saved (or spent) per compression category, as the paper's ledger books them.
  savedRows(d) {
    const line = (color, label, v) => `<div class="row"><span><i class="swatch" style="background:${cssVar(color)}"></i>${label}</span><span>${v > 0 ? '−' : v < 0 ? '+' : ''}${int(Math.abs(v))}</span></div>`;
    if (d.ledger) {
      // [delta, syntax, automation, rest]: rest is the proof rewrite.
      const [, syntax, automation, rest] = d.ledger;
      return [['--c-syntax', 'Syntax optimization', syntax], ['--c-auto', 'Automation rewrite', automation], ['--c-rewrite', 'Proof rewrite', rest]]
        .filter(([, , v]) => v).map(([c, l, v]) => line(c, l, v)).join('');
    }
    if (d.added) {
      if (d.added.syntax) return line('--c-syntax', 'Syntax definition', -d.after);
      return line('--c-added', 'Added', -d.after) + (d.added.uses ? line('--c-syntax', 'Syntax used inside', d.added.uses) : '');
    }
    if (d.status === 'dead') return line('--c-dead', 'Dead code', d.before);
    if (d.status === 'deleted' && !this.isPrep) return line('--c-deleted', 'Deleted', d.before);
    return '';
  }

  // The grip of a bottom sheet, with its close button.
  sheetBar(label, onClose) {
    return h('div', { class: 'sheet-bar' }, h('span', { class: 'grip', 'aria-hidden': 'true' }),
      onClose ? h('button', { type: 'button', class: 'sheet-close', 'aria-label': label, onclick: (e) => { e.stopPropagation(); onClose(); } }, '×') : null);
  }

  renderInspector() {
    const i = this.selected;
    const selected = i >= 0 && i < this.gv.N;
    // Full screen, the declaration lists sit in a sheet that peeks above the bottom edge until tapped.
    const lists = !selected && this.full === 'graph';
    const peek = lists && !this.listExpanded;
    this.inspector.classList.toggle('open', selected || lists);
    this.inspector.classList.toggle('peek', peek);
    this.graphHost.classList.toggle('sheet-open', selected || (lists && !peek));
    this.graphHost.classList.toggle('sheet-peek', peek);
    if (!selected) {
      const changedMost = this.pair ? this.topChanges() : [];
      this.inspector.replaceChildren(
        lists ? this.sheetBar('Fold the list away', peek ? null : () => { this.listExpanded = false; this.renderInspector(); }) : '',
        h('div', { class: 'inspector-empty' },
        h('h3', {}, 'Select a declaration'),
        h('p', {}, `Each circle is one declaration from the ${this.isPrep ? 'raw' : 'baseline'} source, sized by its Lean tokens. `,
          this.narrow.matches ? 'Tap one to read its code before and after, and to walk its dependencies. Pinch to zoom, drag to pan.'
            : 'Click one to read its code before and after, and to walk its dependencies. Scroll to zoom, drag to pan.'),
        changedMost.length ? h('div', { class: 'dep-list' }, h('h4', {}, h('span', {}, this.isPrep ? 'Largest removals' : 'Largest reductions')), this.depList(changedMost)) : null,
        this.gv.protected.size ? h('div', { class: 'dep-list' }, h('h4', {}, h('span', {}, 'Protected results'), h('span', {}, this.gv.protected.size)), this.depList([...this.gv.protected])) : null));
      this.inspector.scrollTop = 0;
      return;
    }
    const d = this.info(i);
    const { deps, users } = this.gv.neighbours(i);
    const statusText = this.catText(d.category);
    const change = d.before ? (1 - d.after / d.before) * 100 : null;
    const sections = [this.sheetBar('Close the declaration', () => this.gv.select(-1))];
    sections.push(h('section', {},
      h('div', { class: 'decl-name' }, breakable(d.name)),
      h('div', { class: 'decl-meta' },
        h('span', { class: 'chip' }, KIND_LABEL[d.kind] || d.kind),
        h('span', { class: 'chip' }, h('i', { class: 'swatch', style: { background: cssVar(CATEGORY_VAR[d.category]) } }), statusText),
        d.protected ? h('span', { class: 'chip', style: { border: '1px solid var(--ink)' } }, 'Protected result') : null,
        shareButton()),
      d.file ? h('div', { class: 'decl-where' }, breakable(`${d.file}:${d.start}–${d.end}`),
        d.afterRange && d.status !== 'unchanged' ? [' → ', breakable(`${d.afterRange[0]}:${d.afterRange[1]}–${d.afterRange[2]}`)] : '') :
        h('div', { class: 'decl-where' }, breakable(`${d.afterRange[0]}:${d.afterRange[1]}–${d.afterRange[2]}`)),
      h('div', { class: 'tok-compare' },
        h('div', {}, h('span', {}, 'Before'), h('b', {}, int(d.before))),
        h('div', {}, h('span', {}, 'After'), h('b', {}, int(d.after))),
        h('div', {}, h('span', {}, 'Change'), h('b', {}, d.status === 'added' ? `+${int(d.after)}` : change == null ? '—' : signedPct(-change, 0)))),
      d.aliases?.length ? h('div', { class: 'muted', style: { fontSize: '0.75rem' } }, `Includes ${d.aliases.length} generated declaration${d.aliases.length === 1 ? '' : 's'} (${d.aliases.slice(0, 3).map((a) => a.split('.').pop()).join(', ')}${d.aliases.length > 3 ? ', …' : ''})`) : null));

    const codeHost = h('div', { class: 'code-host' });
    const statLine = h('span', { class: 'code-stat' });
    sections.push(h('section', {},
      h('div', { class: 'code-tabs' }, h('span', { class: 'eyebrow' }, 'Code'),
        h('button', { type: 'button', class: 'btn icon-btn', title: 'Open the code full size', onclick: () => this.openCode(d) }, expandIcon(), 'Expand')),
      statLine, codeHost));
    this.renderCode(codeHost, statLine, d);

    sections.push(h('section', {}, h('div', { class: 'dep-lists' },
      h('div', { class: 'dep-list' }, h('h4', {}, h('span', { style: { color: 'var(--accent)' } }, 'Uses'), h('span', {}, deps.length)), this.depList(deps)),
      h('div', { class: 'dep-list' }, h('h4', {}, h('span', { style: { color: 'var(--g-users)' } }, 'Used by'), h('span', {}, users.length)), this.depList(users)))));
    this.inspector.replaceChildren(...sections);
    this.inspector.scrollTop = 0;
  }

  topChanges() {
    const n = this.gv.n, out = [];
    for (let i = 0; i < n; i++) {
      const saved = this.graph.nodes.tokens[i] - this.pair.after[i];
      if (saved > 0) out.push([saved, i]);
    }
    return out.sort((a, b) => b[0] - a[0]).slice(0, 25).map((x) => x[1]);
  }

  depList(list) {
    const sorted = [...list].sort((a, b) => this.info(b).before - this.info(a).before);
    const shown = sorted.slice(0, 300);
    const ul = h('ul', {}, shown.map((j) => {
      const d = this.info(j);
      return h('li', {}, h('button', { type: 'button', onclick: () => this.pick(j) },
        h('i', { class: 'dot', style: { background: this.gv.colorOf(j) } }),
        h('span', { class: 'n', title: d.name }, d.name),
        h('span', { class: 'num muted' }, d.status === 'unchanged' || d.status === 'comments' || d.status === 'added' ? int(d.after) : `${int(d.before)}→${int(d.after)}`)));
    }));
    if (sorted.length > shown.length) ul.append(h('li', { class: 'muted', style: { fontSize: '0.72rem', padding: '3px 6px' } }, `and ${sorted.length - shown.length} more`));
    return ul;
  }

  hasBoth(d) { return d.status !== 'added' && d.status !== 'deleted' && d.status !== 'dead' && !!this.pair; }

  async snippet(d) {
    const source = await this.sources();
    const slice = (text, a, b) => (text ?? '').split('\n').slice(a - 1, b);
    const [before, after] = await Promise.all([d.file ? source.before(d.file) : null, d.afterRange ? source.after(d.afterRange[0]) : null]);
    return {
      before: d.file ? slice(before, d.start, d.end) : [],
      after: d.afterRange ? slice(after, d.afterRange[1], d.afterRange[2]) : [],
    };
  }

  // view: 'diff' (unified) | 'split'
  async codeNode(d, view, { wrap = true, context = 3 } = {}) {
    const { before, after } = await this.snippet(d);
    const afterStart = d.afterRange?.[1] ?? 1;
    // A declaration that only exists on one side reads as wholly added or removed;
    // an added one is green but for the lines the ledger books as syntax.
    if (d.status === 'added') return diffBlock([], after, 1, afterStart, { wrap, kindOf: this.lineKinds(d.afterRange?.[0]) });
    if (d.status === 'deleted' || d.status === 'dead') return diffBlock(before, [], d.start, 1, { wrap, removedKindOf: d.status === 'dead' ? () => 'd' : undefined });
    if (!this.hasBoth(d)) return codeBlock(before, d.start, { wrap });
    // A kept declaration's added lines are proof rewrite unless the ledger booked them as automation or syntax.
    const lines = this.lineKinds(d.afterRange?.[0]);
    const kindOf = (n) => lines(n) || 'r';
    const spansOf = this.stepSpans(d.afterRange?.[0]);
    if (view === 'split') return stepTips(splitBlock(before, after, d.start, afterStart, { context, kindOf, spansOf }));
    return stepTips(diffBlock(before, after, d.start, afterStart, { context, wrap, kindOf, spansOf }));
  }

  // After line number -> what the change on it was (a, s, r), for the lines of kept declarations that changed.
  lineKinds(path) {
    const runs = this.pair?.line_kinds?.[path];
    if (!runs) return () => undefined;
    this.kindCache ||= new Map();
    if (!this.kindCache.has(runs)) {
      const map = new Map();
      for (const [a, b, k] of runs) for (let n = a; n <= b; n++) map.set(n, k);
      this.kindCache.set(runs, (n) => map.get(n));
    }
    return this.kindCache.get(runs);
  }

  // After line number -> the changed proof steps on it, [from, to, kind, note] (see markSteps).
  stepSpans(path) {
    const lines = this.pair?.step_spans?.[path];
    return (n) => lines?.[n];
  }

  // Before line number -> 'd' for the lines of declarations deleted as dead code.
  deadLines(path) {
    if (!this.pair || this.isPrep) return () => undefined;
    if (this.deadPair !== this.pair) {
      this.deadPair = this.pair;
      this.dead = new Map();
      const n = this.graph.nodes;
      this.pair.status.forEach((status, i) => {
        if (STATUS[status] !== 'dead' || n.file[i] < 0) return;
        const file = this.graph.files[n.file[i]];
        if (!this.dead.has(file)) this.dead.set(file, new Set());
        for (let l = n.start[i]; l <= n.end[i]; l++) this.dead.get(file).add(l);
      });
    }
    const lines = this.dead.get(path);
    return (l) => (lines?.has(l) ? 'd' : undefined);
  }

  async renderCode(host, stat, d) {
    host.replaceChildren(h('div', { class: 'muted', style: { fontSize: '0.8rem' } }, 'Loading source…'));
    const block = await this.codeNode(d, 'diff');
    host.replaceChildren(block);
    stat.replaceChildren(...codeStat(d, block));
  }

  // Full-size code for one declaration: side by side or unified, with the whole declaration unfolded on request.
  openCode(d) {
    if (!this.dialog) {
      this.dialog = h('dialog', { class: 'code-dialog', 'aria-label': 'Declaration code' });
      this.dialog.addEventListener('click', (e) => { if (e.target === this.dialog) this.dialog.close(); });
      document.body.append(this.dialog);
    }
    const hasBoth = this.hasBoth(d);
    // A phone has no room for two columns, and its code always wraps: only the leaderboard scrolls sideways.
    const phone = matchMedia('(max-width: 759.98px)').matches;
    if (!this.dialogView || (phone && this.dialogView === 'split')) this.dialogView = window.innerWidth > 1000 ? 'split' : 'diff';
    if (this.dialogWrap === undefined || phone) this.dialogWrap = true;
    // Two diffs, side by side or unified; a phone shows only the unified one.
    const views = [['split', 'Side by side'], ['diff', 'Unified']];
    const body = h('div', { class: 'code-dialog-body' });
    const stat = h('span', { class: 'code-stat' });
    const draw = async () => {
      const node = await this.codeNode(d, hasBoth ? this.dialogView : 'before', { wrap: this.dialogWrap, context: 6 });
      body.replaceChildren(node);
      body.classList.toggle('nowrap', !this.dialogWrap);
      stat.replaceChildren(...codeStat(d, node));
    };
    const change = d.before ? (1 - d.after / d.before) * 100 : null;
    this.dialog.replaceChildren(
      h('header', {},
        h('div', { class: 'title' },
          h('div', { class: 'decl-name' }, breakable(d.name)),
          h('div', { class: 'decl-meta' },
            h('span', { class: 'chip' }, KIND_LABEL[d.kind] || d.kind),
            h('span', { class: 'chip' }, h('i', { class: 'swatch', style: { background: cssVar(CATEGORY_VAR[d.category]) } }), this.catText(d.category)),
            h('span', { class: 'decl-where' }, d.file ? `${d.file}:${d.start}–${d.end}` : `${d.afterRange[0]}:${d.afterRange[1]}–${d.afterRange[2]}`),
            h('span', { class: 'num muted', style: { fontSize: '0.75rem' } }, `${int(d.before)} → ${int(d.after)} tokens${change != null && d.status !== 'unchanged' && d.status !== 'comments' && d.status !== 'added' ? ` (${signedPct(-change, 0)})` : ''}`))),
        h('button', { type: 'button', class: 'close', 'aria-label': 'Close', onclick: () => this.dialog.close() }, '×')),
      h('div', { class: 'toolbar' },
        hasBoth && !phone ? seg(views, this.dialogView, (v) => { this.dialogView = v; draw(); }, 'Code view') : h('span'),
        stat,
        phone ? null : h('label', { class: 'chip', style: { cursor: 'pointer' } },
          h('input', { type: 'checkbox', checked: this.dialogWrap, onchange: (e) => { this.dialogWrap = e.target.checked; draw(); } }), 'Wrap lines')),
      body);
    draw();
    if (!this.dialog.open) this.dialog.showModal();
  }

  bindSearch() {
    const input = this.searchInput, list = this.searchList;
    let active = 0, matches = [];
    const run = () => {
      const q = input.value.trim().toLowerCase();
      if (q.length < 2) { list.hidden = true; return; }
      matches = [];
      const n = this.gv.N;
      for (let i = 0; i < n && matches.length < 40; i++) if (this.gv.label(i).toLowerCase().includes(q)) matches.push(i);
      active = 0;
      list.replaceChildren(...matches.map((i, k) => h('li', { role: 'option', 'aria-selected': String(k === 0), onmousedown: (e) => { e.preventDefault(); pick(i); } }, this.gv.label(i))));
      list.hidden = !matches.length;
    };
    const pick = (i) => { list.hidden = true; input.blur(); this.gv.select(i, { focus: true }); };
    // Full screen the search is an icon beside the key; focused, it opens across the top.
    const overlay = input.closest('.canvas-overlay');
    input.addEventListener('focus', () => overlay.classList.add('searching'));
    input.addEventListener('blur', () => setTimeout(() => { if (document.activeElement !== input) overlay.classList.remove('searching'); }, 120));
    input.addEventListener('input', run);
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') { e.stopPropagation(); list.hidden = true; input.blur(); return; }
      if (list.hidden) return;
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault();
        active = (active + (e.key === 'ArrowDown' ? 1 : matches.length - 1)) % matches.length;
        [...list.children].forEach((li, k) => li.setAttribute('aria-selected', String(k === active)));
        list.children[active]?.scrollIntoView({ block: 'nearest' });
      } else if (e.key === 'Enter' && matches[active] != null) pick(matches[active]);
    });
    input.addEventListener('blur', () => setTimeout(() => { list.hidden = true; }, 100));
  }

  // ── files and diff ───────────────────────────────────────────────────
  async renderFiles() {
    if (!this.pair) { this.files.replaceChildren(h('div', { class: 'note' }, 'No verified submission to compare.')); return; }
    const files = this.pair.files;
    let filter = 'changed';
    this.collapsed = new Set();
    const ul = h('ul', { class: 'tree', role: 'tree', 'aria-label': 'Files' });
    const pane = h('div', { class: 'diff-pane' });
    const changedCount = files.filter((f) => f.state !== 'unchanged').length;
    const repoCount = files.filter((f) => !f.harness).length;
    const scored = files.filter((f) => f.scored);
    const before = scored.reduce((x, f) => x + f.before, 0), after = scored.reduce((x, f) => x + f.after, 0);
    const saved = before - after;
    const tdel = scored.reduce((x, f) => x + (f.tdel || 0), 0), tadd = scored.reduce((x, f) => x + (f.tadd || 0), 0);

    const drawList = () => {
      const tree = fileTree(files.filter((f) => filter === 'all' || f.state !== 'unchanged'));
      const rows = [];
      const walk = (node, depth) => {
        for (const d of node.dirs) {
          const open = !this.collapsed.has(d.path);
          rows.push(h('li', { role: 'treeitem', 'aria-expanded': String(open) },
            h('button', { type: 'button', class: 'dir', style: { '--depth': depth }, title: d.path,
              onclick: () => { open ? this.collapsed.add(d.path) : this.collapsed.delete(d.path); drawList(); } },
            h('i', { class: 'chev', 'aria-hidden': 'true' }), icon(open ? 'dir-open' : 'dir'),
            h('span', { class: 'name' }, d.name), treeDiff(d))));
          if (open) walk(d, depth + 1);
        }
        for (const f of node.files) {
          rows.push(h('li', { role: 'treeitem' }, h('button', { type: 'button', class: `file ${f.state}${f.scored ? '' : ' unscored'}`, style: { '--depth': depth },
            'data-path': f.path, 'aria-current': String(f.path === this.filePath), title: fileTitle(f),
            onclick: () => { this.setDrawer(false); this.openFile(f, pane, ul); } },
          h('i', { class: 'chev', 'aria-hidden': 'true', style: { visibility: 'hidden' } }), icon('file'),
          h('span', { class: 'name' }, f.path.slice(f.path.lastIndexOf('/') + 1)),
          f.scored ? treeDiff(f) : h('span', { class: 'delta muted' }, '—'),
          h('i', { class: `state ${f.state}`, 'aria-label': f.state }))));
        }
      };
      walk(tree, 0);
      ul.replaceChildren(...rows);
    };

    const list = h('div', { class: 'file-list' },
      h('div', { class: 'head' },
        h('div', { class: 'saved' },
          h('b', {}, tokens(saved)), h('span', {}, `Lean tokens ${this.isPrep ? 'removed by preprocessing' : 'compressed'}${before ? ` · ${signedPct(-saved / before * 100)}` : ''}`)),
        h('div', { class: 'sub' }, h('span', { class: 'num' }, `${tokens(before)} → ${tokens(after)} · `, tokenDiff(tdel, tadd, tokens), ` · ${changedCount} of ${repoCount} files`),
          seg([['changed', 'Changed'], ['all', 'All']], filter, (v) => { filter = v; drawList(); }, 'Files shown'))),
      ul);
    this.files.replaceChildren(list, pane);
    drawList();
    const first = files.filter((f) => f.state !== 'unchanged').sort((x, y) => (y.before - y.after) - (x.before - x.after))[0];
    const keep = files.find((f) => f.path === this.filePath);
    if (keep || first) await this.openFile(keep || first, pane, ul);
    else pane.replaceChildren(h('div', { class: 'note' }, 'No file changed.'));
  }

  async openFile(f, pane, ul) {
    this.filePath = this.fileShown = f.path;
    this.syncLink();
    for (const b of ul.querySelectorAll('button[data-path]')) b.setAttribute('aria-current', String(b.dataset.path === f.path));
    const head = h('div', { class: 'head' },
      h('div', { style: { display: 'grid', gap: '2px', minWidth: 0 } }, h('span', { class: 'decl-name', style: { fontSize: '0.8rem' } }, breakable(f.path)),
        h('span', { class: 'muted num', style: { fontSize: '0.75rem' } }, f.scored ? [`${int(f.before)} → ${int(f.after)} Lean tokens (${f.before ? signedPct(-(1 - f.after / f.before) * 100) : 'new'}) · `, tokenDiff(f.tdel, f.tadd, int)]
          : f.harness ? 'Written by the benchmark harness · not part of the score' : 'Not part of the score')),
      h('span', { class: `chip ${f.state === 'deleted' ? 'bad' : f.state === 'added' ? 'warn' : ''}` }, f.harness ? 'harness' : f.state), shareButton());
    // A modified file reads as its diff, or whole as it was before or after; the choice carries over to the next file.
    const fileView = f.state === 'modified' ? (this.fileView ||= 'diff') : 'diff';
    const views = f.state === 'modified'
      ? h('div', { class: 'file-views' }, seg([['diff', 'Diff'], ['before', 'Before'], ['after', 'After']], fileView,
        (v) => { this.fileView = v; this.openFile(f, pane, ul); }, 'File view'))
      : null;
    pane.replaceChildren(head, ...(views ? [views] : []), h('div', { class: 'note' }, 'Loading diff…'));
    let result;
    if (f.state === 'modified' && fileView !== 'diff') {
      const source = await this.sources();
      const lines = ((await source[fileView](f.path)) || '').split('\n');
      if (lines.length && lines[lines.length - 1] === '') lines.pop();
      // A whole file is plain rows, cheap enough to show in full up to 20,000 lines.
      result = { pre: codeBlock(lines.slice(0, 20000)), truncated: lines.length > 20000, total: lines.length, limit: 20000, plain: true };
    } else if (f.state === 'modified') {
      this.diff = this.diff || await load(`data/repos/${this.id}/${this.model.short}.diff.json.gz`);
      result = hunksBlock(this.diff[f.path] || [], 4000, this.lineKinds(f.path), this.deadLines(f.path), this.stepSpans(f.path));
      stepTips(result.pre);
    } else if (f.state === 'deleted') {
      const source = await this.sources();
      result = wholeFileBlock((await source.before(f.path)) || '', 'del', 4000, this.deadLines(f.path));
    } else if (f.state === 'added') {
      result = wholeFileBlock(this.pair.text[f.path] || '', 'add', 4000, this.lineKinds(f.path));
    } else {
      const lines = ((f.harness ? this.pair.text[f.path] : await this.baseText(f.path)) || '').split('\n');
      result = { pre: codeBlock(lines.slice(0, 4000)), truncated: lines.length > 4000, total: lines.length };
    }
    if (this.filePath !== f.path) return;
    result.pre.classList.add('wrap');
    const key = result.plain ? null : diffKey(result.pre);
    pane.replaceChildren(head, ...(views ? [views] : []), ...(key ? [h('div', { class: 'file-key' }, key)] : []), result.pre, ...(result.truncated ? [h('div', { class: 'more muted' }, `Showing the first ${int(result.limit || 4000)} of ${int(result.total)} lines.`)] : []));
    pane.scrollTop = 0;
  }
}

// Heartbeats Lean spends elaborating the repository's own declarations, before → after (Lean's
// user unit, the one maxHeartbeats counts; imports excluded), with the change. Not every passing
// submission is measured yet.
function heartbeatStat(before, res) {
  const after = res.heartbeats_after;
  if (before == null || after == null) return null;
  const title = `${int(before)} → ${int(after)} heartbeats to elaborate the repository's own declarations (imports excluded)`;
  return h('div', { class: 'stat', title }, h('span', {}, 'Heartbeats'),
    h('b', {}, `${tokens(before)} → ${tokens(after)}`,
      h('small', {}, signedPct(-res.heartbeats, 1))));
}

function row(label, value) { return h('div', { class: 'row' }, h('span', {}, label), h('span', {}, value)); }
function sw(color, label) { return h('span', {}, h('i', { class: 'swatch', style: { background: cssVar(color) } }), label); }

// Nest paths into folders, GitHub style: folders first, then files, both by name;
// a folder whose only child is another folder is shown as one row (a/b/c).
function fileTree(files) {
  const root = { name: '', path: '', dirs: new Map(), files: [] };
  for (const f of files) {
    const parts = f.path.split('/');
    let node = root;
    for (let i = 0; i < parts.length - 1; i++) {
      if (!node.dirs.has(parts[i])) node.dirs.set(parts[i], { name: parts[i], path: parts.slice(0, i + 1).join('/'), dirs: new Map(), files: [] });
      node = node.dirs.get(parts[i]);
    }
    node.files.push(f);
  }
  const byName = (x, y) => x.name.localeCompare(y.name, undefined, { numeric: true });
  const finish = (node) => {
    let dirs = [...node.dirs.values()].map(finish);
    node.files.sort((x, y) => byName({ name: x.path }, { name: y.path }));
    node.tdel = 0; node.tadd = 0;
    for (const f of node.files) if (f.scored) { node.tdel += f.tdel || 0; node.tadd += f.tadd || 0; }
    for (const d of dirs) { node.tdel += d.tdel; node.tadd += d.tadd; }
    node.dirs = dirs.sort(byName);
    while (node.path && node.files.length === 0 && node.dirs.length === 1) {
      const only = node.dirs[0];
      Object.assign(node, { name: `${node.name}/${only.name}`, path: only.path, dirs: only.dirs, files: only.files });
    }
    return node;
  };
  return finish(root);
}

// A tree row's Lean tokens deleted and added: −542 +60; zero sides are left out.
function treeDiff(x) {
  if (!x.tdel && !x.tadd) return h('span', { class: 'delta muted' }, '0');
  return h('span', { class: 'delta' },
    x.tdel ? h('b', { class: 'del' }, `−${tokens(x.tdel)}`) : null, x.tdel && x.tadd ? ' ' : null,
    x.tadd ? h('b', { class: 'add' }, `+${tokens(x.tadd)}`) : null);
}

// Lean tokens deleted and added, GitHub style: −123 +45.
function tokenDiff(del, add, fmt) {
  if (del == null) return '';
  return h('span', { class: 'tok-diff', title: `${int(del)} Lean tokens deleted, ${int(add)} added` },
    h('b', { class: 'del' }, `−${fmt(del)}`), ' ', h('b', { class: 'add' }, `+${fmt(add)}`), ' tokens');
}

function fileTitle(f) {
  const tok = f.scored ? `${int(f.before)} → ${int(f.after)} Lean tokens${f.tdel != null ? ` · −${int(f.tdel)} +${int(f.tadd)} tokens` : ''}` : f.harness ? 'written by the harness, not scored' : 'not scored';
  return `${f.path}\n${f.state} · ${tok}${f.add || f.del ? ` · +${int(f.add)} −${int(f.del)} lines` : ''}`;
}

const ICONS = {
  dir: 'M1.75 2.5h3.9l1.5 1.5h7.1c.41 0 .75.34.75.75v8.5c0 .41-.34.75-.75.75H1.75A.75.75 0 0 1 1 13.25V3.25c0-.41.34-.75.75-.75Z',
  'dir-open': 'M1.75 2.5h3.9l1.5 1.5h7.1c.41 0 .75.34.75.75V6H3.4L1 12.5V3.25c0-.41.34-.75.75-.75ZM3.9 7H15.5l-2.4 6.5H1.4Z',
  file: 'M3.75 1.5h5.6L13 5.15v9.1c0 .41-.34.75-.75.75h-8.5a.75.75 0 0 1-.75-.75V2.25c0-.41.34-.75.75-.75ZM9 1.75V5.5h3.75',
};
function icon(name) {
  const node = h('span', { class: `ico ${name}`, 'aria-hidden': 'true' });
  node.innerHTML = `<svg viewBox="0 0 16 16" width="14" height="14"><path d="${ICONS[name]}"/></svg>`;
  return node;
}

function codeStat(d, block) {
  const out = [];
  const oneSided = d.status === 'added' || d.status === 'deleted' || d.status === 'dead';
  if (block.dataset.adds !== undefined && !oneSided) {
    out.push(h('span', { class: 'a' }, `+${block.dataset.adds}`), h('span', { class: 'd' }, `−${block.dataset.dels}`));
  }
  const n = d.status === 'added' ? d.afterRange[2] - d.afterRange[1] + 1 : d.end - d.start + 1;
  out.push(h('span', {}, `${int(n)} line${n === 1 ? '' : 's'}${d.status === 'added' ? ' added' : d.status === 'deleted' || d.status === 'dead' ? ' removed' : ' before'}`));
  const key = diffKey(block);
  if (key) out.push(key);
  return out;
}

// A file path or dotted name that may break only after a / or a dot.
function breakable(text) {
  const out = [];
  for (const part of String(text).split(/([/.])/)) {
    if (!part) continue;
    out.push(part);
    if (part === '/' || part === '.') out.push(h('wbr'));
  }
  return out;
}

// An outside page for the repository, as an icon that opens in a new tab.
function iconLink(href, icon, label) {
  return h('a', { class: 'icon-link', href, target: '_blank', rel: 'noopener', title: label, 'aria-label': label },
    h('img', { src: `${ASSETS}logos/${icon}.svg`, alt: '', width: 18, height: 18 }));
}

// Share the address of what is open: the share sheet on touch screens, else copy it.
function shareButton() {
  const text = h('span', {}, 'Copy link');
  const button = h('button', { type: 'button', class: 'btn icon-btn share', title: 'Copy a link to exactly this view' }, linkIcon(), text);
  button.addEventListener('click', async () => {
    const url = location.href;
    try {
      if (navigator.share && matchMedia('(pointer: coarse)').matches) { await navigator.share({ url, title: document.title }); return; }
      await navigator.clipboard.writeText(url);
      text.textContent = 'Link copied';
    } catch {
      text.textContent = 'Copy from the address bar';
    }
    setTimeout(() => { text.textContent = 'Copy link'; }, 1800);
  });
  return button;
}

function linkIcon() {
  const node = h('span', { class: 'ico', 'aria-hidden': 'true' });
  node.innerHTML = '<svg viewBox="0 0 16 16" width="12" height="12"><path d="M6.5 9.5a3 3 0 0 0 4.2 0l2.3-2.3a3 3 0 0 0-4.2-4.2l-.9.9M9.5 6.5a3 3 0 0 0-4.2 0L3 8.8A3 3 0 0 0 7.2 13l.9-.9" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>';
  return node;
}

// Node size, by the graph's own rule (graph.js radius) at a scale where the largest node is 8px across the radius.
// One circle per reference size, as large as a node of that many tokens is drawn at zoom k
// (graph.js shown); those too large to fit the key drop out, the smallest always stays.
function sizeMarks(cap, k) {
  const marks = [10, 100, 1000].map((t) => [t, Math.max(radius(t, cap) * k, Math.min(1.3, cap * k))])
    .filter(([, r], i) => i === 0 || r <= 24);
  return marks.map(([t, r]) => {
    const d = Math.ceil(Math.max(2 * r, 1)) + 1;
    const mark = h('span', { class: 'size-mark', 'aria-hidden': 'true' });
    mark.innerHTML = `<svg width="${d}" height="${d}" viewBox="0 0 ${d} ${d}"><circle cx="${d / 2}" cy="${d / 2}" r="${Math.max(r, 0.5)}" fill="var(--ink-2)"/></svg>`;
    return h('span', { class: 'row' }, mark, int(t));
  });
}

// What the colours of a diff's added lines mean, for the kinds it shows (the compression-origin colours).
const KIND_KEY = [['.add.k-r', '--c-rewrite', 'Proof rewrite'], ['.add.k-a', '--c-auto', 'Automation'], ['.add.k-s', '--c-syntax', 'Syntax'],
  ['.add:not([class*="k-"])', '--c-added', 'Added'], ['.del:not(.k-d)', '--c-deleted', 'Removed'], ['.del.k-d', '--c-dead', 'Dead code']];
function diffKey(block) {
  const shown = KIND_KEY.filter(([selector]) => block.querySelector(selector));
  // Plain green and red need no key.
  if (!shown.some(([selector]) => selector.includes('.k-'))) return null;
  return h('span', { class: 'diff-key' }, shown.map(([, color, label]) => sw(color, label)));
}

function svgIcon(name) {
  const paths = {
    graph: '<circle cx="10" cy="4.5" r="2"/><circle cx="4.5" cy="15" r="2"/><circle cx="15.5" cy="15" r="2"/><path d="M9 6.3 5.5 13.2M11 6.3l3.5 6.9M6.5 15h7"/>',
    diff: '<path d="M6 3.5v7M2.5 7h7M2.5 16.5h7M13.5 3v14"/><path d="M11 14.5l2.5 2.5 2.5-2.5"/>',
    files: '<path d="M3 5h14M3 10h14M3 15h9"/>',
    trace: '<circle cx="4.5" cy="4.5" r="1.6"/><circle cx="4.5" cy="10" r="1.6"/><circle cx="4.5" cy="15.5" r="1.6"/><path d="M8.5 4.5h8M8.5 10h6M8.5 15.5h7.5"/>',
  };
  const node = h('span', { class: 'ico', 'aria-hidden': 'true' });
  node.innerHTML = `<svg viewBox="0 0 20 20" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">${paths[name]}</svg>`;
  return node;
}

function expandIcon() {
  const node = h('span', { class: 'ico', 'aria-hidden': 'true' });
  node.innerHTML = '<svg viewBox="0 0 16 16" width="12" height="12"><path d="M9.5 2.5h4v4M13.5 2.5 9 7M6.5 13.5h-4v-4M2.5 13.5 7 9" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  return node;
}

// Hovering a coloured proof step says which tokens it accounts for.
const STEP_COLOR = { a: '--c-auto', s: '--c-syntax', r: '--c-rewrite' };
function stepTips(node) {
  node.addEventListener('mousemove', (e) => {
    const step = e.target.closest?.('.st');
    if (!step) { hideTip(); return; }
    const kind = [...step.classList].find((c) => c.startsWith('st-')).slice(3);
    const [label, rest] = step.dataset.tip.split(': ');
    showTip(e, `<div class="row"><span><i class="swatch" style="background:${cssVar(STEP_COLOR[kind])}"></i>${esc(label)}</span></div><div>${esc(rest || '')}</div>`);
  });
  node.addEventListener('mouseleave', hideTip);
  return node;
}

// The selected agent's compression origin on a fixed scale from 0 to 100% of
// the baseline, in two lanes. Savings stack right from zero in the upper lane;
// costs (added declarations, proofs that grew) stack back from the end of the
// savings as hatched segments in a thinner lane beneath, so every saving stays
// visible and a tick through both lanes marks the net compression. A net
// increase in size extends the scale left of zero.
function originStrip(origin, baseline, label) {
  const pts = (v) => (v / baseline) * 100;
  const parts = {
    added: pts(origin.added), dead_code: pts(origin.dead_code), deleted: pts(origin.deleted),
    proof_rewrite: pts(origin.proof_simplification + origin.structural_diff), automation: pts(origin.automation),
    syntax_optimization: pts(origin.syntax_optimization),
  };
  const positive = ORIGIN.filter((o) => parts[o.key] > 0);
  const negative = ORIGIN.filter((o) => parts[o.key] < 0);
  const right = positive.reduce((a, o) => a + parts[o.key], 0);
  const net = negative.reduce((a, o) => a + parts[o.key], right);
  const lo = Math.min(0, net), hi = Math.max(100, right);
  const at = (v) => ((v - lo) / (hi - lo)) * 100;
  // [category, from, to] in points: savings from 0 up, costs from the net score up.
  const placed = [];
  let x = 0;
  for (const o of positive) placed.push([o, x, x += parts[o.key]]);
  x = net;
  for (const o of negative) placed.push([o, x, x -= parts[o.key]]);
  const segment = ([o, from, to]) => {
    const v = parts[o.key];
    const color = cssVar(o.color);
    return h('i', {
      class: v < 0 ? 'cost' : null,
      style: { left: `${Math.min(at(from), at(to))}%`, width: `${Math.abs(at(to) - at(from))}%`,
        background: v < 0 ? `repeating-linear-gradient(135deg, ${color} 0 3px, var(--surface) 3px 4.5px)` : color },
      onmousemove: (e) => showTip(e, `<b>${esc(label)}</b><div class="row"><span><i class="swatch" style="background:${color}"></i>${o.label}</span><span>${v > 0 ? '+' : '−'}${Math.abs(v).toFixed(1)} pts</span></div>`),
      onmouseleave: hideTip });
  };
  // Legend in the order the segments appear, left to right.
  const shown = [...positive, ...negative].filter((o) => Math.abs(parts[o.key]) >= 0.05);
  return h('div', { class: 'origin-strip' },
    h('span', { class: 'eyebrow' }, 'Compression origin'),
    h('div', { class: 'origin-bar', role: 'img', 'aria-label': `${ORIGIN.map((o) => `${o.label} ${parts[o.key].toFixed(1)} points`).join(', ')}, net ${net.toFixed(1)}% of the baseline` },
      ...placed.map(segment), lo < 0 ? h('b', { class: 'zero', style: { left: `${at(0)}%` } }) : null,
      negative.length ? h('b', { class: 'net', style: { left: `${at(net)}%` } }) : null),
    h('div', { class: 'origin-axis', 'aria-hidden': 'true' },
      lo < 0 && at(0) > 15 ? h('span', { class: 'start' }, `−${(-lo).toFixed(1)}%`) : null,
      h('span', { class: at(0) < 3 ? 'start' : 'zero', style: { left: `${at(0)}%` } }, '0%'),
      h('span', { class: 'end' }, '100%')),
    h('div', { class: 'origin-keys' }, shown.map((o) =>
      h('span', {}, h('i', { class: 'swatch', style: { background: cssVar(o.color) } }), o.label, h('b', {}, `${parts[o.key] > 0 ? '' : '−'}${Math.abs(parts[o.key]).toFixed(1)}`)))));
}
