// Landing page and model configuration dialogs. Legacy result links redirect to the explorer.
import { h, load, publishedBenchmark, registerLogos } from './util.js?v=e38b2b7f7d';
import { renderLeaderboard, renderMethod, renderCitation } from './leaderboard.js?v=e38b2b7f7d';
import { renderExplore } from './explore.js?v=e38b2b7f7d';

const view = document.getElementById('view');
let board = null;
let boardEl = null;
let openedFrom = null;
const loadConfig = () => import('./config.js?v=e38b2b7f7d');
const RESULTS_STATE = 'leanlean.results.v1';

const styled = () => getComputedStyle(document.documentElement).getPropertyValue('--z0').trim() !== '';
const stylesReady = styled() ? Promise.resolve() : new Promise((resolve) => {
  const link = document.querySelector('link[rel="stylesheet"][href*="site.css"]');
  let settled = false;
  const finish = () => { settled = true; resolve(); };
  const done = () => { if (settled) return; if (styled()) finish(); else requestAnimationFrame(done); };
  link?.addEventListener('load', done, { once: true });
  link?.addEventListener('error', finish, { once: true });
  // Also poll: a stylesheet that finished between the check and the listener fires no event.
  requestAnimationFrame(done);
});

function saveResultsState() {
  if (!boardEl) return;
  try {
    sessionStorage.setItem(RESULTS_STATE, JSON.stringify({
      highestOnly: boardEl.querySelector('.results-filter input').checked,
      tab: boardEl.querySelector('.results-body').dataset.tab,
      table: boardEl.querySelector('.lb-wrap').getState(),
      scrollY: window.scrollY,
    }));
  } catch { /* Navigation still works when browser storage is unavailable. */ }
}

async function restoreResultsState() {
  let state;
  try { state = JSON.parse(sessionStorage.getItem(RESULTS_STATE)); } catch { return; }
  if (!state) {
    boardEl.querySelector('[role="tab"][data-tab="summary"]').click();
    boardEl.querySelector('.board-panel').scrollIntoView();
    return;
  }
  const filter = boardEl.querySelector('.results-filter input');
  if (typeof state.highestOnly === 'boolean' && filter.checked !== state.highestOnly) {
    filter.checked = state.highestOnly;
    filter.dispatchEvent(new Event('change'));
  }
  const tab = ['frontier', 'summary', 'repos'].includes(state.tab) ? state.tab : 'summary';
  boardEl.querySelector(`[role="tab"][data-tab="${tab}"]`).click();
  boardEl.querySelector('.lb-wrap').setState(state.table);
  await document.fonts?.ready;
  requestAnimationFrame(() => window.scrollTo(0, Number.isFinite(state.scrollY) ? state.scrollY : 0));
}

async function route() {
  try {
    const parts = location.hash.replace(/^#\/?/, '').split('/').filter(Boolean).map(decodeURIComponent);
    if (['graph', 'repo', 'trace'].includes(parts[0]) && parts[1]) {
      location.replace(`viewer.html${location.hash}`);
      return;
    }
    const isModel = parts[0] === 'model' && parts[1];
    if (parts.length && !isModel && parts[0] !== 'results') { location.replace('#/'); return; }
    if (!board) {
      await stylesReady;
      board = publishedBenchmark(await load('data/leaderboard.json'));
      registerLogos(board.logos);
    }
    if (!boardEl) {
      const host = h('div');
      renderLeaderboard(host, board);
      boardEl = host.firstElementChild;
      boardEl.append(renderExplore());
      view.replaceChildren(boardEl, renderMethod(board), renderCitation());
      const returning = performance.getEntriesByType('navigation')[0]?.type === 'back_forward';
      if (parts[0] === 'results' || returning) await restoreResultsState();
    }
    if (isModel) {
      const { openModelConfig } = await loadConfig();
      const from = openedFrom;
      const shown = openModelConfig(board, parts[1], () => {
        if (!location.hash.startsWith('#/model/')) return;
        if (from != null) history.back(); else location.replace('#/');
      });
      if (!shown) location.replace('#/');
      return;
    }
    if (document.querySelector('dialog.cfg-dialog[open]')) (await loadConfig()).closeModelConfig();
  } catch (error) {
    console.error(error);
    view.replaceChildren(h('div', { class: 'error' }, 'Could not load the results. ', h('a', { href: './' }, 'Try again')));
    boardEl = null;
  }
}

window.addEventListener('pagehide', saveResultsState);
window.addEventListener('hashchange', (event) => {
  const from = new URL(event.oldURL).hash;
  if (location.hash.startsWith('#/model/') && !from.startsWith('#/model/')) openedFrom = from;
  route();
  window.goatcounter?.count?.({ referrer: event.oldURL });
});
route();
