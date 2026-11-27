// Standalone graph, diff and trace explorer. Hashes retain the existing shareable run format.
import { h, load, publishedBenchmark, preloadLogos } from './util.js?v=e38b2b7f7d';
import { renderRepository } from './repo.js?v=e38b2b7f7d';

const panel = document.getElementById('repository');
const EXAMPLE_HASH = '#/graph/palomar__2026-09-01-000010/opus';
let bench = null;
let shownRepo = null;
let cleanup = null;
let rendering = false;
let pending = false;

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

async function route() {
  if (rendering) { pending = true; return; }
  rendering = true;
  try {
    if (!location.hash || location.hash === '#/') history.replaceState(null, '', EXAMPLE_HASH);
    const [kind, id, short, ...rest] = location.hash.replace(/^#\/?/, '').split('/').filter(Boolean).map(decodeURIComponent);
    if (!['graph', 'repo', 'trace'].includes(kind) || !id) {
      location.replace(`./${location.hash.startsWith('#/model/') ? location.hash : '#/results'}`);
      return;
    }
    await stylesReady;
    if (!bench) {
      bench = publishedBenchmark(await load('data/benchmark.json'));
      await preloadLogos([...bench.models, ...(bench.extra_models || [])].map((m) => m.logo));
    }
    const newRepo = id !== shownRepo;
    if (newRepo && cleanup) { cleanup(); cleanup = null; }
    const result = await renderRepository(panel, bench, id, short, kind, !newRepo, rest.length ? rest.join('/') : null);
    if (typeof result === 'function') cleanup = result;
    shownRepo = id;
    const repo = bench.repositories.find((r) => r.id === id);
    document.title = `${repo?.name || 'Results'} · LeanLean Explorer`;
    if (newRepo) window.scrollTo(0, 0);
  } catch (error) {
    console.error(error);
    if (cleanup) { cleanup(); cleanup = null; }
    shownRepo = null;
    panel.replaceChildren(h('div', { class: 'error' }, 'Could not load this result. ', h('a', { href: './#/results' }, 'Back to leaderboard')));
  } finally {
    rendering = false;
    if (pending) { pending = false; route(); }
  }
}
window.addEventListener('hashchange', (event) => {
  route();
  window.goatcounter?.count?.({ referrer: event.oldURL });
});
route();
