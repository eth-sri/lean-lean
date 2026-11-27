import { h } from './util.js?v=e38b2b7f7d';

// Small, sourced excerpts link to the benchmark runs they illustrate.
const EXAMPLE = 'palomar__2026-09-01-000010/opus';

// Inline our SVG artwork so text uses the same loaded fonts as the viewer.
function previewArt(name) {
  const art = h('div', { class: 'explore-art', 'aria-hidden': 'true' });
  const url = new URL(`../explore-${name}.svg${new URL(import.meta.url).search}`, import.meta.url);
  fetch(url).then((response) => {
    if (!response.ok) throw new Error(`Preview unavailable: ${response.status}`);
    return response.text();
  }).then((svg) => { art.innerHTML = svg; }).catch(() => {
    art.append(h('img', { src: url.href, alt: '', width: 360, height: 270 }));
  });
  return art;
}
export function renderExplore() {
  const previews = [
    { view: 'graph', image: 'graph', title: 'Dependency graph', text: 'See which declarations changed and how they connect.' },
    { view: 'repo', target: 'palomar__2026-08-19-000002/opus/Flypitch4%2FErdos501%2FRealsInZFSet.lean', image: 'diff', title: 'Code diff', text: 'Compare the original code with the agent’s edits.' },
    { view: 'trace', target: 'palomar__2026-09-01-000010/opus', image: 'trace', title: 'Agent trace', text: 'Follow the agent’s steps, tool calls, and outputs.' },
  ];
  return h('section', { class: 'explore-results', 'aria-labelledby': 'explore-results-title' },
    h('div', { class: 'explore-intro' },
      h('div', { class: 'explore-copy' },
        h('h2', { id: 'explore-results-title' }, 'Explore the results'),
        h('p', {}, 'Compare code before and after a run, inspect its dependency graph, and follow the agent’s tool calls. Click a repository score in the leaderboard to open that run.')),
      h('a', { class: 'explore-cta', href: `viewer.html#/repo/${EXAMPLE}` }, 'Explore an example')),
    h('div', { class: 'explore-previews' }, previews.map(({ view, target = EXAMPLE, image, title, text }) =>
      h('a', { class: 'explore-preview', href: `viewer.html#/${view}/${target}` },
        previewArt(image),
        h('div', { class: 'explore-caption' },
          h('h3', {}, title),
          h('p', {}, text))))));
}
