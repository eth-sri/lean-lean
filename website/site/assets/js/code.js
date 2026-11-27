// Lean 4 code rendering: a small line highlighter, snippet diffs, unified hunks.
import { h, esc } from './util.js?v=e38b2b7f7d';

const KEYWORDS = new Set(('theorem lemma def abbrev instance structure class inductive where by have show from fun let in match with ' +
  'if then else do return namespace section end open variable variables universe import private protected noncomputable ' +
  'attribute calc at deriving extends mutual example axiom opaque macro syntax notation infix infixl infixr prefix postfix ' +
  'set_option local scoped termination_by decreasing_by partial unsafe for suffices this Type Prop Sort').split(' '));
const TACTICS = new Set(('simp simp_all simpa simp_rw dsimp rw rwa rewrite rfl exact exact_mod_cast apply intro intros refine refine\' ' +
  'constructor cases rcases obtain induction omega linarith nlinarith norm_num ring ring_nf field_simp positivity aesop grind ' +
  'decide native_decide tauto use exists specialize unfold subst congr ext funext gcongr trivial contradiction exfalso by_contra ' +
  'by_cases push_neg split split_ifs interval_cases fin_cases polyrith linear_combination convert assumption first repeat all_goals ' +
  'any_goals try rintro nth_rewrite nth_rw change calc left right exists_intro push_cast norm_cast bound fun_prop continuity ' +
  'measurability filter_upwards symm trans infer_instance inferInstance haveI letI show_term apply_fun zify qify lift mono').split(' '));

// Highlight one line; `state.block` tracks open /- -/ comments across lines.
// With `cont`, the text continues the previous call's line (used for in-line diff marks).
export function highlight(line, state = { block: 0 }, cont = false) {
  if (cont && state.line) return line ? `<span class="tk-com">${esc(line)}</span>` : '';
  state.line = false;
  let out = '', i = 0;
  const n = line.length;
  while (i < n) {
    if (state.block) {
      const start = i;
      while (i < n && state.block) {
        if (line.startsWith('/-', i)) { state.block++; i += 2; } else if (line.startsWith('-/', i)) { state.block--; i += 2; } else i++;
      }
      out += `<span class="tk-com">${esc(line.slice(start, i))}</span>`;
      continue;
    }
    const c = line[i];
    if (line.startsWith('--', i)) { out += `<span class="tk-com">${esc(line.slice(i))}</span>`; state.line = true; break; }
    if (line.startsWith('/-', i)) { state.block = 1; const start = i; i += 2;
      while (i < n && state.block) { if (line.startsWith('/-', i)) { state.block++; i += 2; } else if (line.startsWith('-/', i)) { state.block--; i += 2; } else i++; }
      out += `<span class="tk-com">${esc(line.slice(start, i))}</span>`; continue; }
    if (c === '"') { let j = i + 1; while (j < n && line[j] !== '"') j += line[j] === '\\' ? 2 : 1; out += `<span class="tk-str">${esc(line.slice(i, j + 1))}</span>`; i = j + 1; continue; }
    if (/[A-Za-z_À-ɏͰ-Ͽἀ-῿]/.test(c)) {
      let j = i + 1; while (j < n && /[A-Za-z0-9_'.!?À-ɏͰ-Ͽἀ-῿₀-ₜ]/.test(line[j])) j++;
      const word = line.slice(i, j);
      out += KEYWORDS.has(word) ? `<span class="tk-kw">${word}</span>` : TACTICS.has(word) ? `<span class="tk-tac">${word}</span>` : esc(word);
      i = j; continue;
    }
    if (line.startsWith(':=', i)) { out += '<span class="tk-sym">:=</span>'; i += 2; continue; }
    if ('⊢←→↔∀∃λ⟨⟩▸·'.includes(c)) { out += `<span class="tk-sym">${esc(c)}</span>`; i++; continue; }
    out += esc(c); i++;
  }
  return out;
}

// A changed line's class: plain `add` (green) or `del` (red) by default; `kindOf(line number)`
// may name what the change was, each kind with its own colour: for an added line a
// (automation), s (syntax) or r (proof rewrite), for a removed one d (dead code).
const lineClass = (base, kindOf, n) => { const k = kindOf?.(n); return k ? `${base} k-${k}` : base; };

// Colour the proof steps of a rendered line: spans [from, to, kind, note] in UTF-16
// columns of the line's text. Each becomes <b class="st st-kind" data-tip="note">, split
// where it crosses highlighting; the word marks of a stepped line give way to the steps.
export function markSteps(el, spans) {
  if (!el || !spans?.length) return;
  for (const m of el.querySelectorAll('mark')) m.replaceWith(...m.childNodes);
  el.normalize();
  for (const [from, to, kind, note] of spans) {
    const walk = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
    const hits = [];
    let pos = 0;
    for (let t; (t = walk.nextNode());) {
      const a = Math.max(from, pos), b = Math.min(to, pos + t.data.length);
      if (a < b && !t.parentElement.closest('.st')) hits.push([t, a - pos, b - pos]);
      pos += t.data.length;
    }
    for (const [t, a, b] of hits) {
      const mid = t.splitText(a);
      mid.splitText(b - a);
      const step = document.createElement('b');
      step.className = `st st-${kind}`;
      step.dataset.tip = note;
      mid.replaceWith(step);
      step.append(mid);
    }
  }
  el.closest('.ln, .sl > span')?.classList.add('stepped');
}

function row(kind, oldNo, newNo, html, gutter = '') {
  const node = h('div', { class: `ln ${kind}` });
  node.innerHTML = `<i class="o">${oldNo ?? ''}</i>${newNo !== undefined ? `<i class="o">${newNo ?? ''}</i>` : ''}<i class="g">${gutter}</i><span>${html || ' '}</span>`;
  return node;
}

export function codeBlock(lines, start = 1, { wrap = false } = {}) {
  const pre = h('pre', { class: `code${wrap ? ' wrap' : ''}` });
  const state = { block: 0 };
  const frag = document.createDocumentFragment();
  lines.forEach((line, k) => frag.append(row('', start + k, undefined, highlight(line, state))));
  pre.append(frag);
  return pre;
}

// Line diff of two snippets: common prefix/suffix, LCS on the middle.
export function lineDiff(a, b) {
  let p = 0; while (p < a.length && p < b.length && a[p] === b[p]) p++;
  let q = 0; while (q < a.length - p && q < b.length - p && a[a.length - 1 - q] === b[b.length - 1 - q]) q++;
  const A = a.slice(p, a.length - q), B = b.slice(p, b.length - q);
  const ops = [];
  for (let i = 0; i < p; i++) ops.push([' ', a[i], i, i]);
  if (A.length * B.length <= 2_500_000) {
    const m = A.length, n = B.length, dp = new Uint32Array((m + 1) * (n + 1));
    for (let i = m - 1; i >= 0; i--) for (let j = n - 1; j >= 0; j--)
      dp[i * (n + 1) + j] = A[i] === B[j] ? dp[(i + 1) * (n + 1) + j + 1] + 1 : Math.max(dp[(i + 1) * (n + 1) + j], dp[i * (n + 1) + j + 1]);
    let i = 0, j = 0;
    while (i < m || j < n) {
      if (i < m && j < n && A[i] === B[j]) { ops.push([' ', A[i], p + i, p + j]); i++; j++; }
      else if (j < n && (i === m || dp[i * (n + 1) + j + 1] >= dp[(i + 1) * (n + 1) + j])) { ops.push(['+', B[j], null, p + j]); j++; }
      else { ops.push(['-', A[i], p + i, null]); i++; }
    }
  } else {
    A.forEach((line, i) => ops.push(['-', line, p + i, null]));
    B.forEach((line, j) => ops.push(['+', line, null, p + j]));
  }
  for (let k = 0; k < q; k++) ops.push([' ', a[a.length - q + k], a.length - q + k, b.length - q + k]);
  return ops;
}

// Group a line diff into blocks: runs of context, and change blocks (removed lines, then added lines).
// Paired removed/added lines get in-line marks on the words that differ.
export function diffBlocks(beforeLines, afterLines, beforeStart = 1, afterStart = 1) {
  const ops = lineDiff(beforeLines, afterLines);
  const sa = { block: 0 }, sb = { block: 0 };
  const blocks = [];
  let adds = 0, dels = 0;
  for (let k = 0; k < ops.length;) {
    if (ops[k][0] === ' ') {
      const rows = [];
      for (; k < ops.length && ops[k][0] === ' '; k++) {
        const [, text, i, j] = ops[k];
        rows.push({ o: beforeStart + i, n: afterStart + j, html: highlight(text, sa) });
        highlight(text, sb);
      }
      blocks.push({ ctx: rows });
      continue;
    }
    const del = [], add = [];
    for (; k < ops.length && ops[k][0] !== ' '; k++) (ops[k][0] === '-' ? del : add).push(ops[k]);
    const pairs = Math.min(del.length, add.length);
    const delRows = del.map(([, text, i], x) => ({ o: beforeStart + i,
      html: x < pairs ? marked(text, add[x][1], sa) : highlight(text, sa) }));
    const addRows = add.map(([, text, , j], x) => ({ n: afterStart + j,
      html: x < pairs ? marked(text, del[x][1], sb) : highlight(text, sb) }));
    dels += del.length; adds += add.length;
    blocks.push({ del: delRows, add: addRows });
  }
  return { blocks, adds, dels };
}

const WORD = /[A-Za-z0-9_'.!?À-ɏͰ-Ͽἀ-῿₀-ₜ]/;
// Highlight `text`, wrapping the part that differs from `other` in <mark>, snapped to whole words.
function marked(text, other, state) {
  let p = 0; while (p < text.length && p < other.length && text[p] === other[p]) p++;
  let q = 0; while (q < text.length - p && q < other.length - p && text[text.length - 1 - q] === other[other.length - 1 - q]) q++;
  while (p > 0 && WORD.test(text[p - 1]) && WORD.test(text[p] ?? '')) p--;
  while (q > 0 && WORD.test(text[text.length - q]) && WORD.test(text[text.length - q - 1] ?? '')) q--;
  const mid = text.slice(p, text.length - q);
  // Lines that share only indentation or punctuation were rewritten, not edited: plain highlighting reads better.
  const shared = text.slice(0, p).trim().length + text.slice(text.length - q).trim().length;
  if (!mid.trim() || shared < 3) return highlight(text, state);
  return highlight(text.slice(0, p), state) + `<mark>${highlight(mid, state, true)}</mark>` + highlight(text.slice(text.length - q), state, true);
}

// A run of unchanged lines. With `context`, long runs fold into a button that expands them:
// a run between changes keeps `context` lines on each side, a leading or trailing run only the side next to a change.
function contextRun(rows, context, at, draw, container) {
  const keep = at === 'mid' ? 2 * context : context;
  if (!context || at === 'all' || rows.length <= keep + 2) { container.append(...rows.map(draw)); return; }
  const head = at === 'start' ? 0 : context, tail = at === 'end' ? 0 : context;
  const hidden = rows.slice(head, rows.length - tail);
  const fold = h('button', { type: 'button', class: 'ln fold', title: 'Show unchanged lines' }, `⋯ ${hidden.length} unchanged line${hidden.length === 1 ? '' : 's'}`);
  fold.addEventListener('click', () => fold.replaceWith(...hidden.map(draw)));
  container.append(...rows.slice(0, head).map(draw), fold, ...rows.slice(rows.length - tail).map(draw));
}
const position = (k, n) => (n === 1 ? 'all' : k === 0 ? 'start' : k === n - 1 ? 'end' : 'mid');

// Unified snippet diff. Options: context (fold unchanged runs; 0 = never), wrap (soft-wrap long lines),
// kindOf (after line number -> kind of change, see lineClass), removedKindOf (the same for before line numbers).
export function diffBlock(beforeLines, afterLines, beforeStart = 1, afterStart = 1, { context = 0, wrap = false, kindOf, removedKindOf, spansOf } = {}) {
  const pre = h('pre', { class: `code${wrap ? ' wrap' : ''}` });
  const { blocks, adds, dels } = diffBlocks(beforeLines, afterLines, beforeStart, afterStart);
  Object.assign(pre.dataset, { adds, dels });
  blocks.forEach((b, k) => {
    if (b.ctx) contextRun(b.ctx, context, position(k, blocks.length), (r) => row('', r.o, r.n, r.html), pre);
    else {
      b.del.forEach((r) => pre.append(row(lineClass('del', removedKindOf, r.o), r.o, '', r.html, '−')));
      b.add.forEach((r) => {
        const node = row(lineClass('add', kindOf, r.n), '', r.n, r.html, '+');
        markSteps(node.lastElementChild, spansOf?.(r.n));
        pre.append(node);
      });
    }
  });
  return pre;
}

// Side-by-side snippet diff: removed lines on the left, facing the lines that replaced them.
export function splitBlock(beforeLines, afterLines, beforeStart = 1, afterStart = 1, { context = 0, kindOf, removedKindOf, spansOf } = {}) {
  const box = h('div', { class: 'code split' });
  const { blocks, adds, dels } = diffBlocks(beforeLines, afterLines, beforeStart, afterStart);
  Object.assign(box.dataset, { adds, dels });
  const side = (kind, no, html) => `<i class="o">${no ?? ''}</i><span class="${kind}">${html || ' '}</span>`;
  const line = (left, right) => { const node = h('div', { class: 'sl' }); node.innerHTML = left + right; return node; };
  blocks.forEach((b, k) => {
    if (b.ctx) contextRun(b.ctx, context, position(k, blocks.length), (r) => line(side('', r.o, r.html), side('', r.n, r.html)), box);
    else for (let x = 0; x < Math.max(b.del.length, b.add.length); x++) {
      const d = b.del[x], a = b.add[x];
      const node = line(d ? side(lineClass('del', removedKindOf, d.o), d.o, d.html) : side('pad', '', ''),
        a ? side(lineClass('add', kindOf, a.n), a.n, a.html) : side('pad', '', ''));
      if (a) markSteps(node.lastElementChild, spansOf?.(a.n));
      box.append(node);
    }
  });
  return box;
}

// Unified hunks: [oldStart, oldLen, newStart, newLen, lines[]]
export function hunksBlock(hunks, limit = 4000, kindOf, removedKindOf, spansOf) {
  const pre = h('pre', { class: 'code' });
  let shown = 0;
  const frag = document.createDocumentFragment();
  for (const [os, , ns, , lines] of hunks) {
    frag.append(row('hunk', '', '', esc(`@@ −${os} +${ns} @@`), ''));
    let o = os, n = ns;
    const state = { block: 0 };
    for (const line of lines) {
      if (shown++ >= limit) break;
      const op = line[0], text = line.slice(1);
      const html = highlight(text, state);
      if (op === '+') {
        const node = row(lineClass('add', kindOf, n), '', n, html, '+');
        markSteps(node.lastElementChild, spansOf?.(n));
        frag.append(node);
        n++;
      }
      else if (op === '-') { frag.append(row(lineClass('del', removedKindOf, o), o, '', html, '−')); o++; }
      else frag.append(row('', o++, n++, html, ''));
    }
    if (shown >= limit) break;
  }
  pre.append(frag);
  const total = hunks.reduce((a, hk) => a + hk[4].length, 0);
  return { pre, truncated: total > limit, total };
}

// A patch as text: unified diffs (git or plain) and Codex's `*** Begin Patch` format.
// File headers and hunk headers become header rows; Lean files are highlighted.
export function patchBlock(text, { wrap = true } = {}) {
  const pre = h('pre', { class: `code${wrap ? ' wrap' : ''}` });
  const lines = String(text).split('\n');
  const frag = document.createDocumentFragment();
  let o = null, n = null, lean = true, state = { block: 0 };
  const file = (path) => { lean = /\.lean$/.test(path.trim()); state = { block: 0 }; };
  for (let k = 0; k < lines.length; k++) {
    const line = lines[k];
    // `--- a/x` followed by `+++ b/x` is a file header; a lone `--- …` is a removed Lean comment.
    if (line.startsWith('--- ') && lines[k + 1]?.startsWith('+++ ')) {
      file(lines[k + 1].slice(4).replace(/^b\//, ''));
      frag.append(row('hunk', '', '', esc(lines[k + 1].slice(4).replace(/^b\//, '')), ''));
      k++;
      continue;
    }
    if (/^(diff --git |index |new file mode|deleted file mode|similarity index|rename (from|to) )/.test(line)) continue;
    const star = line.match(/^\*\*\* (Update|Add|Delete) File: (.*)$/);
    if (star) { file(star[2]); frag.append(row('hunk', '', '', esc(`${star[1] === 'Update' ? '' : `${star[1]} `}${star[2].replace(/^\/testbed\//, '')}`), '')); o = n = null; continue; }
    if (/^\*\*\* (Begin|End) Patch|^\*\*\* End of File/.test(line)) continue;
    if (line.startsWith('@@')) {
      const m = line.match(/^@@ -(\d+)(?:,\d+)? \+(\d+)/);
      [o, n] = m ? [Number(m[1]), Number(m[2])] : [null, null];
      frag.append(row('hunk', '', '', esc(line.replace(/^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@/, '@@ −$1 +$2 @@')), ''));
      continue;
    }
    if (line.startsWith('… ') && line.endsWith(' …')) { frag.append(row('omit', '', '', esc(line), '')); continue; }
    const op = line[0], body = line.slice(1);
    const html = lean ? highlight(body, state) : esc(body);
    if (op === '+') frag.append(row('add', '', n != null ? n++ : '', html, '+'));
    else if (op === '-') frag.append(row('del', o != null ? o++ : '', '', html, '−'));
    else frag.append(row('', o != null ? o++ : '', n != null ? n++ : '', lean ? highlight(line.slice(op === ' ' ? 1 : 0), state) : esc(line.slice(op === ' ' ? 1 : 0)), ''));
  }
  pre.append(frag);
  return pre;
}

export function wholeFileBlock(text, kind, limit = 4000, kindOf) {
  const lines = text.split('\n');
  if (lines.length && lines[lines.length - 1] === '') lines.pop();
  const pre = h('pre', { class: 'code' });
  const frag = document.createDocumentFragment();
  const state = { block: 0 };
  lines.slice(0, limit).forEach((line, k) => frag.append(
    row(lineClass(kind, kindOf, k + 1), kind === 'del' ? k + 1 : '', kind === 'add' ? k + 1 : '', highlight(line, state), kind === 'add' ? '+' : '−')));
  pre.append(frag);
  return { pre, truncated: lines.length > limit, total: lines.length };
}
