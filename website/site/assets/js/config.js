// A model's run configuration (#/model/<short>): harness, sampling settings,
// tools, limits, the task prompt and the harness's own system prompt. Every
// value comes from the frozen run manifests (pipeline/export_benchmark.py,
// run_configs); the system prompts load only when one is opened.
import { h, load, logo, usd, modelLabel } from './util.js?v=e38b2b7f7d';

let dialog = null;

const number = (value) => (typeof value === 'number' && Math.abs(value) >= 1000 ? value.toLocaleString('en-US') : String(value));
const memory = (text) => text.replace(/^(\d+(?:\.\d+)?)g$/i, '$1 GiB');
const duration = (text) => text.replace(/^(\d+)h$/, '$1 hours');
const rate = (value) => `$${value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 3 })}`;
const RATE_LABELS = { input: 'Input', cache_read: 'Cached input', cache_write: 'Cache write', output: 'Output' };

// The task prompt as the agents saw it: the container limits filled in, the
// start time and deadline (different for every run) left as placeholders.
function taskPrompt(prompt, shared) {
  const fill = { container_cpus: String(shared.cpus), container_memory: memory(shared.memory), time_limit: shared.timeout };
  const parts = prompt.split(/(\{\w+\})/);
  return parts.map((part) => {
    const key = part.match(/^\{(\w+)\}$/)?.[1];
    if (!key) return part;
    if (key in fill) return h('mark', { class: 'filled' }, fill[key]);
    return h('mark', { class: 'slot' }, key === 'current_time' ? 'start time, UTC' : key === 'deadline' ? 'deadline, UTC' : key);
  });
}

function settingRow(s) {
  const note = s.note;
  return h('div', { class: 'cfg-row' },
    h('dt', {}, s.label),
    h('dd', {},
      h('span', { class: 'cfg-val' }, number(s.value)),
      note ? h('p', { class: 'cfg-note' }, note) : null));
}

function systemPrompt(model, config) {
  const body = h('div', { class: 'cfg-sys-body' });
  const details = h('details', { class: 'cfg-sys' },
    h('summary', {}, h('span', {}, `Show ${config.harness}'s system prompt`), h('span', { class: 'muted num' }, `${config.system_prompt_chars.toLocaleString('en-US')} characters`)),
    body);
  details.addEventListener('toggle', async () => {
    if (!details.open || body.childElementCount) return;
    body.append(h('p', { class: 'muted' }, 'Loading…'));
    try {
      const prompts = await load('data/system-prompts.json');
      const sp = prompts[model.key];
      body.replaceChildren(
        h('p', { class: 'cfg-note' }, `Captured from ${config.harness} ${config.harness_version} with the run's flags${sp.captured ? ` on ${sp.captured}` : ''}, in a test container: working directories, dates and session details differ from the real runs.`),
        h('pre', { class: 'cfg-pre' }, sp.text));
    } catch (error) {
      body.replaceChildren(h('p', { class: 'error-inline' }, `Could not load the system prompt: ${error.message}`));
    }
  });
  return details;
}

export function openModelConfig(board, short, onClose) {
  const model = board.models.find((m) => m.short === short);
  if (!model?.config) return false;
  const config = model.config;
  const shared = board.run.shared;
  if (!dialog) {
    // Focusable itself, so opening it does not put a focus ring on the close button.
    dialog = h('dialog', { class: 'cfg-dialog', 'aria-labelledby': 'cfg-title', tabindex: '-1' });
    dialog.addEventListener('click', (e) => { if (e.target === dialog) dialog.close(); });
    document.body.append(dialog);
  }
  dialog.onclose = () => onClose?.();

  const effort = config.settings.find((s) => s.key === 'effort');
  dialog.replaceChildren(
    h('header', { style: { '--model': model.color } },
      logo(model, 'sm'),
      h('div', { class: 'title' },
        h('h2', { id: 'cfg-title' }, modelLabel(model)),
        h('p', {}, `${model.provider} · ${config.harness} ${config.harness_version} · ${effort.value} effort`)),
      h('button', { type: 'button', class: 'close', 'aria-label': 'Close', onclick: () => dialog.close() }, '×')),
    h('div', { class: 'cfg-body' },
      h('section', {},
        h('h3', {}, 'Model Configuration'),
        h('dl', { class: 'cfg-list' },
          h('div', { class: 'cfg-row' }, h('dt', {}, 'Harness'), h('dd', {}, h('span', { class: 'cfg-val' }, `${config.harness} ${config.harness_version}`))),
          config.settings.map(settingRow))),
      h('section', {},
        h('h3', {}, 'Tools'),
        h('p', { class: 'cfg-lede' }, `Tools available to ${config.harness} in these runs (${config.tools.length} tools).`),
        h('ul', { class: 'cfg-tools' }, config.tools.map((t) => h('li', {}, h('code', {}, t)))),
        config.tools_note ? h('p', { class: 'cfg-note' }, config.tools_note) : null,
        h('p', { class: 'cfg-note' }, 'From the shell, every agent could also run ', h('code', {}, 'lake'), ', ', h('code', {}, 'lean_verify'), ' and ',
          h('code', {}, 'python3 proof_length.py'), '. ',
          config.restrictions_note || 'We turned off skills, MCP servers and subagents for every agent.')),
      h('section', {},
        h('h3', {}, 'Limits'),
        h('dl', { class: 'cfg-list' },
          h('div', { class: 'cfg-row' }, h('dt', {}, 'Wall clock'), h('dd', {}, h('span', { class: 'cfg-val' }, duration(shared.timeout)))),
          h('div', { class: 'cfg-row' }, h('dt', {}, 'Container'), h('dd', {}, h('span', { class: 'cfg-val' }, `${shared.cpus} CPUs · ${memory(shared.memory)} RAM`))))),
      h('section', {},
        h('h3', {}, 'Cost'),
        h('dl', { class: 'cfg-list' },
          h('div', { class: 'cfg-row' }, h('dt', {}, 'Per repository'), h('dd', {},
            h('span', { class: 'cfg-val' }, usd(model.cost)),
            model.cost_note ? h('p', { class: 'cfg-note' }, model.cost_note) : null)),
          Object.entries(config.rates).map(([kind, value]) => h('div', { class: 'cfg-row' },
            h('dt', {}, RATE_LABELS[kind]),
            h('dd', {}, h('span', { class: 'cfg-val' }, rate(value)), h('span', { class: 'cfg-unit' }, 'per million tokens')))))),
      h('section', {},
        h('h3', {}, 'Task prompt'),
        h('p', { class: 'cfg-lede' }, 'The same for every model.'),
        h('pre', { class: 'cfg-pre' }, taskPrompt(board.prompt, shared))),
      h('section', {},
        h('h3', {}, 'System prompt'),
        h('p', { class: 'cfg-lede' }, `${config.harness}'s built-in system prompt, unchanged; we added no instructions of our own.`),
        systemPrompt(model, config))));
  if (!dialog.open) dialog.showModal();
  dialog.focus();
  dialog.querySelector('.cfg-body').scrollTop = 0;
  return true;
}

export function closeModelConfig() {
  if (dialog?.open) { dialog.onclose = null; dialog.close(); }
}
