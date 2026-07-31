import { $, $$, api, escHtml, fmt } from './core.js';
import { el } from './charts.js';
import { highlight, renderMarkdown } from './markdown.js';
import { registerTab } from './router.js';

/* ---------------------------------------------------------------- code tab ------------ */

/* The second half of the portal: read the project, and ask a model on this machine what a
 * selection is doing. Everything here is inert until the tab is opened for the first time —
 * a dashboard left up overnight should not be listing files or loading a model.
 *
 * Three small engines: a filterable file list, a numbered code pane you select in, and a
 * streaming reader that turns server-sent events into markdown as they arrive. */

const code = {
  files: [],
  dirs: [],
  root: '',
  dir: '',           // the folder being browsed, '' = where the portal is running
  path: null,
  text: '',
  lines: [],
  lang: '',
  doc: null,
  start: null,       // 1-based, inclusive
  end: null,
  snippet: null,     // the exact characters, when the selection is not whole lines
  model: localStorage.getItem('aksharallm-explain-model') || '',
  messages: [],      // the follow-up conversation for the current selection
  answer: '',
  abort: null,
  loaded: false,
  anchor: null,      // gutter click-and-drag origin
};

/** The immediate children of `dir` — the folders you can go down into from here. */
function subdirs(dir) {
  const prefix = dir ? dir + '/' : '';
  const depth = dir ? dir.split('/').length : 0;
  return code.dirs
    .filter((d) => d.startsWith(prefix) && d.split('/').length === depth + 1)
    .map((d) => ({ path: d, name: d.split('/').pop() }));
}

function goto(dir) {
  code.dir = dir;
  localStorage.setItem('aksharallm-code-dir', dir);
  renderTree();
  $('#file-tree').scrollTop = 0;
}

function renderTree() {
  const host = $('#file-tree');
  const crumbs = $('#file-crumbs');
  const q = $('#file-filter').value.trim().toLowerCase();
  host.textContent = '';
  crumbs.textContent = '';

  /* Breadcrumbs are the way back up: every ancestor is one click, including the root. */
  const rootName = code.root.split('/').filter(Boolean).pop() || '/';
  const trail = [{ path: '', name: rootName }];
  if (code.dir) {
    code.dir.split('/').forEach((part, i, all) => {
      trail.push({ path: all.slice(0, i + 1).join('/'), name: part });
    });
  }
  trail.forEach((step, i) => {
    if (i) crumbs.append(Object.assign(document.createElement('span'),
      { className: 'crumb-sep', textContent: '/' }));
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'crumb' + (i === trail.length - 1 ? ' on' : '');
    btn.dataset.dir = step.path;
    btn.textContent = step.name;
    crumbs.appendChild(btn);
  });

  /* Filtering searches the whole tree, not just this folder — otherwise you would have to
   * know where a file is before you could find it. */
  if (q) {
    const hits = code.files.filter((f) => f.path.toLowerCase().includes(q));
    $('#file-count').textContent = `${hits.length} of ${code.files.length} files match`;
    let dir = null;
    for (const f of hits.slice(0, 400)) {
      if (f.dir !== dir) {
        dir = f.dir;
        host.append(Object.assign(document.createElement('div'),
          { className: 'file-dir', textContent: dir || rootName }));
      }
      host.appendChild(fileButton(f));
    }
    if (!hits.length) {
      host.append(Object.assign(document.createElement('div'),
        { className: 'file-dir', textContent: 'nothing matches' }));
    }
    return;
  }

  const folders = subdirs(code.dir);
  const here = code.files.filter((f) => f.dir === code.dir);
  $('#file-count').textContent =
    `${folders.length} folder${folders.length === 1 ? '' : 's'} · ${here.length} file`
    + `${here.length === 1 ? '' : 's'}`;

  if (code.dir) {                       /* the way up, one level */
    const up = document.createElement('button');
    up.type = 'button';
    up.className = 'folder up';
    up.dataset.dir = code.dir.split('/').slice(0, -1).join('/');
    up.textContent = '../';
    host.appendChild(up);
  }
  for (const d of folders) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'folder';
    btn.dataset.dir = d.path;
    btn.textContent = d.name + '/';
    host.appendChild(btn);
  }
  for (const f of here) host.appendChild(fileButton(f));
  if (!folders.length && !here.length) {
    host.append(Object.assign(document.createElement('div'),
      { className: 'file-dir', textContent: 'nothing readable here' }));
  }
}

function fileButton(f) {
  const btn = document.createElement('button');
  btn.className = 'file' + (f.path === code.path ? ' on' : '');
  btn.type = 'button';
  btn.dataset.path = f.path;
  btn.title = `${f.path} · ${fmt.bytes(f.size)}`;
  btn.textContent = f.name;
  return btn;
}

/* --- the code pane -------------------------------------------------------------------- */

export async function openFile(path) {
  try {
    const data = await api(`/api/source/file?path=${encodeURIComponent(path)}`);
    code.path = data.path;
    code.text = data.text;
    code.lang = data.lang;
    code.doc = data.doc;
    code.lines = highlight(data.text, data.lang);
    code.start = code.end = null;
    code.snippet = null;
    code.messages = [];
    code.answer = '';
    $('#explain-out').textContent = '';
    $('#followup-form').hidden = true;
    $('#btn-explain-clear').hidden = true;
    explainStatus('');
    $('#code-path').textContent = data.path;
    $('#code-note').textContent =
      `${fmt.int(data.lines)} lines · ${fmt.bytes(data.size)}`
      + (data.doc ? ` · deep dive: ${data.doc}` : '');
    renderCode();
    /* Follow the file into its folder, so opening a filter hit leaves you somewhere you
     * can keep browsing rather than back at the root. */
    const dir = data.path.includes('/') ? data.path.replace(/\/[^/]*$/, '') : '';
    if (!$('#file-filter').value.trim()) code.dir = dir;
    renderTree();
    localStorage.setItem('aksharallm-code-path', data.path);
    localStorage.setItem('aksharallm-code-dir', code.dir);
    $('#code-view').scrollTop = 0;
  } catch (err) {
    $('#code-note').textContent = err.message;
  }
}

function renderCode() {
  const host = $('#code-view');
  host.textContent = '';
  const frag = document.createDocumentFragment();
  code.lines.forEach((html, i) => {
    const row = document.createElement('div');
    row.className = 'cl';
    row.dataset.line = String(i + 1);
    const num = document.createElement('span');
    num.className = 'ln';
    num.textContent = String(i + 1);
    const src = document.createElement('code');
    src.className = 'lc';
    src.innerHTML = html || '​';
    row.append(num, src);
    frag.appendChild(row);
  });
  host.appendChild(frag);
  markSelection();
}

function setSelection(start, end, snippet) {
  code.start = Math.min(start, end);
  code.end = Math.max(start, end);
  code.snippet = snippet || null;
  code.messages = [];          /* a new selection is a new conversation */
  markSelection();
}

function markSelection() {
  for (const row of $$('#code-view .cl')) {
    const n = Number(row.dataset.line);
    row.classList.toggle('sel', code.start != null && n >= code.start && n <= code.end);
  }
  const label = $('#explain-sel');
  const btn = $('#btn-explain');
  if (code.start == null) {
    label.textContent = code.path ? 'nothing selected' : 'no file open';
    btn.disabled = true;
    return;
  }
  const n = code.end - code.start + 1;
  label.textContent = code.start === code.end
    ? `line ${code.start}` + (code.snippet ? ' (part of it)' : '')
    : `lines ${code.start}–${code.end} · ${n} lines`;
  btn.disabled = false;
}

/** Turn whatever the browser thinks is selected into a line range. Dragging across the
 *  code, double-clicking a word and shift-clicking all end up here. */
function selectionToLines() {
  const sel = window.getSelection();
  if (!sel || sel.isCollapsed || !sel.rangeCount) return;
  const pane = $('#code-view');
  const row = (node) => {
    const el = node && (node.nodeType === 1 ? node : node.parentElement);
    return el && pane.contains(el) ? el.closest('.cl') : null;
  };
  const a = row(sel.anchorNode);
  const b = row(sel.focusNode);
  if (!a || !b) return;
  const text = sel.toString();
  const from = Number(a.dataset.line), to = Number(b.dataset.line);
  const whole = text.trim() === slice(Math.min(from, to), Math.max(from, to)).trim();
  setSelection(from, to, whole ? null : text);
}

const slice = (from, to) => code.text.split('\n').slice(from - 1, to).join('\n');

/* --- asking --------------------------------------------------------------------------- */

function explainStatus(text, kind = '') {
  const box = $('#explain-status');
  box.textContent = text;
  box.className = 'explain-status ' + kind;
}

async function loadModels() {
  const sel = $('#explain-model');
  try {
    const info = await api('/api/explain/models');
    sel.textContent = '';
    const names = info.models.map((m) => m.name);
    if (!names.includes(info.model)) names.unshift(info.model);
    for (const name of names) {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name;
      sel.appendChild(opt);
    }
    code.model = names.includes(code.model) ? code.model : info.model;
    sel.value = code.model;
    if (!info.available) {
      explainStatus(info.error, 'err');
    } else if (!info.models.some((m) => m.name === info.model)) {
      explainStatus(`${info.model} is configured but not pulled — `
        + `run: ollama pull ${info.model}`, 'err');
    }
    /* The explainer and the trainer share one card, and the trainer got there first. */
    const warn = $('#explain-warn');
    if (info.training && info.training.length && !info.on_cpu) {
      warn.innerHTML = `<strong>${escHtml(info.training.join(', '))}</strong> is training `
        + 'on this GPU. Asking here loads the model onto the same card and can run it out '
        + 'of memory. To read while training, set <code>num_gpu: 0</code> under '
        + '<code>explain:</code> in <code>configs/portal.yaml</code> — slower, but it stays '
        + 'off the GPU entirely.';
      warn.hidden = false;
    } else if (info.on_cpu) {
      warn.textContent = 'The explainer is pinned to the CPU (num_gpu: 0), so it cannot '
        + 'disturb a training run — but a 12B model can take minutes to produce its first '
        + 'word this way. A smaller model is the better trade while a run is going.';
      warn.hidden = false;
    } else {
      warn.hidden = true;
    }
  } catch (err) {
    explainStatus(err.message, 'err');
  }
}

/** A reasoning model's scratchpad, folded away. Empty for a model that does not think. */
const thinkingBlock = (text) => (text
  ? `<details class="md-think"><summary>reasoning (${fmt.int(text.length)} chars)</summary>`
    + `<pre>${escHtml(text)}</pre></details>`
  : '');

/** Stream one answer. `question` null means the default ask; `follow` keeps the thread. */
async function explain(question, follow = false) {
  if (code.start == null || code.abort) return;
  const controller = new AbortController();
  code.abort = controller;
  $('#btn-explain').disabled = true;
  $('#btn-explain-stop').hidden = false;
  const started = Date.now();
  explainStatus(`${code.model} is reading ${code.path}…`, 'busy');

  const history = follow ? code.messages.slice() : [];
  if (follow) history.push({ role: 'user', content: question });
  else code.messages = [];

  let answer = '';
  let thinking = '';
  const out = $('#explain-out');
  if (!follow) out.textContent = '';
  const head = follow
    ? `<h4 class="md-ask">${escHtml(question)}</h4>`
    : '';
  const prefix = out.innerHTML + head;

  try {
    const res = await fetch('/api/explain', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Portal': '1' },
      signal: controller.signal,
      body: JSON.stringify({
        path: code.path,
        start: code.start,
        end: code.end,
        snippet: code.snippet,
        model: code.model,
        question: follow ? question : (question || null),
        history,
      }),
    });
    if (!res.ok) {
      let msg = `HTTP ${res.status}`;
      try { msg = (await res.json()).error || msg; } catch { /* no body */ }
      throw new Error(msg);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let stop = false;
    while (!stop) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split('\n\n');
      buffer = frames.pop();
      for (const frame of frames) {
        const line = frame.replace(/^data: /, '').trim();
        if (!line) continue;
        let evt;
        try { evt = JSON.parse(line); } catch { continue; }
        if (evt.error) throw new Error(evt.error);
        /* A reasoning model talks to itself first. Show that it is working, and keep the
         * transcript foldable underneath — but never mix it into the answer. */
        if (evt.thinking) {
          thinking += evt.thinking;
          explainStatus(`${code.model} is thinking… ${fmt.int(thinking.length)} chars`,
            'busy');
        }
        if (evt.delta) {
          answer += evt.delta;
          out.innerHTML = prefix + thinkingBlock(thinking) + renderMarkdown(answer);
          out.scrollTop = out.scrollHeight;
        }
        if (evt.done) stop = true;
      }
    }
    if (follow) code.messages.push({ role: 'user', content: question });
    code.messages.push({ role: 'assistant', content: answer });
    out.innerHTML = prefix + thinkingBlock(thinking) + renderMarkdown(answer);
    if (!answer.trim()) {
      /* A reasoning model that spent its whole budget thinking. Say so, rather than
       * leaving an empty panel that looks like the portal broke. */
      explainStatus(thinking
        ? `${code.model} thought but never answered — it ran out of budget. Raise `
          + 'num_predict, or set think: false, in configs/portal.yaml.'
        : `${code.model} returned nothing.`, 'err');
    } else {
      explainStatus(`${code.model} · ${fmt.dur((Date.now() - started) / 1000)}`
        + (code.doc ? ` · the human-written version is ${code.doc}` : ''));
    }
    $('#followup-form').hidden = false;
    $('#btn-explain-clear').hidden = false;
  } catch (err) {
    if (err.name === 'AbortError') {
      explainStatus(`stopped after ${fmt.dur((Date.now() - started) / 1000)}`);
      if (answer) code.messages.push({ role: 'assistant', content: answer });
    } else {
      explainStatus(err.message, 'err');
    }
  } finally {
    code.abort = null;
    $('#btn-explain-stop').hidden = true;
    $('#btn-explain').disabled = code.start == null;
  }
}

export function wireCode() {
  $('#file-filter').addEventListener('input', renderTree);

  $('#file-tree').addEventListener('click', (e) => {
    const folder = e.target.closest('.folder');
    if (folder) return goto(folder.dataset.dir);
    const btn = e.target.closest('.file');
    if (btn) openFile(btn.dataset.path);
  });

  $('#file-crumbs').addEventListener('click', (e) => {
    const crumb = e.target.closest('.crumb');
    if (crumb) { $('#file-filter').value = ''; goto(crumb.dataset.dir); }
  });

  const pane = $('#code-view');

  /* Click a line number for that line; drag or shift-click down the gutter for a block.
   * The gutter is a separate gesture from selecting text, so a reader can grab a 40-line
   * function without the browser scrolling away under a text selection. */
  pane.addEventListener('mousedown', (e) => {
    const num = e.target.closest('.ln');
    if (!num) return;
    e.preventDefault();
    const line = Number(num.parentElement.dataset.line);
    if (e.shiftKey && code.start != null) setSelection(code.start, line);
    else { code.anchor = line; setSelection(line, line); }
  });
  pane.addEventListener('mouseover', (e) => {
    if (code.anchor == null || !(e.buttons & 1)) return;
    const row = e.target.closest('.cl');
    if (row) setSelection(code.anchor, Number(row.dataset.line));
  });
  window.addEventListener('mouseup', () => { code.anchor = null; });

  /* Selecting the text itself works too, including part of a line. */
  pane.addEventListener('mouseup', () => setTimeout(selectionToLines, 0));
  pane.addEventListener('dblclick', () => setTimeout(selectionToLines, 0));

  $('#explain-model').addEventListener('change', (e) => {
    code.model = e.target.value;
    localStorage.setItem('aksharallm-explain-model', code.model);
  });

  for (const btn of $$('#lenses .ghost')) {
    btn.addEventListener('click', () => {
      for (const other of $$('#lenses .ghost')) other.classList.toggle('on', other === btn);
      if (code.start != null) explain(btn.dataset.ask || null);
    });
  }

  $('#btn-explain').addEventListener('click', () => {
    const lens = $('#lenses .on');
    explain((lens && lens.dataset.ask) || null);
  });
  $('#btn-explain-stop').addEventListener('click', () => code.abort && code.abort.abort());
  $('#btn-explain-clear').addEventListener('click', () => {
    code.messages = [];
    $('#explain-out').textContent = '';
    $('#followup-form').hidden = true;
    $('#btn-explain-clear').hidden = true;
    explainStatus('');
  });

  $('#followup-form').addEventListener('submit', (e) => {
    e.preventDefault();
    const q = $('#followup').value.trim();
    if (!q || code.abort) return;
    $('#followup').value = '';
    explain(q, true);
  });
}

/** First time the tab is opened: fetch the file list and the model list, not before. */
async function openCodeTab() {
  if (code.loaded) return;
  code.loaded = true;
  try {
    const { files, dirs, root } = await api('/api/source');
    code.files = files;
    code.dirs = dirs;
    code.root = root;
    const lastDir = localStorage.getItem('aksharallm-code-dir') || '';
    code.dir = (lastDir && dirs.includes(lastDir)) ? lastDir : '';
    renderTree();
    await loadModels();
    const last = localStorage.getItem('aksharallm-code-path');
    const first = files.find((f) => f.path === last)
      || files.find((f) => f.path === 'aksharallm/model/transformer.py')
      || files.find((f) => f.path.startsWith('aksharallm/'));
    if (first) openFile(first.path);
  } catch (err) {
    code.loaded = false;
    $('#file-count').textContent = err.message;
  }
}

registerTab('code', { open: openCodeTab });
