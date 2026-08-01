/* The Synth tab: make training data with a local teacher, and see what was thrown away.
 *
 * Every button POSTs to /api/synth/*, which shells out to `python -m aksharallm.synth` —
 * this module never generates anything itself, so a job started here and one typed into a
 * terminal append to the same dataset and either can stop the other.
 *
 * The tab leads with the FUNNEL rather than with the sample count, for the same reason the
 * Finetune tab leads with the memory budget rather than the Run button: the number that
 * decides whether generated data is worth training on is not how much of it there is, it is
 * which wall the rest of it hit. 400 samples at a 20% pass rate is four different problems
 * depending on whether the loss was tests_failed, near_duplicate or unparseable, and each
 * one has a different fix.
 */
import { $, $$, api, escHtml, flash, fmt, post } from './core.js';
import { state } from './state.js';
import { registerTab } from './router.js';
import { UNIT_MINUTES, boundPicker, fmtMins, fmtWhen, stopLabel } from './dashboard.js';

const synth = { timer: null, status: null, loaded: false, selected: null,
                detail: null, side: 'kept', budget: null };

/* Why each reason exists, in one line, shown under the biggest one. Kept in step with
 * filters.REJECT_REASONS on the Python side — the server sends the counts, not the prose,
 * because the prose is about what to *do* and belongs next to the bar. */
const WHY = {
  unparseable: 'the teacher ignored the output format — usually fixed by editing the template, not by retrying',
  too_short: 'answers below the length floor; often a truncated reply',
  too_long: 'the teacher rambled past the task — lower num_predict or tighten the prompt',
  boilerplate: 'disclaimers and refusals; harmless here, but they must never reach training data',
  leaked_template: 'the instructions came back in the answer',
  no_entry_point: 'no function to test',
  bad_tests: 'fewer than two asserts, or tests that never mention the function',
  unsafe_code: 'the code touches the filesystem, network or process',
  tests_failed: 'the tests ran and failed — the exercise was wrong. This is the filter earning its keep',
  vacuous_tests: 'the tests passed with the function removed, so they proved nothing',
  sandbox_error: 'the code could not be run at all',
  duplicate: 'an exact repeat',
  near_duplicate: 'a paraphrase of something already kept — a wider seed grid or a different --seed helps',
  identical_pair: 'the preferred and rejected answers were the same',
  teacher_error: 'Ollama failed to answer — an infrastructure problem, not a data one',
};

async function openSynthTab() {
  if (!synth.loaded) {
    synth.loaded = true;
    $('#sy-name').value = $('#sy-name').value || 'py-v1';
  }
  pollSynth();
}

/* ---------------------------------------------------------------- form ---------------- */

function renderSynthForm(st) {
  const recipeSel = $('#sy-recipe');
  if (recipeSel.options.length === 0 && st.recipes) {
    recipeSel.innerHTML = st.recipes.map((r) =>
      `<option value="${escHtml(r.name)}">${escHtml(r.name)} → ${escHtml(r.consumer.toUpperCase())}</option>`).join('');
  }
  const recipe = (st.recipes || []).find((r) => r.name === recipeSel.value);
  if (recipe) {
    $('#sy-recipe-note').innerHTML = `${escHtml(recipe.blurb)}<br>`
      + `<strong>${recipe.verified ? 'The tests are executed' : 'No execution'}</strong> · `
      + `${fmt.int(recipe.grid)} distinct prompt cells (template v${recipe.template_version})`;
  }
  /* The verify checkboxes belong to the one recipe that can be verified. Showing them for
   * `chat` would suggest something is being checked that is not. */
  $('#sy-verify').closest('.sy-check').hidden = !(recipe && recipe.verified);
  $('#sy-mutate').closest('.sy-check').hidden = !(recipe && recipe.verified) || !$('#sy-verify').checked;

  const teachers = st.teachers || {};
  const sel = $('#sy-teacher');
  const wanted = (teachers.defaults || {})[recipeSel.value];
  if (sel.dataset.filled !== String((teachers.models || []).length) || sel.dataset.recipe !== recipeSel.value) {
    sel.dataset.filled = String((teachers.models || []).length);
    sel.dataset.recipe = recipeSel.value;
    const opts = (teachers.models || []).map((m) =>
      `<option value="${escHtml(m.name)}"${m.name === wanted ? ' selected' : ''}>`
      + `${escHtml(m.name)}${m.size ? ` — ${fmt.bytes(m.size)}` : ''}</option>`).join('');
    sel.innerHTML = opts || `<option value="">${escHtml(wanted || 'no models')}</option>`;
  }
  $('#sy-teacher-note').textContent = teachers.error
    ? teachers.error
    : `${teachers.host || ''}${wanted ? ` · this recipe's default is ${wanted}` : ''}`;
  $('#sy-teacher-note').classList.toggle('warn', !!teachers.error);

  /* Contention is *reported*, never decided: the teacher is loaded by Ollama in another
   * process, so unlike the Playground this panel cannot quietly fall back to the CPU. */
  const con = st.contention || {};
  const note = $('#sy-name-note');
  if ((con.training || []).length) {
    note.innerHTML = `<span class="warn">${escHtml(con.reason)}</span>`;
  } else {
    note.innerHTML = 'A directory under <code>data/synth/</code>. Generating into a name '
      + 'that already exists appends to it.';
  }
}

function synthSpec() {
  const spec = {
    recipe: $('#sy-recipe').value,
    name: $('#sy-name').value.trim(),
    n: Number($('#sy-n').value || 100),
    teacher: $('#sy-teacher').value || null,
    seed: Number($('#sy-seed').value || 0),
    dedup: Number($('#sy-dedup').value || 0.6),
    no_verify: !$('#sy-verify').checked,
    no_mutate: !$('#sy-mutate').checked,
  };
  if (synth.budget) spec.stop_in_s = synth.budget;
  return spec;
}

function renderBudgetNote() {
  $('#sy-budget-note').textContent = synth.budget
    ? `budget: ${fmtMins(Math.round(synth.budget / 60))} of wall clock, whichever comes first`
    : 'no time budget — it runs until it has the samples you asked for, or you stop it';
}

/* ---------------------------------------------------------------- status ---------------- */

function renderSynthStatus(st) {
  synth.status = st;
  renderSynthForm(st);

  const cur = st.current;
  const running = st.running;
  $('#sy-run').disabled = running;
  $('#sy-stop').hidden = !running;
  $('#sy-stop').textContent = st.stop && !st.stop.now ? 'Change stop…' : 'Stop…';

  if (cur) {
    const started = cur.started ? new Date(cur.started * 1000).toLocaleTimeString() : '';
    $('#sy-state').textContent =
      `${running ? 'running' : cur.state} — ${cur.recipe} into ${cur.dataset} `
      + `from ${cur.teacher}, started ${started}` + stopLabel(st.stop);
    $('#sy-cmd').innerHTML = `<code>python -m ${escHtml(cur.cmd || '')}</code>`;
  } else {
    $('#sy-state').textContent = 'nothing running';
  }

  const prog = st.progress;
  $('#sy-progress').hidden = !prog;
  if (prog) {
    $('#sy-bar-fill').style.width = `${prog.pct}%`;
    $('#sy-progress-label').textContent =
      `${fmt.int(prog.kept)}/${fmt.int(prog.total)} kept · ${fmt.int(prog.asked)} asked`
      + (prog.pass_rate == null ? '' : ` · ${Math.round(prog.pass_rate * 100)}% survive`);
  }

  renderDatasets(st.datasets || []);
  /* Follow the running job unless the reader picked another dataset to look at; with
   * nothing running, open on the most recently touched one — an empty funnel on a tab whose
   * whole argument is the funnel teaches nothing. */
  if (!synth.selected && cur && cur.dataset) selectDataset(cur.dataset);
  else if (!synth.selected && (st.datasets || []).length) selectDataset(st.datasets[0].name);
  else if (running && synth.selected === (cur || {}).dataset) loadDataset(synth.selected, true);

  const log = $('#sy-log');
  if (st.log && st.log.length) {
    const stick = log.scrollTop + log.clientHeight >= log.scrollHeight - 40;
    log.textContent = st.log.join('\n');
    if (stick) log.scrollTop = log.scrollHeight;
  }
}

function renderDatasets(rows) {
  const host = $('#sy-datasets');
  if (!rows.length) {
    host.innerHTML = '<p class="sy-empty">Nothing generated yet. Pick a recipe, name a '
      + 'dataset and press Generate — a few dozen samples is enough to read the funnel and '
      + 'find out whether the prompt is working.</p>';
    return;
  }
  host.innerHTML = `<table class="sy-table">
    <thead><tr><th>dataset</th><th>recipe</th><th>kept</th><th>asked</th><th>survive</th>
      <th>teacher</th><th>updated</th></tr></thead>
    <tbody>${rows.map((r) => {
      const rate = r.pass_rate == null ? '–' : `${Math.round(r.pass_rate * 100)}%`;
      const when = r.updated ? new Date(r.updated * 1000).toLocaleString() : '–';
      return `<tr data-name="${escHtml(r.name)}"${r.name === synth.selected ? ' class="on"' : ''}>
        <td><code>${escHtml(r.name)}</code></td>
        <td>${escHtml(String(r.recipe || '–'))}</td>
        <td>${fmt.int(r.kept)}</td>
        <td>${fmt.int(r.asked)}</td>
        <td>${rate}</td>
        <td>${escHtml((r.teachers || []).join(', ') || '–')}</td>
        <td>${escHtml(when)}</td></tr>`;
    }).join('')}</tbody></table>`;
}

/* ---------------------------------------------------------------- the funnel ------------ */

function renderFunnel(d) {
  const host = $('#sy-funnel');
  if (!d) { host.innerHTML = ''; return; }
  const drops = Object.entries(d.rejected || {}).sort((a, b) => b[1] - a[1]);
  const worst = drops.length ? drops[0][0] : null;
  const total = d.asked || 1;
  const rate = d.pass_rate == null ? '–' : `${Math.round(d.pass_rate * 100)}%`;
  host.innerHTML = `
    <div class="sy-funnel-head">
      <span><b>${fmt.int(d.kept)}</b> kept</span>
      <span>${fmt.int(d.asked)} asked → ${fmt.int(d.parsed)} parsed → ${fmt.int(d.kept)} kept</span>
      <span><b>${rate}</b> survive every filter</span>
      ${(d.template_versions || []).length > 1
        ? `<span class="warn">template v${(d.template_versions || []).join(' + v')} — this dataset spans two prompts</span>` : ''}
    </div>
    <div class="sy-drops">${drops.map(([reason, n]) => `
      <div class="sy-drop">
        <span class="sy-drop-name">${escHtml(reason)}</span>
        <span class="sy-drop-bar"><span style="width:${Math.min(100, 100 * n / total)}%"></span></span>
        <span class="sy-drop-n">${fmt.int(n)}</span>
      </div>
      ${reason === worst && WHY[reason] ? `<div class="sy-drop-why">${escHtml(WHY[reason])}</div>` : ''}
    `).join('')}</div>`;
}

/* ---------------------------------------------------------------- samples --------------- */

function selectDataset(name) {
  synth.selected = name;
  for (const tr of $$('#sy-datasets tr[data-name]')) {
    tr.classList.toggle('on', tr.dataset.name === name);
  }
  loadDataset(name);
}

async function loadDataset(name, quiet = false) {
  try {
    const d = await api(`/api/synth/dataset?name=${encodeURIComponent(name)}&samples=5&rejects=5`);
    synth.detail = d;
    renderFunnel(d);
    $('#sy-detail-title').textContent = name;
    $('#sy-detail-note').textContent =
      `${fmt.int(d.kept)} kept · ${fmt.int(d.rejected_total)} dropped · ${d.dir}`;
    renderSamples();
  } catch (err) {
    if (!quiet) $('#sy-detail-note').textContent = err.message;
  }
}

function renderSamples() {
  const host = $('#sy-samples');
  const d = synth.detail;
  if (!d) { host.innerHTML = ''; return; }
  const rows = synth.side === 'kept' ? (d.samples || []) : (d.rejects || []);
  if (!rows.length) {
    host.innerHTML = `<p class="sy-empty">${synth.side === 'kept'
      ? 'No samples kept yet.'
      : 'Nothing rejected — which for the python recipe would be surprising.'}</p>`;
    return;
  }
  host.innerHTML = rows.map(synth.side === 'kept' ? keptCard : rejectCard).join('');
}

const field = (label, text, mono) =>
  `<div class="sy-field"><div class="sy-field-label">${escHtml(label)}</div>
   <${mono ? 'pre class="sy-code"' : 'p class="sy-text"'}>${escHtml(text || '')}</${mono ? 'pre' : 'p'}></div>`;

function keptCard(s) {
  const head = `<div class="sy-sample-head">
    <span class="sy-id">${escHtml(s.id || '')}</span>
    <span>${escHtml(s.teacher || '')}</span>
    ${s.verified === true ? '<span class="sy-tag ok">tests passed + mutation-checked</span>' : ''}
    ${s.verified === false ? '<span class="sy-tag bad">not verified</span>' : ''}
  </div>`;
  if (s.kind === 'python') {
    return `<article class="sy-sample">${head}
      ${field('problem', s.problem)}
      ${field('solution', s.solution, true)}
      ${field('tests', s.tests, true)}
      <div class="sy-field-label">${escHtml((s.verify || {}).detail || '')}</div>
    </article>`;
  }
  if (s.kind === 'chat') {
    return `<article class="sy-sample">${head}
      ${field('prompt', s.prompt)}${field('answer', s.answer)}
      <div class="sy-field-label">${escHtml(s.constraint || '')}</div></article>`;
  }
  return `<article class="sy-sample">${head}
    ${field('prompt', s.prompt)}${field('chosen', s.chosen)}${field('rejected', s.rejected)}
    <div class="sy-field-label">flaw: ${escHtml(s.flaw || '')}</div></article>`;
}

function rejectCard(r) {
  return `<article class="sy-sample">
    <div class="sy-sample-head">
      <span class="sy-id">${escHtml(r.seed || '')}</span>
      <span class="sy-tag bad">${escHtml(r.reason || '')}</span>
      <span>${escHtml(r.detail || '')}</span>
    </div>
    ${field('what came back', (r.text || '').slice(0, 1600), true)}</article>`;
}

/* ---------------------------------------------------------------- polling --------------- */

async function pollSynth() {
  clearTimeout(synth.timer);
  if (state.view !== 'synth' || document.hidden) return;
  try {
    const st = await api('/api/synth?lines=250');
    renderSynthStatus(st);
    synth.timer = setTimeout(pollSynth, st.running ? 3000 : 10000);
  } catch (err) {
    $('#sy-state').textContent = `no answer — ${err.message}`;
    synth.timer = setTimeout(pollSynth, 10000);
  }
}

async function startSynth() {
  try {
    $('#sy-run').disabled = true;
    const spec = synthSpec();
    const res = await post('/api/synth/start', spec);
    synth.selected = res.dataset;
    flash(`Generating ${res.n} × ${res.recipe} into ${res.dataset} with ${res.teacher}.`, 'ok');
    $('#sy-log').textContent = 'starting…';
    pollSynth();
  } catch (err) {
    flash(err.message, 'error');
    $('#sy-run').disabled = false;
  }
}

export function wireSynthTab() {
  $('#sy-run').addEventListener('click', startSynth);
  $('#sy-recipe').addEventListener('change', () => {
    if (synth.status) renderSynthForm(synth.status);
  });
  $('#sy-verify').addEventListener('change', () => {
    if (synth.status) renderSynthForm(synth.status);
  });

  /* The same picker as the trainer's session budget and every stop in the portal. A
   * generation run is bounded in exactly the same two ways: a count, or a clock. */
  $('#sy-budget').addEventListener('click', async () => {
    const chosen = await boundPicker.open({
      title: 'Budget for this generation run',
      sub: 'It stops when it has the samples you asked for or when this runs out, '
        + 'whichever is first. Everything written by then is a complete dataset.',
      okLabel: 'Set budget',
      unit: synth.budget ? 'in' : 'none',
      units: [
        { ...UNIT_MINUTES, value: synth.budget || 30 * 60,
          preview: (v) => `stops about ${fmtWhen(v)}` },
        { id: 'none', label: 'No budget', none: true,
          preview: () => 'runs until it has the samples asked for, or until you stop it' },
      ],
    });
    if (chosen === null) return;
    synth.budget = chosen.unit === 'none' ? null : chosen.value;
    renderBudgetNote();
  });

  $('#sy-stop').addEventListener('click', async () => {
    const st = synth.status || {};
    const prog = st.progress;
    const chosen = await boundPicker.open({
      title: 'Stop generating at…',
      sub: prog
        ? `It has ${fmt.int(prog.kept)} of ${fmt.int(prog.total)} samples. Every one of them `
          + 'is already filtered, verified and written — stopping loses nothing but the rest.'
        : 'Every sample already written is kept; stopping loses nothing but the rest.',
      unit: 'in', showNow: true, nowLabel: 'Stop after this sample',
      units: [
        { ...UNIT_MINUTES, preview: (v) => `stops about ${fmtWhen(v)}` },
        { id: 'at', label: 'Samples', noun: 'stop at total samples',
          min: 1, max: 100000, value: prog ? Math.min(prog.total, prog.kept + 50) : 100,
          chips: [10, 50, 100, 250, 500, 1000, 5000],
          snap: [[100, 10], [1000, 50], [10000, 100], [Infinity, 500]],
          fmt: (v) => fmt.int(v),
          preview: (v) => (prog && v <= prog.kept
            ? 'it already has that many — it will stop at once'
            : `stops once the dataset holds ${fmt.int(v)} samples`) },
      ],
    });
    if (!chosen) return;
    const body = chosen === 'now' ? { mode: 'now' }
      : chosen.unit === 'in' ? { mode: 'in', seconds: chosen.value }
      : { mode: 'at', samples: chosen.value };
    try {
      const res = await post('/api/synth/stop', body);
      flash(res.note || 'Stop requested.', 'ok');
    } catch (err) { flash(err.message, 'error'); }
    pollSynth();
  });

  $('#sy-datasets').addEventListener('click', (e) => {
    const tr = e.target.closest('tr[data-name]');
    if (tr) selectDataset(tr.dataset.name);
  });

  for (const chip of $$('.sy-tabs .chip[data-side]')) {
    chip.addEventListener('click', () => {
      synth.side = chip.dataset.side;
      for (const c of $$('.sy-tabs .chip[data-side]')) c.classList.toggle('on', c === chip);
      renderSamples();
    });
  }

  $('#sy-export').addEventListener('click', async () => {
    if (!synth.selected) return flash('Pick a dataset first.', 'error');
    try {
      const res = await post('/api/synth/export', { name: synth.selected });
      flash(`Wrote ${res.rows} rows to ${res.path}. Tokenize it with:  ${res.next}`, 'ok');
    } catch (err) { flash(err.message, 'error'); }
  });

  renderBudgetNote();
}

registerTab('synth', { open: openSynthTab, leave: () => clearTimeout(synth.timer) });
