/* The Tools tab: scores by kind with the held-out kind flagged, two reports compared problem
 * by problem, a live question with its whole transcript, and the two CLI jobs.
 *
 * Files are `tooluse.*`, not `tools.*`, to keep a generic word out of static filenames (the
 * ad-blocker lesson in AGENTS.md: `quant.js` was silently blocked).
 */
import { $, api, escHtml, flash, fmt, post } from './core.js';
import { registerTab } from './router.js';
import { jobBox, pairedLine } from './clijob.js';

const tu = { results: [], held: 'remainder', job: null, busy: false };

function kindsTable(a, b) {
  const kinds = Object.keys(a.by_kind);
  const cell = (r, k, m) => (r ? `${Math.round(r.by_kind[k][m] * r.by_kind[k].n)}/${r.by_kind[k].n}` : '');
  const head = b
    ? '<tr><th>kind</th><th>a: with tool</th><th>b: with tool</th><th>a: without</th><th>b: without</th><th>b called the tool</th></tr>'
    : '<tr><th>kind</th><th>with the tool</th><th>without</th><th>called the tool</th></tr>';
  const held = a.held_out || [tu.held];
  const row = (k) => {
    const name = held.includes(k)
      ? `<span class="tu-held">${escHtml(k)} (held out of set ${escHtml(a.set || 'v1')})</span>` : escHtml(k);
    const last = b || a;
    const called = `${Math.round(last.by_kind[k].called * last.by_kind[k].n)}/${last.by_kind[k].n}`;
    return b
      ? `<tr><td>${name}</td><td>${cell(a, k, 'tool')}</td><td><b>${cell(b, k, 'tool')}</b></td>`
        + `<td>${cell(a, k, 'direct')}</td><td>${cell(b, k, 'direct')}</td><td>${called}</td></tr>`
      : `<tr><td>${name}</td><td><b>${cell(a, k, 'tool')}</b></td><td>${cell(a, k, 'direct')}</td><td>${called}</td></tr>`;
  };
  const tot = (r, m) => `${fmt.pct(r[m])}`;
  const all = b
    ? `<tr><td><b>all</b></td><td>${tot(a, 'tool')}</td><td><b>${tot(b, 'tool')}</b></td><td>${tot(a, 'direct')}</td><td>${tot(b, 'direct')}</td><td>${fmt.pct(b.called, 0)}</td></tr>`
    : `<tr><td><b>all</b></td><td><b>${tot(a, 'tool')}</b></td><td>${tot(a, 'direct')}</td><td>${fmt.pct(a.called, 0)}</td></tr>`;
  return `<thead>${head}</thead><tbody>${kinds.map(row).join('')}${all}</tbody>`;
}

async function show() {
  const a = tu.results.find((r) => r.name === $('#tu-a').value);
  const bName = $('#tu-b').value;
  const b = tu.results.find((r) => r.name === bName);
  if (!a) {
    $('#tu-kinds').innerHTML = '<tbody><tr><td>no reports yet — Score a chat model below</td></tr></tbody>';
    return;
  }
  $('#tu-kinds').innerHTML = kindsTable(a, b);
  $('#tu-cmp-cmd').hidden = true;
  if (b) {
    try {
      const c = await post('/api/tooluse/compare', { a: a.name, b: b.name });
      const p = c.paired;
      $('#tu-paired').innerHTML = `Paired over the same ${c.n} problems (b against a):<br>`
        + pairedLine(p['with the tool'].b_only, p['with the tool'].a_only, p['with the tool'].p, 'with the tool') + '<br>'
        + pairedLine(p['without the tool'].b_only, p['without the tool'].a_only, p['without the tool'].p, 'without the tool');
      $('#tu-cmp-cmd').hidden = false;
      $('#tu-cmp-cmd').textContent = c.command;
    } catch (e) { $('#tu-paired').textContent = e.message; }
  } else {
    const p = a.paired || {};
    $('#tu-paired').innerHTML = `Within this report, with the tool against without: right only with it ${p.tool_only}, `
      + `only without it ${p.direct_only} (sign test p ${p.p != null ? Number(p.p).toPrecision(2) : '–'}).`;
  }
  try {
    const r = await api(`/api/tooluse/report?name=${encodeURIComponent(a.name)}`);
    $('#tu-rows').innerHTML = '<thead><tr><th>kind</th><th>question</th><th>answer</th><th>with tool</th><th>without</th></tr></thead><tbody>'
      + r.rows.map((x) => `<tr><td>${escHtml(x.kind)}</td><td>${escHtml(x.question)}</td><td>${escHtml(x.answer)}</td>`
        + `<td class="${x.tool ? 'tu-ok' : 'tu-no'}">${escHtml(x.tool_final)}</td>`
        + `<td class="${x.direct ? 'tu-ok' : 'tu-no'}">${escHtml(x.direct_final)}</td></tr>`).join('') + '</tbody>';
  } catch (e) { /* the summary above is still right */ }
}

async function load() {
  const res = await api('/api/tooluse');
  tu.results = res.results || [];
  tu.held = res.held_out;
  const opts = tu.results.map((r) => `<option value="${escHtml(r.name)}">${escHtml(r.name)} — set ${escHtml(r.set || 'v1')}, ${fmt.pct(r.tool)} with the tool</option>`).join('');
  const keepA = $('#tu-a').value; const keepB = $('#tu-b').value;
  $('#tu-a').innerHTML = opts;
  $('#tu-b').innerHTML = '<option value="">(nothing)</option>' + opts;
  // Default to the natural pair: the oldest report against the newest (before vs after).
  if (keepA) $('#tu-a').value = keepA;
  else if (tu.results.length > 1) { $('#tu-a').value = tu.results[tu.results.length - 1].name; }
  if (keepB) $('#tu-b').value = keepB;
  else if (tu.results.length > 1) $('#tu-b').value = tu.results[0].name;
  await show();
  const cks = res.checkpoints || [];
  $('#tu-ckpt').innerHTML = cks.map((c) => `<option value="${escHtml(c.rel)}">${escHtml(c.rel)}${c.tool_trained ? ' — trained with the tool' : ''}</option>`).join('');
  $('#tu-ask').disabled = !cks.length;
  $('#tu-eval').disabled = !cks.length;
  $('#tu-status').textContent = `on the ${res.device} — ${res.device_reason}`;
  tu.job.render(res.jobs);
  if (res.jobs && res.jobs.running) tu.job.poll();
}

async function ask() {
  if (tu.busy) return;
  tu.busy = true;
  $('#tu-ask').disabled = true;
  $('#tu-status').textContent = 'thinking… (the first question loads the model)';
  try {
    const res = await post('/api/tooluse/ask', {
      checkpoint: $('#tu-ckpt').value, question: $('#tu-question').value, use_tool: $('#tu-use').checked,
    });
    $('#tu-chat').innerHTML = res.messages.map((m) =>
      `<div class="tu-turn r-${escHtml(m.role)}"><b>${m.role === 'tool' ? 'tool (the sandbox printed)' : escHtml(m.role)}</b>${escHtml(m.content)}</div>`).join('');
    $('#tu-ask-cmd').hidden = false;
    $('#tu-ask-cmd').textContent = res.command;
    $('#tu-status').textContent = `${res.calls} tool call(s) on the ${res.device}; final number: ${res.number ?? 'none'}`;
  } catch (e) {
    $('#tu-status').textContent = e.message;
  } finally {
    tu.busy = false;
    $('#tu-ask').disabled = false;
  }
}

registerTab('tooluse', {
  async open() {
    if (!tu.job) tu.job = jobBox($('#tu-job'), 'tooluse', { onDone: () => load() });
    $('#tu-a').onchange = show;
    $('#tu-b').onchange = show;
    $('#tu-ask').onclick = ask;
    $('#tu-question').onkeydown = (e) => { if (e.key === 'Enter') ask(); };
    $('#tu-eval').onclick = () => tu.job.start({
      kind: 'eval', checkpoint: $('#tu-ckpt').value, label: $('#tu-label').value, set: $('#tu-set').value,
    });
    $('#tu-data').onclick = () => tu.job.start({ kind: 'data', set: $('#tu-set').value });
    await load().catch((e) => flash(`tools: ${e.message}`, 'error'));
  },
  leave() { if (tu.job) tu.job.stop(); },
});
