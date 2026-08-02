/* The Learn tab: the repo as a course you can actually do.
 *
 * Every lesson is a triple — read the doc, open the file, break it and watch a real test go
 * red — so this tab's job is to make those three things one loop instead of three tabs. The
 * check button runs an actual pytest node through /api/learn/check; it is the same node a
 * terminal would run and it records into the same learning/progress.json, so
 * `python -m aksharallm.learn` and this page never disagree about what you have done.
 *
 * The rule worth knowing while reading this file: a lesson completes on RED THEN GREEN. The
 * check passes on a clean checkout, so "it passed" would tick every lesson for someone who
 * never opened a file. The server decides that; this only renders it.
 */
import { $, api, escHtml, flash, fmt, post } from './core.js';
import { state } from './state.js';
import { registerTab, showView } from './router.js';
import { renderMarkdown } from './markdown.js';
import { openFile } from './code.js';

const learn = { data: null, current: null, lesson: null, loaded: false };

const MARKS = { complete: '[x]', broken: '[!]', started: '[~]', todo: '[ ]' };

async function openLearnTab() {
  if (!learn.loaded) {
    learn.loaded = true;
    await refreshList(true);
  } else {
    refreshList(false);
  }
}

async function refreshList(openNext) {
  try {
    const d = await api('/api/learn');
    learn.data = d;
    renderList();
    if (openNext && !learn.current) {
      /* Open on the first unfinished lesson that is unlocked. Arriving at a course and
       * being asked to choose from thirteen items is a worse start than being put back
       * where you were. */
      const want = d.next || (d.lessons[0] && d.lessons[0].id);
      if (want) selectLesson(want);
    }
  } catch (err) {
    $('#ln-count').textContent = `no answer — ${err.message}`;
  }
}

function renderList() {
  const d = learn.data;
  if (!d) return;
  $('#ln-count').textContent = `${d.total} lessons`;
  $('#ln-bar-fill').style.width = `${d.total ? (100 * d.complete) / d.total : 0}%`;
  $('#ln-progress-label').textContent = `${d.complete}/${d.total} complete`;

  /* A lesson that has drifted from the code it describes is the one failure this tab must
   * never hide — it would send someone to break a line that has moved. */
  const problems = $('#ln-problems');
  problems.hidden = !(d.problems || []).length;
  if ((d.problems || []).length) {
    problems.innerHTML = '<strong>These lessons no longer match the code:</strong><br>'
      + d.problems.map(escHtml).join('<br>');
  }

  $('#ln-lessons').innerHTML = d.lessons.map((l) => {
    const st = l.progress.state;
    const cls = ['ln-lesson', l.id === learn.current ? 'on' : '',
      !l.open ? 'locked' : '', st === 'complete' ? 'done' : '',
      st === 'broken' ? 'red' : ''].filter(Boolean).join(' ');
    const sub = !l.open ? l.reason
      : st === 'started' ? 'green, but not broken yet — that is the exercise'
      : st === 'broken' ? 'red now: put it back and run the check again'
      : l.summary;
    return `<li class="${cls}" data-id="${escHtml(l.id)}">
      <span class="ln-mark">${l.open ? MARKS[st] : '[-]'}</span>
      <span class="ln-name">${escHtml(l.title)}</span>
      <span class="ln-sub">${escHtml(sub || '')}</span>
    </li>`;
  }).join('');
}

async function selectLesson(id, keepResult) {
  learn.current = id;
  renderList();
  try {
    const l = await api(`/api/learn/lesson?id=${encodeURIComponent(id)}`);
    learn.lesson = l;
    renderLesson(l, keepResult);
  } catch (err) {
    $('#ln-title').textContent = 'could not open that lesson';
    $('#ln-body').innerHTML = `<p class="ln-empty">${escHtml(err.message)}</p>`;
  }
}

function renderLesson(l, keepResult) {
  $('#ln-title').textContent = l.title;
  const p = l.progress;
  $('#ln-state').textContent = p.complete
    ? `complete · ${p.attempts} check${p.attempts === 1 ? '' : 's'}`
    : p.state === 'broken' ? 'red right now — fix it and check again'
    : p.state === 'started' ? 'green, not broken yet'
    : l.minutes ? `about ${l.minutes} minutes` : '';

  $('#ln-body').innerHTML = renderMarkdown(l.body || '');
  /* A lesson body names files, and a repo-relative link would navigate the portal to a dead
   * URL like /aksharallm/data/loader.py. Send those to the Code tab instead — the same
   * bargain the Docs reader strikes, and the same place the file chips below go. */
  for (const a of $('#ln-body').querySelectorAll('a[href]')) {
    const href = a.getAttribute('href');
    if (/^(https?:|#)/i.test(href)) {
      if (href[0] !== '#') { a.target = '_blank'; a.rel = 'noopener noreferrer'; }
      continue;
    }
    const target = href.split('#')[0];
    if (/\.md$/i.test(target)) a.dataset.doc = target;
    else a.dataset.file = target;
    a.setAttribute('href', '#');
  }

  /* Phase C: the hand-offs. The Code tab opens the exact file the exercise edits, and the
   * Playground opens the probe the lesson talks about — so "read it, look at it, run it"
   * does not mean finding three things by hand. */
  $('#ln-jump').hidden = false;
  $('#ln-open-doc').hidden = !l.doc;
  $('#ln-open-doc').textContent = l.doc ? `Read ${l.doc.split('/').pop()}` : '';
  $('#ln-files').innerHTML = (l.files || []).map((f) =>
    `<button class="chip file" type="button" data-file="${escHtml(f)}">${escHtml(f.split('/').slice(-2).join('/'))}</button>`).join('');
  $('#ln-open-play').hidden = !l.play;

  const locked = $('#ln-locked');
  locked.hidden = l.open;
  locked.textContent = l.open ? '' : `Locked — ${l.reason}. You can still read it.`;

  $('#ln-check').hidden = false;
  $('#ln-verify').textContent = l.verify || '';
  $('#ln-run').disabled = false;
  /* A check that just ran has a fresher answer than the progress file does — including the
   * pytest output and the sentence saying what this particular red or green *means*. Only
   * fall back to the recorded run when arriving at the lesson cold. */
  if (keepResult) return;
  $('#ln-result').hidden = true;
  $('#ln-output').hidden = true;
  const last = (p.runs || []).slice(-1)[0];
  if (last) showResult({ passed: last.passed, summary: last.detail, output: '',
    note: p.complete ? 'Lesson complete.' : 'Last run — press the button to run it again.',
    duration_s: last.duration_s });
}

function showResult(res) {
  const box = $('#ln-result');
  box.hidden = false;
  box.className = `ln-result ${res.passed ? 'pass' : 'fail'}`;
  box.innerHTML = `<span class="ln-verdict">${res.passed ? 'GREEN' : 'RED'}</span>`
    + `<span class="ln-summary">${escHtml(res.summary || '')}</span>`
    + `${escHtml(res.note || '')}`
    + (res.duration_s ? `<span class="ln-sub"> · ${res.duration_s}s</span>` : '');
  const out = $('#ln-output');
  out.hidden = !res.output;
  out.textContent = res.output || '';
}

async function runCheck() {
  const l = learn.lesson;
  if (!l) return;
  $('#ln-run').disabled = true;
  $('#ln-run').textContent = 'Running…';
  try {
    const res = await post('/api/learn/check', { id: l.id, force: !l.open });
    await refreshList(false);
    await selectLesson(l.id, true);
    showResult(res);
  } catch (err) {
    flash(err.message, 'error');
  } finally {
    $('#ln-run').disabled = false;
    $('#ln-run').textContent = 'Run the check';
  }
}

export function wireLearnTab() {
  $('#ln-lessons').addEventListener('click', (e) => {
    const li = e.target.closest('.ln-lesson');
    if (li) selectLesson(li.dataset.id);
  });

  $('#ln-run').addEventListener('click', runCheck);

  $('#ln-reset').addEventListener('click', async () => {
    const l = learn.lesson;
    if (!l) return;
    if (!confirm(`Forget your progress on '${l.title}' and do it again from the start?`)) return;
    try {
      const res = await post('/api/learn/reset', { id: l.id });
      flash(res.note, 'ok');
      await refreshList(false);
      await selectLesson(l.id);
    } catch (err) { flash(err.message, 'error'); }
  });

  /* --- the hand-offs (phase C) --- */
  $('#ln-open-doc').addEventListener('click', () => {
    const l = learn.lesson;
    if (!l || !l.doc) return;
    /* The Docs tab reads the same docs/*.md files this lesson points at. */
    location.hash = 'docs';
    setTimeout(() => window.dispatchEvent(new CustomEvent('open-doc', { detail: l.doc })), 50);
  });

  $('#ln-files').addEventListener('click', (e) => {
    const btn = e.target.closest('[data-file]');
    if (!btn) return;
    /* Straight into the Code tab, on the file the exercise edits — where a local model can
     * explain the lines before you break them. */
    showView('code');
    openFile(btn.dataset.file);
  });

  /* The same two hand-offs, for links written inside the lesson's own prose. */
  $('#ln-body').addEventListener('click', (e) => {
    const file = e.target.closest('a[data-file]');
    if (file) {
      e.preventDefault();
      showView('code');
      openFile(file.dataset.file);
      return;
    }
    const doc = e.target.closest('a[data-doc]');
    if (doc) {
      e.preventDefault();
      location.hash = 'docs';
      setTimeout(() => window.dispatchEvent(new CustomEvent('open-doc', { detail: doc.dataset.doc })), 50);
    }
  });

  $('#ln-open-play').addEventListener('click', () => {
    const l = learn.lesson;
    if (!l || !l.play) return;
    showView('play');
    setTimeout(() => window.dispatchEvent(new CustomEvent('open-probe', { detail: l.play })), 50);
  });
}

registerTab('learn', { open: openLearnTab });
