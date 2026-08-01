import { $, $$, api, escHtml } from './core.js';
import { renderMarkdown } from './markdown.js';
import { registerTab, showView } from './router.js';
import { openFile } from './code.js';

/* --- docs tab: read the same docs/*.md in the portal, diagrams and all ---------------
 * Content comes from /api/source/file (SourceTree already serves .md); the ordered list
 * from /api/docs. Mermaid is vendored locally and loaded LAZILY — only the first time this
 * tab is opened — so a dashboard left up overnight never pays for a 3 MB diagram library. */
const docState = { list: [], path: null, loaded: false };

let mermaidReady = null;
function ensureMermaid() {
  if (window.mermaid) return Promise.resolve(window.mermaid);
  if (!mermaidReady) {
    mermaidReady = new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = '/static/mermaid.min.js';
      s.onload = () => resolve(window.mermaid);
      s.onerror = () => reject(new Error('could not load the diagram library'));
      document.head.appendChild(s);
    });
  }
  return mermaidReady;
}

async function renderDocDiagrams(container) {
  const pres = [...container.querySelectorAll('pre.language-mermaid')];
  if (!pres.length) return;
  let mermaid;
  try { mermaid = await ensureMermaid(); } catch { return; }  // keep the source if it won't load
  mermaid.initialize({ startOnLoad: false, theme: 'dark', securityLevel: 'loose' });
  const nodes = pres.map((pre) => {
    const div = document.createElement('div');
    div.className = 'mermaid';
    div.textContent = pre.textContent;   // the raw diagram source, unescaped
    pre.replaceWith(div);
    return div;
  });
  try { await mermaid.run({ nodes }); } catch { /* a bad diagram just shows its source */ }
}

/** Resolve a doc-relative link ('02-tokenizer.md' from within 'docs/01-data.md') to a
 *  repo-relative path ('docs/02-tokenizer.md'), handling ./ and ../ and a #fragment. */
function resolveDocPath(base, href) {
  const target = href.split('#')[0];
  const parts = base.includes('/') ? base.slice(0, base.lastIndexOf('/')).split('/') : [];
  for (const seg of target.split('/')) {
    if (seg === '..') parts.pop();
    else if (seg && seg !== '.') parts.push(seg);
  }
  return parts.join('/');
}

async function loadDoc(path) {
  docState.path = path;
  for (const li of $$('#docs-list li')) li.classList.toggle('on', li.dataset.path === path);
  const reader = $('#docs-reader');
  reader.innerHTML = '<p class="docs-hint">loading…</p>';
  try {
    const res = await api(`/api/source/file?path=${encodeURIComponent(path)}`);
    reader.innerHTML = renderMarkdown(res.text || '');
    // Rewire links: another doc (.md) loads in this reader; anything external opens in a
    // new tab so a click never navigates the portal away from itself.
    for (const a of reader.querySelectorAll('a[href]')) {
      const href = a.getAttribute('href');
      if (/^https?:/i.test(href)) { a.target = '_blank'; a.rel = 'noopener noreferrer'; }
      else if (/^#/.test(href)) { /* in-page anchor: harmless, leave it */ }
      else if (/\.md(#|$)/i.test(href)) { a.dataset.doc = resolveDocPath(path, href); a.setAttribute('href', '#'); }
      // any other repo-relative link is a source file -> open it in the Code tab, never
      // let it navigate the portal to a dead URL like /aksharallm/data/prepare.py
      else { a.dataset.src = resolveDocPath(path, href); a.setAttribute('href', '#'); }
    }
    reader.scrollTop = 0;
    await renderDocDiagrams(reader);
  } catch (err) {
    reader.innerHTML = `<p class="docs-hint">could not open ${escHtml(path)} — ${escHtml(err.message)}</p>`;
  }
}

/* The Learn tab hands off here: "read docs/06-inference.md" should land on that page, not
 * on whatever was open last. An event rather than an import, so the two tabs stay strangers
 * — docs.js does not need to know the learning path exists. */
window.addEventListener('open-doc', async (e) => {
  const path = e.detail;
  if (!path) return;
  await openDocsTab();
  loadDoc(path);
});

async function openDocsTab() {
  if (docState.loaded) return;
  docState.loaded = true;
  try {
    const res = await api('/api/docs');
    docState.list = res.docs || [];
    $('#docs-list').innerHTML = docState.list.map((d) =>
      `<li data-path="${escHtml(d.path)}"><button type="button">${escHtml(d.title)}</button></li>`).join('');
    $('#docs-list').addEventListener('click', (e) => {
      const li = e.target.closest('li[data-path]');
      if (li) loadDoc(li.dataset.path);
    });
    // In-reader links: another doc loads in place; a source file opens in the Code tab.
    $('#docs-reader').addEventListener('click', (e) => {
      const doc = e.target.closest('a[data-doc]');
      if (doc) { e.preventDefault(); loadDoc(doc.dataset.doc); return; }
      const src = e.target.closest('a[data-src]');
      if (src) {
        e.preventDefault();
        const p = src.dataset.src;
        localStorage.setItem('aksharallm-code-path', p);  // so lazy init opens THIS file
        showView('code');
        openFile(p);
      }
    });
    const first = docState.list.find((d) => d.path.includes('00-')) || docState.list[0];
    if (first) loadDoc(first.path);
  } catch (err) {
    $('#docs-reader').innerHTML = `<p class="docs-hint">could not list docs — ${escHtml(err.message)}</p>`;
  }
}

registerTab('docs', { open: openDocsTab });
