/* Just enough markdown and syntax highlighting for an explanation. Hand-written for the same
 * reason as the charts — and because the input is a stream, so it is re-rendered from scratch
 * on every chunk and has to stay cheap. Shared by the code tab (what the model says) and the
 * docs tab (the real docs/*.md). */

import { escHtml } from './core.js';

/** Just enough markdown for an explanation: fences, headings, lists, bold, inline code.
 *  Hand-written for the same reason as the charts — and because the input is a stream, so
 *  it is re-rendered from scratch on every chunk and has to stay cheap. */
export function renderMarkdown(src) {
  const out = [];
  let list = null;      // 'ul' | 'ol' | null
  let para = [];
  /* The current list item, buffered rather than emitted immediately, so a wrapped item can
   * be joined onto it. Markdown lets an item run over several lines; without this, the
   * second line closed the list and started a paragraph, and the *next* item opened a fresh
   * <ol> numbered from 1 again. Every numbered list in docs/lessons/ hit that. */
  let item = null;

  const flushItem = () => {
    if (item) { out.push(`<li>${inline(item.join(' '))}</li>`); item = null; }
  };
  const closeList = () => {
    flushItem();
    if (list) { out.push(`</${list}>`); list = null; }
  };
  const closePara = () => {
    if (para.length) { out.push(`<p>${inline(para.join(' '))}</p>`); para = []; }
  };
  const inline = (s) => escHtml(s)
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[\s(])\*([^*\n]+)\*/g, '$1<em>$2</em>')
    // [text](url) -> a link. The Docs tab rewires .md links to load in the reader and
    // sends http(s) links to a new tab (see loadDoc); elsewhere they're plain anchors.
    .replace(/(^|[^!])\[([^\]]+)\]\(([^)]+)\)/g, '$1<a href="$3">$2</a>');

  const parts = src.split(/```/);
  parts.forEach((block, i) => {
    if (i % 2) {                                   /* inside a fence */
      closePara(); closeList();
      const nl = block.indexOf('\n');
      const lang = (nl >= 0 ? block.slice(0, nl) : '').trim().toLowerCase().replace(/[^a-z0-9-]/g, '');
      const body = nl >= 0 ? block.slice(nl + 1) : block;
      // Keep the fence language as a class, so the Docs tab can find ```mermaid blocks and
      // render them as diagrams. Harmless to every other caller (just an extra class).
      const cls = lang ? ` language-${lang}` : '';
      out.push(`<pre class="md-code${cls}"><code>${escHtml(body.replace(/\n$/, ''))}</code></pre>`);
      return;
    }
    const lines = block.split('\n');
    const isSep = (s) => /^[\s|:-]+$/.test(s) && s.includes('-') && s.includes('|');
    const cells = (s) => s.trim().replace(/^\||\|$/g, '').split('|').map((c) => c.trim());
    for (let li = 0; li < lines.length; li++) {
      const line = lines[li].replace(/\s+$/, '');
      if (!line.trim()) { closePara(); closeList(); continue; }
      // GFM table: a header row of `| … |`, then a `|---|---|` separator, then body rows.
      if (line.includes('|') && li + 1 < lines.length && isSep(lines[li + 1])) {
        closePara(); closeList();
        const head = cells(line);
        const rows = [];
        li += 2;
        while (li < lines.length && lines[li].trim() && lines[li].includes('|')) {
          rows.push(cells(lines[li])); li++;
        }
        li--;  // the for-loop increment will step past the last consumed row
        out.push('<table><thead><tr>'
          + head.map((c) => `<th>${inline(c)}</th>`).join('') + '</tr></thead><tbody>'
          + rows.map((r) => '<tr>'
            + head.map((_, k) => `<td>${inline(r[k] || '')}</td>`).join('') + '</tr>').join('')
          + '</tbody></table>');
        continue;
      }
      // Blockquote: one or more consecutive `>` lines become one <blockquote>.
      if (/^>\s?/.test(line)) {
        closePara(); closeList();
        const buf = [];
        while (li < lines.length && /^>\s?/.test(lines[li])) {
          buf.push(lines[li].replace(/^>\s?/, '')); li++;
        }
        li--;  // the for-loop increment steps past the last quoted line
        out.push(`<blockquote>${inline(buf.join(' '))}</blockquote>`);
        continue;
      }
      const h = line.match(/^(#{1,4})\s+(.*)$/);
      if (h) {
        closePara(); closeList();
        const lvl = Math.min(6, h[1].length + 2);
        out.push(`<h${lvl}>${inline(h[2])}</h${lvl}>`);
        continue;
      }
      // A horizontal rule. Checked before the list rules, because `---` also matches the
      // `[-*+]` bullet pattern and would otherwise render as an empty list item.
      if (/^\s*(-{3,}|\*{3,}|_{3,})\s*$/.test(line)) {
        closePara(); closeList();
        out.push('<hr>');
        continue;
      }
      const ul = line.match(/^\s*[-*+]\s+(.*)$/);
      const ol = line.match(/^\s*\d+[.)]\s+(.*)$/);
      if (ul || ol) {
        closePara();
        const want = ul ? 'ul' : 'ol';
        if (list !== want) { closeList(); out.push(`<${want}>`); list = want; }
        flushItem();
        item = [(ul || ol)[1]];
        continue;
      }
      // Inside a list, a plain line continues the item it follows rather than ending the
      // list. A blank line (handled above) is what ends it.
      if (item) { item.push(line.trim()); continue; }
      closeList();
      para.push(line.trim());
    }
  });
  closePara(); closeList();
  return out.join('\n');
}

/* --- syntax highlighting -------------------------------------------------------------- */

const KEYWORDS = {
  python: /\b(def|class|return|if|elif|else|for|while|in|not|and|or|is|None|True|False|import|from|as|with|try|except|finally|raise|yield|lambda|global|nonlocal|assert|pass|break|continue|async|await|del)\b/g,
  javascript: /\b(const|let|var|function|return|if|else|for|while|of|in|new|class|extends|try|catch|finally|throw|typeof|instanceof|async|await|null|undefined|true|false|this|export|import|from|delete)\b/g,
  bash: /\b(if|then|elif|else|fi|for|while|do|done|case|esac|function|return|local|export|readonly|source|exit|set|trap)\b/g,
  yaml: null, toml: null, markdown: null, css: null, html: null, json: null,
};

/** Highlight a whole file at once, because the interesting cases span lines: a Python
 *  docstring is not a string on any single line of it. Returns one HTML string per line. */
export function highlight(text, lang) {
  const lines = text.split('\n');
  const kw = KEYWORDS[lang];
  const comment = lang === 'python' || lang === 'bash' || lang === 'yaml' || lang === 'toml'
    ? '#' : (lang === 'javascript' || lang === 'css' ? '//' : null);
  let triple = null;   // the open """ or ''' delimiter, or null

  return lines.map((line) => {
    if (!lang) return escHtml(line);

    /* Multi-line strings first: while one is open, the whole line is string. */
    if (triple) {
      const close = line.indexOf(triple);
      if (close < 0) return `<span class="s">${escHtml(line)}</span>`;
      const head = line.slice(0, close + 3);
      triple = null;
      return `<span class="s">${escHtml(head)}</span>` + hl(line.slice(close + 3));
    }
    const open = lang === 'python' ? line.match(/("""|''')/) : null;
    if (open) {
      const at = open.index;
      const rest = line.slice(at + 3);
      const closeAt = rest.indexOf(open[1]);
      if (closeAt < 0) {
        triple = open[1];
        return hl(line.slice(0, at)) + `<span class="s">${escHtml(line.slice(at))}</span>`;
      }
      return hl(line.slice(0, at))
        + `<span class="s">${escHtml(line.slice(at, at + 3 + closeAt + 3))}</span>`
        + hl(line.slice(at + 3 + closeAt + 3));
    }
    return hl(line);

    function hl(src) {
      if (!src) return '';
      /* Comments swallow the rest of the line, so find the first one that is not inside a
       * quoted string — the cheap way is to walk the line once. */
      if (comment) {
        let q = null;
        for (let i = 0; i < src.length; i++) {
          const c = src[i];
          if (q) { if (c === q && src[i - 1] !== '\\') q = null; continue; }
          if (c === '"' || c === "'") { q = c; continue; }
          if (src.startsWith(comment, i)) {
            return hl(src.slice(0, i)) + `<span class="c">${escHtml(src.slice(i))}</span>`;
          }
        }
      }
      let html = escHtml(src)
        .replace(/(&quot;[^&]*?&quot;|&#39;[^&]*?&#39;|'[^']*'|"[^"]*")/g, '<span class="s">$1</span>');
      if (kw) html = html.replace(kw, '<span class="k">$1</span>');
      html = html.replace(/\b(\d+\.?\d*(e-?\d+)?)\b/g, '<span class="n">$1</span>');
      return html;
    }
  });
}

/* --- file list ------------------------------------------------------------------------ */
