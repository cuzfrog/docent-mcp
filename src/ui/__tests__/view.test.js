import { describe, it, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const HTML = `<!DOCTYPE html>
<div id="connection-status"></div>
<form id="search-form">
  <input id="query">
  <select id="limit">
    <option value="3">3</option>
    <option value="5">5</option>
    <option value="10">10</option>
  </select>
  <button id="search-btn">Search</button>
</form>
<div id="results"></div>
<div id="raw-content"></div>
<button id="copy-raw">Copy</button>
<div id="tool-info"></div>
<div id="results-section"></div>
<template id="result-card-template">
  <div class="result-card">
    <div class="result-title-line">
      <span class="result-kind-badge"></span>
      <span class="result-title"></span>
      <span class="result-score"></span>
    </div>
    <div class="result-source-line">
      <span class="result-source"></span>
      <span class="result-lines"></span>
      <span class="result-freshness"></span>
      <button class="copy-link-btn">Copy link</button>
    </div>
    <div class="result-meta">
      <span class="result-section"></span>
    </div>
    <div class="result-content-wrapper">
      <div class="result-content"></div>
      <button class="copy-content-btn">Copy content</button>
    </div>
    <div class="result-footer">
      <span class="result-footer-text"></span>
    </div>
  </div>
</template>
<template id="error-card-template">
  <div class="error-card"><span class="error-message"></span></div>
</template>
<template id="placeholder-template">
  <div class="result-placeholder">Enter a query above to search.</div>
</template>`;

describe('View', () => {
  let dom;
  let View;

  before(async () => {
    dom = new JSDOM(HTML, { url: 'http://localhost' });
    globalThis.document = dom.window.document;
    globalThis.setTimeout = dom.window.setTimeout;
    globalThis.clearTimeout = dom.window.clearTimeout;

    // Mock clipboard on existing navigator
    if (!navigator.clipboard) {
      Object.defineProperty(navigator, 'clipboard', {
        value: { writeText: async () => {} },
        writable: false,
        configurable: true,
      });
    }

    const mod = await import('../view.js');
    View = mod.View;
  });

  after(() => {
    delete globalThis.document;
    delete globalThis.setTimeout;
    delete globalThis.clearTimeout;
  });

  it('should render empty results as placeholder', () => {
    const view = new View(dom.window.document);
    view.renderResults([]);
    const resultsEl = view.elements.results;
    assert.ok(resultsEl.querySelector('.result-placeholder'));
    assert.equal(resultsEl.children.length, 1);
  });

  it('should render valid file results as result cards', () => {
    const view = new View(dom.window.document);
    const results = [
      {
        title: 'Test Doc',
        sourcePath: '/path/to/doc.md',
        matchedContent: 'some matched content',
        total_score: 0.85,
        semantic_score: 0.95,
        bm25_score: 0.75,
        lineStart: 10,
        lineEnd: 20,
        sectionHeading: 'Intro',
        modifiedAt: '2024-01-15T10:00:00Z',
        kind: 'file',
        sourceRevision: 'a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2',
        isFresh: false,
        indexTime: '2026-05-06T12:00:00Z',
      },
    ];
    view.renderResults(results);
    const resultsEl = view.elements.results;
    const card = resultsEl.querySelector('.result-card');
    assert.ok(card);
    assert.equal(card.querySelector('.result-title').textContent, 'Test Doc');
    assert.equal(card.querySelector('.result-source').textContent, '/path/to/doc.md');
    assert.match(card.querySelector('.result-score').textContent, /100%/);
    assert.equal(card.querySelector('.result-score').title, 'semantic: 0.95, bm25: 0.75, raw: 0.8500');
    assert.ok(card.querySelector('.result-lines'));
    assert.equal(card.querySelector('.result-section').textContent, 'Intro');
    // Kind badge
    const badge = card.querySelector('.result-kind-badge');
    assert.equal(badge.textContent, 'File');
    assert.ok(badge.classList.contains('badge-file'));
    // Freshness badge not rendered for file kind
    assert.ok(!card.querySelector('.result-freshness').textContent);
    // Footer
    const footer = card.querySelector('.result-footer-text');
    assert.match(footer.textContent, /^Modified:/);
    assert.match(footer.textContent, /SHA: a1b2c3d4e5f6/);
  });

  it('should render valid git results with freshness badge', () => {
    const view = new View(dom.window.document);
    const results = [
      {
        title: 'feat: add caching',
        sourcePath: 'src/cache.rs',
        matchedContent: '+ fn get() {}',
        total_score: 0.72,
        semantic_score: 0.85,
        bm25_score: 0.60,
        lineStart: 10,
        lineEnd: 45,
        sectionHeading: null,
        modifiedAt: '2024-03-20T10:00:00Z',
        kind: 'git',
        sourceRevision: 'a1b2c3d7e8f9a1b2c3d7e8f9a1b2c3d7e8f9a1b2',
        isFresh: true,
        indexTime: '2026-05-06T12:00:00Z',
      },
    ];
    view.renderResults(results);
    const resultsEl = view.elements.results;
    const card = resultsEl.querySelector('.result-card');
    assert.ok(card);
    // Kind badge
    const badge = card.querySelector('.result-kind-badge');
    assert.equal(badge.textContent, 'Git');
    assert.ok(badge.classList.contains('badge-git'));
    // Score with breakdown tooltip
    assert.match(card.querySelector('.result-score').textContent, /100%/);
    assert.equal(card.querySelector('.result-score').title, 'semantic: 0.85, bm25: 0.60, raw: 0.7200');
    // Freshness badge
    const freshness = card.querySelector('.result-freshness');
    assert.equal(freshness.textContent, 'Fresh');
    assert.ok(freshness.classList.contains('badge-fresh'));
    // Footer
    const footer = card.querySelector('.result-footer-text');
    assert.match(footer.textContent, /^Committed:/);
    assert.match(footer.textContent, /SHA: a1b2c3d7e8f9/);
  });

  it('should render error card with message', () => {
    const view = new View(dom.window.document);
    view.renderError('Something went wrong');
    const errorCard = view.elements.results.querySelector('.error-card');
    assert.ok(errorCard);
    assert.equal(errorCard.querySelector('.error-message').textContent, 'Something went wrong');
  });

  it('should toggle busy state on form', () => {
    const view = new View(dom.window.document);
    view.renderBusy(true);
    assert.equal(view.elements.query.disabled, true);
    assert.equal(view.elements.limit.disabled, true);
    assert.equal(view.elements.searchBtn.disabled, true);
    assert.ok(view.elements.searchBtn.innerHTML.includes('spinner'));

    view.renderBusy(false);
    assert.equal(view.elements.query.disabled, false);
    assert.equal(view.elements.limit.disabled, false);
    assert.equal(view.elements.searchBtn.disabled, false);
    assert.equal(view.elements.searchBtn.textContent, 'Search');
  });

  it('should set connection status text and class', () => {
    const view = new View(dom.window.document);
    view.renderConnected('connected', 'Connected — protocol: v1');
    assert.ok(view.elements.status.classList.contains('status-connected'));
    assert.equal(view.elements.status.textContent, 'Connected — protocol: v1');
  });

  it('should render raw MCP response as JSON by default', () => {
    const view = new View(dom.window.document);
    const data = { jsonrpc: '2.0', result: { content: [{ text: 'hello' }] } };
    view.renderRawResponse(data);
    const text = view.elements.rawContent.textContent;
    assert.ok(text.includes('"jsonrpc"'));
    assert.ok(text.includes('"content"'));
  });

  it('should toggle to pretty mode and show human-readable content', () => {
    const view = new View(dom.window.document);
    const data = { jsonrpc: '2.0', result: { content: [{ text: 'search result 1' }, { text: 'search result 2' }] } };
    view.renderRawResponse(data);
    view.toggleRawMode();
    const text = view.elements.rawContent.textContent;
    assert.ok(!text.includes('"jsonrpc"'));
    assert.ok(text.includes('search result 1'));
    assert.ok(text.includes('---'));
    assert.ok(text.includes('search result 2'));
    assert.equal(view.rawMode, 'pretty');
  });

  it('should toggle back to raw mode', () => {
    const view = new View(dom.window.document);
    const data = { result: { content: [{ text: 'hello' }] } };
    view.renderRawResponse(data);
    view.toggleRawMode();
    view.toggleRawMode();
    const text = view.elements.rawContent.textContent;
    assert.ok(text.includes('"result"'));
    assert.equal(view.rawMode, 'raw');
  });

  it('should normalize scores relative to top result', () => {
    const view = new View(dom.window.document);
    const results = [
      {
        title: 'First',
        sourcePath: 'a.md',
        matchedContent: 'content a',
        total_score: 0.032,
        semantic_score: 0.9,
        bm25_score: 0.8,
        lineStart: 1,
        lineEnd: 5,
        sectionHeading: null,
        modifiedAt: '2024-01-01T00:00:00Z',
        kind: 'file',
        sourceRevision: 'aabbcc',
        isFresh: false,
        indexTime: '2026-01-01T00:00:00Z',
      },
      {
        title: 'Second',
        sourcePath: 'b.md',
        matchedContent: 'content b',
        total_score: 0.020,
        semantic_score: 0.5,
        bm25_score: 0.4,
        lineStart: 1,
        lineEnd: 5,
        sectionHeading: null,
        modifiedAt: '2024-01-01T00:00:00Z',
        kind: 'file',
        sourceRevision: 'ddeeff',
        isFresh: false,
        indexTime: '2026-01-01T00:00:00Z',
      },
    ];
    view.renderResults(results);
    const cards = view.elements.results.querySelectorAll('.result-card');
    assert.equal(cards.length, 2);
    assert.match(cards[0].querySelector('.result-score').textContent, /100%/);
    assert.match(cards[1].querySelector('.result-score').textContent, /63%/);
    // Check raw score in tooltip
    assert.equal(cards[0].querySelector('.result-score').title, 'semantic: 0.90, bm25: 0.80, raw: 0.0320');
    assert.equal(cards[1].querySelector('.result-score').title, 'semantic: 0.50, bm25: 0.40, raw: 0.0200');
  });
});
