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
    <div class="result-title"></div>
    <div class="result-source-line">
      <span class="result-source"></span>
      <span class="result-lines"></span>
      <button class="copy-link-btn">Copy link</button>
    </div>
    <div class="result-meta">
      <span class="result-score"></span>
      <span class="result-section"></span>
    </div>
    <div class="result-content-wrapper">
      <div class="result-content"></div>
      <button class="copy-content-btn">Copy content</button>
    </div>
    <div class="result-footer">
      <span class="result-modified"></span>
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

  it('should render valid results as result cards', () => {
    const view = new View(dom.window.document);
    const results = [
      {
        title: 'Test Doc',
        sourcePath: '/path/to/doc.md',
        matchedContent: 'some matched content',
        score: 0.95,
        lineStart: 10,
        lineEnd: 20,
        sectionHeading: 'Intro',
        modifiedAt: '2024-01-15T10:00:00Z',
      },
    ];
    view.renderResults(results);
    const resultsEl = view.elements.results;
    const card = resultsEl.querySelector('.result-card');
    assert.ok(card);
    assert.equal(card.querySelector('.result-title').textContent, 'Test Doc');
    assert.equal(card.querySelector('.result-source').textContent, '/path/to/doc.md');
    assert.match(card.querySelector('.result-score').textContent, /95/);
    assert.ok(card.querySelector('.result-lines'));
    assert.equal(card.querySelector('.result-section').textContent, 'Intro');
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
});
