/* View layer — DOM queries, template cloning, rendering, and copy feedback. No fetch(). No protocol knowledge. */

export class View {
  /**
   * @param {Document} [doc]
   */
  constructor(doc = document) {
    this.doc = doc;
    this.elements = {
      status: doc.getElementById('connection-status'),
      form: doc.getElementById('search-form'),
      query: doc.getElementById('query'),
      limit: doc.getElementById('limit'),
      searchBtn: doc.getElementById('search-btn'),
      results: doc.getElementById('results'),
      rawContent: doc.getElementById('raw-content'),
      copyRaw: doc.getElementById('copy-raw'),
      toolInfo: doc.getElementById('tool-info'),
      resultsSection: doc.getElementById('results-section'),
    };
  }

  /**
   * Show connection status.
   * @param {string} state - 'connecting', 'connected', 'error'
   * @param {string} message
   */
  renderConnected(state, message) {
    const el = this.elements.status;
    el.className = `status-${state}`;
    el.textContent = message;
  }

  /**
   * Toggle form busy state.
   * @param {boolean} isBusy
   */
  renderBusy(isBusy) {
    this.elements.query.disabled = isBusy;
    this.elements.limit.disabled = isBusy;
    this.elements.searchBtn.disabled = isBusy;

    if (this.elements.resultsSection) {
      this.elements.resultsSection.setAttribute('aria-busy', String(isBusy));
    }

    if (isBusy) {
      this.elements.searchBtn.innerHTML = '<span class="spinner"></span> Searching…';
    } else {
      this.elements.searchBtn.textContent = 'Search';
    }
  }

  /**
   * Show an error card.
   * @param {string} message
   */
  renderError(message) {
    const template = this.doc.getElementById('error-card-template');
    if (template) {
      const clone = template.content.cloneNode(true);
      const msgEl = clone.querySelector('.error-message');
      if (msgEl) msgEl.textContent = message;
      this.elements.results.appendChild(clone);
    } else {
      this.elements.results.innerHTML = `<div class="error-card">${this.esc(message)}</div>`;
    }
  }

  /**
   * Render normalized search results.
   * @param {import('./search_api.js').NormalizedResult[]} results
   */
  renderResults(results) {
    this.clearResults();

    if (!results || results.length === 0) {
      this.showPlaceholder();
      return;
    }

    const template = this.doc.getElementById('result-card-template');
    if (!template) return;

    for (const result of results) {
      const clone = template.content.cloneNode(true);

      const setText = (sel, text) => {
        const el = clone.querySelector(sel);
        if (el) el.textContent = text;
      };

      setText('.result-title', result.title);
      setText('.result-source', result.sourcePath);

      if (result.lineStart) {
        const lines = `L${result.lineStart}${result.lineEnd !== result.lineStart ? '-L' + result.lineEnd : ''}`;
        setText('.result-lines', lines);
      }

      const linkBtn = clone.querySelector('.copy-link-btn');
      if (linkBtn && result.lineStart) {
        linkBtn.dataset.link = `${result.sourcePath}#L${result.lineStart}${result.lineEnd !== result.lineStart ? '-L' + result.lineEnd : ''}`;
      }

      setText('.result-score', `${(result.score * 100).toFixed(1)}% match`);
      if (result.sectionHeading) {
        setText('.result-section', result.sectionHeading);
      }
      setText('.result-content', result.matchedContent);
      setText('.result-modified', `Last modified at: ${this.formatTime(result.modifiedAt)}`);

      this.elements.results.appendChild(clone);
    }
  }

  /**
   * Show raw MCP response in debug panel.
   * @param {object} data
   */
  renderRawResponse(data) {
    this.elements.rawContent.textContent = JSON.stringify(data, null, 2);
  }

  /**
   * Render tool metadata in the debug/info section.
   * @param {Array} tools
   */
  renderToolInfo(tools) {
    const el = this.elements.toolInfo;
    if (!tools || tools.length === 0) {
      el.innerHTML = '';
      return;
    }

    const html = `
      <details class="tool-details">
        <summary>🔧 Tools</summary>
        ${tools.map(t => `
          <div class="tool-entry">
            <div class="tool-name">${this.esc(t.name)}</div>
            <div class="tool-desc">${this.esc(t.description || '')}</div>
            <details class="schema-details" open>
              <summary>Input Schema</summary>
              <pre class="tool-schema">${this.esc(JSON.stringify(t.inputSchema, null, 2))}</pre>
            </details>
          </div>
        `).join('')}
      </details>`;
    el.innerHTML = html;
  }

  /** Show the placeholder in results area. */
  showPlaceholder() {
    const template = this.doc.getElementById('placeholder-template');
    if (template) {
      const clone = template.content.cloneNode(true);
      this.elements.results.appendChild(clone);
    } else {
      this.elements.results.innerHTML = '<div class="result-placeholder">Enter a query above to search Design Decision Records.</div>';
    }
  }

  /** Clear results area. */
  clearResults() {
    this.elements.results.innerHTML = '';
  }

  /**
   * Initialize copy buttons on results via event delegation.
   * Attaches a single click listener on the results container.
   */
  initCopyButtons() {
    this.elements.results.addEventListener('click', (e) => {
      if (e.target.classList.contains('copy-link-btn')) {
        this._copyWithFeedback(e.target, e.target.dataset.link);
      } else if (e.target.classList.contains('copy-content-btn')) {
        const wrapper = e.target.closest('.result-content-wrapper');
        const content = wrapper ? wrapper.querySelector('.result-content').textContent : '';
        this._copyWithFeedback(e.target, content);
      }
    });
  }

  /**
   * Copy text to clipboard with temporary feedback.
   * @param {HTMLElement} button
   * @param {string} text
   * @returns {Function} cleanup function
   */
  _copyWithFeedback(button, text) {
    if (navigator?.clipboard?.writeText) {
      navigator.clipboard.writeText(text).catch(() => {});
    }
    const originalText = button.textContent;
    button.textContent = 'Copied!';
    const timeoutId = setTimeout(() => {
      button.textContent = originalText;
    }, 1500);
    return () => clearTimeout(timeoutId);
  }

  /**
   * Format ISO date string to YYYY-MM-DD HH:mm:ss.
   * @param {string|null} isoString
   * @returns {string}
   */
  formatTime(isoString) {
    if (!isoString) return 'N/A';
    const d = new Date(isoString);
    if (isNaN(d.getTime())) return isoString;
    const pad = (n) => String(n).padStart(2, '0');
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
  }

  /**
   * Escape HTML entities in a string.
   * @param {string} str
   * @returns {string}
   */
  esc(str) {
    const div = this.doc.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }
}
