/* MCP Streamable HTTP client for docent web UI */

class McpClient {
  constructor() {
    this.sessionId = null;
    this.protocolVersion = null;
    this.requestId = 0;
    this.connected = false;
  }

  async initialize() {
    const body = {
      jsonrpc: '2.0',
      id: ++this.requestId,
      method: 'initialize',
      params: {
        protocolVersion: '2025-11-25',
        capabilities: {},
        clientInfo: {
          name: 'docent-web-ui',
          version: '0.1.0',
        },
      },
    };

    const { sessionId, data } = await this._post(body);

    this.sessionId = sessionId;
    this.protocolVersion = data.result.protocolVersion;

    await this._notifyInitialized();

    this.connected = true;
    return data.result;
  }

  async _notifyInitialized() {
    const body = {
      jsonrpc: '2.0',
      method: 'notifications/initialized',
    };

    await this._post(body);
  }

  async callTool(name, args) {
    const body = {
      jsonrpc: '2.0',
      id: ++this.requestId,
      method: 'tools/call',
      params: { name, arguments: args },
    };

    const { data } = await this._post(body);
    return data;
  }

  async _post(body) {
    const headers = {
      'Content-Type': 'application/json',
      'Accept': 'application/json, text/event-stream',
    };

    if (this.sessionId) {
      headers['Mcp-Session-Id'] = this.sessionId;
    }
    if (this.protocolVersion) {
      headers['MCP-Protocol-Version'] = this.protocolVersion;
    }

    const response = await fetch('/', {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
    });

    const sessionId = response.headers.get('Mcp-Session-Id');

    if (response.status === 202) {
      return { sessionId, data: null };
    }

    if (!response.ok) {
      const text = await response.text().catch(() => '');
      throw new Error(`HTTP ${response.status}: ${text || response.statusText}`);
    }

    const contentType = response.headers.get('Content-Type') || '';

    let data;
    if (contentType.includes('text/event-stream')) {
      data = await this._readSseEvent(response);
    } else {
      data = await response.json();
    }

    return { sessionId, data };
  }

  async _readSseEvent(response) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      const result = this._extractFirstDataEvent(buffer);
      if (result.event) {
        reader.cancel();
        return result.event;
      }
      buffer = result.remainder;
    }

    throw new Error('No response event in SSE stream');
  }

  _extractFirstDataEvent(buf) {
    let pos = 0;

    while (true) {
      const doubleNl = buf.indexOf('\n\n', pos);
      if (doubleNl === -1) {
        return { event: null, remainder: buf.slice(pos) };
      }

      const block = buf.slice(pos, doubleNl);
      pos = doubleNl + 2;

      if (!block.trim() || block.startsWith(':')) continue;

      const data = this._extractDataField(block);
      if (data) {
        return { event: JSON.parse(data), remainder: buf.slice(pos) };
      }
    }
  }

  _extractDataField(block) {
    for (const line of block.split('\n')) {
      if (line.startsWith('data:')) {
        let val = line.slice(5);
        if (val.startsWith(' ')) val = val.slice(1);
        return val || null;
      }
    }
    return null;
  }
}

/* UI logic */

const ui = {
  client: new McpClient(),

  elements: {
    status: document.getElementById('connection-status'),
    form: document.getElementById('search-form'),
    query: document.getElementById('query'),
    limit: document.getElementById('limit'),
    searchBtn: document.getElementById('search-btn'),
    results: document.getElementById('results'),
    rawContent: document.getElementById('raw-content'),
    copyRaw: document.getElementById('copy-raw'),
  },

  async init() {
    this.setFormEnabled(false);
    this.setStatus('connecting', 'Connecting to MCP server…');
    this.initCopyButtons();

    try {
      await this.client.initialize();
      this.setStatus('connected', `Connected — protocol version: ${this.client.protocolVersion}  |  session: ${this.client.sessionId}`);
      this.setFormEnabled(true);
      this.elements.query.focus();
    } catch (err) {
      this.setStatus('error', `Connection failed: ${err.message}`);
      this.showError(`Failed to initialize MCP session: ${err.message}`);
    }
  },

  async onSearch(event) {
    event.preventDefault();

    const query = this.elements.query.value.trim();
    if (!query) return;

    const limit = parseInt(this.elements.limit.value, 10);

    this.setFormEnabled(false);
    this.elements.searchBtn.innerHTML = '<span class="spinner"></span> Searching…';
    this.clearResults();
    this.elements.rawContent.textContent = '';

    try {
      const raw = await this.client.callTool('search_ddr', { query, limit });
      this.showRaw(raw);
      this.showResults(raw);
    } catch (err) {
      this.showError(`Search failed: ${err.message}`);
      this.showRaw({ error: err.message });
    } finally {
      this.setFormEnabled(true);
      this.elements.searchBtn.textContent = 'Search';
    }
  },

  showResults(raw) {
    if (!raw || !raw.result || !raw.result.content || raw.result.content.length === 0) {
      this.showError('No results returned');
      return;
    }

    if (raw.result.isError) {
      const text = raw.result.content.map(c => c.text || '').join('\n');
      this.showError(text);
      return;
    }

    const content = raw.result.content[0];
    if (content.type !== 'text') {
      this.showError(`Unexpected content type: ${content.type}`);
      return;
    }

    let results;
    try {
      results = JSON.parse(content.text);
    } catch {
      this.showError('Failed to parse search results JSON');
      return;
    }

    if (!Array.isArray(results) || results.length === 0) {
      this.elements.results.innerHTML = '<div class="result-placeholder">No matching documents found.</div>';
      return;
    }

    this.elements.results.innerHTML = results.map(r => `
      <div class="result-card">
        <div class="result-title">${this.esc(r.title)}</div>
        <div class="result-source-line">
          <span class="result-source">${this.esc(r.source_path)}</span>
          ${r.line_start ? `<span class="result-lines">L${r.line_start}${r.line_end !== r.line_start ? '-L' + r.line_end : ''}</span>` : ''}
          ${r.line_start ? `<button class="copy-link-btn" data-link="${this.esc(r.source_path)}#L${r.line_start}${r.line_end !== r.line_start ? '-L' + r.line_end : ''}">Copy link</button>` : ''}
        </div>
        <div class="result-meta">
          <span class="result-score">${(r.score * 100).toFixed(1)}% match</span>
          ${r.section_heading ? `<span class="result-section">${this.esc(r.section_heading)}</span>` : ''}
        </div>
        <div class="result-content-wrapper">
          <div class="result-content">${this.esc(r.matched_content)}</div>
          <button class="copy-content-btn">Copy content</button>
        </div>
      </div>
    `).join('');
  },

  initCopyButtons() {
    this.elements.results.addEventListener('click', (e) => {
      if (e.target.classList.contains('copy-link-btn')) {
        navigator.clipboard.writeText(e.target.dataset.link).catch(() => {});
        const txt = e.target.textContent;
        e.target.textContent = 'Copied!';
        setTimeout(() => { e.target.textContent = txt; }, 1500);
      } else if (e.target.classList.contains('copy-content-btn')) {
        const wrapper = e.target.closest('.result-content-wrapper');
        const content = wrapper.querySelector('.result-content').textContent;
        navigator.clipboard.writeText(content).catch(() => {});
        const txt = e.target.textContent;
        e.target.textContent = 'Copied!';
        setTimeout(() => { e.target.textContent = txt; }, 1500);
      }
    });
  },

  showError(message) {
    this.elements.results.innerHTML = `<div class="error-card">${this.esc(message)}</div>`;
  },

  clearResults() {
    this.elements.results.innerHTML = '';
  },

  showRaw(data) {
    this.elements.rawContent.textContent = JSON.stringify(data, null, 2);
  },

  setStatus(type, message) {
    const el = this.elements.status;
    el.className = `status-${type}`;
    el.textContent = message;
  },

  setFormEnabled(enabled) {
    this.elements.query.disabled = !enabled;
    this.elements.limit.disabled = !enabled;
    this.elements.searchBtn.disabled = !enabled;
  },

  esc(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  },
};

/* Wire up event listeners */
document.addEventListener('DOMContentLoaded', () => {
  ui.init();

  ui.elements.form.addEventListener('submit', (e) => ui.onSearch(e));

  ui.elements.copyRaw.addEventListener('click', () => {
    const text = ui.elements.rawContent.textContent;
    navigator.clipboard.writeText(text).catch(() => {});
  });
});
