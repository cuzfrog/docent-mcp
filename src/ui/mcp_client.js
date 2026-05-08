/* MCP Streamable HTTP client — transport layer. No DOM access. No UI state. */

/**
 * @private
 * @param {string} block
 * @returns {string|null}
 */
export function extractDataField(block) {
  for (const line of block.split('\n')) {
    if (line.startsWith('data:')) {
      let val = line.slice(5);
      if (val.startsWith(' ')) val = val.slice(1);
      return val;
    }
  }
  return null;
}

/**
 * @private
 * @param {string} buf
 * @returns {{ event: object|null, remainder: string }}
 */
export function extractFirstDataEvent(buf) {
  let pos = 0;

  while (true) {
    const doubleNl = buf.indexOf('\n\n', pos);
    if (doubleNl === -1) {
      return { event: null, remainder: buf.slice(pos) };
    }

    const block = buf.slice(pos, doubleNl);
    pos = doubleNl + 2;

    if (!block.trim() || block.startsWith(':')) continue;

    const data = extractDataField(block);
    if (data) {
      return { event: JSON.parse(data), remainder: buf.slice(pos) };
    }
  }
}

export class McpClient {
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

  async listTools() {
    const body = {
      jsonrpc: '2.0',
      id: ++this.requestId,
      method: 'tools/list',
    };
    const { data } = await this._post(body);
    return data;
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

      const result = extractFirstDataEvent(buffer);
      if (result.event) {
        reader.cancel();
        return result.event;
      }
      buffer = result.remainder;
    }

    throw new Error('No response event in SSE stream');
  }
}
