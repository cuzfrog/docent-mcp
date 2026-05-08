/* Application controller — wires transport, search API, and view. Owns state transitions. */

import { McpClient } from './mcp_client.js';
import { searchDdr } from './search_api.js';
import { View } from './view.js';

const client = new McpClient();
const view = new View();

const state = {
  connected: false,
  searching: false,
  tools: null,
  lastRaw: null,
};

async function initializeApp() {
  view.renderBusy(true);
  view.renderConnected('connecting', 'Connecting to MCP server…');
  view.initCopyButtons();

  try {
    const result = await client.initialize();
    state.connected = true;
    view.renderConnected('connected', `Connected — protocol version: ${client.protocolVersion}  |  session: ${client.sessionId}`);
    await fetchAndRenderTools();
    view.renderBusy(false);
    view.showPlaceholder();
  } catch (err) {
    view.renderBusy(false);
    view.renderConnected('error', `Connection failed: ${err.message}`);
    view.renderError(`Failed to initialize MCP session: ${err.message}`);
  }
}

async function fetchAndRenderTools() {
  try {
    const data = await client.listTools();
    state.tools = data.result?.tools || [];
  } catch {
    state.tools = [];
  }
  view.renderToolInfo(state.tools);
}

async function handleSearch(event) {
  event.preventDefault();
  if (!state.connected) return;

  const query = view.elements.query.value.trim();
  if (!query) return;

  const limit = parseInt(view.elements.limit.value, 10);

  state.searching = true;
  view.renderBusy(true);
  view.clearResults();
  view.elements.rawContent.textContent = '';

  try {
    const { results, raw, error } = await searchDdr(client, query, limit);
    state.lastRaw = raw;
    view.renderRawResponse(raw);
    if (error) {
      view.renderError(error);
    } else {
      view.renderResults(results);
    }
  } catch (err) {
    view.renderError(`Search failed: ${err.message}`);
    view.renderRawResponse({ error: err.message });
  } finally {
    state.searching = false;
    view.renderBusy(false);
  }
}

function handleCopyRaw() {
  const text = view.elements.rawContent.textContent;
  if (text) {
    navigator.clipboard.writeText(text).catch(() => {});
  }
}

/* Wire up event listeners */
document.addEventListener('DOMContentLoaded', () => {
  initializeApp();

  view.elements.form.addEventListener('submit', handleSearch);
  view.elements.copyRaw.addEventListener('click', handleCopyRaw);
});
