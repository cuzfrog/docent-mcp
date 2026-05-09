/* Search API — protocol constants, tool invocation, and response normalization. No DOM access. */

/**
 * @typedef {Object} NormalizedResult
 * @property {string} title
 * @property {string} sourcePath
 * @property {string} matchedContent
 * @property {number} score
 * @property {number} lineStart
 * @property {number} lineEnd
 * @property {string|null} sectionHeading
 * @property {string|null} modifiedAt
 * @property {string} kind - 'file' or 'git'
 * @property {string} sourceRevision - SHA-256 (file) or commit hash (git)
 * @property {boolean} isFresh - true for latest git commit per file
 * @property {string} indexTime - ISO 8601 index build timestamp
 */

export const PROTOCOL = {
  VERSION: '2025-11-25',
  TOOL_NAME: 'search_ddr',
  SESSION_HEADER: 'Mcp-Session-Id',
  PROTOCOL_VERSION_HEADER: 'MCP-Protocol-Version',
};

/**
 * Call the search_ddr tool and normalize the response.
 * @param {import('./mcp_client.js').McpClient} client
 * @param {string} query
 * @param {number} limit
 * @returns {Promise<{results: NormalizedResult[], raw: object, error: string|null}>}
 */
export async function searchDdr(client, query, limit) {
  const raw = await client.callTool(PROTOCOL.TOOL_NAME, { query, limit });
  return { ...parseSearchResponse(raw), raw };
}

/**
 * Parse raw MCP tool response into normalized results.
 * @param {object|null|undefined} raw
 * @returns {{results: NormalizedResult[], error: string|null, raw: object|null}}
 */
export function parseSearchResponse(raw) {
  if (!raw) {
    return { results: [], error: 'No response from search tool', raw };
  }

  if (raw.result?.isError) {
    const text = (raw.result.content || []).map(c => c.text || '').join('\n');
    return { results: [], error: `Tool returned an error: ${text}`, raw };
  }

  if (!raw.result?.content?.length) {
    return { results: [], error: 'No results returned', raw };
  }

  let parsed;
  try {
    parsed = JSON.parse(raw.result.content[0].text);
  } catch {
    return { results: [], error: 'Failed to parse server response', raw };
  }

  if (!Array.isArray(parsed)) {
    return { results: [], error: 'Unexpected response format', raw };
  }

  const results = [];
  for (const item of parsed) {
    if (!item.title) continue;

    results.push({
      title: item.title,
      sourcePath: item.source_path || '',
      matchedContent: item.matched_content || '',
      score: typeof item.score === 'number' ? item.score : 0,
      lineStart: typeof item.line_start === 'number' ? item.line_start : 0,
      lineEnd: typeof item.line_end === 'number' ? item.line_end : 0,
      sectionHeading: item.section_heading ?? null,
      modifiedAt: item.modified_at ?? null,
      kind: item.kind || 'file',
      sourceRevision: item.source_revision || '',
      isFresh: !!item.is_fresh,
      indexTime: item.index_time ?? null,
    });
  }

  return { results, error: null, raw };
}
