/* Search API — protocol constants, tool invocation, and response normalization. No DOM access. */

/**
 * @typedef {Object} NormalizedResult
 * @property {string} title
 * @property {string} sourcePath
 * @property {string} matchedContent
 * @property {number} total_score
 * @property {number} semantic_score
 * @property {number} bm25_score
 * @property {number} lineStart
 * @property {number} lineEnd
 * @property {string|null} sectionHeading
 * @property {string|null} modifiedAt
 * @property {string} sourceRevision - SHA-256 of file content
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
      total_score: typeof item.total_score === 'number' ? item.total_score : 0,
      semantic_score: typeof item.semantic_score === 'number' ? item.semantic_score : 0,
      bm25_score: typeof item.bm25_score === 'number' ? item.bm25_score : 0,
      lineStart: typeof item.line_start === 'number' ? item.line_start : 0,
      lineEnd: typeof item.line_end === 'number' ? item.line_end : 0,
      sectionHeading: item.section_heading ?? null,
      modifiedAt: item.modified_at ?? null,
      sourceRevision: item.source_revision || '',
    });
  }

  return { results, error: null, raw };
}
