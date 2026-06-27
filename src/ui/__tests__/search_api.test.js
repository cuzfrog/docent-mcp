import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { parseSearchResponse } from '../search_api.js';

describe('parseSearchResponse', () => {
  it('returns error for null/undefined input', () => {
    const r1 = parseSearchResponse(null);
    assert.match(r1.error, /No response/);
    assert.deepEqual(r1.results, []);

    const r2 = parseSearchResponse(undefined);
    assert.match(r2.error, /No response/);
  });

  it('returns error when tool reports isError', () => {
    const raw = { result: { isError: true, content: [{ text: 'Something broke' }] } };
    const r = parseSearchResponse(raw);
    assert.match(r.error, /Something broke/);
    assert.deepEqual(r.results, []);
  });

  it('returns error for empty content', () => {
    const raw = { result: { content: [] } };
    const r = parseSearchResponse(raw);
    assert.match(r.error, /No results/);
  });

  it('returns error for unparseable JSON', () => {
    const raw = { result: { content: [{ type: 'text', text: 'not-json' }] } };
    const r = parseSearchResponse(raw);
    assert.match(r.error, /Failed to parse/);
  });

  it('returns error for non-array JSON', () => {
    const raw = { result: { content: [{ type: 'text', text: '{"obj": true}' }] } };
    const r = parseSearchResponse(raw);
    assert.match(r.error, /Unexpected response format/);
  });

  it('normalizes valid results', () => {
    const raw = {
      result: {
        content: [{
          type: 'text',
          text: JSON.stringify([{
            title: 'Test DDR',
            source_path: '/path/to/doc.md',
            matched_content: 'some content',
            total_score: 0.85,
            semantic_score: 0.95,
            bm25_score: 0.75,
            line_start: 10,
            line_end: 20,
            section_heading: 'Introduction',
            modified_at: '2024-01-15T10:00:00Z',
            source_revision: 'a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2',
          }]),
        }],
      },
    };
    const r = parseSearchResponse(raw);
    assert.equal(r.error, null);
    assert.equal(r.results.length, 1);
    const res = r.results[0];
    assert.equal(res.title, 'Test DDR');
    assert.equal(res.sourcePath, '/path/to/doc.md');
    assert.equal(res.matchedContent, 'some content');
    assert.equal(res.total_score, 0.85);
    assert.equal(res.semantic_score, 0.95);
    assert.equal(res.bm25_score, 0.75);
    assert.equal(res.lineStart, 10);
    assert.equal(res.lineEnd, 20);
    assert.equal(res.sectionHeading, 'Introduction');
    assert.equal(res.modifiedAt, '2024-01-15T10:00:00Z');
    assert.equal(res.sourceRevision, 'a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2');
  });

  it('normalizes results with null fields', () => {
    const raw = {
      result: {
        content: [{
          type: 'text',
          text: JSON.stringify([{
            title: 'Minimal DDR',
            source_path: '/path/doc.md',
            matched_content: 'content',
            total_score: 0.5,
            semantic_score: 0.7,
            bm25_score: 0.3,
            line_start: 1,
            line_end: 1,
            section_heading: null,
            modified_at: null,
          }]),
        }],
      },
    };
    const r = parseSearchResponse(raw);
    assert.equal(r.results.length, 1);
    assert.equal(r.results[0].sectionHeading, null);
    assert.equal(r.results[0].modifiedAt, null);
    assert.equal(r.results[0].sourceRevision, '');
  });

  it('skips items missing required title', () => {
    const raw = {
      result: {
        content: [{
          type: 'text',
          text: JSON.stringify([
            { title: '', source_path: '/a.md', matched_content: 'c', total_score: 0.5, semantic_score: 0.7, bm25_score: 0.3, line_start: 1, line_end: 1, section_heading: null, modified_at: null, source_revision: 'abc123' },
            { title: 'Valid', source_path: '/b.md', matched_content: 'c', total_score: 0.5, semantic_score: 0.7, bm25_score: 0.3, line_start: 1, line_end: 1, section_heading: null, modified_at: null, source_revision: '' },
          ]),
        }],
      },
    };
    const r = parseSearchResponse(raw);
    assert.equal(r.results.length, 1);
    assert.equal(r.results[0].title, 'Valid');
  });

  it('handles unexpected errors gracefully', () => {
    // raw.result exists but content is somehow null
    const raw = { result: { content: null } };
    // Should not throw — return a structured error
    const r = parseSearchResponse(raw);
    assert.ok(r.error);
  });
});
