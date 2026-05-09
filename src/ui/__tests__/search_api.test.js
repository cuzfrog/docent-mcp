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
            score: 0.95,
            line_start: 10,
            line_end: 20,
            section_heading: 'Introduction',
            modified_at: '2024-01-15T10:00:00Z',
            kind: 'file',
            source_revision: 'a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2',
            is_fresh: false,
            index_time: '2026-05-06T12:00:00Z',
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
    assert.equal(res.score, 0.95);
    assert.equal(res.lineStart, 10);
    assert.equal(res.lineEnd, 20);
    assert.equal(res.sectionHeading, 'Introduction');
    assert.equal(res.modifiedAt, '2024-01-15T10:00:00Z');
    assert.equal(res.kind, 'file');
    assert.equal(res.sourceRevision, 'a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2');
    assert.equal(res.isFresh, false);
    assert.equal(res.indexTime, '2026-05-06T12:00:00Z');
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
            score: 0.5,
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
    // Defaults for missing new fields
    assert.equal(r.results[0].kind, 'file');
    assert.equal(r.results[0].sourceRevision, '');
    assert.equal(r.results[0].isFresh, false);
    assert.equal(r.results[0].indexTime, null);
  });

  it('skips items missing required title', () => {
    const raw = {
      result: {
        content: [{
          type: 'text',
          text: JSON.stringify([
            { title: '', source_path: '/a.md', matched_content: 'c', score: 0.5, line_start: 1, line_end: 1, section_heading: null, modified_at: null, kind: 'git', source_revision: 'abc123', is_fresh: true, index_time: '2026-01-01T00:00:00Z' },
            { title: 'Valid', source_path: '/b.md', matched_content: 'c', score: 0.5, line_start: 1, line_end: 1, section_heading: null, modified_at: null, kind: 'file', source_revision: '', is_fresh: false, index_time: null },
          ]),
        }],
      },
    };
    const r = parseSearchResponse(raw);
    assert.equal(r.results.length, 1);
    assert.equal(r.results[0].title, 'Valid');
    assert.equal(r.results[0].kind, 'file');
  });

  it('handles unexpected errors gracefully', () => {
    // raw.result exists but content is somehow null
    const raw = { result: { content: null } };
    // Should not throw — return a structured error
    const r = parseSearchResponse(raw);
    assert.ok(r.error);
  });
});
