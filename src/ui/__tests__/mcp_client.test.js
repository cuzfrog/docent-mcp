import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { extractDataField, extractFirstDataEvent } from '../mcp_client.js';

describe('extractFirstDataEvent', () => {
  it('extracts JSON from a basic SSE block', () => {
    const buf = 'data: {"jsonrpc":"2.0","id":1}\n\n';
    const result = extractFirstDataEvent(buf);
    assert.deepEqual(result.event, { jsonrpc: '2.0', id: 1 });
    assert.equal(result.remainder, '');
  });

  it('skips comments and empty lines', () => {
    const buf = ': comment\n\ndata: {"key":"val"}\n\n';
    const result = extractFirstDataEvent(buf);
    assert.deepEqual(result.event, { key: 'val' });
  });

  it('handles data field with leading space', () => {
    const buf = 'data: {"x":1}\n\n';
    const result = extractFirstDataEvent(buf);
    assert.deepEqual(result.event, { x: 1 });
  });

  it('returns null for incomplete buffer', () => {
    const buf = 'data: {"x":1}\n';
    const result = extractFirstDataEvent(buf);
    assert.equal(result.event, null);
    assert.equal(result.remainder, buf);
  });

  it('handles multiple blocks and returns first', () => {
    const buf = 'data: {"first":1}\n\ndata: {"second":2}\n\n';
    const result = extractFirstDataEvent(buf);
    assert.deepEqual(result.event, { first: 1 });
  });
});

describe('extractDataField', () => {
  it('extracts value from data line', () => {
    assert.equal(extractDataField('data: hello'), 'hello');
    assert.equal(extractDataField('data:  {"nested": true}'), ' {"nested": true}');
  });

  it('returns null for non-data lines', () => {
    assert.equal(extractDataField('event: update'), null);
    assert.equal(extractDataField('id: 1'), null);
    assert.equal(extractDataField(': comment'), null);
  });

  it('handles empty data value', () => {
    assert.equal(extractDataField('data:'), '');
    assert.equal(extractDataField('data: '), '');
  });
});
