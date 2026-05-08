// Test setup — polyfill TextEncoder/TextDecoder for jsdom compatibility
import { TextEncoder, TextDecoder } from 'util';
globalThis.TextEncoder = TextEncoder;
globalThis.TextDecoder = TextDecoder;
