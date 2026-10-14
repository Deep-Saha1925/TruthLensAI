import { describe, expect, it } from 'vitest';

import { formatBytes, formatDuration } from '../utils/formatting.js';
import {
  DEFAULT_LIMITS,
  detectMediaKind,
  getExtension,
  validateFile,
  validateVideoDuration,
} from '../utils/validation.js';

const MB = 1024 * 1024;
const file = (name, type, size) => ({ name, type, size });

describe('formatting', () => {
  it('formats byte sizes', () => {
    expect(formatBytes(512)).toBe('512 B');
    expect(formatBytes(1536)).toBe('1.5 KB');
    expect(formatBytes(10 * MB)).toBe('10.0 MB');
    expect(formatBytes(-1)).toBe('—');
  });

  it('formats durations', () => {
    expect(formatDuration(65.4)).toBe('1:05');
    expect(formatDuration(5.9)).toBe('0:05');
    expect(formatDuration(Number.NaN)).toBe('—');
  });
});

describe('file type detection', () => {
  it('reads extensions case-insensitively and ignores path parts', () => {
    expect(getExtension('Photo.JPG')).toBe('.jpg');
    expect(getExtension('a/b\\c.png')).toBe('.png');
    expect(getExtension('noextension')).toBe('');
  });

  it('classifies by extension', () => {
    expect(detectMediaKind('a.webp')).toBe('image');
    expect(detectMediaKind('a.mkv')).toBe('video');
    expect(detectMediaKind('a.exe')).toBeNull();
  });
});

describe('validateFile', () => {
  it('accepts valid images and videos', () => {
    expect(validateFile(file('a.png', 'image/png', 1000))).toEqual({ ok: true, kind: 'image' });
    expect(validateFile(file('a.mp4', 'video/mp4', 50 * MB))).toEqual({ ok: true, kind: 'video' });
  });

  it('accepts generic MIME types, which browsers send for valid files', () => {
    expect(validateFile(file('a.mkv', '', 1000)).ok).toBe(true);
    expect(validateFile(file('a.mov', 'application/octet-stream', 1000)).ok).toBe(true);
  });

  it('rejects unsupported extensions', () => {
    const result = validateFile(file('payload.exe', 'application/x-msdownload', 100));
    expect(result.ok).toBe(false);
    expect(result.code).toBe('UNSUPPORTED_EXTENSION');
  });

  it('rejects a declared type that contradicts the extension', () => {
    expect(validateFile(file('a.png', 'text/html', 100)).code).toBe('MIME_MISMATCH');
    expect(validateFile(file('a.mp4', 'image/png', 100)).code).toBe('MIME_MISMATCH');
  });

  it('rejects empty files', () => {
    expect(validateFile(file('a.png', 'image/png', 0)).code).toBe('EMPTY_FILE');
  });

  it('enforces per-kind size limits', () => {
    const image = validateFile(file('a.jpg', 'image/jpeg', 11 * MB));
    expect(image.code).toBe('FILE_TOO_LARGE');
    expect(image.message).toContain('10 MB');
    expect(validateFile(file('a.mp4', 'video/mp4', 100 * MB + 1)).code).toBe('FILE_TOO_LARGE');
    expect(validateFile(file('a.mp4', 'video/mp4', 99 * MB)).ok).toBe(true);
  });

  it('uses the limits it is given', () => {
    const strict = { ...DEFAULT_LIMITS, max_image_size_mb: 1 };
    expect(validateFile(file('a.png', 'image/png', 2 * MB), strict).code).toBe('FILE_TOO_LARGE');
  });
});

describe('validateVideoDuration', () => {
  it('blocks videos over the limit', () => {
    expect(validateVideoDuration(61)).toContain('60 seconds');
    expect(validateVideoDuration(60)).toBeNull();
  });

  it('does not block when the duration is unknown', () => {
    expect(validateVideoDuration(undefined)).toBeNull();
    expect(validateVideoDuration(Number.POSITIVE_INFINITY)).toBeNull();
  });
});