import axios from 'axios';
import { describe, expect, it } from 'vitest';

import { ApiError, isRetryable, toApiError } from '../services/api.js';

const serverError = (status, error, headers = {}) => ({
  response: { status, headers, data: { error } },
});

describe('toApiError', () => {
  it("uses the backend's uniform error shape", () => {
    const result = toApiError(
      serverError(415, {
        code: 'UNSUPPORTED_MEDIA_TYPE',
        message: 'File content does not match a supported image format.',
        request_id: 'abc123',
      }),
    );
    expect(result).toBeInstanceOf(ApiError);
    expect(result.code).toBe('UNSUPPORTED_MEDIA_TYPE');
    expect(result.status).toBe(415);
    expect(result.requestId).toBe('abc123');
    expect(result.message).toBe('File content does not match a supported image format.');
  });

  it('adds Retry-After to the message and exposes it', () => {
    const result = toApiError(
      serverError(
        429,
        { code: 'RATE_LIMITED', message: 'Too many requests.', request_id: 'r' },
        { 'retry-after': '7' },
      ),
    );
    expect(result.message).toContain('Try again in 7 s');
    expect(result.retryAfter).toBe(7);
  });

  it('falls back to the status when the body is not the uniform shape', () => {
    const result = toApiError({ response: { status: 502, headers: {}, data: '<html>' } });
    expect(result.code).toBe('HTTP_ERROR');
    expect(result.message).toContain('502');
  });

  it('maps timeouts and network failures', () => {
    expect(toApiError({ code: 'ECONNABORTED' }).code).toBe('TIMEOUT');
    expect(toApiError({ code: 'ETIMEDOUT' }).code).toBe('TIMEOUT');
    expect(toApiError({ code: 'ERR_NETWORK', message: 'Network Error' }).code).toBe('NETWORK_ERROR');
  });

  it('passes cancellations through untouched', () => {
    const cancel = new axios.CanceledError();
    expect(toApiError(cancel)).toBe(cancel);
    expect(axios.isCancel(toApiError(cancel))).toBe(true);
  });
});

describe('isRetryable', () => {
  it('retries transient failures only', () => {
    expect(isRetryable(new ApiError('x', { code: 'NETWORK_ERROR' }))).toBe(true);
    expect(isRetryable(new ApiError('x', { status: 429 }))).toBe(true);
    expect(isRetryable(new ApiError('x', { status: 503 }))).toBe(true);
    expect(isRetryable(new ApiError('x', { status: 415 }))).toBe(false);
    expect(isRetryable(new ApiError('x', { status: 413 }))).toBe(false);
    expect(isRetryable(new Error('plain'))).toBe(false);
  });
});