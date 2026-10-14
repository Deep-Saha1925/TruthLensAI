import axios from 'axios';

const IMAGE_TIMEOUT_MS = 60_000;
// Longer than the server's VIDEO_TIMEOUT_SEC so the server's own 504 arrives first.
const VIDEO_TIMEOUT_MS = 300_000;

const client = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '/api',
  timeout: 15_000,
});

/** Every failure the UI sees is an ApiError with a message that is safe to display. */
export class ApiError extends Error {
  constructor(
    message,
    { code = 'UNKNOWN', status = null, requestId = null, retryAfter = null } = {},
  ) {
    super(message);
    this.name = 'ApiError';
    this.code = code;
    this.status = status;
    this.requestId = requestId;
    this.retryAfter = retryAfter;
  }
}

export const isCancelled = axios.isCancel;

/** Whether sending the same request again could plausibly succeed. */
export function isRetryable(error) {
  if (!(error instanceof ApiError)) return false;
  return error.status === null || error.status === 429 || error.status >= 500;
}

function retryAfterSeconds(response) {
  const seconds = Number.parseInt(response?.headers?.['retry-after'], 10);
  return Number.isFinite(seconds) && seconds > 0 ? seconds : null;
}

export function toApiError(error) {
  if (axios.isCancel(error)) return error;

  const response = error.response;
  const body = response?.data?.error; // the backend's uniform error shape
  if (body?.message) {
    const retryAfter = retryAfterSeconds(response);
    const suffix = retryAfter ? ` Try again in ${retryAfter} s.` : '';
    return new ApiError(`${body.message}${suffix}`, {
      code: body.code,
      status: response.status,
      requestId: body.request_id,
      retryAfter,
    });
  }
  if (response) {
    return new ApiError(`The server responded with status ${response.status}.`, {
      code: 'HTTP_ERROR',
      status: response.status,
    });
  }
  if (error.code === 'ECONNABORTED' || error.code === 'ETIMEDOUT') {
    return new ApiError('The server took too long to respond.', { code: 'TIMEOUT' });
  }
  return new ApiError(
    'Cannot reach the analysis server. If you uploaded a very large file, the server may have closed the connection.',
    { code: 'NETWORK_ERROR' },
  );
}

function assertResult(data) {
  if (!data || typeof data.id !== 'string') {
    throw new ApiError('The server returned an unexpected response.', { code: 'BAD_RESPONSE' });
  }
}

async function upload(path, file, { signal, onUploadProgress, timeout }) {
  const body = new FormData();
  body.append('file', file); // the server reads only the extension of the client filename

  let data;
  try {
    // No Content-Type header: the browser must add the multipart boundary itself.
    ({ data } = await client.post(path, body, { signal, onUploadProgress, timeout }));
  } catch (error) {
    throw toApiError(error);
  }
  assertResult(data);
  return data;
}

export async function getModels({ signal } = {}) {
  try {
    const { data } = await client.get('/models', { signal });
    return data;
  } catch (error) {
    throw toApiError(error);
  }
}

export function detectImage(file, { signal, onUploadProgress } = {}) {
  return upload('/detect/image', file, { signal, onUploadProgress, timeout: IMAGE_TIMEOUT_MS });
}

export function detectVideo(file, { signal, onUploadProgress } = {}) {
  return upload('/detect/video', file, { signal, onUploadProgress, timeout: VIDEO_TIMEOUT_MS });
}

export async function getResult(resultId, { signal } = {}) {
  let data;
  try {
    ({ data } = await client.get(`/results/${encodeURIComponent(resultId)}`, { signal }));
  } catch (error) {
    throw toApiError(error);
  }
  assertResult(data);
  return data;
}