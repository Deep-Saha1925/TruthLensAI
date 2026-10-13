import axios from 'axios';

const client = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '/api',
  timeout: 15_000,
});

/** Every failure the UI sees is an ApiError with a message that is safe to display. */
export class ApiError extends Error {
  constructor(message, { code = 'UNKNOWN', status = null, requestId = null } = {}) {
    super(message);
    this.name = 'ApiError';
    this.code = code;
    this.status = status;
    this.requestId = requestId;
  }
}

export const isCancelled = axios.isCancel;

function toApiError(error) {
  if (axios.isCancel(error)) return error;

  const body = error.response?.data?.error; // the backend's uniform error shape
  if (body?.message) {
    return new ApiError(body.message, {
      code: body.code,
      status: error.response.status,
      requestId: body.request_id,
    });
  }
  if (error.response) {
    return new ApiError(`The server responded with status ${error.response.status}.`, {
      code: 'HTTP_ERROR',
      status: error.response.status,
    });
  }
  if (error.code === 'ECONNABORTED') {
    return new ApiError('The server took too long to respond.', { code: 'TIMEOUT' });
  }
  return new ApiError('Cannot reach the analysis server.', { code: 'NETWORK_ERROR' });
}

export async function getModels({ signal } = {}) {
  try {
    const { data } = await client.get('/models', { signal });
    return data;
  } catch (error) {
    throw toApiError(error);
  }
}

// Phase 6 adds detectImage, detectVideo and getResult here.